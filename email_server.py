import datetime
import logging
import re
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)

import pandas as pd
from fastapi import FastAPI, Form, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from email_service import parse_search_query
from email_utils import (
    get_attachment_file,
    get_one_email,
    get_thread_for_email,
    load_email_content_search,
    load_email_db,
)

if TYPE_CHECKING:
    import duckdb

db_connections: Dict[str, Union["duckdb.DuckDBPyConnection"]] = {}

EMAILS_PER_PAGE = 5


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load the database and initialize VSS extension
    db = load_email_db()
    db_connections["duckdb"] = load_email_content_search(db)
    yield
    # Clean up the DB connections
    if duckdb_con := db_connections.get("duckdb"):
        if hasattr(duckdb_con, "close"):
            duckdb_con.close()
    db_connections.clear()


# --- Setup and Configuration ---
# 1. Initialize FastAPI
app = FastAPI(title="FastAPI HTMX Email Client", lifespan=lifespan)

# 2. Configure Static Files (for CSS/JS/HTMX)
# Assumes static assets (output.css, bundle.js) are placed in a 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Configure Templating (for the base HTML page)
# Assumes index.html is placed in a 'templates' directory
templates = Jinja2Templates(directory="templates")


def create_list_item_fragment(
    email: Dict[str, Union[str, int]],
    is_last: bool = False,
    next_page: int = 0,
    query: str = "",
    folder: str = "",
) -> str:
    """Generates the HTML for a single email item."""
    preview_text = email["excerpt"]

    template = templates.get_template("email_list.jinja")
    output = template.render(
        email_id=email["message_id"],
        sender_name=email["from_email"],
        email_date=email["date"],
        email_subject=email["subject"],
        preview_text=preview_text,
        is_last=is_last,
        next_page=next_page,
        query=query,
        folder=folder,
    )
    return output


def create_detail_fragment(
    email_meta: Mapping[Any, Any],
    email_content: Optional[str],
    attachments: List[Dict[str, Union[str, bytes, int]]],
    is_in_thread: List[Dict[str, Union[str, int]]],
) -> str:
    """Generates the HTML for the email detail pane."""
    email_detail_template = templates.get_template("email_detail.jinja")
    thread_id = is_in_thread[0].get("thread_id") if is_in_thread else None
    output = email_detail_template.render(
        email_id=email_meta["message_id"],
        email_subject=email_meta["subject"],
        email_sender=email_meta["from_email"],
        email_date=email_meta["date"],
        email_body=email_content,
        has_attachment=email_meta["has_attachment"],
        attachments=attachments,
        thread=len(is_in_thread),
        thread_id=thread_id,
    )
    return output


def create_thread_detail_fragment(
    email_meta: Optional[Dict[str, Union[str, int]]],
    email_content: Optional[str],
    attachments: Optional[List[Dict[str, Union[str, bytes, int]]]],
    enriched_thread: List[Dict[str, Any]],
) -> str:
    """
    Generates the HTML for the thread detail pane.

    Args:
        email_meta: Unused (kept for backwards compatibility)
        email_content: Unused (kept for backwards compatibility)
        attachments: Unused (kept for backwards compatibility)
        enriched_thread: List of already-enriched email dicts with parsed content

    Returns:
        HTML string for thread detail view
    """
    email_thread_detail_template = templates.get_template("email_detail_thread.jinja")
    output = email_thread_detail_template.render(
        thread_emails=enriched_thread,
        thread_count=len(enriched_thread),
        thread_id=enriched_thread[0].get("thread_id") if enriched_thread else None,
    )
    return output


# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Route to serve the base HTML template."""
    # Templates.TemplateResponse requires the request object
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats/layout", response_class=HTMLResponse)
async def stats_layout(request: Request) -> HTMLResponse:
    """Route to serve the base HTML template."""
    from email_service import get_stats_summary

    stats_template = templates.get_template("stats.jinja")

    # Use service layer to get stats summary
    stats = get_stats_summary(db_connections["duckdb"])

    return HTMLResponse(
        content=stats_template.render(
            all_emails=stats["all_emails"],
            days_timespan=stats["days_timespan"],
            avg_size=stats["avg_size"],
        )
    )


@app.get("/api/stats/data/{query_name}", response_class=JSONResponse)
async def stats_data(query_name: str) -> List[Dict[str, Any]]:
    """Route to serve stats time series data."""
    from email_service import get_stats_time_series

    # Use service layer to get time series data
    return get_stats_time_series(db_connections["duckdb"], query_name)


@app.get("/api/inbox/layout", response_class=HTMLResponse)
async def inbox_layout(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("mail_list.jinja", {"request": request})


@app.get("/api/sent/layout", response_class=HTMLResponse)
async def sent_layout(request: Request) -> HTMLResponse:
    mail_list_template = templates.get_template("mail_list.jinja")
    rendered_mail_list = mail_list_template.render(folder="Sent")
    return HTMLResponse(content=rendered_mail_list)


@app.post("/api/search", response_class=HTMLResponse)
async def handle_search(search_input: Annotated[str, Form()]) -> HTMLResponse:
    parsed_search_query = parse_search_query(search_input)
    return await email_list(page=1, query=search_input)


@app.get("/api/email/list", response_class=HTMLResponse)
async def email_list(
    page: int = Query(1, ge=1),
    query: Optional[str] = None,
    folder: Optional[str] = None,
) -> HTMLResponse:
    """HTMX route to load the initial list and handle infinite scrolling."""
    from email_service import search_emails

    # Use service layer to get emails
    result = search_emails(
        db=db_connections["duckdb"],
        query=query,
        page=page,
        page_size=EMAILS_PER_PAGE,
        folder=folder,
    )

    page_emails = result["emails"]
    has_more = result["has_more"]
    next_page = result["next_page"]

    # Generate HTML fragments
    html_fragments = ""
    if has_more:
        for i, email in enumerate(page_emails):
            is_last = i == len(page_emails) - 1
            html_fragments += create_list_item_fragment(
                email,
                is_last=is_last,
                next_page=next_page,
                query=query or "",
                folder=folder or "",
            )
    else:
        for i, email in enumerate(page_emails):
            html_fragments += create_list_item_fragment(
                email,
                is_last=False,
                next_page=-1,
                query=query or "",
                folder=folder or "",
            )
        html_fragments += """
            <div class="text-center p-4 text-gray-600 border-t border-gray-700">End of Inbox.</div>
        """

    return HTMLResponse(content=html_fragments)


@app.get("/api/email/{email_id:path}", response_class=HTMLResponse)
async def email_detail(email_id: str) -> HTMLResponse:
    """HTMX route to load the detail pane content."""
    from email_service import get_email_with_thread

    # Use service layer to get email details
    result = get_email_with_thread(db=db_connections["duckdb"], email_id=email_id)

    if result is None:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return HTMLResponse(
        content=create_detail_fragment(
            result["email_meta"],
            result["email_content"],
            result["attachments"],
            result["thread"],
        )
    )


@app.get("/api/email_thread/{thread_id}", response_class=HTMLResponse)
async def email_thread_detail(thread_id: str) -> HTMLResponse:
    """HTMX route to load the detail pane content."""
    from email_service import get_thread_with_emails

    # Use service layer to get enriched thread emails
    enriched_thread = get_thread_with_emails(
        db=db_connections["duckdb"], thread_id=thread_id
    )

    if enriched_thread is None:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Thread not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return HTMLResponse(
        content=create_thread_detail_fragment(None, None, None, enriched_thread)
    )


@app.get("/api/attachment/{email_id:path}/{attachment_id}", response_class=Response)
async def get_attachment(email_id: str, attachment_id: str) -> Response:
    """HTMX route to load the detail pane content."""

    attachment = get_attachment_file(db_connections["duckdb"], email_id, attachment_id)

    if not attachment or "content" not in attachment:
        return Response(
            content="Attachment not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    content_bytes = attachment["content"]
    content_type_val = attachment["content_type"]

    if isinstance(content_bytes, bytes) and isinstance(content_type_val, str):
        return Response(
            content=content_bytes,
            media_type=content_type_val,
            headers={"content-disposition": f'attachment; filename="{attachment_id}"'},
        )
    else:
        return Response(
            content="Invalid attachment data",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("email_server:app", host="0.0.0.0", port=8000, reload=True)
