import datetime
import logging
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import (TYPE_CHECKING, Annotated, Any, AsyncGenerator, Dict, List,
                    Optional, Union)

logger = logging.getLogger(__name__)

import pandas as pd
from fastapi import FastAPI, Form, Query, Request, status
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               Response)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from email_service import parse_search_query
from email_utils import (get_attachment_file, get_basic_stats,
                         get_domains_by_count, get_email_count, get_email_list,
                         get_email_sizes_in_time, get_one_email,
                         get_one_thread, get_string_email_from_mboxfile,
                         get_thread_for_email, load_email_content_search,
                         load_email_db, parse_email)

if TYPE_CHECKING:
    import duckdb

db_connections: Dict[str, Union["duckdb.DuckDBPyConnection"]] = {}

EMAILS_PER_PAGE = 5


# Cached wrapper functions for stats (cached with LRU, maxsize=128)
@lru_cache(maxsize=128)
def get_cached_basic_stats() -> List[pd.DataFrame]:
    """Cached version of get_basic_stats."""
    return get_basic_stats(db_connections["duckdb"])


@lru_cache(maxsize=128)
def get_cached_email_sizes_in_time() -> pd.DataFrame:
    """Cached version of get_email_sizes_in_time."""
    return get_email_sizes_in_time(db_connections["duckdb"])


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
    email_meta: Dict[str, Union[str, int]],
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
    thread: List[Dict[str, Union[str, int]]],
) -> str:
    """Generates the HTML for the email detail pane."""
    # Parse content for each email in the thread
    enriched_thread = []
    for email in thread:
        email_start = email.get("email_start")
        email_end = email.get("email_end")
        if isinstance(email_start, int) and isinstance(email_end, int):
            email_raw_string = get_string_email_from_mboxfile(email_start, email_end)
            parsed_email = parse_email(email_raw_string)
            # Enrich the email dict with parsed content
            enriched_email: Dict[str, Any] = dict(
                email
            )  # Create a copy with flexible typing
            enriched_email["parsed_body"] = parsed_email.get("body", ("", ""))[
                1
            ]  # Get HTML content
            enriched_email["attachments"] = parsed_email.get("attachments", [])
            enriched_thread.append(enriched_email)
        else:
            # If we can't parse, just add the email as-is
            enriched_thread.append(email)

    email_thread_detail_template = templates.get_template("email_detail_thread.jinja")
    output = email_thread_detail_template.render(
        thread_emails=enriched_thread,
        thread_count=len(thread),
        thread_id=thread[0].get("thread_id") if thread else None,
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
    stats_template = templates.get_template("stats.jinja")

    basic_stats = get_cached_basic_stats()
    all_emails = basic_stats[0].to_dict(orient="records")[0].get("all_emails")
    avg_size = basic_stats[1].to_dict(orient="records")[0].get("avg_size")
    first_seen = basic_stats[2].to_dict(orient="records")[0].get("first_seen")
    last_seen = basic_stats[2].to_dict(orient="records")[0].get("last_seen")

    days_timespan = 0
    if first_seen is not None and last_seen is not None:
        days_timespan = (last_seen - first_seen).days / 365

    return HTMLResponse(
        content=stats_template.render(
            all_emails=all_emails,
            days_timespan=days_timespan,
            avg_size=avg_size,
        )
    )


@app.get("/api/stats/data/{query_name}", response_class=JSONResponse)
async def stats_data(query_name: str) -> List[Dict[str, Any]]:
    """Route to serve the base HTML template."""
    if query_name == "dates_size":
        basic_stats = get_cached_email_sizes_in_time()
        if not basic_stats.empty:
            # Convert date column to ISO format string for JSON serialization if it's datetime
            if pd.api.types.is_datetime64_any_dtype(basic_stats["date"]):
                basic_stats["date"] = basic_stats["date"].dt.strftime("%Y-%m-%d")
            result: List[Dict[str, Any]] = basic_stats.to_dict(orient="records")  # type: ignore[assignment]
            return result
    elif query_name == "domains_count":
        domain_stats = get_domains_by_count(db_connections["duckdb"])
        if not domain_stats.empty:
            result: List[Dict[str, Any]] = domain_stats.to_dict(orient="records")  # type: ignore[assignment]
            return result
    return []


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

    start_index = (page - 1) * EMAILS_PER_PAGE
    end_index = start_index + EMAILS_PER_PAGE

    # Parse search query if provided
    if query:
        additional_criteria = parse_search_query(query)
    else:
        additional_criteria = None

    # Handle RAG semantic search if rag: query provided
    rag_message_ids: Optional[pd.DataFrame] = None
    if additional_criteria and "rag" in additional_criteria:
        from email_utils import rag_search_duckdb

        rag_query = additional_criteria.pop(
            "rag"
        )  # Remove from criteria, handle separately
        rag_message_ids = rag_search_duckdb(
            db_connections["duckdb"],
            rag_query,
            n_results=100,  # Get more RAG results, then filter/paginate
        )

    db_conn = db_connections["duckdb"]
    page_emails_df = get_email_list(
        db_conn,
        criteria=(
            {"limit": EMAILS_PER_PAGE, "offset": start_index}
            if rag_message_ids is None
            else {"limit": EMAILS_PER_PAGE, "offset": 0}
        ),
        additional_criteria=additional_criteria,
        sent=folder == "Sent",
        rag_message_ids=(
            rag_message_ids[start_index:end_index]["message_id"].tolist()
            if rag_message_ids is not None and not rag_message_ids.empty
            else None
        ),
    )
    if rag_message_ids is not None and not rag_message_ids.empty:
        page_emails_df = pd.merge(
            page_emails_df, rag_message_ids[["message_id", "dist"]], on="message_id"
        ).sort_values(by="dist", ascending=True)

    page_emails = page_emails_df.to_dict(orient="records")
    if rag_message_ids is not None and not rag_message_ids.empty:
        all_emails = rag_message_ids.shape[0]
    elif additional_criteria:
        all_emails = get_email_count(db_conn, additional_criteria)
    else:
        all_emails = get_email_count(
            db_conn
        )  # TODO this should reflect the size of the currently retrieved results
    html_fragments = ""

    has_more = end_index < all_emails

    if has_more:
        next_page = page + 1
        for i, email in enumerate(page_emails):
            is_last = i == len(page_emails) - 1
            html_fragments += create_list_item_fragment(
                email,  # type: ignore[arg-type]
                is_last=is_last,
                next_page=next_page,
                query=query or "",
                folder=folder or "",
            )
    else:
        for i, email in enumerate(page_emails):
            html_fragments += create_list_item_fragment(
                email,  # type: ignore[arg-type]
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

    email_meta_list = get_one_email(db_connections["duckdb"], email_id).to_dict(
        orient="records"
    )
    if isinstance(email_meta_list, list) and email_meta_list:
        email_meta: Dict[str, Union[str, int]] = email_meta_list[0]  # type: ignore[assignment]
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    if email_meta:
        email_start = email_meta.get("email_start")
        email_end = email_meta.get("email_end")
        if isinstance(email_start, int) and isinstance(email_end, int):
            email_raw_string = get_string_email_from_mboxfile(email_start, email_end)
        else:
            return HTMLResponse(
                content="<div class='p-8 text-center text-red-400'>Error: Invalid email boundaries.</div>",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        parsed_email = parse_email(email_raw_string)
        attachments = parsed_email.get("attachments")
        email_content = parsed_email.get("body")
        is_in_thread = get_thread_for_email(db_connections["duckdb"], email_id).to_dict(
            orient="records"
        )
        return HTMLResponse(
            content=create_detail_fragment(
                email_meta,
                email_content[1],
                attachments,
                is_in_thread,  # type: ignore[index, arg-type]
            )
        )
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )


@app.get("/api/email_thread/{thread_id}", response_class=HTMLResponse)
async def email_thread_detail(thread_id: str) -> HTMLResponse:
    """HTMX route to load the detail pane content."""

    thread_meta = get_one_thread(db_connections["duckdb"], thread_id).to_dict(
        orient="records"
    )

    if thread_meta:
        return HTMLResponse(
            content=create_thread_detail_fragment(None, None, None, thread_meta)  # type: ignore[arg-type]
        )
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
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
