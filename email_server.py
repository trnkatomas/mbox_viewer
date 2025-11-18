import datetime
import time
from contextlib import asynccontextmanager
import re
from typing import Optional, Annotated, Dict, List, Union, AsyncGenerator, TYPE_CHECKING, cast

import pandas as pd

from fastapi import FastAPI, Query, Request, status, Form
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from email_utils import (
    EMAIL_DETAILS,
    get_email_count,
    get_email_content,
    get_email_list,
    get_one_email,
    get_one_thread,
    load_email_content_search,
    load_email_db,
    get_string_email_from_mboxfile,
    parse_email,
    get_attachment_file,
    get_thread_for_email,
    get_basic_stats,
    get_email_sizes_in_time,
)

if TYPE_CHECKING:
    import chromadb
    import duckdb

db_connections: Dict[str, Union["chromadb.Collection", "duckdb.DuckDBPyConnection"]] = {}

EMAILS_PER_PAGE = 5


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load the ML model
    db_connections["duckdb"] = load_email_db()
    db_connections["chromadb"] = load_email_content_search()
    yield
    # Clean up the DB connections
    if duckdb_con := db_connections.get("duckdb"):
        duckdb_con.close()  # type: ignore
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
    email_thread_detail_template = templates.get_template("email_detail_thread.jinja")
    output = email_thread_detail_template.render(
        email_id=thread[0]["message_id"],
        email_subject=thread[0]["subject"],
        email_sender=thread[0]["from_email"],
        email_date=thread[0]["date"],
        email_body=thread[0],
        has_attachment=thread[0]["has_attachment"],
        attachments=[],
        thread=len(thread),
        thread_id=thread[0].get("thread_id"),
    )
    return output


def parse_search_input(query: str) -> Dict[str, str]:
    regex = r"(from|subject|rag|label):(\"(.*)\"|([^ \n]+))"
    matches = re.finditer(regex, query, re.MULTILINE)
    matches_dict: Dict[str, str] = {}
    to_discard: List[int] = []
    for matchNum, match in enumerate(matches, start=1):
        print(
            "Match {matchNum} was found at {start}-{end}: {match}".format(
                matchNum=matchNum,
                start=match.start(),
                end=match.end(),
                match=match.group(),
            )
        )
        to_discard.extend(list(range(match.start(), match.end())))
        matches_dict.update({match[1]: match[2]})
    remainder = [c for i, c in enumerate(query) if i not in to_discard]
    matches_dict["excerpt"] = "".join(remainder)
    return matches_dict


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

    basic_stats = get_basic_stats(db_connections["duckdb"])  # type: ignore
    all_emails = basic_stats[0].to_dict(orient="records")[0].get("all_emails", 0)
    avg_size = basic_stats[1].to_dict(orient="records")[0].get("avg_size", 0)
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
async def stats_data(query_name: str) -> Union[List[Dict[str, Union[str, int, float]]], Dict[str, str]]:
    """Route to serve the base HTML template."""
    if query_name == "dates_size":
        basic_stats = get_email_sizes_in_time(db_connections["duckdb"])  # type: ignore
        if not basic_stats.empty:
            return basic_stats.to_dict(orient="records")  # type: ignore[return-value]
    return {}


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
    parsed_search_query = parse_search_input(search_input)
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

    if query:
        additional_criteria = parse_search_input(query)
    else:
        additional_criteria = None

    page_emails = get_email_list(
        db_connections["duckdb"],  # type: ignore
        criteria={"limit": EMAILS_PER_PAGE, "offset": start_index},
        additional_criteria=additional_criteria,
        sent=folder == "Sent",
    ).to_dict(orient="records")
    all_emails = get_email_count(db_connections["duckdb"])  # type: ignore
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


@app.get("/api/email/{email_id}", response_class=HTMLResponse)
async def email_detail(email_id: str) -> HTMLResponse:
    """HTMX route to load the detail pane content."""

    email_meta_list = get_one_email(db_connections["duckdb"], email_id).to_dict(  # type: ignore
        orient="records"
    )
    if isinstance(email_meta_list, list) and email_meta_list:
        email_meta: Dict[str, Union[str, int]] = email_meta_list[0]  # type: ignore[assignment]
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    email_raw_string = get_string_email_from_mboxfile(
        int(email_meta.get("email_line_start", 0)),
        int(email_meta.get("email_line_end", 0)),
    )
    parsed_email = parse_email(email_raw_string)
    attachments = parsed_email.get("attachments", [])
    email_body = parsed_email.get("body", ("", None))
    is_in_thread = get_thread_for_email(db_connections["duckdb"], email_id).to_dict(  # type: ignore
        orient="records"
    )

    if isinstance(attachments, list) and isinstance(email_body, tuple):
        return HTMLResponse(
            content=create_detail_fragment(
                email_meta,
                email_body[1],
                attachments,
                is_in_thread,  # type: ignore
            )
        )
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Failed to parse email.</div>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.get("/api/email_thread/{thread_id}", response_class=HTMLResponse)
async def email_thread_detail(thread_id: str) -> HTMLResponse:
    """HTMX route to load the detail pane content."""

    thread_meta = get_one_thread(db_connections["duckdb"], thread_id).to_dict(  # type: ignore
        orient="records"
    )

    if thread_meta:
        # for each email in thread
        # email_raw_string = get_string_email_from_mboxfile(thread_meta.get('email_line_start'), thread_meta.get('email_line_end'))
        # parsed_email = parse_email(email_raw_string)
        # attachments = parsed_email.get('attachments')
        # email_content = parsed_email.get('body')
        # is_in_thread = get_thread_for_email(db_connections['duckdb'], the).to_dict(orient='records')
        return HTMLResponse(
            content=create_thread_detail_fragment(None, None, None, thread_meta)  # type: ignore[arg-type]
        )
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )


@app.get("/api/attachment/{email_id}/{attachment_id}", response_class=Response)
async def get_attachment(email_id: str, attachment_id: str) -> Response:
    """HTMX route to load the detail pane content."""

    attachment = get_attachment_file(db_connections["duckdb"], email_id, attachment_id)  # type: ignore

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
