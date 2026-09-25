import datetime
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator, Dict, Iterator, List, Optional
from urllib.parse import quote, urlencode

from fastapi import Depends, FastAPI, Form, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import email_service
from email_render import render_email_frame
from email_utils import (
    InvalidQueryError,
    RagUnavailableError,
    check_index_freshness,
    get_mbox_path,
    load_email_db,
)

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

db_connections: Dict[str, "duckdb.DuckDBPyConnection"] = {}

EMAILS_PER_PAGE = 25


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_connections["duckdb"] = load_email_db()
    _warn_if_index_stale(db_connections["duckdb"])
    yield
    if duckdb_con := db_connections.get("duckdb"):
        duckdb_con.close()
    db_connections.clear()


def _warn_if_index_stale(db: "duckdb.DuckDBPyConnection") -> None:
    """Messages are read by byte offset, so a replaced mbox would show the wrong mail."""
    try:
        problem = check_index_freshness(db, get_mbox_path())
    except Exception as e:
        logger.warning("Could not verify the index against the mbox: %s", e)
        return
    if problem:
        logger.error(
            "The index does not match the mbox (%s); messages will not display correctly. "
            "Run `python email_utils.py index --rebuild`.",
            problem,
        )


app = FastAPI(title="FastAPI HTMX Email Client", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def format_date(value: Any) -> str:
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def format_size(size: Any) -> str:
    size = float(size or 0)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


templates.env.filters["format_date"] = format_date
templates.env.filters["format_size"] = format_size


def get_db() -> Iterator["duckdb.DuckDBPyConnection"]:
    """A cursor per request: DuckDB connections must not be shared across threads."""
    cursor = db_connections["duckdb"].cursor()
    try:
        yield cursor
    finally:
        cursor.close()


Db = Annotated[Any, Depends(get_db)]


def _message_view(
    email_meta: Dict[str, Any],
    body: Optional[str],
    body_type: str,
    attachments: List[Dict[str, Any]],
    inline_images: Optional[Dict[str, Any]],
    allow_remote: bool,
) -> Dict[str, Any]:
    frame_html, has_remote = render_email_frame(body, body_type, inline_images, allow_remote)
    return {
        "message_id": email_meta["message_id"],
        "subject": email_meta.get("subject"),
        "from_email": email_meta.get("from_email"),
        "to_email": email_meta.get("to_email"),
        "date": email_meta.get("date"),
        "attachments": attachments,
        "frame_html": frame_html,
        "remote_blocked": has_remote and not allow_remote,
    }


def _message_fragment(text: str, tone: str = "gray") -> str:
    template = templates.env.from_string(
        '<div class="p-6 text-center text-sm text-{{ tone }}-600">{{ text }}</div>'
    )
    return template.render(text=text, tone=tone)


def content_disposition(filename: str) -> str:
    """RFC 6266 header with an ASCII fallback and the exact UTF-8 name."""
    ascii_name = filename.encode("ascii", "ignore").decode().replace("\\", "_").replace('"', "_")
    ascii_name = "".join(ch for ch in ascii_name if ch.isprintable()) or "attachment"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/stats/layout", response_class=HTMLResponse)
def stats_layout(db: Db) -> HTMLResponse:
    stats = email_service.get_stats_summary(db)
    return HTMLResponse(
        templates.get_template("stats.jinja").render(
            all_emails=stats["all_emails"],
            days_timespan=stats["days_timespan"],
            avg_size=stats["avg_size"],
        )
    )


@app.get("/api/stats/data/{query_name}", response_class=JSONResponse)
def stats_data(query_name: str, db: Db) -> List[Dict[str, Any]]:
    return email_service.get_stats_time_series(db, query_name)


@app.get("/api/inbox/layout", response_class=HTMLResponse)
def inbox_layout() -> HTMLResponse:
    return HTMLResponse(templates.get_template("mail_list.jinja").render(folder=""))


@app.get("/api/sent/layout", response_class=HTMLResponse)
def sent_layout() -> HTMLResponse:
    return HTMLResponse(templates.get_template("mail_list.jinja").render(folder="Sent"))


@app.post("/api/search", response_class=HTMLResponse)
def handle_search(
    search_input: Annotated[str, Form()],
    db: Db,
    folder: Annotated[str, Form()] = "",
) -> HTMLResponse:
    return email_list(db=db, page=1, query=search_input, folder=folder)


@app.get("/api/email/list", response_class=HTMLResponse)
def email_list(
    db: Db,
    page: int = Query(1, ge=1),
    query: Optional[str] = None,
    folder: Optional[str] = None,
) -> HTMLResponse:
    """HTMX route for the initial list and infinite scrolling."""
    try:
        result = email_service.search_emails(
            db=db, query=query, page=page, page_size=EMAILS_PER_PAGE, folder=folder
        )
    except (RagUnavailableError, InvalidQueryError) as e:
        return HTMLResponse(_message_fragment(str(e), tone="red"))

    next_url = None
    if result["has_more"]:
        params = {"page": result["next_page"], "query": query, "folder": folder}
        next_url = "/api/email/list?" + urlencode({k: v for k, v in params.items() if v})

    item_template = templates.get_template("email_list.jinja")
    emails = result["emails"]
    html_fragments = "".join(
        item_template.render(
            email=email,
            next_url=next_url if i == len(emails) - 1 else None,
        )
        for i, email in enumerate(emails)
    )
    if not result["has_more"]:
        text = "No more emails." if emails or page > 1 else "No emails match this search."
        html_fragments += _message_fragment(text)
    return HTMLResponse(html_fragments)


@app.get("/api/email/{email_id:path}", response_class=HTMLResponse)
def email_detail(email_id: str, db: Db, remote: bool = False) -> HTMLResponse:
    """HTMX route for the detail pane."""
    result = email_service.get_email_with_thread(db=db, email_id=email_id)
    if result is None:
        return HTMLResponse(
            _message_fragment("Error: Email not found.", tone="red"),
            status_code=status.HTTP_404_NOT_FOUND,
        )

    email_meta = result["email_meta"]
    message = _message_view(
        email_meta,
        result["email_content"],
        result.get("body_type", "HTML"),
        result["attachments"] or [],
        result.get("inline_images"),
        remote,
    )
    thread = result["thread"]
    return HTMLResponse(
        templates.get_template("email_detail.jinja").render(
            message=message,
            thread_count=len(thread),
            thread_id=email_meta.get("thread_id") if len(thread) > 1 else None,
            remote_url=f"/api/email/{quote(email_id, safe='')}?remote=true",
        )
    )


@app.get("/api/email_thread/{thread_id}", response_class=HTMLResponse)
def email_thread_detail(thread_id: str, db: Db, remote: bool = False) -> HTMLResponse:
    enriched_thread = email_service.get_thread_with_emails(db=db, thread_id=thread_id)
    if enriched_thread is None:
        return HTMLResponse(
            _message_fragment("Error: Thread not found.", tone="red"),
            status_code=status.HTTP_404_NOT_FOUND,
        )

    messages = [
        _message_view(
            email,
            email.get("parsed_body"),
            email.get("body_type", "HTML"),
            email.get("attachments") or [],
            email.get("inline_images"),
            remote,
        )
        for email in enriched_thread
    ]
    return HTMLResponse(
        templates.get_template("email_detail_thread.jinja").render(
            messages=messages,
            thread_count=len(messages),
            remote_url=f"/api/email_thread/{quote(thread_id, safe='')}?remote=true",
        )
    )


@app.get("/api/attachment/{email_id:path}/{index:int}", response_class=Response)
def get_attachment_route(email_id: str, index: int, db: Db) -> Response:
    attachment = email_service.get_attachment(db, email_id, index)
    if not attachment or not isinstance(attachment.get("content"), bytes):
        return Response("Attachment not found", status_code=status.HTTP_404_NOT_FOUND)

    return Response(
        content=attachment["content"],
        media_type=attachment.get("content_type") or "application/octet-stream",
        headers={
            "Content-Disposition": content_disposition(attachment.get("filename") or "attachment"),
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "email_server:app",
        host=os.getenv("MBOX_VIEWER_HOST", "127.0.0.1"),
        port=int(os.getenv("MBOX_VIEWER_PORT", "8000")),
        reload=True,
    )
