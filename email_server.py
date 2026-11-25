import datetime
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
    Mailbox,
    RagUnavailableError,
    check_index_freshness,
    discover_mailboxes,
    load_email_db,
)

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

EMAILS_PER_PAGE = 25
MAILBOX_COOKIE = "mailbox"
MAILBOX_HEADER = "X-Mailbox"


class MailboxUnavailable(Exception):
    """The requested mailbox doesn't exist or has no index yet."""


class MailboxRegistry:
    """
    Known mailboxes and a lazily opened read-only connection per mailbox.
    Connections stay open, so the indexer can't rewrite an index while the
    viewer is running (DuckDB allows one writer or many readers).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mailboxes: Dict[str, Mailbox] = {}
        self._connections: Dict[str, "duckdb.DuckDBPyConnection"] = {}
        self._problems: Dict[str, Optional[str]] = {}

    def refresh(self) -> List[Mailbox]:
        mailboxes = discover_mailboxes()
        with self._lock:
            self._mailboxes = {m.id: m for m in mailboxes}
        return mailboxes

    def all(self) -> List[Mailbox]:
        return list(self._mailboxes.values())

    def get(self, mailbox_id: str) -> Optional[Mailbox]:
        if mailbox_id not in self._mailboxes:
            self.refresh()  # a new mbox may have been added since the last scan
        return self._mailboxes.get(mailbox_id)

    def default(self) -> Optional[Mailbox]:
        if not self._mailboxes:
            self.refresh()
        return next(iter(self._mailboxes.values()), None)

    def connection(self, mailbox: Mailbox) -> "duckdb.DuckDBPyConnection":
        with self._lock:
            con = self._connections.get(mailbox.id)
            if con is None:
                if not mailbox.indexed:
                    raise MailboxUnavailable(
                        f"The mailbox \u201c{mailbox.name}\u201d hasn't been indexed yet. "
                        f"Run `python email_utils.py index --mailbox {mailbox.id}`."
                    )
                con = self._connections[mailbox.id] = load_email_db(mailbox.db_path)
                self._problems[mailbox.id] = _index_problem(con, mailbox)
            return con

    def problem(self, mailbox: Mailbox) -> Optional[str]:
        """Why the index can't be trusted (e.g. the mbox was replaced), if it can't."""
        return self._problems.get(mailbox.id)

    def close(self) -> None:
        with self._lock:
            for con in self._connections.values():
                con.close()
            self._connections.clear()
            self._problems.clear()


def _index_problem(db: "duckdb.DuckDBPyConnection", mailbox: Mailbox) -> Optional[str]:
    """Messages are read by byte offset, so a replaced mbox would show the wrong mail."""
    try:
        problem = check_index_freshness(db, mailbox.mbox_path)
    except Exception as e:
        logger.warning("Could not verify the index of %r against its mbox: %s", mailbox.name, e)
        return None
    if problem:
        logger.error(
            "The index of %r does not match its mbox (%s); messages will not display correctly. "
            "Run `python email_utils.py index --mailbox %s --rebuild`.",
            mailbox.name,
            problem,
            mailbox.id,
        )
    return problem


registry = MailboxRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    mailboxes = registry.refresh()
    if mailboxes:
        logger.info("Mailboxes: %s", ", ".join(f"{m.id} ({m.mbox_path})" for m in mailboxes))
    else:
        logger.warning("No mailboxes found: set MBOX_FILE_PATH or MBOX_DIR")
    yield
    registry.close()


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


def resolve_mailbox(request: Request) -> Mailbox:
    """
    The mailbox a request is for: the X-Mailbox header (sent by every htmx
    request on the page), a ?mailbox= query parameter (plain links and chart
    data), then the remembered cookie, then the first mailbox. An explicit id
    that doesn't exist is an error rather than silently showing another mailbox.
    """
    requested = request.headers.get(MAILBOX_HEADER) or request.query_params.get("mailbox")
    if requested:
        mailbox = registry.get(requested)
        if mailbox is None:
            raise MailboxUnavailable(f"There is no mailbox \u201c{requested}\u201d.")
        return mailbox
    remembered = request.cookies.get(MAILBOX_COOKIE)
    mailbox = (registry.get(remembered) if remembered else None) or registry.default()
    if mailbox is None:
        raise MailboxUnavailable("No mailboxes found. Set MBOX_DIR or MBOX_FILE_PATH.")
    return mailbox


@dataclass
class MailboxContext:
    mailbox: Mailbox
    db: Any


def get_mailbox_context(
    mailbox: Annotated[Mailbox, Depends(resolve_mailbox)],
) -> Iterator[MailboxContext]:
    """A cursor per request: DuckDB connections must not be shared across threads."""
    cursor = registry.connection(mailbox).cursor()
    try:
        yield MailboxContext(mailbox=mailbox, db=cursor)
    finally:
        cursor.close()


Ctx = Annotated[MailboxContext, Depends(get_mailbox_context)]
CurrentMailbox = Annotated[Mailbox, Depends(resolve_mailbox)]


@app.exception_handler(MailboxUnavailable)
def mailbox_unavailable_handler(request: Request, exc: MailboxUnavailable) -> Response:
    if request.url.path.startswith("/api/stats/data/"):
        return JSONResponse([])
    # 200 so htmx swaps the message in place of the content
    return HTMLResponse(_message_fragment(str(exc), tone="amber"))


def _message_view(
    mailbox: Mailbox,
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
        "mailbox_id": mailbox.id,
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
def index(request: Request, mailbox: Optional[str] = None) -> HTMLResponse:
    mailboxes = registry.refresh()
    by_id = {m.id: m for m in mailboxes}
    current = (
        by_id.get(mailbox or "")
        or by_id.get(request.cookies.get(MAILBOX_COOKIE) or "")
        or (mailboxes[0] if mailboxes else None)
    )
    response = templates.TemplateResponse(
        request, "index.html", {"mailboxes": mailboxes, "current": current}
    )
    if current:
        response.set_cookie(
            MAILBOX_COOKIE, current.id, max_age=365 * 24 * 3600, httponly=True, samesite="lax"
        )
    return response


@app.get("/api/mailboxes", response_class=JSONResponse)
def list_mailboxes() -> List[Dict[str, Any]]:
    return [{"id": m.id, "name": m.name, "indexed": m.indexed} for m in registry.refresh()]


def _mailbox_notice(mailbox: Mailbox) -> Optional[str]:
    if not mailbox.indexed:
        return None  # the list itself explains this
    try:
        registry.connection(mailbox)
    except Exception as e:
        return f"The index can't be opened: {e}"
    problem = registry.problem(mailbox)
    if problem:
        return (
            f"This index doesn't match the mbox file ({problem}), so messages may show the "
            f"wrong content. Rebuild it with `python email_utils.py index --mailbox {mailbox.id} --rebuild`."
        )
    return None


@app.get("/api/stats/layout", response_class=HTMLResponse)
def stats_layout(ctx: Ctx) -> HTMLResponse:
    stats = email_service.get_stats_summary(ctx.db)
    return HTMLResponse(
        templates.get_template("stats.jinja").render(
            all_emails=stats["all_emails"],
            days_timespan=stats["days_timespan"],
            avg_size=stats["avg_size"],
        )
    )


@app.get("/api/stats/data/{query_name}", response_class=JSONResponse)
def stats_data(query_name: str, ctx: Ctx) -> List[Dict[str, Any]]:
    return email_service.get_stats_time_series(ctx.db, query_name)


def _mail_list_layout(mailbox: Mailbox, folder: str) -> HTMLResponse:
    return HTMLResponse(
        templates.get_template("mail_list.jinja").render(
            folder=folder, mailbox=mailbox, notice=_mailbox_notice(mailbox)
        )
    )


@app.get("/api/inbox/layout", response_class=HTMLResponse)
def inbox_layout(mailbox: CurrentMailbox) -> HTMLResponse:
    return _mail_list_layout(mailbox, "")


@app.get("/api/sent/layout", response_class=HTMLResponse)
def sent_layout(mailbox: CurrentMailbox) -> HTMLResponse:
    return _mail_list_layout(mailbox, "Sent")


@app.post("/api/search", response_class=HTMLResponse)
def handle_search(
    search_input: Annotated[str, Form()],
    ctx: Ctx,
    folder: Annotated[str, Form()] = "",
) -> HTMLResponse:
    return email_list(ctx=ctx, page=1, query=search_input, folder=folder)


@app.get("/api/email/list", response_class=HTMLResponse)
def email_list(
    ctx: Ctx,
    page: int = Query(1, ge=1),
    query: Optional[str] = None,
    folder: Optional[str] = None,
) -> HTMLResponse:
    """HTMX route for the initial list and infinite scrolling."""
    try:
        result = email_service.search_emails(
            db=ctx.db, query=query, page=page, page_size=EMAILS_PER_PAGE, folder=folder
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
def email_detail(email_id: str, ctx: Ctx, remote: bool = False) -> HTMLResponse:
    """HTMX route for the detail pane."""
    result = email_service.get_email_with_thread(
        db=ctx.db, email_id=email_id, mbox_path=ctx.mailbox.mbox_path
    )
    if result is None:
        return HTMLResponse(
            _message_fragment("Error: Email not found.", tone="red"),
            status_code=status.HTTP_404_NOT_FOUND,
        )

    email_meta = result["email_meta"]
    message = _message_view(
        ctx.mailbox,
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
def email_thread_detail(thread_id: str, ctx: Ctx, remote: bool = False) -> HTMLResponse:
    enriched_thread = email_service.get_thread_with_emails(
        db=ctx.db, thread_id=thread_id, mbox_path=ctx.mailbox.mbox_path
    )
    if enriched_thread is None:
        return HTMLResponse(
            _message_fragment("Error: Thread not found.", tone="red"),
            status_code=status.HTTP_404_NOT_FOUND,
        )

    messages = [
        _message_view(
            ctx.mailbox,
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
def get_attachment_route(email_id: str, index: int, ctx: Ctx) -> Response:
    attachment = email_service.get_attachment(ctx.db, email_id, index, ctx.mailbox.mbox_path)
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
