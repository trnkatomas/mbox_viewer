import datetime
import time
from contextlib import asynccontextmanager
import re
from typing import Optional, Annotated

from fastapi import FastAPI, Query, Request, status, Form
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

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

db_connections = {}

EMAILS_PER_PAGE = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the database and initialize VSS extension
    db_connections["duckdb"] = load_email_db()
    db_connections["duckdb"] = load_email_content_search(db_connections["duckdb"])
    yield
    # Clean up the DB connections
    if duckdb_con := db_connections.get("duckdb"):
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
    email, is_last: bool = False, next_page: int = 0, query: str = "", folder: str = ""
):
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


def create_detail_fragment(email_meta, email_content, attachments, is_in_thread):
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


def create_thread_detail_fragment(email_meta, email_content, attachments, thread):
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


def parse_search_input(query: str):
    """
    Parse search query for special filters.

    Supported filters:
    - from:email - Filter by sender
    - subject:text - Filter by subject
    - rag:text - Semantic search
    - from_date:YYYY-MM-DD - Filter from date
    - to_date:YYYY-MM-DD - Filter to date
    """
    regex = r"(from|subject|rag|from_date|to_date|label):(\"(.*)\"|([^ \n]+))"
    matches = re.finditer(regex, query, re.MULTILINE)
    matches_dict = {}
    to_discard = []
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
        # Extract the value (either quoted or unquoted)
        value = match[3] if match[3] else match[4]
        matches_dict.update({match[1]: value})
    remainder = [c for i, c in enumerate(query) if i not in to_discard]
    matches_dict["excerpt"] = "".join(remainder).strip()
    return matches_dict


# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Route to serve the base HTML template."""
    # Templates.TemplateResponse requires the request object
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats/layout", response_class=HTMLResponse)
async def stats_layout(request: Request):
    """Route to serve the base HTML template."""
    stats_template = templates.get_template("stats.jinja")

    basic_stats = get_basic_stats(db_connections["duckdb"])
    all_emails = basic_stats[0].to_dict(orient="records")[0].get("all_emails")
    avg_size = basic_stats[1].to_dict(orient="records")[0].get("avg_size")
    first_seen = basic_stats[2].to_dict(orient="records")[0].get("first_seen")
    last_seen = basic_stats[2].to_dict(orient="records")[0].get("last_seen")
    return HTMLResponse(
        content=stats_template.render(
            all_emails=all_emails,
            days_timespan=(last_seen - first_seen).days / 365,
            avg_size=avg_size,
        )
    )


@app.get("/api/stats/data/{query_name}", response_class=JSONResponse)
async def stats_layout(query_name: str):
    """Route to serve the base HTML template."""
    if query_name == "dates_size":
        basic_stats = get_email_sizes_in_time(db_connections["duckdb"])
        if not basic_stats.empty:
            return basic_stats.to_dict(orient="records")
    else:
        return {}


@app.get("/api/inbox/layout", response_class=HTMLResponse)
async def inbox_layout(request: Request):
    return templates.TemplateResponse("mail_list.jinja", {"request": request})


@app.get("/api/sent/layout", response_class=HTMLResponse)
async def inbox_layout(request: Request):
    mail_list_template = templates.get_template("mail_list.jinja")
    rendered_mail_list = mail_list_template.render(folder="Sent")
    return HTMLResponse(content=rendered_mail_list)


@app.post("/api/search", response_class=HTMLResponse)
async def handle_search(search_input: Annotated[str, Form()]):
    parsed_search_query = parse_search_input(search_input)
    return await email_list(page=1, query=search_input)


@app.get("/api/email/list", response_class=HTMLResponse)
async def email_list(
    page: int = Query(1, ge=1),
    query: Optional[str] = None,
    folder: Optional[str] = None,
):
    """HTMX route to load the initial list and handle infinite scrolling."""

    start_index = (page - 1) * EMAILS_PER_PAGE
    end_index = start_index + EMAILS_PER_PAGE

    # Parse search query if provided
    if query:
        additional_criteria = parse_search_input(query)
    else:
        additional_criteria = None

    # Handle RAG semantic search if rag: query provided
    rag_message_ids = None
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

    page_emails_df = get_email_list(
        db_connections["duckdb"],
        criteria={"limit": EMAILS_PER_PAGE, "offset": start_index},
        additional_criteria=additional_criteria,
        sent=folder == "Sent",
        rag_message_ids=rag_message_ids[start_index:EMAILS_PER_PAGE]["message_id"].tolist() if rag_message_ids is not None else None,
    )
    if rag_message_ids is not None:
        page_emails_df = pd.merge(
            page_emails_df, rag_message_ids[['message_id', 'dist']], on="message_id"
        ).sort_values(by="dist", ascending=True)

    page_emails = page_emails_df.to_dict(orient="records")
    all_emails = get_email_count(db_connections["duckdb"]) # TODO this should reflect the size of the currently retrieved results
    html_fragments = ""

    has_more = end_index < all_emails

    if has_more:
        next_page = page + 1
        for i, email in enumerate(page_emails):
            is_last = i == len(page_emails) - 1
            html_fragments += create_list_item_fragment(
                email, is_last=is_last, next_page=next_page, query=query, folder=folder
            )
    else:
        for i, email in enumerate(page_emails):
            html_fragments += create_list_item_fragment(
                email, is_last=False, next_page=-1, query=query, folder=folder
            )
        html_fragments += """
            <div class="text-center p-4 text-gray-600 border-t border-gray-700">End of Inbox.</div>
        """

    return HTMLResponse(content=html_fragments)


@app.get("/api/email/{email_id}", response_class=HTMLResponse)
async def email_detail(email_id: str):
    """HTMX route to load the detail pane content."""

    email_meta = get_one_email(db_connections["duckdb"], email_id).to_dict(
        orient="records"
    )
    if isinstance(email_meta, list) and email_meta:
        email_meta = email_meta[0]

    if email_meta:
        email_raw_string = get_string_email_from_mboxfile(
            email_meta.get("email_start"), email_meta.get("email_end")
        )
        parsed_email = parse_email(email_raw_string)
        attachments = parsed_email.get("attachments")
        email_content = parsed_email.get("body")
        is_in_thread = get_thread_for_email(db_connections["duckdb"], email_id).to_dict(
            orient="records"
        )
        return HTMLResponse(
            content=create_detail_fragment(
                email_meta, email_content[1], attachments, is_in_thread
            )
        )
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )


@app.get("/api/email_thread/{thread_id}", response_class=HTMLResponse)
async def email_thread_detail(thread_id: str):
    """HTMX route to load the detail pane content."""

    thread_meta = get_one_thread(db_connections["duckdb"], thread_id).to_dict(
        orient="records"
    )

    if thread_meta:
        # for each email in thread
        # email_raw_string = get_string_email_from_mboxfile(thread_meta.get('email_start'), thread_meta.get('email_end'))
        # parsed_email = parse_email(email_raw_string)
        # attachments = parsed_email.get('attachments')
        # email_content = parsed_email.get('body')
        # is_in_thread = get_thread_for_email(db_connections['duckdb'], the).to_dict(orient='records')
        return HTMLResponse(
            content=create_thread_detail_fragment(None, None, None, thread_meta)
        )
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )


@app.get("/api/attachment/{email_id}/{attachment_id}", response_class=Response)
async def get_attachment(email_id: str, attachment_id: str):
    """HTMX route to load the detail pane content."""

    attachment = get_attachment_file(db_connections["duckdb"], email_id, attachment_id)

    if not attachment or "content" not in attachment:
        return Response(
            content="Attachment not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return Response(
        content=attachment["content"],
        media_type=attachment["content_type"],
        headers={"content-disposition": f'attachment; filename="{attachment_id}"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("email_server:app", host="0.0.0.0", port=8000, reload=True)
