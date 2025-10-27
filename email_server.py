import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from email_utils import (
    EMAIL_DETAILS,
    get_email_count,
    get_email_content,
    get_email_list,
    get_one_email,
    load_email_content_search,
    load_email_db,
)

db_connections = {}

EMAILS_PER_PAGE = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    db_connections["duckdb"] = load_email_db()
    db_connections["chromadb"] = load_email_content_search()
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


def create_list_item_fragment(email, is_last: bool = False, next_page: int = 0):
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
        next_page=next_page
    )
    return output


def create_detail_fragment(email_meta, email_content):
    """Generates the HTML for the email detail pane."""
    email_detail_template = templates.get_template("email_detail.jinja")
    output = email_detail_template.render(
        email_subject=email_meta["subject"],
        email_sender=email_meta["from_email"],
        email_date=email_meta["date"],
        email_body=email_content,
    )
    return output


# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Route to serve the base HTML template."""
    # Templates.TemplateResponse requires the request object
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/email/list", response_class=HTMLResponse)
async def email_list(page: int = Query(1, ge=1)):
    """HTMX route to load the initial list and handle infinite scrolling."""

    start_index = (page - 1) * EMAILS_PER_PAGE
    end_index = start_index + EMAILS_PER_PAGE

    page_emails = get_email_list(
        db_connections["duckdb"],
        criteria={"limit": EMAILS_PER_PAGE, "offset": start_index},
    ).to_dict(orient="records")
    all_emails = get_email_count(db_connections["duckdb"])
    html_fragments = ""

    has_more = end_index < all_emails

    if has_more:
        next_page = page + 1
        for i, email in enumerate(page_emails):
            is_last = i == len(page_emails)-1
            html_fragments += create_list_item_fragment(email, is_last=is_last, next_page=next_page)
    else:
        for i, email in enumerate(page_emails):
            html_fragments += create_list_item_fragment(email, is_last=False, next_page=-1)
        html_fragments += """
            <div class="text-center p-4 text-gray-600 border-t border-gray-700">End of Inbox.</div>
        """

    return HTMLResponse(content=html_fragments)


@app.get("/api/email/{email_id}", response_class=HTMLResponse)
async def email_detail(email_id: str):
    """HTMX route to load the detail pane content."""

    email_meta = get_one_email(db_connections["duckdb"], email_id).to_dict(orient="records")
    if isinstance(email_meta, list) and email_meta:
        email_meta = email_meta[0]

    if email_meta:
        email = get_email_content(email_meta.get('email_line_start'), email_meta.get('email_line_end'))

        return HTMLResponse(content=create_detail_fragment(email_meta, email))
    else:
        return HTMLResponse(
            content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>",
            status_code=status.HTTP_404_NOT_FOUND,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
