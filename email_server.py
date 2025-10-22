from fastapi import FastAPI, Request, status, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time

from email_utils import EMAIL_DETAILS, load_email_db, load_email_content_search, get_one_email, get_email_list
from typing import Optional

EMAILS_PER_PAGE = 5

from contextlib import asynccontextmanager

db_connections = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    db_connections['duckdb'] = load_email_db()
    db_connections['chromadb'] = load_email_content_search()
    yield
    # Clean up the DB connections
    if duckdb_con := db_connections.get('duckdb'):
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


def create_list_item_fragment(email):
    """Generates the HTML for a single email item."""
    preview_text = email['excerpt']

    template = templates.get_template('email_list.jinja')
    output = template.render(email_id=email['message_id'],
                             sender_name=email['from_email'],
                             email_date=email['date'],
                             email_subject=email['subject'],
                             preview_text=preview_text)
    return output


def create_detail_fragment(email):
    """Generates the HTML for the email detail pane."""
    email_detail_template = templates.get_template('email_detail.jinja')
    output = email_detail_template.render(email_subject=email['subject'],
                                          email_sender=email['from_email'],
                                          email_date=email['date'],
                                          email_body=email['excerpt'])
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
    #page_emails = EMAIL_DETAILS[start_index:end_index]

    page_emails = get_email_list(db_connections['duckdb'], criteria=None).to_dict(orient='records')

    html_fragments = ""
    for email in page_emails:
        html_fragments += create_list_item_fragment(email)

    has_more = end_index < len(EMAIL_DETAILS)

    if has_more:
        next_page = page + 1
        # HTMX load-more trigger (note: hx-get sends the next page number)
        html_fragments += f"""
            <div id="load-more-trigger"
                class="text-center p-4 text-gray-500 hover:text-blue-400 cursor-pointer border-t border-gray-700/50"
                hx-get="/api/email/list?page={next_page}" 
                hx-trigger="intersect once"
                hx-swap="outerHTML"
            >
                <span class="animate-pulse">Loading more emails (Page {next_page})...</span>
            </div>
        """
    else:
        html_fragments += """
            <div class="text-center p-4 text-gray-600 border-t border-gray-700">End of Inbox.</div>
        """
        
    return HTMLResponse(content=html_fragments)

@app.get("/api/email/{email_id}", response_class=HTMLResponse)
async def email_detail(email_id: str):
    """HTMX route to load the detail pane content."""
    
    #email = next((e for e in EMAIL_DETAILS if e['id'] == email_id), None)
    email = get_one_email(db_connections['duckdb'], email_id).to_dict(orient='records')
    if isinstance(email, list) and email:
        email = email[0]

    if email:
        return HTMLResponse(content=create_detail_fragment(email))
    else:
        return HTMLResponse(content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>", 
                            status_code=status.HTTP_404_NOT_FOUND)





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)