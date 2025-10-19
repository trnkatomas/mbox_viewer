from fastapi import FastAPI, Request, status, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
from typing import Optional

# --- Setup and Configuration ---
# 1. Initialize FastAPI
app = FastAPI(title="FastAPI HTMX Email Client")

# 2. Configure Static Files (for CSS/JS/HTMX)
# Assumes static assets (output.css, bundle.js) are placed in a 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Configure Templating (for the base HTML page)
# Assumes index.html is placed in a 'templates' directory
templates = Jinja2Templates(directory="templates")

# --- Dummy Data and Constants (Same as before) ---
EMAIL_DETAILS = [
    {
        "id": 1,
        "subject": "Project Status Update: Q4 Goals",
        "sender": "Jane Doe <jane.doe@corp.com>",
        "date": "Oct 15, 2025",
        "body": """
            <p class="text-gray-300">Hi Team,</p>
            <p class="mt-4 text-gray-300">I wanted to provide a quick update on our Q4 goals. The new design system migration is 90% complete, and we are on track to launch the new marketing site by Friday.</p>
            <p class="mt-4 text-gray-300">Please review the attached mockups (simulated) and provide feedback by end-of-day tomorrow. We can discuss any blockers during our stand-up.</p>
            <p class="mt-8 text-gray-300">Best,<br>Jane</p>
        """
    },
    {
        "id": 2, "subject": "Upcoming Holiday Schedule", "sender": "HR Department <hr@corp.com>", "date": "Oct 14, 2025",
        "body": "<p class=\"text-gray-300\">All Employees,</p><p class=\"mt-4 text-gray-300\">The holiday schedule is finalized and available on the internal portal.</p>"
    },
    {
        "id": 3, "subject": "Quick Question Regarding the New Logo", "sender": "Mark Smith <mark.smith@design.io>", "date": "Oct 13, 2025",
        "body": "<p class=\"text-gray-300\">Could you confirm the HEX code for the dark blue in the new logo design?</p>"
    },
    {
        "id": 4, "subject": "Reminder: PTO Request Deadline", "sender": "Jenny Wilson <jenny@corp.com>", "date": "Oct 12, 2025",
        "body": "<p class=\"text-gray-300\">Just a friendly reminder that all Paid Time Off (PTO) requests for the first quarter must be submitted by the end of this week.</p>"
    },
    {
        "id": 5, "subject": "Meeting Confirmation: Q1 Budget Review", "sender": "Finance Team <finance@corp.com>", "date": "Oct 12, 2025",
        "body": "<p class=\"text-gray-300\">This is a confirmation for our Q1 Budget Review meeting scheduled for 10/25 at 2:00 PM.</p>"
    },
    {
        "id": 6, "subject": "Your recent purchase has been shipped", "sender": "Shipping <no-reply@shop.com>", "date": "Oct 10, 2025",
        "body": "<p class=\"text-gray-300\">Great news! Your order #90210 has been shipped and is expected to arrive within 3-5 business days.</p>"
    },
    {
        "id": 7, "subject": "Weekly Dev Sync Agenda", "sender": "Project Manager <pm@corp.com>", "date": "Oct 9, 2025",
        "body": "<p class=\"text-gray-300\">Please review the agenda for our weekly sync.</p>"
    },
    {
        "id": 8, "subject": "New Policy Update: Remote Work", "sender": "HR Department <hr@corp.com>", "date": "Oct 9, 2025",
        "body": "<p class=\"text-gray-300\">Our remote work policy has been updated, allowing for two days per week remote.</p>"
    },
    {
        "id": 9, "subject": "Lunch options for tomorrow", "sender": "Michael <michael@corp.com>", "date": "Oct 8, 2025",
        "body": "<p class=\"text-gray-300\">Hey, thinking of ordering Italian tomorrow. Let me know if you want to join the group order!</p>"
    },
    {
        "id": 10, "subject": "Your monthly service bill is ready", "sender": "Billing <billing@service.com>", "date": "Oct 8, 2025",
        "body": "<p class=\"text-gray-300\">Your bill for the month of October is now available. Log in to your account to view the details.</p>"
    },
]
EMAILS_PER_PAGE = 5

# --- Utility Functions for HTML Fragments (Templates could be used instead) ---

def create_list_item_fragment(email):
    """Generates the HTML for a single email item."""
    is_unread = email['id'] <= 3
    sender_name = email['sender'].split('<')[0].strip()
    preview_text = email['body'].replace(
        '<[^>]*?>', '', 100).split('</p>')[0].replace('<p class="text-gray-300">', '')
    
    # Crucial: hx-get now points to the real FastAPI endpoint
    return f"""
    <div
        class="list-item email-list-item cursor-pointer p-4 border-b border-gray-700 hover:bg-gray-800 transition duration-200 border-l-4 border-l-transparent"
        hx-get="/api/email/{email['id']}"
        hx-target="#detail-pane"
        hx-swap="innerHTML"
        hx-indicator="#loading-indicator"
    >
        <div class="flex justify-between items-center">
            <p class="{'font-semibold text-white' if is_unread else 'font-medium text-gray-300'} truncate">{sender_name}</p>
            <span class="text-xs {'text-blue-400 font-bold' if is_unread else 'text-gray-400'}">{email['date']}</span>
        </div>
        <p class="text-sm {'font-medium text-gray-200' if is_unread else 'text-gray-400'} truncate mt-1">{email['subject']}</p>
        <p class="text-xs text-gray-500 truncate mt-1">{preview_text}...</p>
    </div>
    """

def create_detail_fragment(email):
    """Generates the HTML for the email detail pane."""
    return f"""
        <div class="border-b border-gray-700 p-6 flex justify-between items-start">
            <div>
                <h2 class="text-2xl font-bold text-white">{email['subject']}</h2>
                <p class="text-sm text-gray-400 mt-1">From: <span class="text-white">{email['sender']}</span></p>
            </div>
            <p class="text-sm text-gray-400 pt-1">{email['date']}</p>
        </div>
        <div class="p-6 overflow-y-auto custom-scrollbar flex-grow bg-gray-800 text-lg">
            {email['body']}
        </div>
        <div class="p-4 border-t border-gray-700 flex space-x-3 bg-gray-700/50 flex-shrink-0">
            <button class="flex items-center space-x-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition duration-150 shadow-lg">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h10a2 2 0 012 2v6a2 2 0 01-2 2H3a2 2 0 01-2-2v-6a2 2 0 012-2zM7 10V5a2 2 0 012-2h6a2 2 0 012 2v5"></path></svg>
                <span>Reply</span>
            </button>
            <button class="flex items-center space-x-1 px-4 py-2 bg-gray-600 hover:bg-gray-500 text-gray-200 text-sm font-medium rounded-lg transition duration-150 shadow-lg">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 4h4M4 7l4-4M15 10l4-4M15 10l-4 4M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                <span>Forward</span>
            </button>
        </div>
    """

# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Route to serve the base HTML template."""
    # Templates.TemplateResponse requires the request object
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/email/list", response_class=HTMLResponse)
async def email_list(page: int = Query(1, ge=1)):
    """HTMX route to load the initial list and handle infinite scrolling."""
    
    # Simulate a network delay
    time.sleep(0.5) 

    start_index = (page - 1) * EMAILS_PER_PAGE
    end_index = start_index + EMAILS_PER_PAGE
    page_emails = EMAIL_DETAILS[start_index:end_index]

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
async def email_detail(email_id: int):
    """HTMX route to load the detail pane content."""
    
    # Simulate a network delay
    time.sleep(0.2) 
    
    email = next((e for e in EMAIL_DETAILS if e['id'] == email_id), None)
    
    if email:
        return HTMLResponse(content=create_detail_fragment(email))
    else:
        return HTMLResponse(content="<div class='p-8 text-center text-red-400'>Error: Email not found.</div>", 
                            status_code=status.HTTP_404_NOT_FOUND)

