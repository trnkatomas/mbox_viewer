"""
MCP (Model Context Protocol) Server for mbox-viewer

Exposes email search, retrieval, and statistics as MCP tools so that
AI agents can browse and query the email archive programmatically.

Usage:
    uv run mcp_server.py               # run as MCP server (stdio transport)
    uv run mcp dev mcp_server.py       # run with MCP inspector for testing

Environment variables (same as the web app):
    MBOX_FILE_PATH   Path to the .mbox file (required)
    OLLAMA_URL       Ollama embedding endpoint (for rag: search)
    OLLAMA_MODEL     Ollama model name (default: nomic-embed-text)
"""

import html
import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from email_service import (
    get_attachment,
    get_email_with_thread,
    get_stats_summary,
    get_stats_time_series,
    get_thread_with_emails,
    search_emails,
)
from email_utils import load_email_content_search, load_email_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "mbox-viewer",
    instructions=(
        "Access an email archive stored in mbox format. "
        "Use search_emails to find messages, get_email to read a single message, "
        "get_thread to read a full conversation, and get_stats for archive statistics. "
        "The search query supports filters: from:, subject:, label:, "
        "from_date: (YYYY-MM-DD), to_date: (YYYY-MM-DD), and rag: for semantic search."
    ),
)

# Single shared DB connection (read-only, thread-safe)
_db: Any = None


def _get_db() -> Any:
    global _db
    if _db is None:
        _db = load_email_db()
        try:
            load_email_content_search(_db)
        except Exception as exc:  # VSS extension may not be available
            logger.warning("Could not load VSS extension (rag: search disabled): %s", exc)
    return _db


# ---------------------------------------------------------------------------
# HTML → plain-text helper
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._parts)).strip()


def _html_to_text(content: str) -> str:
    """Strip HTML tags and decode entities to produce readable plain text."""
    stripper = _HTMLStripper()
    try:
        # HTMLParser with convert_charrefs=True (the default) decodes entities
        # automatically in handle_data, so no pre-unescaping is needed.
        stripper.feed(content)
        return stripper.get_text()
    except Exception:
        return content


def _serialisable(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable values (e.g. datetimes, bytes)."""
    import datetime

    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return f"<binary {len(obj)} bytes>"
    return obj


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_emails_tool(
    query: str,
    page: int = 1,
    page_size: int = 20,
    folder: Optional[str] = None,
) -> str:
    """Search emails in the archive.

    Args:
        query: Search string. Supports plain keywords and structured filters:
               - from:email@example.com
               - subject:"some subject"
               - label:label_name
               - from_date:YYYY-MM-DD
               - to_date:YYYY-MM-DD
               - rag:natural language query  (semantic / vector search)
               Combine filters freely, e.g. "from:alice rag:budget proposal"
        page: Page number (1-indexed, default 1)
        page_size: Results per page (default 20, max 100)
        folder: Optional folder filter. Use "Sent" to search sent mail only.

    Returns:
        JSON with keys:
          emails       – list of email summaries
          total_count  – total number of matching emails
          has_more     – whether more pages exist
          next_page    – page number to fetch next (or -1 if none)
    """
    db = _get_db()
    page_size = min(page_size, 100)
    result = search_emails(db, query, page, page_size, folder)

    # Trim each email to the fields that are useful for an agent
    summary_fields = {
        "message_id", "subject", "from_email", "to_email",
        "date", "excerpt", "has_attachment", "thread_id", "labels",
    }
    result["emails"] = [
        {k: v for k, v in email.items() if k in summary_fields}
        for email in result["emails"]
    ]

    return json.dumps(_serialisable(result), ensure_ascii=False, indent=2)


@mcp.tool()
def get_email_tool(email_id: str, include_html: bool = False) -> str:
    """Retrieve the full content of a single email.

    Args:
        email_id: The message_id of the email (as returned by search_emails_tool).
        include_html: If True, return the raw HTML body instead of plain text.
                      Default is False (plain text is easier for agents to read).

    Returns:
        JSON with keys:
          message_id    – unique identifier
          subject       – subject line
          from_email    – sender address
          to_email      – recipient address(es)
          date          – ISO-8601 date string
          labels        – list of labels / folders
          has_attachment– 0 or 1
          thread_id     – conversation thread identifier
          body          – email body (plain text by default)
          attachments   – list of {filename, content_type, size_bytes}
          thread        – list of other emails in the same thread (summaries only)
    """
    db = _get_db()
    data = get_email_with_thread(db, email_id)
    if data is None:
        return json.dumps({"error": f"Email not found: {email_id}"})

    meta = data.get("email_meta", {})
    body_raw = data.get("email_content") or ""
    body = body_raw if include_html else _html_to_text(body_raw)

    attachments = [
        {
            "filename": a.get("filename"),
            "content_type": a.get("content_type"),
            "size_bytes": len(a.get("content", b"")) if isinstance(a.get("content"), bytes) else 0,
        }
        for a in (data.get("attachments") or [])
        if isinstance(a, dict)
    ]

    thread_summary = [
        {k: v for k, v in e.items() if k in {"message_id", "subject", "from_email", "date"}}
        for e in (data.get("thread") or [])
    ]

    output = {
        "message_id": meta.get("message_id"),
        "subject": meta.get("subject"),
        "from_email": meta.get("from_email"),
        "to_email": meta.get("to_email"),
        "date": meta.get("date"),
        "labels": meta.get("labels"),
        "has_attachment": meta.get("has_attachment"),
        "thread_id": meta.get("thread_id"),
        "body": body,
        "attachments": attachments,
        "thread": thread_summary,
    }

    return json.dumps(_serialisable(output), ensure_ascii=False, indent=2)


@mcp.tool()
def get_thread_tool(thread_id: str, include_html: bool = False) -> str:
    """Retrieve all emails in a conversation thread.

    Args:
        thread_id: Thread identifier (as returned by search_emails_tool or get_email_tool).
        include_html: If True, return raw HTML bodies. Default False (plain text).

    Returns:
        JSON array of email objects, each with:
          message_id, subject, from_email, to_email, date, labels,
          has_attachment, body, attachments
        Emails are ordered oldest-first.
    """
    db = _get_db()
    emails = get_thread_with_emails(db, thread_id)
    if emails is None:
        return json.dumps({"error": f"Thread not found: {thread_id}"})

    result = []
    for e in emails:
        body_raw = e.get("parsed_body") or ""
        body = body_raw if include_html else _html_to_text(body_raw)

        attachments = [
            {
                "filename": a.get("filename"),
                "content_type": a.get("content_type"),
                "size_bytes": len(a.get("content", b"")) if isinstance(a.get("content"), bytes) else 0,
            }
            for a in (e.get("attachments") or [])
            if isinstance(a, dict)
        ]

        result.append({
            "message_id": e.get("message_id"),
            "subject": e.get("subject"),
            "from_email": e.get("from_email"),
            "to_email": e.get("to_email"),
            "date": e.get("date"),
            "labels": e.get("labels"),
            "has_attachment": e.get("has_attachment"),
            "body": body,
            "attachments": attachments,
        })

    return json.dumps(_serialisable(result), ensure_ascii=False, indent=2)


@mcp.tool()
def get_attachment_info_tool(email_id: str, filename: str) -> str:
    """Get metadata about a specific email attachment.

    Returns attachment metadata (filename, content_type, size).
    Binary content is not returned to keep responses manageable.

    Args:
        email_id: The message_id of the email containing the attachment.
        filename: The filename of the attachment (as listed in get_email_tool).

    Returns:
        JSON with keys: filename, content_type, size_bytes
    """
    db = _get_db()
    attachment = get_attachment(db, email_id, filename)
    if attachment is None:
        return json.dumps({"error": f"Attachment '{filename}' not found in email {email_id}"})

    content = attachment.get("content", b"")
    return json.dumps({
        "filename": attachment.get("filename"),
        "content_type": attachment.get("content_type"),
        "size_bytes": len(content) if isinstance(content, bytes) else 0,
    })


@mcp.tool()
def get_stats_tool() -> str:
    """Get summary statistics for the email archive.

    Returns:
        JSON with keys:
          all_emails    – total number of emails
          avg_size      – average email size in bytes
          days_timespan – archive span in years
          first_seen    – date of oldest email (ISO-8601)
          last_seen     – date of newest email (ISO-8601)
    """
    db = _get_db()
    stats = get_stats_summary(db)
    return json.dumps(_serialisable(stats), ensure_ascii=False, indent=2)


@mcp.tool()
def get_stats_time_series_tool(query_name: str) -> str:
    """Get time-series statistics for the email archive.

    Args:
        query_name: One of:
          - "dates_size"    – monthly cumulative email sizes over time
          - "domains_count" – top sender domains by message count

    Returns:
        JSON array of data-point objects.
        For dates_size: [{date, cumulative_size_mb}, ...]
        For domains_count: [{domain, count}, ...]
    """
    db = _get_db()
    data = get_stats_time_series(db, query_name)
    return json.dumps(_serialisable(data), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
