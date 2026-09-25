"""
MCP (Model Context Protocol) Server for mbox-viewer

Exposes email search, retrieval, and statistics as MCP tools so that
AI agents can browse and query the email archive programmatically.

Usage:
    uv run mcp_server.py               # run as MCP server (stdio transport)
    uv run mcp dev mcp_server.py       # run with MCP inspector for testing

Environment variables (same as the web app):
    MBOX_FILE_PATH   Path to the .mbox file (required)
    EMAILS_DB_PATH   Path to the DuckDB index (default: emails.db)
    OLLAMA_URL       Ollama embedding endpoint (for rag: search)
    OLLAMA_MODEL     Embedding model, only used if the index doesn't record one
                     (default: embeddinggemma)
    LOG_LEVEL        Logging level (default: INFO)
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import requests

from mcp.server.fastmcp import FastMCP

from email_service import (
    get_attachment,
    get_email_with_thread,
    get_stats_summary,
    get_stats_time_series,
    get_thread_with_emails,
    search_emails,
)
from email_utils import (
    DEFAULT_OLLAMA_URL,
    InvalidQueryError,
    RagUnavailableError,
    get_embedding_config,
    html_to_text,
    load_email_db,
)

# Log to stderr so we don't corrupt the stdio MCP transport on stdout
logging.basicConfig(
    stream=sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "mbox-viewer",
    instructions=(
        "You have access to a personal email archive in mbox format.\n"
        "\n"
        "## Typical workflow\n"
        "1. Call search_emails_tool to find relevant emails — it returns summaries with "
        "a message_id for each match.\n"
        "2. Call get_email_tool with a message_id to read the full body of one email.\n"
        "3. Call get_thread_tool with a thread_id (returned by search or get_email) to "
        "read the full conversation.\n"
        "4. If something seems broken, call get_server_status_tool first — it shows "
        "whether the database loaded and which env vars are set.\n"
        "\n"
        "## Search query syntax\n"
        "All filters are combined with AND. Text matches are case-insensitive substrings.\n"
        "  from:alice@example.com       — sender contains 'alice@example.com'\n"
        "  subject:budget               — subject contains 'budget'; quote multi-word values:"
        " subject:\"team meeting\"\n"
        "  label:Inbox                  — exact label/folder match\n"
        "  from_date:2024-01-01         — emails on or after this date (YYYY-MM-DD)\n"
        "  to_date:2024-06-30           — emails on or before this date, inclusive (YYYY-MM-DD)\n"
        "  rag:discuss the Q3 roadmap   — semantic search; the phrase runs to the next filter "
        "or the end of the query, results are ordered by relevance\n"
        "  plain keywords               — matched against the subject and the email text\n"
        "\n"
        "Combined example: 'from:alice subject:budget from_date:2024-01-01 to_date:2024-12-31'\n"
        "\n"
        "## Important constraints\n"
        "- rag: requires the Ollama embedding service; if it is down the search returns an "
        "error instead of results (check get_server_status_tool).\n"
        "- Use folder='Sent' parameter on search_emails_tool to restrict to sent mail.\n"
    ),
)

# Single shared DB connection (read-only, thread-safe)
_db: Any = None


def _get_db() -> Any:
    global _db
    if _db is None:
        mbox_path = os.environ.get("MBOX_FILE_PATH", "<not set>")
        ollama_url = os.environ.get("OLLAMA_URL", "<not set>")
        ollama_model = os.environ.get("OLLAMA_MODEL", "<not set>")
        logger.info(
            "Initialising database connection — MBOX_FILE_PATH=%r OLLAMA_URL=%r OLLAMA_MODEL=%r",
            mbox_path, ollama_url, ollama_model,
        )
        try:
            _db = load_email_db()
        except Exception:
            logger.exception("load_email_db() failed — MBOX_FILE_PATH=%r", mbox_path)
            raise
        logger.info("Database loaded successfully")
    return _db


# ---------------------------------------------------------------------------
# HTML → plain-text helper
# ---------------------------------------------------------------------------


def _html_to_text(content: str) -> str:
    return html_to_text(content)


def _body_for_agent(body: Optional[str], body_type: Optional[str], include_html: bool) -> str:
    body = body or ""
    if include_html or (body_type and body_type != "HTML"):
        return body
    return _html_to_text(body)


def _attachment_summary(attachment: Dict[str, Any]) -> Dict[str, Any]:
    size = attachment.get("size_bytes")
    if size is None:
        content = attachment.get("content")
        size = len(content) if isinstance(content, bytes) else 0
    return {
        "filename": attachment.get("filename"),
        "content_type": attachment.get("content_type"),
        "size_bytes": size,
    }


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
    """Search emails in the archive. Returns summaries; use get_email_tool to read a full message.

    Args:
        query: Space-separated filters, all combined with AND. Available filters:
               from:value        — sender address contains value (substring match)
               subject:value     — subject contains value (substring match); quote
                                   multi-word values: subject:"team meeting"
               label:value       — exact label/folder match, e.g. label:Inbox
               from_date:YYYY-MM-DD — only emails on or after this date
               to_date:YYYY-MM-DD   — only emails on or before this date
               rag:free text     — semantic search over email content, runs to the
                                   next filter or the end; results ordered by relevance
                                   (requires Ollama; use get_server_status_tool to check)
               plain keywords    — matched against the subject and the email text

               Examples:
                 "from:alice subject:budget"
                 "from:boss@corp.com from_date:2024-01-01 to_date:2024-03-31"
                 "label:Inbox rag:flight booking confirmation"
                 "subject:invoice from_date:2023-06-01"

        page: Page number, 1-indexed (default 1).
        page_size: Results per page (default 20, max 100).
        folder: Pass "Sent" to restrict results to the Sent folder.

    Returns:
        JSON with keys:
          emails       – list of email summaries (each has message_id, subject,
                         from_email, to_email, date, excerpt, thread_id, labels)
          total_count  – total number of matching emails
          has_more     – whether more pages exist
          next_page    – page number to pass for the next page (-1 if none)
        or {"error": ...} if the query is invalid or semantic search is unavailable.
    """
    logger.info(
        "search_emails_tool called: query=%r page=%d page_size=%d folder=%r",
        query, page, page_size, folder,
    )
    db = _get_db()
    page_size = min(page_size, 100)
    try:
        result = search_emails(db, query, page, page_size, folder)
    except (RagUnavailableError, InvalidQueryError) as exc:
        logger.warning("search_emails_tool rejected query %r: %s", query, exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception:
        logger.exception("search_emails_tool failed: query=%r", query)
        raise

    # Trim each email to the fields that are useful for an agent
    summary_fields = {
        "message_id", "subject", "from_email", "to_email",
        "date", "excerpt", "has_attachment", "thread_id", "labels", "dist",
    }
    result["emails"] = [
        {k: v for k, v in email.items() if k in summary_fields}
        for email in result["emails"]
    ]

    logger.info(
        "search_emails_tool returning %d/%d results",
        len(result["emails"]), result.get("total_count", 0),
    )
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
    logger.info("get_email_tool called: email_id=%r include_html=%r", email_id, include_html)
    db = _get_db()
    try:
        data = get_email_with_thread(db, email_id)
    except Exception:
        logger.exception("get_email_tool failed: email_id=%r", email_id)
        raise
    if data is None:
        logger.warning(
            "get_email_tool: get_email_with_thread returned None for email_id=%r "
            "(email may not exist in the database)",
            email_id,
        )
        return json.dumps({"error": f"Email not found: {email_id}"})

    meta = data.get("email_meta", {})
    body = _body_for_agent(data.get("email_content"), data.get("body_type"), include_html)

    attachments = [
        _attachment_summary(a) for a in (data.get("attachments") or []) if isinstance(a, dict)
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

    logger.info(
        "get_email_tool returning email subject=%r thread_size=%d attachments=%d",
        output.get("subject"), len(output.get("thread") or []), len(attachments),
    )
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
    logger.info("get_thread_tool called: thread_id=%r include_html=%r", thread_id, include_html)
    db = _get_db()
    try:
        emails = get_thread_with_emails(db, thread_id)
    except Exception:
        logger.exception("get_thread_tool failed: thread_id=%r", thread_id)
        raise
    if emails is None:
        logger.warning(
            "get_thread_tool: get_thread_with_emails returned None for thread_id=%r "
            "(thread may not exist in the database)",
            thread_id,
        )
        return json.dumps({"error": f"Thread not found: {thread_id}"})

    result = []
    for e in emails:
        body = _body_for_agent(e.get("parsed_body"), e.get("body_type"), include_html)

        attachments = [
            _attachment_summary(a) for a in (e.get("attachments") or []) if isinstance(a, dict)
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

    logger.info("get_thread_tool returning %d emails for thread %r", len(result), thread_id)
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
    logger.info("get_attachment_info_tool called: email_id=%r filename=%r", email_id, filename)
    db = _get_db()
    try:
        attachment = get_attachment(db, email_id, filename)
    except Exception:
        logger.exception("get_attachment_info_tool failed: email_id=%r filename=%r", email_id, filename)
        raise
    if attachment is None:
        logger.warning(
            "get_attachment_info_tool: attachment not found: email_id=%r filename=%r",
            email_id, filename,
        )
        return json.dumps({"error": f"Attachment '{filename}' not found in email {email_id}"})

    summary = _attachment_summary(attachment)
    logger.info("get_attachment_info_tool returning attachment: %r %d bytes", filename, summary["size_bytes"])
    return json.dumps(summary)


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
    logger.info("get_stats_tool called")
    db = _get_db()
    try:
        stats = get_stats_summary(db)
    except Exception:
        logger.exception("get_stats_tool failed")
        raise
    logger.info("get_stats_tool returning stats: %s", stats)
    return json.dumps(_serialisable(stats), ensure_ascii=False, indent=2)


def _ollama_reachable(embed_url: str) -> bool:
    base = embed_url.split("/api/")[0]
    try:
        return requests.get(f"{base}/api/tags", timeout=2).status_code == 200
    except requests.RequestException:
        return False


@mcp.tool()
def get_server_status_tool() -> str:
    """Return diagnostic information about the MCP server configuration and state.

    Useful for troubleshooting: shows whether the database is loaded, which
    environment variables are set, and whether semantic (rag:) search can work.

    Returns:
        JSON with keys:
          db_loaded         – whether the database has been initialised
          mbox_file_path    – value of MBOX_FILE_PATH env var (or null)
          mbox_file_exists  – whether the file exists on disk
          ollama_url        – embedding endpoint in use
          ollama_reachable  – whether the Ollama server answered
          embedding_model   – model the index was built with (used for queries)
          embedding_dim     – vector size of the index
          embedded_emails   – number of emails with at least one embedding
          rag_available     – whether rag: search should work right now
    """
    logger.info("get_server_status_tool called")
    mbox_path = os.environ.get("MBOX_FILE_PATH")
    ollama_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    status: dict[str, Any] = {
        "db_loaded": False,
        "mbox_file_path": mbox_path,
        "mbox_file_exists": os.path.isfile(mbox_path) if mbox_path else False,
        "ollama_url": ollama_url,
        "ollama_reachable": _ollama_reachable(ollama_url),
        "embedding_model": None,
        "embedding_dim": None,
        "embedded_emails": 0,
    }

    try:
        db = _get_db()
        status["db_loaded"] = True
        config = get_embedding_config(db)
        status["embedding_model"] = config.model
        status["embedding_dim"] = config.dim
        if config.dim is not None:
            row = db.execute("select count(distinct message_id) from embeddings").fetchone()
            status["embedded_emails"] = int(row[0]) if row else 0
    except Exception as exc:
        status["db_error"] = str(exc)

    status["rag_available"] = bool(status["ollama_reachable"] and status["embedded_emails"])
    logger.info("get_server_status_tool: %s", status)
    return json.dumps(status, ensure_ascii=False, indent=2)


@mcp.tool()
def get_stats_time_series_tool(query_name: str) -> str:
    """Get time-series statistics for the email archive.

    Args:
        query_name: One of:
          - "dates_size"    – monthly cumulative email sizes over time
          - "domains_count" – top sender domains by message count

    Returns:
        JSON array of data-point objects.
        For dates_size: [{date, count}, ...] where count is the cumulative size in bytes
        For domains_count: [{domain, count}, ...]
    """
    logger.info("get_stats_time_series_tool called: query_name=%r", query_name)
    db = _get_db()
    try:
        data = get_stats_time_series(db, query_name)
    except Exception:
        logger.exception("get_stats_time_series_tool failed: query_name=%r", query_name)
        raise
    logger.info("get_stats_time_series_tool returning %d data points", len(data) if isinstance(data, list) else -1)
    return json.dumps(_serialisable(data), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
