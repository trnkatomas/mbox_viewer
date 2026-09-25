"""
Email Service Layer

Business logic for email operations, decoupled from the presentation layer
(email_server.py, mcp_server.py) and data access (email_utils.py).

Services return plain data structures, never HTML, and have no FastAPI
dependencies so they can be tested without the HTTP layer.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from email_utils import (
    Email,
    embed_query,
    get_basic_stats,
    get_domains_by_count,
    get_email_count,
    get_email_list,
    get_email_sizes_in_time,
    get_one_email,
    get_one_thread,
    load_and_parse_email,
    parse_email,
    read_mbox_slice,
)

logger = logging.getLogger(__name__)

_FILTER_RE = re.compile(r'\b(from|subject|rag|from_date|to_date|label):(?:"([^"]*)"|(\S+))')


def parse_search_query(query: str) -> Dict[str, str]:
    """
    Parse search query string into structured criteria.

    Example:
        >>> parse_search_query('from:john@example.com subject:"team meeting" important')
        {'from': 'john@example.com', 'subject': 'team meeting', 'excerpt': 'important'}

    Supported filters: from:, subject:, rag:, from_date:, to_date:, label:
    (values with spaces must be quoted). Remaining text becomes 'excerpt',
    a keyword search. A rag: value runs to the next filter or the end, so
    "rag:flight to prague" searches for the whole phrase.
    """
    criteria: Dict[str, str] = {}
    remainder_parts: List[str] = []
    last_end = 0
    rag_open = False
    for match in _FILTER_RE.finditer(query):
        between = query[last_end : match.start()]
        if rag_open:
            criteria["rag"] = f"{criteria['rag']} {between.strip()}".strip()
        else:
            remainder_parts.append(between)
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        criteria[key] = value
        rag_open = key == "rag" and match.group(2) is None
        last_end = match.end()
    tail = query[last_end:]
    if rag_open:
        criteria["rag"] = f"{criteria['rag']} {tail.strip()}".strip()
    else:
        remainder_parts.append(tail)

    criteria["excerpt"] = " ".join(" ".join(remainder_parts).split())
    return criteria


def search_emails(
    db: Any,
    query: Optional[str],
    page: int,
    page_size: int,
    folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search and paginate emails; a rag: filter switches to semantic ranking.

    Raises:
        RagUnavailableError: rag: was requested but the query couldn't be embedded.
        InvalidQueryError: a filter value is malformed (e.g. a bad date).

    Returns a dict with emails, total_count, has_more and next_page.
    """
    criteria = parse_search_query(query) if query else {}
    rag_query = criteria.pop("rag", None)
    query_vec = embed_query(db, rag_query) if rag_query else None
    sent = folder == "Sent"
    logger.debug("search_emails: query=%r criteria=%r folder=%r rag=%r", query, criteria, folder, rag_query)

    offset = (page - 1) * page_size
    page_df = get_email_list(db, page_size, offset, criteria, sent, query_vec)
    total_count = get_email_count(db, criteria, sent, query_vec)
    has_more = offset + page_size < total_count

    return {
        "emails": page_df.to_dict(orient="records"),
        "total_count": total_count,
        "has_more": has_more,
        "next_page": page + 1 if has_more else -1,
        "semantic": query_vec is not None,
    }


def _thread_records(db: Any, thread_id: Optional[str]) -> List[Dict[str, Any]]:
    if not thread_id:
        return []
    return get_one_thread(db, thread_id).to_dict(orient="records")  # type: ignore[return-value]


def get_email_with_thread(db: Any, email_id: str) -> Optional[Dict[str, Any]]:
    """
    Email metadata, parsed body and thread summary, or None if not found.

    Keys: email_meta, email_content, body_type, attachments (without content),
    inline_images, thread.
    """
    email_df = get_one_email(db, email_id)
    if email_df.empty:
        return None
    try:
        email_meta = email_df.to_dict(orient="records")[0]
        parsed = load_and_parse_email(Email.from_dict(email_meta))
        body_type, content = parsed["body"]
        return {
            "email_meta": email_meta,
            "email_content": content,
            "body_type": body_type,
            "attachments": parsed["attachments"],
            "inline_images": parsed["inline_images"],
            "thread": _thread_records(db, email_meta.get("thread_id")),
        }
    except (KeyError, ValueError, IndexError, OSError) as e:
        logger.error("Failed to load email %s: %s", email_id, e)
        return None


def get_thread_with_emails(db: Any, thread_id: str) -> Optional[List[Dict[str, Any]]]:
    """Thread emails oldest-first with parsed content, or None if not found."""
    thread = _thread_records(db, thread_id)
    if not thread:
        return None
    return enrich_thread_emails(db, thread)


def enrich_thread_emails(
    db: Any, thread: Sequence[Mapping[Any, Any]]
) -> List[Dict[str, Any]]:
    """Add parsed_body, body_type, attachments and inline_images to each email."""
    enriched_thread = []
    for email_dict in thread:
        enriched: Dict[str, Any] = dict(email_dict)
        try:
            parsed = load_and_parse_email(Email.from_dict(email_dict))
            enriched["body_type"], enriched["parsed_body"] = parsed["body"]
            enriched["attachments"] = parsed["attachments"]
            enriched["inline_images"] = parsed["inline_images"]
        except (KeyError, ValueError, OSError) as e:
            logger.warning("Failed to parse email in thread: %s", e)
        enriched_thread.append(enriched)
    return enriched_thread


def get_stats_summary(db: Any) -> Dict[str, Any]:
    """Total count, average size (bytes), span in years, first/last date."""
    basic_stats = get_basic_stats(db)
    all_emails = basic_stats[0].to_dict(orient="records")[0].get("all_emails")
    avg_size = basic_stats[1].to_dict(orient="records")[0].get("avg_size")
    first_seen = basic_stats[2].to_dict(orient="records")[0].get("first_seen")
    last_seen = basic_stats[2].to_dict(orient="records")[0].get("last_seen")

    years_timespan = 0.0
    if first_seen is not None and last_seen is not None and not pd.isna(first_seen):
        years_timespan = (last_seen - first_seen).days / 365

    return {
        "all_emails": all_emails,
        "avg_size": avg_size if avg_size is not None and not pd.isna(avg_size) else 0,
        "days_timespan": years_timespan,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


def get_stats_time_series(db: Any, query_name: str) -> List[Dict[str, Any]]:
    """Time series for 'dates_size' or 'domains_count'; [] for unknown names."""
    if query_name == "dates_size":
        stats_df = get_email_sizes_in_time(db)
        if not stats_df.empty:
            if pd.api.types.is_datetime64_any_dtype(stats_df["date"]):
                stats_df["date"] = stats_df["date"].dt.strftime("%Y-%m-%d")
            return stats_df.to_dict(orient="records")  # type: ignore[return-value]
    elif query_name == "domains_count":
        stats_df = get_domains_by_count(db)
        if not stats_df.empty:
            return stats_df.to_dict(orient="records")  # type: ignore[return-value]
    return []


def get_attachment(
    db: Any, email_id: str, attachment_id: Union[int, str]
) -> Optional[Dict[str, Any]]:
    """
    One attachment with its content, selected by position (int) or filename (str).
    Positions are unambiguous when several attachments share a filename.
    """
    email_df = get_one_email(db, email_id)
    if email_df.empty:
        logger.warning("Email %s not found", email_id)
        return None
    try:
        email_meta = email_df.to_dict(orient="records")[0]
        raw = read_mbox_slice(int(email_meta["email_start"]), int(email_meta["email_end"]))
        attachments = parse_email(raw)["attachments"]
    except (KeyError, ValueError, OSError) as e:
        logger.error("Failed to retrieve attachment %s: %s", attachment_id, e)
        return None

    if isinstance(attachment_id, int):
        if 0 <= attachment_id < len(attachments):
            return attachments[attachment_id]  # type: ignore[no-any-return]
    else:
        for attachment in attachments:
            if attachment.get("filename") == attachment_id:
                return attachment  # type: ignore[no-any-return]
    logger.warning("Attachment %s not found in email %s", attachment_id, email_id)
    return None
