"""
Email Service Layer

This module contains business logic for email operations, decoupled from
the presentation layer (FastAPI routes) and data access layer (email_utils).

Architecture:
- Routes (email_server.py): Handle HTTP, validate input, return responses
- Services (this file): Business logic, orchestration, data transformation
- Data Access (email_utils.py): Database queries, raw data retrieval
- Models (email_utils.py): Data structures (Email dataclass)

Design Principles:
- Services return data structures, NOT HTML
- Services are framework-agnostic (no FastAPI dependencies)
- Services are testable without HTTP layer
- Business logic is centralized here, not scattered in routes
"""

import logging
import mmap
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Mbox file path - must be set via MBOX_FILE_PATH environment variable
# Set to None if not provided; functions using this will fail with clear error messages
MBOX_FILE_PATH = os.getenv("MBOX_FILE_PATH")


# ============================================================================
# Service Function Signatures (to be implemented in subsequent phases)
# ============================================================================


def parse_search_query(query: str) -> Dict[str, str]:
    """
    Parse search query string into structured criteria.

    Extracts special filters like from:, subject:, rag:, from_date:, to_date:, label:

    Args:
        query: Raw search query string

    Returns:
        Dictionary of search criteria with:
        - Extracted filter values (from, subject, rag, from_date, to_date, label)
        - excerpt: Remaining text after filters are removed

    Example:
        >>> parse_search_query("from:john@example.com subject:meeting important")
        {'from': 'john@example.com', 'subject': 'meeting', 'excerpt': 'important'}

    Supported filters:
    - from:email - Filter by sender
    - subject:text - Filter by subject (supports quoted strings)
    - rag:text - Semantic search query
    - from_date:YYYY-MM-DD - Filter from date
    - to_date:YYYY-MM-DD - Filter to date
    - label:text - Filter by label
    """
    import re

    regex = r"(from|subject|rag|from_date|to_date|label):(\"(.*)\"|([^ \n]+))"
    matches = re.finditer(regex, query, re.MULTILINE)
    matches_dict: Dict[str, str] = {}
    to_discard: List[int] = []

    for matchNum, match in enumerate(matches, start=1):
        logger.debug(
            "Match %d was found at %d-%d: %s",
            matchNum,
            match.start(),
            match.end(),
            match.group(),
        )
        to_discard.extend(list(range(match.start(), match.end())))
        # Extract the value (either quoted or unquoted)
        value = match[3] if match[3] else match[4]
        matches_dict.update({match[1]: value})

    # Extract remainder as excerpt
    remainder = [c for i, c in enumerate(query) if i not in to_discard]
    matches_dict["excerpt"] = "".join(remainder).strip()

    return matches_dict


def search_emails(
    db: Any,  # duckdb.DuckDBPyConnection
    query: Optional[str],
    page: int,
    page_size: int,
    folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search and paginate emails with optional RAG semantic search.

    Orchestrates:
    - Query parsing
    - RAG search if needed
    - Database querying
    - Pagination
    - Result merging and sorting

    Args:
        db: Database connection
        query: Search query string (optional)
        page: Page number (1-indexed)
        page_size: Number of results per page
        folder: Folder filter (e.g., "Sent")

    Returns:
        Dictionary containing:
        - emails: List of email records
        - total_count: Total matching emails
        - has_more: Boolean indicating more results available
    """
    from email_utils import get_email_count, get_email_list, rag_search_duckdb

    # Calculate pagination indices
    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    # Parse search query if provided
    additional_criteria = parse_search_query(query) if query else None

    # Handle RAG semantic search if rag: query provided
    rag_message_ids: Optional[pd.DataFrame] = None
    if additional_criteria and "rag" in additional_criteria:
        rag_query = additional_criteria.pop(
            "rag"
        )  # Remove from criteria, handle separately
        rag_message_ids = rag_search_duckdb(
            db,
            rag_query,
            n_results=100,  # Get more RAG results, then filter/paginate
        )

    # Fetch emails from database
    page_emails_df = get_email_list(
        db,
        criteria=(
            {"limit": page_size, "offset": start_index}
            if rag_message_ids is None
            else {"limit": page_size, "offset": 0}
        ),
        additional_criteria=additional_criteria,
        sent=folder == "Sent",
        rag_message_ids=(
            rag_message_ids[start_index:end_index]["message_id"].tolist()
            if rag_message_ids is not None and not rag_message_ids.empty
            else None
        ),
    )

    # Merge RAG results if needed
    if rag_message_ids is not None and not rag_message_ids.empty:
        page_emails_df = pd.merge(
            page_emails_df, rag_message_ids[["message_id", "dist"]], on="message_id"
        ).sort_values(by="dist", ascending=True)

    # Convert to list of dicts
    page_emails = page_emails_df.to_dict(orient="records")

    # Calculate total count
    if rag_message_ids is not None and not rag_message_ids.empty:
        total_count = rag_message_ids.shape[0]
    elif additional_criteria:
        total_count = get_email_count(db, additional_criteria)
    else:
        total_count = get_email_count(db)

    # Calculate if there are more results
    has_more = end_index < total_count

    return {
        "emails": page_emails,
        "total_count": total_count,
        "has_more": has_more,
        "next_page": page + 1 if has_more else -1,
    }


def get_email_with_thread(db: Any, email_id: str) -> Optional[Dict[str, Any]]:
    """
    Get email details with thread information.

    Args:
        db: Database connection
        email_id: Email message ID

    Returns:
        Dictionary containing:
        - email_meta: Email metadata dict
        - email_content: Parsed email content (HTML)
        - attachments: List of attachments
        - thread: List of emails in the same thread
        Or None if email not found
    """
    from email_utils import Email, get_one_email, get_thread_for_email, load_and_parse_email

    # Get email from database
    email_df = get_one_email(db, email_id)
    if email_df.empty:
        return None

    try:
        # Convert to dict and Email dataclass
        email_meta = email_df.to_dict(orient="records")[0]
        email = Email.from_dict(email_meta)

        # Load and parse email content
        parsed_email = load_and_parse_email(email)
        attachments = parsed_email.get("attachments")
        email_content = parsed_email.get("body")

        # Get thread information
        thread_df = get_thread_for_email(db, email_id)
        thread = thread_df.to_dict(orient="records")

        return {
            "email_meta": email_meta,
            "email_content": email_content[1] if email_content else None,  # Get HTML content
            "attachments": attachments,
            "thread": thread,
        }
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Failed to load email {email_id}: {e}")
        return None


def get_thread_with_emails(db: Any, thread_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get thread emails and enrich them with parsed content.

    Args:
        db: Database connection
        thread_id: Thread ID

    Returns:
        List of enriched email dicts with parsed content, or None if thread not found
    """
    from email_utils import get_one_thread

    # Get thread from database
    thread_df = get_one_thread(db, thread_id)
    thread = thread_df.to_dict(orient="records")

    if not thread:
        return None

    # Enrich emails in thread
    return enrich_thread_emails(db, thread)


def enrich_thread_emails(
    db: Any, thread: Sequence[Mapping[Any, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich thread emails with parsed content.

    Loads and parses each email in the thread, adding:
    - parsed_body: HTML content
    - attachments: List of attachments

    Args:
        db: Database connection
        thread: List of email metadata dicts

    Returns:
        List of enriched email dicts with parsed content
    """
    from email_utils import Email, load_and_parse_email

    enriched_thread = []
    for email_dict in thread:
        try:
            # Convert to Email dataclass and parse
            email = Email.from_dict(email_dict)
            parsed_email = load_and_parse_email(email)

            # Enrich the email dict with parsed content
            enriched_email: Dict[str, Any] = dict(email_dict)
            enriched_email["parsed_body"] = parsed_email.get("body", ("", ""))[
                1
            ]  # Get HTML content
            enriched_email["attachments"] = parsed_email.get("attachments", [])
            enriched_thread.append(enriched_email)
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse email in thread: {e}")
            # Add original email dict if parsing fails (convert to dict)
            enriched_thread.append(dict(email_dict))

    return enriched_thread


def get_stats_summary(db: Any) -> Dict[str, Any]:
    """
    Get email statistics summary.

    Processes stats DataFrames and returns structured summary.

    Args:
        db: Database connection

    Returns:
        Dictionary containing:
        - all_emails: Total email count
        - avg_size: Average email size
        - days_timespan: Time span in years
        - first_seen: First email date
        - last_seen: Last email date
    """
    from email_utils import get_basic_stats

    basic_stats = get_basic_stats(db)
    all_emails = basic_stats[0].to_dict(orient="records")[0].get("all_emails")
    avg_size = basic_stats[1].to_dict(orient="records")[0].get("avg_size")
    first_seen = basic_stats[2].to_dict(orient="records")[0].get("first_seen")
    last_seen = basic_stats[2].to_dict(orient="records")[0].get("last_seen")

    days_timespan = 0
    if first_seen is not None and last_seen is not None:
        days_timespan = (last_seen - first_seen).days / 365

    return {
        "all_emails": all_emails,
        "avg_size": avg_size,
        "days_timespan": days_timespan,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


def get_stats_time_series(db: Any, query_name: str) -> List[Dict[str, Any]]:
    """
    Get time series statistics data.

    Args:
        db: Database connection
        query_name: Type of stats query (dates_size, domains_count)

    Returns:
        List of dicts containing time series data, or empty list if query not supported
    """
    from email_utils import get_domains_by_count, get_email_sizes_in_time

    if query_name == "dates_size":
        stats_df = get_email_sizes_in_time(db)
        if not stats_df.empty:
            # Convert date column to ISO format string for JSON serialization if it's datetime
            if pd.api.types.is_datetime64_any_dtype(stats_df["date"]):
                stats_df["date"] = stats_df["date"].dt.strftime("%Y-%m-%d")
            return stats_df.to_dict(orient="records")  # type: ignore[return-value]
    elif query_name == "domains_count":
        stats_df = get_domains_by_count(db)
        if not stats_df.empty:
            return stats_df.to_dict(orient="records")  # type: ignore[return-value]

    return []


# ============================================================================
# Mbox File Access with mmap (Thread-Safe, Efficient for Large Files)
# ============================================================================


@lru_cache(maxsize=512)
def read_mbox_slice(start: int, end: int) -> bytes:
    """
    Read a slice from the mbox file using memory-mapped I/O.

    This function uses mmap for efficient random access to large mbox files,
    which is ideal for read-only operations. Results are cached to avoid
    repeated reads of the same email.

    Args:
        start: Start byte position
        end: End byte position

    Returns:
        Byte slice from the mbox file

    Note:
        mmap is thread-safe for read operations and more efficient than
        seek/read for random access patterns, especially with large files.
    """
    if MBOX_FILE_PATH is None:
        raise ValueError(
            "MBOX_FILE_PATH environment variable must be set to use this function"
        )
    with open(MBOX_FILE_PATH, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm[start:end]


def read_mbox_slices_concurrent(
    regions: List[Tuple[int, int]], max_workers: int = 5
) -> List[bytes]:
    """
    Read multiple regions from mbox file concurrently using mmap.

    Useful for loading multiple emails or attachments in parallel.

    Args:
        regions: List of (start, end) byte position tuples
        max_workers: Maximum number of concurrent threads

    Returns:
        List of byte slices corresponding to each region

    Example:
        >>> regions = [(100, 200), (10_000, 11_000), (1_000_000, 1_010_000)]
        >>> results = read_mbox_slices_concurrent(regions)
    """
    if MBOX_FILE_PATH is None:
        raise ValueError(
            "MBOX_FILE_PATH environment variable must be set to use this function"
        )
    with open(MBOX_FILE_PATH, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(lambda region: mm[region[0] : region[1]], regions)
                )
            return results


def get_attachment(
    db: Any, email_id: str, attachment_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get email attachment with validation.

    Retrieves attachment content from the mbox file using efficient
    memory-mapped I/O.

    Args:
        db: Database connection
        email_id: Email message ID
        attachment_id: Attachment filename

    Returns:
        Dictionary containing:
        - filename: Attachment filename
        - content: Binary content
        - content_type: MIME type
        - size_bytes: Size in bytes
        Or None if attachment not found
    """
    from email_utils import get_one_email, parse_email

    # Get email metadata from database
    email_df = get_one_email(db, email_id)
    if email_df.empty:
        logger.warning(f"Email {email_id} not found")
        return None

    try:
        email_data = email_df.to_dict(orient="records")[0]

        # Read email content from mbox using mmap
        email_start = email_data.get("email_start")
        email_end = email_data.get("email_end")

        if email_start is None or email_end is None:
            logger.error(f"Email {email_id} missing byte positions")
            return None

        email_raw = read_mbox_slice(email_start, email_end)

        # Parse email and extract attachments
        parsed = parse_email(email_raw)
        attachments = parsed.get("attachments", [])

        # Type narrowing: ensure attachments is a list
        if not isinstance(attachments, list):
            logger.error(f"Invalid attachments format for email {email_id}")
            return None

        # Find the requested attachment
        for attachment in attachments:
            if isinstance(attachment, dict) and attachment.get("filename") == attachment_id:
                return attachment

        logger.warning(f"Attachment {attachment_id} not found in email {email_id}")
        return None

    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Failed to retrieve attachment {attachment_id}: {e}")
        return None
