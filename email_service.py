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
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


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
    raise NotImplementedError("To be implemented in Phase 6")
