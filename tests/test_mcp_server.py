"""Tests for MCP server tools in mcp_server.py."""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(result: str) -> dict | list:
    return json.loads(result)


# ---------------------------------------------------------------------------
# _html_to_text
# ---------------------------------------------------------------------------


class TestHtmlToText:
    def test_strips_tags(self):
        from mcp_server import _html_to_text

        assert _html_to_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_decodes_entities(self):
        from mcp_server import _html_to_text

        assert "&amp;" not in _html_to_text("AT&amp;T")
        assert "AT&T" in _html_to_text("AT&amp;T")

    def test_collapses_whitespace(self):
        from mcp_server import _html_to_text

        result = _html_to_text("<p>line one</p>\n\n<p>line two</p>")
        assert "\n\n" not in result

    def test_plain_text_passthrough(self):
        from mcp_server import _html_to_text

        assert _html_to_text("just plain text") == "just plain text"

    def test_empty_string(self):
        from mcp_server import _html_to_text

        assert _html_to_text("") == ""


# ---------------------------------------------------------------------------
# _serialisable
# ---------------------------------------------------------------------------


class TestSerialisable:
    def test_datetime_converted(self):
        import datetime

        from mcp_server import _serialisable

        dt = datetime.datetime(2024, 1, 15, 12, 0, 0)
        result = _serialisable({"date": dt})
        assert result["date"] == "2024-01-15T12:00:00"

    def test_bytes_converted(self):
        from mcp_server import _serialisable

        result = _serialisable({"content": b"hello"})
        assert "<binary" in result["content"]
        assert "5" in result["content"]

    def test_nested_structures(self):
        import datetime

        from mcp_server import _serialisable

        data = {"emails": [{"date": datetime.date(2024, 1, 1), "body": b"x"}]}
        result = _serialisable(data)
        assert result["emails"][0]["date"] == "2024-01-01"
        assert "<binary" in result["emails"][0]["body"]

    def test_plain_types_unchanged(self):
        from mcp_server import _serialisable

        data = {"count": 42, "name": "test", "flag": True, "nothing": None}
        assert _serialisable(data) == data


# ---------------------------------------------------------------------------
# search_emails_tool
# ---------------------------------------------------------------------------

MOCK_SEARCH_RESULT = {
    "emails": [
        {
            "message_id": "<test1@example.com>",
            "subject": "Test Email 1",
            "from_email": "sender@example.com",
            "to_email": "recipient@example.com",
            "date": "2024-01-01",
            "excerpt": "Body of email 1",
            "has_attachment": 0,
            "thread_id": "thread1",
            "labels": ["inbox"],
            "extra_field": "should be stripped",
        }
    ],
    "total_count": 1,
    "has_more": False,
    "next_page": -1,
}


class TestSearchEmailsTool:
    def test_returns_valid_json(self):
        from mcp_server import search_emails_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.search_emails", return_value=MOCK_SEARCH_RESULT
        ):
            result = _load_json(search_emails_tool("test query"))
            assert "emails" in result
            assert "total_count" in result
            assert "has_more" in result

    def test_strips_non_summary_fields(self):
        from mcp_server import search_emails_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.search_emails", return_value=MOCK_SEARCH_RESULT
        ):
            result = _load_json(search_emails_tool("test query"))
            email = result["emails"][0]
            assert "extra_field" not in email
            assert "message_id" in email
            assert "subject" in email

    def test_page_size_capped_at_100(self):
        from mcp_server import search_emails_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.search_emails", return_value=MOCK_SEARCH_RESULT
        ) as mock_search:
            search_emails_tool("query", page_size=500)
            _, kwargs = mock_search.call_args
            # page_size is passed positionally: (db, query, page, page_size, folder)
            args = mock_search.call_args[0]
            assert args[3] <= 100

    def test_folder_passed_through(self):
        from mcp_server import search_emails_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.search_emails", return_value=MOCK_SEARCH_RESULT
        ) as mock_search:
            search_emails_tool("query", folder="Sent")
            args = mock_search.call_args[0]
            assert args[4] == "Sent"

    def test_empty_result(self):
        from mcp_server import search_emails_tool

        empty = {"emails": [], "total_count": 0, "has_more": False, "next_page": -1}
        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.search_emails", return_value=empty
        ):
            result = _load_json(search_emails_tool("nothing"))
            assert result["emails"] == []
            assert result["total_count"] == 0


# ---------------------------------------------------------------------------
# get_email_tool
# ---------------------------------------------------------------------------

MOCK_EMAIL_DATA = {
    "email_meta": {
        "message_id": "<test1@example.com>",
        "subject": "Test Subject",
        "from_email": "sender@example.com",
        "to_email": "recipient@example.com",
        "date": "2024-01-01 12:00:00",
        "labels": ["inbox"],
        "has_attachment": 0,
        "thread_id": "thread1",
    },
    "email_content": "<p>Hello <b>world</b></p>",
    "attachments": [
        {"filename": "doc.pdf", "content_type": "application/pdf", "content": b"pdf_bytes"}
    ],
    "thread": [
        {"message_id": "<test1@example.com>", "subject": "Test Subject", "from_email": "sender@example.com", "date": "2024-01-01"}
    ],
}


class TestGetEmailTool:
    def test_returns_plain_text_by_default(self):
        from mcp_server import get_email_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_email_with_thread", return_value=MOCK_EMAIL_DATA
        ):
            result = _load_json(get_email_tool("<test1@example.com>"))
            assert "<p>" not in result["body"]
            assert "Hello" in result["body"]
            assert "world" in result["body"]

    def test_returns_html_when_requested(self):
        from mcp_server import get_email_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_email_with_thread", return_value=MOCK_EMAIL_DATA
        ):
            result = _load_json(get_email_tool("<test1@example.com>", include_html=True))
            assert "<p>" in result["body"]

    def test_not_found_returns_error(self):
        from mcp_server import get_email_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_email_with_thread", return_value=None
        ):
            result = _load_json(get_email_tool("<ghost@example.com>"))
            assert "error" in result

    def test_attachment_metadata_included(self):
        from mcp_server import get_email_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_email_with_thread", return_value=MOCK_EMAIL_DATA
        ):
            result = _load_json(get_email_tool("<test1@example.com>"))
            assert len(result["attachments"]) == 1
            att = result["attachments"][0]
            assert att["filename"] == "doc.pdf"
            assert att["content_type"] == "application/pdf"
            assert att["size_bytes"] == len(b"pdf_bytes")

    def test_binary_content_not_in_output(self):
        from mcp_server import get_email_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_email_with_thread", return_value=MOCK_EMAIL_DATA
        ):
            raw = get_email_tool("<test1@example.com>")
            # Should be valid JSON with no raw binary
            json.loads(raw)
            assert "content" not in json.loads(raw)["attachments"][0]

    def test_thread_summary_only(self):
        from mcp_server import get_email_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_email_with_thread", return_value=MOCK_EMAIL_DATA
        ):
            result = _load_json(get_email_tool("<test1@example.com>"))
            for entry in result["thread"]:
                assert set(entry.keys()) <= {"message_id", "subject", "from_email", "date"}


# ---------------------------------------------------------------------------
# get_thread_tool
# ---------------------------------------------------------------------------

MOCK_THREAD = [
    {
        "message_id": "<t1@example.com>",
        "subject": "Hello",
        "from_email": "a@example.com",
        "to_email": "b@example.com",
        "date": "2024-01-01",
        "labels": ["inbox"],
        "has_attachment": 0,
        "parsed_body": "<p>First message</p>",
        "attachments": [],
    },
    {
        "message_id": "<t2@example.com>",
        "subject": "Re: Hello",
        "from_email": "b@example.com",
        "to_email": "a@example.com",
        "date": "2024-01-02",
        "labels": ["inbox"],
        "has_attachment": 0,
        "parsed_body": "<p>Reply here</p>",
        "attachments": [],
    },
]


class TestGetThreadTool:
    def test_returns_all_emails_in_thread(self):
        from mcp_server import get_thread_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_thread_with_emails", return_value=MOCK_THREAD
        ):
            result = _load_json(get_thread_tool("thread1"))
            assert len(result) == 2

    def test_body_is_plain_text_by_default(self):
        from mcp_server import get_thread_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_thread_with_emails", return_value=MOCK_THREAD
        ):
            result = _load_json(get_thread_tool("thread1"))
            assert "<p>" not in result[0]["body"]
            assert "First message" in result[0]["body"]

    def test_html_body_when_requested(self):
        from mcp_server import get_thread_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_thread_with_emails", return_value=MOCK_THREAD
        ):
            result = _load_json(get_thread_tool("thread1", include_html=True))
            assert "<p>" in result[0]["body"]

    def test_not_found_returns_error(self):
        from mcp_server import get_thread_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_thread_with_emails", return_value=None
        ):
            result = _load_json(get_thread_tool("nonexistent"))
            assert "error" in result


# ---------------------------------------------------------------------------
# get_attachment_info_tool
# ---------------------------------------------------------------------------


class TestGetAttachmentInfoTool:
    def test_returns_metadata(self):
        from mcp_server import get_attachment_info_tool

        mock_att = {"filename": "report.pdf", "content_type": "application/pdf", "content": b"bytes"}
        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_attachment", return_value=mock_att
        ):
            result = _load_json(get_attachment_info_tool("<msg@example.com>", "report.pdf"))
            assert result["filename"] == "report.pdf"
            assert result["content_type"] == "application/pdf"
            assert result["size_bytes"] == len(b"bytes")

    def test_binary_content_not_returned(self):
        from mcp_server import get_attachment_info_tool

        mock_att = {"filename": "f.bin", "content_type": "application/octet-stream", "content": b"\x00\x01\x02"}
        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_attachment", return_value=mock_att
        ):
            result = _load_json(get_attachment_info_tool("<msg@example.com>", "f.bin"))
            assert "content" not in result

    def test_not_found_returns_error(self):
        from mcp_server import get_attachment_info_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_attachment", return_value=None
        ):
            result = _load_json(get_attachment_info_tool("<msg@example.com>", "missing.pdf"))
            assert "error" in result


# ---------------------------------------------------------------------------
# get_stats_tool
# ---------------------------------------------------------------------------


class TestGetStatsTool:
    def test_returns_all_stat_keys(self):
        import datetime

        from mcp_server import get_stats_tool

        mock_stats = {
            "all_emails": 5000,
            "avg_size": 12345,
            "days_timespan": 3.5,
            "first_seen": datetime.datetime(2020, 1, 1),
            "last_seen": datetime.datetime(2024, 1, 1),
        }
        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_stats_summary", return_value=mock_stats
        ):
            result = _load_json(get_stats_tool())
            assert result["all_emails"] == 5000
            assert result["avg_size"] == 12345
            assert result["days_timespan"] == 3.5
            assert "2020" in result["first_seen"]
            assert "2024" in result["last_seen"]

    def test_datetimes_serialised_as_strings(self):
        import datetime

        from mcp_server import get_stats_tool

        mock_stats = {
            "all_emails": 1,
            "avg_size": 100,
            "days_timespan": 0.1,
            "first_seen": datetime.datetime(2024, 6, 15),
            "last_seen": datetime.datetime(2024, 6, 16),
        }
        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_stats_summary", return_value=mock_stats
        ):
            raw = get_stats_tool()
            # Must be valid JSON — would raise if datetimes were not serialised
            json.loads(raw)


# ---------------------------------------------------------------------------
# get_stats_time_series_tool
# ---------------------------------------------------------------------------


class TestGetStatsTimeSeriesTool:
    def test_dates_size_query(self):
        from mcp_server import get_stats_time_series_tool

        mock_data = [{"date": "2024-01-01", "cumulative_size_mb": 100}]
        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_stats_time_series", return_value=mock_data
        ):
            result = _load_json(get_stats_time_series_tool("dates_size"))
            assert isinstance(result, list)
            assert result[0]["date"] == "2024-01-01"

    def test_domains_count_query(self):
        from mcp_server import get_stats_time_series_tool

        mock_data = [{"domain": "example.com", "count": 50}]
        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_stats_time_series", return_value=mock_data
        ):
            result = _load_json(get_stats_time_series_tool("domains_count"))
            assert result[0]["domain"] == "example.com"

    def test_empty_for_unknown_query(self):
        from mcp_server import get_stats_time_series_tool

        with patch("mcp_server._get_db", return_value=MagicMock()), patch(
            "mcp_server.get_stats_time_series", return_value=[]
        ):
            result = _load_json(get_stats_time_series_tool("unsupported_query"))
            assert result == []
