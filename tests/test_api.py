"""Tests for FastAPI endpoints in email_server.py."""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_db_connections():
    """Mock database connections for testing."""
    mock_duckdb = Mock()

    # Setup mock responses
    mock_duckdb.execute = Mock()
    mock_duckdb.close = Mock()

    with patch("email_server.db_connections", {"duckdb": mock_duckdb}):
        yield {"duckdb": mock_duckdb}


@pytest.fixture
def client(mock_db_connections):
    """Create a test client with mocked dependencies."""
    # Mock the lifespan to avoid actual DB connections
    with patch(
        "email_server.load_email_db", return_value=mock_db_connections["duckdb"]
    ):
        from email_server import app

        with TestClient(app) as test_client:
            yield test_client


class TestMainEndpoints:
    """Tests for main application endpoints."""

    def test_root_endpoint(self, client):
        """Test that root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_inbox_layout(self, client):
        """Test inbox layout endpoint."""
        response = client.get("/api/inbox/layout")
        assert response.status_code == 200

    def test_sent_layout(self, client):
        """Test sent folder layout endpoint."""
        response = client.get("/api/sent/layout")
        assert response.status_code == 200

    def test_stats_layout(self, client):
        """Test statistics layout endpoint."""
        import datetime

        mock_stats_summary = {
            "all_emails": 100,
            "avg_size": 5000,
            "days_timespan": 1.0,
            "first_seen": datetime.datetime(2024, 1, 1),
            "last_seen": datetime.datetime(2024, 12, 31),
        }

        with patch("email_service.get_stats_summary", return_value=mock_stats_summary):
            response = client.get("/api/stats/layout")
            assert response.status_code == 200


class TestEmailEndpoints:
    """Tests for email-related endpoints."""

    def test_email_list_endpoint(self, client):
        """Test email list endpoint."""
        mock_result = {
            "emails": [
                {
                    "message_id": "<test1@example.com>",
                    "subject": "Test 1",
                    "from_email": "sender1@example.com",
                    "date": "2024-01-01",
                    "excerpt": "Body 1",
                    "has_attachment": False,
                },
                {
                    "message_id": "<test2@example.com>",
                    "subject": "Test 2",
                    "from_email": "sender2@example.com",
                    "date": "2024-01-02",
                    "excerpt": "Body 2",
                    "has_attachment": True,
                },
            ],
            "total_count": 10,
            "has_more": True,
            "next_page": 2,
        }

        with patch("email_service.search_emails", return_value=mock_result):
            response = client.get("/api/email/list?page=1")
            assert response.status_code == 200

    def test_email_list_with_search(self, client):
        """Test email list with search query."""
        mock_result = {
            "emails": [
                {
                    "message_id": "<test1@example.com>",
                    "subject": "Test Email",
                    "from_email": "sender@example.com",
                    "date": "2024-01-01",
                    "excerpt": "Body",
                    "has_attachment": False,
                }
            ],
            "total_count": 10,
            "has_more": True,
            "next_page": 2,
        }

        with patch("email_service.search_emails", return_value=mock_result):
            response = client.get("/api/email/list?page=1&query=test")
            assert response.status_code == 200

    def test_email_detail_endpoint(self, client):
        """Test single email detail endpoint."""
        mock_result = {
            "email_meta": {
                "message_id": "<test1@example.com>",
                "subject": "Test Email",
                "from_email": "sender@example.com",
                "to_email": "recipient@example.com",
                "date": "2024-01-01 12:00:00",
                "has_attachment": 0,
                "email_start": 0,
                "email_end": 100,
                "thread_id": "thread1",
                "excerpt": "Test excerpt",
                "labels": [],
            },
            "email_content": "This is the email body",
            "attachments": [],
            "thread": [],
        }

        with patch("email_service.get_email_with_thread", return_value=mock_result):
            response = client.get("/api/email/<test1@example.com>")
            assert response.status_code == 200

    def test_email_detail_not_found(self, client):
        """Test email detail with non-existent email."""
        with patch("email_service.get_email_with_thread", return_value=None):
            response = client.get("/api/email/<nonexistent@example.com>")
            assert response.status_code == 404

    def test_email_thread_endpoint(self, client):
        """Test email thread endpoint."""
        mock_enriched_thread = [
            {
                "message_id": "<test1@example.com>",
                "subject": "Re: Test",
                "from_email": "sender1@example.com",
                "to_email": "recipient@example.com",
                "date": "2024-01-01 12:00:00",
                "thread_id": "thread1",
                "has_attachment": 0,
                "email_start": 0,
                "email_end": 100,
                "excerpt": "Test1",
                "labels": [],
                "parsed_body": "<p>Test email body</p>",
                "attachments": [],
            },
            {
                "message_id": "<test2@example.com>",
                "subject": "Re: Test",
                "from_email": "sender2@example.com",
                "to_email": "recipient@example.com",
                "date": "2024-01-01 13:00:00",
                "thread_id": "thread1",
                "has_attachment": 0,
                "email_start": 100,
                "email_end": 200,
                "excerpt": "Test2",
                "labels": [],
                "parsed_body": "<p>Test email body</p>",
                "attachments": [],
            },
        ]

        with patch(
            "email_service.get_thread_with_emails", return_value=mock_enriched_thread
        ):
            response = client.get("/api/email_thread/thread1")
            assert response.status_code == 200


class TestSearchEndpoint:
    """Tests for search endpoint."""

    def test_search_basic(self, client):
        """Test basic search."""
        mock_result = {
            "emails": [
                {
                    "message_id": "<test1@example.com>",
                    "subject": "Test Email",
                    "from_email": "sender@example.com",
                    "date": "2024-01-01",
                    "excerpt": "Body",
                    "has_attachment": False,
                }
            ],
            "total_count": 10,
            "has_more": True,
            "next_page": 2,
        }

        with patch("email_service.search_emails", return_value=mock_result):
            response = client.post("/api/search", data={"search_input": "test query"})
            assert response.status_code == 200

    def test_search_empty_query(self, client):
        """Test search with empty query."""
        mock_result = {
            "emails": [],
            "total_count": 0,
            "has_more": False,
            "next_page": -1,
        }

        with patch("email_service.search_emails", return_value=mock_result):
            response = client.post("/api/search", data={"search_input": ""})
            assert response.status_code == 200


class TestStatsEndpoint:
    """Tests for statistics endpoints."""

    def test_stats_basic(self, client):
        """Test basic stats endpoint - returns empty list for unsupported query."""
        response = client.get("/api/stats/data/basic_stats")
        assert response.status_code == 200
        data = response.json()
        # The endpoint returns [] for unknown query names
        assert data == []

    def test_stats_email_sizes(self, client):
        """Test email sizes over time endpoint."""
        mock_data = [
            {"date": "2024-01-01", "count": 1000},
            {"date": "2024-02-01", "count": 1500},
        ]

        with patch("email_service.get_stats_time_series", return_value=mock_data):
            response = client.get("/api/stats/data/dates_size")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2

    def test_stats_unknown_query(self, client):
        """Test stats endpoint with unknown query name."""
        response = client.get("/api/stats/data/unknown_query")
        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestAttachmentEndpoint:
    """Tests for attachment download endpoint."""

    def test_download_attachment(self, client):
        """Test downloading an attachment."""
        mock_attachment = {
            "filename": "test.pdf",
            "content": b"PDF content",
            "content_type": "application/pdf",
        }

        with patch("email_server.get_attachment_file", return_value=mock_attachment):
            response = client.get("/api/attachment/<test1@example.com>/test.pdf")
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert "attachment" in response.headers["content-disposition"]

    def test_attachment_not_found(self, client):
        """Test downloading non-existent attachment."""
        with patch("email_server.get_attachment_file", return_value={}):
            response = client.get("/api/attachment/<test1@example.com>/nonexistent.pdf")
            assert response.status_code == 404
