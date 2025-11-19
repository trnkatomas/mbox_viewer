"""Tests for FastAPI endpoints in email_server.py."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


@pytest.fixture
def mock_db_connections():
    """Mock database connections for testing."""
    mock_duckdb = Mock()

    # Setup mock responses
    mock_duckdb.execute = Mock()
    mock_duckdb.close = Mock()

    with patch('email_server.db_connections', {'duckdb': mock_duckdb}):
        yield {'duckdb': mock_duckdb}


@pytest.fixture
def client(mock_db_connections):
    """Create a test client with mocked dependencies."""
    # Mock the lifespan to avoid actual DB connections
    with patch('email_server.load_email_db', return_value=mock_db_connections['duckdb']):
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
        with patch('email_server.get_email_count', return_value=10):
            response = client.get("/api/inbox/layout")
            assert response.status_code == 200

    def test_sent_layout(self, client):
        """Test sent folder layout endpoint."""
        with patch('email_server.get_email_count', return_value=5):
            response = client.get("/api/sent/layout")
            assert response.status_code == 200

    def test_stats_layout(self, client):
        """Test statistics layout endpoint."""
        import datetime
        mock_stats = [
            pd.DataFrame({'all_emails': [100]}),
            pd.DataFrame({'avg_size': [5000]}),
            pd.DataFrame({
                'first_seen': [datetime.datetime(2024, 1, 1)],
                'last_seen': [datetime.datetime(2024, 12, 31)]
            })
        ]

        with patch('email_server.get_basic_stats', return_value=mock_stats):
            response = client.get("/api/stats/layout")
            assert response.status_code == 200


class TestEmailEndpoints:
    """Tests for email-related endpoints."""

    def test_email_list_endpoint(self, client):
        """Test email list endpoint."""
        mock_emails = pd.DataFrame({
            'message_id': ['<test1@example.com>', '<test2@example.com>'],
            'subject': ['Test 1', 'Test 2'],
            'from_email': ['sender1@example.com', 'sender2@example.com'],
            'date': ['2024-01-01', '2024-01-02'],
            'excerpt': ['Body 1', 'Body 2'],
            'has_attachment': [False, True]
        })

        with patch('email_server.get_email_list', return_value=mock_emails):
            with patch('email_server.get_email_count', return_value=10):
                response = client.get("/api/email/list?page=1")
                assert response.status_code == 200

    def test_email_list_with_search(self, client):
        """Test email list with search query."""
        mock_emails = pd.DataFrame({
            'message_id': ['<test1@example.com>'],
            'subject': ['Test Email'],
            'from_email': ['sender@example.com'],
            'date': ['2024-01-01'],
            'excerpt': ['Body'],
            'has_attachment': [False]
        })

        with patch('email_server.get_email_list', return_value=mock_emails):
            with patch('email_server.get_email_count', return_value=10):
                response = client.get("/api/email/list?page=1&query=test")
                assert response.status_code == 200

    def test_email_detail_endpoint(self, client):
        """Test single email detail endpoint."""
        mock_email = pd.DataFrame({
            'message_id': ['<test1@example.com>'],
            'subject': ['Test Email'],
            'from_email': ['sender@example.com'],
            'date': ['2024-01-01 12:00:00'],
            'has_attachment': [False],
            'email_start': [0],
            'email_end': [100]
        })

        mock_raw_email = b'From: sender@example.com\nSubject: Test\n\nBody'
        mock_parsed_email = {
            'body': ('Plain Text', 'This is the email body'),
            'attachments': []
        }

        with patch('email_server.get_one_email', return_value=mock_email):
            with patch('email_server.get_string_email_from_mboxfile', return_value=mock_raw_email):
                with patch('email_server.parse_email', return_value=mock_parsed_email):
                    with patch('email_server.get_thread_for_email', return_value=pd.DataFrame()):
                        response = client.get("/api/email/<test1@example.com>")
                        assert response.status_code == 200

    def test_email_detail_not_found(self, client):
        """Test email detail with non-existent email."""
        with patch('email_server.get_one_email', return_value=pd.DataFrame()):
            response = client.get("/api/email/<nonexistent@example.com>")
            # Should handle gracefully - may return 200 with empty content or 404
            assert response.status_code in [200, 404]

    def test_email_thread_endpoint(self, client):
        """Test email thread endpoint."""
        mock_thread = pd.DataFrame({
            'message_id': ['<test1@example.com>', '<test2@example.com>'],
            'subject': ['Re: Test', 'Re: Test'],
            'from_email': ['sender1@example.com', 'sender2@example.com'],
            'date': ['2024-01-01 12:00:00', '2024-01-01 13:00:00'],
            'thread_id': ['thread1', 'thread1'],
            'has_attachment': [False, False],
            'email_start': [0, 100],
            'email_end': [100, 200]
        })

        mock_parsed_email = {
            'body': ('text/html', '<p>Test email body</p>'),
            'attachments': []
        }

        with patch('email_server.get_one_thread', return_value=mock_thread):
            with patch('email_server.get_string_email_from_mboxfile', return_value=b'From: test@example.com\nSubject: Test\n\nBody'):
                with patch('email_server.parse_email', return_value=mock_parsed_email):
                    response = client.get("/api/email_thread/thread1")
                    assert response.status_code == 200


class TestSearchEndpoint:
    """Tests for search endpoint."""

    def test_search_basic(self, client):
        """Test basic search."""
        mock_results = pd.DataFrame({
            'message_id': ['<test1@example.com>'],
            'subject': ['Test Email'],
            'from_email': ['sender@example.com'],
            'date': ['2024-01-01'],
            'excerpt': ['Body'],
            'has_attachment': [False]
        })

        with patch('email_server.get_email_list', return_value=mock_results):
            with patch('email_server.get_email_count', return_value=10):
                response = client.post("/api/search", data={"search_input": "test query"})
                assert response.status_code == 200

    def test_search_empty_query(self, client):
        """Test search with empty query."""
        with patch('email_server.get_email_list', return_value=pd.DataFrame()):
            with patch('email_server.get_email_count', return_value=0):
                response = client.post("/api/search", data={"search_input": ""})
                assert response.status_code == 200


class TestStatsEndpoint:
    """Tests for statistics endpoints."""

    def test_stats_basic(self, client):
        """Test basic stats endpoint - returns empty dict for unsupported query."""
        response = client.get("/api/stats/data/basic_stats")
        assert response.status_code == 200
        data = response.json()
        # The endpoint returns {} for unknown query names
        assert data == {}

    def test_stats_email_sizes(self, client):
        """Test email sizes over time endpoint."""
        mock_sizes = pd.DataFrame({
            'date': ['2024-01-01', '2024-02-01'],
            'count': [1000, 1500]
        })

        with patch('email_server.get_email_sizes_in_time', return_value=mock_sizes):
            response = client.get("/api/stats/data/dates_size")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2

    def test_stats_unknown_query(self, client):
        """Test stats endpoint with unknown query name."""
        response = client.get("/api/stats/data/unknown_query")
        assert response.status_code == 200
        data = response.json()
        assert data == {}


class TestAttachmentEndpoint:
    """Tests for attachment download endpoint."""

    def test_download_attachment(self, client):
        """Test downloading an attachment."""
        mock_attachment = {
            'filename': 'test.pdf',
            'content': b'PDF content',
            'content_type': 'application/pdf'
        }

        with patch('email_server.get_attachment_file', return_value=mock_attachment):
            response = client.get("/api/attachment/<test1@example.com>/test.pdf")
            assert response.status_code == 200
            assert response.headers['content-type'] == 'application/pdf'
            assert 'attachment' in response.headers['content-disposition']

    def test_attachment_not_found(self, client):
        """Test downloading non-existent attachment."""
        with patch('email_server.get_attachment_file', return_value={}):
            response = client.get("/api/attachment/<test1@example.com>/nonexistent.pdf")
            assert response.status_code == 404
