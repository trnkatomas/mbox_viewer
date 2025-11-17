"""Tests for email_utils.py functions."""
import pytest
from email_utils import (
    to_string,
    parse_email,
    get_one_email,
    get_email_count,
    get_one_thread,
    get_basic_stats,
    get_email_sizes_in_time,
    MboxReader,
    _extract_body_content,
)
from email.parser import BytesParser
from email.policy import default


class TestToString:
    """Tests for to_string utility function."""

    def test_bytes_to_string(self):
        """Test converting bytes to string."""
        result = to_string(b"Hello World")
        assert result == "Hello World"
        assert isinstance(result, str)

    def test_bytearray_to_string(self):
        """Test converting bytearray to string."""
        result = to_string(bytearray(b"Hello"))
        assert result == "Hello"

    def test_string_passthrough(self):
        """Test that strings pass through unchanged."""
        result = to_string("Already a string")
        assert result == "Already a string"

    def test_non_ascii_bytes(self):
        """Test handling of non-ASCII bytes."""
        result = to_string(b"Hello\xff\xfeWorld")
        assert "Hello" in result
        assert "World" in result


class TestParseEmail:
    """Tests for parse_email function."""

    def test_parse_plain_text_email(self, sample_email_plain):
        """Test parsing a plain text email."""
        result = parse_email(sample_email_plain)

        assert 'body' in result
        assert 'attachments' in result
        body_type, body_content = result['body']
        assert body_type == 'Plain Text'
        assert 'plain text email' in body_content

    def test_parse_html_email(self, sample_email_html):
        """Test parsing an HTML email."""
        result = parse_email(sample_email_html)

        assert 'body' in result
        body_type, body_content = result['body']
        assert body_type == 'HTML'
        assert 'Hello World' in body_content

    def test_parse_invalid_email(self):
        """Test parsing invalid email data."""
        result = parse_email(b"Not a valid email")
        # Should not crash, but may return error or empty body
        assert 'body' in result
        assert 'attachments' in result


class TestExtractBodyContent:
    """Tests for _extract_body_content function."""

    def test_extract_plain_text(self, sample_email_plain):
        """Test extracting plain text body."""
        msg = BytesParser(policy=default).parsebytes(sample_email_plain)
        body_type, body_content = _extract_body_content(msg)

        assert body_type == 'Plain Text'
        assert 'plain text email' in body_content

    def test_extract_html(self, sample_email_html):
        """Test extracting HTML body."""
        msg = BytesParser(policy=default).parsebytes(sample_email_html)
        body_type, body_content = _extract_body_content(msg)

        assert body_type == 'HTML'
        assert '<h1>Hello World</h1>' in body_content

    def test_html_priority_over_plain(self):
        """Test that HTML is prioritized over plain text."""
        multipart_email = b"""From: test@example.com
To: user@example.com
Subject: Multipart Test
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain

Plain text version

--boundary123
Content-Type: text/html

<html><body>HTML version</body></html>

--boundary123--
"""
        msg = BytesParser(policy=default).parsebytes(multipart_email)
        body_type, body_content = _extract_body_content(msg)

        assert body_type == 'HTML'
        assert 'HTML version' in body_content


class TestMboxReader:
    """Tests for MboxReader class."""

    def test_read_mbox_file(self, sample_mbox_file):
        """Test reading emails from MBOX file."""
        with MboxReader(sample_mbox_file) as reader:
            emails = list(reader)

        assert len(emails) == 3
        # Each item should be (message, (start_pos, end_pos))
        msg1, pos1 = emails[0]
        assert msg1['Subject'] == 'Test Email 1'
        assert msg1['From'] == 'sender@example.com'

    def test_mbox_context_manager(self, sample_mbox_file):
        """Test that MboxReader works as context manager."""
        with MboxReader(sample_mbox_file) as reader:
            assert reader.handle is not None
            assert not reader.handle.closed

        # File should be closed after exiting context
        assert reader.handle.closed

    def test_mbox_byte_positions(self, sample_mbox_file):
        """Test that byte positions are tracked correctly."""
        with MboxReader(sample_mbox_file) as reader:
            emails = list(reader)

        for msg, (start, end) in emails:
            assert start < end
            assert start >= 0


class TestDatabaseFunctions:
    """Tests for database query functions."""

    def test_get_one_email(self, test_db):
        """Test retrieving a single email."""
        result = get_one_email(test_db, '<test1@example.com>')

        assert not result.empty
        assert result.iloc[0]['subject'] == 'Test Email 1'
        assert result.iloc[0]['from_email'] == 'sender@example.com'

    def test_get_one_email_not_found(self, test_db):
        """Test retrieving non-existent email."""
        result = get_one_email(test_db, '<nonexistent@example.com>')
        assert result.empty

    def test_get_email_count(self, test_db):
        """Test counting total emails."""
        count = get_email_count(test_db)
        assert count == 3

    def test_get_email_count_empty_db(self):
        """Test count on empty database."""
        import duckdb
        empty_db = duckdb.connect(":memory:")
        empty_db.execute("""
            CREATE TABLE emails (
                message_id VARCHAR PRIMARY KEY,
                subject VARCHAR,
                from_email VARCHAR,
                date TIMESTAMP,
                excerpt VARCHAR,
                has_attachment BOOLEAN,
                email_start INTEGER,
                email_end INTEGER,
                thread_id VARCHAR,
                labels VARCHAR
            )
        """)

        count = get_email_count(empty_db)
        assert count == 0
        empty_db.close()

    def test_get_one_thread(self, test_db):
        """Test retrieving emails in a thread."""
        result = get_one_thread(test_db, 'thread1')

        assert not result.empty
        assert all(result['thread_id'] == 'thread1')

    def test_get_basic_stats(self, test_db):
        """Test retrieving basic statistics."""
        stats = get_basic_stats(test_db)

        assert len(stats) == 3
        all_emails, all_size, all_timespan = stats

        # Check we got dataframes back
        assert not all_emails.empty
        assert not all_size.empty
        assert not all_timespan.empty

        # Check email count
        assert all_emails.iloc[0]['all_emails'] == 3

    def test_get_email_sizes_in_time(self, test_db):
        """Test getting email size statistics over time."""
        result = get_email_sizes_in_time(test_db)

        assert not result.empty
        assert 'date' in result.columns
        assert 'count' in result.columns
