"""Pytest fixtures for mbox_viewer tests."""

import os
import tempfile
from email.message import EmailMessage
from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def sample_mbox_file():
    """Create a temporary MBOX file with sample emails."""
    content = b"""From sender@example.com Mon Jan 01 00:00:00 2024
From: sender@example.com
To: recipient@example.com
Subject: Test Email 1
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <test1@example.com>

This is the body of the first test email.

From another@example.com Mon Jan 02 00:00:00 2024
From: another@example.com
To: recipient@example.com
Subject: Test Email 2
Date: Tue, 02 Jan 2024 13:00:00 +0000
Message-ID: <test2@example.com>
Content-Type: text/html

<html><body><p>This is an HTML email.</p></body></html>

From attach@example.com Mon Jan 03 00:00:00 2024
From: attach@example.com
To: recipient@example.com
Subject: Email with Attachment
Date: Wed, 03 Jan 2024 14:00:00 +0000
Message-ID: <test3@example.com>
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary123"

--boundary123
Content-Type: text/plain

This email has an attachment.

--boundary123
Content-Type: application/pdf; name="test.pdf"
Content-Disposition: attachment; filename="test.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBl
--boundary123--

"""

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".mbox") as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def sample_email_html():
    """Sample HTML email content."""
    return b"""From: test@example.com
To: user@example.com
Subject: HTML Test
Message-ID: <html-test@example.com>
Content-Type: text/html

<html>
<body>
<h1>Hello World</h1>
<p>This is a test email.</p>
</body>
</html>
"""


@pytest.fixture
def sample_email_plain():
    """Sample plain text email content."""
    return b"""From: test@example.com
To: user@example.com
Subject: Plain Test
Message-ID: <plain-test@example.com>
Content-Type: text/plain

Hello,

This is a plain text email.

Best regards
"""


@pytest.fixture
def test_db():
    """Create an in-memory test database with sample data."""
    con = duckdb.connect(":memory:")

    # Create emails table
    con.execute(
        """
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
    """
    )

    # Insert sample data
    con.execute(
        """
        INSERT INTO emails VALUES
        ('<test1@example.com>', 'Test Email 1', 'sender@example.com',
         '2024-01-01 12:00:00', 'This is the body...', false, 0, 100, 'thread1', 'inbox'),
        ('<test2@example.com>', 'Test Email 2', 'another@example.com',
         '2024-01-02 13:00:00', 'This is an HTML...', false, 100, 200, 'thread2', 'inbox'),
        ('<test3@example.com>', 'Email with Attachment', 'attach@example.com',
         '2024-01-03 14:00:00', 'This email has...', true, 200, 300, 'thread3', 'sent')
    """
    )

    yield con

    con.close()


@pytest.fixture
def mock_mbox_path(sample_mbox_file, monkeypatch):
    """Mock the MBOX_FILE_PATH environment variable."""
    monkeypatch.setenv("MBOX_FILE_PATH", sample_mbox_file)
    return sample_mbox_file
