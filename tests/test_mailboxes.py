"""Multiple mailboxes: discovery, indexing, switching in the web app and MCP server."""

import json
import os
from email.message import EmailMessage
from typing import List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

import email_utils
from email_utils import discover_mailboxes, index_mailboxes

# (message id, subject, body, attachment filename or None)
Msg = Tuple[str, str, str, Optional[str]]


def write_mbox(path: str, messages: List[Msg]) -> None:
    chunks = []
    for i, (message_id, subject, body, attachment) in enumerate(messages):
        msg = EmailMessage()
        msg["From"] = "Sender <sender@example.com>"
        msg["To"] = "me@example.com"
        msg["Subject"] = subject
        msg["Date"] = f"Mon, 0{i + 1} Jan 2024 10:00:00 +0000"
        msg["Message-ID"] = message_id
        msg.set_content(body)
        if attachment:
            msg.add_attachment(b"attachment bytes", maintype="application", subtype="pdf", filename=attachment)
        chunks.append(f"From sender@example.com Mon Jan 0{i + 1} 10:00:00 2024\n".encode() + msg.as_bytes() + b"\n")
    with open(path, "wb") as f:
        f.write(b"".join(chunks))


@pytest.fixture
def mail_env(tmp_path, monkeypatch):
    """MBOX_DIR with two indexed mailboxes that share a Message-ID, plus INDEX_DIR."""
    mbox_dir = tmp_path / "mail"
    index_dir = tmp_path / "index"
    mbox_dir.mkdir()
    index_dir.mkdir()
    write_mbox(
        str(mbox_dir / "personal.mbox"),
        [("<shared@example.com>", "Holiday photos", "Pictures from the mountains.", None)],
    )
    write_mbox(
        str(mbox_dir / "work.mbox"),
        [
            ("<shared@example.com>", "Quarterly report", "The numbers are in.", "report.pdf"),
            ("<work-2@example.com>", "Standup notes", "Nothing blocking.", None),
        ],
    )
    monkeypatch.delenv("MBOX_FILE_PATH", raising=False)
    monkeypatch.setenv("MBOX_DIR", str(mbox_dir))
    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    results = index_mailboxes(discover_mailboxes(), embeddings=False)
    assert all("error" not in r for r in results.values()), results
    return mbox_dir, index_dir


@pytest.fixture
def web(mail_env):
    from email_server import app, registry

    registry.close()
    with TestClient(app) as client:
        yield client
    registry.close()


class TestDiscovery:
    def test_scans_dir_recursively_and_skips_hidden(self, tmp_path, monkeypatch):
        (tmp_path / "Takeout" / "Mail").mkdir(parents=True)
        (tmp_path / ".zfs").mkdir()
        for name in ["a.mbox", "Takeout/Mail/All mail.mbox", ".zfs/snap.mbox", ".hidden.mbox", "notes.txt"]:
            (tmp_path / name).write_bytes(b"From x Mon Jan 1 00:00:00 2024\n\n")
        monkeypatch.delenv("MBOX_FILE_PATH", raising=False)
        monkeypatch.setenv("MBOX_DIR", str(tmp_path))
        monkeypatch.setenv("INDEX_DIR", "/idx")

        mailboxes = {m.id: m for m in discover_mailboxes()}

        assert set(mailboxes) == {"a", "takeout-mail-all-mail"}
        assert mailboxes["takeout-mail-all-mail"].name == "Takeout/Mail/All mail"
        assert mailboxes["a"].db_path == os.path.join("/idx", "a.db")

    def test_single_file_keeps_its_index_and_is_not_duplicated(self, tmp_path, monkeypatch):
        (tmp_path / "emails.mbox").write_bytes(b"From x Mon Jan 1 00:00:00 2024\n\n")
        monkeypatch.setenv("MBOX_FILE_PATH", str(tmp_path / "emails.mbox"))
        monkeypatch.setenv("EMAILS_DB_PATH", "/legacy/emails.db")
        monkeypatch.setenv("MBOX_DIR", str(tmp_path))

        mailboxes = discover_mailboxes()

        assert len(mailboxes) == 1
        assert mailboxes[0].db_path == "/legacy/emails.db"

    def test_mailboxes_never_share_an_index(self, tmp_path, monkeypatch):
        for name in ["takeout.mbox", "emails.mbox", "work-.mbox", "work_.mbox"]:
            (tmp_path / name).write_bytes(b"From x Mon Jan 1 00:00:00 2024\n\n")
        monkeypatch.setenv("MBOX_FILE_PATH", str(tmp_path / "takeout.mbox"))
        monkeypatch.setenv("EMAILS_DB_PATH", str(tmp_path / "idx" / "emails.db"))
        monkeypatch.setenv("MBOX_DIR", str(tmp_path))
        monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))

        mailboxes = discover_mailboxes()

        db_paths = [m.db_path for m in mailboxes]
        assert len(set(db_paths)) == len(db_paths) == 4
        assert {m.id for m in mailboxes} == {"takeout", "emails-2", "work", "work-2"}

    def test_missing_single_file_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MBOX_FILE_PATH", str(tmp_path / "gone.mbox"))
        monkeypatch.delenv("MBOX_DIR", raising=False)
        assert discover_mailboxes() == []


class TestIndexing:
    def test_each_mailbox_gets_its_own_index(self, mail_env):
        _, index_dir = mail_env
        assert sorted(os.listdir(index_dir)) == ["personal.db", "work.db"]

    def test_one_broken_mbox_does_not_stop_the_others(self, mail_env):
        mbox_dir, _ = mail_env
        (mbox_dir / "broken.mbox").write_bytes(b"this is not an mbox\n")

        results = index_mailboxes(discover_mailboxes(), embeddings=False)

        assert "error" in results["broken"]
        assert "error" not in results["work"]


class TestWebSwitching:
    def test_menu_lists_mailboxes_and_remembers_choice(self, web):
        page = web.get("/").text
        assert 'id="mailbox-select"' in page
        assert '<option value="personal" selected>' in page
        assert '<option value="work" >' in page

        page = web.get("/?mailbox=work").text
        assert '<option value="work" selected>' in page
        assert web.cookies.get("mailbox") == "work"
        assert '"X-Mailbox": "work"' in page

    def test_list_follows_the_header(self, web):
        personal = web.get("/api/email/list", headers={"X-Mailbox": "personal"}).text
        work = web.get("/api/email/list", headers={"X-Mailbox": "work"}).text
        assert "Holiday photos" in personal and "Quarterly report" not in personal
        assert "Quarterly report" in work and "Holiday photos" not in work

    def test_cookie_is_used_without_header(self, web):
        web.get("/?mailbox=work")
        assert "Standup notes" in web.get("/api/email/list").text

    def test_same_message_id_reads_the_right_mbox(self, web):
        url = "/api/email/%3Cshared%40example.com%3E"
        personal = web.get(url, headers={"X-Mailbox": "personal"}).text
        work = web.get(url, headers={"X-Mailbox": "work"}).text
        assert "Pictures from the mountains." in personal
        assert "The numbers are in." in work and "mountains" not in work

    def test_attachment_link_names_the_mailbox(self, web):
        detail = web.get("/api/email/%3Cshared%40example.com%3E", headers={"X-Mailbox": "work"}).text
        assert "/api/attachment/%3Cshared%40example.com%3E/0?mailbox=work" in detail
        response = web.get("/api/attachment/%3Cshared%40example.com%3E/0?mailbox=work")
        assert response.status_code == 200
        assert response.content == b"attachment bytes"

    def test_unknown_mailbox_is_an_error_not_a_fallback(self, web):
        for bad in ["nope", "../../etc/passwd"]:
            text = web.get("/api/email/list", headers={"X-Mailbox": bad}).text
            assert "There is no mailbox" in text
            assert "Holiday photos" not in text

    def test_chart_data_takes_query_parameter(self, web):
        assert web.get("/api/stats/data/domains_count?mailbox=work").json()[0]["count"] == 2
        assert web.get("/api/stats/data/domains_count?mailbox=nope").json() == []

    def test_unindexed_mailbox_explains_itself(self, web, mail_env):
        mbox_dir, _ = mail_env
        write_mbox(str(mbox_dir / "new.mbox"), [("<n@example.com>", "New", "Body", None)])

        assert "new (not indexed)" in " ".join(web.get("/").text.split())
        text = web.get("/api/email/list", headers={"X-Mailbox": "new"}).text
        assert "hasn&#39;t been indexed yet" in text
        assert "--mailbox new" in text

    def test_replaced_mbox_shows_a_warning(self, web, mail_env):
        mbox_dir, _ = mail_env
        write_mbox(str(mbox_dir / "work.mbox"), [("<other@example.com>", "Different", "Different file", None)])

        layout = web.get("/api/inbox/layout", headers={"X-Mailbox": "work"}).text

        assert "doesn&#39;t match the mbox file" in layout

    def test_no_mailboxes_configured(self, tmp_path, monkeypatch):
        from email_server import app, registry

        monkeypatch.delenv("MBOX_FILE_PATH", raising=False)
        monkeypatch.setenv("MBOX_DIR", str(tmp_path))
        registry.close()
        with TestClient(app) as client:
            assert "No mailboxes found" in client.get("/").text
            assert "No mailboxes found" in client.get("/api/email/list").text
        registry.close()


class TestMcpMailboxes:
    @pytest.fixture(autouse=True)
    def _fresh_connections(self):
        import mcp_server

        for con in mcp_server._dbs.values():
            con.close()
        mcp_server._dbs.clear()
        yield
        for con in mcp_server._dbs.values():
            con.close()
        mcp_server._dbs.clear()

    def test_list_and_search_by_mailbox(self, mail_env):
        from mcp_server import list_mailboxes_tool, search_emails_tool

        listed = json.loads(list_mailboxes_tool())
        assert [m["id"] for m in listed] == ["personal", "work"]

        work = json.loads(search_emails_tool("", mailbox="work"))
        assert work["total_count"] == 2
        default = json.loads(search_emails_tool(""))
        assert default["total_count"] == 1

    def test_get_email_reads_the_right_mbox(self, mail_env):
        from mcp_server import get_email_tool

        work = json.loads(get_email_tool("<shared@example.com>", mailbox="work"))
        assert "The numbers are in." in work["body"]

    def test_unknown_mailbox_is_reported(self, mail_env):
        from mcp_server import search_emails_tool

        result = json.loads(search_emails_tool("", mailbox="nope"))
        assert "Unknown mailbox" in result["error"]
