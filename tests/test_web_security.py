"""Login, host checking and the self-signed HTTPS setup."""

import datetime
import os
import socket
import ssl
import stat
import threading
import time
import urllib.request
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import web_security
from web_security import (
    LoginThrottle,
    SecuritySettings,
    create_session_token,
    ensure_tls,
    hash_password,
    load_settings,
    safe_next_path,
    session_valid,
    verify_password,
)

PASSWORD = "correct horse battery staple"


@pytest.fixture(scope="module")
def password_hash() -> str:
    return hash_password(PASSWORD)


@pytest.fixture
def secured(monkeypatch, password_hash) -> Iterator[TestClient]:
    from email_server import app, registry

    monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
    monkeypatch.setenv("MBOX_VIEWER_PASSWORD_HASH", password_hash)
    monkeypatch.setenv("ALLOWED_HOSTS", "testserver,nas.local")
    registry.close()
    with TestClient(app, follow_redirects=False) as client:
        yield client
    registry.close()


def login(client: TestClient, password: str = PASSWORD, next_path: str = "/"):
    return client.post("/login", data={"password": password, "next": next_path})


class TestPasswords:
    def test_hash_round_trip(self, password_hash):
        assert verify_password(PASSWORD, password_hash)
        assert not verify_password("wrong", password_hash)

    def test_hash_is_safe_to_paste_into_compose(self, password_hash):
        assert "$" not in password_hash and password_hash.startswith("scrypt:17:8:1:")

    def test_malformed_hash_never_verifies(self):
        for stored in ["", "plaintext", "scrypt:1:1:1:x:y", "md5:a:b:c:d:e"]:
            assert not verify_password(PASSWORD, stored)


class TestSettings:
    def test_refuses_to_start_without_password(self, monkeypatch):
        monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
        with pytest.raises(RuntimeError, match="password"):
            load_settings()

    def test_rejects_malformed_hash_at_startup(self, monkeypatch):
        monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
        monkeypatch.setenv("MBOX_VIEWER_PASSWORD_HASH", "REPLACE_ME")
        with pytest.raises(ValueError):
            load_settings()

    def test_plain_password_is_hashed_in_memory(self, monkeypatch):
        monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
        monkeypatch.setenv("MBOX_VIEWER_PASSWORD", PASSWORD)
        settings = load_settings()
        assert settings.password_hash and PASSWORD not in settings.password_hash
        assert verify_password(PASSWORD, settings.password_hash)

    def test_session_key_persists_and_is_private(self, monkeypatch, tmp_path, password_hash):
        monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
        monkeypatch.setenv("MBOX_VIEWER_PASSWORD_HASH", password_hash)
        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        first = load_settings().session_secret
        assert load_settings().session_secret == first
        assert stat.S_IMODE(os.stat(tmp_path / "session.key").st_mode) == 0o600


class TestSessions:
    def _settings(self, password_hash: str, max_age: int = 3600) -> SecuritySettings:
        return SecuritySettings(True, password_hash, ["testserver"], "secret", max_age)

    def test_valid_token(self, password_hash):
        settings = self._settings(password_hash)
        assert session_valid(settings, create_session_token(settings))

    def test_tampered_and_foreign_tokens_rejected(self, password_hash):
        settings = self._settings(password_hash)
        token = create_session_token(settings)
        assert not session_valid(settings, token[:-2] + "xx")
        other = SecuritySettings(True, password_hash, [], "other-secret", 3600)
        assert not session_valid(settings, create_session_token(other))

    def test_expired_token_rejected(self, password_hash):
        settings = self._settings(password_hash, max_age=-1)
        assert not session_valid(settings, create_session_token(settings))

    def test_password_change_logs_everyone_out(self, password_hash):
        before = self._settings(password_hash)
        token = create_session_token(before)
        after = self._settings(hash_password("a new password"))
        assert not session_valid(after, token)

    def test_next_path_stays_on_site(self):
        assert safe_next_path("/api/email/list?page=2") == "/api/email/list?page=2"
        for bad in [None, "", "https://evil.example", "//evil.example", "/\\evil.example"]:
            assert safe_next_path(bad) == "/"


class TestHostCheck:
    def test_unknown_host_rejected(self, secured):
        response = secured.get("/login", headers={"Host": "evil.example"})
        assert response.status_code == 400

    def test_allowed_host_with_port(self, secured):
        assert secured.get("/login", headers={"Host": "nas.local:8000"}).status_code == 200

    def test_lookalike_host_rejected(self, secured):
        assert secured.get("/login", headers={"Host": "nas.local.evil.example"}).status_code == 400


class TestLoginFlow:
    def test_pages_redirect_to_login(self, secured):
        response = secured.get("/?mailbox=work")
        assert response.status_code == 303
        assert response.headers["location"] == "/login?next=%2F%3Fmailbox%3Dwork"

    def test_api_and_htmx_requests_get_401(self, secured):
        assert secured.get("/api/email/list").status_code == 401
        assert secured.get("/api/stats/data/dates_size").status_code == 401
        htmx = secured.get("/api/email/list", headers={"HX-Request": "true"})
        assert htmx.status_code == 401 and htmx.headers["HX-Redirect"] == "/login"

    def test_attachments_need_login(self, secured):
        assert secured.get("/api/attachment/%3Ca%40b%3E/0?mailbox=x").status_code == 401

    def test_static_files_and_ca_are_public(self, secured):
        assert secured.get("/static/email_frames.js").status_code == 200
        assert secured.get("/tls/ca.pem").status_code in (200, 404)

    def test_wrong_password(self, secured):
        response = login(secured, "nope")
        assert response.status_code == 401
        assert "Wrong password" in response.text
        assert web_security.SESSION_COOKIE not in secured.cookies

    def test_login_sets_strict_cookie_and_redirects(self, secured):
        response = login(secured, next_path="/?mailbox=work")
        assert response.status_code == 303
        assert response.headers["location"] == "/?mailbox=work"
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert secured.get("/").status_code == 200

    def test_login_never_redirects_off_site(self, secured):
        assert login(secured, next_path="//evil.example").headers["location"] == "/"

    def test_logout(self, secured):
        login(secured)
        response = secured.post("/logout")
        assert response.status_code == 303
        assert secured.get("/").status_code == 303

    def test_throttling(self, secured):
        for _ in range(5):
            assert login(secured, "nope").status_code == 401
        blocked = login(secured)  # even the right password waits
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) > 0

    def test_secure_cookie_over_https(self, monkeypatch, password_hash):
        from email_server import app

        monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
        monkeypatch.setenv("MBOX_VIEWER_PASSWORD_HASH", password_hash)
        with TestClient(app, base_url="https://testserver", follow_redirects=False) as client:
            assert "secure" in login(client).headers["set-cookie"].lower()


class TestThrottle:
    def test_overall_limit_spans_clients(self):
        throttle = LoginThrottle(per_client=100, overall=3)
        for i in range(3):
            throttle.record_failure(f"10.0.0.{i}")
        assert throttle.retry_after("10.0.0.99") > 0

    def test_success_resets_client(self):
        throttle = LoginThrottle(per_client=2, overall=100)
        throttle.record_failure("a")
        throttle.record_failure("a")
        assert throttle.retry_after("a") > 0
        throttle.reset("a")
        assert throttle.retry_after("a") == 0


class TestTls:
    def test_ca_signs_server_cert_with_host_names(self, tmp_path):
        from cryptography import x509

        files = ensure_tls(["nas.local", "192.168.1.10"], str(tmp_path))
        ca = x509.load_pem_x509_certificate(open(files.ca_cert, "rb").read())
        cert = x509.load_pem_x509_certificate(open(files.cert, "rb").read())

        cert.verify_directly_issued_by(ca)
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert set(san.get_values_for_type(x509.DNSName)) == {"localhost", "nas.local"}
        assert {str(ip) for ip in san.get_values_for_type(x509.IPAddress)} == {"127.0.0.1", "192.168.1.10"}
        lifetime = cert.not_valid_after_utc - cert.not_valid_before_utc
        assert lifetime <= datetime.timedelta(days=826)
        assert stat.S_IMODE(os.stat(files.key).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(tmp_path / "tls" / "ca-key.pem").st_mode) == 0o600

    def test_reissues_only_when_hosts_change(self, tmp_path):
        files = ensure_tls(["nas.local"], str(tmp_path))
        ca_before, cert_before = open(files.ca_cert).read(), open(files.cert).read()

        ensure_tls(["nas.local"], str(tmp_path))
        assert open(files.cert).read() == cert_before

        ensure_tls(["nas.local", "nas.lan"], str(tmp_path))
        assert open(files.cert).read() != cert_before
        assert open(files.ca_cert).read() == ca_before  # devices keep trusting the same CA

    def test_real_https_connection_verifies_against_the_ca(self, tmp_path, monkeypatch, password_hash):
        """Start the real server over TLS and connect trusting only the generated CA."""
        import uvicorn

        from email_server import app

        monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
        monkeypatch.setenv("MBOX_VIEWER_PASSWORD_HASH", password_hash)
        monkeypatch.setenv("ALLOWED_HOSTS", "localhost")
        files = ensure_tls(["localhost"], str(tmp_path))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, ssl_certfile=files.cert,
                           ssl_keyfile=files.key, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            context = ssl.create_default_context(cafile=files.ca_cert)
            with urllib.request.urlopen(f"https://localhost:{port}/login", context=context) as response:
                assert response.status == 200
                assert "Sign in" in response.read().decode()
            plain = ssl.create_default_context()  # system trust store only
            with pytest.raises(urllib.error.URLError):
                urllib.request.urlopen(f"https://localhost:{port}/login", context=plain)
        finally:
            server.should_exit = True
            thread.join(timeout=5)


class TestSecurityMode:
    @pytest.fixture(autouse=True)
    def _no_overrides(self, monkeypatch):
        monkeypatch.delenv("MBOX_VIEWER_AUTH", raising=False)
        monkeypatch.delenv("MBOX_VIEWER_TLS", raising=False)

    def test_full_is_the_default(self):
        assert web_security.auth_enabled() and web_security.tls_enabled()

    def test_local_turns_off_login_and_tls(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MBOX_VIEWER_SECURITY", "local")
        read_only = tmp_path / "ro"
        read_only.mkdir(mode=0o500)
        monkeypatch.setenv("STATE_DIR", str(read_only / "state"))

        settings = load_settings()  # no password, no writable state folder needed

        assert not settings.auth_enabled and not web_security.tls_enabled()
        assert not (read_only / "state").exists()

    def test_individual_switches_override_the_mode(self, monkeypatch, password_hash):
        monkeypatch.setenv("MBOX_VIEWER_SECURITY", "local")
        monkeypatch.setenv("MBOX_VIEWER_AUTH", "on")
        with pytest.raises(RuntimeError, match="password"):
            load_settings()

        monkeypatch.setenv("MBOX_VIEWER_SECURITY", "full")
        monkeypatch.delenv("MBOX_VIEWER_AUTH")
        monkeypatch.setenv("MBOX_VIEWER_TLS", "off")
        monkeypatch.setenv("MBOX_VIEWER_PASSWORD_HASH", password_hash)
        assert load_settings().auth_enabled and not web_security.tls_enabled()

    def test_typos_fail_instead_of_guessing(self, monkeypatch):
        monkeypatch.setenv("MBOX_VIEWER_SECURITY", "lcoal")
        with pytest.raises(ValueError, match="MBOX_VIEWER_SECURITY"):
            web_security.auth_enabled()
        monkeypatch.setenv("MBOX_VIEWER_SECURITY", "full")
        monkeypatch.setenv("MBOX_VIEWER_TLS", "of")
        with pytest.raises(ValueError, match="MBOX_VIEWER_TLS"):
            web_security.tls_enabled()

    def test_viewer_shows_that_login_is_off(self, monkeypatch, password_hash):
        from email_server import app

        monkeypatch.setenv("MBOX_VIEWER_SECURITY", "local")
        with TestClient(app) as client:
            assert "No login: anyone who can reach this viewer" in client.get("/").text

        monkeypatch.setenv("MBOX_VIEWER_SECURITY", "full")
        monkeypatch.setenv("MBOX_VIEWER_PASSWORD_HASH", password_hash)
        with TestClient(app) as client:
            login(client)
            page = client.get("/").text
            assert "No login" not in page and 'action="/logout"' in page
