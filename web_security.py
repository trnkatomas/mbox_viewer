"""
Login, host checking and HTTPS for the web viewer.

CLI:
    python web_security.py hash-password   # prompt for a password, print its hash
    python web_security.py init            # create the session key and TLS files
"""

import base64
import datetime
import hashlib
import hmac
import ipaddress
import logging
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from itsdangerous import BadSignature, URLSafeTimedSerializer

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1"
DEFAULT_SESSION_HOURS = 24 * 7
SESSION_COOKIE = "mbox_session"

# ---------------------------------------------------------------------------
# Passwords: scrypt with OWASP's recommended cost (N=2^17, r=8, p=1, 128 MiB)
# ---------------------------------------------------------------------------

SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P = 17, 8, 1
SCRYPT_MAXMEM = 256 * 1024 * 1024
# ":" separators, base64url fields: no "$", which compose files would try to interpolate
HASH_PREFIX = "scrypt"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _scrypt(password: str, salt: bytes, log_n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**log_n, r=r, p=p, maxmem=SCRYPT_MAXMEM, dklen=32
    )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P)
    return ":".join([HASH_PREFIX, str(SCRYPT_LOG_N), str(SCRYPT_R), str(SCRYPT_P), _b64(salt), _b64(digest)])


def _parse_hash(stored: str) -> Tuple[int, int, int, bytes, bytes]:
    parts = stored.strip().split(":")
    if len(parts) != 6 or parts[0] != HASH_PREFIX:
        raise ValueError("not a password hash from `python web_security.py hash-password`")
    log_n, r, p = (int(x) for x in parts[1:4])
    if not (10 <= log_n <= 22 and 1 <= r <= 32 and 1 <= p <= 16):
        raise ValueError("password hash has out-of-range parameters")
    return log_n, r, p, _unb64(parts[4]), _unb64(parts[5])


def verify_password(password: str, stored: str) -> bool:
    try:
        log_n, r, p, salt, expected = _parse_hash(stored)
    except ValueError:
        return False
    return hmac.compare_digest(_scrypt(password, salt, log_n, r, p), expected)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class SecuritySettings:
    auth_enabled: bool
    password_hash: Optional[str]
    allowed_hosts: List[str]
    session_secret: str
    session_max_age: int

    def host_allowed(self, host_header: str) -> bool:
        """Same matching as Starlette's TrustedHostMiddleware: exact, '*.domain' or '*'."""
        if host_header.startswith("["):  # IPv6 literal, optionally with a port
            host = host_header[: host_header.find("]") + 1]
        else:
            host = host_header.rsplit(":", 1)[0]
        host = host.lower()
        for pattern in self.allowed_hosts:
            if pattern == "*" or host == pattern:
                return True
            if pattern.startswith("*.") and host.endswith(pattern[1:]):
                return True
        return False


def allowed_hosts_from_env() -> List[str]:
    return [h.strip().lower() for h in os.getenv("ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS).split(",") if h.strip()]


SECURITY_MODES = ("full", "local")


def security_mode() -> str:
    """
    MBOX_VIEWER_SECURITY: "full" (default) requires a login and serves HTTPS;
    "local" turns both off, for a quick look at an mbox on your own machine.
    """
    mode = (os.getenv("MBOX_VIEWER_SECURITY") or "full").strip().lower()
    if mode not in SECURITY_MODES:
        raise ValueError(f"MBOX_VIEWER_SECURITY must be one of {', '.join(SECURITY_MODES)}, not {mode!r}")
    return mode


def _env_flag(name: str, default: bool) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    if value in ("on", "1", "true", "yes"):
        return True
    if value in ("off", "0", "false", "no"):
        return False
    raise ValueError(f"{name} must be on or off, not {value!r}")


def auth_enabled() -> bool:
    """Login required? MBOX_VIEWER_AUTH overrides the security mode."""
    return _env_flag("MBOX_VIEWER_AUTH", security_mode() == "full")


def tls_enabled() -> bool:
    """Serve HTTPS? MBOX_VIEWER_TLS overrides the security mode."""
    return _env_flag("MBOX_VIEWER_TLS", security_mode() == "full")


def get_state_dir() -> str:
    return os.getenv("STATE_DIR", "state")


def _read_or_create_secret(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            secret = f.read().strip()
        if secret:
            return secret
    except FileNotFoundError:
        pass
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        secret = secrets.token_urlsafe(48)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secret)
        return secret
    except OSError as e:
        logger.debug("Could not create %s: %s", path, e)
        return None


def load_settings(state_dir: Optional[str] = None) -> SecuritySettings:
    """
    Read security settings from the environment. With login enabled (the
    default) a password is required, so a misconfigured server refuses to
    start instead of serving the mailbox to everyone.
    """
    state_dir = state_dir or get_state_dir()
    login_required = auth_enabled()

    password_hash = os.getenv("MBOX_VIEWER_PASSWORD_HASH") or None
    if password_hash:
        _parse_hash(password_hash)  # fail at startup, not at the first login
    elif os.getenv("MBOX_VIEWER_PASSWORD"):
        password_hash = hash_password(os.environ["MBOX_VIEWER_PASSWORD"])
    if login_required and not password_hash:
        raise RuntimeError(
            "No login password configured. Set MBOX_VIEWER_PASSWORD_HASH (from "
            "`python web_security.py hash-password`), or MBOX_VIEWER_SECURITY=local for a "
            "viewer only you can reach."
        )
    if not login_required:
        logger.warning("Login is off: anyone who can reach the viewer can read the mail")

    allowed_hosts = allowed_hosts_from_env()

    secret: Optional[str]
    if not login_required:
        # Nothing to sign, so don't require a writable state folder
        secret = secrets.token_urlsafe(48)
    else:
        secret = os.getenv("MBOX_VIEWER_SESSION_SECRET") or _read_or_create_secret(
            os.path.join(state_dir, "session.key")
        )
    if not secret:
        logger.warning(
            "No session key in %s and it isn't writable; logins will not survive a restart. "
            "Run `python web_security.py init`.",
            state_dir,
        )
        secret = secrets.token_urlsafe(48)

    hours = float(os.getenv("MBOX_VIEWER_SESSION_HOURS", str(DEFAULT_SESSION_HOURS)))
    return SecuritySettings(
        auth_enabled=login_required,
        password_hash=password_hash,
        allowed_hosts=allowed_hosts,
        session_secret=secret,
        session_max_age=int(hours * 3600),
    )


# ---------------------------------------------------------------------------
# Sessions: a signed, expiring cookie. The token carries a fingerprint of the
# password hash, so changing the password logs every session out.
# ---------------------------------------------------------------------------


def _serializer(settings: SecuritySettings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.session_secret, salt="mbox-viewer-session")


def _password_fingerprint(settings: SecuritySettings) -> str:
    return hashlib.sha256((settings.password_hash or "").encode()).hexdigest()[:16]


def create_session_token(settings: SecuritySettings) -> str:
    return _serializer(settings).dumps({"pw": _password_fingerprint(settings)})


def session_valid(settings: SecuritySettings, token: Optional[str]) -> bool:
    if not token:
        return False
    try:
        data = _serializer(settings).loads(token, max_age=settings.session_max_age)
    except BadSignature:
        return False
    return isinstance(data, dict) and hmac.compare_digest(
        str(data.get("pw", "")), _password_fingerprint(settings)
    )


# ---------------------------------------------------------------------------
# Login throttling: per client and overall, so guessing stays slow even from
# many addresses. In memory, which is enough for a single viewer process.
# ---------------------------------------------------------------------------


class LoginThrottle:
    def __init__(self, per_client: int = 5, overall: int = 30, window_seconds: int = 15 * 60) -> None:
        self.per_client = per_client
        self.overall = overall
        self.window = window_seconds
        self._failures: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> Deque[float]:
        failures = self._failures[key]
        while failures and failures[0] <= now - self.window:
            failures.popleft()
        return failures

    def retry_after(self, client: str) -> int:
        """Seconds until this client may try again, 0 if it may now."""
        now = time.monotonic()
        with self._lock:
            waits = []
            for key, limit in ((client, self.per_client), ("*", self.overall)):
                failures = self._prune(key, now)
                if len(failures) >= limit:
                    waits.append(failures[0] + self.window - now)
            return int(max(waits)) + 1 if waits else 0

    def record_failure(self, client: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._failures[client].append(now)
            self._failures["*"].append(now)

    def reset(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)


def safe_next_path(target: Optional[str]) -> str:
    """Only same-site paths, so the login form can't be used to redirect elsewhere."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


# ---------------------------------------------------------------------------
# TLS: a local certificate authority signs the server certificate. Trusting
# the CA once on each device removes the browser warning, and later server
# certificates (new host names, renewals) need no re-trusting.
# ---------------------------------------------------------------------------

CA_DAYS = 3650
# Apple platforms reject TLS server certificates valid for longer than 825 days
SERVER_DAYS = 825
RENEW_BEFORE_DAYS = 30


@dataclass
class TlsFiles:
    ca_cert: str
    cert: str
    key: str


def tls_files(state_dir: Optional[str] = None) -> TlsFiles:
    tls_dir = os.path.join(state_dir or get_state_dir(), "tls")
    return TlsFiles(
        ca_cert=os.path.join(tls_dir, "ca.pem"),
        cert=os.path.join(tls_dir, "cert.pem"),
        key=os.path.join(tls_dir, "key.pem"),
    )


def _san_names(hosts: Sequence[str]) -> List[str]:
    names = {"localhost", "127.0.0.1"}
    names.update(h for h in hosts if h and h != "*")
    return sorted(names)


def _write_private(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _write_public(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def ensure_tls(hosts: Sequence[str], state_dir: Optional[str] = None) -> TlsFiles:
    """
    Create the CA and server certificate if missing, and reissue the server
    certificate when it nears expiry or the host names changed.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    files = tls_files(state_dir)
    tls_dir = os.path.dirname(files.cert)
    os.makedirs(tls_dir, exist_ok=True)
    ca_key_path = os.path.join(tls_dir, "ca-key.pem")
    now = datetime.datetime.now(datetime.timezone.utc)

    if os.path.exists(files.ca_cert) and os.path.exists(ca_key_path):
        with open(files.ca_cert, "rb") as f:
            ca_cert = x509.load_pem_x509_certificate(f.read())
        with open(ca_key_path, "rb") as f:
            ca_key = serialization.load_pem_private_key(f.read(), password=None)
    else:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"MBox Viewer local CA {secrets.token_hex(4)}")])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=CA_DAYS))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                    key_encipherment=False, data_encipherment=False, key_agreement=False,
                    encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        _write_private(
            ca_key_path,
            ca_key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ),
        )
        _write_public(files.ca_cert, ca_cert.public_bytes(serialization.Encoding.PEM))
        logger.info("Created local certificate authority %s", files.ca_cert)

    wanted = _san_names(hosts)
    if os.path.exists(files.cert) and os.path.exists(files.key):
        with open(files.cert, "rb") as f:
            current = x509.load_pem_x509_certificate(f.read())
        try:
            san = current.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            have = sorted([str(n) for n in san.get_values_for_type(x509.DNSName)]
                          + [str(ip) for ip in san.get_values_for_type(x509.IPAddress)])
        except x509.ExtensionNotFound:
            have = []
        if (
            have == wanted
            and current.issuer == ca_cert.subject
            and current.not_valid_after_utc - now > datetime.timedelta(days=RENEW_BEFORE_DAYS)
        ):
            return files

    alt_names: List[x509.GeneralName] = []
    for name in wanted:
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(name.strip("[]"))))
        except ValueError:
            alt_names.append(x509.DNSName(name))
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, wanted[0])]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=SERVER_DAYS))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()), critical=False  # type: ignore[arg-type]
        )
        .sign(ca_key, hashes.SHA256())  # type: ignore[arg-type]
    )
    _write_private(
        files.key,
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )
    _write_public(files.cert, cert.public_bytes(serialization.Encoding.PEM))
    logger.info("Issued server certificate for %s", ", ".join(wanted))
    return files


if __name__ == "__main__":
    import argparse
    import getpass

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Login and HTTPS setup for the web viewer.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hash-password", help="prompt for a password and print its hash")
    commands.add_parser(
        "init", help="create the session key and TLS certificates in STATE_DIR (run where it is writable)"
    )
    args = parser.parse_args()

    if args.command == "hash-password":
        first = getpass.getpass("Password: ")
        if len(first) < 12:
            parser.error("use at least 12 characters")
        if getpass.getpass("Repeat: ") != first:
            parser.error("passwords don't match")
        print(hash_password(first))
        # Only the hash goes to stdout, so `$(... hash-password)` captures it cleanly
        print("Set MBOX_VIEWER_PASSWORD_HASH to the line above.", file=sys.stderr)
    else:
        state_dir = get_state_dir()
        if not _read_or_create_secret(os.path.join(state_dir, "session.key")):
            parser.error(f"could not create {state_dir}/session.key")
        files = ensure_tls(allowed_hosts_from_env(), state_dir)
        print(f"Session key: {os.path.join(state_dir, 'session.key')}")
        print(f"Certificate: {files.cert}")
        print(f"CA to trust on your devices: {files.ca_cert}")
