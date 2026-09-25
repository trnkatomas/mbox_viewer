"""
Data access, parsing and indexing for the mbox archive.

The DuckDB file holds metadata (``emails``), chunk embeddings (``embeddings``)
and index settings (``index_meta``). Raw messages are never copied into the
database; they are read back from the mbox by byte offset.

CLI:
    python email_utils.py index [--rebuild]   # build or resume the index
    python email_utils.py index --embeddings off  # metadata only, no Ollama needed
    python email_utils.py migrate             # upgrade a pre-chunking database
"""

import email.utils
import hashlib
import logging
import mmap
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.message import Message
from email.parser import BytesParser
from email.policy import default
from html.parser import HTMLParser
from types import TracebackType
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import duckdb
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "emails.db"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"

# Embedding models are trained with task prefixes; using the wrong ones (or
# none) silently degrades retrieval, so they are keyed by model family.
MODEL_PREFIXES: Dict[str, Tuple[str, str]] = {
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
    "nomic-embed-text": ("search_query: ", "search_document: "),
}

# Cosine distance cut-off for semantic search hits (0 = identical, 2 = opposite).
# With embeddinggemma, relevant mail typically lands at 0.5-0.75 and emails
# embedded from empty text around 0.8+, so results are ranked and this only
# drops the clearly unrelated tail.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.75"))

EXCERPT_CHARS = 200
BODY_TEXT_MAX_CHARS = 100_000
CHUNK_CHARS = 2000  # roughly 500 tokens
CHUNK_OVERLAP_CHARS = 200
MAX_CHUNKS_PER_EMAIL = 20


class RagUnavailableError(RuntimeError):
    """Semantic search was requested but the query could not be embedded."""


class InvalidQueryError(ValueError):
    """A search filter has a value that cannot be used (e.g. a malformed date)."""


@dataclass
class Email:
    """Email metadata from database."""

    message_id: str
    subject: str
    from_email: str
    to_email: str
    date: datetime
    excerpt: str
    has_attachment: int
    email_start: int
    email_end: int
    thread_id: str
    labels: List[str] = field(default_factory=list)
    content_type: Optional[str] = None
    mbox_file_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[Any, Any]) -> "Email":
        """Create Email from dictionary (e.g., from DataFrame record)."""
        return cls(
            message_id=data["message_id"],
            subject=data.get("subject", ""),
            from_email=data.get("from_email", ""),
            to_email=data.get("to_email", ""),
            date=data["date"],
            excerpt=data.get("excerpt", ""),
            has_attachment=data.get("has_attachment", 0),
            email_start=int(data["email_start"]),
            email_end=int(data["email_end"]),
            thread_id=data.get("thread_id", ""),
            labels=data.get("labels", []),
            content_type=data.get("content_type"),
            mbox_file_id=data.get("mbox_file_id"),
        )


def get_mbox_path() -> str:
    path = os.getenv("MBOX_FILE_PATH")
    if not path:
        raise ValueError("MBOX_FILE_PATH environment variable must be set")
    return path


def get_db_path() -> str:
    return os.getenv("EMAILS_DB_PATH", DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# mbox reading
# ---------------------------------------------------------------------------

# A real From_ separator carries a timestamp ("From x Mon Jan 01 00:00:00 2024");
# requiring it keeps an unescaped "From " at the start of a body line from
# splitting a message in two.
_FROM_LINE_RE = re.compile(rb"^From \S+.*\d{1,2}:\d{2}(:\d{2})?")


def _is_from_line(line: bytes) -> bool:
    return line.startswith(b"From ") and _FROM_LINE_RE.match(line) is not None


class MboxReader:
    """Streams messages from an mbox file along with their byte offsets."""

    def __init__(self, filename: str, start: int = 0) -> None:
        self.handle = open(filename, "rb")
        self.start = start
        self.handle.seek(start)
        first_line = self.handle.readline()
        self.handle.seek(start)
        # Resuming exactly at the end of the file is fine: there is nothing new.
        at_eof = start > 0 and not first_line
        if not at_eof and not first_line.startswith(b"From "):
            self.handle.close()
            where = f" at byte {start}" if start else ""
            raise ValueError(f"{filename} does not look like an mbox file{where}")

    def __enter__(self) -> "MboxReader":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        self.handle.close()

    def __iter__(self) -> Iterator[Tuple[Message, Tuple[int, int]]]:
        lines: List[bytes] = []
        start = pos = self.start
        for line in self.handle:
            if lines and _is_from_line(line):
                yield _message_from_lines(lines), (start, pos)
                lines = []
                start = pos
            lines.append(line)
            pos += len(line)
        if lines:
            yield _message_from_lines(lines), (start, pos)


def _message_from_lines(lines: List[bytes]) -> Message:
    return BytesParser(policy=default).parsebytes(b"".join(lines))


def read_mbox_slice(start: int, end: int) -> bytes:
    """Read one raw message from the mbox by byte offsets."""
    with open(get_mbox_path(), "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm[start:end]


def to_string(_content: Union[bytes, bytearray, str]) -> str:
    if isinstance(_content, (bytes, bytearray)):
        return bytes(_content).decode("utf-8", errors="replace")
    return _content


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "head", "title", "noscript"}
    _BLOCK = {
        "p", "div", "br", "tr", "li", "ul", "ol", "table", "h1", "h2", "h3",
        "h4", "h5", "h6", "blockquote", "pre", "hr", "section", "article",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return normalize_whitespace("".join(self._parts))


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces within lines and drop blank lines."""
    lines = (re.sub(r"[ \t\r\f\v ]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def html_to_text(content: str) -> str:
    """Convert HTML to readable plain text, keeping paragraph breaks."""
    extractor = _TextExtractor()
    try:
        extractor.feed(content)
        extractor.close()
        return extractor.text()
    except Exception:
        return content


def _is_attachment_part(part: Message) -> bool:
    disposition = part.get("Content-Disposition")
    return bool(part.get_filename()) or (
        disposition is not None and str(disposition).lower().startswith("attachment")
    )


def _iter_body_parts(msg: Message) -> Iterator[Tuple[str, str]]:
    """Yield (content_type, decoded_text) for every inline text part."""
    for i, part in enumerate(msg.walk()):
        if part.is_multipart() or _is_attachment_part(part):
            continue
        if part.get_content_maintype() != "text":
            continue
        content_type = part.get_content_type()
        try:
            content = part.get_content()  # type: ignore[attr-defined]
            if not isinstance(content, str):
                content = to_string(content)
        except Exception as e:
            # Unknown or wrong charsets are common in old mail; fall back to raw bytes.
            logger.debug("Part %d (%s) failed to decode: %s", i, content_type, e)
            payload = part.get_payload(decode=True)
            content = to_string(payload) if isinstance(payload, bytes) else ""
        yield content_type, content


def _extract_body_content(msg: Message) -> Tuple[str, Optional[str]]:
    """
    Pick the body to display: the last HTML part, else the first plain-text part.
    Returns (body_type, content) where body_type is "HTML", "Plain Text" or "None".
    """
    html_body: Optional[str] = None
    text_body: Optional[str] = None
    for content_type, content in _iter_body_parts(msg):
        if content_type == "text/html":
            html_body = content
        elif content_type == "text/plain" and text_body is None:
            text_body = content
    if html_body:
        return ("HTML", html_body)
    if text_body:
        return ("Plain Text", text_body)
    return ("None", None)


def extract_index_text(msg: Message) -> str:
    """Text used for search and embeddings: plain part preferred over HTML."""
    html_body: Optional[str] = None
    for content_type, content in _iter_body_parts(msg):
        if content_type == "text/plain" and content.strip():
            return normalize_whitespace(content)
        if content_type == "text/html" and html_body is None:
            html_body = content
    return html_to_text(html_body) if html_body else ""


def _extract_attachments(
    msg: Message, include_content: bool = True
) -> List[Dict[str, Any]]:
    """List attachment parts; the decoded bytes are included only on request."""
    attachments: List[Dict[str, Any]] = []
    for part in msg.walk():
        if part.is_multipart() or not _is_attachment_part(part):
            continue
        filename = part.get_filename()
        try:
            payload = part.get_payload(decode=True)
            if not isinstance(payload, bytes):
                payload = b"" if payload is None else str(payload).encode("utf-8")
        except Exception as e:
            logger.error("Error decoding attachment %s: %s", filename, e)
            payload = b""
        attachment: Dict[str, Any] = {
            "filename": filename or "Untitled",
            "content_type": part.get_content_type(),
            "size_bytes": len(payload),
            "content_id": _content_id(part),
        }
        if include_content:
            attachment["content"] = payload
        attachments.append(attachment)
    return attachments


def _content_id(part: Message) -> Optional[str]:
    cid = part.get("Content-ID")
    return str(cid).strip().strip("<>") if cid else None


def _extract_inline_images(msg: Message) -> Dict[str, Tuple[str, bytes]]:
    """Map Content-ID -> (mime type, bytes) for images referenced as cid: URLs."""
    images: Dict[str, Tuple[str, bytes]] = {}
    for part in msg.walk():
        if part.is_multipart() or part.get_content_maintype() != "image":
            continue
        cid = _content_id(part)
        if not cid:
            continue
        payload = part.get_payload(decode=True)
        if isinstance(payload, bytes):
            images[cid] = (part.get_content_type(), payload)
    return images


def parse_email(
    raw_email: bytes, include_attachment_content: bool = True
) -> Dict[str, Any]:
    """
    Parse a raw message into its display body, attachments and inline images.

    Returns a dict with:
    - 'body': (body_type, content or None)
    - 'attachments': list of {filename, content_type, size_bytes, content_id[, content]}
    - 'inline_images': {content_id: (mime_type, bytes)}
    """
    try:
        msg = BytesParser(policy=default).parsebytes(raw_email)
    except Exception as e:
        logger.error("Failed to parse raw email: %s", e)
        return {"body": ("Error", f"Parsing failed: {e}"), "attachments": [], "inline_images": {}}

    return {
        "body": _extract_body_content(msg),
        "attachments": _extract_attachments(msg, include_attachment_content),
        "inline_images": _extract_inline_images(msg),
    }


def load_and_parse_email(email_meta: Email, include_attachment_content: bool = False) -> Dict[str, Any]:
    raw = read_mbox_slice(email_meta.email_start, email_meta.email_end)
    return parse_email(raw, include_attachment_content)


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------


def load_email_db(db_name: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_name or get_db_path(), read_only=True)


def get_one_email(db: duckdb.DuckDBPyConnection, email_id: str) -> pd.DataFrame:
    return db.execute("select * from emails where message_id = ? limit 1", [email_id]).df()


def get_one_thread(db: duckdb.DuckDBPyConnection, thread_id: str) -> pd.DataFrame:
    """Emails in a thread, oldest first. An empty id is not a thread."""
    if not thread_id:
        return pd.DataFrame()
    return db.execute(
        "select * from emails where thread_id = ? order by date asc", [thread_id]
    ).df()


def get_basic_stats(db: duckdb.DuckDBPyConnection) -> List[pd.DataFrame]:
    row = db.execute(
        "select count(*) as all_emails, avg(email_end - email_start) as avg_size,"
        " min(date) as first_seen, max(date) as last_seen from emails"
    ).df()
    return [row[["all_emails"]], row[["avg_size"]], row[["first_seen", "last_seen"]]]


def get_email_sizes_in_time(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Cumulative archive size (bytes) per month."""
    return db.execute(
        """
        select month as date, sum(size) over (order by month) as count
        from (
            select date_trunc('month', date) as month, sum(email_end - email_start) as size
            from emails where date is not null group by month
        )
        order by month
        """
    ).df()


def get_domains_by_count(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Top 10 sender domains by message count."""
    return db.execute(
        """
        select lower(regexp_extract(from_email, '@([^>\\s]+)', 1)) as domain, count(*) as count
        from emails
        where from_email is not null
        group by domain
        having domain != ''
        order by count desc
        limit 10
        """
    ).df()


def _has_column(db: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    rows = db.execute(
        "select 1 from information_schema.columns where table_name = ? and column_name = ?",
        [table, column],
    ).fetchall()
    return bool(rows)


def _parse_iso_date(value: str, key: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise InvalidQueryError(f"{key} must be a date in YYYY-MM-DD format, got {value!r}")


def build_email_filter(
    criteria: Optional[Mapping[str, str]],
    sent: bool = False,
    keyword_column: str = "excerpt",
) -> Tuple[str, List[Any]]:
    """
    Translate parsed search criteria into a WHERE clause and its parameters.
    All filters combine with AND; text matches are case-insensitive substrings.
    """
    clauses: List[str] = []
    params: List[Any] = []
    criteria = criteria or {}

    if value := criteria.get("from"):
        clauses.append("from_email ilike ?")
        params.append(f"%{value}%")
    if value := criteria.get("subject"):
        clauses.append("subject ilike ?")
        params.append(f"%{value}%")
    if value := criteria.get("label"):
        clauses.append("list_contains(labels, ?)")
        params.append(value)
    if value := criteria.get("excerpt"):
        clauses.append(f"(subject ilike ? or {keyword_column} ilike ?)")
        params.extend([f"%{value}%", f"%{value}%"])
    if value := criteria.get("from_date"):
        clauses.append("date >= ?")
        params.append(_parse_iso_date(value, "from_date"))
    if value := criteria.get("to_date"):
        # to_date is inclusive of the whole day
        clauses.append("date < ?")
        params.append(_parse_iso_date(value, "to_date") + timedelta(days=1))
    if sent:
        clauses.append("list_contains(labels, 'Sent')")

    return (" and ".join(clauses) or "true"), params


def _search_sql(
    db: duckdb.DuckDBPyConnection,
    select: str,
    criteria: Optional[Mapping[str, str]],
    sent: bool,
    query_vec: Optional[Sequence[float]],
) -> Tuple[str, List[Any]]:
    keyword_column = (
        "coalesce(body_text, excerpt)" if _has_column(db, "emails", "body_text") else "excerpt"
    )
    where, params = build_email_filter(criteria, sent, keyword_column)
    if query_vec is None:
        return f"select {select} from emails where {where}", params

    dim = len(query_vec)
    sql = f"""
        with scored as (
            select message_id, min(array_cosine_distance(vec, ?::FLOAT[{dim}])) as dist
            from embeddings where vec is not null
            group by message_id
        )
        select {select} from emails join scored using (message_id)
        where scored.dist < ? and {where}
    """
    return sql, [list(query_vec), RAG_MAX_DISTANCE] + params


def get_email_list(
    db: duckdb.DuckDBPyConnection,
    limit: int = 30,
    offset: int = 0,
    criteria: Optional[Mapping[str, str]] = None,
    sent: bool = False,
    query_vec: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    One page of emails matching the filters. With ``query_vec`` the results are
    semantic-search hits ordered by distance (``dist`` column), otherwise newest first.
    """
    select = "emails.*" if query_vec is None else "emails.*, scored.dist"
    sql, params = _search_sql(db, select, criteria, sent, query_vec)
    order = "date desc" if query_vec is None else "scored.dist asc"
    return db.execute(f"{sql} order by {order} limit ? offset ?", params + [limit, offset]).df()


def get_email_count(
    db: duckdb.DuckDBPyConnection,
    criteria: Optional[Mapping[str, str]] = None,
    sent: bool = False,
    query_vec: Optional[Sequence[float]] = None,
) -> int:
    sql, params = _search_sql(db, "count(*)", criteria, sent, query_vec)
    row = db.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingConfig:
    model: str
    dim: Optional[int]
    query_prefix: str
    document_prefix: str


def _prefixes_for(model: str) -> Tuple[str, str]:
    return MODEL_PREFIXES.get(model.split(":")[0], ("", ""))


def get_embedding_config(db: duckdb.DuckDBPyConnection) -> EmbeddingConfig:
    """
    The embedding settings the index was built with. Queries must use the same
    model and prefixes, so these take precedence over OLLAMA_MODEL.
    """
    meta: Dict[str, str] = {}
    if _table_exists(db, "index_meta"):
        meta = dict(db.execute("select key, value from index_meta").fetchall())

    model = meta.get("embedding_model") or os.getenv("OLLAMA_MODEL") or DEFAULT_EMBEDDING_MODEL
    query_prefix, document_prefix = _prefixes_for(model)
    dim: Optional[int] = int(meta["embedding_dim"]) if "embedding_dim" in meta else None
    if dim is None and _table_exists(db, "embeddings"):
        row = db.execute(
            "select data_type from information_schema.columns"
            " where table_name = 'embeddings' and column_name = 'vec'"
        ).fetchone()
        match = re.search(r"\[(\d+)\]", row[0]) if row else None
        dim = int(match.group(1)) if match else None
    return EmbeddingConfig(
        model=model,
        dim=dim,
        query_prefix=meta.get("query_prefix", query_prefix),
        document_prefix=meta.get("document_prefix", document_prefix),
    )


def _table_exists(db: duckdb.DuckDBPyConnection, table: str) -> bool:
    rows = db.execute(
        "select 1 from information_schema.tables where table_name = ?", [table]
    ).fetchall()
    return bool(rows)


def get_ollama_embeddings(
    texts: Sequence[str],
    model: Optional[str] = None,
    server_url: Optional[str] = None,
    timeout: float = 120,
) -> Optional[List[List[float]]]:
    """Embed a batch of texts with one Ollama call. Returns None on failure."""
    server_url = server_url or os.getenv("OLLAMA_URL") or DEFAULT_OLLAMA_URL
    model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_EMBEDDING_MODEL
    try:
        response = requests.post(
            server_url, json={"model": model, "input": list(texts)}, timeout=timeout
        )
    except requests.RequestException as e:
        logger.error("Failed to reach Ollama at %s: %s", server_url, e)
        return None
    if response.status_code != 200:
        logger.error("Ollama embedding failed with status %s: %s", response.status_code, response.text[:200])
        return None
    embeddings = response.json().get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        logger.error("Unexpected Ollama response: expected %d embeddings", len(texts))
        return None
    return embeddings


def get_ollama_embedding(
    text: str, server_url: Optional[str] = None, model: Optional[str] = None
) -> Optional[List[float]]:
    result = get_ollama_embeddings([text], model=model, server_url=server_url, timeout=30)
    return result[0] if result else None


def pull_ollama_model(model: str, server_url: Optional[str] = None) -> bool:
    """Ask Ollama to download ``model``; blocks until done. Returns success."""
    server_url = server_url or os.getenv("OLLAMA_URL") or DEFAULT_OLLAMA_URL
    pull_url = server_url.split("/api/", 1)[0] + "/api/pull"
    logger.info("Embedding model '%s' unavailable; trying to pull it via %s", model, pull_url)
    try:
        response = requests.post(pull_url, json={"model": model, "stream": False}, timeout=3600)
    except requests.RequestException as e:
        logger.error("Failed to pull '%s': %s", model, e)
        return False
    if response.status_code != 200:
        logger.error("Pulling '%s' failed with status %s: %s", model, response.status_code, response.text[:200])
        return False
    return True


def embed_query(db: duckdb.DuckDBPyConnection, query_text: str) -> List[float]:
    """Embed a search query with the index's model; raises RagUnavailableError."""
    config = get_embedding_config(db)
    if config.dim is None:
        raise RagUnavailableError(
            "Semantic search is unavailable: the index has no embeddings "
            "(it was built without them; re-run `python email_utils.py index` with Ollama available)"
        )
    vec = get_ollama_embedding(config.query_prefix + query_text, model=config.model)
    if not vec:
        raise RagUnavailableError(
            f"Semantic search is unavailable: could not embed the query with "
            f"'{config.model}' (is Ollama running at {os.getenv('OLLAMA_URL', DEFAULT_OLLAMA_URL)}?)"
        )
    if config.dim is not None and len(vec) != config.dim:
        raise RagUnavailableError(
            f"Embedding model '{config.model}' returned {len(vec)} dimensions but the "
            f"index was built with {config.dim}; rebuild the index or change OLLAMA_MODEL"
        )
    return vec


def rag_search_duckdb(
    db: duckdb.DuckDBPyConnection, query_text: str, n_results: int = 50
) -> pd.DataFrame:
    """Semantic search returning message_id, subject and dist, closest first."""
    if not query_text:
        return pd.DataFrame()
    vec = embed_query(db, query_text)
    return get_email_list(db, limit=n_results, query_vec=vec)[["message_id", "subject", "dist"]]


def chunk_text(
    text: str,
    max_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
    max_chunks: int = MAX_CHUNKS_PER_EMAIL,
) -> List[str]:
    """Split text into overlapping chunks, breaking on whitespace where possible."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start + max_chars // 2, end)
            newline = text.rfind("\n", start + max_chars // 2, end)
            end = max(space, newline) if max(space, newline) > start else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

EMAILS_DDL = """
create table if not exists emails (
    message_id text primary key,
    subject text,
    from_email text,
    to_email text,
    date timestamp,
    has_attachment integer,
    excerpt text,
    body_text text,
    labels text[],
    content_type text,
    mbox_file_id text,
    email_start bigint,
    email_end bigint,
    thread_id text
)
"""

INDEX_META_DDL = "create table if not exists index_meta (key text primary key, value text)"


def _embeddings_ddl(dim: int) -> str:
    return f"""
    create table if not exists embeddings (
        message_id text,
        chunk_index integer,
        mbox_file_id text,
        vec FLOAT[{dim}],
        primary key (message_id, chunk_index)
    )
    """


def _file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed: Optional[datetime]
    try:
        parsed = email.utils.parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        from dateparser import parse

        parsed = parse(str(value))
    if parsed is None:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _header(message: Message, name: str) -> str:
    value = message.get(name)
    return str(value).strip() if value is not None else ""


def _email_row(
    message: Message, message_id: str, text: str, mbox_file_id: str, boundaries: Tuple[int, int]
) -> List[Any]:
    labels_header = _header(message, "X-Gmail-Labels")
    return [
        message_id,
        _header(message, "Subject"),
        _header(message, "From"),
        _header(message, "To"),
        _parse_date(message.get("Date")),
        sum(1 for part in message.walk() if not part.is_multipart() and _is_attachment_part(part)),
        " ".join(text.split())[:EXCERPT_CHARS],
        text[:BODY_TEXT_MAX_CHARS],
        [label.strip() for label in labels_header.split(",") if label.strip()],
        _header(message, "Content-Type") or None,
        mbox_file_id,
        boundaries[0],
        boundaries[1],
        _header(message, "X-GM-THRID"),
    ]


def _check_schema(con: duckdb.DuckDBPyConnection) -> None:
    if _table_exists(con, "emails") and not _has_column(con, "emails", "body_text"):
        raise RuntimeError(
            "The database uses the old schema. Run `python email_utils.py migrate` "
            "or `python email_utils.py index --rebuild`."
        )


def check_index_freshness(
    db: duckdb.DuckDBPyConnection, mbox_path: str, samples: int = 16
) -> Optional[str]:
    """
    Check that the byte offsets stored in the index still point at the same
    messages. Appending to the mbox keeps the index valid; replacing it (e.g.
    with a new Google Takeout export) does not. Returns None when the index
    matches, otherwise what is wrong. Only a handful of messages are read, so
    this is cheap even for a large mbox on network storage.
    """
    if not _table_exists(db, "emails"):
        return None
    row = db.execute("select max(email_end) from emails").fetchone()
    if not row or row[0] is None:
        return None
    indexed_end = int(row[0])
    size = os.path.getsize(mbox_path)
    if size < indexed_end:
        return f"the mbox is {size} bytes but the index refers to byte {indexed_end}"

    rows = db.execute(
        """
        select message_id, email_start, email_end from emails
        where email_start = (select min(email_start) from emails)
           or email_end = (select max(email_end) from emails)
           or message_id in (select message_id from emails order by hash(message_id) limit ?)
        """,
        [samples],
    ).fetchall()
    with open(mbox_path, "rb") as f:
        for message_id, start, end in rows:
            f.seek(int(start))
            head = f.read(min(int(end) - int(start), 256 * 1024))
            if not _is_from_line(head.split(b"\n", 1)[0]):
                return f"no message starts at byte {start} (expected {message_id})"
            headers = BytesParser(policy=default).parsebytes(head, headersonly=True)
            if _header(headers, "Message-ID") != message_id:
                return f"the message at byte {start} is not {message_id}"
        if size > indexed_end:
            f.seek(indexed_end)
            if not _is_from_line(f.readline()):
                return f"the data after byte {indexed_end} does not start a new message"
    return None


def build_index(
    mbox_path: Optional[str] = None,
    db_path: Optional[str] = None,
    rebuild: bool = False,
    model: Optional[str] = None,
    batch_size: int = 32,
    embeddings: Union[bool, str] = True,
    rebuild_if_stale: bool = False,
) -> Dict[str, int]:
    """
    Index the mbox into DuckDB in two passes: message metadata (no Ollama
    needed), then embeddings for every message that does not have them yet.
    Both passes pick up where a previous run stopped, and only the part of
    the mbox after the last indexed message is read, so re-running against an
    unchanged or appended-to mbox is cheap. Embedding failures abort the run
    (everything committed so far is kept) instead of storing empty vectors.

    ``embeddings`` is True, False, or "auto" to skip them (with a warning)
    when Ollama cannot provide the model. If the mbox no longer matches the
    stored offsets the run fails, or with ``rebuild_if_stale`` the index is
    rebuilt from scratch.
    """
    mbox_path = mbox_path or get_mbox_path()
    db_path = db_path or get_db_path()

    stats = {
        "indexed": 0,
        "skipped_existing": 0,
        "skipped_no_id": 0,
        "embedded": 0,
        "chunks": 0,
        "resumed_at": 0,
    }

    with duckdb.connect(db_path) as con:
        if not rebuild:
            _check_schema(con)
            problem = check_index_freshness(con, mbox_path)
            if problem and not rebuild_if_stale:
                raise RuntimeError(f"The index does not match the mbox ({problem}); use --rebuild")
            if problem:
                logger.warning("The index does not match the mbox (%s); rebuilding", problem)
                rebuild = True
        if rebuild:
            for table in ("emails", "embeddings", "index_meta"):
                con.execute(f"drop table if exists {table}")

        con.execute(EMAILS_DDL)
        con.execute(INDEX_META_DDL)
        # Checked before reading the mbox so a missing model fails fast.
        embedder = _prepare_embeddings(con, model, optional=embeddings == "auto") if embeddings else None
        _index_messages(con, mbox_path, stats)
        if embedder:
            _embed_missing(con, embedder[0], embedder[1], batch_size, stats)
        con.execute("checkpoint")

    logger.info("Indexing finished: %s", stats)
    return stats


def _prepare_embeddings(
    con: duckdb.DuckDBPyConnection, model: Optional[str], optional: bool = False
) -> Optional[Tuple[str, str]]:
    """
    Validate the model against the index and create the table; returns (model,
    document prefix), or None when Ollama is unavailable and ``optional``.
    """
    config = get_embedding_config(con)
    model = model or config.model
    if config.dim is not None and config.model != model:
        raise RuntimeError(
            f"Index was built with '{config.model}', not '{model}'; use --rebuild to switch models"
        )
    query_prefix, document_prefix = _prefixes_for(model)

    probe_text = [document_prefix + "dimension probe"]
    probe = get_ollama_embeddings(probe_text, model=model, timeout=60)
    if not probe and pull_ollama_model(model):
        probe = get_ollama_embeddings(probe_text, model=model, timeout=60)
    if not probe:
        message = f"Could not get an embedding from Ollama with model '{model}'"
        if optional:
            logger.warning("%s; indexing without embeddings (rag: search unavailable)", message)
            return None
        raise RuntimeError(f"{message} (use --embeddings off to index without semantic search)")
    dim = len(probe[0])
    if config.dim is not None and config.dim != dim:
        raise RuntimeError(f"Model returns {dim}-d vectors but the index has {config.dim}-d; use --rebuild")

    con.execute(_embeddings_ddl(dim))
    con.executemany(
        "insert or replace into index_meta values (?, ?)",
        [
            ["embedding_model", model],
            ["embedding_dim", str(dim)],
            ["query_prefix", query_prefix],
            ["document_prefix", document_prefix],
        ],
    )
    return model, document_prefix


def _index_messages(con: duckdb.DuckDBPyConnection, mbox_path: str, stats: Dict[str, int]) -> None:
    import tqdm

    row = con.execute("select max(email_end) from emails").fetchone()
    resume_at = int(row[0]) if row and row[0] is not None else 0
    stats["resumed_at"] = resume_at
    if resume_at and resume_at >= os.path.getsize(mbox_path):
        logger.info("No new messages in the mbox since the last run")
        return

    mbox_file_id = _file_sha256(mbox_path)
    logger.info("mbox SHA256: %s", mbox_file_id)
    seen = {r[0] for r in con.execute("select message_id from emails").fetchall()}
    pending: List[List[Any]] = []

    def flush() -> None:
        if not pending:
            return
        con.begin()
        con.executemany(
            "insert into emails (message_id, subject, from_email, to_email, date,"
            " has_attachment, excerpt, body_text, labels, content_type, mbox_file_id,"
            " email_start, email_end, thread_id) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            pending,
        )
        con.commit()
        stats["indexed"] += len(pending)
        pending.clear()

    with MboxReader(mbox_path, start=resume_at) as mbox:
        for message, boundaries in tqdm.tqdm(mbox, unit="email", desc="Reading mbox"):
            message_id = _header(message, "Message-ID")
            if not message_id:
                stats["skipped_no_id"] += 1
                continue
            if message_id in seen:
                stats["skipped_existing"] += 1
                continue
            seen.add(message_id)

            try:
                text = extract_index_text(message)
            except Exception as e:
                logger.warning("Failed to extract text from %s: %s", message_id, e)
                text = ""
            pending.append(_email_row(message, message_id, text, mbox_file_id, boundaries))
            if len(pending) >= 500:
                flush()
    flush()


def _embed_missing(
    con: duckdb.DuckDBPyConnection,
    model: str,
    document_prefix: str,
    batch_size: int,
    stats: Dict[str, int],
) -> None:
    """Embed every indexed message without embeddings, from the text stored in the index."""
    import tqdm

    missing = [
        r[0]
        for r in con.execute(
            "select message_id from emails e where not exists"
            " (select 1 from embeddings x where x.message_id = e.message_id) order by email_start"
        ).fetchall()
    ]
    pending: List[Tuple[str, str, List[str]]] = []

    def flush() -> None:
        if not pending:
            return
        texts = [document_prefix + chunk for _, _, chunks in pending for chunk in chunks]
        vectors = get_ollama_embeddings(texts, model=model)
        if vectors is None:
            raise RuntimeError(
                "Embedding failed; progress so far is saved, re-run the index command to resume"
            )
        vec_iter = iter(vectors)
        embedding_rows = [
            [message_id, chunk_index, mbox_file_id, next(vec_iter)]
            for message_id, mbox_file_id, chunks in pending
            for chunk_index in range(len(chunks))
        ]
        con.begin()
        con.executemany("insert into embeddings values (?, ?, ?, ?)", embedding_rows)
        con.commit()
        stats["embedded"] += len(pending)
        stats["chunks"] += len(embedding_rows)
        pending.clear()

    with tqdm.tqdm(total=len(missing), unit="email", desc="Embedding") as progress:
        for i in range(0, len(missing), 500):
            rows = con.execute(
                "select message_id, subject, body_text, mbox_file_id from emails"
                " join (select unnest(?::text[]) as message_id) using (message_id)",
                [missing[i : i + 500]],
            ).fetchall()
            for message_id, subject, body_text, mbox_file_id in rows:
                text = body_text or ""
                chunks = chunk_text(f"{subject}\n\n{text}" if subject else text)
                if chunks:
                    pending.append((message_id, mbox_file_id, chunks))
                if sum(len(c) for _, _, c in pending) >= batch_size:
                    flush()
            progress.update(len(rows))
    flush()


def migrate_db(db_path: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Upgrade a database created before chunked embeddings: widen byte offsets
    to BIGINT, drop the experimental HNSW index and unused columns, add the new
    columns and record the embedding model. The result is written to a fresh
    file (which also reclaims space); the original is kept as ``<db>.bak``.
    """
    db_path = db_path or get_db_path()
    tmp_path = db_path + ".migrating"
    backup_path = db_path + ".bak"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with duckdb.connect(db_path, read_only=True) as source:
        config = get_embedding_config(source)
    model = model or config.model
    if config.dim is None:
        raise RuntimeError("No embeddings table found; nothing to migrate")

    with duckdb.connect(tmp_path) as con:
        escaped_path = db_path.replace("'", "''")
        con.execute(f"attach '{escaped_path}' as old (read_only)")
        con.execute(EMAILS_DDL)
        con.execute(
            """
            insert into emails
            select message_id, subject, from_email, to_email, date, has_attachment,
                   excerpt, null, labels, content_type, mbox_file_id,
                   email_start::bigint, email_end::bigint, thread_id
            from old.emails
            where message_id is not null
            qualify row_number() over (partition by message_id order by email_start) = 1
            """
        )
        con.execute(_embeddings_ddl(config.dim))
        con.execute(
            """
            insert into embeddings
            select message_id, 0, mbox_file_id, vec from old.embeddings
            where message_id is not null and vec is not null
            qualify row_number() over (partition by message_id) = 1
            """
        )
        query_prefix, document_prefix = _prefixes_for(model)
        con.execute(INDEX_META_DDL)
        con.executemany(
            "insert into index_meta values (?, ?)",
            [
                ["embedding_model", model],
                ["embedding_dim", str(config.dim)],
                ["query_prefix", query_prefix],
                ["document_prefix", document_prefix],
            ],
        )
        con.execute("detach old")
        con.execute("checkpoint")

    os.replace(db_path, backup_path)
    os.replace(tmp_path, db_path)
    return backup_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Build or upgrade the mbox search index.")
    commands = parser.add_subparsers(dest="command", required=True)

    env_embeddings = os.getenv("INDEX_EMBEDDINGS", "on").lower()
    index_cmd = commands.add_parser("index", help="index new messages (resumable)")
    index_cmd.add_argument("--rebuild", action="store_true", help="drop existing tables first")
    index_cmd.add_argument("--model", help="Ollama embedding model (default: OLLAMA_MODEL or embeddinggemma)")
    index_cmd.add_argument("--batch-size", type=int, default=32, help="chunks per Ollama request")
    index_cmd.add_argument(
        "--embeddings",
        choices=["on", "off", "auto"],
        default={"1": "on", "true": "on", "0": "off", "false": "off"}.get(env_embeddings, env_embeddings),
        help="off: no Ollama needed, rag: search unavailable; auto: off if Ollama is unreachable"
        " (default: INDEX_EMBEDDINGS or on)",
    )
    index_cmd.add_argument(
        "--no-embeddings", dest="embeddings", action="store_const", const="off", help="same as --embeddings off"
    )
    index_cmd.add_argument(
        "--rebuild-if-stale",
        action="store_true",
        help="rebuild instead of failing when the mbox was replaced since the last run",
    )

    migrate_cmd = commands.add_parser("migrate", help="upgrade a database built by an older version")
    migrate_cmd.add_argument("--model", help="model the existing embeddings were built with")

    args = parser.parse_args()
    if args.command == "index" and args.embeddings not in ("on", "off", "auto"):
        parser.error(f"INDEX_EMBEDDINGS must be on, off or auto, not {args.embeddings!r}")
    if args.command == "index":
        print(
            build_index(
                rebuild=args.rebuild,
                model=args.model,
                batch_size=args.batch_size,
                embeddings={"on": True, "off": False}.get(args.embeddings, "auto"),
                rebuild_if_stale=args.rebuild_if_stale,
            )
        )
    else:
        print(f"Migrated; original kept at {migrate_db(model=args.model)}")
