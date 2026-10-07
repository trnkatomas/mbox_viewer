"""End-to-end tests: index a small mbox into DuckDB with a fake embedder, then search it."""

import hashlib
import math
from typing import List, Optional, Sequence

import duckdb
import pytest

import email_utils
from email_utils import (
    InvalidQueryError,
    MboxReader,
    RagUnavailableError,
    build_email_filter,
    build_index,
    chunk_text,
    get_email_count,
    get_email_list,
    get_embedding_config,
    get_one_thread,
    check_index_freshness,
    migrate_db,
)

MBOX = b"""From alice@example.com Mon Jan 01 10:00:00 2024
From: Alice <alice@example.com>
To: me@example.com
Subject: Flight booking to Prague
Date: Mon, 01 Jan 2024 10:00:00 +0100
Message-ID: <flight@example.com>
X-Gmail-Labels: Inbox,Travel
X-GM-THRID: 111

Your flight to Prague is confirmed. Boarding pass attached.
From here on, please check in online.

From me@example.com Tue Jan 02 11:00:00 2024
From: me@example.com
To: alice@example.com
Subject: Re: Flight booking to Prague
Date: Tue, 02 Jan 2024 11:00:00 +0000
Message-ID: <reply@example.com>
X-Gmail-Labels: Sent
X-GM-THRID: 111

Thanks, see you at the airport.

From bob@example.com Wed Jan 03 12:00:00 2024
From: Bob <bob@Example.com>
To: me@example.com
Subject: Invoice
Date: Wed, 03 Jan 2024 12:00:00 +0000
Message-ID: <invoice@example.com>
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="outer"

--outer
Content-Type: multipart/alternative; boundary="inner"

--inner
Content-Type: text/plain; charset="utf-8"
Content-Transfer-Encoding: 8bit

Faktura za leden \xc4\x8desky, platba do p\xc3\xa1tku.

--inner
Content-Type: text/html; charset="utf-8"

<html><body><p>Faktura</p></body></html>
--inner--

--outer
Content-Type: application/pdf; name="invoice.pdf"
Content-Disposition: attachment; filename="invoice.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQK
--outer--

"""


def _fake_vector(text: str, dim: int = 16) -> List[float]:
    vec = [0.0] * dim
    for word in text.lower().split():
        word = word.strip(".,:!?")
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def fake_ollama(monkeypatch):
    calls: List[int] = []

    def fake_embeddings(texts: Sequence[str], model: Optional[str] = None, **kwargs) -> List[List[float]]:
        calls.append(len(texts))
        return [_fake_vector(t) for t in texts]

    monkeypatch.setattr(email_utils, "get_ollama_embeddings", fake_embeddings)
    return calls


@pytest.fixture
def mbox_path(tmp_path, monkeypatch):
    path = tmp_path / "test.mbox"
    path.write_bytes(MBOX)
    monkeypatch.setenv("MBOX_FILE_PATH", str(path))
    return str(path)


@pytest.fixture
def indexed_db(tmp_path, mbox_path, fake_ollama):
    db_path = str(tmp_path / "emails.db")
    build_index(mbox_path, db_path, model="fake-model")
    con = duckdb.connect(db_path, read_only=True)
    yield con
    con.close()


class TestMboxReader:
    def test_from_at_start_of_body_line_does_not_split(self, mbox_path):
        with MboxReader(mbox_path) as reader:
            messages = list(reader)
        assert len(messages) == 3
        assert "From here on" in messages[0][0].get_content()

    def test_offsets_cover_file(self, mbox_path):
        with MboxReader(mbox_path) as reader:
            bounds = [b for _, b in reader]
        assert bounds[0][0] == 0
        assert all(prev[1] == nxt[0] for prev, nxt in zip(bounds, bounds[1:]))
        assert bounds[-1][1] == len(MBOX)

    def test_rejects_non_mbox(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_bytes(b"hello\n")
        with pytest.raises(ValueError):
            MboxReader(str(path))


class TestBuildIndex:
    def test_indexes_all_messages_with_text(self, indexed_db):
        rows = indexed_db.execute(
            "select message_id, excerpt, body_text, labels, has_attachment from emails order by date"
        ).fetchall()
        assert [r[0] for r in rows] == ["<flight@example.com>", "<reply@example.com>", "<invoice@example.com>"]
        invoice = rows[2]
        # nested multipart with UTF-8: decoded with diacritics, plain part preferred
        assert "česky" in invoice[2]
        assert "pátku" in invoice[1]
        assert invoice[4] == 1
        assert rows[0][3] == ["Inbox", "Travel"]

    def test_dates_normalised_to_utc(self, indexed_db):
        (value,) = indexed_db.execute(
            "select date from emails where message_id = '<flight@example.com>'"
        ).fetchone()
        assert value.hour == 9

    def test_offsets_are_bigint(self, indexed_db):
        types = dict(indexed_db.execute(
            "select column_name, data_type from information_schema.columns where table_name = 'emails'"
        ).fetchall())
        assert types["email_start"] == "BIGINT"

    def test_records_embedding_config(self, indexed_db):
        config = get_embedding_config(indexed_db)
        assert config.model == "fake-model"
        assert config.dim == 16

    def test_rerun_skips_existing(self, tmp_path, mbox_path, fake_ollama):
        db_path = str(tmp_path / "again.db")
        first = build_index(mbox_path, db_path, model="fake-model")
        second = build_index(mbox_path, db_path, model="fake-model")
        assert first["indexed"] == 3
        assert second["indexed"] == 0
        # nothing after the last indexed message, so the mbox is not re-read
        assert second["resumed_at"] == len(MBOX)
        assert second["skipped_existing"] == 0

    def test_embedding_failure_aborts_without_null_vectors(self, tmp_path, mbox_path, monkeypatch):
        calls = {"n": 0}

        def flaky(texts, model=None, **kwargs):
            calls["n"] += 1
            return [_fake_vector(t) for t in texts] if calls["n"] == 1 else None

        monkeypatch.setattr(email_utils, "get_ollama_embeddings", flaky)
        db_path = str(tmp_path / "flaky.db")
        with pytest.raises(RuntimeError, match="resume"):
            build_index(mbox_path, db_path, model="fake-model")
        with duckdb.connect(db_path, read_only=True) as con:
            assert con.execute("select count(*) from embeddings where vec is null").fetchone()[0] == 0

    def test_refuses_model_switch_without_rebuild(self, tmp_path, mbox_path, fake_ollama):
        db_path = str(tmp_path / "switch.db")
        build_index(mbox_path, db_path, model="fake-model")
        with pytest.raises(RuntimeError, match="rebuild"):
            build_index(mbox_path, db_path, model="other-model")


class TestIncrementalIndexing:
    NEW_MESSAGE = b"""From carol@example.com Thu Jan 04 09:00:00 2024
From: Carol <carol@example.com>
To: me@example.com
Subject: Lunch
Date: Thu, 04 Jan 2024 09:00:00 +0000
Message-ID: <lunch@example.com>

Lunch on Friday?

"""

    def test_appended_messages_are_indexed_from_the_old_end(self, tmp_path, mbox_path, fake_ollama):
        db_path = str(tmp_path / "grow.db")
        build_index(mbox_path, db_path, model="fake-model")
        with open(mbox_path, "ab") as f:
            f.write(self.NEW_MESSAGE)
        stats = build_index(mbox_path, db_path, model="fake-model")
        assert stats["resumed_at"] == len(MBOX)
        assert stats["indexed"] == 1
        assert stats["embedded"] == 1
        with duckdb.connect(db_path, read_only=True) as con:
            assert check_index_freshness(con, mbox_path) is None
            start, end = con.execute(
                "select email_start, email_end from emails where message_id = '<lunch@example.com>'"
            ).fetchone()
        assert (start, end) == (len(MBOX), len(MBOX) + len(self.NEW_MESSAGE))

    def test_metadata_only_then_backfill_embeddings(self, tmp_path, mbox_path, monkeypatch):
        def no_ollama(*args, **kwargs):
            raise AssertionError("Ollama must not be called with embeddings=False")

        monkeypatch.setattr(email_utils, "get_ollama_embeddings", no_ollama)
        db_path = str(tmp_path / "meta.db")
        stats = build_index(mbox_path, db_path, embeddings=False)
        assert stats["indexed"] == 3
        with duckdb.connect(db_path, read_only=True) as con:
            assert get_email_count(con) == 3
            with pytest.raises(RagUnavailableError, match="no embeddings"):
                email_utils.embed_query(con, "flight")

        monkeypatch.setattr(
            email_utils, "get_ollama_embeddings", lambda texts, **k: [_fake_vector(t) for t in texts]
        )
        stats = build_index(mbox_path, db_path, model="fake-model")
        assert stats["indexed"] == 0
        assert stats["embedded"] == 3
        with duckdb.connect(db_path, read_only=True) as con:
            vec = _fake_vector("flight booking prague confirmed")
            results = get_email_list(con, 10, 0, query_vec=vec)
            assert results["message_id"].iloc[0] == "<flight@example.com>"

    def test_missing_model_is_pulled(self, tmp_path, mbox_path, monkeypatch):
        pulled: List[str] = []

        def embeddings(texts, model=None, **kwargs):
            return [_fake_vector(t) for t in texts] if pulled else None

        monkeypatch.setattr(email_utils, "get_ollama_embeddings", embeddings)
        monkeypatch.setattr(email_utils, "pull_ollama_model", lambda model, **k: pulled.append(model) or True)
        build_index(mbox_path, str(tmp_path / "pull.db"), model="fake-model")
        assert pulled == ["fake-model"]

    def test_auto_mode_skips_embeddings_without_ollama(self, tmp_path, mbox_path, monkeypatch):
        monkeypatch.setattr(email_utils, "get_ollama_embeddings", lambda *a, **k: None)
        monkeypatch.setattr(email_utils, "pull_ollama_model", lambda *a, **k: False)
        db_path = str(tmp_path / "auto.db")
        stats = build_index(mbox_path, db_path, embeddings="auto")
        assert stats["indexed"] == 3
        assert stats["embedded"] == 0
        with pytest.raises(RuntimeError, match="Ollama"):
            build_index(mbox_path, str(tmp_path / "strict.db"), embeddings=True)

    def test_replaced_mbox_is_detected(self, tmp_path, mbox_path, fake_ollama):
        db_path = str(tmp_path / "stale.db")
        build_index(mbox_path, db_path, model="fake-model")
        # Same messages, different order: offsets no longer line up.
        messages = MBOX.split(b"\nFrom ")
        reordered = b"From " + b"\nFrom ".join([messages[1], messages[0][len(b"From "):]] + messages[2:])
        with open(mbox_path, "wb") as f:
            f.write(reordered)

        with duckdb.connect(db_path, read_only=True) as con:
            assert check_index_freshness(con, mbox_path) is not None
        with pytest.raises(RuntimeError, match="does not match"):
            build_index(mbox_path, db_path, model="fake-model")

        stats = build_index(mbox_path, db_path, model="fake-model", rebuild_if_stale=True)
        assert stats["indexed"] == 3
        with duckdb.connect(db_path, read_only=True) as con:
            assert check_index_freshness(con, mbox_path) is None

    def test_truncated_mbox_is_detected(self, tmp_path, mbox_path, fake_ollama):
        db_path = str(tmp_path / "short.db")
        build_index(mbox_path, db_path, model="fake-model")
        with open(mbox_path, "wb") as f:
            f.write(MBOX[: len(MBOX) // 2])
        with duckdb.connect(db_path, read_only=True) as con:
            assert "bytes" in (check_index_freshness(con, mbox_path) or "")


class TestSearch:
    def test_keyword_searches_full_body_case_insensitively(self, indexed_db):
        emails = get_email_list(indexed_db, 10, 0, {"excerpt": "AIRPORT"})
        assert emails["message_id"].tolist() == ["<reply@example.com>"]

    def test_sent_filter_applies_to_count(self, indexed_db):
        assert get_email_count(indexed_db, {}, sent=True) == 1
        assert get_email_count(indexed_db) == 3

    def test_label_and_keyword_combine(self, indexed_db):
        assert get_email_count(indexed_db, {"label": "Travel", "excerpt": "prague"}) == 1
        assert get_email_count(indexed_db, {"label": "Travel", "excerpt": "invoice"}) == 0

    def test_to_date_is_inclusive(self, indexed_db):
        assert get_email_count(indexed_db, {"to_date": "2024-01-02"}) == 2

    def test_invalid_date_is_reported(self, indexed_db):
        with pytest.raises(InvalidQueryError):
            get_email_count(indexed_db, {"from_date": "last week"})

    def test_semantic_search_ranks_by_distance(self, indexed_db):
        vec = _fake_vector("flight booking prague confirmed")
        results = get_email_list(indexed_db, 10, 0, query_vec=vec)
        assert results["message_id"].iloc[0] == "<flight@example.com>"
        assert results["dist"].is_monotonic_increasing
        assert get_email_count(indexed_db, query_vec=vec) == len(results)

    def test_semantic_search_combines_with_filters(self, indexed_db):
        vec = _fake_vector("flight booking prague")
        results = get_email_list(indexed_db, 10, 0, {"from": "me@"}, query_vec=vec)
        assert results["message_id"].tolist() == ["<reply@example.com>"]

    def test_thread_is_ordered_and_empty_id_is_not_a_thread(self, indexed_db):
        thread = get_one_thread(indexed_db, "111")
        assert thread["message_id"].tolist() == ["<flight@example.com>", "<reply@example.com>"]
        assert get_one_thread(indexed_db, "").empty


class TestRagFailure:
    def test_search_raises_instead_of_returning_everything(self, indexed_db, monkeypatch):
        from email_service import search_emails

        monkeypatch.setattr(email_utils, "get_ollama_embeddings", lambda *a, **k: None)
        with pytest.raises(RagUnavailableError):
            search_emails(indexed_db, "rag:flight", 1, 10)

    def test_dimension_mismatch_is_reported(self, indexed_db, monkeypatch):
        from email_service import search_emails

        monkeypatch.setattr(email_utils, "get_ollama_embeddings", lambda texts, **k: [[0.1] * 4 for _ in texts])
        with pytest.raises(RagUnavailableError, match="dimensions"):
            search_emails(indexed_db, "rag:flight", 1, 10)


class TestMigrate:
    def test_migrates_legacy_schema(self, tmp_path):
        db_path = str(tmp_path / "legacy.db")
        with duckdb.connect(db_path) as con:
            con.execute(
                "create table emails (i integer, line integer, subject text, excerpt text, message_id text,"
                " from_email text, to_email text, date datetime, has_attachment integer, labels text[],"
                " content_type text, mbox_file_id text, email_start integer, email_end integer, thread_id text)"
            )
            con.execute("create table embeddings (id integer, mbox_file_id text, message_id text, vec FLOAT[4])")
            con.execute(
                "insert into emails values (null, null, 's', 'e', '<a@b>', 'f', 't', '2024-01-01', 0,"
                " ['Inbox'], null, 'h', 0, 2000000000, '')"
            )
            con.execute("insert into embeddings values (null, 'h', '<a@b>', [1, 0, 0, 0])")
            con.execute("insert into embeddings values (null, 'h', '<c@d>', null)")

        backup = migrate_db(db_path, model="embeddinggemma")

        with duckdb.connect(db_path, read_only=True) as con:
            assert con.execute("select email_end from emails").fetchone()[0] == 2000000000
            types = dict(con.execute(
                "select column_name, data_type from information_schema.columns where table_name = 'emails'"
            ).fetchall())
            assert types["email_end"] == "BIGINT" and "body_text" in types and "i" not in types
            assert con.execute("select count(*) from embeddings").fetchone()[0] == 1
            config = get_embedding_config(con)
            assert (config.model, config.dim) == ("embeddinggemma", 4)
            assert config.query_prefix.startswith("task: search result")
            # keyword search falls back to the excerpt for migrated rows
            assert get_email_count(con, {"excerpt": "e"}) == 1
        assert backup.endswith(".bak")


class TestHelpers:
    def test_chunk_text_overlaps_and_caps(self):
        text = " ".join(f"word{i}" for i in range(2000))
        chunks = chunk_text(text, max_chars=500, overlap=50, max_chunks=5)
        assert len(chunks) == 5
        assert all(len(c) <= 500 for c in chunks)
        assert chunks[0].split()[-1] in chunks[1]

    def test_chunk_text_empty(self):
        assert chunk_text("   ") == []

    def test_filter_uses_parameters_not_interpolation(self):
        where, params = build_email_filter({"from": "x' or 1=1 --"})
        assert "x'" not in where
        assert params == ["%x' or 1=1 --%"]
