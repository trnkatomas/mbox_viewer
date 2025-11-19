import email
from email.message import Message
from email.policy import default
import os
from typing import Dict, List, Optional, Tuple, Union, Generator
from types import TracebackType

import numpy as np
import numpy.typing as npt
import pandas as pd
from readabilipy import simple_json_from_html_string
import email
from email.policy import default
from email.parser import BytesParser
import textwrap
import io
import logging  # New import for structured debugging and information

# Set up a logger for the module
logger = logging.getLogger(__name__)

import chromadb
import requests

import duckdb

EMAIL_DETAILS = [
    {
        "id": 2,
        "subject": "Upcoming Holiday Schedule",
        "sender": "HR Department <hr@corp.com>",
        "date": "Oct 14, 2025",
        "body": '<p class="text-gray-300">All Employees,</p><p class="mt-4 text-gray-300">The holiday schedule is finalized and available on the internal portal.</p>',
    },
    {
        "id": 3,
        "subject": "Quick Question Regarding the New Logo",
        "sender": "Mark Smith <mark.smith@design.io>",
        "date": "Oct 13, 2025",
        "body": '<p class="text-gray-300">Could you confirm the HEX code for the dark blue in the new logo design?</p>',
    },
]

# Default MBOX file path - can be overridden with MBOX_FILE_PATH environment variable
mboxfilename = os.getenv(
    "MBOX_FILE_PATH", "/Users/tomastrnka/Downloads/bigger_example.mbox"
)


class MboxReader:
    def __init__(self, filename: str) -> None:
        self.handle = open(filename, "rb")
        assert self.handle.readline().startswith(b"From ")
        # move the position in file back to zero
        self.handle.seek(0)

    def __enter__(self) -> "MboxReader":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        self.handle.close()

    def __iter__(self) -> Generator[Tuple[Message, Tuple[int, int]], None, None]:
        return iter(self.__next__())

    def __next__(self) -> Generator[Tuple[Message, Tuple[int, int]], None, None]:
        lines: List[bytes] = []
        line_counter = 0
        bytes_start_counter = 0
        bytes_end_counter = 0
        while True:
            line = self.handle.readline()
            if line == b"" or line.startswith(b"From ") and line_counter > 0:
                yield email.message_from_bytes(b"".join(lines), policy=default), (
                    bytes_start_counter,
                    bytes_end_counter,
                )
                bytes_start_counter = bytes_end_counter
                bytes_end_counter += len(line)
                if line == b"":
                    break
                lines = [line]
                continue
            lines.append(line)
            line_counter += 1
            bytes_end_counter += len(line)


def to_string(_content: Union[bytes, bytearray, str]) -> str:
    if isinstance(_content, (bytes, bytearray)):
        string_content = _content.decode("ascii", errors="ignore")
    else:
        string_content = _content
    return string_content


def load_email_db(db_name: str = "emails.db") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_name, read_only=True)
    return con


def get_one_email(db: duckdb.DuckDBPyConnection, email_id: str) -> pd.DataFrame:
    rel = db.execute("select * from emails where message_id == ? limit 1", [email_id])
    return rel.df()


def get_basic_stats(db: duckdb.DuckDBPyConnection) -> List[pd.DataFrame]:
    all_emails = db.execute(
        "select count(distinct message_id) as all_emails from emails limit 1"
    ).df()
    all_size = db.execute(
        "select avg(email_end - email_start) as avg_size from emails limit 1"
    ).df()
    all_timespan = db.execute(
        "select min(date) as first_seen, max(date) as last_seen from emails limit 1"
    ).df()
    return [all_emails, all_size, all_timespan]


def get_email_sizes_in_time(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    stats_query = """
    with raw_data as (
        select 1 as dummy, date_trunc('month', date) as mmonth, (email_end-email_start) as size from emails
    ),
    monthly_sizes as (
        select mmonth, sum(size) as sizes from raw_data group by mmonth
    )
    select 1 as dummy, mmonth as date, sum(sizes) over (partition by dummy order by mmonth) as count from monthly_sizes
    """
    results = db.execute(stats_query).df()
    return results


def get_domains_by_count(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Get email count by sender domain, limited to top domains."""
    stats_query = """
    with domain_counts as (
        select
            regexp_extract(from_email, '@([^>]+)') as domain,
            count(*) as count
        from emails
        where from_email is not null
        group by domain
        order by count desc
        limit 10
    )
    select domain, count from domain_counts
    """
    results = db.execute(stats_query).df()
    return results


def get_thread_for_email(db: duckdb.DuckDBPyConnection, email_id: str) -> pd.DataFrame:
    email_from_db = get_one_email(db, email_id)
    if not email_from_db.empty:
        email_dict = email_from_db.to_dict(orient="records")[0]
        thread_id_val = email_dict.get("thread_id")
        if isinstance(thread_id_val, str):
            return get_one_thread(db, thread_id_val)
    return pd.DataFrame()


def get_one_thread(db: duckdb.DuckDBPyConnection, thread_id: str) -> pd.DataFrame:
    rel = db.execute("select * from emails where thread_id == ?", [thread_id])
    return rel.df()


def get_email_count(db: duckdb.DuckDBPyConnection, additional_criteria: Optional[Dict[str, str]] = None) -> int:
    if additional_criteria:
        additional_conditions, where_statements = process_additional_criteria(additional_criteria)
        if where_statements:
            where_statement = " AND ".join(where_statements)
            rel = db.execute(
                f"with conditional_selection as (select * from emails where {where_statement} order by date desc)"
                f"select count(distinct message_id) as email_count from conditional_selection",
                additional_conditions
            )
        else:
            rel = db.execute("select count(distinct message_id) as email_count from emails")
    else:
        rel = db.execute("select count(distinct message_id) as email_count from emails")
    df = rel.df()
    if not df.empty:
        return int(df.email_count.values[0])
    else:
        return 0


def get_attachment_file(db: duckdb.DuckDBPyConnection, email_id: str, attachment_name: str) -> Dict[str, Union[str, bytes, int]]:
    email_data_list = get_one_email(db, email_id=email_id).to_dict(orient="records")
    if isinstance(email_data_list, list) and email_data_list:
        email_data = email_data_list[0]
        email_raw_string = get_string_email_from_mboxfile(
            email_data.get("email_start"), email_data.get("email_end")  # type: ignore[arg-type]
        )
        attachments = parse_email(email_raw_string).get("attachments")
        for a in attachments:  # type: ignore[union-attr]
            if attachment_name == a.get("filename"):  # type: ignore[union-attr]
                return a  # type: ignore[return-value]
    return {}


def _extract_attachments(msg: Message) -> List[Dict[str, Union[str, bytes, int]]]:
    """
    Internal function to iterate and extract all attachment parts.
    Returns the full decoded binary content in the 'content' field.
    """
    attachments: List[Dict[str, Union[str, bytes, int]]] = []

    # Iterate through all parts of the email
    for i, part in enumerate(msg.walk()):
        # Skip container parts
        if part.is_multipart():
            continue

        content_type = part.get_content_type()
        filename = part.get_filename()
        content_disposition = part.get("Content-Disposition")

        # An attachment is typically identified by a filename OR a Content-Disposition: attachment
        is_attachment = filename or (
            content_disposition and content_disposition.startswith("attachment")
        )

        if is_attachment:
            logger.debug(
                f"[{i:02d}] Identified attachment: {filename} ({content_type})"
            )

            try:
                # get_payload(decode=True) extracts the raw, decoded binary content
                payload_raw = part.get_payload(decode=True)
                if payload_raw is None:
                    payload = b"[Empty attachment content]"
                elif isinstance(payload_raw, bytes):
                    payload = payload_raw
                else:
                    payload = str(payload_raw).encode("utf-8")
            except Exception as e:
                logger.error(f"Error decoding attachment content for {filename}: {e}")
                payload = f"[Error decoding attachment content: {e}]".encode("utf-8")

            # Truncate content for a clean preview (for logging/summary)
            content_preview = payload[:50]

            attachments.append(
                {
                    "filename": filename or "Untitled",
                    "content_type": content_type,
                    "size_bytes": len(payload),
                    "content": payload,  # Returns the full binary content
                    "content_preview": content_preview,  # A small preview for easy viewing
                }
            )

    return attachments


def _extract_body_content(msg: Message) -> Tuple[str, Optional[str]]:
    """
    Internal function to extract and prioritize HTML or plain text body.
    Priority: HTML > Plain Text.
    Returns: A tuple (body_type, body_content_string)
    """
    email_body_html: Optional[str] = None
    email_body_text: Optional[str] = None

    for i, part in enumerate(msg.walk()):
        # Skip container parts or parts clearly identified as attachments
        content_disposition = part.get("Content-Disposition")
        if (
            part.is_multipart()
            or part.get_filename()
            or (
                content_disposition is not None
                and content_disposition.startswith("attachment")
            )
        ):
            continue

        content_type = part.get_content_type()
        main_type = part.get_content_maintype()

        is_body_part = main_type == "text"

        if is_body_part:
            try:
                # .get_content() automatically decodes the payload into a string
                content_raw = part.get_content()  # type: ignore[attr-defined]
                if isinstance(content_raw, str):
                    content = content_raw
                else:
                    content = str(content_raw)
            except Exception as e:
                logger.error(
                    f"Error decoding body content (Part {i:02d}, Type {content_type}): {e}"
                )
                content = f"[Error decoding body content: {e}]"

            if content_type == "text/html":
                email_body_html = content
                logger.debug(f"[{i:02d}] Stored HTML Body.")
            elif content_type == "text/plain":
                # Only store if not already set, in case of multiple plain parts
                if email_body_text is None:
                    email_body_text = content
                logger.debug(f"[{i:02d}] Stored Plain Text Body.")

    # 5. Determine which body to show (HTML first, then Plain Text)
    if email_body_html:
        logger.info("Prioritizing HTML body content.")
        return ("HTML", email_body_html)
    elif email_body_text:
        logger.info("Falling back to Plain Text body content.")
        return ("Plain Text", email_body_text)
    else:
        logger.info("No recognizable body content found.")
        return ("None", None)


# --- Main Library Function ---


def parse_email(
    raw_email: bytes,
) -> Dict[str, Union[Tuple[str, Optional[str]], List[Dict[str, Union[str, bytes, int]]]]]:
    """
    Parses a raw email string to extract prioritized body content and all attachments.

    Args:
        raw_email_string: The complete raw content of an email message.

    Returns:
        A dictionary containing:
        - 'body': A tuple (body_type, body_content_string or None)
        - 'attachments': A list of attachment dictionaries.
    """

    logger.info("Starting email parsing process...")

    # 1. Convert string to bytes and parse the email message
    try:
        msg = BytesParser(policy=default).parsebytes(raw_email)
    except Exception as e:
        logger.error(f"Failed to parse raw email string: {e}")
        return {"body": ("Error", f"Parsing failed: {e}"), "attachments": []}

    # 2. Extract content using dedicated functions
    body_info = _extract_body_content(msg)
    attachments_list = _extract_attachments(msg)

    logger.info(f"Finished parsing. Found {len(attachments_list)} attachments.")

    return {"body": body_info, "attachments": attachments_list}


def get_string_email_from_mboxfile(email_start: int, email_end: int) -> bytes:
    with open(mboxfilename, "rb") as infile:
        infile.seek(email_start)
        data = infile.read(email_end - email_start)
        return data


def get_email_content(email_start: int, email_end: int) -> str:
    with open(mboxfilename, "rb") as infile:
        infile.seek(email_start)
        data = infile.read(email_end - email_start)
        parsed_email = email.message_from_bytes(data, policy=default)
        email_content = _extract_body_content(parsed_email)
        email_attachments = _extract_attachments(parsed_email)

        content = ""
        try:
            if len(list(parsed_email.iter_parts())) > 0:
                for part in parsed_email.iter_parts():
                    disposition = part.get("Content-Disposition")
                    if disposition and "attachment" in disposition:
                        continue
                    current_content = to_string(part.get_content())
                    content += current_content
            else:
                content = to_string(parsed_email.get_content())
        except Exception as e:
            print(e)
        for a in parsed_email.iter_attachments():
            # TODO return attachments, generate new email block for getting the attachments from the source file
            print(f"attachment: {a.get_filename()}")
        return content


def surround_with_wildcards(input: str) -> str:
    return f"%{input}%"


def get_similar_vectors(
    db: duckdb.DuckDBPyConnection, vec: npt.NDArray[np.float64]
) -> pd.DataFrame:
    rel = db.execute("""SELECT *, array_distance(vec, ?::FLOAT[768]) as dist
    FROM
    embeddings
    ORDER
    BY
    dist
    LIMIT
    20;""",
        [vec],
    )
    return rel.df()


def process_additional_criteria(additional_criteria: Optional[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    additional_conditions = []
    where_statements = []
    if additional_criteria:
        # Email address filter
        if "from" in additional_criteria:
            where_statements.append("from_email like ?")
            additional_conditions.append(
                surround_with_wildcards(additional_criteria["from"])
            )
        if "subject" in additional_criteria:
            where_statements.append("subject like ?")
            additional_conditions.append(
                surround_with_wildcards(additional_criteria["subject"])
            )
        if "label" in additional_criteria:
            where_statements.append("? in labels")
            additional_conditions.append(additional_criteria["label"])
        else:
            if excerpt := additional_criteria.get("excerpt"):
                where_statements.append("excerpt like ?")
                additional_conditions.append(surround_with_wildcards(excerpt))

        # Date range filters
        if (
                "from_date" in additional_criteria
                and additional_criteria["from_date"]
        ):
            where_statements.append("date >= ?")
            additional_conditions.append(additional_criteria["from_date"])

        if "to_date" in additional_criteria and additional_criteria["to_date"]:
            where_statements.append("date <= ?")
            additional_conditions.append(additional_criteria["to_date"])
    return additional_conditions, where_statements


def get_email_list(
    db: duckdb.DuckDBPyConnection, criteria: Optional[Dict[str, int]] = None, additional_criteria: Optional[Dict[str, str]] = None, sent: bool = False, rag_message_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get email list with optional filtering.

    Args:
        db: DuckDB connection
        criteria: Dict with 'limit' and 'offset' for pagination
        additional_criteria: Dict with search filters (from, subject, excerpt, from_date, to_date)
        sent: Boolean to filter for sent emails
        rag_message_ids: List of message IDs from RAG search to filter by
    """
    if not criteria:
        rel = db.sql("select * from emails order by date desc limit 30")
        return rel.df()
    else:
        if "limit" in criteria and "offset" in criteria:
            additional_conditions, where_statements = process_additional_criteria(additional_criteria)
            # Sent folder filter
            if sent:
                where_statements.append("? IN labels")
                additional_conditions.append("Sent")

            # RAG search results filter
            if rag_message_ids:
                # Create placeholders for the IN clause
                placeholders = ",".join(["?" for _ in rag_message_ids])
                where_statements.append(f"message_id IN ({placeholders})")
                additional_conditions.extend(rag_message_ids)

            # Execute query
            if where_statements:
                where_statement = " AND ".join(where_statements)
                db.execute(
                    f"select * from emails where {where_statement} order by date desc limit ? offset ?",
                    additional_conditions + [criteria["limit"], criteria["offset"]],
                )
            else:
                db.execute(
                    "select * from emails order by date desc limit ? offset ?",
                    [criteria["limit"], criteria["offset"]],
                )
        else:
            db.execute("select * from emails order by date desc limit 1")
        return db.df()


def load_email_content_search(db: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """
    Initialize DuckDB for vector search by installing and loading VSS extension.

    Args:
        db: DuckDB connection

    Returns:
        The same db connection with VSS loaded
    """
    try:
        # Install and load the VSS extension for vector similarity search
        db.execute("INSTALL vss;")
        db.execute("LOAD vss;")
        logger.info("DuckDB VSS extension loaded successfully")
    except Exception as e:
        logger.warning(f"VSS extension may already be installed: {e}")

    return db


def get_ollama_embedding(text: str, server_url: Optional[str] = None, model: Optional[str] = None) -> Optional[List[float]]:
    """
    Get embedding vector from Ollama server.

    Args:
        text: Text to embed
        server_url: Ollama API endpoint (defaults to OLLAMA_URL env var or http://localhost:11434/api/embed)
        model: Embedding model to use (defaults to OLLAMA_MODEL env var or nomic-embed-text)

    Returns:
        List of floats representing the embedding vector, or None on failure
    """
    import requests

    # Get configuration from environment variables if not provided
    if server_url is None:
        server_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embed")
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "embeddinggemma")

    try:
        response = requests.post(
            server_url, json={"model": model, "input": text}, timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            # Ollama returns embeddings in different formats depending on version
            if "embeddings" in result:
                emb = result["embeddings"][0] if isinstance(result["embeddings"], list) else result["embeddings"]
                return emb  # type: ignore[no-any-return]
            elif "embedding" in result:
                return result["embedding"]  # type: ignore[no-any-return]
            else:
                logger.error(f"Unexpected Ollama response format: {result.keys()}")
                return None
        else:
            logger.error(f"Ollama embedding failed with status {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Failed to get embedding from Ollama: {e}")
        return None


def rag_search_duckdb(db: duckdb.DuckDBPyConnection, query_text: str, n_results: int = 50) -> pd.DataFrame:
    """
    Perform semantic search using DuckDB's VSS extension with cosine distance.

    Args:
        db: DuckDB connection with VSS loaded
        query_text: The search query text
        n_results: Number of results to return (default 50)

    Returns:
        List of message IDs from semantically similar emails
    """
    try:
        if not query_text:
            return pd.DataFrame()

        # Get embedding for the query (uses OLLAMA_URL env var)
        query_vec = get_ollama_embedding(query_prefix + query_text)

        if not query_vec:
            logger.warning("Failed to generate embedding for RAG search")
            return pd.DataFrame()

        # Perform vector similarity search using array_cosine_distance
        # Lower distance = more similar (0 = identical, 2 = opposite)
        rel = db.execute(
            """
            with dists as (
                SELECT message_id, array_cosine_distance(vec, ?::FLOAT[768]) as dist
                FROM embeddings
                ORDER BY dist ASC
                LIMIT ?
            )
            select emails.message_id, subject, dists.dist
            from emails join
            dists on dists.message_id == emails.message_id 
            where dist < 0.5
            order by dist asc
        """,
            [query_vec, n_results],
        )

        results = rel.df()
        logger.info(
            f"RAG search for '{query_text}' returned {results.shape[0]} results"
        )

        return results

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return pd.DataFrame()


def process(drop_previous_table: bool = False) -> None:
    # imports
    from hashlib import sha256

    import chromadb
    import duckdb
    import numpy as np
    import tqdm
    from dateparser import parse

    from sentence_transformers import CrossEncoder

    # DBs initialization
    # chroma_client = chromadb.Client()
    chroma_client = chromadb.PersistentClient(path="emails.chromadb")
    emails_collection = chroma_client.get_or_create_collection(name="emails")

    sha = sha256()
    BUF_SIZE = 65536
    with open(mboxfilename, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha.update(data)
    mbox_file_hash = sha.hexdigest()
    print("SHA256: {0}".format(mbox_file_hash))

    con = duckdb.connect("emails.db")
    if drop_previous_table:
        con.sql("drop table if exists emails")
        con.sql("drop table if exists embeddings")
        con.sql(
            "create table embeddings"
            " (id integer,"
            "  mbox_file_id text,"
            "  message_id text,"
            "  vec FLOAT[768])"
        )
        con.sql(
            "create table emails "
            "(i integer,"
            " line integer,"
            " subject text,"
            " excerpt text,"
            " message_id text,"
            " from_email text,"
            " to_email text,"
            " date datetime,"
            " has_attachment integer,"
            " labels text[],"
            " content_type text,"
            " mbox_file_id text,"
            " email_start integer,"
            " email_end integer,"
            " thread_id text)"
        )
        con.close()

    with duckdb.connect("emails.db") as con:
        with MboxReader(mboxfilename) as mbox:
            for message, boundaries in tqdm.tqdm(mbox):
                # print(message['From'], message['To'], message['Subject'], message['Date'], len(list(message.iter_parts())), len(list(message.iter_attachments())))
                content = ""
                whole_text = ""
                try:
                    if len(list(message.iter_parts())) > 0:
                        for part in message.iter_parts():
                            current_content = to_string(part.get_content())
                            content += current_content
                    else:
                        content = to_string(message.get_content())
                    parsed = simple_json_from_html_string(content)
                    whole_text = "\n".join([x["text"] for x in parsed["plain_text"]])
                except:
                    pass
                if not message["Message-ID"]:
                    continue
                emails_collection.add(
                    ids=[message["Message-ID"]], documents=[whole_text]
                )
                # the content should be chunked into ~ 500 tokens
                d_encoded = ollama_embeddings(
                    document_prefix + whole_text,
                    "http://localhost:11434/api/embed",
                    "embeddinggemma",
                )
                vector_to_insert = [message["Message-ID"], mbox_file_hash, d_encoded]
                con.execute(
                    f"""insert into embeddings 
                                (message_id, mbox_file_id, vec) values (?, ?, ?)""",
                    vector_to_insert,
                )
                data_to_insert = [
                    message["From"],
                    message["To"],
                    message["Subject"],
                    parse(message["Date"]),
                    message["Message-ID"],
                    len(list(message.iter_attachments())),
                    whole_text[:30],
                    message.get("X-Gmail-Labels", "").split(","),
                    message.get("Content-Type"),
                    mbox_file_hash,
                    boundaries[0],
                    boundaries[1],
                    message.get("X-GM-THRID", ""),
                ]
                con.execute(
                    f"""insert into emails 
                (from_email,
                 to_email,
                 subject,
                 date, 
                 message_id, 
                 has_attachment, 
                 excerpt,
                 labels,
                 content_type,
                 mbox_file_id,
                 email_start,
                 email_end,
                 thread_id
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    data_to_insert[:13],
                )

        # check that the DB has been filled in
        res = con.sql(
            "select subject, sum(has_attachment>0), count(*) as cnt from emails group by subject order by cnt desc limit 3"
        )

        con.execute("INSTALL vss; LOAD vss")
        con.execute("SET hnsw_enable_experimental_persistence = TRUE;")
        con.execute(
            "CREATE INDEX cosine_idx ON embeddings USING HNSW (vec) WITH (metric = 'cosine')"
        )

    print(res.df())

    def query_collection(collection: chromadb.Collection, query: str) -> Dict[str, List[List[str]]]:
        # Query the results
        query = "vyrizena objednavka"
        results = collection.query(query_texts=[query], n_results=2)
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        documents = results.get("documents")
        if documents and len(documents) > 0 and documents[0]:
            scores = model.predict([(query, doc) for doc in documents[0]])
            print(documents[0][np.argmax(scores)])
        return results  # type: ignore[return-value]


query_prefix = "task: search result | query: "
document_prefix = "title: none | text: "


def ollama_embeddings(text: str, server_url: str, model: str) -> npt.NDArray[np.float64]:
    response = requests.post(server_url, json={"model": model, "input": text})
    if response.status_code == 200:
        json_data = response.json()
        return np.squeeze(np.array(json_data["embeddings"]))
    else:
        return np.array([])


if __name__ == "__main__":
    import argparse

    arguments = argparse.ArgumentParser()

    arguments.add_argument("delete_table", action="store_true")

    parsed_args = arguments.parse_args()

    # process(parsed_args.delete_table)
    # be careful the database creation took an hour and a half
    process(drop_previous_table=False)


"""
    import requests

    a = ollama_embeddings(
        "what the heck is going on",
        "http://localhost:11434/api/embed",
        "embeddinggemma",
    )

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    def get_distances(documents, queries):

        q_encoded = [
            ollama_embeddings(
                query_prefix + q, "http://localhost:11434/api/embed", "embeddinggemma"
            )
            for q in queries
        ]
        d_encoded = [
            ollama_embeddings(
                document_prefix + d,
                "http://localhost:11434/api/embed",
                "embeddinggemma",
            )
            for d in documents
        ]

        d_emb = np.vstack([np.array(d["embeddings"]) for d in d_encoded])
        q_emb = np.vstack([np.array(q["embeddings"]) for q in q_encoded])

        print(q_emb.shape, d_emb.shape)
        return cosine_similarity(q_emb, d_emb)

    # supposed output array([[0.30018332, 0.63578754, 0.49253921, 0.48859104]])
"""
