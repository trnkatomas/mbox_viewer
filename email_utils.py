import email
from email.policy import default
from readabilipy import simple_json_from_html_string
import email
from email.policy import default
from email.parser import BytesParser
import textwrap
import io
import logging # New import for structured debugging and information

# Set up a logger for the module
logger = logging.getLogger(__name__)

import chromadb
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

# mboxfilename = "/Users/tomastrnka/Downloads/email_sample.mbox"
mboxfilename = '/Users/tomastrnka/Downloads/bigger_example.mbox'


class MboxReader:
    def __init__(self, filename):
        self.handle = open(filename, "rb")
        assert self.handle.readline().startswith(b"From ")
        # move the position in file back to zero
        self.handle.seek(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.handle.close()

    def __iter__(self):
        return iter(self.__next__())

    def __next__(self):
        lines = []
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


def to_string(_content):
    if isinstance(_content, (bytes, bytearray)):
        string_content = _content.decode("ascii", errors="ignore")
    else:
        string_content = _content
    return string_content


def load_email_db(db_name="emails.db"):
    con = duckdb.connect(db_name, read_only=True)
    return con


def get_one_email(db, email_id):
    rel = db.execute("select * from emails where message_id == ? limit 1", [email_id])
    return rel.df()


def get_email_count(db):
    rel = db.execute("select count(distinct message_id) as email_count from emails")
    df = rel.df()
    if not df.empty:
        return df.email_count.values[0]
    else:
        return 0


def _extract_attachments(msg):
    """
    Internal function to iterate and extract all attachment parts.
    Returns the full decoded binary content in the 'content' field.
    """
    attachments = []

    # Iterate through all parts of the email
    for i, part in enumerate(msg.walk()):
        # Skip container parts
        if part.is_multipart():
            continue

        content_type = part.get_content_type()
        filename = part.get_filename()
        content_disposition = part.get('Content-Disposition')

        # An attachment is typically identified by a filename OR a Content-Disposition: attachment
        is_attachment = filename or (content_disposition and content_disposition.startswith('attachment'))

        if is_attachment:
            logger.debug(f"[{i:02d}] Identified attachment: {filename} ({content_type})")

            try:
                # get_payload(decode=True) extracts the raw, decoded binary content
                payload = part.get_payload(decode=True)
            except Exception as e:
                logger.error(f"Error decoding attachment content for {filename}: {e}")
                payload = f"[Error decoding attachment content: {e}]".encode('utf-8')

            # Truncate content for a clean preview (for logging/summary)
            content_preview = payload[:50]

            attachments.append({
                'filename': filename or 'Untitled',
                'content_type': content_type,
                'size_bytes': len(payload),
                'content': payload,  # Returns the full binary content
                'content_preview': content_preview  # A small preview for easy viewing
            })

    return attachments


def _extract_body_content(msg):
    """
    Internal function to extract and prioritize HTML or plain text body.
    Priority: HTML > Plain Text.
    Returns: A tuple (body_type, body_content_string)
    """
    email_body_html = None
    email_body_text = None

    for i, part in enumerate(msg.walk()):
        # Skip container parts or parts clearly identified as attachments
        if part.is_multipart() or part.get_filename() or (
                part.get('Content-Disposition') and part.get('Content-Disposition').startswith('attachment')):
            continue

        content_type = part.get_content_type()
        main_type = part.get_content_maintype()

        is_body_part = main_type == 'text'

        if is_body_part:
            try:
                # .get_content() automatically decodes the payload into a string
                content = part.get_content()
            except Exception as e:
                logger.error(f"Error decoding body content (Part {i:02d}, Type {content_type}): {e}")
                content = f"[Error decoding body content: {e}]"

            if content_type == 'text/html':
                email_body_html = content
                logger.debug(f"[{i:02d}] Stored HTML Body.")
            elif content_type == 'text/plain':
                # Only store if not already set, in case of multiple plain parts
                if email_body_text is None:
                    email_body_text = content
                logger.debug(f"[{i:02d}] Stored Plain Text Body.")

    # 5. Determine which body to show (HTML first, then Plain Text)
    if email_body_html:
        logger.info("Prioritizing HTML body content.")
        return ('HTML', email_body_html)
    elif email_body_text:
        logger.info("Falling back to Plain Text body content.")
        return ('Plain Text', email_body_text)
    else:
        logger.info("No recognizable body content found.")
        return ('None', None)


# --- Main Library Function ---

def parse_email(raw_email_string: str):
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
        msg = BytesParser(policy=default).parsebytes(raw_email_string.encode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to parse raw email string: {e}")
        return {'body': ('Error', f"Parsing failed: {e}"), 'attachments': []}

    # 2. Extract content using dedicated functions
    body_info = _extract_body_content(msg)
    attachments_list = _extract_attachments(msg)

    logger.info(f"Finished parsing. Found {len(attachments_list)} attachments.")

    return {
        'body': body_info,
        'attachments': attachments_list
    }


def get_email_content(email_start, email_end):
    with open(mboxfilename, 'rb') as infile:
        infile.seek(email_start)
        data = infile.read(email_end - email_start)
        parsed_email = email.message_from_bytes(data, policy=default)
        email_content = _extract_body_content(parsed_email)
        email_attachments = _extract_attachments(parsed_email)

        content = ''
        try:
            if len(list(parsed_email.iter_parts())) > 0:
                for part in parsed_email.iter_parts():
                    if is_attachment := part.get('Content-Disposition') and 'attachment' in is_attachment:
                        continue
                    current_content = to_string(part.get_content())
                    content += current_content
            else:
                content = to_string(parsed_email.get_content())
        except Exception as e:
            print(e)
        for a in parsed_email.iter_attachments():
            # TODO return attachments, generate new email block for getting the attachments from the source file
            print(f'attachment: {a.get_filename()}')
        return content


def get_email_list(db, criteria=None):
    if not criteria:
        rel = db.sql("select * from emails order by date desc limit 30")
        # return db.fetchall()
        return rel.df()
    else:
        if "limit" in criteria and "offset" in criteria:
            db.execute(
                "select * from emails order by date desc limit ? offset ?",
                [criteria["limit"], criteria["offset"]],
            )
        else:
            db.execute("select * from emails order by date desc limit 1")
        return db.df()


def load_email_content_search(email_embeddings_name="emails.chromadb"):
    chroma_client = chromadb.PersistentClient(path="emails.chromadb")
    emails_collection = chroma_client.get_or_create_collection(name="emails")
    return emails_collection


def process(drop_previous_table=False):
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
        " email_line_start integer,"
        " email_line_end integer,"
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
                 email_line_start,
                 email_line_end,
                 thread_id
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    data_to_insert[:13],
                )

        # check that the DB has been filled in
        res = con.sql(
            "select subject, sum(has_attachment>0), count(*) as cnt from emails group by subject order by cnt desc limit 3"
        )
        print(res.df())

    def query_collection(collection, query):
        # Query the results
        query = "vyrizena objednavka"
        results = collection.query(query_texts=[query], n_results=2)
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        scores = model.predict([(query, doc) for doc in results["documents"][0]])
        print(results["documents"][0][np.argmax(scores)])
        return results


if __name__ == "__main__":
    import argparse

    arguments = argparse.ArgumentParser()

    arguments.add_argument("delete_table", action="store_true")

    parsed_args = arguments.parse_args()

    process(parsed_args.delete_table)
