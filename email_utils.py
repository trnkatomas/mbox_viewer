
import email
from email.policy import default
import duckdb
import chromadb

EMAIL_DETAILS = [
    {
        "id": 1,
        "subject": "Project Status Update: Q4 Goals",
        "sender": "Jane Doe <jane.doe@corp.com>",
        "date": "Oct 15, 2025",
        "body": """
            <p class="text-gray-300">Hi Team,</p>
            <p class="mt-4 text-gray-300">I wanted to provide a quick update on our Q4 goals. The new design system migration is 90% complete, and we are on track to launch the new marketing site by Friday.</p>
            <p class="mt-4 text-gray-300">Please review the attached mockups (simulated) and provide feedback by end-of-day tomorrow. We can discuss any blockers during our stand-up.</p>
            <p class="mt-8 text-gray-300">Best,<br>Jane</p>
        """
    },
    {
        "id": 2, "subject": "Upcoming Holiday Schedule", "sender": "HR Department <hr@corp.com>", "date": "Oct 14, 2025",
        "body": "<p class=\"text-gray-300\">All Employees,</p><p class=\"mt-4 text-gray-300\">The holiday schedule is finalized and available on the internal portal.</p>"
    },
    {
        "id": 3, "subject": "Quick Question Regarding the New Logo", "sender": "Mark Smith <mark.smith@design.io>", "date": "Oct 13, 2025",
        "body": "<p class=\"text-gray-300\">Could you confirm the HEX code for the dark blue in the new logo design?</p>"
    },
    {
        "id": 4, "subject": "Reminder: PTO Request Deadline", "sender": "Jenny Wilson <jenny@corp.com>", "date": "Oct 12, 2025",
        "body": "<p class=\"text-gray-300\">Just a friendly reminder that all Paid Time Off (PTO) requests for the first quarter must be submitted by the end of this week.</p>"
    },
    {
        "id": 5, "subject": "Meeting Confirmation: Q1 Budget Review", "sender": "Finance Team <finance@corp.com>", "date": "Oct 12, 2025",
        "body": "<p class=\"text-gray-300\">This is a confirmation for our Q1 Budget Review meeting scheduled for 10/25 at 2:00 PM.</p>"
    },
    {
        "id": 6, "subject": "Your recent purchase has been shipped", "sender": "Shipping <no-reply@shop.com>", "date": "Oct 10, 2025",
        "body": "<p class=\"text-gray-300\">Great news! Your order #90210 has been shipped and is expected to arrive within 3-5 business days.</p>"
    },
    {
        "id": 7, "subject": "Weekly Dev Sync Agenda", "sender": "Project Manager <pm@corp.com>", "date": "Oct 9, 2025",
        "body": "<p class=\"text-gray-300\">Please review the agenda for our weekly sync.</p>"
    },
    {
        "id": 8, "subject": "New Policy Update: Remote Work", "sender": "HR Department <hr@corp.com>", "date": "Oct 9, 2025",
        "body": "<p class=\"text-gray-300\">Our remote work policy has been updated, allowing for two days per week remote.</p>"
    },
    {
        "id": 9, "subject": "Lunch options for tomorrow", "sender": "Michael <michael@corp.com>", "date": "Oct 8, 2025",
        "body": "<p class=\"text-gray-300\">Hey, thinking of ordering Italian tomorrow. Let me know if you want to join the group order!</p>"
    },
    {
        "id": 10, "subject": "Your monthly service bill is ready", "sender": "Billing <billing@service.com>", "date": "Oct 8, 2025",
        "body": "<p class=\"text-gray-300\">Your bill for the month of October is now available. Log in to your account to view the details.</p>"
    },
]

class MboxReader:
    def __init__(self, filename):
        self.handle = open(filename, 'rb')
        assert self.handle.readline().startswith(b'From ')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.handle.close()

    def __iter__(self):
        return iter(self.__next__())

    def __next__(self):
        lines = []
        while True:
            line = self.handle.readline()
            if line == b'' or line.startswith(b'From '):
                yield email.message_from_bytes(b''.join(lines), policy=default)
                if line == b'':
                    break
                lines = []
                continue
            lines.append(line)


def load_email_db(db_name='emails.db'):
    con = duckdb.connect(db_name, read_only=True)
    return con


def get_one_email(db, email_id):
    rel = db.execute('select * from emails where message_id == ? limit 1', [email_id])
    return rel.df()


def get_email_list(db, criteria=None):
    if not criteria:
        rel = db.sql('select * from emails limit 30')
        #return db.fetchall()
        return rel.df()
    else:
        db.execute('select * from emails limit 1')
        return db.fetchall()


def load_email_content_search(email_embeddings_name='emails.chromadb'):
    chroma_client = chromadb.PersistentClient(path="emails.chromadb")
    emails_collection = chroma_client.get_or_create_collection(name="emails")
    return emails_collection

def process():
    # imports 
    import numpy as np
    import chromadb
    from sentence_transformers import CrossEncoder
    import tqdm
    import duckdb
    from dateparser import parse
    from readabilipy import simple_json_from_html_string

    # DBs initialization
    #chroma_client = chromadb.Client()
    chroma_client = chromadb.PersistentClient(path="emails.chromadb")
    emails_collection = chroma_client.get_or_create_collection(name="emails")
    mboxfilename = '/Users/tomastrnka/Downloads/email_sample.mbox'
    mboxfilename = '/Users/tomastrnka/Downloads/bigger_example.mbox'

    con = duckdb.connect("emails.db")
    con.sql('create table emails (i integer, line integer, subject text, excerpt text, message_id text, from_email text, to_email text, date datetime, has_attachment bool)')
    con.close()


    def to_string(content):
        if isinstance(content, (bytes, bytearray)):
            bcontent = content.decode('ascii', errors='ignore')
        else:
            bcontent = content
        return bcontent

    with duckdb.connect("emails.db") as con:
        with MboxReader(mboxfilename) as mbox:
            for message in tqdm.tqdm(mbox):
                #print(message['From'], message['To'], message['Subject'], message['Date'], len(list(message.iter_parts())), len(list(message.iter_attachments())))
                content = ''
                whole_text = ''
                try:
                    if len(list(message.iter_parts())) > 0:
                        for part in message.iter_parts():
                            current_content = to_string(part.get_content())
                            content += current_content
                    else:
                        content = to_string(message.get_content())            
                    parsed = simple_json_from_html_string(content)
                    whole_text = '\n'.join([x['text'] for x in parsed['plain_text']])
                except:
                    pass
                if not message['Message-ID']:
                    continue
                emails_collection.add(ids=[message['Message-ID']], documents=[whole_text])
                con.execute(f'''insert into emails (from_email, to_email, subject, date, message_id, has_attachment, excerpt) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           [message["From"], message["To"], message["Subject"], parse(message["Date"]), message["Message-ID"], len(list(message.iter_attachments()))>0, whole_text[:30]])


    # check that the DB has been filled in
    con.sql('select subject, sum(has_attachment), count(*) as cnt from emails group by subject order by cnt desc limit 10')


    def query_collection(collection, query):
        # Query the results
        query = 'vyrizena objednavka'
        results = collection.query(query_texts=[query], n_results=2)
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        scores = model.predict([(query, doc) for doc in results["documents"][0]])
        print(results["documents"][0][np.argmax(scores)])
        return results




