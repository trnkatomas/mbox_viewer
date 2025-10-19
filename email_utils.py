
import email
from email.policy import default

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

mboxfilename = 'email_sample.mbox'
counter = 0
with MboxReader(mboxfilename) as mbox:
    for message in mbox:
        counter += 1
        print(f"{message['From']} -> {message['To']}: {message['Subject']}")
        for part in message.iter_attachments():
            fn = part.get_filename()
            print(f'\tattachment: {fn}')
