# Docker Setup

The compose stack reads your mbox from wherever it lives (a local directory or
a NAS dataset) without copying it into the image.

## Architecture

- **indexer**: a one-shot job that brings the DuckDB index up to date, then exits.
- **mbox-viewer**: the web app. It starts once the indexer has finished successfully.
- **ollama** (optional, `rag` profile): embeddings for semantic `rag:` search.

The mbox is mounted read-only in both app containers. The index directory is
writable only for the indexer. Messages are read from the mbox by byte offset
on demand, so only the index (metadata, extracted text, embeddings) is kept
separately.

On each start the indexer:

1. checks that the index still matches the mbox, and rebuilds it if the file was replaced;
2. reads only the part of the mbox after the last indexed message, so appended mail is added
   and an unchanged mbox costs nothing;
3. embeds messages that don't have embeddings yet, if Ollama is available. The first run pulls
   the model automatically.

## Quick look without any setup

To browse an mbox on your own machine, one container is enough. It indexes the
folder (without embeddings, so no Ollama) and serves it at
http://localhost:8000 with no login and no HTTPS:

```bash
docker run --rm -p 127.0.0.1:8000:8000 --user "$(id -u):$(id -g)" \
  -e MBOX_VIEWER_SECURITY=local -e MBOX_DIR=/data/mbox -e INDEX_DIR=/data/index \
  -v /path/to/mail:/data/mbox:ro -v "$PWD/index:/data/index" \
  ghcr.io/trnkatomas/mbox_viewer:latest \
  sh -c "python email_utils.py index --embeddings off && python email_server.py"
```

`MBOX_VIEWER_SECURITY=local` is what turns the protection off, so keep the
`127.0.0.1:` in `-p`: without it Docker publishes the port to your whole
network. Indexes in `./index` are reused on the next run.

## Quick start

Create a `.env` file next to `docker-compose.yml`:

```bash
MBOX_DIR=/path/to/mail          # directory with your .mbox files
INDEX_DIR=./index               # where the indexes are kept; must be writable by PUID
ALLOWED_HOSTS=localhost,127.0.0.1,nas.local,192.168.1.10   # names you open it by
MBOX_VIEWER_PASSWORD_HASH=      # filled in below
COMPOSE_PROFILES=rag            # remove to run without Ollama / rag: search
```

Create the login password's hash and paste it into `MBOX_VIEWER_PASSWORD_HASH`:

```bash
docker compose run --rm --no-deps indexer python web_security.py hash-password
```

Then:

```bash
docker compose up -d
docker compose logs -f indexer    # first run: indexing progress
```

Open https://localhost:8000 once the indexer has exited (see *HTTPS and login*
below about the certificate warning).

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `MBOX_DIR` | `./mbox_files` | Host directory with the mbox files, mounted read-only. Every `*.mbox` file in it (subfolders included, hidden ones skipped) is a mailbox |
| `MBOX_FILE` | `emails.mbox` | Optional: the mbox in `MBOX_DIR` whose index is `emails.db` (indexes built before multiple mailboxes were supported keep working) |
| `INDEX_DIR` | `./index` | Host directory for the indexes, one `<mailbox-id>.db` per mailbox |
| `PUID` / `PGID` | `1000` | User the containers run as |
| `COMPOSE_PROFILES` | *(empty)* | `rag` also starts Ollama |
| `INDEX_EMBEDDINGS` | `auto` | `on` fails when Ollama is unavailable, `off` never embeds, `auto` embeds when Ollama is reachable |
| `OLLAMA_MODEL` | `embeddinggemma` | Embedding model; switching needs a rebuild (below) |
| `MBOX_VIEWER_SECURITY` | `full` | `full`: login and HTTPS. `local`: neither, for a viewer only you can reach |
| `MBOX_VIEWER_PASSWORD_HASH` | *(required with `full`)* | Login password hash from `python web_security.py hash-password` |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated names/IPs the viewer may be reached by (`*.example.lan` works); also the names on its HTTPS certificate |
| `MBOX_VIEWER_BIND` | `127.0.0.1` | Host address the viewer is published on; `0.0.0.0` for the LAN |
| `MBOX_VIEWER_PORT` | `8000` | Host port the viewer is published on |

## HTTPS and login

The viewer only serves HTTPS and asks for a password before showing anything.

- **Password**: a single password, stored only as a scrypt hash in
  `MBOX_VIEWER_PASSWORD_HASH` (see *Setting the password* below). Without one
  the viewer refuses to start. After five wrong attempts from one address
  (thirty overall) logins pause for fifteen minutes.
- **Sessions** last seven days (`MBOX_VIEWER_SESSION_HOURS`) in a signed,
  `HttpOnly`, `SameSite=Strict` cookie. *Sign out* ends one; changing the
  password ends all of them.
- **Host names**: requests addressed to a name not in `ALLOWED_HOSTS` are
  refused. Besides keeping the certificate correct, this stops other websites
  from reaching the viewer through your browser (DNS rebinding).
- **Certificate**: on each start the indexer creates, if missing, a small local
  certificate authority and a server certificate for the `ALLOWED_HOSTS` names,
  in `INDEX_DIR/state/tls/`. The server certificate is reissued when the names
  change or it nears expiry; the authority stays the same, so a device only has
  to trust it once. Until then, browsers show a warning that you can click
  through. To trust it, open `https://<host>:8000/tls/ca.pem` (the link is also
  on the login page) and:
  - **macOS**: open the file, then in Keychain Access set it to *Always Trust*.
  - **iOS / iPadOS**: install the downloaded profile (*Settings → Profile
    Downloaded*), then enable it under *Settings → General → About →
    Certificate Trust Settings*.
  - **Windows**: open the file, *Install Certificate → Local Machine → Trusted
    Root Certification Authorities*.
  - **Android**: *Settings → Security → Encryption & credentials → Install a
    certificate → CA certificate*.
  - **Firefox** keeps its own list: *Settings → Privacy & Security →
    Certificates → View Certificates → Authorities → Import*.

  Only trust it if you set the server up yourself: the authority can vouch for
  any name. Its private key (`ca-key.pem`) never leaves the index folder.

### Setting the password

The viewer never stores the password itself, only a hash of it. You create the
hash once with the `hash-password` command, which asks for the password twice
(at least 12 characters) and prints one line starting with `scrypt:`. That line
is the value of `MBOX_VIEWER_PASSWORD_HASH`. It contains no `$`, so it can be
pasted into `.env` and compose files as is. The hash works anywhere, so you can
create it on any machine that has the image or this repository.

**Running locally** (from a checkout of this repository):

```bash
uv run python web_security.py hash-password
```

Put the printed line in your environment or `.env`:
`MBOX_VIEWER_PASSWORD_HASH=scrypt:17:8:1:...`

**Docker Compose**:

```bash
docker compose run --rm --no-deps indexer python web_security.py hash-password
```

Paste the line into `.env` next to `docker-compose.yml`
(`MBOX_VIEWER_PASSWORD_HASH=scrypt:17:8:1:...`), then run
`docker compose up -d`. Compose recreates the viewer with the new value.

**TrueNAS**: open *System → Shell* and run

```bash
docker run --rm -it ghcr.io/trnkatomas/mbox_viewer:latest python web_security.py hash-password
```

Then edit the app (*Apps → mbox-viewer → Edit*) and replace
`REPLACE_WITH_PASSWORD_HASH` in the viewer service, keeping the quotes:
`MBOX_VIEWER_PASSWORD_HASH: "scrypt:17:8:1:..."`. Saving redeploys the app.

**Changing or resetting it**: there is no way to recover a password from its
hash, so a forgotten password is replaced the same way: create a new hash and
swap it in. Every open session ends when the hash changes.

`MBOX_VIEWER_PASSWORD` with the plain password also works, but then the
password is readable by anyone who can inspect the container's environment,
so prefer the hash.

### Turning it off

`MBOX_VIEWER_SECURITY=local` disables both the login and
HTTPS (the host-name check stays on). Use it only when the port is reachable
from your own machine alone, e.g. published as `-p 127.0.0.1:8000:8000`; the
viewer logs a warning and shows a *No login* notice in its sidebar. Each part
can also be switched on its own, overriding the mode: `MBOX_VIEWER_AUTH=on|off`
and `MBOX_VIEWER_TLS=on|off` (e.g. login without HTTPS behind a reverse proxy
that terminates TLS). Misspelled values stop the server rather than guessing.

All containers drop every Linux capability and run with `no-new-privileges`.
The indexer and viewer also have a read-only root filesystem: the only
writable paths are `/tmp` (a tmpfs) and, for the indexer, `INDEX_DIR`. Ollama
publishes no port; the indexer and viewer reach it over the compose network,
and models can be pulled with `docker exec mbox-ollama ollama pull <model>`.

## Running on TrueNAS (Community Edition)

Use `docker-compose.truenas.yml`, not `docker-compose.yml`. TrueNAS's
*Install via YAML* runs the file exactly as written and has no `.env` file, so
the `${...}` settings in the main compose file would silently fall back to
defaults like `./mbox_files` that don't exist on the NAS. The TrueNAS file uses
literal values and the prebuilt image `ghcr.io/trnkatomas/mbox_viewer`, so
nothing is built on the NAS.

1. **Create two datasets**, one for the mbox (e.g. `/mnt/POOL/mail`) and one
   for the index (e.g. `/mnt/POOL/apps/mbox`). The app runs as the `apps` user
   (uid/gid 568): give it read access to the mail dataset and write access to
   the index dataset (*Datasets → Edit Permissions*).
2. **Copy the mbox** into the mail dataset.
3. **Install**: *Apps → Discover Apps → ⋮ → Install via YAML*, name it
   `mbox-viewer`, and paste `docker-compose.truenas.yml`. Replace every
   `/mnt/POOL/...` path, and `emails.mbox` if your file is named differently
   (it appears in both services).
4. **First start**: the `indexer` container indexes the mailbox and exits; the
   `viewer` starts once it has finished. Follow progress in the indexer's logs.
   The viewer is then at `https://<nas-ip>:8000`.

Before installing, set the two security values in the YAML (see *HTTPS and
login* above): `ALLOWED_HOSTS` in both services, to the names and IP you will
open the viewer by, and `MBOX_VIEWER_PASSWORD_HASH` in the viewer. To create
the hash, run this in the TrueNAS shell (*System → Shell*):

```bash
docker run --rm -it ghcr.io/trnkatomas/mbox_viewer:latest python web_security.py hash-password
```

**Multiple mailboxes**: every `*.mbox` file in the mail dataset is a mailbox,
and the viewer's sidebar has a *Mailbox* menu to switch between them. Each
mailbox has its own index file in the index dataset (named after the file,
e.g. `work-2024.mbox` → `work-2024.db`). To add one, copy the file into the
mail dataset and restart the app so the indexer picks it up; until then it is
listed as *not indexed*.

After **replacing the mbox** with a new export, restart the app: the indexer
detects the mismatch and rebuilds. If the new export only *appends* to the old
file, just the new messages are indexed.

The image tag is `latest` by default. Once versions are tagged (`v1.2.3`), pin
one (`:1.2.3`) so updates happen only when you edit the tag.

### Storage and performance

- The index is much smaller than the mbox (no attachments, text capped per
  message) but is read constantly while searching. SSD-backed storage (e.g. the
  apps pool) helps, but spinning disks work.
- **Embeddings on a CPU-only NAS are slow**: expect hours for a large mailbox.
  The TrueNAS file therefore starts without Ollama (`INDEX_EMBEDDINGS: "off"`),
  which gives you everything except `rag:` search. Options for adding it:
  - Uncomment the `ollama` service and the indexer's `depends_on`, and set
    `INDEX_EMBEDDINGS` to `auto`. Only missing embeddings are computed.
  - Build the index on a faster machine and copy `emails.db` into the index
    dataset. The index stores byte offsets, not paths, so it works against an
    identical copy of the mbox; the indexer verifies that on start.
  - Point `OLLAMA_URL` at an Ollama running on another machine with a GPU, and
    set `INDEX_EMBEDDINGS` to `auto`.

### Other NAS systems

Anywhere you can run `docker compose` over SSH, the main `docker-compose.yml`
with a `.env` file (see *Quick start*) works too; set `PUID`/`PGID` to the user
that owns the datasets.

## Manual indexing

```bash
docker compose run --rm indexer python email_utils.py index --rebuild   # every mailbox, from scratch
docker compose run --rm indexer python email_utils.py index --mailbox work-2024 --rebuild   # just one
docker compose run --rm indexer python email_utils.py index --help
```

Stop the viewer first (`docker compose stop mbox-viewer`). DuckDB can't write
the index while another process has it open.

## Alternative embedding models

Set `OLLAMA_MODEL` (e.g. `nomic-embed-text` or `mxbai-embed-large`), then
rebuild the index as above. The model name, vector size and prompt prefixes
are stored in the index, so searches always use the model the index was
built with.

## GPU support

If you have an NVIDIA GPU, uncomment the `deploy` section of the `ollama`
service in `docker-compose.yml`.

## Troubleshooting

- **Viewer doesn't start**: `docker compose logs indexer`. The viewer waits for
  the indexer to succeed.
- **Permission denied writing the index**: `INDEX_DIR` must be writable by
  `PUID`/`PGID`.
- **"Semantic search is unavailable"**: the index has no embeddings, or Ollama
  isn't running. Enable the `rag` profile and restart.
- **Wrong message content shown**: the mbox changed without the index being
  rebuilt. The viewer logs an error about it at startup; restart the stack to
  let the indexer rebuild.
