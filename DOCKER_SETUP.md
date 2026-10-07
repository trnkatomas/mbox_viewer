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

## Quick start

Create a `.env` file next to `docker-compose.yml`:

```bash
MBOX_DIR=/path/to/mail          # directory containing the mbox
MBOX_FILE=emails.mbox           # file name inside MBOX_DIR
INDEX_DIR=./index               # where emails.db is kept; must be writable by PUID
COMPOSE_PROFILES=rag            # remove to run without Ollama / rag: search
```

Then:

```bash
docker compose up -d
docker compose logs -f indexer    # first run: indexing progress
```

Open http://localhost:8000 once the indexer has exited.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `MBOX_DIR` | `./mbox_files` | Host directory with the mbox, mounted read-only |
| `MBOX_FILE` | `emails.mbox` | mbox file name inside `MBOX_DIR` |
| `INDEX_DIR` | `./index` | Host directory for `emails.db` |
| `PUID` / `PGID` | `1000` | User the containers run as |
| `COMPOSE_PROFILES` | *(empty)* | `rag` also starts Ollama |
| `INDEX_EMBEDDINGS` | `auto` | `on` fails when Ollama is unavailable, `off` never embeds, `auto` embeds when Ollama is reachable |
| `OLLAMA_MODEL` | `embeddinggemma` | Embedding model; switching needs a rebuild (below) |
| `MBOX_VIEWER_BIND` | `127.0.0.1` | Host address the viewer is published on |
| `MBOX_VIEWER_PORT` | `8000` | Host port the viewer is published on |

The viewer has no authentication. Setting `MBOX_VIEWER_BIND=0.0.0.0` makes
your mail readable by anyone on the network, so only do that on a trusted LAN
or behind a reverse proxy that adds a login.

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
   The viewer is then at `http://<nas-ip>:8000`.

The viewer has no login. Anyone who can reach port 8000 can read the mail, so
keep it on a trusted network or put it behind a reverse proxy that adds
authentication.

To **choose a different mbox**, change `MBOX_FILE_PATH` in both services. The
index belongs to one specific file (it stores byte offsets), so give each
mailbox its own index dataset; pointing an existing index at a different file
makes the indexer rebuild it for the new one.

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
docker compose run --rm indexer python email_utils.py index --rebuild   # from scratch
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
