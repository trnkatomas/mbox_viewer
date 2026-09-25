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

## Running on a NAS (TrueNAS Scale and similar)

TrueNAS Scale 24.10+ runs custom apps from compose YAML (*Apps → Discover Apps
→ Custom App → Install via YAML*), or you can use `docker compose` over SSH.

- Point `MBOX_DIR` at the dataset holding the mbox. It is only ever read.
- Put `INDEX_DIR` on a dataset for app data and make it writable by the
  container user. On TrueNAS that is typically the `apps` user, so set
  `PUID=568` and `PGID=568`.
- The index is much smaller than the mbox (no attachments, text capped per
  message) but is read constantly while searching. SSD-backed storage (e.g. the
  apps pool) helps, but spinning disks work.
- **Embeddings on a CPU-only NAS are slow**: expect hours for a large mailbox.
  Options:
  - Leave out the `rag` profile. You get everything except `rag:` search, and
    can add embeddings later; only missing ones are computed.
  - Build the index on a faster machine and copy `emails.db` into `INDEX_DIR`.
    The index stores byte offsets, not paths, so it works against an identical
    copy of the mbox. The indexer verifies that on start.
  - Point `OLLAMA_URL` in `docker-compose.yml` at an Ollama running on another
    machine with a GPU.

After replacing the mbox (e.g. with a new Takeout export), restart the stack.
The indexer detects the mismatch and rebuilds. If the new export only
*appends* to the old file, just the new messages are indexed.

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
