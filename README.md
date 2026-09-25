# MBOX Viewer

A web-based email viewer for MBOX archive files with search, threading, and statistics.

## Features

- **MBOX File Support**: View and browse email archives in standard MBOX format
- **Search**: Keyword search over subject and body text with filters (`from:`, `subject:`, `label:`, dates)
- **Semantic Search**: `rag:` queries ranked by embedding similarity (Ollama), combinable with filters
- **Email Threading**: Conversation grouping
- **Statistics Dashboard**: Email volume over time and sender distribution with D3.js charts
- **Responsive UI**: Built with Tailwind CSS and HTMX
- **Safe HTML Rendering**: Email bodies are sanitized (nh3) and shown in an opaque-origin sandboxed iframe with a strict CSP; remote images are blocked until you opt in per message
- **Attachment Support**: Download email attachments
- **Folder Organization**: Inbox and Sent folder views

## Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** - Web framework
- **DuckDB** - Embedded SQL database for email metadata
- **Uvicorn** - ASGI server

### Frontend
- **HTMX** - Dynamic HTML updates
- **Tailwind CSS** - Styling
- **D3.js** - Data visualizations
- **Jinja2** - Server-side templating

## Installation

### Prerequisites
- Python 3.10 or higher
- An MBOX file to view

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mbox_viewer
   ```

2. **Install dependencies**

   Using `uv` (recommended):
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up your MBOX file**

   Set the `MBOX_FILE_PATH` environment variable:
   ```bash
   export MBOX_FILE_PATH="/path/to/your/emails.mbox"
   ```

4. **Index the MBOX file**

   Requires [Ollama](https://ollama.com) with an embedding model (`ollama pull embeddinggemma`):
   ```bash
   uv run python email_utils.py index
   ```
   Indexing is resumable: if it is interrupted or Ollama goes away, run the same
   command again and it continues where it stopped. New messages appended to the
   mbox are picked up the same way, and only the new part of the file is read.
   Use `--rebuild` to start from scratch (for example to switch embedding models).

   Without Ollama, `--embeddings off` builds a metadata-only index (everything but
   `rag:` search works); a later run with Ollama adds the missing embeddings.
   `--embeddings auto` embeds only if Ollama is reachable.

   Messages are read from the mbox by byte offset, so the index must match the
   file. If the mbox is replaced (e.g. by a new Takeout export), `index` refuses
   to continue and the server logs an error at startup; `--rebuild-if-stale`
   rebuilds automatically instead.

   **Upgrading a database from an older version:** run
   `uv run python email_utils.py migrate` once. It widens byte offsets to BIGINT
   (the old INTEGER columns overflow on mbox files over 2 GB), drops the
   experimental HNSW index and unused columns, and records the embedding model.
   The original file is kept as `emails.db.bak`. Migrated emails keep their old
   whole-message embeddings; `index --rebuild` re-embeds everything with chunking
   and correct charset handling, which also fixes emails whose text was lost by
   the old extractor.

5. **Start the server**
   ```bash
   uvicorn email_server:app --reload
   ```

6. **Open your browser**

   Navigate to `http://localhost:8000`

## Configuration

### Environment Variables

- `MBOX_FILE_PATH`: Path to your MBOX file (required)
- `EMAILS_DB_PATH`: Path to the DuckDB index (default: `emails.db`)
- `OLLAMA_URL`: Ollama embedding endpoint (default: `http://localhost:11434/api/embed`)
- `OLLAMA_MODEL`: Embedding model used when indexing (default: `embeddinggemma`).
  The model, vector size and prompt prefixes are stored in the index, and searches
  always use the stored model, so changing this variable cannot silently break search.
- `RAG_MAX_DISTANCE`: Cosine-distance cut-off for `rag:` hits (default: `0.75`)
- `MBOX_VIEWER_HOST` / `MBOX_VIEWER_PORT`: Bind address for `python email_server.py`
  (default: `127.0.0.1:8000`). The app has no authentication, so only expose it
  beyond localhost behind something that adds it.
- `INDEX_EMBEDDINGS`: Default for `index --embeddings` (`on`, `off` or `auto`; default `on`)
- `LOG_LEVEL`: Logging level for the MCP server and indexer (default: `INFO`)

### Database Files

`emails.db` (DuckDB) holds three tables: `emails` (metadata, excerpt, extracted body
text and byte offsets into the mbox), `embeddings` (one vector per ~500-token chunk)
and `index_meta` (embedding settings). Raw messages and attachments are always read
back from the mbox, never copied. The file is excluded from git.

Older versions also wrote a ChromaDB store to `emails.chromadb/`. Nothing reads it
any more; it can be deleted.

## Usage

### Search Syntax

- **Basic search**: `meeting notes` (subject and body text, case-insensitive)
- **From filter**: `from:john@example.com`
- **Subject filter**: `subject:invoice` or `subject:"team meeting"`
- **Label filter**: `label:important`
- **Dates**: `from_date:2024-01-01 to_date:2024-06-30` (both inclusive)
- **Semantic**: `rag:flight to prague` (the phrase runs to the next filter or the end)

## Development

### Development Setup

1. **Install development dependencies**
   ```bash
   uv sync --group dev
   ```

2. **Code formatting**
   ```bash
   black .
   isort .
   ```

3. **Run the development server**
   ```bash
   uvicorn email_server:app --reload --port 8000
   ```

### Project Structure

```
mbox_viewer/
├── email_server.py           # FastAPI application and routes
├── email_service.py          # Business logic shared by the web app and MCP server
├── email_utils.py            # mbox reading, parsing, queries, indexing CLI
├── email_render.py           # Sanitization and sandboxed iframe documents
├── mcp_server.py             # MCP tools for AI agents
├── ollama_agent.py           # Minimal Ollama agent using the MCP server
├── templates/                # Jinja2 templates
│   ├── index.html           # Main layout
│   ├── mail_list.jinja      # Email list container
│   ├── email_list.jinja     # Email list items
│   ├── _message.jinja       # Shared message card macro
│   ├── email_detail.jinja   # Email detail view
│   ├── email_detail_thread.jinja # Thread view
│   └── stats.jinja          # Statistics dashboard
├── static/                   # Static assets
│   ├── sim_script.js        # D3.js visualizations
│   └── email_frames.js      # Iframe sizing and list selection
├── pyproject.toml           # Project dependencies
└── Dockerfile               # Docker configuration
```

### API Endpoints

- `GET /` - Main application
- `GET /api/inbox/layout` - Inbox view
- `GET /api/sent/layout` - Sent folder view
- `GET /api/stats/layout` - Statistics dashboard
- `GET /api/email/list` - Email list (paginated, supports search)
- `GET /api/email/{email_id}?remote=true` - Email detail (`remote` loads remote images)
- `GET /api/email_thread/{thread_id}?remote=true` - Email thread view
- `GET /api/attachment/{email_id}/{index}` - Download the attachment at that position
- `POST /api/search` - Search emails
- `GET /api/stats/data/{query_name}` - Statistics data

## MCP Interface (AI Agent Access)

The MCP server (`mcp_server.py`) exposes the email archive as an
[MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, letting
AI agents browse and query emails programmatically via the same tools available
in the web interface.

### Available tools

| Tool | Description |
|---|---|
| `search_emails_tool` | Search by keyword, sender, subject, date range, label, or semantic query (`rag:`) |
| `get_email_tool` | Read the full body and metadata of a single email |
| `get_thread_tool` | Retrieve all emails in a conversation thread |
| `get_attachment_info_tool` | Get metadata (filename, type, size) of an attachment |
| `get_stats_tool` | Summary statistics for the archive |
| `get_stats_time_series_tool` | Time-series data (email volume, top sender domains) |
| `get_server_status_tool` | Diagnostics: DB loaded, env vars set, VSS availability |

### Connecting to Claude Desktop

Add the server to your Claude Desktop configuration file
(`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "mbox-viewer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mbox_viewer", "mcp_server.py"],
      "env": {
        "MBOX_FILE_PATH": "/path/to/your/archive.mbox"
      }
    }
  }
}
```

Restart Claude Desktop after saving. The 7 mbox-viewer tools will appear in the
tool picker (hammer icon) in the chat input.

To enable semantic (`rag:`) search, also set `OLLAMA_URL` and `OLLAMA_MODEL` in
the `env` block — see the Configuration section for details.

### Connecting to Claude Code (CLI)

Run the server as a project-scoped MCP server:

```bash
claude mcp add mbox-viewer \
  --command "uv" \
  --args "run,--directory,/path/to/mbox_viewer,mcp_server.py" \
  -- \
  MBOX_FILE_PATH=/path/to/your/archive.mbox
```

Or add it manually to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "mbox-viewer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mbox_viewer", "mcp_server.py"],
      "env": {
        "MBOX_FILE_PATH": "/path/to/your/archive.mbox"
      }
    }
  }
}
```

### Testing the server

**1. Protocol smoke test** — send raw MCP messages via stdin and inspect the
responses:

```bash
MBOX_FILE_PATH=/path/to/archive.mbox uv run python mcp_server.py <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_server_status_tool","arguments":{}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_stats_tool","arguments":{}}}
EOF
```

A healthy response for `get_server_status_tool` will show `"db_loaded": true`
and `"mbox_file_exists": true`. `get_stats_tool` should return `all_emails > 0`.

**2. Interactive browser inspector** — launches the MCP Inspector UI:

```bash
MBOX_FILE_PATH=/path/to/archive.mbox uv run mcp dev mcp_server.py
```

Open `http://localhost:5173` to call tools interactively and inspect payloads.

**3. Unit tests**

```bash
uv run pytest tests/test_mcp_server.py -v
```

### Typical agent workflow

1. Call `get_server_status_tool` to confirm the server is configured correctly.
2. Call `search_emails_tool` with a query to find relevant emails — results
   include a `message_id` and `thread_id` for each match.
3. Call `get_email_tool` with a `message_id` to read the full body of one email.
4. Call `get_thread_tool` with a `thread_id` to read the whole conversation.

### Search query syntax

All filters are combined with AND and can be mixed freely in a single string:

| Filter | Example | Notes |
|---|---|---|
| `from:` | `from:alice@example.com` | Substring match on sender |
| `subject:` | `subject:"team meeting"` | Substring match; quote multi-word values |
| `label:` | `label:Inbox` | Exact label match |
| `from_date:` | `from_date:2024-01-01` | Emails on or after this date (YYYY-MM-DD) |
| `to_date:` | `to_date:2024-06-30` | Emails on or before this date, inclusive (YYYY-MM-DD) |
| `rag:` | `rag:flight booking` | Semantic search ranked by relevance; runs to the next filter (requires Ollama, returns an error if it is down) |
| plain keywords | `invoice overdue` | Case-insensitive match on subject and body text |

Example combined query:
```
from:boss@corp.com subject:budget from_date:2024-01-01 to_date:2024-12-31
```

## Docker Deployment

The mbox is never copied into the image: it is mounted read-only and the index
is built inside the stack by a one-shot `indexer` service before the viewer
starts. This works well with the mbox on a NAS (TrueNAS, Synology, ...):

```bash
MBOX_DIR=/mnt/tank/mail MBOX_FILE=takeout.mbox INDEX_DIR=/mnt/tank/apps/mbox-viewer \
  docker compose up -d
```

Add `COMPOSE_PROFILES=rag` to also run Ollama for `rag:` search. See
[DOCKER_SETUP.md](DOCKER_SETUP.md) for all settings.

## License

MIT License - see [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
