# MBOX Viewer

A web-based email viewer for MBOX archive files with search, threading, and statistics.

## Features

- **MBOX File Support**: View and browse email archives in standard MBOX format
- **Search**: Full-text search with query syntax (`from:`, `subject:`, `label:`)
- **Email Threading**: Conversation grouping
- **Statistics Dashboard**: Email volume over time and sender distribution with D3.js charts
- **Responsive UI**: Built with Tailwind CSS and HTMX
- **Sandboxed HTML Rendering**: Safe email body display with iframe isolation
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

4. **Process the MBOX file** (first time only)

   This will create the database:
   ```bash
   python -c "from email_utils import process; process()"
   ```

5. **Start the server**
   ```bash
   uvicorn email_server:app --reload
   ```

6. **Open your browser**

   Navigate to `http://localhost:8000`

## Configuration

### Environment Variables

- `MBOX_FILE_PATH`: Path to your MBOX file (required)
  - Default: `/Users/tomastrnka/Downloads/bigger_example.mbox`

### Database Files

The application creates a DuckDB database file:
- `emails.db` - Email metadata and content

This file is automatically created during the processing step and is excluded from git.

## Usage

### Search Syntax

- **Basic search**: `meeting notes`
- **From filter**: `from:john@example.com`
- **Subject filter**: `subject:invoice`
- **Label filter**: `label:important`

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
   uvicorn email_server:app --reload --host 0.0.0.0 --port 8000
   ```

### Project Structure

```
mbox_viewer/
├── email_server.py           # FastAPI application and routes
├── email_utils.py            # Email parsing and database utilities
├── templates/                # Jinja2 templates
│   ├── index.html           # Main layout
│   ├── mail_list.jinja      # Email list container
│   ├── email_list.jinja     # Email list items
│   ├── email_detail.jinja   # Email detail view
│   └── stats.jinja          # Statistics dashboard
├── static/                   # Static assets
│   ├── sim_script.js        # D3.js visualizations
│   └── iframe_wrapper.js    # Iframe management
├── pyproject.toml           # Project dependencies
└── Dockerfile               # Docker configuration
```

### API Endpoints

- `GET /` - Main application
- `GET /api/inbox/layout` - Inbox view
- `GET /api/sent/layout` - Sent folder view
- `GET /api/stats/layout` - Statistics dashboard
- `GET /api/email/list` - Email list (paginated, supports search)
- `GET /api/email/{email_id}` - Email detail
- `GET /api/email_thread/{thread_id}` - Email thread view
- `GET /api/attachment/{email_id}/{attachment_id}` - Download attachment
- `POST /api/search` - Search emails
- `GET /api/stats/data/{query_name}` - Statistics data

## Docker Deployment

### Build the Docker image
```bash
docker build -t mbox-viewer .
```

### Run the container
```bash
docker run -p 8000:8000 \
  -v /path/to/your/emails.mbox:/data/emails.mbox \
  -e MBOX_FILE_PATH=/data/emails.mbox \
  mbox-viewer
```

## License

MIT License - see [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
