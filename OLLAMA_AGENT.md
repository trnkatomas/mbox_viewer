# Ollama Agent for mbox-viewer

`ollama_agent.py` connects a locally-running Ollama LLM to the mbox-viewer
MCP server so you can query your email archive in plain language.

## Prerequisites

- Ollama running locally with a tool-capable model pulled:
  ```bash
  ollama pull qwen2.5:7b   # or llama3.1:8b, mistral-nemo, qwen2.5:14b, …
  ```
- The `ollama` Python package:
  ```bash
  uv add ollama
  # or: pip install ollama
  ```
- `MBOX_FILE_PATH` pointing at a processed mbox archive (see main README).

## Usage

```bash
MBOX_FILE_PATH=/path/to/your.mbox python ollama_agent.py "who sent me the most emails?"

# choose a different model
MBOX_FILE_PATH=/path/to/your.mbox python ollama_agent.py --model llama3.1:8b "find invoices from last year"

# show tool calls and response previews
MBOX_FILE_PATH=/path/to/your.mbox python ollama_agent.py --verbose "summarise my emails from alice"

# interactive prompt (query omitted)
MBOX_FILE_PATH=/path/to/your.mbox python ollama_agent.py
```

## How it works

1. Spawns `mcp_server.py` as a subprocess (stdio MCP transport).
2. Fetches available tools (`search_emails`, `get_email`, `get_thread`, `get_stats`, …)
   and converts their schemas to Ollama's function-calling format.
3. Runs a tool-calling loop: the model decides which tools to call, the agent
   executes them via MCP and feeds results back, until the model produces a
   final plain-text answer.

## Semantic (RAG) search

If you have embeddings indexed, the agent can use the `rag:` filter inside
`search_emails` for semantic queries. This requires Ollama to also serve an
embedding model (set `OLLAMA_URL` and `OLLAMA_MODEL` in `.env`):

```bash
ollama pull nomic-embed-text
```

See `.env.example` for configuration details.
