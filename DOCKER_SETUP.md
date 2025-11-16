# Docker Setup with Ollama for RAG Search

This guide explains how to run MBOX Viewer with Ollama for semantic search capabilities.

## Architecture

The Docker Compose setup includes two containers:
- **ollama**: Runs Ollama with embedding models for semantic search
- **mbox-viewer**: The main application server

## Quick Start

### 1. Start the containers

```bash
docker-compose up -d
```

This will:
- Pull and start the Ollama container
- Build and start the MBOX Viewer application
- Create a persistent volume for Ollama models

### 2. Pull the embedding model

```bash
docker exec -it mbox-ollama ollama pull nomic-embed-text
```

The `nomic-embed-text` model is recommended for email embeddings (768 dimensions, good quality/speed trade-off).

### 3. Verify Ollama is working

```bash
curl http://localhost:11434/api/embed \
  -d '{
    "model": "nomic-embed-text",
    "input": "test query"
  }'
```

### 4. Access the application

Open http://localhost:8000 in your browser.

## Configuration

### Environment Variables

Set these in `docker-compose.yml` or `.env` file:

- `MBOX_FILE_PATH`: Path to your MBOX file (default: `/app/mbox_files/emails.mbox`)
- `OLLAMA_URL`: Ollama API endpoint (default: `http://ollama:11434/api/embed`)
- `OLLAMA_MODEL`: Embedding model to use (default: `nomic-embed-text`)

### Volume Mounts

- `./emails.db`: DuckDB database with email metadata and embeddings
- `./mbox_files`: Directory containing your MBOX files (read-only)
- `ollama_data`: Persistent volume for Ollama models

## Using RAG Search

Once the embeddings are generated (see below), you can use semantic search:

```
rag:project deadline discussions
```

This will find emails semantically similar to your query, even if they don't contain the exact words.

### Combine with other filters

```
rag:budget approval from_date:2024-01-01 to_date:2024-06-30
```

## Generating Embeddings

Before RAG search works, you need to generate embeddings for your emails:

```bash
# SSH into the container
docker exec -it mbox-viewer-app bash

# Run the embedding generation script
python -c "from email_utils import process_embeddings; process_embeddings()"
```

**Note**: This can take a while depending on the number of emails.

## Alternative Embedding Models

You can use different Ollama models:

```bash
# Smaller, faster (384 dims)
docker exec -it mbox-ollama ollama pull all-minilm

# Larger, better quality (1024 dims)
docker exec -it mbox-ollama ollama pull mxbai-embed-large
```

Update `OLLAMA_MODEL` and adjust the embedding dimension in `rag_search_duckdb()` accordingly.

## GPU Support

If you have NVIDIA GPU, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Troubleshooting

### Ollama not responding

```bash
docker logs mbox-ollama
docker restart mbox-ollama
```

### Check if model is loaded

```bash
docker exec -it mbox-ollama ollama list
```

### Application can't connect to Ollama

Ensure the containers are on the same network:

```bash
docker network inspect mbox_viewer_default
```

## Stopping the services

```bash
docker-compose down
```

To also remove volumes:

```bash
docker-compose down -v
```
