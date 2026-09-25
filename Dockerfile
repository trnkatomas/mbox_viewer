FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY *.py ./
COPY templates templates
COPY static static

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

# 0.0.0.0 is needed inside the container; restrict exposure with the published
# port instead (e.g. -p 127.0.0.1:8000:8000).
CMD ["uvicorn", "email_server:app", "--host", "0.0.0.0", "--port", "8000"]
