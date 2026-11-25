# syntax=docker/dockerfile:1.7

# --- Tailwind CSS -------------------------------------------------------------
# Built with Tailwind's standalone CLI (a single binary, no Node.js), pinned by
# version and SHA-256. It runs on the build machine's own architecture: the CSS
# is identical for every platform, so multi-arch builds don't emulate it.
# Only the binary for the build architecture is downloaded.

FROM --platform=$BUILDPLATFORM python:3.11-slim-bookworm AS tailwind-amd64
ADD --chmod=755 \
    --checksum=sha256:4af3198c015616ea7d6617974ec3d70d987ecc00c1ca8463b0a30fd65cc7c06e \
    https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.19/tailwindcss-linux-x64 \
    /usr/local/bin/tailwindcss

FROM --platform=$BUILDPLATFORM python:3.11-slim-bookworm AS tailwind-arm64
ADD --chmod=755 \
    --checksum=sha256:e5b2d27694daa80cc52ec29553ba2c6bd43d86bd51a9d633ed24058b9c05a676 \
    https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.19/tailwindcss-linux-arm64 \
    /usr/local/bin/tailwindcss

FROM tailwind-${BUILDARCH} AS css
WORKDIR /src
# Everything Tailwind scans for class names (see tailwind.config.js)
COPY tailwind.config.js *.py ./
COPY templates templates
COPY static/src static/src
COPY static/*.js static/
RUN tailwindcss -c tailwind.config.js -i static/src/tailwind.css -o /out/tailwind.css --minify

# `docker build --target css-export --output type=local,dest=static/vendor .`
# writes the CSS into the checkout, for running the app without Docker
FROM scratch AS css-export
COPY --from=css /out/tailwind.css /tailwind.css

# --- Application --------------------------------------------------------------

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
COPY --from=css /out/tailwind.css static/vendor/tailwind.css

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

# 0.0.0.0 is needed inside the container; restrict exposure with the published
# port instead (e.g. -p 127.0.0.1:8000:8000). Serves HTTPS with the certificate
# from STATE_DIR (see web_security.py).
ENV MBOX_VIEWER_HOST=0.0.0.0
CMD ["python", "email_server.py"]
