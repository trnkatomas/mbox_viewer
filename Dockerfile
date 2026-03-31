# ----------------------------------------------------------------------
# 1. Builder Stage (For Frontend Assets)
# This stage handles all bundling and minification using Webpack and Tailwind CLI.
# ----------------------------------------------------------------------
FROM node:18-alpine AS builder

# Set the working directory
WORKDIR /app

# 1. Prepare environment and install necessary Node packages
# We install Tailwind CLI, Webpack CLI, and the Webpack core library.
RUN npm init -y
RUN npm install tailwindcss webpack webpack-cli --save-dev

# 2. Create minimal source files needed for the build (Source files)
RUN mkdir -p src templates static

# Create input CSS file with Tailwind directives
RUN echo "@tailwind base; @tailwind components; @tailwind utilities;" > src/input.css

# Create a minimal Tailwind configuration file
RUN echo "module.exports = { content: ['./templates/*.html'], theme: { extend: {}, }, plugins: [], }" > tailwind.config.js

# Create the JS entry point file (src/app.js)
RUN echo "import { init } from './htmx-simulator';" > src/app.js
# Create a simulated module that Webpack will bundle
RUN echo "/* Minimal HTMX Simulator */ export function init() { console.log('HTMX and custom logic bundled by Webpack.'); }; init();" > src/htmx-simulator.js

# Create a placeholder HTML file for Tailwind's content scanner
# This placeholder ensures the necessary Tailwind classes for the layout are kept.
RUN echo '<html><body class="bg-gray-900 text-gray-100"><div hx-get="/api/email/list"></div></body></html>' > templates/index.html 

# 3. RUN the Frontend Builds

# 3a. Run Webpack for JS Bundling
# Bundles src/app.js and its dependencies into static/bundle.js
RUN npx webpack --entry ./src/app.js --output-path ./static --output-filename bundle.js --mode production

# 3b. Run the Tailwind CLI for CSS Purging and Minification
# This generates the minified, purged CSS file based on the templates/index.html placeholder.
RUN npx tailwindcss -i src/input.css -o static/output.css --minify

# ----------------------------------------------------------------------
# 2. Final Stage (For Python/FastAPI Application)
# ----------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user and group (uid/gid 1000)
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --no-create-home --shell /sbin/nologin appuser

WORKDIR /app

# Install dependencies as root before dropping privileges
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY email_server.py email_service.py email_utils.py mcp_server.py ./
COPY templates/ templates/
COPY static/ static/

# Copy the built frontend assets from the builder stage
COPY --from=builder /app/static/ static/

# Hand ownership of the working directory to the app user
# (the mbox file and db are mounted as volumes and must be pre-chowned on the host)
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "email_server:app", "--host", "0.0.0.0", "--port", "8000"]

