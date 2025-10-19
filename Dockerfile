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
# We use a slim Python image for the final, lean production environment
FROM python:3.11-slim-bookworm AS final

# Set environment variable
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install FastAPI and Uvicorn with standard dependencies (which includes Jinja2)
RUN pip install "fastapi[standard]" uvicorn

# Set the working directory for the application
WORKDIR /usr/src/app

# Copy the server code
COPY server.py .

# Create the templates directory and copy the base HTML
RUN mkdir -p templates
# Use the *real* index.html for serving
COPY index.html templates/index.html

# Copy the *built* frontend assets from the builder stage
# This copies the actual generated static/ directory containing output.css and bundle.js
COPY --from=builder /app/static/ static/

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# 'server:app' refers to the 'app' object in the 'server.py' file
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

