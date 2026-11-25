# Web app (Streamlit). Build from the repo root:
#   docker build -f docker/webapp.Dockerfile -t rides-webapp .
# It only calls the API over HTTP, so it needs the "webapp" dependency group, not torch or MLflow.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app

# 1. Dependencies of the "webapp" group, pinned by uv.lock (streamlit also brings requests, numpy, pandas)
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --only-group webapp --no-emit-project --no-hashes --output-file requirements.txt \
    && uv pip install --system --no-cache -r requirements.txt \
    && rm requirements.txt

# 2. Application code
COPY webapp/ webapp/

# 3. Run as a regular user, not root
RUN useradd --create-home app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]
CMD ["streamlit", "run", "webapp/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
