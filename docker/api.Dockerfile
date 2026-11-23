# Prediction API (FastAPI). Build from the repo root:
#   docker build -f docker/api.Dockerfile -t rides-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app

# 1. Dependencies pinned by uv.lock, with torch in its CPU build: the default Linux build pulls
#    several GB of NVIDIA CUDA libraries that are useless without a GPU. So: install everything
#    except torch and its CUDA libraries, then torch at the version uv.lock pins, from the CPU index.
#    Copied before the code so this slow layer is cached until the dependencies change.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --output-file requirements.txt \
    && TORCH_VERSION="$(sed -n 's/^torch==\([^ ;]*\).*/\1/p' requirements.txt)" \
    && test -n "$TORCH_VERSION" \
    && grep -v -E "^(torch|triton|nvidia-|cuda-)" requirements.txt > requirements-cpu.txt \
    && uv pip install --system --no-cache -r requirements-cpu.txt \
    && uv pip install --system --no-cache "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && rm requirements.txt requirements-cpu.txt

# 2. Application code
COPY src/ src/

# 3. Run as a regular user, not root
RUN useradd --create-home app
USER app

EXPOSE 8000
# /health answers 503 until a model is loaded, so the container is "healthy" only when it can predict
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "rides.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
