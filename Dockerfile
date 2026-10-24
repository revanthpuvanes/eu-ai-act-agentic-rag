FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PORT=8080

WORKDIR /app

# Minimal system dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch to avoid CUDA dependencies.
# Keep the exact pinned torch version from requirements.txt.
RUN python -m pip install --no-cache-dir \
      torch==2.14.1 \
      --index-url https://download.pytorch.org/whl/cpu

# The remaining dependencies can reuse the installed torch.
RUN grep -v '^torch==' requirements.txt > /tmp/requirements-api.txt \
    && python -m pip install --no-cache-dir \
       -r /tmp/requirements-api.txt

# Pre-download the embedding model into the image.
RUN python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code and processed legal corpus.
COPY src/ ./src/
COPY data/processed/ ./data/processed/

# Precompute legal corpus embeddings during the Docker build.
RUN mkdir -p data/cache/embeddings \
    && python -c "from src.retrieval.semantic import SemanticRetriever; SemanticRetriever.from_jsonl('data/processed/chunks.jsonl')"


EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
