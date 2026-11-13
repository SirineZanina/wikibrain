# Build from the repo root:  docker build -f server/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# Dependencies before source, so editing code does not re-run this layer.
COPY server/requirements.txt ./server/requirements.txt

# CPU-only torch. The default PyPI wheel drags in ~3GB of CUDA libraries that this
# image will never use.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r server/requirements.txt

# Bake the embedding model into the image so a cold container needs no network and
# the first request is not slowed by a model download. Kept as an ARG so it cannot
# drift from EMBED_MODEL_LOCAL and silently fall back to a runtime download.
ARG EMBED_MODEL_LOCAL=all-MiniLM-L6-v2
ENV EMBED_MODEL_LOCAL=$EMBED_MODEL_LOCAL
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBED_MODEL_LOCAL'])"

COPY server ./server

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=4)"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
