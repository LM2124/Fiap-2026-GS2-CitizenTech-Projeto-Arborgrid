# Multi-stage: compila o frontend (Node) e as dependencias Python em estagios separados;
# a imagem final so leva o runtime Python, o backend e o build estatico, rodando sem root.

# 1. Frontend React/Vite -> /build/app/static (outDir "../app/static" do vite.config.ts)
FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


# 2. Dependencias Python num venv isolado
FROM python:3.15-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip==24.3.1 \
    && pip install --no-cache-dir -r requirements.txt


# 3. Runtime enxuto
FROM python:3.15-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ARBORGRID_MANIFEST_PATH="/app/data/manifest.json" \
    ARBORGRID_KNOWLEDGE_BASE_PATH="/app/data/knowledge_base.jsonl" \
    ARBORGRID_SAMPLE_GOLD_PATH="/app/data/sample_gold_upsert.json" \
    ARBORGRID_SAMPLES_DIR="/app/data/samples" \
    ARBORGRID_ONNX_MODEL_PATH="/app/models/segformer_v3_int8_dyn_matmul.onnx" \
    ARBORGRID_MEDIA_DIR="/app/media" \
    ARBORGRID_GOLD_STATE_PATH="/app/state/gold_risk_scores_runtime.json"

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --shell /usr/sbin/nologin --create-home appuser \
    && mkdir -p /app/data /app/media /app/state /app/models \
    && chown -R appuser:appgroup /app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup app /app/app
COPY --from=frontend --chown=appuser:appgroup /build/app/static /app/app/static
COPY --chown=appuser:appgroup data /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
