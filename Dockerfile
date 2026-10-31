# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime-deps

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ocr.txt requirements-ocr-nodeps.txt ./

RUN pip install --upgrade pip

# Application runtime stack (FastAPI, SQLAlchemy, Alembic, Uvicorn, etc.)
# Installed before the OCR layers below: its pins are all unrelated
# package names to the OCR stack, so installing it first is safe and
# keeps this layer cached independently of OCR dependency churn.
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir -r requirements-ocr.txt

# pyclipper's prebuilt wheel bundles a static zlib that crashes when loaded
# alongside paddlepaddle's; build from source against the system libz.
# Own layer so a flaky build-tool download doesn't redo the big install below.
RUN pip install --no-cache-dir --retries 10 --resume-retries 10 --timeout 60 \
    --no-deps --no-binary pyclipper pyclipper==1.4.0

RUN pip install --no-cache-dir --retries 10 --resume-retries 20 --timeout 60 \
    --no-deps -r requirements-ocr-nodeps.txt

# A broken native OCR stack fails the build, not the first request.
RUN python -c "import cv2, fitz, paddle, paddleocr, PIL; \
print('cv2', cv2.__version__, '| PyMuPDF', fitz.VersionBind, '| paddle', paddle.__version__, \
'| paddleocr', paddleocr.__version__, '| Pillow', PIL.__version__)"


# --- OCR models: fetched once, verified against models/ocr/manifest.json -------
# Verified archives in models/ocr/archives/ (a local, git-ignored cache) are
# used as they are; anything missing comes from the project mirror (default)
# or, only when OCR_MODEL_SOURCE=upstream is passed, from PaddleOCR's servers.
# A private mirror needs a token, given as a BuildKit secret
# (--secret id=github_token,...); it never reaches an image layer.
FROM python:3.12-slim AS ocr-models
ARG OCR_MODEL_SOURCE=mirror
COPY models/ocr/manifest.json models/ocr/NOTICE.md scripts/fetch_ocr_models.py /build/
RUN --mount=type=bind,source=models/ocr,target=/context-models \
    --mount=type=secret,id=github_token,required=false \
    mkdir -p /tmp/archives \
    && (cp /context-models/archives/*.tar /tmp/archives/ 2>/dev/null || true) \
    && python /build/fetch_ocr_models.py \
        --manifest /build/manifest.json \
        --archive-dir /tmp/archives \
        --source "$OCR_MODEL_SOURCE" \
        --token-file /run/secrets/github_token \
        --dest /opt/build-ai/models/paddleocr \
    && cp /build/manifest.json /build/NOTICE.md /opt/build-ai/models/paddleocr/ \
    && rm -rf /tmp/archives


# --- Embedding model: the pinned revision in models/embedding/manifest.json --
# Verified files in models/embedding/files/ (a local, git-ignored cache) are
# used as they are; anything missing comes from the pinned Hugging Face
# revision. Every file is checked against size and sha256 before install.
FROM python:3.12-slim AS embedding-model
COPY models/embedding/manifest.json scripts/fetch_embedding_model.py /build/
RUN --mount=type=bind,source=models/embedding,target=/context-model \
    python /build/fetch_embedding_model.py \
        --manifest /build/manifest.json \
        --cache-dir /context-model/files \
        --dest /opt/build-ai/models/embedding \
    && cp /build/manifest.json /opt/build-ai/models/embedding/


FROM runtime-deps AS app

COPY --from=ocr-models /opt/build-ai/models /opt/build-ai/models
COPY --from=embedding-model /opt/build-ai/models/embedding /opt/build-ai/models/embedding
# Not writable for a non-root user. The image still runs as root (a
# pre-existing gap), which this does not prevent.
RUN chmod -R a-w /opt/build-ai/models
# PaddleOCR and the embedder load only these verified models; nothing is
# downloaded at runtime.
ENV OCR_MODEL_DIR=/opt/build-ai/models/paddleocr
ENV EMBEDDING_MODEL_DIR=/opt/build-ai/models/embedding

# Only what the service needs: no tests or test tooling, development model
# files, evaluation data, local caches or build leftovers from the working
# tree. scripts/ carries the runtime-only offline OCR check.
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY docker ./docker
COPY scripts ./scripts
COPY models/ocr/manifest.json ./models/ocr/manifest.json

RUN chmod +x docker/entrypoint.sh

# A broken embedding stack fails the build, not the first ingestion.
RUN python -c "import asyncio; \
from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder; \
v = asyncio.run(OnnxMiniLMEmbedder('/opt/build-ai/models/embedding', threads=1).embed_text('build check')); \
assert len(v) == 384 and abs(sum(x * x for x in v) - 1) < 1e-4, len(v); print('embedding ok', len(v))"

EXPOSE 8000

ENTRYPOINT ["./docker/entrypoint.sh"]


# --- What production runs, and the default target ------------------------------
# The marker makes the service refuse learning memory while the production gate
# is closed, whatever APP_ENVIRONMENT says (app/core/learning_memory_gate.py). A
# file, since an ENV could be overridden at runtime. Only the boundary gate
# builds --target app, to exercise memory in its development topology.
FROM app AS production

RUN touch /opt/build-ai/production-image && chmod a-w /opt/build-ai/production-image
