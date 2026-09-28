FROM python:3.12-slim

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

COPY . .

RUN chmod +x docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./docker/entrypoint.sh"]
