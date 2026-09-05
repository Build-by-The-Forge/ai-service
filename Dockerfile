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

COPY requirements-ocr.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-ocr.txt
RUN pip install --no-cache-dir --no-deps paddleocr==2.7.0

COPY . .

RUN chmod +x docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./docker/entrypoint.sh"]
