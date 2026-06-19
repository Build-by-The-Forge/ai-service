# AI Service - Build Core

Production-grade AI service for CBT + Learning Platform.

## Features

- **FastAPI** - Modern, fast web framework
- **Modular Architecture** - Clean separation of concerns
- **RAG Pipeline** - Retrieval-Augmented Generation support
- **OCR Pipeline** - Document parsing and text extraction
- **Tree Generation** - Knowledge tree generation and traversal
- **LLM Integration** - Ready for OpenAI and other LLM providers

## Setup

```bash
# Install dependencies
pip install -e .

# Setup environment
cp .env.example .env

# Run the service
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/v1/health/` - Health check
- `POST /api/v1/ai/chat` - Chat endpoint
- `POST /api/v1/ai/read` - Read endpoint
- `POST /api/v1/ai/tree` - Tree generation endpoint
- `POST /api/v1/ai/retrieve` - Retrieval endpoint
- `POST /api/v1/ai/generate` - Generation endpoint
- `POST /api/v1/ai/ocr` - OCR endpoint

## Project Structure

```
app/
├── api/          # API routes (v1, v2, etc.)
├── core/         # Config, logging, exceptions
├── db/           # Database models and sessions
├── schemas/      # Pydantic request/response models
├── services/     # Business logic (AI, RAG, OCR, Tree)
├── rag/          # RAG pipeline
└── utils/        # Utilities (time, text, id)
```

## Development

```bash
# Run tests
pytest

# Format code
black app/

# Lint
ruff check app/
```

## Deployment

This repository includes production-ready container and proxy configuration.

### Build and run locally

```bash
make up
```

### Health checks

- `http://localhost/api/v1/system/health`
- `http://localhost/api/v1/system/ready`

### Stop the stack

```bash
make down
```

### Logs

```bash
make logs
```

### Run migrations

```bash
make migrate
```
