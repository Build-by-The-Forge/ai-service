# Architecture Notes

## Authentication
- JWT-based authentication is handled through the auth package.
- Protected routes rely on dependency-based authentication and role checks.

## AI Orchestrator
- The AI orchestrator coordinates the LLM provider, retrieval service, and memory service.
- It builds prompts for chat, explanation, and question generation flows.

## OCR Pipeline
- OCR services are modularized under the OCR package.
- PDF parsing is isolated in the PDF parser service so the rest of the pipeline can remain backend-agnostic.

## RAG Pipeline
- Retrieval is currently a tenant-aware placeholder that can be replaced with a vector store implementation.
- The service is designed to be swapped without changing the orchestrator contract.

## Exam Flow
- Exams are created, listed, updated, and deleted through the exam API routes.
- The core exam services manage scoring, state, and session lifecycle.

## WebSocket Flow
- WebSocket exam routes are routed through the exam websocket module.
- They are intended to keep live exam state synchronized with the backend service layer.

## Tenant Isolation
- Tenant context is set through middleware and resolved from request headers.
- The retrieval and memory services use that context to scope data by school.

## Database Design
- SQLAlchemy models define relational entities such as exams, users, sessions, and learning memory.
- Alembic handles schema migrations for the service.
