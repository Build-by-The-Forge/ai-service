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
- The `RetrievalService` resolves current tenant context and returns school-scoped documents.

## OCR Pipeline
- OCR services are modularized under the OCR package.
- PDF parsing is isolated in the PDF parser service so the rest of the pipeline can remain backend-agnostic.
- `OCRService` combines the OCR provider, text cleaner, and PDF parser to normalize raw text.

## Exam Flow
- Exams are created, listed, updated, and deleted through the exam API routes.
- The core exam services manage scoring, state, and session lifecycle.
- `ExamEngine` coordinates tenant-aware question loading and session creation.

## Tenant Isolation
- Tenant context is set through middleware and resolved from request headers.
- The retrieval and memory services use that context to scope data by school.
- The `ExamEngine` and `MemoryService` both depend on `tenant.get_school_id()` to keep requests isolated.

## WebSocket Flow
- WebSocket exam routes are routed through the exam websocket module.
- They are intended to keep live exam state synchronized with the backend service layer.

## Tenant Isolation
- Tenant context is set through middleware and resolved from request headers.
- The retrieval and memory services use that context to scope data by school.

## Database Design
- SQLAlchemy models define relational entities such as exams, users, sessions, and learning memory.
- Alembic handles schema migrations for the service.
