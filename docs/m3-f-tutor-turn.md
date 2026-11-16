# M3-F: the AI Tutor turn, contract freeze

**Status: proposed for review.** Written against ai-service `2c76c93` and build-backend `e9067ed`. The design is build-backend `docs/m3-f-design.md` (#81, amended in #83). The mobile contract is build-backend #90 (`contracts/mobile/openapi.yaml`).

**Summary:** one signed route answers one conversation turn. The backend owns the conversation, its IDs, its keys and its storage; the AI service gets the question, at most three of the student's earlier questions, and the denial list, and returns an answer, its structured citations and every source it retrieved. Nothing of the turn is kept here.

## Decisions for the reviewer

| # | Decision | Recommendation |
|---|---|---|
| T1 | What history the turn sees | **The student's own earlier questions only, at most three, oldest first.** Never earlier answers: an answer is source-derived text, and carrying it into a new turn would let a revoked document shape the new answer without being in its `sources_used`. Questions are the student's own text, so the provenance check stays on `sources_used` alone |
| T2 | What history is used for | **The prompt only.** Retrieval still runs on the current message, as measured in P3-5. Rewriting a follow-up into a retrieval query changes retrieval quality and needs its own measured PR. Until then, a follow-up the passages for its own words don't cover gets the "not covered" reply, which is safe |
| T3 | When the backend sends history | **Only after the mobile contract's "each turn is answered on its own" sentence is amended**, in the backend's contract PR. Until then the backend sends an empty list, and the AI side behaves exactly as with no history |
| T4 | A new route or the legacy `/v1/ai/chat` | **A new route**, `POST /v1/ai/tutor/turn`. `/v1/ai/chat` keeps serving the legacy backend route until it retires (design decision 4), and separately stops returning context passages and memory (G-18, its own PR) |
| T5 | The model budget | The `chat` capability's, unchanged: one turn is one grounded answer, the same work |

## Request

`POST /v1/ai/tutor/turn`, Service Signature v1 with a user context carrying `ai.use`. Identity, tenant and correlation come only from the signed context. Unknown fields are refused with 422, so a body naming a tenant, user, conversation or turn fails.

| Field | Type | Rule |
|---|---|---|
| `message` | string | 1 to 4,000 characters. The question being answered, and the only retrieval query |
| `earlier_questions` | list of strings | Optional, default empty. At most 3, each 1 to 4,000 characters, oldest first. The student's own earlier questions in this conversation, never answers (T1). Prompt context only (T2) |
| `denied_assets` | list of asset IDs | Required, may be empty. At most 1,000 (M3-D). Excluded before ranking, whatever their scope |
| `response_nonce` | string | Required. The backend's single-use nonce, covered by the response signature |

There's no conversation ID, turn number or idempotency key: those are the backend's.

## Response

`200`, signed, the M3-D sourced envelope:

```
{ "success": true, "message": "...",
  "data": { "answer": str, "grounded": bool, "citations": [Citation] },
  "sources_used": [ { "asset_id": str, "run_number": int, "attempt": int } ] }
```

`Citation`: `number`, `chunk_id`, `document_id`, `page_start` (nullable), `page_end` (nullable), `excerpt`, `run_number`, `attempt`.

| Field | Guarantee |
|---|---|
| `answer` | Model text with `[n]` markers. Nothing is appended to it: citation details stay in `citations`. The prompt shows each passage as `[n] (pages) text`, with no document or chunk identifier, so the model has none to copy into the answer |
| `grounded` | `false` means `answer` is the fixed "not covered" reply and `citations` is empty. The model isn't asked when nothing was retrieved |
| `citations` | Exactly the passages the answer cites, one per distinct `[n]`, ordered by number. An answer citing any number it wasn't given, or citing nothing, is replaced by "not covered" |
| `excerpt` | The cited chunk's stored text, verbatim and in full. Never model output. The backend cuts it to 500 characters |
| `document_id`, `run_number`, `attempt` | The exact document version the cited chunk came from. Every citation's version is in `sources_used` |
| `sources_used` | Every document version retrieval returned for the turn, cited or not, from the retrieval recorder, never the model. Empty only when nothing was retrieved |

**Not returned**, ever: uncited passages (no `context_used`), learning memory (no `memory_used`), retrieval scores, storage paths, tenant or user IDs, provider text.

## Retrieval

The turn retrieves through the same evidence boundary as chat: the verified tenant only, documents with a retrievable scope only (M3-D, `asset_scopes`), `denied_assets` excluded before ranking, top 5 passages, and every passage's version recorded in `sources_used` before it can reach the prompt.

## Failures

| Status | Body | When | Backend maps it to |
|---|---|---|---|
| 401 / 403 | signature failure | Bad or replayed signature, no user context, no `ai.use` | `AI_UPSTREAM_ERROR` (a backend fault) |
| 422 | validation error | Message empty or too long, more than 3 earlier questions, an unknown field, a bad asset ID, more than 1,000 denied assets | `AI_UPSTREAM_ERROR`: the backend validated first |
| 503 | `AI_SERVICE_UNAVAILABLE`, fixed body | Model failure, timeout, rate limit, open circuit, no approved provider, budget exhausted | `AI_UNAVAILABLE` |
| 500 | fixed body | Anything else, logged by exception type only | `AI_UPSTREAM_ERROR` |

`ANSWER_WITHHELD` and `TURN_IN_PROGRESS` are the backend's alone: the first follows its provenance check of `sources_used` and citations, the second its turn keys. A response without `sources_used`, or with a citation whose version isn't in it, is a broken contract the backend withholds.

## What the AI service keeps

Nothing of the turn. No conversation state, no question or answer text, no memory write (learning memory isn't read either, and `GATE_OPEN` stays `False`). The replay guard keeps the request's nonce, as for every signed call. No message or history text is logged.

## Delivery

| PR | Closes |
|---|---|
| This one: the turn route, with the request, response and guarantees above | The AI side of G-17 |
| `/v1/ai/chat` stops returning `context_used` and `memory_used` | The AI side of G-18 |
| Backend (paired): routes, migration, provenance on every turn | G-17, G-04 (backend part) |
