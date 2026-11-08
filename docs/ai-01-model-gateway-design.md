# AI-01 design: chat, explain and question generation through `ModelGateway`

**Status: design for review. Nothing here is implemented.** Written against ai-service `2506dfd`. It carries out decision 3 of the master execution blueprint (backend #107): move chat, explanations and question generation off `LLMProvider` onto `ModelGateway` before Gemini is added, so the provider adapter inherits the gateway's resilience, budget and fallback controls.

**Summary:** today two model layers exist.
- `services/ai` (`LLMProvider`, `OpenAILLM`, `DeterministicLLM`) serves `/v1/ai/chat`, `/explain` and `/generate-questions`. It has no timeout, retry, circuit breaker, rate limit or budget.
- `core/models` (`ModelGateway`) has all of those, but only `services/agents/executor.py` uses it.

Reading the gateway for this move turned up **five defects** that would ship to students the moment the routes moved onto it. The first two are about where student passages go and what is logged, so they're fixed before the move, not after.

## What the gateway does today, and what's wrong

| # | Finding | Where | Consequence once routes use it |
|---|---|---|---|
| F1 | After the preferred provider and the policy's fallbacks, the loop appends **every other registered provider** | `gateway.py`, `generate` | Once Gemini is registered, a student passage can reach a provider the policy never named, and one without a data-processing record (G-33) |
| F2 | Provider error messages embed the provider's response body, and the gateway logs `str(exc)` on every failure | `providers/openai.py` `_map_http_error`; `gateway.py` | Provider response text reaches the logs. It's not ours to log, and an invalid-request error can quote the request |
| F3 | A missing `OPENAI_API_KEY` silently becomes `"sk-mock-key"` | `providers/openai.py` `__init__` | A misconfigured deployment sends real requests with a fake key and fails as a 401 per request, instead of refusing to start |
| F4 | `BudgetTracker` has no limits unless `set_budget` is called (nobody calls it), and it keeps a usage entry per scope key forever | `budget.py` | The budget control is a no-op, and keyed per request it's an unbounded in-memory map |
| F5 | Retries default to 3 per provider at a 30 s timeout, with backoff | `contracts.py` `ModelPolicy` | One chat can hold a request for more than 90 s per provider. That's longer than the backend's 60 s call deadline (`ai_service_timeout_seconds`), so the backend gives up while the provider is still paid for |

## Decisions for the reviewer

| # | Decision | Recommendation |
|---|---|---|
| 1 | Which providers a request may reach (F1) | **Only the policy's preferred provider and its named fallbacks, each on an allowlist.** The allowlist, `MODEL_PROVIDERS_APPROVED`, holds providers with a data-processing record (G-33). An unlisted provider is never called, even as a fallback. In strict environments the list must be explicit, and while G-33 is open it holds none, so student passages go to no production provider (the programme rule) |
| 2 | Provider error text (F2) | Exceptions carry the provider's **status code and error type only**, never the response body. Logs record provider, model, status, attempt and latency. The body is discarded |
| 3 | A missing key (F3) | **Settings validation refuses to start** in strict environments when an approved provider has no key. There is no placeholder key anywhere |
| 4 | Budgets (F4) | **Per-capability limits** in settings (calls, tokens and cost per request), checked before each call. Usage is keyed per request and dropped when the request ends, so nothing accumulates. A tenant-level daily budget needs shared state (Redis or PostgreSQL) and is its own later PR |
| 5 | Retries and time (F5) | **One deadline per request**, below the backend's call deadline, shared by all attempts and providers. Retries happen only for timeouts, 429 and 5xx, and only while the deadline allows. The default is 1 retry, not 3 |
| 6 | Deterministic mode | `DeterministicLLM` becomes a gateway **provider adapter** named `deterministic`, so CI and the boundary gate exercise the real gateway path. It stays refused in strict environments, as today |
| 7 | Structured output | `ModelRequest` gains an optional `structured: {kind, hints}` field. Real adapters send JSON mode (`response_format`) and ignore the hints; the deterministic adapter answers from them, as `generate_json` does now |
| 8 | What happens to `LLMProvider` | A thin `GatewayLLM` keeps the `generate` and `generate_json` interface for the orchestrator and `StructuredQuestionGenerator` during the move, then they call the gateway directly. `OpenAILLM` and the `openai` SDK pin go once nothing imports them |

## Request context

The gateway needs a `ModelInvocationContext` (tenant, execution, task, correlation) for rate limiting, budgets and logs. The routes already have it from the verified signed call:
- `tenant_id` and `user_id` come from `VerifiedCall.context`;
- `correlation_id` is the signed request's ID;
- `execution_id` is new per request.

The orchestrator takes this context as an argument. It doesn't read it from a global.

## Order of PRs

Each PR has separate commits, with the tests first.

1. **Gateway safety, before any route moves.** Covers F1 to F5, and decisions 1 to 5. The agent executor, the gateway's one current user, keeps working. Its tests show unlisted providers are never called, error bodies never reach logs, a strict start without a key fails, and the deadline bounds the attempts.
2. **The deterministic adapter and structured output** (decisions 6 and 7). Deterministic mode then runs through the gateway.
3. **The move.** `get_ai_orchestrator` builds a `GatewayLLM` over the gateway, with the request context threaded from the routes (decision 8). Chat, explain and generate-questions answer exactly as before in deterministic mode: the same prompts, grounding checks, `sources_used` and signed envelopes.
4. **Removal.** `OpenAILLM`, `LLMProvider`'s implementations and the `openai` dependency go.

Gemini (AI-01's provider work) comes after PR 4, as a new adapter behind the allowlist.

## What must not change

- **Grounding.** `_grounded` still refuses to ask the model without passages, and `check_grounded` still replaces an uncited answer.
- **Denial and provenance.** `denied_assets`, `sources_used`, the response nonce and the signed envelope are untouched.
- **Learning memory.** It stays gated (`GATE_OPEN=False`); chat continues without it.
- **Data.** No production provider receives student or institutional passages until G-33. Decision 1 makes that a property of the code, not of configuration discipline.

## Acceptance tests

| Test | PR |
|---|---|
| A request whose preferred provider fails never reaches a registered provider that isn't on the allowlist or among its named fallbacks | 1 |
| A provider error's body never appears in a log record or in the exception message | 1 |
| A strict-environment start with an approved provider and no key fails validation | 1 |
| The request deadline bounds total time across retries and fallbacks; a non-retryable error isn't retried | 1 |
| A per-capability call limit refuses the call after it; usage doesn't outlive the request | 1 |
| The deterministic adapter's chat, explain and questions match today's `DeterministicLLM` output for the same prompts | 2 |
| `/v1/ai/chat`, `/explain` and `/generate-questions` are unchanged in deterministic mode: the existing contract, signature and retrieval tests pass untouched | 3 |
| Nothing imports `OpenAILLM` or `openai` | 4 |
