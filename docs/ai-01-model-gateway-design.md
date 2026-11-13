# AI-01 design: chat, explain and question generation through `ModelGateway`

**Status:** the design was approved in #48.
- **Merged:** PR 1, gateway safety (#49), and its cleanup-bound follow-up (#51).
- **Merged:** PR 2, the deterministic adapter and structured output (#50).
- **In review:** PR 3, the route migration, implementing the decisions below.

Originally written against ai-service `2506dfd`. It carries out decision 3 of the master execution blueprint (backend #107): move chat, explanations and question generation off `LLMProvider` onto `ModelGateway` before Gemini is added, so the provider adapter inherits the gateway's resilience, budget and fallback controls.

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
4. **AI-10's generator onto the gateway**, scoped separately (decision P3-5). `CBTAIGenerator`, used only by the unrouted `CBTFusionEngine`, moves off `LLMProvider`.
5. **Removal.** `OpenAILLM`, `DeterministicLLM`, `LLMProvider`, the dead `OpenAIEmbeddingAdapter` and the `openai` dependency go.
   - **Condition:** the PR includes an explicit inventory: a repository-wide search, with its output, showing that no required code imports or invokes a direct provider path.
   - It can't merge while a required path, such as AI-10's generator, still depends on the old layer.

Gemini (AI-01's provider work) comes after PR 4, as a new adapter behind the allowlist.

## PR 3: the reviewer's decisions (2026-11-12)

These were decided before any PR 3 code, and the implementation follows them. PR 3 moves **chat, explain and question generation** onto the gateway, and nothing else.

| # | Question | Decision |
|---|---|---|
| P3-1 | The gateway's errors at the route | A gateway execution failure is a **503** with a fixed body. Invalid client input keeps its **4xx**. A malformed request is not a provider outage |
| P3-2 | `MODEL_PROVIDERS_APPROVED` empty in staging or production | **Fail closed**: no external provider is called, and chat, explain and questions answer 503. There is no fallback to an unapproved provider and no implicit approval by default. Deterministic mode stays for development and tests only, and is never a production escape hatch |
| P3-3 | Per-capability budgets | **Two model calls per request** at most, plus token and cost caps, configurable per capability with conservative defaults. The accounting rules are below |
| P3-4 | Request context | It comes from the **verified signed call**. Nothing in the request body supplies identity or tenant |
| P3-5 | Direct provider paths | **Every model execution passes through the gateway.** `OpenAIEmbeddingAdapter` is removed. AI-10's `CBTAIGenerator` moves to the gateway in its own PR, and until then it is a recorded legacy path |

### P3-1: errors

| Failure | Status | Body |
|---|---|---|
| Any `ModelGatewayError` during execution. This covers a provider that is unavailable or timed out, a rate limit, an open circuit breaker, no eligible provider, exhausted fallbacks, an exhausted budget, the request deadline, a provider refusing our request (`InvalidRequestError`), or a missing or rejected key | **503** | Fixed: `{"success": false, "code": "AI_SERVICE_UNAVAILABLE", "message": "AI service unavailable", "data": {}}`. The backend already treats an AI-service 503 as `AI_SERVICE_UNAVAILABLE` |
| An invalid request body | **422**, as today | The route validates the body before any model call |
| A question-generation answer that isn't one JSON object (`InvalidStructuredOutputError`) | **200**, as today | The questions are counted as discarded, and the existing contract is unchanged |

The body never carries a provider response, an exception message, a credential, a prompt or a passage.

The log line has safe structured fields only: exception type, `execution_id`, capability, the gateway's provider name, HTTP status when there is one, `request_id` and attempt. It follows #52's rule.

### P3-3: budget accounting

| Rule | How |
|---|---|
| 1. The first attempt and the one retry share the request's call allowance | The gateway reserves **per attempt** against the request's scope (`max_model_calls=2`) |
| 2. Only explicitly retryable transient failures are retried | `ModelRetryPolicy.is_retryable`: timeouts, 429 and 5xx, from the error's class, never its text |
| 3. No retry after a validation error, an exhausted budget, a cancellation or an expired deadline | `BudgetExceededError` is raised by the reservation, before the call. A validation or structured-output failure is non-retryable. Cancellation propagates. A retry whose backoff would outlast the deadline isn't attempted |
| 4. Token and cost budgets are reserved before execution | `BudgetTracker.reserve`: prompt estimate plus `max_tokens`, at the adapter's price |
| 5. The reservation is reconciled against actual usage when the provider reports it | `reconcile(reservation, response.usage)` |
| 6. An unused reservation is released however execution ends, cancellation and exceptions included | `release` on every failure path, including a call left behind after its request ended (#51), whose late settlement is ignored |
| 7. Concurrent calls can't each exceed the request's budget | The check and the hold happen with no `await` between them |
| One overall deadline for the attempt and the retry | `deadline_seconds=45` spans both (#49 F5), so two bounded calls can't double the request's time |

**Defaults** come from settings and can be changed per capability. No capability is unlimited, and no budget is implied by a fallback.

| Capability | Calls | Tokens | Cost |
|---|---|---|---|
| `chat` | 2 | 12,000 | $0.05 |
| `explain` | 2 | 12,000 | $0.05 |
| `generate_questions` | 2 | 16,000 | $0.08 |

The budget scope is the request's `execution_id`. It is set when the request starts and dropped when it ends (`BudgetTracker.scope`), so nothing accumulates across requests. A tenant-level daily budget needs shared state and is later work (design decision 4).

### P3-4: request context

| Field | Source |
|---|---|
| `tenant_id` | The verified signed call's context |
| `correlation_id` | The signed request's ID |
| `execution_id` | New for each request. It is the budget scope and the parent of every attempt |
| `task_id` | The capability: `chat`, `explain` or `generate_questions` |
| Attempt ID | New for **each attempt**, and logged with the `execution_id` and `correlation_id`. A retry is separately traceable under the same parent |

The orchestrator takes the context as an argument. It doesn't read it from a global.

### Deterministic mode doesn't change

The gateway's model layer sends what `OpenAILLM` sent: a fixed system message, then the prompt as the user message. The deterministic adapter answers the last user message, so every pinned answer in `tests/unit/test_deterministic_answers.py` still holds.

### P3-5: remaining direct provider paths

| Path | Evidence | Disposition |
|---|---|---|
| `OpenAIEmbeddingAdapter` (`app/services/rag/embedder.py`) | Never constructed: no import, injection, setting, environment variable, workflow, Dockerfile or doc refers to it. Production embeds with local ONNX MiniLM (`EMBEDDING_MODEL_DIR`, fetched in the Dockerfile) | **Removed** in the removal PR |
| `CBTAIGenerator` (`app/services/generation/cbt_ai_generator.py`) | Reached only from `CBTFusionEngine` (AI-10), which no route calls, and from unit tests | **A known legacy path.** It moves to the gateway in its own PR, because AI-10 is on the roadmap (stage 6). It is not deleted, and it is not an approved second boundary |
| `OpenAILLM`, `DeterministicLLM`, `LLMProvider` | After PR 3, used only by `CBTAIGenerator` and tests | Removed in the removal PR, once the AI-10 migration has merged |
| The `openai` SDK (`requirements.txt`, `pyproject.toml`) | Used by `OpenAILLM` and the dead embedder | Removed last, under the inventory condition above |

### The related data-flow findings

Two exception-text paths that cross a boundary were found during #52. They are separate from AI-01:
- **#53:** tool results going into the agent's context;
- **#54:** ingestion failure callbacks to the backend.

The reviewer has decided their safe contracts.

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
