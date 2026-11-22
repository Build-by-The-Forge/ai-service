# Bounded agentic study workflows: ai-service architecture

**Status: implemented foundation, unreachable by design.** Release 2 and Pro-only. Release 1's tutor stays naive RAG. Nothing here is mounted on a route, names a model provider, holds a database connection or makes a network call. Entitlement (`agentic_workflows`) is enforced later, by the backend. Learning Memory stays gated off (`GATE_OPEN` is `False`); nothing in `app/core/agentic` or `app/services/agentic` reads or writes it.

Inputs: the draft study-planner design (build-backend draft PR #123), the offline content-pack contract (#122), the implementation plan (#116), and PRD sections 22 to 26.1 and 146.7. Where this code differs from those drafts, section 11 lists it, and `docs/agentic-reviewer-decisions.md` asks the reviewer to decide.

## 1. Where things are

| Layer | Path | Imports | Purpose |
|---|---|---|---|
| Portable core | `app/core/agentic/` | stdlib, pydantic, `app.core.safe_log` | Contracts, schema validator, registry, ports, tools, runtime, checks, stores, MCP adapter, evaluation kit |
| Server adapters | `app/services/agentic/` | the core, retrieval, the gateway's model interface | Retrieval door over the chat evidence path; the model door and the runtime's model over `ModelCaller` |
| Evaluation data | `data/agent_eval/v1/` | | 24 synthetic tasks (digest-pinned) and the recorded baseline |
| Runner | `scripts/run_agent_eval.py` | | Offline; scripted policies only |

The existing `AgentExecutor`, `app/core/tools` and `app/services/tools` are untouched. They stay as they are (decision D1).

## 2. Contracts (`contracts.py`, `schema.py`)

Three Pydantic models, all `extra="forbid"`, frozen, versioned (`schema_version: 1`; another version is refused) and JSON round-trippable. No sets, datetimes, callables, provider types or server types.

- **`ToolContract`**: name, version, description, input and output schema (both required; a JSON-Schema subset that refuses what it cannot enforce, and input schemas must close `additionalProperties`), required permissions and capabilities, side effect (`read`, `propose`, `write_through_backend`), idempotency (`natural`, `keyed`, `none`), retry class and count, deadline, boundaries (`retrieval`, `backend`, `model`, `mcp`: there is no database boundary), `produces_evidence`, size limits. Consistency rules are enforced at construction: a read is natural; a proposal or write is keyed; a write needs the backend boundary; evidence needs the retrieval boundary.
- **`AgentTask`**: task and execution ids, type, goal, `CallerContext` (tenant, user, permissions, capabilities, `denied_assets`, `official_attempt_active`), required capabilities, allowed tools, evidence requirements, budget (steps, iterations, tokens, cost, retries), one deadline, execution policy, completion criteria, and the data the backend supplied (`SuppliedData`: topics, practice evidence, plan, dates).
- **`ExecutionCheckpoint`**: execution and task ids, task digest, step, state, call records (tool and model, with digests, never arguments), result references with hashes, budget consumed, retry count, cancellation, outcome code.

States: `pending`, `running`, then one of `completed`, `abstained`, `failed`, `cancelled`, `timed_out`, `budget_exhausted`, `loop_detected`, `refused`. There is no way out of an end state.

## 3. The registry: one path for every call (`registry.py`)

`ToolRegistry.execute` is the only way a tool runs: native runtime, MCP client or test. In order: unknown tool; the task's allow-list; exam mode; permissions and capabilities from the signed context; side-effect policy (write-through-backend is off by default); cancelled or past deadline; argument size, serialisability, tenant or user named in arguments, schema; idempotency (keyed tools: a repeat returns the first outcome; the same key with other arguments is a conflict; concurrent calls run once; an outcome that may be retried or stopped in flight is not recorded, so the same key settles it); the tighter of the tool's and the run's deadline; cancellation mid-call; bounded retries of transient failures only; output schema and size; one audit record.

A handler receives `ToolServices`, whose doors (retrieval, backend, model) open only if the contract declared the boundary; reaching for another fails closed. The audit record holds ids, digests, codes and counts, never arguments or results. Failures reach the model as a code, a retry flag and a fixed message; an exception's text goes nowhere (`describe` only, as in #52).

A test walks the syntax tree of both agentic packages and fails on any import of a database, network client, route or provider.

## 4. The eight tools (`tools.py`)

| Tool | Side effect | Permission | Boundaries | How |
|---|---|---|---|---|
| `search_course_material` | read | `content:read` | retrieval | `gather` (tenant, scope, `denied_assets`, recorder); refuses without `denied_assets` |
| `explain_concept` | read | `content:read` | retrieval, model | passages in; model writes; `check_grounded`, else `NOT_COVERED` |
| `generate_practice` | propose | `assessment:generate` | retrieval, model | model drafts; a question with no real source or not exactly one key is dropped |
| `evaluate_response` | read | `content:read` | none | normalised comparison with the key; provisional; no score field |
| `identify_knowledge_gaps` | read | `progress:read` | none | recency-weighted error rate from supplied evidence; thin topics reported, not ranked |
| `create_study_plan` | propose | `plan:propose` | none | deterministic schedule; ends before the exam; requires student approval |
| `recommend_next_activity` | read | `plan:read` | none | overdue item, next item, weakest topic, or nothing |
| `update_learning_progress` | propose | `progress:propose` | backend | proposal only, on evidence the request supplied; mastery and scores are not fields |

All require capability `agentic_workflows`. BUILD computes figures; the model narrates.

## 5. The runtime (`runtime.py`)

`AgentRuntime.run(task, control)` and `resume(task, control)`. Each step the `AgentModel` returns one `Decision`: call one tool, or finish.

- **Checkpoints** are saved before the run goes on. `resume` rebuilds from the checkpoint and stored results, cannot reset the budget, refuses a different task, user or tenant, detects a changed stored result, and re-issues an interrupted call under the same idempotency key. A finished run resumes to its result without any call.
- **Budgets**: steps, iterations (tool rounds), tokens (including tool-internal model tokens), cost, retries. Reaching one ends the run `budget_exhausted` with the gathered tool results marked partial.
- **One deadline and one cancellation** (`RunControl`) reach every model and tool call. Cancelling mid-call stops it and records the effect as unknown. A cancel that arrives with a final answer wins.
- **Loops**: the same tool and arguments N times (default 3; the Nth is not executed) or N tool rounds in a row that return nothing new (default 3).
- **Retries** reuse keys, so no side effect is duplicated (tested with a backend that accepts and then loses the reply, and a process killed mid-call).
- **Exam mode**: refused before any call, and again on resume. The flag is excluded from the task digest.
- **Foundation checks** (`checks.py`): never invent a source (answers and `source_refs` arguments must be chunks retrieval returned in this run), abstain on thin evidence (counted by source version), never invent a grade or mastery (a figure presented as one must be a tool-computed number).
- **Sanitised**: results carry a code and a fixed message; model, store and observer exceptions are logged by type and place.

| Terminal state | Student-visible meaning |
|---|---|
| `completed` | A grounded, checked answer |
| `abstained` | `NOT_COVERED` |
| `budget_exhausted`, `loop_detected`, `timed_out` | Partial tool results if any, never a half-finished claim |
| `failed` | A fixed message; the answer, if any, is withheld |
| `cancelled`, `refused` | Nothing was produced |

The runtime's model door is `ModelCallerAgentModel` (`app/services/agentic/model.py`): one `generate_json` call, kind `agent_step`, through the gateway's one model interface, so approval, deadlines, budgets and sanitised errors apply as for chat. The deterministic offline model cannot drive an agent and fails safe.

## 6. MCP (`mcp.py`)

An in-process JSON-RPC subset (`tools/list`, `tools/call`) with a transport Protocol. **Export**: identity is fixed when the server is built and no parameter can set it; calls run `registry.execute`. **Consume**: an external tool is registered only through an integrator policy (`mcp_` name, the integrator's description and permissions, read or propose only); its schema is checked and tightened; its output is untrusted, size-checked and never evidence. Native execution stays the default. No real transport exists; stdio and HTTP need their own review.

## 7. Evaluation (`evalkit/`)

24 original synthetic tasks in eight categories (tool selection, argument accuracy, loop detection, abstention, budget exhaustion, cancellation, permission denial, replay and idempotency), pinned by digest. Scoring is deterministic from run facts: state, code, tool selection, argument accuracy, forbidden tools, effects, citations, efficiency. A task passes when every component is 1. Traces record decisions and tool results; `ReplayAgentModel` hands decisions back by step, so any run, including a resumed one, reproduces exactly.

Baseline (recorded, no bar set; `data/agent_eval/v1/baseline.json`):

| Policy | Pass rate | Mean |
|---|---|---|
| reference (scripted) | 1.000 | 1.000 |
| looping | 0.375 | 0.676 |
| naive | 0.125 | 0.641 |
| overreach | 0.125 | 0.634 |

The negative controls show the scoring discriminates; the pass rate is the sharper measure. **Plugging a real model in later:** write a reviewed script that has provider approval, a budget and network allowance, and pass `run_suite` a factory `lambda task: ModelCallerAgentModel(gateway_llm, capability, invocation_for(task))`. Record its traces, replay them for free, and set bars only after a baseline on real material.

## 8. Tenant isolation, provenance, scope

Tenant and user come only from `CallerContext`; arguments naming another tenant or user are refused. Retrieval goes through the chat evidence path with `denied_assets`, so scope, denied documents and the recorder (no passage without its source version) apply. Citations carry `run_number` and `attempt`. A source the backend revokes mid-run is the backend's provenance check on the whole answer, as for chat (`ANSWER_WITHHELD`).

## 9. Online and on a device

Reusable on a device as written: contracts, schema validator, registry, tools, checks, grounding, runtime logic, evaluation kit. Server-shaped, recorded so a device implementation knows what to replace:

| Assumption | Where | On a device |
|---|---|---|
| `asyncio` tasks, events, locks | registry, runtime | any cooperative scheduler with timeouts and cancellation |
| Wall-clock `deadline_epoch` | `AgentTask` | the lease's monotonic clock (offline contract section 5) |
| In-memory stores | stores, idempotency store | a local database; the idempotency store must survive a restart |
| Estimated tokens (4 characters) | `ports.estimate_tokens` | the local runtime's real counts |
| Pydantic | contracts | a port of the schemas; the JSON is the contract |
| `ModelCallerAdapter`, `RetrievalEvidencePort` | `app/services/agentic` | local model and pack retrieval implementing the same Protocols |
| Backend proposal | `BackendPort` | queued locally, applied at sync, provisional until validated |

## 10. Limits

Tokens are estimated where the model interface reports no usage. The figure check is textual. Cancelled and timed-out runs are terminal. Only in-memory checkpoint, result and idempotency stores exist, so a real restart loses the idempotency record (the backend's keyed dedupe is what then protects a proposal). No router (naive versus agentic) and no planner `AgentDefinition` exist: those need the entitlement design.

## 11. Differences from the draft specs

| Draft | This code | Why |
|---|---|---|
| Side effects `READ_ONLY`, `ANALYTICAL`, `PROPOSAL`, `MUTATING` (existing enum) | `read`, `propose`, `write_through_backend` (as briefed) | `ANALYTICAL` is a read; the write class is off by default |
| `search_course_material` wraps `content.search` | calls `evidence.gather` directly, adds sources | the same retrieval path with the recorder; no second tool plane |
| `generate_practice` wraps `draft.question` | drafts through the model door and drops unsourced questions | `draft.question` returns a fixed template, not drafted from passages |
| Plan items carry `source_refs` | optional, copied from arguments, and the runtime refuses refs retrieval did not return | a plan tool cannot verify them itself |
| `update_learning_progress` evidence is "real practice rows" | evidence refs must be supplied attempt refs; so only practice-backed changes are possible | decision D7 |
| Planner ceilings 6 steps, 8 calls, 45 s | defaults 8 steps, 6 iterations; set per task | to be set from measurement |
| Audit trail is the `AgentEvent` stream | `ToolAuditRecord` per call and the checkpoint's call records | the old stream is unused; D12 |
| Permissions `progress:read`, `progress:propose`, `plan:read`, `plan:propose` (proposed) | used as proposed | PRD section 24's READ/WRITE categories are not mapped, D3 |
| Open decision 5: assessment-history inputs | `SuppliedData` in the signed request | a proposal for the backend contract, D8 |
