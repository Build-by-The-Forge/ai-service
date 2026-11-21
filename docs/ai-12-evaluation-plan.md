# AI-12 evaluation plan: the 100-task suite, metrics, gates and the paid-call checklist

Status: plan. Built at $0: no provider key and no funded account exist, and no paid
call may be made until the checklist in section 6 is satisfied. All threshold values are
**"to be set from the baseline, then approved"**; nothing here picks a model or assigns a
role. Context: the offline benchmark (`benchmarks/offline/`), the approved master plan
(Release 2 capability; Android 14+, 4 GB RAM minimum, English only, one active model at
a time on the minimum tier) and PRD §146.

## 1. What the suite must decide

1. Is a **Free online model** good enough: grounded answers, correct citations, reliable
   abstention?
2. Is the **smallest local model** (Qwen3.5 0.8B first) good enough for Free: the same
   essentials, plus a minimum abstention threshold, before it ships?
3. Which provider and model give the best **cost per validated task** for paid tiers?

The same suite runs against local models (`bench.py`, `litert_bench.py`) and hosted
models (`hosted_bench.py`), so the numbers are comparable on quality. They are not
comparable on latency or cost across those two kinds, and reports must say so.

## 2. The 100-task suite

Built on the 43-task original set (dataset A), which stays as the regression core in CI.
All passages are original or synthetic; real course PDFs (dataset B) stay local and are
never sent to a provider.

| Block | Tasks | Composition |
|---|---:|---|
| **RAG (grounded answering)** | 50 | 14 single-passage answerable (10 exist) · 10 multi-passage answerable (facts in two passages) · 8 partial evidence (the passage is on topic but lacks the asked detail; the right answer abstains or states the gap) · 12 unanswerable (6 exist) · 4 conflicting passages (two sources disagree; the answer must say so) · 2 long-context (all passages in the prompt) |
| **Orchestration** | 30 | 10 tool selection and arguments (6 exist) · 6 no-tool-needed (answer or greet without calling a tool) · 6 route decisions (answer from passages / call a tool / abstain / defer to the online path) · 4 bounded multi-step (two ordered tool calls with a hard step limit) · 4 refusal and scope (out-of-scope or unsafe requests handled without a tool call) |
| **Question generation** | 20 | 8 multiple-choice from one passage (1 exists) · 4 from two passages · 4 short-answer with a marking guide · 4 at stated difficulty levels. Each must be answerable from the passage, have one defensible key, plausible distractors and a source id |

Rules: English only for the Release 2 gate. The existing Yoruba, Hausa and Igbo tasks
stay as an exploratory block, outside the 100 and outside every gate, until native
speakers have reviewed them.

### Authoring and human review

- Every task has an **expected outcome** written by one person and **reviewed by a
  second**: answer facts (with accepted alternatives), the supporting passage ids, the
  correct action (answer, tool, abstain), or for question generation a review rubric.
- Disagreements are resolved by a named third reviewer and logged. A task is `verified`
  only with two named reviewers; `draft` tasks are never scored (the same rule as
  dataset B).
- Model-drafted tasks are allowed as drafts only. A model's output is never the ground
  truth for scoring that model or any other.
- Question-generation outputs are scored on a human rubric (correct key, answerable from
  the passage, distractor quality, clarity) on a sample of at least 30 percent per model,
  with the same two-reviewer rule. Automatic checks (valid JSON, four options, source id
  present, key in range) run on all.
- The suite is frozen with a content hash before the first scored run; changes create a
  new version and invalidate earlier comparisons.

## 3. Metrics

| Metric | Definition | Already measured by |
|---|---|---|
| Schema accuracy | share of replies that parse and validate against the task's JSON Schema, split by instruction-only and constrained decoding | `scoring.score_*`, `summary.*.schema_ok_rate` |
| Hallucination | share of unanswerable or partial-evidence tasks where the model answers instead of abstaining, plus answers containing facts absent from the passages | `en_hallucination_rate` (facts check to extend) |
| Citation correctness | precision (cited ids that support the answer), recall (supporting ids cited), and faithful (correct answer, only supporting ids cited) | `citation_precision`, `faithful` |
| Abstention | true abstention on unanswerable tasks, and false abstention on answerable ones (both reported; one alone can be gamed) | `en_abstention_rate`, `en_false_abstention_rate` |
| Orchestration accuracy | correct tool and arguments, correct no-tool decisions, step limit respected | `tool_call_accuracy` (route and multi-step to add) |
| Generation validity | automatic structure checks plus the human rubric score | `mcq_ok`; rubric sheet to add |
| Consistency | share of repeated sampled runs agreeing with the greedy run | `sampled_consistency` |
| Cost per validated task | total cost divided by tasks that are correct, schema-valid and faithful (not cost per request) | `hosted_bench.py` records cost per request; division in the report |
| Latency | TTFT and total, median and p90 (hosted: network included; local: device-measured) | `ttft_*`, `latency_s` |

Reporting rules: report counts as well as rates (n of N), because 100 tasks cannot
separate models that differ by a few points; give no confidence claim from fewer than
three repeats; keep local and hosted results in separate tables.

## 4. Gates

Thresholds are set **after** a baseline run, proposed in the report with the evidence,
then approved by the user and recorded here. They are frozen before candidates are
compared, so a threshold is never adjusted to let a favoured model pass.

| Gate | Applies to | Must show | Threshold |
|---|---|---|---|
| G1 grounded answer | Free online model | answer correctness on answerable RAG tasks | to be set from the baseline, then approved |
| G2 citation correctness | Free online model | citation precision and faithful rate | to be set from the baseline, then approved |
| G3 reliable abstention | Free online model | true abstention on unanswerable and partial-evidence tasks, and a ceiling on false abstention | to be set from the baseline, then approved |
| G4 schema reliability | all candidates | schema accuracy under the deployment mode (constrained decoding) | to be set from the baseline, then approved |
| L1 to L3 | smallest local model | the same essentials as G1 to G3 | to be set from the baseline, then approved |
| **L4 minimum abstention** | smallest local model | abstention at or above a **minimum bar before it ships to Free**, with hallucination below a ceiling | to be set from the baseline, then approved |
| L5 device fit | smallest local model | peak RAM, cold start, TTFT and tokens per second on the 4 GB tier, battery and thermal from device runs | to be set from device runs, then approved |

A model that passes quality but not L5 does not ship on the 4 GB tier; one that fails
L4 does not ship to Free whatever its other scores.

## 5. Order of work (all at $0)

1. Finish the desktop baseline for the local models (running now, one at a time).
2. Grow dataset A to the 100-task suite and review it (two reviewers per task).
3. Run the baseline, propose thresholds, get approval, freeze them.
4. Device runs on 4 GB, 6 GB and 8 GB+ phones (`device/DEVICE_PROCEDURE.md`).
5. Dry-run `hosted_bench.py` for each candidate provider to size the worst-case spend.
6. Stop. A paid run waits for section 6.

## 6. Authorisation checklist before any paid call

Every box must be ticked, by the user or with the user's written approval in the PR or
issue, before `--confirm-spend` is used. There is no standing approval; each run needs
its own.

- [ ] The wallet is funded and the user has approved this provider and this model id.
- [ ] The API key sits in an environment variable or a file **outside the repository**;
      it was never pasted into a chat, a commit, a log or a result file.
- [ ] The price table is a file outside the code, `verified` is true, and `source`
      (the provider's pricing page URL) and `verified_on` (a date within 7 days) are set.
- [ ] The provider's data terms (retention, training on inputs) are read and accepted by
      the user.
- [ ] The dataset is marked synthetic with no third-party material (`DATASET.json`);
      no student data and no dataset B text is in the prompt.
- [ ] A dry run was reviewed: request count and worst-case cost are stated in the
      approval.
- [ ] `--max-usd`, `--max-requests` and `--max-tokens-per-request` are set, and the
      USD cap is no more than the approved amount.
- [ ] The base URL is https and belongs to the approved provider.
- [ ] The result file location is decided and is not committed with any key material.
- [ ] After the run: the spend recorded in the result is compared with the provider's
      dashboard, the stop reason is read, and the wallet balance is reported.

## 7. What exists and what is pending

| Piece | State |
|---|---|
| 43-task dataset A, scorers, runners (llama.cpp, LiteRT-LM, hosted) | built and self-tested |
| Hosted mode caps, key handling, synthetic-only, https, price-table gate | built; tested against a fake server only |
| 100-task suite and two-reviewer review | to do |
| Gate thresholds | to be set from the baseline, then approved |
| Paid calls | none made; none possible (no key, no funding, no flag) |
