# Topic coverage for AI-10's exam generator

Status: measured 2026-11-16. Required before AI-10 (`cbt_exam`) gets any
route (review of ai-service #62). This change adds no route.

## The gap

Retrieval returns its top passages even for a topic they don't cover, and
its scores can't tell (ADR 001). The support checks keep any question its
cited passages support, whatever it is about. So before this change:

- an exam on a topic the material doesn't teach still reached the model,
  with unrelated passages;
- a question those passages supported was kept, even if it was about
  another topic.

## The checks

Both are in `app/services/generation/topic_coverage.py`. They are lexical
and deterministic, and they apply only to AI-10's exam generator. The routed
practice-question path is unchanged.

| Stage | Check | If it fails |
|---|---|---|
| Before generation | At least 60% of the topic's content terms appear in one retrieved passage | The model isn't asked and nothing is sent (`grounded: false`) |
| After generation | At least 40% of the topic's content terms appear in a passage the question cites | The question is discarded (`rejected: off_topic`) |

## The evaluation set (`data/topic_coverage/v1`)

The set is written over the retrieval evaluation corpus and its 48
past-question distractors (`data/retrieval_eval/v1`). All of it is original
course-style text written for this purpose.

- **192 topics**, labelled covered or not:
  - 92 covered: 58 named as in the notes, 22 paraphrased and 12 broad;
  - 70 hard near misses: same course and subject, often sharing the notes'
    words, but not taught (for example facilitated diffusion, capacitors in
    series and parallel, or the standard error of the mean);
  - 30 out-of-corpus subjects.

  Every covered topic names its page and quotes a span from it. Every
  uncovered one says what is missing.
- **298 (topic, question) pairs** from 62 four-option questions: 111 on
  topic and 187 off topic. Every question passes the support and exam
  checks against the page it cites, so every off-topic pair is the hard
  case: supported, but not about the topic. Each question's off-topic
  pairs include a topic from the same document, a same-course near miss and
  an out-of-corpus subject.
- **Splits.** Topics are split into dev and test, and a labelled pair never
  mixes the two. Thresholds were chosen on dev only, by the lowest balanced
  error (FP rate + FN rate, the stricter on a tie). Test was held out and is
  only reported.

**Status: unreviewed.** Claude drafted the labels, and no person has
reviewed them yet. The same author wrote the labels and the candidate
checks. The held-out split guards against tuning, not against the author's
blind spots.

## Method

`scripts/evaluate_topic_coverage.py` runs the production path:

1. The corpus and distractors (158 chunks) are chunked, embedded with
   all-MiniLM-L6-v2 and indexed in `PgChunkIndex` (pgvector).
2. Each topic's passages come from `evidence.gather` with the generator's own
   query, `"<course> <topic>"`.
3. The question side needs no index, because each question cites a page.

A false positive (FP) accepts an uncovered topic or an off-topic question.
A false negative (FN) rejects a covered topic or an on-topic question.
Intervals are Wilson 95%.

## Results

The baseline (behaviour before this change) is shown first, then the
candidates. Each candidate's threshold was chosen on dev; its test row is
the held-out result.

**Before generation: does the material cover the topic?**

| Check | Split | FP | FN |
|---|---|---|---|
| Today: retrieval returned anything | test | 50/50 (100%, 92.9–100%) | 0/45 (0%, 0–7.9%) |
| Dense score, best passage ≥ 0.387 | test | 6/50 (12.0%, 5.6–23.8%) | 21/45 (46.7%, 32.9–60.9%) |
| Terms in all passages ≥ 0.6 | test | 4/50 (8.0%, 3.1–18.8%) | 16/45 (35.6%, 23.2–50.2%) |
| **Terms in one passage ≥ 0.6 (selected)** | dev | 3/50 (6.0%, 2.1–16.2%) | 12/47 (25.5%, 15.2–39.5%) |
| **Terms in one passage ≥ 0.6 (selected)** | **test** | **3/50 (6.0%, 2.1–16.2%)** | **16/45 (35.6%, 23.2–50.2%)** |

On test, the selected check:

- accepted 3 of 35 near misses and none of 15 out-of-corpus topics;
- rejected 8 of 28 directly named, 7 of 11 paraphrased and 1 of 6 broad
  topics.

**After generation: is a supported question on the topic?**

| Check | Split | FP | FN |
|---|---|---|---|
| Today: support checks only | test | 85/85 (100%, 95.7–100%) | 0/48 (0%, 0–7.4%) |
| Terms in the question ≥ 0.25 | test | 16/85 (18.8%, 11.9–28.4%) | 17/48 (35.4%, 23.4–49.6%) |
| Terms in question + anchor paragraph ≥ 0.4 | test | 9/85 (10.6%, 5.7–18.9%) | 13/48 (27.1%, 16.6–41.0%) |
| **Terms in a cited passage ≥ 0.4 (selected)** | dev | 19/102 (18.6%, 12.3–27.3%) | 14/63 (22.2%, 13.7–33.9%) |
| **Terms in a cited passage ≥ 0.4 (selected)** | **test** | **14/85 (16.5%, 10.1–25.8%)** | **6/48 (12.5%, 5.9–24.7%)** |

On test, the selected check passed these off-topic questions:

- 8 of 28 from the same document;
- 5 of 28 for a near miss;
- none of 28 for an out-of-corpus subject.

**Both checks together**, on the question pairs (a pair is kept only if its
topic passes coverage and the question passes relevance):

| Split | FP: off-topic question kept | FN: on-topic question lost |
|---|---|---|
| Today | 85/85 (100%) | 0/48 (0%) |
| dev | 8/102 (7.8%, 4.0–14.7%) | 28/63 (44.4%, 32.9–56.7%) |
| **test** | **8/85 (9.4%, 4.9–17.5%)** | **17/48 (35.4%, 23.4–49.6%)** |

## What this does and doesn't show

- The checks cut the rate of off-topic exam questions on this set from 100%
  to about 9% (95% CI 5–17%). They don't make it zero. A near miss made of
  the notes' words (capacitors in series and parallel) passes coverage, and
  a question citing a passage that also mentions the topic passes relevance.
- They fail safe, but often. About a third of covered topics are refused
  when they are paraphrased ("Averages", "Dispersion"), when they are broad
  and the notes never name them ("Thermochemistry"), or when retrieval's
  top five passages miss the right page. The student is told that the
  material doesn't cover the topic. No ungrounded exam is produced.
- Most of those refusals come from retrieval, not the check. On test, the
  page that teaches the topic was among the top five passages for only 29
  of 45 covered topics. Of the 16 false negatives, 11 had no such page, and
  5 are the check's own: words it doesn't match, such as "laws" against
  "law". Better retrieval recall would lower the coverage FN rate without
  changing the check.
- The dense score was measured and did worse than the lexical check on both
  error kinds. That agrees with ADR 001.
- These numbers describe this set, which is small, self-authored and
  unreviewed. They aren't a measurement on real course documents. A
  real-document check (as in the P3-5 tier-2 evaluation) and a human review
  of the labels are follow-ups.

## Regression in CI

- **Unit tests**, with no index:
  - `test_topic_coverage_dataset.py` checks the set's consistency;
  - `test_topic_coverage.py` checks the thresholds against the baseline.
    It also reproduces the relevance measurement exactly, and holds the
    named regression cases and known limits.
- **The "Retrieval quality gate" job** runs
  `tests/integration/test_topic_coverage_quality.py` on the production
  path. It fails if:
  - the coverage or end-to-end errors rise by more than one topic or pair
    above `baseline.json`; or
  - the production check decides any topic differently from the evaluated
    signal.
- **The no-route tripwire** stays in `tests/unit/test_cbt_exam_grounded.py`.
- **The strict xfail from the #62 review** now passes, so it is an ordinary
  test. That test is `test_a_supported_question_on_another_topic_is_rejected`.

Changing the set, the checks or retrieval means re-measuring and committing
a new baseline in the same change:

```bash
python scripts/evaluate_topic_coverage.py --write-baseline
```
