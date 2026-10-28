# ADR 001: Retrieval strategy and evidence grounding

## Status

Accepted, 2026-10-27 (P3-5). Revisit when the real-document evaluation is
recorded, or when a change to retrieval moves the baseline.

## Context

Chat, explanations and question generation answer students from their
school's uploaded course material. Two questions had to be settled with
evidence rather than assumption:

1. How to retrieve: dense (meaning-based) search, keyword search, or both
   combined.
2. When retrieved material is good enough to answer from, so that the
   service never presents an answer as grounded in a document when it is
   not.

Constraints already decided: local models only (D3: all-MiniLM-L6-v2 on
ONNX Runtime, no hosted API), and Postgres with pgvector on Supabase (D4).
Every query is scoped to the verified tenant inside the SQL.

## Decision

1. **Retrieval is hybrid.** For each question, run dense search (pgvector,
   cosine) and keyword search (Postgres full-text search, query terms
   combined with OR), then merge the two rankings with Reciprocal Rank
   Fusion. Both read the durable `document_chunks` index.
2. **Grounding is decided by citation, not by score.** The model gets the
   tenant's top five passages in full, numbered, with their document and
   page. It must answer only from them and cite them as `[n]`, or say that
   the material does not cover the question. A deterministic check
   (`check_grounded`) keeps an answer only if it cites at least one passage
   it was given and nothing else; otherwise the reply is a fixed "not
   covered" message. When nothing is retrieved, the model is not called.
   Question generation produces no questions without passages.

## Evidence

**Evaluation set v1** (`data/retrieval_eval/v1`), committed and run in CI:

- **Corpus:** 8 original course-style documents across six courses
  (MTS 102, PHY 102, CHM 101, BIO 101, CSC 101 and STA 111), written for
  the set so it can be committed.
- **Questions:** 56 answerable student questions:
  - 12 direct;
  - 14 paraphrased;
  - 12 informal;
  - 12 mathematical;
  - 6 needing more than one passage.

  There are also 5 out-of-corpus questions.
- **Expected evidence:** each answerable question names its evidence as a
  span quoted verbatim from a labelled page.
- **Distractors:** 48 generated past-question papers on the same subjects,
  the realistic competition real tenants' material provides. That makes
  158 chunks in total.

**Method:**

- The production path is used: chunker, local embedder, `PgChunkIndex`.
- The evaluation tenant is fixed, so runs are repeatable; three runs gave
  byte-identical reports.
- A question counts as answered only if a top-5 chunk is from the
  expected document, covers the expected page and contains the expected
  span. Multi-passage questions need every span.

**Baseline:**

| Strategy | recall@5 | MRR@10 | Questions missed |
|---|---|---|---|
| Dense | 0.929 | 0.819 | 4 |
| Keyword | 0.857 | 0.715 | 8 |
| **Hybrid** | **0.946** | **0.838** | 3 |

Hybrid is best on both metrics here, but its lead over dense is one
question of 56. That is **promising, not established**: the set is too
small to show a significant difference. Hybrid is kept because its two
sides miss different questions. The questions all three strategies miss
are formula-only questions (such as `q = mcΔT` or a sample variance). The
local D3 benchmark on real documents (28 queries, 2,371 chunks) agreed in
direction: keyword search was strongest alone, and dense search covered
some of its misses.

**Why not a score threshold:**

- The top dense score of answerable questions ranged from 0.58 to 0.83.
- For 15 out-of-corpus questions it ranged from 0.53 to 0.73.
- A cutoff of 0.74 rejects every out-of-corpus question, but also two
  thirds of the answerable ones.

Because the ranges overlap, no single score cutoff separates the two
groups.

## Consequences

- **The baseline is a measured regression reference, not a quality
  guarantee or a release target.**
  - CI's "Retrieval quality gate" fails if any strategy's recall@5 or
    MRR@10 drops below the baseline. The tolerance is one question, or 0.02
    MRR.
  - Improving retrieval means committing a re-measured baseline with the
    change.
- The numbers describe evaluation set v1, which is original material
  written for the purpose. They are not a measurement of retrieval quality
  on real course documents. That measurement is the tier-2 evaluation
  (`docs/p3-5-tier2-evaluation.md`, aggregates only). On 141 real FUTA
  documents, recall@5 is dense 0.675, hybrid 0.625 and keyword 0.325, with
  no significant difference between dense and hybrid at n = 40. The choice
  of hybrid is to be re-examined against that set (its follow-up 1).
- A grounded answer can still misstate what a passage says. The citation
  check proves *which* evidence was used, not that the answer is faithful
  to it. Generated questions are produced from passages, but are not yet
  checked one by one for traceability, answerability or unsupported
  premises; that check is tracked as a follow-up.
- Abstention depends on the model following the grounding instructions.
  The deterministic check catches answers that cite nothing or cite
  passages they were not given. It does not catch a model that cites a
  passage while answering from outside it.
- Citations are returned as structured data (passage number, chunk,
  document, page span). The backend does not yet relay them to clients;
  the citation contract for the app, including access checks on the
  cited source, belongs to the Mobile API Contracts work.
