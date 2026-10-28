# P3-5 tier-2 retrieval evaluation: real FUTA course documents

Measured on 2026-10-28. This is the evaluation that ADR 001 deferred until the ground truth (#17) was verified: retrieval quality on real course material, as opposed to evaluation set v1, which was written for the purpose. **Only aggregates are recorded here.** The corpus, the questions and their evidence spans stay on the machine that ran them, because the documents are third-party course material awaiting a licence review.

## Method

- **Corpus:** `data/ocr_evaluation/pdfs/1st Semester/`, the user's first-semester collection. It has 141 PDFs and 3,250 pages across BIO 101/103, CHE 101, CVE 105, GNS 101/103, MEE 101, MTS 101 and PHY 101/103.
- **Extraction:** `scripts/build_retrieval_corpus.py` reads each page the way the P3-2 extractor does without OCR: `classify_page`, then `text_layer`, with its per-page reading order.
  - 1,785 pages were read: 1,749 native and 36 mixed.
  - 1,465 pages were not indexed: 1,459 scanned, which need OCR, and 6 empty.
  - OCR of the scanned pages (seconds of CPU per page) was left out of this run, so **this measures retrieval over text-layer pages only.**
- **Indexing:** the production path, the same one tier 1 uses (`scripts/evaluate_retrieval.py`):
  - `chunk_document`;
  - the local all-MiniLM-L6-v2 embedder;
  - `PgChunkIndex` on PostgreSQL 17 with pgvector 0.8.0.

  All 141 documents produced 2,289 chunks under one evaluation tenant. The run took 198 s, then the tenant was deleted.
- **Questions:** 40, one per page, written by Claude.
  - **The pages:** 40 documents drawn with a fixed seed (20261028) from the 84 that have a page of at least 600 characters, then one such page drawn from each.
  - **The questions:** student-style, phrased in the student's own words where the page allowed. Each names a verbatim evidence span from its page.
  - **Scoring:** as in tier 1, a retrieved chunk counts only if it contains the span on the right page.
  - **Duplicates:** the collection holds many duplicated documents (the same past questions under different file names), so a span is accepted on every page where it appears. 22 of the 40 spans appear on 2 to 6 pages.
  - **Categories:** 26 questions are on past-question banks and 14 on explanatory notes. 7 of the pages have a text layer that has lost its spaces (`Thefemaleconestake3yearstomature`). Those are reported separately.
- **Ceiling:** every evidence span lies inside a single chunk. Chunking loses nothing, so every miss below is a ranking miss.

The local question set's sha256 is `313db888…a28f`, which ties these figures to the exact set.

## Results

recall@5, with its 95% Wilson interval (n = 40), and MRR@10:

| Strategy | recall@5 | 95% interval | MRR@10 |
|---|---|---|---|
| Dense (MiniLM-L6) | **0.675** | 0.52–0.80 | 0.587 |
| Hybrid (RRF, the production default) | 0.625 | 0.47–0.76 | 0.519 |
| Keyword (PostgreSQL FTS) | 0.325 | 0.20–0.48 | 0.295 |

By category (recall@5):

| Category | Questions | Dense | Hybrid | Keyword |
|---|---|---|---|---|
| Past-question banks | 20 | 0.70 | 0.65 | 0.30 |
| Explanatory notes | 13 | 0.85 | 0.77 | 0.54 |
| Pages whose text layer lost its spaces | 7 | 0.29 | 0.29 | 0.00 |

For comparison, evaluation set v1 (tier 1) gives hybrid 0.946, dense 0.929 and keyword 0.857.

## What it says

1. **Real documents are much harder than the tier-1 set.** Recall@5 falls from about 0.93 to about 0.65. The tier-1 baseline is a regression reference, as ADR 001 says, and must not be read as the quality students will see.
2. **The hybrid is not shown to help on real documents.**
   - Dense found 4 questions that the hybrid missed; the hybrid found 2 that dense missed. With 40 questions that difference is not significant (McNemar, 6 discordant pairs), so the honest reading is "no measured benefit".
   - Keyword search is weak on this material (0.325). The question banks repeat the same vocabulary across hundreds of near-identical items, so its ranking places other questions on the topic ahead of the one asked. Fusing that ranking in costs the hybrid some of dense's hits.
3. **Text layers without spaces defeat keyword search entirely** (0 of 7) and most dense matches too (2 of 7). They are 7 of the 40 sampled pages, so they are not rare. They are a P3-2 extraction problem: the text layer exists but is unusable. Such a page should be treated as needing OCR, or have its words re-segmented, before chunking.
4. **Duplicated documents waste top-k slots.** 22 of 40 spans appear in two or more documents. Duplicates do not count as misses here, but in production they repeat one source across the results a student sees.

## Follow-ups (not changed by this document)

These are proposals, each to be measured on this set before and after:

1. **Retrieval weighting:** compare RRF with dense-weighted fusion and with dense alone on tier 2, and keep tier 1 from regressing. ADR 001's choice of hybrid stands until that comparison is made.
2. **Spaceless text layers:** detect a page whose text layer has implausibly few spaces for its length, and route it to OCR, or re-segment its words. Then re-measure category 3.
3. **Duplicate assets:** detect duplicate content (the same text hash within a tenant) at ingestion, and collapse duplicates in results.
4. **Scanned pages:** extend tier 2 to the 1,459 scanned pages once OCR throughput allows a full run. They are 45% of the collection's pages and are not measured here.
5. **A larger question set** narrows the intervals: 40 questions give about ±0.15.

## Reproducing

On a machine with the collection, with PostgreSQL + pgvector (`DATABASE_URL`) and the model files (`EMBEDDING_MODEL_DIR`):

```
python scripts/build_retrieval_corpus.py "data/ocr_evaluation/pdfs/1st Semester" /tmp/tier2
# write /tmp/tier2/questions.json (local only; same format as data/retrieval_eval/v1)
python scripts/evaluate_retrieval.py --dataset /tmp/tier2 --out /tmp/tier2/report.json
```

Keep the output directory outside the repository: it holds document text.
