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
  - **Two independent classifications.** Each question has a source type and a text-layer quality, and the two overlap:
    - **Source type:** 26 questions are on past-question banks and 14 on explanatory notes.
    - **Text-layer quality:** 33 pages have a normal text layer, and 7 have one that has lost its spaces (`Thefemaleconestake3yearstomature`). Of those 7, 6 are question banks and 1 is explanatory notes.
- **Ceiling:** every evidence span lies inside a single chunk. Chunking loses nothing, so every miss below is a ranking miss.

The local question set's sha256 is `313db888…a28f`, which ties these figures to the exact set.

## Results

recall@5, with its 95% Wilson interval (n = 40), and MRR@10:

| Strategy | Found in top 5 | recall@5 | 95% interval | MRR@10 |
|---|---|---|---|---|
| Dense (MiniLM-L6) | 27 of 40 | 0.675 | 0.52–0.80 | 0.587 |
| Hybrid (RRF, the production default) | 25 of 40 | 0.625 | 0.47–0.76 | 0.519 |
| Keyword (PostgreSQL FTS) | 13 of 40 | 0.325 | 0.20–0.48 | 0.295 |

The intervals show sampling uncertainty with 40 questions. They overlap, so the scores above are observations, not a ranking established beyond chance.

By source type (the rows partition all 40 questions):

| Source type | Questions | Dense | Hybrid | Keyword |
|---|---|---|---|---|
| Past-question banks | 26 | 16 (0.62) | 15 (0.58) | 6 (0.23) |
| Explanatory notes | 14 | 11 (0.79) | 10 (0.71) | 7 (0.50) |
| **Total** | **40** | **27** | **25** | **13** |

By text-layer quality (the rows partition all 40 questions, across both source types):

| Text layer | Questions | Dense | Hybrid | Keyword |
|---|---|---|---|---|
| Normal | 33 | 25 (0.76) | 23 (0.70) | 13 (0.39) |
| Lost its spaces | 7 (6 banks, 1 notes) | 2 (0.29) | 2 (0.29) | 0 (0.00) |
| **Total** | **40** | **27** | **25** | **13** |

For comparison, evaluation set v1 (tier 1) gives hybrid 0.946, dense 0.929 and keyword 0.857.

## What it says

1. **Real documents are much harder than the tier-1 set.** Recall@5 falls from about 0.93 to about 0.65. The tier-1 baseline is a regression reference, as ADR 001 says, and must not be read as the quality students will see.
2. **The hybrid has not demonstrated a benefit on real documents. Equally, dense has not been shown to be better.**
   - Dense found 27 of 40 and the hybrid 25. Dense found 4 questions the hybrid missed; the hybrid found 2 that dense missed. With 6 discordant pairs that difference is not significant (McNemar).
   - Keyword search is weak on this material (13 of 40). The question banks repeat the same vocabulary across hundreds of near-identical items, so its ranking places other questions on the topic ahead of the one asked. That may cost the hybrid some of dense's hits, but this sample cannot show it.
   - **The production default (hybrid) is kept.** This evaluation does not justify switching it. Changing the default needs the comparison in follow-up 1.
3. **Text layers without spaces defeat keyword search entirely** (0 of 7) and most dense matches too (2 of 7). They are 7 of the 40 sampled pages, so they are not rare. They are a P3-2 extraction problem: the text layer exists but is unusable. Such a page should be treated as needing OCR, or have its words re-segmented, before chunking.
4. **Duplicated documents waste top-k slots.** 22 of 40 spans appear in two or more documents. Duplicates do not count as misses here, but in production they repeat one source across the results a student sees.

## Limitations

- **The questions were written by Claude and have not been reviewed by a person.**
  - The only automated check is mechanical: each evidence span is verified verbatim on its page and in a single chunk.
  - Nobody has checked that each question is representative of what students ask, answerable from its span alone, or well formulated.
  - Human review of a sample is the next step before these figures are relied on beyond comparison between strategies.
- **The scanned pages are not measured.** 1,459 pages (45% of the collection) need OCR. These figures describe retrieval over text-layer pages only, not the whole collection.
- **Duplicates.** Duplicated documents can fill several result slots with one source. Accepting a span wherever it appears keeps that from counting as a miss, but it does not measure what deduplication would change.
- **The 7 spaceless pages are an extraction problem first.** Their retrieval scores reflect unusable input text. Any fix (OCR routing or re-segmentation) must be measured as its own change.
- **Small sample.** With n = 40 the 95% intervals are about ±0.15, wide enough to hide differences of the size observed.

## Follow-ups (not changed by this document)

These are proposals, each to be measured on this set before and after:

1. **Retrieval weighting:** compare dense alone, the current hybrid and a dense-weighted fusion on the same questions. Report recall@5 and MRR@10, and keep the tier-1 regression gate. ADR 001's choice of hybrid stands until that comparison is made.
2. **Spaceless text layers:** detect a page whose text layer has implausibly few spaces for its length, and route it to OCR or re-segment its words. Measure it separately, on the spaceless rows above.
3. **Duplicate assets:** detect duplicate content (the same text hash within a tenant) at ingestion, and collapse duplicates in results. Evaluate this as its own change; it is not assumed to improve recall.
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
