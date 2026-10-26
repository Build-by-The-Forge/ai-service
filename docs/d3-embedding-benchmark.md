# D3: local embedding benchmark (2026-10-26)

Decision record for the embedding model used by P3-4 (durable index) and
P3-5 (retrieval). Constraint, set by the product owner: **local models only**.
No document text goes to a hosted API.

## Method

- **Corpus.** The seven evaluation documents in `data/ocr_evaluation`
  (5 PDFs, 2 photos) were extracted and chunked by the merged P3-2/P3-3
  pipeline (`DocumentExtractor`, `chunk_document`, 1200 characters with
  200 overlap). That gave 87 target chunks. To make ranking realistic, 2,284
  distractor chunks were added from the native-text pages of the local
  `1st Semester` corpus, chunked the same way, for 2,371 chunks in total.
  None of this text is committed.
- **Queries.** 28 questions, each labelled with the document and page that
  answers it. They mix natural-language questions ("Who first determined
  the charge of the electron?") with notation-heavy ones ("derivative of
  tan(cos x)"). Two further queries about a handwritten graph photo were
  excluded, because its OCR text is illegible and no retriever can answer
  them; that is a P3-2 OCR limitation, not a retrieval one. The queries
  quote corpus content, so they stay local; P3-5's committed recall@5
  harness will use its own fixture.
- **Metrics.** A hit is any top-ranked chunk from the expected document whose
  page span covers an expected page. The table reports recall@5 and MRR@10.
- **Runtime.** fastembed 0.7.3 (Apache-2.0, ONNX Runtime, no PyTorch). Each
  model ran in its own container with `--network none`, 4 CPUs and 2 GB of
  memory, which is a realistic worker size. Models were downloaded
  beforehand, in a step with no documents mounted.
- **Hybrid.** Hybrid rows fuse the BM25 and dense rankings by reciprocal rank
  fusion (k = 60). This is the fusion `HybridRetrievalEngine` already
  implements.

## Results

| Retriever | recall@5 | MRR@10 | chunks/s | query ms | peak RSS | model size | licence |
|---|---|---|---|---|---|---|---|
| BM25 alone | **0.857** | 0.656 | n/a (no model) | <1 | – | – | – |
| all-MiniLM-L6-v2 (384-d) | 0.786 | 0.588 | **15.4** | 17 | **785 MB** | 87 MB | Apache-2.0 |
| bge-small-en-v1.5, int8 (384-d) | 0.714 | 0.598 | 4.0 | 17 | 1,489 MB | 65 MB | MIT |
| jina-embeddings-v2-small-en (512-d) | 0.500 | 0.404 | 3.3 | 19 | 2,078 MB | 125 MB | Apache-2.0 |
| BM25 + MiniLM-L6 (RRF) | 0.750 | 0.675 | 15.4 | 17 | 785 MB | 87 MB | |
| BM25 + bge-small (RRF) | 0.821 | **0.691** | 4.0 | 17 | 1,489 MB | 65 MB | |

Not measured:

- bge-base-en-v1.5 and snowflake-arctic-embed-s: their downloads timed out.
- gte-base: not available from fastembed's mirror.

The base-size models are about three times bge-small's compute, which
conflicts with the speed-first priority.

## Findings

1. **The set is small: with 28 queries, one query moves recall by 0.036.**
   Differences under about 0.07 are within noise. The firm conclusions are
   the large gaps: jina-v2-small is clearly worse, and MiniLM-L6 is about 4×
   faster than bge-small at half the memory.
2. **Keyword retrieval is the strongest single signal on course material.**
   BM25 has the best recall@5. All four of its misses are notation-heavy
   queries ("limit of sin x over x", "derivative of tan(cos x)"), where the
   page text uses symbols (→, superscripts, `xe5y`). That is the gap the
   dense side is meant to cover.
3. **The query set leans lexical.** I wrote the queries while reading the
   pages, so they share wording with the text. That favours BM25. Students'
   real questions paraphrase more, which is where dense retrieval earns its
   cost. The P3-5 harness should therefore add paraphrased queries.
4. **Throughput is real compute, not a fault.** Two points matter for the
   P3-4 implementation:
   - ONNX Runtime must have its threads pinned to the CPU quota. Unpinned
     it oversubscribes, and bge-small runs at 2.3 chunks/s instead of 4.2.
   - Batches must be bounded. fastembed's default batch of 256 was
     OOM-killed at 2 GB; batch 32 fits comfortably.

   At 15 chunks/s, MiniLM-L6 embeds the largest corpus document (538 pages,
   about 1,000 chunks) in about a minute on 4 CPUs. bge-small takes about
   four minutes.

## Recommendation

- **Hybrid retrieval: Postgres full-text search plus all-MiniLM-L6-v2,
  fused by the existing RRF engine.**
  - The keyword index is the primary signal. It is free at ingest time and
    needs no extension on any managed Postgres.
  - MiniLM-L6 is the fastest and smallest candidate, with quality within
    noise of bge-small. 384-d vectors keep the pgvector index small.
  - Apache-2.0, ONNX-only (no PyTorch in the image).
- **Swap-ready.**
  - Store the model id and dimension with every chunk, so a change of model
    (for example to bge-small, if the P3-5 harness with paraphrased queries
    shows a real gain) is a re-embedding job, not a schema change.
  - Load the model files into the image at build time, never at runtime,
    so the worker needs no network access to a model hub.
- **Operational settings for P3-4:**
  - `threads` equals the worker's CPU quota;
  - batch size 32;
  - one model instance per worker process, loaded lazily once, like the
    OCR engine.
