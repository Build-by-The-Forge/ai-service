# AI-12: what the three offline components need from each other

PRD §0.6 and §146 require offline **generative** AI on the student's phone. It is
three components that only work together:

1. **Local inference**: the small model and its runtime (llama.cpp or LiteRT-LM).
2. **Offline retrieval and storage**: downloaded, versioned course packs, SQLite
   metadata, FTS5/BM25, a local vector index with a small embedding model, and hybrid
   retrieval with course and document filters.
3. **Offline orchestration**: the on-device runtime that routes a request, calls
   retrieval, builds the prompt, calls the model and deterministic tools, decides to
   abstain, and syncs later.

The mobile app talks only to orchestration through a stable contract and never needs to
know which model, local or cloud, answered. The AI-12 benchmark
(`benchmarks/offline/`) measures the inference side of these contracts; the numbers
below refer to its report.

## Contracts shared by all three

| Contract | Shape | Why |
|---|---|---|
| Passage id | `<pack>/<doc>/<page>[#chunk]`, stable across pack versions | citations must survive updates and map back to a page |
| Grounded reply | `{"answer": str, "citations": [passage id], "answerable": bool}` | the benchmark's schema; abstention is a field, not a phrase |
| Device tier | from physical RAM: 4 GB, 6 GB, 8 GB+ (never "virtual RAM") | picks the model, context size and passage budget |
| Evidence threshold | per pack, set by retrieval evaluation | lets orchestration abstain before spending battery on generation |
| Telemetry (local, opt-in sync) | TTFT, tokens/s, peak memory, abstained, error code | routing and the next benchmark round use real-device numbers |

## Local inference needs

- **From retrieval and storage:** at most about three page-sized passages
  (about 1,800 characters each) per question on 4 GB phones. The benchmark's `long`
  mode measures what a full ten-passage context costs in time to first token. It also
  needs stable passage ids to cite, and plain text: OCR has already run, and scanned
  pages that still need OCR are never passed as evidence.
- **From orchestration:** the model's own chat template with thinking turned off,
  per-task token limits, grammar-constrained JSON (the `schema` mode) for every
  structured reply, the tool list, cancellation, and load and unload decisions under
  memory pressure. Android kills a background app before it gives up memory.

## Offline retrieval and storage needs

- **From local inference:** a memory budget. The peak RSS figures in the report are
  what the LLM takes, and the embedding model and the SQLite and vector indexes have to
  fit in what remains on a 4 GB phone. It also needs the tokenizer's counts, so chunks
  fit the context the tier allows.
- **From orchestration:** pack manifests (version, sha256, signature, licence and
  access scope, revocation), the student's course filters, and the sync schedule.
  Packs are downloaded once over mobile data, so pack size is a cost to the student
  just like model size.

## Offline orchestration needs

- **From local inference:** one call shape per runtime: OpenAI-compatible chat with
  `tools` and `response_format: json_schema` for llama.cpp, and a Conversation with a
  `ResponseFormat` for LiteRT-LM. It also needs error codes for out-of-memory, a killed
  process and timeouts, and the speed figures, so it can decide when local generation
  is too slow and a deterministic answer or the online queue is better.
- **From retrieval and storage:** retrieval scores to compare with the evidence
  threshold (abstain or ask the student to sync, rather than let a small model guess),
  page provenance for every passage, and deterministic lookups (offline CBT items,
  definitions) exposed as tools. The tool-call figures in the report decide whether
  small models can be trusted to choose them.
- **Shared with online:** the same reply schema, citation format and permissions as
  the online Gemini path, so an offline answer can be checked or upgraded when the
  phone reconnects, without uploading conversations automatically (PRD §146.11).
