# AI-12 offline AI benchmark: report

**Status: draft, results pending.** Device runs on Android phones and the Nigerian-language
model are pending. No model is selected here. The desktop numbers rank candidates and
rule out what cannot fit; the choice is made from device runs.

Harness and registry verified 2026-11-16 (two portfolio models added 2026-11-20); the numbers below come from the harness in this folder (dataset A: 43 original tasks).

## 1. Candidates verified (2026-11-16)

Sizes, revisions and sha256 come from the Hugging Face model API (`?blobs=true`);
licences and language notes come from each official model card. Everything is pinned in
[models.json](models.json).

| Model | Role | Licence | File benchmarked | Download | Languages (card) |
|---|---|---|---|---:|---|
| [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | low-memory baseline | Apache-2.0 | unsloth Q4_K_M GGUF | 533 MB | "201 languages and dialects"; Yoruba, Igbo, Hausa not named |
| [Qwen3.5 2B](https://huggingface.co/Qwen/Qwen3.5-2B) | main lightweight candidate | Apache-2.0 | unsloth Q4_K_M GGUF | 1,281 MB | as above |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | mobile alternative | Apache-2.0 | unsloth Q4_K_M GGUF; LiteRT-LM `.litertlm` | 3,107 MB; 2,588 MB | 35+ supported, 140+ pre-trained; 2.3B effective, 5.1B with per-layer embeddings |
| [Gemma 3n E2B](https://huggingface.co/google/gemma-3n-E2B-it) | mobile alternative | Gemma Terms of Use (accepted by the user) | unsloth Q4_K_M GGUF | 3,027 MB | see card |
| [Ministral 3 3B Instruct 2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | higher-memory candidate | Apache-2.0 | Mistral's own Q4_K_M GGUF | 2,147 MB | 11 named; no Nigerian language |
| [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B) | higher-capability tier | Apache-2.0 | unsloth Q4_K_M GGUF | 2,741 MB | as Qwen3.5 |
| [Phi-4-mini-instruct 3.8B](https://huggingface.co/microsoft/Phi-4-mini-instruct) | reasoning comparison | MIT | unsloth Q4_K_M GGUF | 2,492 MB | 23 listed; no Yoruba, Igbo or Hausa |
| Nigerian-language model | multilingual candidate | pending | pending | - | name, source, weights and licence not yet supplied |

**Glimmer 2B is not a downloadable model.** No checkpoint of that name exists on Hugging
Face or in release coverage. "Glimmer" is Meta's
[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Apache-2.0,
about 30B parameters: a 2B perception encoder plus a 28B decoder). Even at 4 bits it is
far beyond phone memory, so it stays out of the mobile shortlist. The earlier "Glimmer
2B" direction remains in the architecture history as a proposed configuration, not as an
available model.

## 2. Method

- **Runtimes.** llama.cpp v1.0.0 (commit `b1b770d9`, CPU, built from source) runs every
  GGUF. Google AI Edge LiteRT-LM (`litert-lm-api` 0.19.2, CPU) runs the Gemma 4 E2B
  `.litertlm`. The two are different deployment paths, so their rows are not compared as
  if the execution environments were the same.
- **Settings.** 4 threads, 4,096-token context, thinking disabled, greedy decoding
  (temperature 0, top-k 1). The sampled runs use temperature 0.7, top-k 20, top-p 0.8,
  three seeds.
- **Modes.** `free`: instruction-only JSON (raw reliability). `schema`: the runtime
  constrains output to the JSON schema (how BUILD would deploy). `long`: the English
  grounded questions with all ten passages (about 2.5k tokens) instead of three.
  `sampled`: consistency against the greedy run.
- **Memory.** Peak RSS is the kernel's `VmHWM` for the llama-server process. Model files
  are memory-mapped, so this includes the pages of weights actually touched. For
  LiteRT-LM it is the whole Python process.
- **Scoring** is deterministic ([scoring.py](scoring.py)): exact facts, citation subsets,
  abstention flags and tool arguments. Explanation quality and every Nigerian-language
  output also go to a native-speaker review sheet.

### Desktop hardware

Lenovo IdeaPad Slim 3: Intel Core i5-13420H, 8 GB DDR5, Windows 11. Runs happen in WSL2
(Ubuntu), which sees **3.7 GB of RAM** and 12 logical CPUs. Docker Desktop was
unavailable, and the 3.7 GB ceiling is close to what a 4 GB phone leaves an app, which
makes it a useful first filter. It is not a phone: x86 cores, desktop memory bandwidth,
and no thermal limit comparable to a phone's.

## 3. Results (desktop, dataset A)

**Pending.** The desktop queue (Qwen3.5 0.8B first, then Qwen3-1.7B, SmolLM2-1.7B,
Qwen3.5 2B, Ministral 3 3B, Gemma 4 E2B on LiteRT-LM, Gemma 3n E2B, Qwen3.5 4B,
Phi-4-mini, Gemma 4 E2B on llama.cpp) runs one model at a time. Result files appear in
`runs/desktop/` and the tables below are generated from them with `report.py`; no
number is quoted here before it is measured. A model that cannot run in the 3.7 GB WSL
ceiling gets a `<model>.failed.txt` and is reported as "does not fit".

## 4. Real course PDFs (dataset B)

The local corpus (`data/ocr_evaluation/pdfs`, third-party student documents) was
extracted read-only by [pdf_eval.py](pdf_eval.py): **5 documents, 61 pages; 43 pages
have extractable text and 18 are scanned and need OCR first.** A review template with 43
draft task slots exists locally (git-ignored). **No dataset B task is verified yet, so
there are no dataset B scores.** A person must write and check each question against
the PDF; generated questions are never scored as ground truth. No PDF text leaves the
machine or enters the repository, and result files redact dataset B outputs.

## 5. Pending

| Item | Needs |
|---|---|
| Device runs (4 GB floor, 6 GB, 8 GB+; Android 14+) | exact phone models and physical RAM from the user; then [device/DEVICE_PROCEDURE.md](device/DEVICE_PROCEDURE.md) |
| Battery and thermal | device runs only |
| Nigerian-language model | name, source, weights and licence, verified before it enters the registry |
| Nigerian-language quality | native speakers review the Yoruba, Hausa and Igbo prompts, then `runs/desktop/human_review.csv` |
| Dataset B scores | human-verified tasks over the 43 text pages; OCR for the 18 scanned pages |
| Larger task set | 43 tasks is a baseline; differences of one or two tasks are noise |

## 6. Recommendations (provisional, before any result)

No model roles are assigned. Qwen3.5 0.8B, Qwen3-1.7B and SmolLM2-1.7B are the target
portfolio (master plan, 2026-11-20); everything else is a comparison. Next steps:
get phones to run the device procedure, set the abstention gate from the baseline
(see [the evaluation plan](../../docs/ai-12-evaluation-plan.md)), grow the task set
to 100 reviewed tasks, and have native speakers review the Nigerian-language outputs
only if those languages come back into scope.
