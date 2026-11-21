# AI-12 offline AI benchmark: report

**Status: draft. Desktop results are in (10 runs); device runs, the Nigerian-language model and native-speaker review are pending.** No model is selected here. The desktop numbers rank candidates and
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

GitHub-hosted `ubuntu-latest` runners: 4 vCPUs, about 16 GB RAM, one fresh runner per
model, **CPU model differs between jobs** (recorded in every result file's `hardware`).
llama.cpp is built for a portable AVX2 baseline (`AI12_PORTABLE=1`) so the binaries are
identical across runners. The maintainer's 8 GB laptop was tried first (WSL2 sees 3.7 GB);
it was abandoned because llama-server's host prompt cache grew it past the limit and the
OOM killer and the WSL service ended runs while other work shared the machine. Those
aborted attempts are not reported. None of this is a phone: x86 server cores, 16 GB, no
thermal limit. It ranks candidates and bounds memory; it does not predict phone speed.

## 3. Results (desktop, dataset A)

Run 44853974362 (workflow `offline-benchmark`, branch `bench/ai12-desktop-3`): each model ran
alone on a fresh GitHub-hosted 4-vCPU, 16 GB Ubuntu runner, with llama.cpp built for a
portable AVX2 baseline (nine GGUF models) and LiteRT-LM 0.19.2 for Gemma 4 E2B. The
tables below are generated by `report.py` from the files in `runs/desktop/`.

**How to read them.**

- The runner CPU differed between jobs (Intel Xeon 8370C, 8573C, 6975P-C; AMD EPYC 7763,
  9V45, 9V74; each result file records it). **Speed figures are indicative only and are not
  comparable across models.** Memory and quality figures do not depend on the CPU: quality
  runs are greedy.
- The runner has 16 GB. Peak RSS says what each model needs, not whether the phone has it;
  it includes weights memory-mapped from the file. A 4 GB phone leaves an app well under
  4 GB, so a peak near or above 4 GB (Gemma 4 E2B, Gemma 3n, Ministral 3 3B, Phi-4-mini,
  Qwen3.5 4B at a 4,096-token context) would not fit it without changes. That is an
  inference from desktop memory, not a phone measurement.
- 43 tasks is a baseline. There are only **6 unanswerable and 6 tool-call tasks**, so one
  task is 17 points; read counts, not small gaps. Free mode is instruction-only JSON;
  schema mode is grammar-constrained (how BUILD would deploy).
- Nigerian-language prompts and the proxy scores are unreviewed: **no language-quality
  claim is made.** `runs/desktop/human_review.csv` (250 outputs) awaits native speakers.

#### Footprint and speed

| Model | Runtime | Download MB | Peak RSS MB | Cold start s | TTFT median s | Decode tok/s | Prefill tok/s | llama-bench tg128 | llama-bench pp512 | Switch s (unload + warm reload) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-3n-e2b-q4km | llama.cpp llama-server | 3027 | 4315 | 2.16 | 5.86 | 12.4 | 41 | 13.6 | 45 | 2.2 |
| gemma-4-e2b-litertlm | LiteRT-LM cpu | 2588 | 3576 | 3.20 | 4.41 | 23.6 | 158 | - | - | - |
| gemma-4-e2b-q4km | llama.cpp llama-server | 3107 | 4318 | 1.64 | 1.15 | 23.8 | 214 | 26.7 | 240 | 1.7 |
| ministral-3-3b-q4km | llama.cpp llama-server | 2147 | 4012 | 2.51 | 7.27 | 9.1 | 30 | 11.0 | 33 | 2.6 |
| phi-4-mini-q4km | llama.cpp llama-server | 2492 | 4289 | 2.05 | 6.38 | 8.5 | 31 | 10.0 | 34 | 2.1 |
| qwen3-1.7b-q4km | llama.cpp llama-server | 1107 | 2295 | 1.33 | 3.09 | 24.9 | 75 | 29.9 | 80 | 1.5 |
| qwen3.5-0.8b-q4km | llama.cpp llama-server | 533 | 936 | 0.98 | 1.61 | 30.7 | 162 | 32.9 | 168 | 1.1 |
| qwen3.5-2b-q4km | llama.cpp llama-server | 1281 | 2006 | 1.44 | 3.95 | 21.2 | 66 | 21.1 | 71 | 1.5 |
| qwen3.5-4b-q4km | llama.cpp llama-server | 2741 | 4327 | 1.74 | 6.59 | 13.2 | 40 | 13.2 | 42 | 1.9 |
| smollm2-1.7b-q4km | llama.cpp llama-server | 1056 | 2621 | 1.32 | 3.68 | 16.4 | 62 | 24.2 | 68 | 1.4 |

#### Structured output and tools (free = instruction only; schema = grammar-constrained)

| Model | JSON strict (free) | JSON parseable (free) | Schema-valid (free) | Schema-valid (schema) | Extraction correct (free / schema) | Tool call correct | Tool name correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-3n-e2b-q4km | 68% | 97% | 97% | 97% | 83% / 83% | 17% | 17% |
| gemma-4-e2b-litertlm | 39% | 97% | 94% | 97% | 100% / 83% | 100% | 100% |
| gemma-4-e2b-q4km | 55% | 100% | 77% | 97% | 100% / 100% | 100% | 100% |
| ministral-3-3b-q4km | 0% | 87% | 87% | 84% | 83% / 100% | 100% | 100% |
| phi-4-mini-q4km | 68% | 84% | 19% | 87% | 100% / 100% | 17% | 17% |
| qwen3-1.7b-q4km | 84% | 84% | 84% | 100% | 83% / 83% | 100% | 100% |
| qwen3.5-0.8b-q4km | 87% | 87% | 64% | 87% | 100% / 100% | 67% | 83% |
| qwen3.5-2b-q4km | 90% | 90% | 84% | 90% | 100% / 100% | 100% | 100% |
| qwen3.5-4b-q4km | 100% | 100% | 100% | 100% | 100% / 100% | 100% | 100% |
| smollm2-1.7b-q4km | 87% | 90% | 26% | 97% | 17% / 83% | 17% | 17% |

#### Grounded answers in English (10 answerable, 6 unanswerable)

free/schema: 3 passages per question, greedy. long: all 10 passages (context handling). sampled: temperature 0.7, repeated (consistency = share of sampled runs whose correctness matches the greedy run).

| Model | Mode | Answer correct | Faithful (correct + only supporting citations) | Citation precision | False abstention | Abstained when unanswerable | Hallucinated when unanswerable | No usable flag when unanswerable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-3n-e2b-q4km | free | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| gemma-3n-e2b-q4km | schema | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| gemma-3n-e2b-q4km | long | 100% | 80% | 90% | 0% | 67% | 33% | 0% |
| gemma-3n-e2b-q4km | sampled | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| gemma-4-e2b-litertlm | free | 90% | 90% | 100% | 0% | 83% | 17% | 0% |
| gemma-4-e2b-litertlm | schema | 80% | 80% | 100% | 0% | 83% | 17% | 0% |
| gemma-4-e2b-litertlm | long | 100% | 100% | 100% | 0% | 67% | 17% | 17% |
| gemma-4-e2b-litertlm | sampled | 93% | 93% | 100% | 0% | 83% | 6% | 11% |
| gemma-4-e2b-q4km | free | 70% | 70% | 100% | 0% | 83% | 17% | 0% |
| gemma-4-e2b-q4km | schema | 100% | 100% | 100% | 0% | 83% | 17% | 0% |
| gemma-4-e2b-q4km | long | 50% | 50% | 100% | 0% | 83% | 17% | 0% |
| gemma-4-e2b-q4km | sampled | 77% | 77% | 100% | 0% | 83% | 17% | 0% |
| ministral-3-3b-q4km | free | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| ministral-3-3b-q4km | schema | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| ministral-3-3b-q4km | long | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| ministral-3-3b-q4km | sampled | 97% | 97% | 100% | 0% | 100% | 0% | 0% |
| phi-4-mini-q4km | free | 0% | 0% | 100% | 0% | 0% | 0% | 100% |
| phi-4-mini-q4km | schema | 100% | 100% | 100% | 0% | 67% | 33% | 0% |
| phi-4-mini-q4km | long | 0% | 0% | 100% | 0% | 0% | 0% | 100% |
| phi-4-mini-q4km | sampled | 0% | 0% | 100% | 0% | 0% | 0% | 100% |
| qwen3-1.7b-q4km | free | 100% | 100% | 100% | 0% | 0% | 17% | 83% |
| qwen3-1.7b-q4km | schema | 100% | 100% | 100% | 0% | 83% | 17% | 0% |
| qwen3-1.7b-q4km | long | 100% | 100% | 100% | 0% | 0% | 50% | 50% |
| qwen3-1.7b-q4km | sampled | 93% | 93% | 100% | 0% | 6% | 17% | 78% |
| qwen3.5-0.8b-q4km | free | 40% | 40% | 100% | 0% | 17% | 67% | 17% |
| qwen3.5-0.8b-q4km | schema | 100% | 90% | 90% | 0% | 17% | 83% | 0% |
| qwen3.5-0.8b-q4km | long | 50% | 40% | 80% | 0% | 0% | 50% | 50% |
| qwen3.5-0.8b-q4km | sampled | 30% | 30% | 97% | 0% | 44% | 44% | 11% |
| qwen3.5-2b-q4km | free | 90% | 90% | 100% | 0% | 100% | 0% | 0% |
| qwen3.5-2b-q4km | schema | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| qwen3.5-2b-q4km | long | 90% | 90% | 100% | 0% | 83% | 17% | 0% |
| qwen3.5-2b-q4km | sampled | 90% | 90% | 100% | 0% | 100% | 0% | 0% |
| qwen3.5-4b-q4km | free | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| qwen3.5-4b-q4km | schema | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| qwen3.5-4b-q4km | long | 90% | 90% | 100% | 0% | 100% | 0% | 0% |
| qwen3.5-4b-q4km | sampled | 100% | 100% | 100% | 0% | 100% | 0% | 0% |
| smollm2-1.7b-q4km | free | 0% | 0% | 73% | 0% | 0% | 0% | 100% |
| smollm2-1.7b-q4km | schema | 80% | 40% | 57% | 0% | 0% | 100% | 0% |
| smollm2-1.7b-q4km | long | 0% | 0% | 0% | 0% | 0% | 17% | 83% |
| smollm2-1.7b-q4km | sampled | 10% | 3% | 70% | 0% | 0% | 28% | 72% |

| Model | Consistency (sampled vs greedy) | Long-context TTFT median s | Long-context prompt tokens |
|---|---:|---:|---:|
| gemma-3n-e2b-q4km | 100% | 46.52 | 1938 |
| gemma-4-e2b-litertlm | 98% | 12.07 | 1942 |
| gemma-4-e2b-q4km | 96% | 9.58 | 1942 |
| ministral-3-3b-q4km | 98% | 62.05 | 1825 |
| phi-4-mini-q4km | 100% | 55.75 | 1750 |
| qwen3-1.7b-q4km | 94% | 26.07 | 1826 |
| qwen3.5-0.8b-q4km | 75% | 12.21 | 1954 |
| qwen3.5-2b-q4km | 96% | 29.11 | 1954 |
| qwen3.5-4b-q4km | 100% | 46.59 | 1954 |
| smollm2-1.7b-q4km | 94% | 30.81 | 1876 |

#### Nigerian languages (unreviewed prompts; 3 grounded tasks + 1 explanation each)

| Model | Yoruba grounded (free / schema) | Hausa grounded | Igbo grounded | Yoruba explain: language share / terms | Hausa explain | Igbo explain | English explain terms |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-3n-e2b-q4km | 67% / 67% | 33% / 0% | 67% / 67% | 100% / 0% | 100% / 100% | 100% / 100% | 78% |
| gemma-4-e2b-litertlm | 67% / 67% | 67% / 0% | 33% / 0% | 92% / 0% | 100% / 0% | 100% / 0% | 78% |
| gemma-4-e2b-q4km | 0% / 67% | 67% / 67% | 33% / 33% | 98% / 100% | 95% / 100% | 100% / 100% | 89% |
| ministral-3-3b-q4km | 67% / 67% | 0% / 0% | 67% / 67% | 100% / 0% | 100% / 100% | 67% / 0% | 89% |
| phi-4-mini-q4km | 0% / 67% | 0% / 0% | 0% / 33% | 0% / 100% | 100% / 100% | 100% / 0% | 89% |
| qwen3-1.7b-q4km | 33% / 33% | 0% / 0% | 0% / 0% | 12% / 100% | 100% / 0% | 100% / 100% | 78% |
| qwen3.5-0.8b-q4km | 33% / 33% | 33% / 33% | 33% / 33% | 84% / 0% | 100% / 50% | 100% / 100% | 89% |
| qwen3.5-2b-q4km | 67% / 67% | 0% / 0% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 89% |
| qwen3.5-4b-q4km | 100% / 100% | 67% / 67% | 100% / 100% | 98% / 100% | 100% / 100% | 100% / 100% | 89% |
| smollm2-1.7b-q4km | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 100% | 0% / 100% | 100% / 100% | 78% |

Hardware: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz, 4 logical CPUs visible, 15982 MB RAM visible, Linux 7.0.0-1020-azure. Threads 4, context 4096, temperature 0.


### What the numbers show (observations, not selections)

**Smallest candidate, Qwen3.5 0.8B** (533 MB download, 936 MB peak RSS, 1.0 s cold start):
with constrained JSON it answered 10 of 10 answerable questions correctly and 9 of 10
faithfully (right answer, only supporting passages cited). On the 6 unanswerable
questions it abstained on **1 and answered anyway on 5** (83% hallucination); instruction-only
JSON was valid for 64% of replies and its tool calls were right on 4 of 6. With all ten
passages in the prompt it answered 5 of 10 and agreed with its own greedy run in only 75%
of sampled repeats. On this baseline it shows no reliable abstention, which is the one
property the Free-tier gate requires first. The threshold is not set yet; this is a first
measurement, not a verdict.

**Qwen3-1.7B** (1,107 MB, 2.3 GB peak): constrained, 10 of 10 answers correct and 5 of 6
abstentions; 6 of 6 tool tasks right (one is in Yoruba). Instruction-only, 16% of replies were invalid JSON
(for example a stray quote after `false`), which constrained decoding removes. With all
ten passages, abstention fell to 0 of 6.

**SmolLM2-1.7B** (1,056 MB, 2.6 GB peak; English only per its card): did not follow the
reply format without constraint (16 of its 16 English grounded replies had no usable `answerable` flag), and
constrained it answered 8 of 10 but faithfully on only 4, abstained on 0 of 6, and got 1 of 6 tool calls.

**Comparisons.** Qwen3.5 2B (1,281 MB, 2.0 GB): 9 to 10 of 10 answers, 6 of 6 abstentions and 6 of 6
tool tasks. Qwen3.5 4B (2,741 MB, 4.3 GB): perfect in both free and constrained modes.
Ministral 3 3B (2,147 MB, 4.0 GB): 6 of 6 abstentions, 6 of 6 tool calls, the slowest
prefill. Gemma 4 E2B (3,107 MB, 4.3 GB; 2,588 MB and 3.6 GB on LiteRT-LM): 5 of 6
abstentions on both runtimes, 6 of 6 tool calls; the two runtimes differ (llama.cpp 100%
constrained vs LiteRT-LM 80% answer correctness), so they are separate deployment rows.
Gemma 3n E2B (3,027 MB, 4.3 GB): strong grounded answers but 1 of 6 tool calls.
Phi-4-mini (2,492 MB, 4.3 GB): omitted the `answerable` flag unless constrained, 1 of 6
tool calls.

**Model switching** (unload plus warm reload, file in the page cache, so the best case):
1.1 s for Qwen3.5 0.8B up to 2.6 s for Ministral 3 3B on the runners. The first load after a
download or an eviction on a phone is the figure the model manager needs, and only a device
run gives it.

**Context cost.** Prefill with about 1,900 tokens of passages took 9.6 to 62 s to the first
token on these runners (median per model in the long-context table). A phone will be slower,
so passage budgets for the 4 GB tier need device measurement before they are set.

### Provisional roles (not assigned)

Observed on this baseline only, pending device runs, a larger task set and approved gates:
Qwen3.5 2B is the only model that fits a 4 GB budget (2.0 GB peak) and is clean on answers,
abstention and tool calls; Qwen3-1.7B fits and is clean when constrained but weaker with
long contexts; Qwen3.5 0.8B fits easily but does not abstain; SmolLM2-1.7B does not meet
the format and tool-call needs; the 4 GB-plus models are candidates for 6 GB and 8 GB+
tiers only.

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
| Device runs, minimum tier | **Samsung Galaxy A06**: 4 GB physical RAM (RAM Plus must be off and recorded), 64 GB, USB; Android version and SoC to be read from the device (believed Helio G85 class, unverified). Needs the owner's go-ahead, `adb` (approval requested, not installed) and an exclusive window shared with the mobile-ui lane; [device/DEVICE_PROCEDURE.md](device/DEVICE_PROCEDURE.md) |
| Device runs, mid and high tiers (6 GB, 8 GB+) | phones not yet named |
| Battery and thermal | device runs only |
| Nigerian-language model | name, source, weights and licence, verified before it enters the registry |
| Nigerian-language quality | native speakers review the Yoruba, Hausa and Igbo prompts, then `runs/desktop/human_review.csv` |
| Dataset B scores | human-verified tasks over the 43 text pages; OCR for the 18 scanned pages |
| Larger task set | 43 tasks is a baseline; differences of one or two tasks are noise |

## 6. Recommendations

1. **Run the Galaxy A06** (4 GB, Android 14+, RAM Plus off) before any model is chosen.
   Peak RAM, cold start, thermal and battery decide the 4 GB tier; the desktop numbers
   only rule models out.
2. **Do not rely on Qwen3.5 0.8B for abstention.** It answered 10 of 10 answerable questions
   but only 1 of 6 unanswerable ones correctly. Before it ships to Free: test it on the
   100-task suite, and consider enforcing abstention outside the model (a retrieval-score
   threshold before generation), measured as a system. Set the abstention bar from the
   baseline, get it approved, then compare.
3. **Treat Qwen3.5 2B and Qwen3-1.7B as the 4 GB-tier candidates to put on the phone first**
   (2.0 and 2.3 GB peak). Ask whether the Free tier's "smallest local model" should be the
   smallest one that clears the abstention bar rather than the smallest one.
4. **Use grammar-constrained JSON on device.** It raised schema validity for eight of the ten
   runs (Phi-4-mini 19% to 87%, SmolLM2 26% to 97%, Qwen3-1.7B 84% to 100%, Qwen3.5 0.8B
   64% to 87%) and left Qwen3.5 4B and Gemma 3n unchanged. The exception is Ministral 3 3B
   (87% free, 84% constrained), which needs a look at its constrained outputs before it is
   trusted. The model manager and tool contracts should assume constrained decoding.
5. **Keep passages to about three per question on the 4 GB tier** until device TTFT is
   known: ten passages cost 10 to 60 s to first token on server CPUs.
6. **Drop SmolLM2-1.7B from the target portfolio unless it is retested** with a prompt
   written for it; on this baseline it fails abstention, citations and tool calls, and it
   has no Nigerian-language support.
7. **Grow to the 100 reviewed tasks** (see the evaluation plan) before ranking models that
   differ by one or two tasks, and have native speakers review `human_review.csv` only if
   Nigerian languages return to scope.
8. **Build the adb-driven runner** so the A06 can be tested without installing Termux on a
   shared phone, and an Android arm64 CI build to feed it.
