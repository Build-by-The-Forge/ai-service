# AI-12 offline AI benchmark

Measures which small models and runtimes can give BUILD's students useful, grounded,
offline answers on their own phones (PRD §0.6, §146). It is a benchmark, not a mobile
integration. Results and the recommendation are in [REPORT.md](REPORT.md).

## What it measures

| Area | How |
|---|---|
| Download size | file size of the pinned model file |
| Peak RAM | kernel `VmHWM` of llama-server (LiteRT-LM: whole process peak) |
| Cold start | spawn until the model reports loaded |
| TTFT, tokens/s | streamed requests plus server timings; llama-bench pp512/tg128 |
| JSON reliability | strict and lenient parse, schema validity; free vs grammar-constrained |
| Tool calls | tool name and argument correctness, including "no tool needed" |
| Grounded answers | answer facts, citation precision and faithfulness |
| Abstention | unanswerable questions: abstained vs hallucinated |
| Context handling | the same questions with all ten passages (`long` mode) |
| Consistency | three temperature-0.7 runs compared with the greedy run |
| Nigerian languages | Yoruba, Hausa and Igbo grounded tasks and explanations; a language-share proxy plus a native-speaker review sheet |
| Battery, thermal | device runs only (see `device/DEVICE_PROCEDURE.md`) |

## Layout

| Path | Contents |
|---|---|
| `tasks/` | dataset A: 10 original course-style passages, 43 tasks, 4 tool definitions (no third-party text) |
| `scoring.py` | deterministic scorers (stdlib only) |
| `bench.py` | llama.cpp runner (llama-server + llama-bench) |
| `litert_bench.py` | Google AI Edge LiteRT-LM runner (`.litertlm` models) |
| `report.py` | comparison tables and the native-speaker review CSV |
| `models.json`, `fetch_model.py` | pinned registry (revision, size, sha256, licence, card) and verifying fetcher |
| `pdf_eval.py` | dataset B: local-only evaluation over real course PDFs, human-verified tasks only |
| `desktop/` | llama.cpp build (rootless script or Dockerfile) and the desktop run script |
| `device/` | Android procedure: Termux build, device run, adb battery and thermal monitor |
| `runs/` | result files; `local/` (git-ignored) holds dataset B |

## Desktop quick start (Linux or WSL)

```bash
bash benchmarks/offline/desktop/build_llama_cpp.sh          # llama.cpp v1.0.0, pinned
bash benchmarks/offline/desktop/run_desktop.sh qwen3.5-0.8b-q4km
python benchmarks/offline/report.py benchmarks/offline/runs/desktop \
  --tables benchmarks/offline/runs/desktop/TABLES.md \
  --review benchmarks/offline/runs/desktop/human_review.csv
```

Model downloads need the user's approval of the file, source and size. Weights never
enter the repository (`.gitignore`).

## Rules for reading the numbers

- Desktop numbers rank candidates. They are not phone numbers: decide on phones only
  from device runs.
- A llama.cpp row and a LiteRT-LM row for the same model family are different
  deployment paths, not the same measurement twice.
- Yoruba, Hausa and Igbo prompts are `unreviewed`. Make no language-quality claim
  until native speakers have reviewed both the prompts and `human_review.csv`.
- The task set is a baseline (43 tasks). Small differences between models are noise
  at this size. Grow it, and dataset B, before a final choice.
