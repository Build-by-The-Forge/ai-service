### Footprint and speed

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

### Structured output and tools (free = instruction only; schema = grammar-constrained)

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

### Grounded answers in English (10 answerable, 6 unanswerable)

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

### Nigerian languages (unreviewed prompts; 3 grounded tasks + 1 explanation each)

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
