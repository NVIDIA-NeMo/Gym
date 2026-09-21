# Agent Security Bench — FDR baseline models

Upstream `agiresearch/ASB` @ `1f561dccf92d` (arXiv:2410.02644). 27 conditions x 400 rows = 10,800 rollouts per model.

Cells are the mean of per-attack-type rates, matching the paper. Rows whose plan never parsed are excluded from ASR and utility and reported separately; see `benchmarks/asb/METRICS.md` for denominators and the five disclosed deviations.


## Agent Attack

| Model | DPI ASR | DPI RR | OPI ASR | OPI RR | Memory Poisoning ASR | Memory Poisoning RR | Mixed Attack ASR | Mixed Attack RR | PoT Backdoor ASR | PoT Backdoor RR | Average ASR | Average RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 86.91% | 15.02% | 34.35% | 4.70% | 14.15% | 0.60% | 96.01% | 8.99% | 31.00% | 7.40% | 52.48% | 7.34% |
| **kimi-k3** | 67.04% | 20.46% | 27.32% | 29.70% | 12.15% | 12.70% | 75.35% | 19.91% | 7.25% | 20.80% | 37.82% | 20.71% |
| **nemotron-3.5-super-vl** | 95.39% | 5.86% | 32.65% | 1.75% | 13.05% | 0.25% | 98.53% | 7.85% | 81.75% | 2.26% | 64.27% | 3.59% |
| **qwen3.5-122b-a10b** | 83.06% | 30.50% | 43.45% | 4.46% | 16.45% | 0.25% | 85.54% | 20.55% | 8.50% | 6.78% | 47.40% | 12.51% |

## Utility controls

| Model | Clean task success | PoT Clean ASR (should be low) |
|---|---|---|
| **nemotron-3-ultra-550b** | 71.25% | 27.25% |
| **kimi-k3** | 65.00% | 8.00% |
| **nemotron-3.5-super-vl** | 87.75% | 47.25% |
| **qwen3.5-122b-a10b** | 72.00% | 14.75% |

## Defenses Against DPI

| Model | DPI ASR | Delimiter ASR-d | Paraphrase ASR-d | Instruction ASR-d |
|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 76.49% | 57.19% | 73.64% | 38.15% |
| **kimi-k3** | 52.75% | 14.00% | 50.75% | 25.75% |
| **nemotron-3.5-super-vl** | 90.08% | 78.55% | 90.37% | 61.15% |
| **qwen3.5-122b-a10b** | 79.73% | 70.08% | 77.89% | 67.20% |

## Defenses Against OPI

| Model | OPI ASR | Delimiter ASR-d | Instruction ASR-d | Sandwich ASR-d |
|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 28.25% | 24.75% | 23.50% | 23.75% |
| **kimi-k3** | 27.53% | 27.27% | 28.64% | 27.59% |
| **nemotron-3.5-super-vl** | 30.75% | 35.25% | 31.75% | 30.50% |
| **qwen3.5-122b-a10b** | 54.25% | 54.75% | 41.50% | 27.50% |

## Coverage and failure accounting

`Landed` is rows that reached the rollouts file; `missing` never did. A model scored on 10,777 of 10,800 rows is not the same measurement as one scored on 10,800, and that difference disappears if only the landed count is published -- so the expected count is stated rather than left to be inferred.

| Model | Expected | Landed | Missing | Scored | Workflow failure | Judge sidecar | Plan salvage |
|---|---|---|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 10800 | 10800 | 0 | 10373 | 427 (4.0%) | 0 | 0.0% |
| **kimi-k3** | 10800 | 10777 | **23** (0.21%) | 10739 | 38 (0.4%) | 0 | 40.1% |
| **nemotron-3.5-super-vl** | 10800 | 10800 | 0 | 10517 | 283 (2.6%) | 0 | 4.2% |
| **qwen3.5-122b-a10b** | 10800 | 10775 | **25** (0.23%) | 10556 | 219 (2.0%) | 0 | 0.1% |

## Per-condition detail

Every condition expects 400 rows. A short `n` is flagged, because a denominator below that is a coverage gap rather than a design choice -- and where the refusal judge caused it, a *biased* gap concentrated in the most adversarial rows. See METRICS.md.

| Model | Condition | n | short by | scored | ASR | RR | Utility | Workflow failure |
|---|---|---|---|---|---|---|---|---|
| nemotron-3-ultra-550b | `clean.combined_attack.no_defense.all` | 400 | - | 400 | 0.00% | 0.25% | 71.25% | 0 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.delimiters_defense.all` | 400 | - | 306 | 57.19% | 7.25% | 24.84% | 94 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.direct_paraphrase_defense.all` | 400 | - | 349 | 73.64% | 8.06% | 10.60% | 51 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.instructional_prevention.all` | 400 | - | 249 | 38.15% | 4.75% | 40.16% | 151 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.no_defense.all` | 400 | - | 353 | 76.49% | 9.02% | 11.61% | 47 |
| nemotron-3-ultra-550b | `direct_prompt_injection.context_ignoring.no_defense.all` | 400 | - | 376 | 85.11% | 20.96% | 3.19% | 24 |
| nemotron-3-ultra-550b | `direct_prompt_injection.escape_characters.no_defense.all` | 400 | - | 395 | 91.14% | 15.66% | 1.27% | 5 |
| nemotron-3-ultra-550b | `direct_prompt_injection.fake_completion.no_defense.all` | 400 | - | 393 | 91.60% | 11.59% | 3.05% | 7 |
| nemotron-3-ultra-550b | `direct_prompt_injection.naive.no_defense.all` | 400 | - | 399 | 90.23% | 17.88% | 1.25% | 1 |
| nemotron-3-ultra-550b | `memory_attack.combined_attack.no_defense.all` | 400 | - | 400 | 14.25% | 1.00% | 66.75% | 0 |
| nemotron-3-ultra-550b | `memory_attack.context_ignoring.no_defense.all` | 400 | - | 400 | 13.00% | 0.50% | 66.75% | 0 |
| nemotron-3-ultra-550b | `memory_attack.escape_characters.no_defense.all` | 400 | - | 400 | 14.00% | 0.75% | 66.50% | 0 |
| nemotron-3-ultra-550b | `memory_attack.fake_completion.no_defense.all` | 400 | - | 400 | 14.75% | 0.50% | 65.50% | 0 |
| nemotron-3-ultra-550b | `memory_attack.naive.no_defense.all` | 400 | - | 400 | 14.75% | 0.25% | 69.75% | 0 |
| nemotron-3-ultra-550b | `mixed_attack.escape_characters.no_defense.all` | 400 | - | 385 | 96.62% | 10.10% | 0.26% | 15 |
| nemotron-3-ultra-550b | `mixed_attack.fake_completion.no_defense.all` | 400 | - | 382 | 96.34% | 6.84% | 0.26% | 18 |
| nemotron-3-ultra-550b | `mixed_attack.naive.no_defense.all` | 400 | - | 386 | 95.08% | 10.03% | 1.04% | 14 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.delimiters_defense.all` | 400 | - | 400 | 24.75% | 6.75% | 65.00% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.instructional_prevention.all` | 400 | - | 400 | 23.50% | 5.76% | 63.75% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.no_defense.all` | 400 | - | 400 | 28.25% | 2.25% | 55.25% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.ob_sandwich_defense.all` | 400 | - | 400 | 23.75% | 5.75% | 57.75% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.context_ignoring.no_defense.all` | 400 | - | 400 | 20.25% | 5.75% | 60.50% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.escape_characters.no_defense.all` | 400 | - | 400 | 48.50% | 6.00% | 43.50% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.fake_completion.no_defense.all` | 400 | - | 400 | 40.25% | 4.00% | 48.75% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.naive.no_defense.all` | 400 | - | 400 | 34.50% | 5.50% | 48.25% | 0 |
| nemotron-3-ultra-550b | `pot_backdoor.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 31.00% | 7.40% | 94.00% | 0 |
| nemotron-3-ultra-550b | `pot_clean.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 27.25% | 0.00% | 92.00% | 0 |
| kimi-k3 | `clean.combined_attack.no_defense.all` | 400 | - | 400 | 0.00% | 6.03% | 65.00% | 0 |
| kimi-k3 | `direct_prompt_injection.combined_attack.delimiters_defense.all` | 400 | - | 400 | 14.00% | 9.80% | 54.25% | 0 |
| kimi-k3 | `direct_prompt_injection.combined_attack.direct_paraphrase_defense.all` | 400 | - | 400 | 50.75% | 14.29% | 31.25% | 0 |
| kimi-k3 | `direct_prompt_injection.combined_attack.instructional_prevention.all` | 400 | - | 400 | 25.75% | 10.10% | 45.50% | 0 |
| kimi-k3 | `direct_prompt_injection.combined_attack.no_defense.all` | 400 | - | 400 | 52.75% | 14.61% | 27.00% | 0 |
| kimi-k3 | `direct_prompt_injection.context_ignoring.no_defense.all` | 400 | - | 400 | 55.00% | 18.09% | 27.25% | 0 |
| kimi-k3 | `direct_prompt_injection.escape_characters.no_defense.all` | 400 | - | 400 | 76.00% | 25.13% | 14.25% | 0 |
| kimi-k3 | `direct_prompt_injection.fake_completion.no_defense.all` | 400 | - | 399 | 75.69% | 21.16% | 11.78% | 1 |
| kimi-k3 | `direct_prompt_injection.naive.no_defense.all` | 400 | - | 400 | 75.75% | 23.31% | 13.25% | 0 |
| kimi-k3 | `memory_attack.combined_attack.no_defense.all` | 400 | - | 400 | 11.00% | 14.04% | 58.00% | 0 |
| kimi-k3 | `memory_attack.context_ignoring.no_defense.all` | 400 | - | 400 | 12.00% | 14.82% | 58.00% | 0 |
| kimi-k3 | `memory_attack.escape_characters.no_defense.all` | 400 | - | 400 | 13.75% | 9.55% | 57.75% | 0 |
| kimi-k3 | `memory_attack.fake_completion.no_defense.all` | 400 | - | 400 | 12.00% | 12.28% | 58.00% | 0 |
| kimi-k3 | `memory_attack.naive.no_defense.all` | 400 | - | 400 | 12.00% | 12.81% | 58.50% | 0 |
| kimi-k3 | `mixed_attack.escape_characters.no_defense.all` | 400 | - | 398 | 74.62% | 22.06% | 12.81% | 2 |
| kimi-k3 | `mixed_attack.fake_completion.no_defense.all` | 400 | - | 367 | 76.57% | 15.37% | 8.99% | 33 |
| kimi-k3 | `mixed_attack.naive.no_defense.all` | 400 | - | 398 | 74.87% | 22.31% | 10.80% | 2 |
| kimi-k3 | `observation_prompt_injection.combined_attack.delimiters_defense.all` | 396 | **-4** | 396 | 27.27% | 32.91% | 61.87% | 0 |
| kimi-k3 | `observation_prompt_injection.combined_attack.instructional_prevention.all` | 391 | **-9** | 391 | 28.64% | 50.39% | 69.82% | 0 |
| kimi-k3 | `observation_prompt_injection.combined_attack.no_defense.all` | 396 | **-4** | 396 | 27.53% | 33.76% | 63.89% | 0 |
| kimi-k3 | `observation_prompt_injection.combined_attack.ob_sandwich_defense.all` | 395 | **-5** | 395 | 27.59% | 39.34% | 67.34% | 0 |
| kimi-k3 | `observation_prompt_injection.context_ignoring.no_defense.all` | 399 | **-1** | 399 | 26.57% | 39.24% | 65.41% | 0 |
| kimi-k3 | `observation_prompt_injection.escape_characters.no_defense.all` | 400 | - | 400 | 27.75% | 25.77% | 59.75% | 0 |
| kimi-k3 | `observation_prompt_injection.fake_completion.no_defense.all` | 400 | - | 400 | 27.50% | 24.74% | 58.50% | 0 |
| kimi-k3 | `observation_prompt_injection.naive.no_defense.all` | 400 | - | 400 | 27.25% | 25.00% | 63.75% | 0 |
| kimi-k3 | `pot_backdoor.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 7.25% | 20.80% | 55.00% | 0 |
| kimi-k3 | `pot_clean.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 8.00% | 9.50% | 55.50% | 0 |
| nemotron-3.5-super-vl | `clean.combined_attack.no_defense.all` | 400 | - | 400 | 0.00% | 0.00% | 87.75% | 0 |
| nemotron-3.5-super-vl | `direct_prompt_injection.combined_attack.delimiters_defense.all` | 400 | - | 373 | 78.55% | 3.00% | 16.35% | 27 |
| nemotron-3.5-super-vl | `direct_prompt_injection.combined_attack.direct_paraphrase_defense.all` | 400 | - | 353 | 90.37% | 3.50% | 8.50% | 47 |
| nemotron-3.5-super-vl | `direct_prompt_injection.combined_attack.instructional_prevention.all` | 400 | - | 381 | 61.15% | 2.76% | 31.23% | 19 |
| nemotron-3.5-super-vl | `direct_prompt_injection.combined_attack.no_defense.all` | 400 | - | 353 | 90.08% | 4.50% | 7.37% | 47 |
| nemotron-3.5-super-vl | `direct_prompt_injection.context_ignoring.no_defense.all` | 400 | - | 360 | 94.72% | 7.25% | 5.56% | 40 |
| nemotron-3.5-super-vl | `direct_prompt_injection.escape_characters.no_defense.all` | 400 | - | 383 | 96.08% | 5.79% | 2.61% | 17 |
| nemotron-3.5-super-vl | `direct_prompt_injection.fake_completion.no_defense.all` | 400 | - | 378 | 98.41% | 4.26% | 1.59% | 22 |
| nemotron-3.5-super-vl | `direct_prompt_injection.naive.no_defense.all` | 400 | - | 383 | 97.65% | 7.50% | 1.31% | 17 |
| nemotron-3.5-super-vl | `memory_attack.combined_attack.no_defense.all` | 400 | - | 400 | 14.00% | 0.50% | 77.25% | 0 |
| nemotron-3.5-super-vl | `memory_attack.context_ignoring.no_defense.all` | 400 | - | 400 | 12.75% | 0.25% | 79.50% | 0 |
| nemotron-3.5-super-vl | `memory_attack.escape_characters.no_defense.all` | 400 | - | 400 | 11.25% | 0.00% | 78.25% | 0 |
| nemotron-3.5-super-vl | `memory_attack.fake_completion.no_defense.all` | 400 | - | 400 | 14.25% | 0.00% | 78.75% | 0 |
| nemotron-3.5-super-vl | `memory_attack.naive.no_defense.all` | 400 | - | 400 | 13.00% | 0.50% | 77.50% | 0 |
| nemotron-3.5-super-vl | `mixed_attack.escape_characters.no_defense.all` | 400 | - | 386 | 97.93% | 8.50% | 1.30% | 14 |
| nemotron-3.5-super-vl | `mixed_attack.fake_completion.no_defense.all` | 400 | - | 382 | 99.21% | 5.78% | 0.52% | 18 |
| nemotron-3.5-super-vl | `mixed_attack.naive.no_defense.all` | 400 | - | 385 | 98.44% | 9.27% | 1.82% | 15 |
| nemotron-3.5-super-vl | `observation_prompt_injection.combined_attack.delimiters_defense.all` | 400 | - | 400 | 35.25% | 1.75% | 76.50% | 0 |
| nemotron-3.5-super-vl | `observation_prompt_injection.combined_attack.instructional_prevention.all` | 400 | - | 400 | 31.75% | 2.75% | 75.00% | 0 |
| nemotron-3.5-super-vl | `observation_prompt_injection.combined_attack.no_defense.all` | 400 | - | 400 | 30.75% | 0.75% | 79.00% | 0 |
| nemotron-3.5-super-vl | `observation_prompt_injection.combined_attack.ob_sandwich_defense.all` | 400 | - | 400 | 30.50% | 1.50% | 76.50% | 0 |
| nemotron-3.5-super-vl | `observation_prompt_injection.context_ignoring.no_defense.all` | 400 | - | 400 | 29.50% | 2.25% | 81.50% | 0 |
| nemotron-3.5-super-vl | `observation_prompt_injection.escape_characters.no_defense.all` | 400 | - | 400 | 33.50% | 1.00% | 78.25% | 0 |
| nemotron-3.5-super-vl | `observation_prompt_injection.fake_completion.no_defense.all` | 400 | - | 400 | 36.25% | 3.00% | 77.75% | 0 |
| nemotron-3.5-super-vl | `observation_prompt_injection.naive.no_defense.all` | 400 | - | 400 | 33.25% | 1.75% | 76.75% | 0 |
| nemotron-3.5-super-vl | `pot_backdoor.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 81.75% | 2.26% | 93.25% | 0 |
| nemotron-3.5-super-vl | `pot_clean.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 47.25% | 0.00% | 93.00% | 0 |
| qwen3.5-122b-a10b | `clean.combined_attack.no_defense.all` | 400 | - | 400 | 0.00% | 0.00% | 72.00% | 0 |
| qwen3.5-122b-a10b | `direct_prompt_injection.combined_attack.delimiters_defense.all` | 392 | **-8** | 371 | 70.08% | 37.37% | 2.43% | 21 |
| qwen3.5-122b-a10b | `direct_prompt_injection.combined_attack.direct_paraphrase_defense.all` | 400 | - | 380 | 77.89% | 37.37% | 1.32% | 20 |
| qwen3.5-122b-a10b | `direct_prompt_injection.combined_attack.instructional_prevention.all` | 397 | **-3** | 375 | 67.20% | 46.83% | 2.67% | 22 |
| qwen3.5-122b-a10b | `direct_prompt_injection.combined_attack.no_defense.all` | 393 | **-7** | 375 | 79.73% | 36.46% | 0.53% | 18 |
| qwen3.5-122b-a10b | `direct_prompt_injection.context_ignoring.no_defense.all` | 400 | - | 365 | 79.18% | 34.45% | 1.10% | 35 |
| qwen3.5-122b-a10b | `direct_prompt_injection.escape_characters.no_defense.all` | 396 | **-4** | 377 | 85.68% | 25.39% | 1.59% | 19 |
| qwen3.5-122b-a10b | `direct_prompt_injection.fake_completion.no_defense.all` | 399 | **-1** | 376 | 85.64% | 28.06% | 0.27% | 23 |
| qwen3.5-122b-a10b | `direct_prompt_injection.naive.no_defense.all` | 399 | **-1** | 382 | 85.08% | 28.13% | 1.05% | 17 |
| qwen3.5-122b-a10b | `memory_attack.combined_attack.no_defense.all` | 400 | - | 400 | 15.25% | 0.25% | 63.50% | 0 |
| qwen3.5-122b-a10b | `memory_attack.context_ignoring.no_defense.all` | 400 | - | 400 | 17.25% | 0.00% | 63.75% | 0 |
| qwen3.5-122b-a10b | `memory_attack.escape_characters.no_defense.all` | 400 | - | 400 | 17.00% | 0.25% | 61.50% | 0 |
| qwen3.5-122b-a10b | `memory_attack.fake_completion.no_defense.all` | 400 | - | 400 | 16.50% | 0.25% | 64.25% | 0 |
| qwen3.5-122b-a10b | `memory_attack.naive.no_defense.all` | 400 | - | 400 | 16.25% | 0.50% | 61.25% | 0 |
| qwen3.5-122b-a10b | `mixed_attack.escape_characters.no_defense.all` | 400 | - | 384 | 85.16% | 19.54% | 0.78% | 16 |
| qwen3.5-122b-a10b | `mixed_attack.fake_completion.no_defense.all` | 399 | **-1** | 385 | 85.71% | 22.37% | 0.26% | 14 |
| qwen3.5-122b-a10b | `mixed_attack.naive.no_defense.all` | 400 | - | 386 | 85.75% | 19.74% | 0.00% | 14 |
| qwen3.5-122b-a10b | `observation_prompt_injection.combined_attack.delimiters_defense.all` | 400 | - | 400 | 54.75% | 11.25% | 34.25% | 0 |
| qwen3.5-122b-a10b | `observation_prompt_injection.combined_attack.instructional_prevention.all` | 400 | - | 400 | 41.50% | 18.30% | 39.25% | 0 |
| qwen3.5-122b-a10b | `observation_prompt_injection.combined_attack.no_defense.all` | 400 | - | 400 | 54.25% | 10.03% | 28.50% | 0 |
| qwen3.5-122b-a10b | `observation_prompt_injection.combined_attack.ob_sandwich_defense.all` | 400 | - | 400 | 27.50% | 6.25% | 53.50% | 0 |
| qwen3.5-122b-a10b | `observation_prompt_injection.context_ignoring.no_defense.all` | 400 | - | 400 | 23.75% | 1.75% | 66.00% | 0 |
| qwen3.5-122b-a10b | `observation_prompt_injection.escape_characters.no_defense.all` | 400 | - | 400 | 48.75% | 4.75% | 42.50% | 0 |
| qwen3.5-122b-a10b | `observation_prompt_injection.fake_completion.no_defense.all` | 400 | - | 400 | 48.25% | 4.25% | 44.75% | 0 |
| qwen3.5-122b-a10b | `observation_prompt_injection.naive.no_defense.all` | 400 | - | 400 | 42.25% | 1.50% | 46.50% | 0 |
| qwen3.5-122b-a10b | `pot_backdoor.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 8.50% | 6.78% | 69.25% | 0 |
| qwen3.5-122b-a10b | `pot_clean.naive.no_defense.all.with_perspicacious_discernment` | 400 | - | 400 | 14.75% | 0.25% | 78.50% | 0 |
