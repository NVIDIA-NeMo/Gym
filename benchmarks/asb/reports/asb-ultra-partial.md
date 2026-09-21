# Agent Security Bench — FDR baseline models

Upstream `agiresearch/ASB` @ `1f561dccf92d` (arXiv:2410.02644). 27 conditions x 400 rows = 10,800 rollouts per model.

Cells are the mean of per-attack-type rates, matching the paper. Rows whose plan never parsed are excluded from ASR and utility and reported separately; see `benchmarks/asb/METRICS.md` for denominators and the four disclosed deviations.


## Agent Attack

| Model | DPI ASR | DPI RR | OPI ASR | OPI RR | Memory Poisoning ASR | Memory Poisoning RR | Mixed Attack ASR | Mixed Attack RR | PoT Backdoor ASR | PoT Backdoor RR | Average ASR | Average RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 86.91% | 15.02% | 34.35% | 4.70% | 14.15% | 0.60% | 96.01% | 8.99% | 31.00% | 7.40% | 52.48% | 7.34% |

## Utility controls

| Model | Clean task success | PoT Clean ASR (should be low) |
|---|---|---|
| **nemotron-3-ultra-550b** | 71.25% | 27.25% |

## Defenses Against DPI

| Model | DPI ASR | Delimiter ASR-d | Paraphrase ASR-d | Instruction ASR-d |
|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 76.49% | 57.19% | 73.64% | 38.15% |

## Defenses Against OPI

| Model | OPI ASR | Delimiter ASR-d | Instruction ASR-d | Sandwich ASR-d |
|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 28.25% | 24.75% | 23.50% | 23.75% |

## Coverage and failure accounting

| Model | Rows | Scored | Workflow failure | Judge sidecar | Plan salvage |
|---|---|---|---|---|---|
| **nemotron-3-ultra-550b** | 10800 | 10373 | 427 (4.0%) | 0 | 0.0% |

## Per-condition detail

| Model | Condition | n | scored | ASR | RR | Utility | Workflow failure |
|---|---|---|---|---|---|---|---|
| nemotron-3-ultra-550b | `clean.combined_attack.no_defense.all` | 400 | 400 | 0.00% | 0.25% | 71.25% | 0 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.delimiters_defense.all` | 400 | 306 | 57.19% | 7.25% | 24.84% | 94 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.direct_paraphrase_defense.all` | 400 | 349 | 73.64% | 8.06% | 10.60% | 51 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.instructional_prevention.all` | 400 | 249 | 38.15% | 4.75% | 40.16% | 151 |
| nemotron-3-ultra-550b | `direct_prompt_injection.combined_attack.no_defense.all` | 400 | 353 | 76.49% | 9.02% | 11.61% | 47 |
| nemotron-3-ultra-550b | `direct_prompt_injection.context_ignoring.no_defense.all` | 400 | 376 | 85.11% | 20.96% | 3.19% | 24 |
| nemotron-3-ultra-550b | `direct_prompt_injection.escape_characters.no_defense.all` | 400 | 395 | 91.14% | 15.66% | 1.27% | 5 |
| nemotron-3-ultra-550b | `direct_prompt_injection.fake_completion.no_defense.all` | 400 | 393 | 91.60% | 11.59% | 3.05% | 7 |
| nemotron-3-ultra-550b | `direct_prompt_injection.naive.no_defense.all` | 400 | 399 | 90.23% | 17.88% | 1.25% | 1 |
| nemotron-3-ultra-550b | `memory_attack.combined_attack.no_defense.all` | 400 | 400 | 14.25% | 1.00% | 66.75% | 0 |
| nemotron-3-ultra-550b | `memory_attack.context_ignoring.no_defense.all` | 400 | 400 | 13.00% | 0.50% | 66.75% | 0 |
| nemotron-3-ultra-550b | `memory_attack.escape_characters.no_defense.all` | 400 | 400 | 14.00% | 0.75% | 66.50% | 0 |
| nemotron-3-ultra-550b | `memory_attack.fake_completion.no_defense.all` | 400 | 400 | 14.75% | 0.50% | 65.50% | 0 |
| nemotron-3-ultra-550b | `memory_attack.naive.no_defense.all` | 400 | 400 | 14.75% | 0.25% | 69.75% | 0 |
| nemotron-3-ultra-550b | `mixed_attack.escape_characters.no_defense.all` | 400 | 385 | 96.62% | 10.10% | 0.26% | 15 |
| nemotron-3-ultra-550b | `mixed_attack.fake_completion.no_defense.all` | 400 | 382 | 96.34% | 6.84% | 0.26% | 18 |
| nemotron-3-ultra-550b | `mixed_attack.naive.no_defense.all` | 400 | 386 | 95.08% | 10.03% | 1.04% | 14 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.delimiters_defense.all` | 400 | 400 | 24.75% | 6.75% | 65.00% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.instructional_prevention.all` | 400 | 400 | 23.50% | 5.76% | 63.75% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.no_defense.all` | 400 | 400 | 28.25% | 2.25% | 55.25% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.combined_attack.ob_sandwich_defense.all` | 400 | 400 | 23.75% | 5.75% | 57.75% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.context_ignoring.no_defense.all` | 400 | 400 | 20.25% | 5.75% | 60.50% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.escape_characters.no_defense.all` | 400 | 400 | 48.50% | 6.00% | 43.50% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.fake_completion.no_defense.all` | 400 | 400 | 40.25% | 4.00% | 48.75% | 0 |
| nemotron-3-ultra-550b | `observation_prompt_injection.naive.no_defense.all` | 400 | 400 | 34.50% | 5.50% | 48.25% | 0 |
| nemotron-3-ultra-550b | `pot_backdoor.naive.no_defense.all.with_perspicacious_discernment` | 400 | 400 | 31.00% | 7.40% | 94.00% | 0 |
| nemotron-3-ultra-550b | `pot_clean.naive.no_defense.all.with_perspicacious_discernment` | 400 | 400 | 27.25% | 0.00% | 92.00% | 0 |
