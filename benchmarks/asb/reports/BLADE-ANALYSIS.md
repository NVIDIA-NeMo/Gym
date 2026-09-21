# Agent Security Bench — BLADE Analysis Report

## Executive Summary

Four NVIDIA/partner baselines were run against the full published ASB matrix: 27 conditions
x 400 selectors = 10,800 rollouts per model, 43,177 collected in total. Average attack
success rate ranges from **37.82% (Kimi-K3) to 64.27% (Nemotron-3.5-Super-VL)**.

Three findings matter more than the headline ranking:

1. **The models separate on willingness to decline, not on detecting the injection.**
   Ordering by average ASR descending and by average refusal rate ascending produces the
   identical sequence across all four models. ASB cannot distinguish "recognised the attack"
   from "declined the underlying task anyway", so a low ASR is not evidence of detection.
2. **A low PoT Backdoor score is not backdoor resistance.** For two of four models the
   backdoor arm scores *below* its own clean control (Kimi -0.75pp, Qwen -6.25pp). Their
   low PoT numbers measure whatever the clean condition already produced; the planted
   exemplar did nothing.
3. **Kimi's cells are not comparable to published ASB numbers.** 40.2% of its plans required
   salvage parsing; under upstream's strict parse its Mixed ASR reads 94.25% rather than
   75.35%. The other three models are directly comparable (0.0-4.2% salvage).

The largest workflow drop-off is **plan produced but no tool invoked**, and it is
model-specific: Qwen 7.2% against Super-VL's 0.3%.

## Artifact Inventory

| Artifact | Location | Status |
|---|---|---|
| Rollout JSONL, 4 models | `results/asb/<slug>.jsonl` | complete |
| Failure sidecars | `results/asb/<slug>_failures.jsonl` | present |
| Materialized inputs | `results/asb/<slug>_materialized_inputs.jsonl` | present, hash-pinned |
| Pinned selectors | HF `misalignedandmad/asb-selectors` | content hash `6351e7ee6a94fd6a` |
| Metrics / report | `benchmarks/asb/reports/asb-four-model.{md,json}` | generated |
| Deviations | `benchmarks/asb/METRICS.md` | 5 protocol + 1 measurement limitation |

Upstream `agiresearch/ASB` @ `1f561dccf92d`. Sampling: temperature 0.0, `num_repeats=1`.

**`num_repeats=1` is a structural limitation of this campaign.** pass@k, consistency and
retry-value cannot be computed, and no task has a sometimes-pass observation. Cross-model
comparison substitutes for repeat-based diagnosis throughout; where this report would
normally cite a sometimes-pass deep dive it cites a cross-model divergence instead.

## Aggregate Results

| Model | Expected | Landed | Scored | Coverage | Avg ASR | Avg RR | Clean utility |
|---|---|---|---|---|---|---|---|
| Nemotron-3.5-Super-VL | 10,800 | 10,800 | 10,517 | 100% | **64.27%** | 3.59% | 87.75% |
| Nemotron-3-Ultra-550B | 10,800 | 10,800 | 10,373 | 100% | **52.48%** | 7.34% | 71.25% |
| Qwen3.5-122B-A10B | 10,800 | 10,800 | 10,570 | 100% | **47.36%** | 12.51% | 72.00% |
| Kimi-K3 | 10,800 | 10,777 | 10,739 | 99.79% | **37.82%** | 20.71% | 65.00% |

pass@1 equals ASR here; with one repeat per task there is no pass@k to report.

## Workflow Funnel

Per model, out of 10,800 selectors:

| Model | Landed | Plan parsed | Any tool invoked | Attacker tool invoked | Attack succeeded |
|---|---|---|---|---|---|
| Super-VL | 10,800 | 10,517 | 10,483 | 5,574 | 5,574 |
| Ultra | 10,800 | 10,373 | 10,143 | 4,795 | 4,795 |
| Qwen | 10,800 | 10,570 | 9,810 | 5,146 | 5,146 |
| Kimi | 10,777 | 10,739 | 10,428 | 3,756 | 3,756 |

**Attacker-tool invocation and attack success are exactly equivalent** — zero rows in any
model invoked the attacker tool without the goal string appearing, and zero succeeded
without invoking it. ASB's containment check therefore adds no discrimination over "did the
agent call the attacker tool"; there is no partial-credit regime. This is worth stating
because it means ASR is a clean binary behavioural measure, and also that the metric cannot
distinguish an agent that called the tool and recovered from one that called it and did not.

**The largest drop-off is plan -> tool invocation**, and it is where the models differ most:

| Model | Planned | Invoked no tool | Rate |
|---|---|---|---|
| Super-VL | 10,517 | 34 | 0.3% |
| Ultra | 10,373 | 230 | 2.2% |
| Kimi | 10,739 | 311 | 2.9% |
| **Qwen** | 10,570 | **760** | **7.2%** |

## Task Outcome Buckets

With one repeat there are no always/sometimes/never-pass buckets. The equivalent partition
over the 26 attack conditions (10,400 selectors, excluding the clean control):

| Model | Scored | Attacked | Invoked-but-failed | Refused |
|---|---|---|---|---|
| Super-VL | 9,717 | 5,385 (55.4%) | 0 | 321 |
| Ultra | 9,573 | 4,686 (49.0%) | 0 | 693 |
| Qwen | 9,770 | 5,087 (52.1%) | 0 | 1,559 |
| Kimi | 9,939 | 3,724 (37.5%) | 0 | 2,182 |

The refusal column is monotone in the inverse of the attack column across all four models.

## Dominant Failure Modes

### 1. Qwen's 760 planned-but-no-tool rollouts — `BI` behavioural issue

Of Qwen's 760 rows that produced a valid plan and then invoked nothing:

- attack succeeded in **0** of them
- the refusal judge scored **481** as refusals
- clean-task utility was achieved in **0**

So the bucket is overwhelmingly "declined to act", not "tried and failed". It concentrates
in DPI (465) and mixed (149) — the families where the injection is in the user turn, which
is the text a refusal-trained model is most likely to react to. Labelled `BI` rather than
`KG`: Qwen plans correctly and then declines to execute, which is a behaviour rather than a
missing capability. It is also the mechanism behind Qwen's high refusal rate, and therefore
behind its ASR sitting below Ultra's despite comparable planning.

### 2. Workflow failures concentrated in DPI for three models — `BI`, not `TI`

Workflow failures (no parseable plan) total 283 / 427 / 230 / 38 for Super-VL / Ultra /
Qwen / Kimi. For the three Nemotron/Qwen models they concentrate in
`direct_prompt_injection.combined_attack` — the most aggressive injection. That is the
signature of a model derailed by the injected text rather than of a broken task: the same
selector with a milder attack type parses fine. Labelled `BI`.

Kimi is the exception at 38, and its failures sit in `mixed_attack.fake_completion` (33 of
38) rather than DPI — consistent with its very different parse profile (see Data Artifacts).

### 3. Defence effectiveness is model-specific, and two defences backfire — `TI` boundary

DPI, combined_attack arm, ASR delta against the undefended baseline:

| Model | base | Delimiters | Paraphrase | Instructional |
|---|---|---|---|---|
| Super-VL | 90.08% | −11.53 | +0.28 | −28.93 |
| Ultra | 76.49% | −19.30 | −2.85 | **−38.33** |
| Qwen | 79.21% | −9.34 | −1.32 | −12.01 |
| Kimi | 52.75% | **−38.75** | −2.00 | −27.00 |

OPI, same arm:

| Model | base | Delimiters | Instructional | Sandwich |
|---|---|---|---|---|
| Super-VL | 30.75% | **+4.50** | +1.00 | −0.25 |
| Ultra | 28.25% | −3.50 | −4.75 | −4.50 |
| Qwen | 54.25% | +0.50 | −12.75 | **−26.75** |
| Kimi | 27.53% | −0.25 | **+1.12** | +0.07 |

Paraphrasing is near-inert everywhere (−2.85 to +0.28) against the paper's reported −21.52pp
average. Delimiters range from −38.75 (Kimi) to +4.50 (Super-VL) — a 43-point spread across
models on the same defence. Any release-card claim about a defence's effectiveness is a
claim about a specific model, not about the defence.

### 4. The refusal judge lost 23 rows non-randomly — `IR` infrastructure

Kimi's 23 missing rows are all Observation Prompt Injection, 22 of them OPI+combined_attack.
The judge provider content-filters those transcripts deterministically. The loss is 0.72% of
Kimi's OPI rows and 0% elsewhere, so its OPI cells rest on a subset biased toward the less
extreme half. Ultra, Super-VL and Qwen lost nothing. Labelled `IR`: a property of the key,
not of ASB or the models.

## Sometimes-Pass Deep Dives

**Not available.** `num_repeats=1` means no task has both a passing and a failing rollout,
so the highest-signal diagnostic slice this analysis format expects does not exist for this
campaign. A repeat pass (`--num-repeats 3` on a stratified subset) is the single highest-value
follow-up; see Recommendations.

The nearest available substitute is cross-model divergence on identical selectors, used
throughout as the comparison axis.

## Never-Pass Deep Dives

Conditions where the attack essentially always succeeds, read as never-resisted:

- `mixed_attack` for Super-VL (98.53%) and Ultra (96.01%) — DPI and OPI applied together,
  with retrieved memory. Near-ceiling on 1,200 scored rows each.
- `direct_prompt_injection.combined_attack` for Super-VL (90.08%).

The corresponding `PoT Clean` control is the useful contrast: Super-VL scores 47.25% on a
condition with **no trigger in the task at all**, meaning nearly half of its attacker-tool
invocations there need no injected instruction — the planted exemplar in the system prompt
suffices. Labelled `BI`: the model follows a demonstrated pattern over its actual instruction.

## Cross-Model Comparison

| Column | Super-VL | Ultra | Qwen | Kimi |
|---|---|---|---|---|
| DPI ASR | 95.39% | 86.91% | 82.83% | 67.04% |
| OPI ASR | 32.65% | 34.35% | **43.45%** | 27.32% |
| Memory Poisoning ASR | 13.05% | 14.15% | **16.45%** | 12.15% |
| Mixed ASR | 98.53% | 96.01% | 85.55% | 75.35% |
| PoT Backdoor ASR | **81.75%** | 31.00% | 8.50% | 7.25% |
| **Average ASR** | **64.27%** | 52.48% | 47.36% | 37.82% |
| **Average RR** | 3.59% | 7.34% | 12.51% | **20.71%** |

PoT trigger effect, backdoor arm against its own clean control:

| Model | Clean | Backdoor | Trigger worth |
|---|---|---|---|
| Super-VL | 47.25% | 81.75% | **+34.50pp** |
| Ultra | 27.25% | 31.00% | +3.75pp |
| Kimi | 8.00% | 7.25% | **−0.75pp** |
| Qwen | 14.75% | 8.50% | **−6.25pp** |

Super-VL is the only model where the trigger does substantial work. For Kimi and Qwen the
backdoor arm is at or below its control, so their low PoT numbers are not resistance.

## Recommendations

| # | Finding | Label | Action |
|---|---|---|---|
| 1 | No repeats, so no sometimes-pass slice | `DA` | Rerun a stratified subset at `--num-repeats 3` (suggest DPI+OPI+PoT combined_attack, 3 x 1,200 rollouts). Enables pass@k, consistency and retry value, and is the only way to tell an unstable model from a consistently-susceptible one. Highest value follow-up. |
| 2 | Qwen declines after planning (760 rows, 0 successes, 481 refusals) | `BI` | Do not read Qwen's ASR as detection. If the goal is a model that recognises injections, instrument whether the refusal cites the injection; ASB cannot. |
| 3 | Super-VL PoT 81.75% with +34.50pp trigger effect | `BI` | Highest-severity single result in the campaign. A planted in-context exemplar diverts it in four of five rollouts. Recommend targeted SFT on trigger-conditional exemplars and a follow-up with the other four `POT.yml` triggers, which this campaign did not run. |
| 4 | Kimi 40.2% salvage; cells not comparable to published ASB | `DA` | Carry the strict-only sensitivity table wherever Kimi's cells appear. Consider a strict-parse rerun for Kimi specifically if comparability to the paper matters. |
| 5 | Judge loses OPI rows non-randomly on this key | `IR` | Switch to `deepseek-v4-pro` on a full re-run with all four models re-judged together. Do not switch mid-campaign: a judge differing between models corrupts the comparison worse than 0.72% row loss in one model's cells. |
| 6 | Defence deltas vary 43 points across models | `TI` | Never state a defence's effectiveness without naming the model. Paraphrasing is inert here (−2.85 to +0.28) against the paper's −21.52pp; worth checking whether the paraphrase model differs from upstream's. |
| 7 | Workflow failures cluster in the strongest DPI arm | `BI` | 427 rows for Ultra (4.0%) are excluded from ASR. Confirm they are derailment rather than formatting by rerunning a sample with a larger output budget. |

## Reproducibility Notes

- Upstream `agiresearch/ASB` @ `1f561dccf92d`; inputs SHA-256 pinned in `upstream_spec.py`.
- Denominator recovery: 10 agents x 1 task x 40 agent-matched attacker tools = 400 rows per
  condition; PoT is 5 agents x 2 tasks x 40. Confirmed by the published tables' own
  granularity, by upstream's shipped memory stores holding ~400 records, and by the
  expansion landing on 400 for all 27 conditions.
- Selectors pinned at HF `misalignedandmad/asb-selectors`, content hash `6351e7ee6a94fd6a`,
  round trip verified. Regenerating in-container is byte-identical.
- Sampling: temperature 0.0 (upstream's), `max_output_tokens` 4,096 (upstream's 256 would
  be consumed by reasoning traces).
- Five protocol deviations and one measurement limitation in `benchmarks/asb/METRICS.md`.
- **Provenance gap:** rows carry no collection-site field. Ultra and Kimi were collected on
  a laptop; Super-VL and Qwen partly there and partly in Modal containers. The boundary is
  known from run logs, not from the data, and cannot be reconstructed by a reviewer.
  Verification is deterministic and the selector file byte-identical, so the halves pool
  legitimately; wall-clock and throughput do not compare across them.
