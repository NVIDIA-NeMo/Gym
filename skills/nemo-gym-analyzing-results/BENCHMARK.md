# Skill Benchmark: nemo-gym-analyzing-results

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-gym-analyzing-results`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 22 evaluation tasks (12 positive, 10 negative)
- Dataset digest: `sha256:9ba64109e5762407a9bc346ce41485130c46c287fd4f9c173b5e48de0fea2367` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemo-gym-analyzing-results | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-analyzing-results | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nemo-gym-analyzing-results`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.7% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.7% → 100.0% (+2.3 points) | 100.0% → 95.5% (-4.5 points) |
| Correctness | 86.4% → 92.7% (+6.3 points) | 95.5% → 97.3% (+1.8 points) |
| Discoverability | 95.8% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 77.6% → 87.9% (+10.3 points) | 89.0% → 86.4% (-2.6 points) |
| Efficiency | 96.9% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.4% → 93.7% (-1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.7% → 83.7% (+1.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,232,442 | 3,804,846 | +427,596 | +11.24% | skill 22/22; base 22/22 |
| claude-code | baseline-counted-failures-as-zero | 397,161 | 162,942 | +234,219 | +143.74% | skill 1/1; base 1/1 |
| claude-code | different-task-sets | 275,306 | 209,722 | +65,584 | +31.27% | skill 1/1; base 1/1 |
| claude-code | failed-rollouts-inflate-candidate | 240,572 | 199,625 | +40,947 | +20.51% | skill 1/1; base 1/1 |
| claude-code | flipped-task-is-noise | 269,416 | 319,724 | -50,308 | -15.73% | skill 1/1; base 1/1 |
| claude-code | gap-to-published-number | 241,376 | 208,430 | +32,946 | +15.81% | skill 1/1; base 1/1 |
| claude-code | healthy-but-unchecked | 239,118 | 201,358 | +37,760 | +18.75% | skill 1/1; base 1/1 |
| claude-code | is-candidate-better-interval-contains-zero | 231,130 | 185,897 | +45,233 | +24.33% | skill 1/1; base 1/1 |
| claude-code | judge-failures-in-score | 123,927 | 143,369 | -19,442 | -13.56% | skill 1/1; base 1/1 |
| claude-code | negative-add-public-benchmark | 150,700 | 71,947 | +78,753 | +109.46% | skill 1/1; base 1/1 |
| claude-code | negative-choose-training-tasks | 36,178 | 107,766 | -71,588 | -66.43% | skill 1/1; base 1/1 |
| claude-code | negative-custom-metric | 187,219 | 532,656 | -345,437 | -64.85% | skill 1/1; base 1/1 |
| claude-code | negative-plan-repeats-in-advance | 36,844 | 72,549 | -35,705 | -49.22% | skill 1/1; base 1/1 |
| claude-code | negative-resume-killed-run | 185,146 | 107,495 | +77,651 | +72.24% | skill 1/1; base 1/1 |
| claude-code | negative-run-a-benchmark | 193,662 | 106,947 | +86,715 | +81.08% | skill 1/1; base 1/1 |
| claude-code | negative-run-cannot-connect | 71,587 | 107,506 | -35,919 | -33.41% | skill 1/1; base 1/1 |
| claude-code | negative-slow-evaluation | 36,603 | 36,298 | +305 | +0.84% | skill 1/1; base 1/1 |
| claude-code | negative-training-reward-flat | 36,515 | 72,026 | -35,511 | -49.30% | skill 1/1; base 1/1 |
| claude-code | negative-welch-function | 35,864 | 35,802 | +62 | +0.17% | skill 1/1; base 1/1 |
| claude-code | pass-at-k-drops-tasks | 329,698 | 282,333 | +47,365 | +16.78% | skill 1/1; base 1/1 |
| claude-code | single-repeat-no-interval | 221,036 | 149,580 | +71,456 | +47.77% | skill 1/1; base 1/1 |
| claude-code | tasks-that-never-pass | 323,816 | 253,133 | +70,683 | +27.92% | skill 1/1; base 1/1 |
| claude-code | tasks-that-sometimes-pass | 369,568 | 237,741 | +131,827 | +55.45% | skill 1/1; base 1/1 |
| codex | All cases | 3,234,867 | 3,044,488 | +190,379 | +6.25% | skill 22/22; base 22/22 |
| codex | baseline-counted-failures-as-zero | 281,173 | 335,728 | -54,555 | -16.25% | skill 1/1; base 1/1 |
| codex | different-task-sets | 200,426 | 266,995 | -66,569 | -24.93% | skill 1/1; base 1/1 |
| codex | failed-rollouts-inflate-candidate | 316,185 | 301,015 | +15,170 | +5.04% | skill 1/1; base 1/1 |
| codex | flipped-task-is-noise | 189,684 | 117,012 | +72,672 | +62.11% | skill 1/1; base 1/1 |
| codex | gap-to-published-number | 296,918 | 224,223 | +72,695 | +32.42% | skill 1/1; base 1/1 |
| codex | healthy-but-unchecked | 165,349 | 216,050 | -50,701 | -23.47% | skill 1/1; base 1/1 |
| codex | is-candidate-better-interval-contains-zero | 157,120 | 170,748 | -13,628 | -7.98% | skill 1/1; base 1/1 |
| codex | judge-failures-in-score | 94,695 | 81,900 | +12,795 | +15.62% | skill 1/1; base 1/1 |
| codex | negative-add-public-benchmark | 52,886 | 80,774 | -27,888 | -34.53% | skill 1/1; base 1/1 |
| codex | negative-choose-training-tasks | 84,861 | 73,071 | +11,790 | +16.13% | skill 1/1; base 1/1 |
| codex | negative-custom-metric | 79,917 | 82,617 | -2,700 | -3.27% | skill 1/1; base 1/1 |
| codex | negative-plan-repeats-in-advance | 18,693 | 38,439 | -19,746 | -51.37% | skill 1/1; base 1/1 |
| codex | negative-resume-killed-run | 65,860 | 51,941 | +13,919 | +26.80% | skill 1/1; base 1/1 |
| codex | negative-run-a-benchmark | 74,861 | 95,131 | -20,270 | -21.31% | skill 1/1; base 1/1 |
| codex | negative-run-cannot-connect | 74,299 | 78,598 | -4,299 | -5.47% | skill 1/1; base 1/1 |
| codex | negative-slow-evaluation | 60,875 | 82,079 | -21,204 | -25.83% | skill 1/1; base 1/1 |
| codex | negative-training-reward-flat | 35,395 | 55,750 | -20,355 | -36.51% | skill 1/1; base 1/1 |
| codex | negative-welch-function | 18,155 | 13,577 | +4,578 | +33.72% | skill 1/1; base 1/1 |
| codex | pass-at-k-drops-tasks | 171,111 | 150,982 | +20,129 | +13.33% | skill 1/1; base 1/1 |
| codex | single-repeat-no-interval | 99,734 | 90,950 | +8,784 | +9.66% | skill 1/1; base 1/1 |
| codex | tasks-that-never-pass | 378,584 | 252,008 | +126,576 | +50.23% | skill 1/1; base 1/1 |
| codex | tasks-that-sometimes-pass | 318,086 | 184,900 | +133,186 | +72.03% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,467,309 | 6,849,334 | +617,975 | +9.02% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (5814 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nemo-gym-analyzing-results/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (474 chars, recommend 50-150) (`skills/nemo-gym-analyzing-results/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nemo-gym-analyzing-results/SKILL.md`)
- **LOW** SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/nemo-gym-analyzing-results/SKILL.md`)

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
