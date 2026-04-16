# Arena Resources Server

Pairwise LLM-judge benchmark for measuring general chat quality. The policy model's response is evaluated against a fixed baseline using a two-game comparison to cancel positional bias.

Supports multiple datasets in the same server:
- **lmarena-260311** — 1,475 multi-lingual questions, single baseline (`configs/lmarena_260311.yaml`)
- **arena-hard-v2.0** — 750 questions across `hard_prompt` and `creative_writing`, per-category baselines (`configs/arena_hard_v2.yaml`)

## Benchmark

- **Questions:** Open-ended questions spanning diverse topics (practical advice, creative writing, coding, reasoning, etc.)
- **Baseline:** Pre-computed answers from a fixed reference model (or per-category models), stored in the JSONL dataset
- **Evaluation:** LLM-as-a-judge pairwise comparison (policy vs. baseline)

## Reward Formula

Two games are played per question to cancel positional bias:
- Game 1: policy model = **A**, baseline = **B**
- Game 2: baseline = **A**, policy model = **B**

Each game produces a verdict. Strong verdicts (`[[A>>B]]`, `[[B>>A]]`) count `verdict_weight` times (default: 3) in the average:

```
scores = weighted_scores(game1, perspective=A) + weighted_scores(game2, perspective=B)
reward = mean(scores)
```

| Game 1 verdict | Game 2 verdict | Reward (weight=3) |
|----------------|----------------|-------------------|
| `[[A>>B]]` | `[[B>>A]]` | 1.0 |
| `[[A>B]]` | `[[B>A]]` | 1.0 |
| `[[A>>B]]` | `[[A=B]]` | 0.875 |
| `[[A>B]]` | `[[A=B]]` | 0.75 |
| `[[A=B]]` | `[[A=B]]` | 0.5 |
| `[[B>A]]` | `[[B>A]]` | 0.5 |
| `[[A=B]]` | `[[A>B]]` | 0.25 |
| `[[B>A]]` | `[[A>B]]` | 0.0 |
| `[[B>>A]]` | `[[A>>B]]` | 0.0 |

The `mean/reward` reported by nemo-gym is the mean across all tasks weighted this way. Note, this is different from `win_rate` which pools all weighted score items before computing a Bradley-Terry win probability (see Output Metrics below).

## Data Schema

Each JSONL line:
```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "<question>"}, ...]
  },
  "question_id": "<uid>",
  "question": "<question text, passed verbatim to judge>",
  "baseline_answer": "<baseline model's pre-computed answer>",
  "category": "<category>"
}
```

`baseline_answer` can be either a plain string (v0.1 format) or a dict `{"answer": "<text>"}` (v0.2 format, which allows additional metadata fields). Both are accepted.

`category` is omitted for single-category datasets (lmarena-260311). For multi-category datasets (arena-hard-v2.0), it is used to select the per-category baseline and style constants.

`responses_create_params.input` contains the full conversation history (all messages) so that the policy model receives the correct context for multi-turn questions. For single-turn questions this is a single user message. The `question` field contains the same text for single-turn questions, or the flattened conversation (`[User]: ...\n\n[Assistant]: ...\n\n[User]: ...`) for multi-turn questions.

`verifier_metadata` is not used — `question`, `question_id`, `baseline_answer`, and `category` are top-level fields.

## Configuration

Requires two model servers:
- `policy_model` — the model being evaluated (any `responses_api_models` variant)
- `judge_model` — a strong judge model; uses `openai_model` (see below)

Example endpoint configuration in `env.yaml`:
```yaml
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: sk-...
policy_model_name: nvidia/nvidia/nemotron-3-super-v3
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: sk-...
judge_model_name: gcp/google/gemini-3.1-pro-preview
```

### Judge defaults

The judge is configured with `temperature: 0.0` (deterministic) and `max_output_tokens: 65536`. These are set in the dataset config YAML under `judge_responses_create_params` and do not need to be passed on the command line.

For the policy model, `max_output_tokens: 65536` is the recommended value in the workflow commands below.

### `style_control`

Optional config field (default `true`). When `true`, `win_rate` is a Bradley-Terry win probability after removing length/formatting bias via pre-computed style offsets. When `false`, `win_rate` is a bootstrap mean of weighted scores.

```yaml
lmarena_260311:
  resources_servers:
    arena:
      style_control: true
```

### `verdict_weight`

Optional config field (default `3`). Increase to give strong verdicts more influence; set to `1` for a simple unweighted average. Override in YAML:

```yaml
lmarena_260311:
  resources_servers:
    arena:
      verdict_weight: 3
```

### `max_rollout_failure_rate`

Optional config field (default `0.01` = 1%). `compute_metrics` raises a `ValueError` if the fraction of failed rollouts exceeds this threshold, since scores computed from too many failures are unreliable.

A rollout is considered failed if gen_answer produced no output, or if any of its two judge games returned an unparseable verdict. Both cases count toward `rollout_failure_rate`. Self-comparison rollouts are excluded from this denominator (a warning is logged if any are detected).

Override in YAML:

```yaml
lmarena_260311:
  resources_servers:
    arena:
      max_rollout_failure_rate: 0.01
```

### `judge_concurrency`

Optional config field (default `16`). Maximum number of concurrent HTTP calls to the judge endpoint. Each `verify()` call launches 2 judge games concurrently, so the effective question-level parallelism is `judge_concurrency / 2`.

Lower this if the judge returns 429 rate-limit errors. The policy model 504 timeouts are unrelated and controlled by the framework-level parallelism of `ng_collect_rollouts`.

```yaml
lmarena_260311:
  resources_servers:
    arena:
      judge_concurrency: 16
```

## Implementation Notes

### Judge prompt

The judge system prompt instructs the judge to:
1. Generate its own answer to the question first
2. Compare both assistants' answers against its own
3. Output exactly one of six verdict labels as its final verdict

The user message follows this template:
```
<|Conversation History|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>
```

### Verdict extraction

Six verdict labels are recognised:

| Label | Meaning |
|-------|---------|
| `[[A>>B]]` | A significantly better |
| `[[A>B]]` | A slightly better |
| `[[A=B]]` | Tie |
| `[[B>A]]` | B slightly better |
| `[[B>>A]]` | B significantly better |
| `[[BB]]` | Both bad |

The **rightmost** occurrence in the judge's response is taken as the verdict. Judges typically state their final decision at the end after reasoning, so rightmost is more reliable than first. If no label is found, `verdict` is `None` (parse failure, scored as 0.0).

### Thinking block stripping

Before the policy answer is sent to the judge, `<think>…</think>` and `<thinking>…</thinking>` blocks are stripped. This ensures reasoning models (which emit internal chain-of-thought in these tags) are judged only on their visible answer, matching what a user would see.

### Bootstrap parameters

`win_rate` uses 100 bootstrap rounds with seed 42 for reproducibility.

For **multi-category datasets** (e.g. arena-hard-v2.0 with `hard_prompt` and `creative_writing`), each bootstrap round resamples all categories independently and averages their per-category BT win rates into one combined value. The CI (`win_rate_ci_lower`, `win_rate_ci_upper`) is then the 2.5th/97.5th percentile of those 100 combined values — it is the CI of the joint average, not the average of per-category CIs. This gives equal weight to each category regardless of question count.

For **single-category datasets** (e.g. lmarena-260311, where all questions share one category), this is equivalent to a standard per-question bootstrap.

## Full Evaluation Workflow

### 1. Start servers (keep running in a separate terminal)

```bash
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/arena/configs/lmarena_260311.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

Use `openai_model` for the policy if it serves `/v1/responses`; use `vllm_model` if it serves `/v1/chat/completions`.

Check all servers are healthy:
```bash
ng_status
```

### 2. Smoke test (quick sanity check)

Two options depending on whether you have the full dataset downloaded:

**Without full dataset** — use the 5 committed example entries:
```bash
ng_collect_rollouts \
    +agent_name=lmarena_260311_simple_agent \
    +input_jsonl_fpath=resources_servers/arena/data/lmarena_260311/example.jsonl \
    +output_jsonl_fpath=results/<run_name>/rollouts.jsonl \
    +num_repeats=1 \
    "+responses_create_params={max_output_tokens: 65536, temperature: 1.0}"
```

**With full dataset** — take the first 10 questions:
```bash
ng_collect_rollouts \
    +agent_name=lmarena_260311_simple_agent \
    +input_jsonl_fpath=resources_servers/arena/data/lmarena_260311/lmarena_260311_validation.jsonl \
    +output_jsonl_fpath=results/<run_name>/rollouts.jsonl \
    +num_repeats=1 \
    +limit=10 \
    "+responses_create_params={max_output_tokens: 65536, temperature: 1.0}"
```

### 3. Full evaluation (1,475 questions)

```bash
ulimit -n 65536
ng_collect_rollouts \
    +agent_name=lmarena_260311_simple_agent \
    +input_jsonl_fpath=resources_servers/arena/data/lmarena_260311/lmarena_260311_validation.jsonl \
    +output_jsonl_fpath=results/<run_name>/rollouts.jsonl \
    +num_repeats=1 \
    +num_samples_in_parallel=8 \
    +resume_from_cache=true \
    "+responses_create_params={max_output_tokens: 65536, temperature: 1.0}"
```

- `ulimit -n 65536` — raises the open file descriptor limit; needed when aiohttp opens many connections.
- `+num_samples_in_parallel=8` — limits concurrent policy model requests. Tune down (4–6) if the policy endpoint returns 504s; tune up (16–32) if it is fast and the judge is the bottleneck.
- `+resume_from_cache=true` — if the run is interrupted, restarting the same command resumes from where it left off without re-running completed rollouts. Always include this for long runs.

To reduce judge 429 rate-limit errors, lower `judge_concurrency` when starting the servers:
```bash
ng_run \
    "+config_paths=[resources_servers/arena/configs/lmarena_260311.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
    "+lmarena_260311.resources_servers.arena.judge_concurrency=4"
```

Outputs written alongside the rollouts file:
- `rollouts_materialized_inputs.jsonl` — inputs annotated with task/rollout indices and `agent_ref`
- `rollouts_aggregate_metrics.json` — full metrics summary (see Output Metrics below)

### 4. Reward profiling (per-task pass rates)

```bash
ng_reward_profile \
    +materialized_inputs_jsonl_fpath=results/<run_name>/rollouts_materialized_inputs.jsonl \
    +rollouts_jsonl_fpath=results/<run_name>/rollouts.jsonl \
    +output_jsonl_fpath=results/<run_name>/profiled.jsonl \
    +pass_threshold=1.0
```

### 5. Print results

```bash
python scripts/print_aggregate_results.py \
    +jsonl_fpath=results/<run_name>/rollouts_reward_profiling.jsonl
```

Key metric: `mean/reward`. Ranking correlates with LM Arena leaderboard.

### 6. Recompute metrics from saved rollouts

The full metrics (`win_rate`, CIs, etc.) are written to `rollouts_aggregate_metrics.json` automatically at the end of step 3. To recompute them from the saved JSONL without re-running any model calls, run the following as a script (not in a REPL) with the venv active:

```bash
source .venv/bin/activate
python - <<'EOF'
import json
from collections import defaultdict
from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.arena.app import ArenaResourcesServer, ArenaResourcesServerConfig

rollouts_by_task = defaultdict(list)
with open("results/<run_dir>/rollouts.jsonl") as f:
    for line in f:
        r = json.loads(line)
        rollouts_by_task[r["_ng_task_index"]].append(r)

tasks = [rollouts_by_task[i] for i in sorted(rollouts_by_task)]
all_rollouts = [r for task in tasks for r in task]

config = ArenaResourcesServerConfig(
    host="0.0.0.0", port=8080, entrypoint="", name="",
    judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
    judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    # Raise this if compute_metrics raises "Too many failed rollouts".
    max_rollout_failure_rate=0.01,
)
server = ArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
metrics = server.compute_metrics(tasks)

# Framework-level metrics (averaged across all rollouts, not per-task).
metrics["mean/reward"] = sum(r["reward"] for r in all_rollouts) / len(all_rollouts)
usage = [r["response"]["usage"] for r in all_rollouts if r.get("response", {}).get("usage")]
if usage:
    metrics["mean/input_tokens"] = sum(u["input_tokens"] for u in usage) / len(usage)
    metrics["mean/output_tokens"] = sum(u["output_tokens"] for u in usage) / len(usage)

print(json.dumps(metrics, indent=2))
EOF
```

No servers need to be running — `compute_metrics` is pure data processing over the saved rollout dicts.

## Output Metrics

| Metric | Description |
|--------|-------------|
| `mean/reward` | Framework metric: mean of per-task rewards — every question contributes equally. |
| `win_rate` | Bootstrap win rate (style-controlled BT or plain mean; see below). |
| `win_rate_ci_lower` | Bootstrap 2.5th percentile for `win_rate`. |
| `win_rate_ci_upper` | Bootstrap 97.5th percentile for `win_rate`. |
| `style_control` | `true` if `win_rate` uses style-controlled BT, `false` if plain bootstrap mean. |
| `mean/input_tokens` | Framework metric: mean input token count per rollout. |
| `mean/output_tokens` | Framework metric: mean output token count per rollout. |
| `rollout_failure_rate` | Fraction of rollouts where any game failed to produce a parseable verdict (gen_answer returned empty, or any game verdict is `None`). Compared against `max_rollout_failure_rate`. Self-comparison rollouts are excluded from the denominator. |
| `BB_rollout_count` | Number of rollouts where at least one game returned `[[BB]]` (both bad). The judge produced a valid verdict, but the rollout is excluded from `win_rate` scoring. |

### `mean/reward` vs `win_rate`

**`mean/reward`** (framework metric) — per-task, equal-weight average:
```
per_task_reward = mean(weighted_scores_game1 + weighted_scores_game2)
mean/reward     = mean(per_task_reward for all tasks)
```
Each question contributes equally regardless of verdict strength.

**`win_rate`** — per-category win rate, unweighted-averaged across categories:

`[[BB]]` and parse failures are excluded. Strong verdicts (`>>`) contribute `verdict_weight` items instead of 1.

Battles are first grouped by `category`. A win rate is computed independently per category, then those win rates are averaged with equal weight — so a dataset with 500 hard_prompt and 250 creative_writing questions gives each category a 50% contribution, not 67%/33%. For single-category datasets the grouping has no effect.

The per-category formula depends on `style_control`:

- **`style_control=true` (default)** — Bradley-Terry win probability after removing length/formatting bias:
  ```
  P(policy wins battle i) = sigmoid(θ + style_offset_i)
  style_offset_i = z_scored_features_i @ pre_computed_style_coefs
  ```
  θ is fit independently per category per bootstrap round. A model that wins primarily by being longer/more formatted will score lower here than with `style_control=false`.

- **`style_control=false`** — bootstrap mean of weighted scores, computed per category then averaged.

See [Bootstrap parameters](#bootstrap-parameters) for how the CI is derived from per-category bootstrap rounds.

**Style features** (4 dimensions per judgment):
- `f0` — token length differential: `(policy_len − base_len) / (policy_len + base_len)`
- `f1-3` — markdown density differentials for headers, lists, bold: `(policy_density − base_density) / (policy_density + base_density + 1)`

The normalisation factors (`style_norm_mean`, `style_norm_std`) and regression coefficients (`style_coefs`) are pre-computed once from a multi-model reference dataset by running `scripts/precompute_style_constants.py`, and stored in the dataset YAML config. They are **not recomputed** at evaluation time — doing so would make the style adjustment dependent on the current model's own outputs. Only the single quality coefficient `θ` is fit per bootstrap sample.

### Updating pre-computed style constants

When new reference models are added or a different arena-hard-auto dataset / judge is used, recompute the style constants:

```bash
# lmarena-260311
python resources_servers/arena/scripts/precompute_style_constants.py \
    --arena_data_dir /path/to/arena-hard-auto/data/lmarena-260311 \
    --judge gemini-3.1-pro-preview
```

**`--judge`** — subdirectory name under `model_judgment/` (e.g. `gemini-3.1-pro-preview`). If omitted, all judgment files across all judge subdirectories are pooled.

**`--include_models`** — whitelist of evaluated model names (JSONL stems under `model_answer/`, without `.jsonl`). If omitted, all models in `model_answer/` are used.

**`--baseline_models`** — baseline model names to load from `model_answer/`. Each judgment row identifies its baseline via a `baseline` field; rows whose baseline isn't in this list are skipped. If omitted, all model answer files are available as potential baselines. For lmarena-260311, use `--baseline_models baseline` (the aggregated `baseline.jsonl`).

**`--n_rounds`** (default `100`) — number of bootstrap rounds. Style coefficients are averaged over `n_rounds` bootstrap samples for stable estimates.

The baseline for each judgment row is determined by the row's `baseline` field — the same script works for any arena-hard-auto dataset.

The script uses contrast coding (`+1/-1`): each battle row has `+1` for the evaluated model and `-1` for its specific baseline. This avoids the ill-conditioned Hessian that standard one-hot produces when a dominant model wins near-100% of its battles.

Then paste the printed YAML snippets into `style_norm_mean`, `style_norm_std`, and `style_coefs` in the dataset config YAML.

## Baseline Results

Reference numbers for validating the evaluation setup. All runs use `num_repeats=1`, `max_output_tokens=65536`, `temperature=1.0` on the full 1,475-question validation set.

| Model | Judge | `style_control` | `win_rate` | 95% CI | `mean/reward` |
|-------|-------|----------------|-----------|--------|---------------|
| `nvidia/nvidia/nemotron-3-super-v3` | `gcp/google/gemini-3.1-pro-preview` | `false` | 0.186 | [0.179, 0.194] | 0.218 |
| `nvidia/nvidia/nemotron-3-super-v3` | `gcp/google/gemini-3.1-pro-preview` | `true` | 0.180 | [0.172, 0.187] | 0.218 |

Both endpoints hosted on `https://inference-api.nvidia.com/v1`. Benchmark version: v0.2.

## Changelog

The config name tracks the question dataset vintage. The version number tracks evaluation methodology changes.

| Dataset | Version | Changes |
|---------|---------|---------|
| lmarena-260311 | v0.2 | Harder baseline is a mix of stronger reference models (Kimi-K2.5, GLM-5, and Qwen3.5-397B-A17B with Qwen/Qwen3-Next-80B-A3B-Instruct as fallback). |
| lmarena-260311 | v0.1 | Initial release. Baseline: Qwen/Qwen3-Next-80B-A3B-Instruct. |

## Tests

```bash
ng_test +entrypoint=resources_servers/arena
# or, without venv isolation:
ng_dev_test +entrypoint=resources_servers/arena
```
