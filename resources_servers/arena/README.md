# Arena Resources Server

Evaluate a model's chat capabilities on diverse conversational prompts. An LLM judge compares each response with a saved baseline in both answer orderings to reduce positional bias.

## Choose a benchmark

| Benchmark | Prompts | Main differences |
|---|---:|---|
| `lmarena_v3` | 3,713 | Recommended. Exact human-battle opponents, equal verdict weights, `both bad` as a tie, reference-length style control. |
| `lmarena_v2` | 1,475 | Legacy/deprecated. Synthetic baselines, strong verdict weight 3, `both bad` excluded, Bradley-Terry style control. |

## Setup

Run commands from the repository root with the project environment active. Add your endpoint settings to `env.yaml`:

```yaml
policy_base_url: https://YOUR_MODEL_ENDPOINT/v1
policy_api_key: YOUR_MODEL_API_KEY
policy_model_name: YOUR_MODEL_ID
judge_api_key: YOUR_JUDGE_API_KEY
```

Gym loads these values automatically, so evaluation commands do not need endpoint, model, or API-key flags. The benchmark's dummy judge key only allows dataset preparation without credentials; `env.yaml` overrides it during evaluation.

The benchmark config defines the judge and generation settings. Both LMArena versions use policy reasoning, `temperature: 1.0`, and `top_p: 0.95`.

Download the validation data from the internal GitLab registry:

```bash
gym eval prepare --config benchmarks/lmarena_v3/config.yaml
```

This creates `benchmarks/lmarena_v3/data/lmarena_v3_validation.jsonl` with 3,713 rows.

## Run the benchmark

This command generates responses, judges them, and reports scores:

```bash
mkdir -p results/lmarena_v3/my-model

gym eval run \
    --config benchmarks/lmarena_v3/config.yaml \
    --agent lmarena_v3_benchmark_agent \
    --output results/lmarena_v3/my-model/rollouts.jsonl \
    --split benchmark \
    --resume \
    --concurrency 64 \
    ++reuse_existing_data_preparation=true
```

Replace the output path as needed. Use `--limit 3` for a smoke test. For concurrent runs, add a distinct `++head_server.port=<port>` to each command.

## Validation data

Each JSONL row contains:

```json
{
  "category": "lmarena_v3",
  "responses_create_params": {"input": [{"role": "user", "content": "<message>"}]},
  "question_id": "<question ID>",
  "question": "<judge-visible conversation>",
  "baseline_answer": "<baseline answer>",
  "baseline_model": "<baseline model name>",
  "other_answer": "<other model answer>",
  "other_model": "<other model name>",
  "winner": "<human verdict>",
  "style_reference_token_count": "<reference token count>",
  "is_lmarena_v2_prompt": "<whether the prompt is derived from lmarena_v2>",
  "metadata": "<prompt metadata>"
}
```

`responses_create_params.input` contains the complete conversation sent to the policy model. `question` is the same user message for single-turn prompts and a flattened conversation for multi-turn prompts. The `other_*`, `winner`, and style-reference fields are lmarena_v3 metadata.

## Scoring

| Metric | Meaning |
|---|---|
| `mean/reward` | Arithmetic mean of rewards returned by Arena verification; empty responses and failed judge games receive zero. `[[BB]]` is assigned a 0.5 reward. No bootstrap is applied. |
| `win_rate_no_SC` | Raw judge score computed from valid judgments, excluding failures. |
| `win_rate` | Style-controlled win rate. |
| `*_ci95_lower`, `*_ci95_upper` | Bootstrap 95% confidence interval, resampling individual judge games. |
| `verbosity_acceptance_rate` | Fraction of valid lmarena_v3 responses accepted by reference-length style control. |
| `max_token_reached_rate` | Fraction of lmarena_v3 responses that generated to the policy output limit. They receive score 0 and do not count as rollout failures. |
| `context_window_exceeded_rate` | Fraction of lmarena_v3 policy requests where prompt tokens plus requested output exceeded the model context window. They receive score 0 and do not count as rollout failures. |
| `rollout_failure_rate` | Fraction excluded because judgments were missing or unparseable. |
| `response_tokens/*`, `reasoning_tokens/*` | Response and reasoning length statistics. |
| `arena/*`, `taxonomy-language/*`, `taxonomy-task-type/*` | Prompt count and ON/OFF scores for slices with at least 50 prompts. |

`lmarena_v3` keeps the judge score when `0.5 < response length / reference length < 1.75`; otherwise it assigns zero. The reference length is the median length from reference models. For short references (at most 100 tokens), responses from 0 to 175 tokens are accepted.

`lmarena_v2` instead uses fixed Bradley-Terry style coefficients and excludes `[[BB]]` from its `win_rate*` metrics.

## Generate responses only

Generate responses without calling the judge:

```bash
mkdir -p results/lmarena_v3/my-model

gym eval run \
    --config benchmarks/lmarena_v3/config.yaml \
    --agent lmarena_v3_benchmark_agent \
    --output results/lmarena_v3/my-model/responses.jsonl \
    --split benchmark \
    --resume \
    --concurrency 64 \
    ++reuse_existing_data_preparation=true \
    ++lmarena_v3_benchmark_resources_server.resources_servers.arena.generation_only=true
```

No judge calls are made during generation-only runs.

## Recompute scores

No model calls are needed:

```bash
resources_servers/arena/.venv/bin/python resources_servers/arena/scripts/compute_rollout_scores.py \
    results/lmarena_v3/my-model/judged.jsonl \
    --version lmarena_v3
```

## Scripts

Run these scripts with `resources_servers/arena/.venv/bin/python` so Arena-specific dependencies are available.

| Script | When to use it |
|---|---|
| `compute_rollout_scores.py` | Recompute standard Arena metrics from saved rollouts. |
| `compute_rollout_scores_by_custom_taxonomy.py` | Score saved rollouts using a supplied question-to-label taxonomy. |
| `count_rollout_tokens.py` | Inspect response, reasoning, and baseline token lengths in one rollout file. |
| `count_rollout_tokens_directory.py` | Run the token-length report across matching rollout files in a directory. |
| `create_replay_validation_from_arena_eval.py` | Convert human Arena evaluation records into prompts for replaying model responses; use OFF scores because its reference length is only a placeholder. |
| `fit_anchored_elo.py` | Estimate one model's Elo while holding opponent Elo values fixed. |
| `remove_failed_rollouts.py` | Remove failed rollouts so `--resume` regenerates and rejudges them. |
| `summarize_prompt_blend.py` | Report prompt lengths, turn counts, and taxonomy distribution for a benchmark. |

## Tests

```bash
gym env test --resources-server arena
```
