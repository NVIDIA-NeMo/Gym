# RoleMRC Resources Server

Role-play machine-reading-comprehension scoring, ported from the nemo-evaluator
BYOB benchmarks `rolemrc` / `rolemrc_judge`. The model plays a character and
answers questions about supplied passages while respecting the character's
knowledge range, speech style, and instruction priority.

Source dataset: [`Junrulu/RoleMRC`](https://huggingface.co/datasets/Junrulu/RoleMRC)
(`roleMRC_test.jsonl`). Upstream eval: `RoleMRC/evaluation/{evaluation,llm_judge}.py`.

## Two scoring modes

Selected by `config.mode`; one app, two configs.

| Config | `mode` | Reward | Notes |
|--------|--------|--------|-------|
| `configs/rolemrc.yaml` | `reference` | ROUGE-L vs the gold reply | BLEU / METEOR / BERTScore ride along on the verify response. |
| `configs/rolemrc_judge.yaml` | `judge` | mean 0/1 over relevant aspects | One judge call per aspect, per the row's `task` (see `_EVALUATION_CONFIG`). |

The five judge aspects are `knowledge_range`, `style_compliance`,
`nested_instruction`, `multi_turn_instruction`, and `instruction_priority`.
Which aspects fire is determined by the row's `task` field.

Results are broken down by RoleMRC **dimension** (`on_scene_dialogue`,
`multi_turn`, `nested_instruction`, `instruction_priority`), derived from the
`task` suffix in `compute_metrics`.

> **Reasoning models:** verify() strips a leading `<think>…</think>` block
> before scoring. When serving a reasoning model, also run the policy server
> with `--reasoning-parser <name>` so reasoning is split off upstream.

## Prepare the dataset

The committed `data/example*.jsonl` are tiny synthetic samples for tests and
smoke runs. Build the full test split from Hugging Face:

```bash
python resources_servers/rolemrc/prepare_rolemrc.py
# -> data/test.jsonl (reference) and data/test_judge.jsonl (judge subset)
```

Set `ROLEMRC_LOCAL_JSONL=/path/to/roleMRC_test.jsonl` to convert a
pre-downloaded file instead of fetching from the Hub.

## BERTScore

`include_bertscore: true` (default) matches the upstream benchmark but
downloads a roberta-large checkpoint on first use. Set it to `false` (and drop
`bert-score` from `requirements.txt`) for a lightweight ROUGE/BLEU/METEOR-only
reward signal.

## Judge model

`rolemrc_judge.yaml` defaults the judge to `policy_model`. Point
`judge_model_server` at a dedicated, stronger judge server to reduce
self-grading bias.

## Run

```bash
# Reference metrics
gym env start --resources-server rolemrc --model-type vllm_model

# LLM-as-judge
gym env start --resources-server rolemrc/rolemrc_judge --model-type vllm_model
```

## Example rollouts and metrics

`data/example_rollouts.jsonl` and `data/example_metrics.json` are committed
and can be regenerated at any time with the scripts below (no servers needed):

```bash
# Regenerate synthetic rollouts (ROUGE/BLEU/METEOR scored, no model call)
python resources_servers/rolemrc/generate_example_rollouts.py

# Aggregate rollouts -> per-dimension metrics summary
python resources_servers/rolemrc/generate_example_metrics.py

# Inspect
tail -n 1 resources_servers/rolemrc/data/example_rollouts.jsonl | jq .reward
cat resources_servers/rolemrc/data/example_metrics.json | jq .
```

To collect rollouts from a live model instead:

```bash
gym eval run --no-serve \
    --agent rolemrc_simple_agent \
    --input resources_servers/rolemrc/data/example.jsonl \
    --output resources_servers/rolemrc/data/example_rollouts.jsonl

tail -n 1 resources_servers/rolemrc/data/example_rollouts.jsonl | jq | less
```

## Test

```bash
gym env test --resources-server rolemrc
```
