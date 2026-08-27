# RoleMRC Resources Server

Role-play machine-reading-comprehension scoring. The model plays a character
and answers questions about supplied passages while respecting the character's
knowledge range, speech style, and instruction priority.

Source dataset: [`Junrulu/RoleMRC`](https://huggingface.co/datasets/Junrulu/RoleMRC)
(`roleMRC_test.jsonl`). Upstream eval: `RoleMRC/evaluation/{evaluation,llm_judge}.py`.

## Two scoring modes

Selected by `config.mode`; one app, two configs.

| Config | `mode` | Reward | Notes |
|--------|--------|--------|-------|
| `configs/rolemrc.yaml` | `reference` | ROUGE-L vs the gold reply | BLEU / METEOR / BERTScore ride along on the verify response. BLEU is unsmoothed 4-gram BLEU matching upstream's `evaluate.load("bleu")`, so a response with no matching 4-gram scores exactly 0. |
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

## Metrics

**The headline metric is `judge/avg_simple_no_mt`** — RoleMRC's `AvgS(noMT)`.

`compute_metrics` emits everything the published RoleMRC report quotes:

| Key | Report line | Definition |
|-----|-------------|------------|
| `auto/<metric>/mean` | `Auto Rouge1:… BERTScore-F1:…` | corpus mean of each reference metric |
| `aspect/<aspect>/mean` | `Judge Knowledge:… Priority:…` | mean 0/1 verdict per aspect, over **judge calls** |
| `judge/avg_simple` | `AvgSimple` | unweighted mean of the 5 aspect means |
| `judge/avg_weighted` | `AvgWeighted` | mean over every judge call (aspect means weighted by call count) |
| `judge/avg_simple_no_mt` | `AvgS(noMT)` | unweighted mean of the 4 aspects **excluding** `multi_turn_instruction` |
| `mean_reward` | *(none)* | per-row mean of `reward` — matches no published number |

Two things are easy to get wrong here:

- **The judge aggregates are per-aspect, not per-row.** A row fires one judge
  call per aspect its `task` maps to, and the on-scene `*_answer_with_narration`
  / `*_refused_no_narration` tasks fire two, so the 1400-row split produces 1600
  calls. `mean_reward` averages rows; the report averages aspects. The two are
  different numbers and neither can be derived from the other.
- **`mean_reward` is not the benchmark score.** In `reference` mode it is
  ROUGE-L alone; in `judge` mode it is a mean-of-row-means.

### When the harness only calls `/verify`

The table above is produced by `compute_metrics`, which runs only when the
caller asks for it via `/aggregate_metrics`. A harness that drives this server
through `/verify` alone never triggers it and is left with per-row rewards to
average itself — which yields `mean_reward`, not any published number.

`verify()` returns every input those roll-ups need alongside `reward` (the
per-aspect 0/1 scores, the reference metrics), so a harness that persists whole
verify responses per row loses nothing. `score_rolemrc_report.py` rebuilds the
report lines from such per-row output:

```bash
python resources_servers/rolemrc/score_rolemrc_report.py <per-row-output>.jsonl
```
```
Auto  Rouge1:0.2118 Rouge2:0.0470 RougeL:0.1315 BLEU:0.0135 METEOR:0.2516 BERTScore-F1:0.8438
Judge Knowledge:66.7% Style:98.2% Nested:77.8% Maintain:53.8% Priority:36.9% AvgSimple:66.7% AvgWeighted:70.8% AvgS(noMT):69.9%
      judge calls: 1600 (Knowledge=600, Style=400, Nested=158, Maintain=400, Priority=42)
      >> headline metric AvgS(noMT) = 69.9%
```

Pass `--json` for machine-readable output. That script is the one file here that
knows about a specific harness's on-disk format; the server itself does not.

### Known divergence: `AvgWeighted` call counts

**Compare against published RoleMRC numbers on `AvgS(noMT)`, not `AvgWeighted`.**

Solving for the aspect weights implied by the 28 published model runs recovers
call counts of (knowledge 601, style 400, nested 159, multi-turn 400, priority
**84**), max residual 0.07 pp. Every count lands within 1 of this dataset's true
counts except `instruction_priority`, which is exactly doubled — the raw
`roleMRC_test.jsonl` has 42 rows for the two `-refused` tasks that fire it.

No alternative weighting explains it: pinning the true 42:400 priority-to-style
ratio forces nested to 168 and multi-turn to 433 (both contradicted by the
dataset) and still leaves 0.69 pp of error, and no priority *score* can rescue
the 42-call model — the value it would require is negative. So those published
runs counted 42 judge calls twice.

We report the honest count, which puts our `AvgWeighted` ~0.7 pp (up to 1.3 pp)
above the published value. `AvgSimple` and `AvgS(noMT)` are
count-independent, so the headline metric is unaffected and directly comparable.

The expected split is pinned in `prepare_rolemrc.py:_EXPECTED_TASK_COUNTS` and
asserted on every prep run, so a truncated download or a revised upstream split
fails loudly instead of quietly shifting the scores. Override with
`ROLEMRC_ALLOW_DATASET_DRIFT=1` if upstream legitimately changes.

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

The judge runs on its own model server instance, `judge_model`, registered in
both judge configs and wired by the `+judge_base_url` / `+judge_api_key` /
`+judge_model_name` overrides. It is deliberately separate from `policy_model`
(the model under test) — sharing the two means the model grades its own output.

### Reasoning judges

A reasoning judge (gpt-5.x and friends) needs two departures from the gpt-4.1
defaults, both config-only:

- **Drop `temperature` / `top_p`.** Reasoning models reject them outright
  (`Unsupported parameter: 'top_p' is not supported with this model`). Only
  params actually set in `judge_responses_create_params` are sent, so deleting
  the lines is enough; per run, `~<instance>.…judge_responses_create_params.top_p`
  does the same. Such a judge samples at the provider's defaults, so its verdicts
  are not reproducible the way a temperature-0 judge's are.
- **Pin `reasoning.effort`.** The model server validates the reply into
  `NeMoGymResponse`, whose allowed values come from the pinned `openai` SDK, and
  a newer provider default (`effort: "none"`) 500s every call — scoring the run
  0. The reply echoes the request, so asking for a value the SDK knows avoids it.

To reuse the policy model as the judge anyway (cheap smoke tests, no second
endpoint available), set `judge_model_server.name` back to `policy_model` and
drop the `+judge_*` overrides.

### `rolemrc_judge.yaml` vs `rolemrc_judge_serve.yaml`

| Config | Registers | Use when |
|--------|-----------|----------|
| `configs/rolemrc_judge.yaml` | judge scorer + `judge_model` + `rolemrc_judge_simple_agent` | gym generates the answers; compose a model server that provides `policy_model` |
| `configs/rolemrc_judge_serve.yaml` | judge scorer + `judge_model` only | an external harness generates the answers and only calls `/verify` |

Every server instance in the merged config is validated at startup, agents
included. So composing `rolemrc_judge.yaml` into a generation-free service
fails on the agent's unresolvable `policy_model` reference even though nothing
would ever call that agent — hence the serve-only variant, which is the judge
counterpart of `rolemrc_serve.yaml`.

## Run

```bash
# Reference metrics
gym env start --resources-server rolemrc --model-type vllm_model

# LLM-as-judge (needs a judge endpoint on top of the policy one)
gym env start --resources-server rolemrc/rolemrc_judge --model-type vllm_model \
  +judge_base_url=https://api.openai.com/v1/ \
  +judge_api_key=$OPEN_AI_KEY \
  +judge_model_name=gpt-4.1
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
