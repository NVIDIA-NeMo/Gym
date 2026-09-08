# AgentIF Resources Server

Agentic **instruction-following** scoring, ported from
[THU-KEG/AgentIF](https://github.com/THU-KEG/AgentIF) (707 real-world agentic
scenarios). Each row carries a list of natural-language constraints the
model's response must satisfy; constraints are scored by a pipeline of typed
checkers declared in the data itself.

Source dataset: [`THU-KEG/AgentIF`](https://huggingface.co/datasets/THU-KEG/AgentIF)
(`eval.json`). Upstream eval: `1.evaluation_api.py`.

## Checker pipeline

Each constraint's `evaluation` list is a sequence of typed steps, run in order
against a `working` value that starts as the model's response:

| Step type | Mechanism |
|-----------|-----------|
| `llm` | Calls the judge model; rewrites `working` to the judge's output text. |
| `llm_conditional_check` | Calls the judge to test if the constraint applies; the constraint is skipped (`None`) if not. |
| `code` | Executes the dataset-provided `check_following(response)` in a fresh globals dict (fresh per call, so concurrent rollouts never interfere). |

The pipeline resolves to `True` / `False` / `None` (`None` = unscored: a
conditional guard was unmet, the judge call failed, or the code checker
raised). A preceding failed/`None` code check short-circuits any later `llm` /
`llm_conditional_check` steps for that constraint (upstream parity).

> **Reasoning models:** verify() strips a leading `<think>…</think>` block
> before scoring. When serving a reasoning model, also run the policy server
> with `--reasoning-parser <name>` so reasoning is split off upstream.

## Metrics

Two headline metrics follow upstream `1.evaluation_api.py`:

- **ISR** (Instruction Success Rate) — `1.0` when every scored constraint in a
  row passed and none errored, else `0.0`; this is the per-row reward returned
  by `verify()`.
- **CSR** (Constraint Success Rate) — fraction of scored constraints
  (`n_true / (n_true + n_false)`) that passed, corpus-wide.

`compute_metrics` also reports:

- `mean_reward` — mean per-row ISR.
- `by_dimension/{vanilla,condition,example}/accuracy` — per-dimension
  breakdown (`unconditional` / `conditional` / `example_driven` in the data).
- `by_type/{formatting,semantic,tool}/accuracy` — per-type breakdown
  (`formatting` / `semantic` / `resource` in the data).
- `n_null_total` — total unscored constraints across the corpus (diagnostics),
  split into `n_skipped_total` and `n_error_total`.

Unscored constraints come in two flavours, and they are treated differently:

- **skipped** — an `llm_conditional_check` gate answered `NO`, so the constraint
  does not apply to this response. Excluded from ISR and CSR (upstream parity).
- **error** — a judge call failed, a `code` checker raised, or the judge returned
  no usable `YES` / `NO` verdict (including ambiguous output mentioning both).
  Errors count as evaluation failures: the row scores ISR `0.0` rather than
  silently dropping the constraint.

`get_key_metrics` surfaces `isr`, `csr`, and `mean_reward` as the report
headline.

## Prepare the dataset

The committed `data/example.jsonl` is a tiny synthetic sample for tests and
smoke runs. Build the full 707-row dataset from Hugging Face:

```bash
python resources_servers/agentif/prepare_agentif.py
# -> data/train.jsonl
```

Set `AGENTIF_LOCAL_JSON=/path/to/eval.json` to convert a pre-downloaded file
instead of fetching from the Hub.

There's also a gym-native benchmark entry (`benchmarks/agentif/`) that reuses
`prepare_agentif.build_row` to write the whole dataset to
`benchmarks/agentif/data/agentif_benchmark.jsonl` for the `gym eval` /
nemo-evaluator `gym://` flow — see `benchmarks/agentif/README.md`.

## Judge model

Scoring `llm` / `llm_conditional_check` steps requires a judge model.
Recommended: `gpt-4o-mini` (upstream used `gpt-4o-2024-11-20`). Supply
credentials at launch:

```bash
gym env start --resources-server agentif --model-type vllm_model \
    +judge_base_url=... +judge_api_key=... +judge_model_name=...
```

`configs/agentif.yaml` wires a policy agent/model + judge for full
eval/training. `configs/agentif_serve.yaml` is a serve-only config (resources
server + judge, no policy) for the nemo-evaluator `gym://...protocol=native`
flow.

Judge concurrency is bounded by `asyncio.Semaphore` (default 32); `code`
checkers run in an executor so they never block the event loop.

## Run

```bash
gym env start --resources-server agentif --model-type vllm_model \
    +judge_base_url=... +judge_api_key=... +judge_model_name=...
```

## Example rollouts and metrics

`data/example_rollouts.jsonl` and `data/example_metrics.json` are committed
and show live examples of rollouts and metrics.

To collect rollouts from a live model instead:

```bash
gym eval run --no-serve \
    --agent agentif_simple_agent \
    --input resources_servers/agentif/data/example.jsonl \
    --output resources_servers/agentif/data/example_rollouts.jsonl

tail -n 1 resources_servers/agentif/data/example_rollouts.jsonl | jq | less
```

## Test

```bash
gym env test --resources-server agentif
```

## Notes

- `verified: false` — not yet baselined. Reward profiling across an
  open-instruct / open-thinking / closed-source model suite is a follow-up
  before flipping to `verified: true`.
- `code` checkers use `exec` in a fresh per-call globals dict (never the
  process `globals()`). The AgentIF dataset is trusted (curated by THU-KEG);
  checkers only inspect the model's response string.
