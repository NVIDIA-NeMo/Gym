# IHEval (gym-native)

[IHEval](https://github.com/ytyz1307zzh/IHEval) is an **instruction-hierarchy**
benchmark: it measures whether a model respects the priority order of
system / user / tool instructions across *aligned*, *conflict*, and *reference*
settings.

This entry runs IHEval through the **gym-native** eval path against the
**whole** dataset and reports the exact upstream headline — the task-macro
**hierarchy** `result_score` (aggregate conflict score) — computed by the
`iheval` resources server's `compute_metrics` and surfaced via `get_key_metrics`.
A plain per-row-mean eval cannot produce this number; only the gym-native
`compute_metrics` path does.

## Relationship to the resources server

Scoring, aggregation, and the upstream rule-based checkers all live in the
`iheval` resources server (`resources_servers/iheval/`). This benchmark only
supplies data and wiring; it chains to `resources_servers/iheval/configs/iheval.yaml`.

## Data shape

`resources_servers/iheval/prepare_iheval.py` builds the whole dataset
(`data/test.jsonl`, all tasks/settings) in **Chat-Completions** shape. The
gym-native `simple_agent` speaks the
**Responses API** (`/v1/responses`), so `prepare.py` here re-shapes the tool-use
rows (`get-webpage`, `slack-user`) to Responses items
(`function_call` / `function_call_output`, top-level function tools). All other
rows are plain `{role, content}` messages and pass through unchanged. Gold and
routing fields are untouched, so the resources server's `verify()` is unaffected.

## Prepare data

```bash
gym eval prepare --benchmark iheval
```

Builds `resources_servers/iheval/data/test.jsonl` if missing (downloads
`github.com/ytyz1307zzh/IHEval`, or set `IHEVAL_REPO_DIR`), then writes the
Responses-shaped `benchmarks/iheval/data/iheval_benchmark.jsonl`.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark iheval
```

## Collecting rollouts and scoring

```bash
gym eval run --no-serve \
    --agent iheval_benchmark_simple_agent \
    --input benchmarks/iheval/data/iheval_benchmark.jsonl \
    --output results/iheval_rollouts.jsonl \
    --num-repeats 1
```

The headline `result_score` (hierarchy conflict score) is selected by the
resources server's `accuracy_mode` (default `hierarchy`; see
`resources_servers/iheval/README.md`). Override with
`++iheval.resources_servers.iheval.accuracy_mode=hierarchy_sysprompt`.
