# Finance Agent v2 (FABv2) — NeMo Gym benchmark

A NeMo Gym integration of the official
[Vals Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2)
that **reuses Vals's own tool code directly** (tools-only wrap): the upstream
`finance_agent.tools.*` classes are imported and exposed as HTTP endpoints, and
the generic `finance_agent` agent loop drives them. Scoring uses **our own**
`[[0]]/[[1]]/[[2]]` judge (path A) — the public FABv2 release ships no official
grader (see [Licensing & grading](#licensing--grading)).

This package (`benchmarks/finance_agent_v2/`) is the public **evaluation recipe**:
it downloads/converts the public question set and wires the run via the gym CLI.
The server code, tests, and `gym env test` fixtures live in
`resources_servers/finance_agent_v2/` — see that
[README](../../resources_servers/finance_agent_v2/README.md) for tool details,
the caching design, dataset/label schema, and full licensing. (Training uses the
same resources server run on externally generated SDG data, so there is no
`environments/finance_agent_v2/` entry.)

## Layout

| Path | Purpose |
|------|---------|
| `config.yaml` | Thin benchmark overlay: `config_paths`-chains to the resources server + `finance_agent` agent config and overrides the dataset to the frozen FABv2 set. Resolved by `gym eval prepare/run --benchmark finance_agent_v2`. |
| `prepare.py` | `gym eval prepare` entry point **and** CSV→JSONL converter (`--input/--output`): builds benchmark JSONL from a raw Vals public export (downloading it if absent), synthesizing the GOLD `expected_answer` from the rubric criteria and copying the raw `rubric` through for reference only. |
| `scripts/prefetch_prices.py` | Sequential Tiingo prefetch into the on-disk cache (idempotent/resumable). |
| `scripts/compare_runs.py` | Compare rollout JSONLs by per-question reward (cache-fidelity checks). |
| `data/vals_v2_public_27q.jsonl` | The public 27-question eval set. **Not committed** — `prepare.py` regenerates it from the upstream Vals public export, so the whole `data/` dir is gitignored. The `gym env test` fixtures (`example.jsonl`, `example_rollouts.jsonl`, `example_metrics.json`) live with the server in `resources_servers/finance_agent_v2/data/`. |

## Setup

Secrets: model endpoints go in `env.yaml` at the repo root (gitignored); tool API
keys are read straight from your shell by the config (`${oc.env:SEC_API_KEY}` etc.).

```bash
export OPENAI_API_KEY=...   # policy + judge
export SEC_API_KEY=...      # edgar_search (sec-api.io)
export TAVILY_API_KEY=...   # web_search (Tavily)
export TIINGO_API_KEY=...   # price_history (Tiingo)
# Persistent, shared cache root (survives across jobs; served on cache hits):
export FINANCE_AGENT_V2_CACHE_DIR=/shared/cache/finance_agent_v2
```

```yaml
# env.yaml — model endpoints only
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

search_judge_model_base_url: https://api.openai.com/v1
search_judge_model_api_key: ${oc.env:OPENAI_API_KEY}
search_judge_model_name: gpt-5-mini
```

## Run (gym CLI, NeMo Gym >= 0.5.0)

`prepare.py` imports the upstream `finance_agent` package, and `gym eval prepare`
imports the prepare module in-process, so run from the resource server's venv
(which has both `gym` and `finance_agent`), at the repo root:

```bash
source resources_servers/finance_agent_v2/.venv/bin/activate

# 1. Build (or rebuild) the public 27Q set. Downloads the Vals public CSV if no
#    local source is present under data/.
gym eval prepare --benchmark finance_agent_v2

# 2. Evaluate against it (auto-serves the resources server + agent).
gym eval run --benchmark finance_agent_v2 \
  -c responses_api_models/openai_model/configs/openai_model.yaml \
  --output results/finance_agent_v2_27q.jsonl --limit 27 --concurrency 4
```

To run against a prebuilt JSONL directly (skip the config-driven prep), use the
`--no-serve --input` path against already-started servers:

```bash
gym eval run --no-serve --agent finance_agent \
  --input benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl \
  --output results/finance_agent_v2_27q.jsonl --limit 27 --concurrency 4
```

Rewards land in the output JSONL (`reward` 1.0 = judge rated `[[2]]`; else 0.0).
Swap in `responses_api_models/vllm_model/configs/vllm_model.yaml` and point
`policy_*` at your endpoint to run on a self-hosted model.

## Caching

Tiingo/SEC responses are cached to disk (byte-identical to live calls) to absorb
rate limits and make re-runs reproducible; `use_cache: true` by default. Point
`cache_dir` (or `FINANCE_AGENT_V2_CACHE_DIR`) at a shared absolute path so it is
reused across seeds/jobs. Full design in the
[resources-server README](../../resources_servers/finance_agent_v2/README.md#caching-pricing--sec).

## Licensing & grading

Environment code (this recipe + `resources_servers/finance_agent_v2/`) is
**Apache-2.0**. Tools are **imported, not vendored**: `finance-agent` (MIT) and
`model-library` (MIT, NVIDIA fork). The dataset derives from the **public** Vals
FABv2 release (subject to that project's terms). The public release ships **no
official grader** — reward is computed by **our own** `[[N]]` judge against a GOLD
`expected_answer` synthesized from the public rubric criteria; the dataset's
`rubric` field is propagated for reference only and is **not** used for scoring.
Vals's private per-criterion rubric grader is licensed and deliberately not
reproduced here. Details:
[resources-server README](../../resources_servers/finance_agent_v2/README.md#licensing).
