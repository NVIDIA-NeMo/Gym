# Finance Agent v2 (FABv2) — NeMo Gym benchmark

A NeMo Gym integration of the official
[Vals Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2)
that **reuses Vals's own tool code directly** (tools-only wrap): the upstream
`finance_agent.tools.*` classes are imported and exposed as HTTP endpoints, and
the `finance_agent_v2` agent loop drives them. Scoring uses **our own**
per-criterion rubric judge (path A) — the public FABv2 release ships no official
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
| `prepare.py` | `gym eval prepare` entry point **and** CSV→JSONL converter (`--input/--output`): builds benchmark JSONL from a raw Vals public export (downloading it if absent), copying the raw `rubric` through verbatim — criteria text *and* the `modifiers` carrying severity/`must_pass`, all of which are scoring inputs — and concatenating the criteria into a human-readable `expected_answer` reference. Depends only on the standard library, so it runs in the repo-root venv like every other benchmark's prepare script. |
| `upstream_spec.json` | Committed snapshot of the upstream system/question prompts and the tool JSON schemas that `prepare.py` bakes into each sample. Generated from the installed `finance_agent` package by `resources_servers/finance_agent_v2/scripts/export_upstream_spec.py`; **do not hand-edit**. It exists because `gym eval prepare` imports `prepare.py` into the root `gym` process, where `finance_agent` (a server dependency) is not installed. The server's `tests/test_upstream_spec.py` re-derives it from the package and fails if it has drifted, and `prepare.py` refuses to run if its `_UPSTREAM_SHA` and the snapshot disagree. |
| `data/vals_v2_public_27q.jsonl` | The public 27-question eval set. **Not committed** — `prepare.py` regenerates it from the upstream Vals public export, so the whole `data/` dir is gitignored. The `gym env test` fixtures (`example.jsonl`, `example_rollouts.jsonl`, `example_metrics.json`) live with the server in `resources_servers/finance_agent_v2/data/`. |

Operational scripts (cache prefetch, run comparison, offline rescoring, run
reports) live with the server, not here, because they are shared by eval, training,
and SDG: `resources_servers/finance_agent_v2/scripts/`.

## Setup

Nothing is hardcoded in the committed configs, and nothing below is required for
the config to *resolve* — `gym eval prepare` and `gym env resolve` work on a clean
checkout. Each value resolves as **config key → shell environment → null-safe
default**, so you can supply it either way.

Tool API keys are easiest to export; a key you leave unset simply registers its
tool as unavailable instead of failing startup.

```bash
export OPENAI_API_KEY=...   # policy + judge
export SEC_API_KEY=...      # edgar_search (sec-api.io)
export TAVILY_API_KEY=...   # web_search (Tavily)
export TIINGO_API_KEY=...   # price_history (Tiingo)
# Persistent, shared cache root (survives across jobs; served on cache hits):
export FINANCE_AGENT_V2_CACHE_DIR=/shared/cache/finance_agent_v2
```

Model endpoints go in `env.yaml` at the repo root (gitignored — never commit a
populated one). The judge is a **separate** server from the policy on purpose: if
the policy graded its own answers, two models evaluated on this benchmark would
be scored by two different judges and the numbers would not be comparable. Pin
`search_judge_model_*` for any leaderboard or cross-model claim.

```yaml
# env.yaml — model endpoints only (tool keys come from the shell, above)
# Policy model, also reused for retrieve_information. For a self-hosted model
# point these at your vLLM endpoint and run with vllm_model.yaml.
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

# Judge model used by /verify. Any OpenAI-compatible endpoint, including a local
# vLLM. Defaults to gpt-5-mini on api.openai.com if omitted.
search_judge_model_base_url: https://api.openai.com/v1
search_judge_model_api_key: ${oc.env:OPENAI_API_KEY}
search_judge_model_name: gpt-5-mini
```

## Run (gym CLI, NeMo Gym >= 0.5.0)

Run from the repo root in the root venv. `gym eval prepare` imports `prepare.py`
into the `gym` process, and prepare reads the upstream prompts and tool schemas from
the committed `upstream_spec.json`, so no server venv is needed to build the dataset:

```bash
source .venv/bin/activate

# 1. Build (or rebuild) the public 27Q set. Downloads the Vals public CSV if no
#    local source is present under data/.
gym eval prepare --benchmark finance_agent_v2

# 2. Evaluate against it (auto-serves the resources server + agent).
gym eval run --benchmark finance_agent_v2 \
  -c responses_api_models/openai_model/configs/openai_model.yaml \
  --output results/finance_agent_v2_27q.jsonl --limit 27 --concurrency 4
```

For a model comparison, name the model on the command line and take repeats — 27
questions is small enough that a single pass cannot separate two models (see
[report_run.py](../../resources_servers/finance_agent_v2/README.md#analyzing-a-finished-run)
for the confidence intervals):

```bash
gym eval run --benchmark finance_agent_v2 --split benchmark \
  --model-type openai_model -m gpt-5.5 \
  --output results/fabv2_27q_gpt55_repeat3.jsonl \
  --num-repeats 3 --concurrency 4
```

To reuse one set of servers across several runs, start them in one terminal and
collect in another. The agent name is the config's benchmark agent, not the
component name:

```bash
# Terminal 1 — serve
gym env start --benchmark finance_agent_v2 \
  -c responses_api_models/openai_model/configs/openai_model.yaml

# Terminal 2 — collect (repeat as needed against the same servers)
gym eval run --no-serve --agent finance_agent_v2_benchmark_agent \
  --input benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl \
  --output results/finance_agent_v2_27q.jsonl --limit 27 --concurrency 4
```

Rewards land in the output JSONL. `reward` is **Partial Credit**: the
severity-weighted share of rubric criteria passed, forced to 0.0 if any criterion
marked `must_pass` failed. `rubric_all_pass` is the stricter every-criterion-passed
flag, and `judge_error` marks rows the judge could not score (filter those out
rather than reading them as zeros). Swap in
`responses_api_models/vllm_model/configs/vllm_model.yaml` and point `policy_*` at
your endpoint to run on a self-hosted model.

> Read `mean/reward` through
> `resources_servers/finance_agent_v2/scripts/report_run.py`, which prints
> question-level bootstrap confidence intervals. At 27 questions the interval is wide
> enough that small gaps between models are usually noise.

## Caching

Tiingo/SEC responses are cached to disk (byte-identical to live calls) to absorb
rate limits and make re-runs reproducible; `use_cache: true` by default. Point
`cache_dir` (or `FINANCE_AGENT_V2_CACHE_DIR`) at a shared absolute path so it is
reused across seeds/jobs. Full design in the
[resources-server README](../../resources_servers/finance_agent_v2/README.md#caching-pricing--sec).

## Licensing & grading

Environment code (this recipe + `resources_servers/finance_agent_v2/`) is
**Apache-2.0**. Tools are **imported, not vendored**: `finance-agent` (MIT) and
`model-library` (MIT). The dataset derives from the **public** Vals
FABv2 release (subject to that project's terms). The public release ships **no
official grader** — reward is computed by **our own** judge, run once per criterion
of the public `rubric` field and voted over repeated calls. Vals's private grader
is licensed and deliberately not reproduced here, so scores are not directly
comparable to their published numbers. Licensing detail:
[resources-server README](../../resources_servers/finance_agent_v2/README.md#licensing).

Before comparing anything here to the Vals leaderboard, read
[reading these metrics against the Vals leaderboard](../../resources_servers/finance_agent_v2/README.md#reading-these-metrics-against-the-vals-leaderboard).
`mean/rubric_partial_credit` and `mean/rubric_all_pass` now match their Accuracy and
All-Pass *definitions*, but not their numbers: we measure on the 27 public questions
rather than their private 450-question split, with our own judge instead of their
three-model jury. `rubric_fraction` has no counterpart there at all.
