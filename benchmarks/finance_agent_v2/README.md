# Finance Agent v2 (FABv2) — NeMo Gym benchmark

A NeMo Gym integration of the official
[Vals Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2)
that **reuses Vals's own tool code directly** (tools-only wrap): the upstream
`finance_agent.tools.*` classes are imported and exposed as HTTP endpoints, and
the `finance_agent_v2` agent loop drives them. The public FABv2 release ships no
official grader, so scoring uses our own per-criterion rubric judge.

This package is the public **evaluation recipe**: it downloads and converts the
public question set and wires the run via the gym CLI. The server code, tests, and
`gym env test` fixtures live in `resources_servers/finance_agent_v2/` — see that
[README](../../resources_servers/finance_agent_v2/README.md) for tool details, the
caching design, the label schema, and full licensing. Training uses the same
resources server on externally generated SDG data, so there is no
`environments/finance_agent_v2/` entry.

## Layout

| Path | Purpose |
|------|---------|
| `config.yaml` | Thin benchmark overlay: `config_paths`-chains to the resources server + agent config and overrides the dataset to the frozen FABv2 set. |
| `prepare.py` | `gym eval prepare` entry point and CSV→JSONL converter (`--input/--output`). Copies the raw `rubric` through verbatim — criteria text *and* the `modifiers` carrying severity and `must_pass`, all of which are scoring inputs. Standard library only, so it runs in the repo-root venv like every other prepare script. |
| `upstream_spec.json` | Generated snapshot of the upstream prompts and tool JSON schemas that `prepare.py` bakes into each sample; **do not hand-edit**. Regenerate with `resources_servers/finance_agent_v2/scripts/export_upstream_spec.py`. |
| `data/vals_v2_public_27q.jsonl` | The public 27-question eval set. **Not committed** — `prepare.py` regenerates it from the upstream Vals export, so `data/` is gitignored. |

## Setup

Everything goes in `env.yaml` at the repo root, which is gitignored. Nothing is
hardcoded in the committed configs, and only the policy endpoint is required for a
run: a tool key left unset registers its tool as unavailable instead of failing
startup.

The judge is a **separate** server from the policy on purpose. If the policy graded
its own answers, two models evaluated here would be scored by two different judges
and the numbers would not be comparable.

```yaml
# env.yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

search_judge_model_base_url: https://api.openai.com/v1
search_judge_model_api_key: ${oc.env:OPENAI_API_KEY}
search_judge_model_name: gpt-5-mini

sec_api_key: ${oc.env:SEC_API_KEY}                 # edgar_search (sec-api.io)
tavily_api_key: ${oc.env:TAVILY_API_KEY}           # web_search (Tavily)
pricing_data_api_key: ${oc.env:TIINGO_API_KEY}     # price_history (Tiingo)

# Persistent, shared cache root (survives across jobs; served on cache hits):
finance_agent_v2_cache_dir: /shared/cache/finance_agent_v2
```

`${oc.env:VAR}` reads the value at resolve time, so no secret is written to disk;
replace it with a literal if you prefer. Every key above except the `policy_*` ones
also resolves straight from the environment, as config key → environment variable →
null-safe default, so exporting `SEC_API_KEY`, `TAVILY_API_KEY`, `TIINGO_API_KEY` or
`FINANCE_AGENT_V2_CACHE_DIR` works without naming them here — useful in CI and batch
jobs where secrets already arrive in the environment.

## Run

Run from the repo root in the root venv. No server venv is needed to build the
dataset, since prepare reads the upstream prompts and tool schemas from the
committed `upstream_spec.json`.

```bash
source .venv/bin/activate

# 1. Build (or rebuild) the public 27Q set. Downloads the Vals public CSV if no
#    local source is present under data/.
gym eval prepare --benchmark finance_agent_v2

# 2. Evaluate against it (auto-serves the resources server + agent).
gym eval run --benchmark finance_agent_v2 \
  -c responses_api_models/openai_model/configs/openai_model.yaml \
  --output results/finance_agent_v2_27q.jsonl --concurrency 4
```

For a model comparison, name the model and take repeats — 27 questions is small
enough that a single pass cannot separate two models:

```bash
gym eval run --benchmark finance_agent_v2 \
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
  --output results/finance_agent_v2_27q.jsonl --concurrency 4
```

`reward` is **Partial Credit**: the severity-weighted share of rubric criteria
passed, forced to 0.0 if any criterion marked `must_pass` failed. `rubric_all_pass`
is the stricter every-criterion-passed flag, and `judge_error` marks rows the judge
could not score — filter those out rather than reading them as zeros.

`mean/rubric_partial_credit` and `mean/rubric_all_pass` match the *definitions* of
the Vals leaderboard's Accuracy and All-Pass columns, but not their numbers: this
measures the 27 public questions rather than their private 450-question split, with
our own judge rather than their three-model jury. `rubric_fraction` is ungated and
unweighted, so it has no counterpart there at all.

## Licensing & grading

Environment code (this recipe + `resources_servers/finance_agent_v2/`) is
**Apache-2.0**. Tools are **imported, not vendored**: `finance-agent` (MIT) and
`model-library` (MIT). The dataset derives from the **public** Vals FABv2 release,
subject to that project's terms. Vals's private grader is licensed and deliberately
not reproduced here. Detail:
[resources-server README](../../resources_servers/finance_agent_v2/README.md#licensing).
