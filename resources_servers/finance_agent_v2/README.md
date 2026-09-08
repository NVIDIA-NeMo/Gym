# Finance Agent v2 (FABv2) Resource Server

A NeMo Gym integration of the official [Vals Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2)
that **reuses Vals's own tool code directly** (tools-only wrap) instead of
reimplementing it. The upstream `finance_agent.tools.*` classes are imported and
exposed as HTTP endpoints; the shared `responses_api_agents/finance_agent` loop
drives them, configured to Vals's v2 harness policy in
`configs/finance_agent_v2.yaml`. The public release ships no grader, so scoring is
our own per-criterion rubric judge — see [Verification](#verification).

`tests/test_upstream_parity.py` checks that configured policy against the
installed upstream package, so a pin bump that changes the harness fails here.

## Tools (imported from `finance_agent.tools`)

| Tool | Description | Requires |
|------|-------------|----------|
| `web_search` | Tavily web search (`TavilyWebSearch`) | `tavily_api_key` |
| `edgar_search` | sec-api.io full-text EDGAR search (`EDGARSearch`) | `sec_api_key` |
| `price_history` | Tiingo daily OHLC for equity/etf/crypto/fx (`PriceHistory`) | `pricing_data_api_key` |
| `parse_html_page` | Fetch + parse a page to text, store under a key (`ParseHtmlPage`) | — |
| `retrieve_information` | LLM over stored docs via `{{key}}` prompts (`RetrieveInformation`) | `retrieval_model_server` |
| `calculator` | Safe arithmetic via simpleeval (`Calculator`) | — |
| `submit_final_result` | Submit the final answer; ends the loop (`SubmitFinalResult`) | — |

`parse_html_page` and `retrieve_information` share a per-session data storage
(`state`) dict, scoped by the HTTP session cookie. A tool whose required key or
model is not configured is registered as unavailable and its endpoint returns a
clear error, so the agent can route around it.

## Dependencies

Both upstream pins are exact so the benchmark stays reproducible:

```
-e nemo-gym[dev] @ ../../
model-library==0.1.25
finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@<pinned-sha>
```

`model-library` declares `openai[aiohttp]>=2.28.0`, which cannot resolve against
the vllm-compatible `openai` that Gym pins. The upstream code runs fine on the
older client, so only the declared floor is the problem, and `overrides.txt` drops
it with `openai[aiohttp]<=2.7.2`. That line must mirror the cap in the root
`pyproject.toml`, because `uv --override` replaces *every* declared constraint on a
package, Gym's included; a test asserts the two stay in sync.

`finance-agent` publishes no tags and is not on PyPI, so it is pinned to a commit.
nemo-gym requires Python >=3.13.14, the binding floor here.

## Setup (`env.yaml`)

Run this environment with **two configs**: this environment config
(`configs/finance_agent_v2.yaml`) plus a model config
(`responses_api_models/openai_model/configs/openai_model.yaml`, or
`vllm_model.yaml` for a self-hosted endpoint).

Endpoints and tool API keys all go in `env.yaml` at the repo root, which is
gitignored. Only the policy endpoint is required: a tool key left unset simply
registers its tool as unavailable instead of failing startup.

```yaml
# env.yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

# Judge model used by /verify. Kept separate from the policy on purpose, so that
# models evaluated on this benchmark are all graded by the same judge.
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
resolves as config key → environment variable → null-safe default, so exporting
`SEC_API_KEY`, `TAVILY_API_KEY`, `TIINGO_API_KEY` or `FINANCE_AGENT_V2_CACHE_DIR`
works without naming them here — useful in CI and batch jobs where secrets already
arrive in the environment. The `policy_*` keys have no default and must come from
`env.yaml` or a CLI override.

Grading works without network egress, but the tools do not: they call Tavily,
sec-api.io and Tiingo at rollout time.

## Run

The public benchmark recipe (dataset + `prepare.py` + config) lives in
`benchmarks/finance_agent_v2/`; the server code and its `gym env test` fixtures
stay here. There is deliberately no `environments/finance_agent_v2/` entry:
training runs this same resources server on externally generated SDG data.

```bash
# Unit tests + example-data validation for this resources server
gym env test --resources-server finance_agent_v2

# Build the public 27Q benchmark JSONL (downloads the Vals CSV if absent)
gym eval prepare --benchmark finance_agent_v2

# End-to-end (prepare + start servers + collect rollouts) on the benchmark set
gym eval run --benchmark finance_agent_v2 \
  -c responses_api_models/openai_model/configs/openai_model.yaml
```

Add `--limit 3 --concurrency 3` for a smoke run. `gym eval run` writes the scored
rollouts plus `<output>_aggregate_metrics.json`.

Each prepared sample carries the upstream prompts and tool JSON schemas, read at
prepare time from `benchmarks/finance_agent_v2/upstream_spec.json` — a generated
snapshot, because `finance_agent` is installed in this server's venv only while
`gym eval prepare` runs in the repo-root venv. Bumping the pin means editing
`requirements.txt` and `_UPSTREAM_SHA` in `prepare.py` together, `uv sync`, then
regenerating the snapshot from this venv and re-running `gym eval prepare`:

```bash
cd resources_servers/finance_agent_v2 && source .venv/bin/activate
python scripts/export_upstream_spec.py
```

`tests/test_upstream_spec.py` fails if the snapshot drifts from the installed
package, and `prepare.py` refuses to run when the two shas disagree.

## Caching

Tiingo and SEC calls are rate-limited and, within the pinned window, return
immutable data, so the server caches them to disk. Each `Cached*` class subclasses
the upstream tool, overrides only its network method, and stores the **raw upstream
response**, which untouched upstream code then re-serializes. A hit is therefore
byte-identical to a live call and survives an upstream formatting bump.

| Source | Cached unit |
|--------|-------------|
| `price_history` (Tiingo) | per-`(endpoint, ticker)` master of raw records, sliced on read by the upstream `_records_to_csv` |
| `edgar_search` (sec-api.io) | raw `filings` list keyed by the normalized request |
| `parse_html_page` on sec.gov filing URLs | parsed text at `sec_filings/<cik>/<accession>/<primary-doc>.txt`; general web is not cached |

Caching is on by default. Point `cache_dir` at a shared absolute path so it
survives across jobs:

```yaml
use_cache: true                            # true = read+write, false = off
cache_dir: /shared/cache/finance_agent_v2  # null -> ~/.cache/nemo_gym/finance_agent_v2
```

`cache_dir` also reads `FINANCE_AGENT_V2_CACHE_DIR`. Tiingo responses are cached
verbatim and never re-derived, and extending a window keeps already-cached rows, so
an adjusted price level cannot drift with the fetch date between re-runs.

To populate the cache ahead of an offline eval:

```bash
python resources_servers/finance_agent_v2/scripts/prefetch_prices.py \
  --cache-dir /shared/cache/finance_agent_v2 \
  --tickers AAPL MSFT NVDA --asset-class equity
```

## Verification

Each criterion in the dataset's `rubric` is judged separately against the answer
the agent passed to `submit_final_result`. Per criterion the judge returns a binary
verdict as JSON; calls repeat until `judge_required_successes` (default 3) replies
parse, capped at `judge_max_attempts` (default 10). Errors and unparseable replies
consume an attempt and are retried; only a reply that ran out of output budget
escalates that budget. The criterion's score is the majority of those verdicts, so
an odd count cannot tie.

Reward is **Partial Credit**: the severity-weighted pass fraction, forced to `0.0`
if any `must_pass` dealbreaker failed. Weights come from each criterion's
`modifiers` in the dataset and are applied after judging — the judge never sees
them, so a criterion cannot be graded more harshly for being expensive to fail.
Criteria without `modifiers` default to severity 1.0 and non-gating, so an
unweighted dataset scores as a plain pass fraction.

| Field | Meaning |
|---|---|
| `reward` | Partial Credit — the weighted pass fraction, zeroed by any failed dealbreaker |
| `rubric_partial_credit` | the same number under Vals's name for it |
| `rubric_weighted_fraction` | Partial Credit *before* gating; the gap is what the dealbreakers cost |
| `rubric_all_pass` | `true` only when every criterion resolved and passed — Vals's All-Pass |
| `rubric_fraction` | criteria passed / total, unweighted |
| `rubric_dealbreakers_failed` / `rubric_dealbreakers_total` | how many `must_pass` criteria there were, and how many failed |
| `rubric_unresolved` | criteria that never got enough parsable verdicts |
| `rubric_judgements[]` | per-criterion votes, evidence, weights used, and sampled failed replies |
| `judge_error` | set when scoring could not complete — **filter these out rather than reading them as zeros** |

A criterion the judge never resolved is a judge failure, not a miss, so it scores
`null` rather than `0` and flags the row via `judge_error`. It still weighs as
not-passed, because treating an absent verdict as a pass would inflate exactly the
runs whose judging broke. A row with no rubric at all scores 0 with `judge_error`
set, so the agent and tools can be validated before labels exist.

The agent records why its loop ended in `response.metadata.stop_reason`
(`done_tool`, `max_turns`, `max_time`, `max_output_tokens`, `error`). Under
dealbreaker gating a truncated trajectory scores the same as a confidently wrong
one, so check this before reading a low score as a capability result.

Because both the judge model and its prompt are ours, scores are **not** comparable
to Vals's published numbers. The prompt lives in
`prompt_templates/finance_agent_v2_rubric_judge.yaml`.

`verify` is a pure function of the request body plus this server's config, so the
server declares `REVERIFY_MODE = STATELESS`: after changing the judge prompt or
parameters you can rescore stored rollouts without re-running the policy.

```bash
gym eval reverify --benchmark finance_agent_v2 --model-type openai_model \
  --inputs results/<run>_materialized_inputs.jsonl \
  --rollouts results/<run>.jsonl \
  --output results/<run>_rejudged.jsonl --concurrency 4
```

Select the **same** config the rollouts were collected with: reverify routes each
row by the agent name stored on it. `--model-type` is required either way, because
the `retrieval_model_server` reference must resolve even though reverify only calls
`/verify`.

## Licensing

**This environment's code** (everything under `resources_servers/finance_agent_v2/`
and `benchmarks/finance_agent_v2/`) is licensed under **Apache-2.0**, consistent
with NeMo Gym and the SPDX headers in each source file.

**Upstream dependencies** (imported, not vendored — see `requirements.txt`):

| Package | Source | License |
|---------|--------|---------|
| `finance-agent` (`finance_agent.tools` / `finance_agent.prompt`; also the source of the benchmark's `upstream_spec.json`) | [vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2) | MIT |
| `model-library` (`model_library.*`) | [vals-ai/model-library](https://github.com/vals-ai/model-library)@`0.1.25`, from PyPI | MIT |

We import these packages at install time and do not copy their source into this
repo, so their MIT terms apply to that code as distributed upstream.

**Dataset.** The `example.jsonl` fixtures here and the prepared benchmark JSONL
derive from the **public** Vals Finance Agent Benchmark v2 release; use is subject
to that project's terms. The public release ships no official grader.

**Grading is our own.** Vals's private grader was obtained under a separate license
and is **deliberately not reproduced** here; the judge prompt was written from
scratch, so scores are not directly comparable to Vals's published numbers.
