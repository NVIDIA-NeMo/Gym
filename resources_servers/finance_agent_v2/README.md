# Finance Agent v2 (FABv2) Resource Server

A NeMo Gym integration of the official [Vals Finance Agent Benchmark v2](https://github.com/vals-ai/finance-agent-v2)
that **reuses Vals's own tool code directly** (tools-only wrap) instead of
reimplementing it. The upstream `finance_agent.tools.*` classes are imported and
exposed as HTTP endpoints; the existing nemo-gym `finance_agent` agent loop drives
them. Scoring uses our own judge (path A) — see [Verification](#verification).

This is the v2 counterpart to `resources_servers/finance_sec_search` (v1). The key
difference: v1 reimplements the tools; v2 imports them from upstream so the tool
descriptions, parameters, and behavior track Vals automatically via a dependency bump.

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
| `sec_filing_search` | **Optional** (training/SDG): data.sec.gov ticker→CIK filing listing (local `SecFilingSearch`, not from Vals) | — |

`parse_html_page` and `retrieve_information` share a per-session **data storage**
(`state`) dict, scoped by the HTTP session cookie. A tool whose required key/model
is not configured is registered as unavailable and its endpoint returns a clear
error (the agent can route around it). New in v2 vs v1: `calculator`, `price_history`.
The upstream date clamp is `MAX_END_DATE = 2026-03-01` (enforced inside the Vals tools).

**SEC tool surface (`enabled_sec_tools`).** `edgar_search` (sec-api.io full-text
search) is byte-parity with Vals — use it for **eval** (the default). For
**training/SDG** you can additionally enable `sec_filing_search`, a cheaper
data.sec.gov ticker→CIK listing that needs no key; it changes the tool surface vs
Vals and is **not** byte-parity, so keep it out of eval. See
[Caching](#caching-pricing--sec).

## Dependencies

`requirements.txt` pins both upstream packages from git (`model-library` is **not**
on PyPI; `finance-agent` requires `model-library==0.1.25`, i.e. tag `v0.1.25`):

```
-e nemo-gym[dev] @ ../../
model-library @ git+https://github.com/vals-ai/model-library.git@v0.1.25
finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@<pinned-sha>
```

Both nemo-gym and finance-agent require Python >=3.12.

## Setup (`env.yaml`)

This is a self-contained Gym environment: run it with **two configs** — this
environment config (`configs/finance_agent_v2.yaml`) plus a model config
(`responses_api_models/openai_model/configs/openai_model.yaml` for OpenAI, or
`responses_api_models/vllm_model/configs/vllm_model.yaml` for a self-hosted
vLLM endpoint).

Secrets live in `env.yaml` at the repo root (gitignored — never commit a populated
copy). Copy `env.yaml.example` and fill in. Only the **model endpoints** go here;
the **tool API keys are read directly from your shell** by the environment config
(`${oc.env:SEC_API_KEY}` etc.), so just export them:

```bash
export OPENAI_API_KEY=...        # policy + judge (OpenAI)
export SEC_API_KEY=...           # edgar_search (sec-api.io)
export TAVILY_API_KEY=...        # web_search (Tavily)
export TIINGO_API_KEY=...        # price_history (Tiingo)
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

> Note: "airgap-friendly" applies to **grading** only. The v2 tools call external
> APIs (Tavily, sec-api.io, Tiingo) at rollout time and need network egress + keys.
> A tool whose key is unset simply registers as unavailable (no crash).

## Run

The gym CLI (NeMo Gym >= 0.5.0) drives everything. The public benchmark recipe
(dataset + `prepare.py` + config) lives in `benchmarks/finance_agent_v2/`; the
server code and its `gym env test` fixtures (`data/example*.jsonl`,
`data/example_metrics.json`) stay here under `resources_servers/finance_agent_v2/`.
There is no `environments/finance_agent_v2/` entry: training runs this same
resources server on externally generated SDG data.

```bash
# Unit tests + example-data validation for this resources server
gym env test --resources-server finance_agent_v2

# Build the public 27Q benchmark JSONL (downloads the Vals CSV if absent)
gym eval prepare --benchmark finance_agent_v2

# End-to-end (prepare + start servers + collect rollouts) on the benchmark set
gym eval run --benchmark finance_agent_v2 \
  -c responses_api_models/openai_model/configs/openai_model.yaml
```

### Quickstart: public 27-question smoke run (OpenAI gpt-5-mini)

Run the Vals public question set end-to-end on the OpenAI API (`gpt-5-mini` for
both policy and judge), limited to 3 rollouts to confirm the agent + tools +
binary grading path works. Two configs only: the environment config and the
OpenAI model config.

1. **Secrets** — populate `env.yaml` (model endpoints) and export the tool keys
   (`OPENAI_API_KEY`, `SEC_API_KEY`, `TAVILY_API_KEY`, `TIINGO_API_KEY`) as shown
   in [Setup](#setup-envyaml).

2. **Prepare** the benchmark JSONL. `gym eval prepare` runs
   `benchmarks/finance_agent_v2/prepare.py`, which downloads the raw Vals public
   CSV from [finance-agent-v2](https://github.com/vals-ai/finance-agent-v2) (if no
   local source is present) and converts it. Prompts come from
   `finance_agent.prompt` and tool schemas from `finance_agent.tools`. The public
   CSV ships only rubric *criteria* (no single gold answer), so the script
   synthesizes a GOLD `expected_answer` from those criteria for our judge. The
   CSV's `rubric` is also copied through verbatim for reference only (it is **not**
   used for reward). Run from the resource server's venv (has `finance_agent`):

   ```bash
   source resources_servers/finance_agent_v2/.venv/bin/activate
   gym eval prepare --benchmark finance_agent_v2
   ```

3. **Start the servers** — resources server + OpenAI model config:

   ```bash
   gym env start --resources-server finance_agent_v2 \
     -c responses_api_models/openai_model/configs/openai_model.yaml
   ```

4. **Collect rollouts** against the running servers, limited to 3 questions
   (`--no-serve` reuses the servers from step 3):

   ```bash
   gym eval run --no-serve \
     --agent finance_agent \
     --input benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl \
     --output results/finance_agent_v2_smoke.jsonl \
     --limit 3 --concurrency 3
   ```

   Rewards land in `results/finance_agent_v2_smoke.jsonl` (1.0 = judge rated
   `[[2]]`, i.e. the answer covers all required facts in the GOLD reference;
   else 0.0). Drop `--limit` to run the full set. To run on a
   self-hosted model, use `-c responses_api_models/vllm_model/configs/vllm_model.yaml`
   and point `policy_*` in `env.yaml` at your vLLM endpoint.

## Caching (pricing / SEC)

Tiingo and SEC calls are rate-limited and (within the pinned window) return
**immutable** data, so the server can cache them to disk. Caching is a thin
**wrapper around the imported upstream tools** — never a reimplementation: each
`Cached*` class subclasses the upstream tool and overrides only its network
method, stores the **raw upstream response**, and re-serializes it with the
untouched upstream code. A cache hit is therefore **byte-identical** to a live
call (and survives a future upstream formatting/SHA bump without a refetch). See
`cache.py` and `cached_tools.py`.

Caching is on by default (`use_cache: true`); point `cache_dir` at a shared
absolute path (config, or `FINANCE_AGENT_V2_CACHE_DIR`) so it survives across
jobs. When `use_cache` is on the cache both serves hits and persists misses;
set `use_cache: false` to run every tool live.

```yaml
use_cache: true                            # true = read+write, false = off
cache_dir: /shared/cache/finance_agent_v2  # null -> ~/.cache/nemo_gym/finance_agent_v2
```

`edgar_search` and `sec_filing_search` share the same on-disk cache root and
document layout: whichever tool the agent used to locate a filing, the actual
document bytes are fetched by `parse_html_page` and land at the same
`sec_filings/<cik>/<accession>/<file>` path, so the two never re-download the
same document.

What gets cached, and why it stays in sync with Vals:

| Source | Cached unit | Parity |
|--------|-------------|--------|
| `price_history` (Tiingo) | per-`(endpoint, ticker)` master of raw records; sliced on read via the upstream `_records_to_csv` | byte-identical |
| `edgar_search` (sec-api.io) | raw `filings` list in `edgar_search/<query-slug>_<hash>.json` (request stored alongside for debugging) | byte-identical |
| `parse_html_page` on **sec.gov** filing URLs | parsed text at `sec_filings/<cik>/<accession>/<primary-doc>.txt` | identical parse; general web is **not** cached |

**Price as-of / drift.** Raw prices are immutable, and returns/ratios are
invariant to the adjustment as-of date; only an *adjusted absolute level* can
drift with the fetch date. We cache Tiingo's response **verbatim** (never
re-derive adjusted), and on a window extension keep already-cached rows, so
re-runs are reproducible.

**Prefetch (offline eval).** Populate the cache sequentially ahead of an eval so
runtime price queries are served from disk instead of hitting Tiingo:

```bash
python benchmarks/finance_agent_v2/scripts/prefetch_prices.py \
  --cache-dir /shared/cache/finance_agent_v2 \
  --tickers AAPL MSFT NVDA --asset-class equity
```

## Dataset & labels (path-A scoring)

The public FABv2 release ships **only question strings** (no ground truth/grader).
`benchmarks/finance_agent_v2/prepare.py` (the `gym eval prepare` entry point)
loads input from its `data/` dir by this precedence:

1. `data/labeled.jsonl` — labeled rows (enables real scoring)
2. `data/public.jsonl` — rows with at least `{question}`
3. `data/public.txt` — one question per line (FABv2 public format)
4. `data/public.csv` — raw Vals public CSV (`question`/`prompt` column)

**Labeled JSONL schema** (one object per line):

```json
{"question": "...", "expected_answer": "...", "rubric": "[{\"operator\": \"...\", \"criteria\": \"...\"}]"}
```

- `expected_answer` is the GOLD reference used by our `[[N]]` judge — this is what
  drives reward.
- `rubric` is propagated from the public CSV verbatim for reference/completeness
  only. It is **not** consumed by scoring. (The public FABv2 release has no
  official grader; Vals's private per-criterion rubric grader is licensed and is
  deliberately not reproduced here.)
- To source labels at scale, publish a labeled set to the GitLab Model Registry
  (mirrors v1's `finance_sec_search_vals_200_eval`) and point the dataset entry in
  `benchmarks/finance_agent_v2/config.yaml` at it (`type: benchmark` +
  `gitlab_identifier`).

**Interim dry-run:** with no labels, `/verify` returns `reward=0` so the agent +
tools path can be validated before ground truth is available.

## Verification

The public FABv2 release ships **no official grader**, so scoring uses **our own**
approximation: the legacy `[[0]]/[[1]]/[[2]]` judge from
`resources_servers/finance_sec_search`. The public CSV has no single gold answer,
so the GOLD `expected_answer` is synthesized from the rubric criteria (see
`benchmarks/finance_agent_v2/prepare.py`); the judge awards `[[2]]` (reward 1.0) only when
the answer covers all required facts. The dataset's `rubric` field plays **no**
role in reward.

Set `reward_mode` in the resource server config:

| Mode | Mapping | Use |
|------|---------|-----|
| `binary` | `[[2]]` → 1.0, else 0.0 | **default (public)** — strict pass/fail |
| `scaled` | `[[0]]`/`[[1]]`/`[[2]]` → 0.0/0.5/1.0 | shaped reward (training) |

Judge prompts live in `prompt_templates/`.

## File structure

```
resources_servers/finance_agent_v2/         # server code + gym env test fixtures
├── app.py                         # Resource server: tool endpoints + retrieval shim + verify
├── cache.py                       # ToolCache: namespaced atomic disk cache + read/write policy
├── cached_tools.py                # Cached* wrappers (price/edgar/parse) + SecFilingSearch
├── requirements.txt               # Pins nemo-gym + Vals model-library (NVIDIA fork) + finance-agent
├── env.yaml.example
├── configs/
│   └── finance_agent_v2.yaml      # Resources-server config used by gym env test / gym dataset collate
│                                  # (the benchmark recipe config_paths-chains to this; no duplication)
├── prompt_templates/              # judge / retrieval (loaded at runtime; server cwd = this dir)
├── data/                          # gym env test fixtures: example.jsonl (5), example_metrics.json, example_rollouts.jsonl
└── tests/                         # test_app.py (server), test_cache.py (cache layer)

benchmarks/finance_agent_v2/                # public eval recipe (self-contained)
├── config.yaml                    # Thin overlay: config_paths -> resources config + _inherit_from + benchmark dataset
├── prepare.py                     # gym eval prepare entry point + CSV->JSONL converter (builds tool schemas from upstream classes)
├── scripts/
│   ├── prefetch_prices.py         # Sequential Tiingo prefetch into the cache (idempotent/resumable)
│   └── compare_runs.py            # Compare rollout JSONLs by per-question reward
└── data/                          # gitignored; prepare.py regenerates vals_v2_public_27q.jsonl from the upstream Vals export
```

## Licensing

**This environment's code** (everything under `resources_servers/finance_agent_v2/`
and `benchmarks/finance_agent_v2/`) is licensed under **Apache-2.0**, consistent
with NeMo Gym and the SPDX headers in each source file (`app.py`,
`tests/test_app.py`, `benchmarks/finance_agent_v2/prepare.py`).

**Upstream dependencies** (imported, not vendored — see `requirements.txt`):

| Package | Source | License |
|---------|--------|---------|
| `finance-agent` (`finance_agent.tools` / `finance_agent.prompt`) | [vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2) | MIT |
| `model-library` (`model_library.*`) | NVIDIA fork of [vals-ai/model-library](https://github.com/vals-ai/model-library)@`v0.1.25` (openai floor dropped, static version) | MIT |

We import these packages at install time and do not copy their source into this
repo, so their MIT terms apply to that code as distributed by the upstream/fork.

**Dataset.** The `example.jsonl` fixtures (here) and
`benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl` derive
from the **public** Vals Finance Agent Benchmark v2 release
([vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2)); use is
subject to that project's terms. The public release ships **no official grader**.

**Grading is our own.** Reward is computed by our `[[0]]/[[1]]/[[2]]` judge (an
approximation reused from `resources_servers/finance_sec_search`) against a GOLD
`expected_answer` we synthesize from the public rubric criteria. The dataset's
`rubric` field is propagated **for reference only** and is not used for scoring.
Vals's private per-criterion rubric grader (prompts + reward logic) was obtained
under a separate license and is **deliberately not reproduced** in this public
code.
