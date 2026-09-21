# Finance SEC Search Resource Server

Financial information retrieval using SEC EDGAR filings with optional web search via Tavily.

**Companies listed in the [SEC company tickers file](https://www.sec.gov/files/company_tickers.json) are supported.** Set `supplementary_tickers_fpath` (and pass `--supplementary_tickers` to prefetch) to overlay extra ticker→CIK mappings that live SEC has dropped.

## Tools

| Tool | Description |
|------|-------------|
| `sec_filing_search` | Search SEC EDGAR for filing metadata by stock ticker symbol |
| `edgar_search` | Full-text search a read-only local SQLite FTS5 index |
| `parse_html_page` | Fetch and parse an HTML page, optionally cache SEC content, and store it under a key |
| `retrieve_information` | Query stored documents via LLM prompt with `{{key}}` placeholders |
| `submit_final_result` | Submit the final answer (keeps model in tool-calling mode until ready) |
| `web_search` | Internet search via Tavily API (optional — requires `tavily_api_key`) |

If `tavily_api_key` is not configured, `web_search` returns an error directing the model to use SEC tools instead.

## Setup

### env.yaml

Create `env.yaml` in the Gym root:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini

search_judge_model_base_url: https://api.openai.com/v1
search_judge_model_api_key: ${oc.env:OPENAI_API_KEY}
search_judge_model_name: gpt-5-mini

# Optional: set TAVILY_API_KEY to enable web_search.
tavily_api_key: ${oc.env:TAVILY_API_KEY,null}

# Required when the dataset exposes edgar_search.
local_edgar_index_path: /path/to/sap500_sec_fts.sqlite

# Optional per-process JSONL latency records.
local_edgar_metrics_dir: /path/to/search_metrics

# Optional. Defaults to the index path plus '.metadata' when that file exists.
local_edgar_metadata_path: /path/to/sap500_sec_fts.sqlite.metadata

# Filing content. Reads try cache_dir, then sec_dump_path, then SEC.gov.
# use_cache defaults to false, which fetches fresh every time; set it true for
# training so filings are fetched once and reused.
cache_dir: /path/to/shared/cache/finance_sec_search
use_cache: true
sec_dump_path: /path/to/step-0-download/data

# Latest filing date the date-filtered tools will return. Set this to the cutoff
# the dataset's prompts were written against.
max_end_date: "2025-04-07"
```

The config uses `${oc.select:tavily_api_key,null}`, so `web_search` is disabled
when `tavily_api_key` is omitted or null.

## Local EDGAR index

`edgar_search` reads a read-only SQLite full-text index of SEC filings. Build
its metadata sidecar once per index, which is what keeps common queries under a
second instead of tens of seconds:

```bash
python resources_servers/sec_local_index/scripts/build_local_edgar_metadata.py \
  --index /path/to/sap500_sec_fts.sqlite
```

See [local-edgar-index.md](../sec_local_index/docs/local-edgar-index.md) for the schema an
index must have, the column formats that matter, and how to obtain one.

## Cache Management

The resource server can cache SEC data locally to avoid redundant API calls and
enable offline operation after the first fetch. Caching is opt-in.

### Enabling / disabling the cache (`use_cache`)

| `use_cache` | Behavior |
|-------------|----------|
| `false` (default) | The on-disk cache is fully bypassed — no cache directories are created and **every request fetches fresh filings live**. |
| `true` | The on-disk cache under `cache_dir` is read and written: ticker mappings, filing metadata, and parsed filing content are cached and reused across requests/runs. |

Keep `use_cache: false` (the default) for **eval**, so every rollout is scored
against current filings. Set it to `true` for **training**, where thousands of
concurrent rollouts would otherwise be rate-limited by SEC.gov.

### Settings for training runs

Training needs more than `use_cache`. The full set, with the eval default shown
for contrast:

| Setting | Eval default | Training | Why |
|---------|--------------|----------|-----|
| `use_cache` | `false` | `true` | Serve filings from `cache_dir` instead of live SEC.gov. |
| `cache_dir` | `~/.cache/...` | shared absolute path | Must be visible from every rollout worker. |
| `supplementary_tickers_fpath` | `null` | path to the overlay JSON | Resolve filers that live SEC no longer lists. |
| `max_end_date` | `null` | dataset cutoff | Latest filing date any date-filtered tool returns. |
| `reward_mode` | `binary` | `scaled` | Partial credit: `[[1]]` scores 0.5 instead of 0.0. |
| `max_rollout_time_seconds` | `null` | e.g. `1800` | Without it a stalled rollout blocks the batch. |

Set `max_end_date` to the date the dataset's prompts were written against.
Leaving it `null` lets the agent retrieve filings published after that date. It
also matters for `edgar_search`, which falls back to an internal default when
`max_end_date` is unset — searches are then silently capped at that date rather
than at your dataset's cutoff.

One agent-side setting lives in the `responses_api_agents` block rather than
here:

```yaml
responses_api_agents:
  finance_agent:
    continue_if_not_tool_call: false
```

Eval leaves this `true`, so a text-only turn gets a nudge and the episode
continues. Training sets it `false` because RL trainers require each turn's
prompt tokens to extend the previous turn's. The consequence is that a
text-only turn ends the episode before `submit_final_result`, which scores 0 —
so track how many rollouts never submit, not just mean reward.

With a local corpus or index available, also set `sec_dump_path` and
`local_edgar_index_path` (see [Local EDGAR index](#local-edgar-index)).

### What is cached

| Directory | Contents |
|-----------|----------|
| `filings_metadata/{CIK}.json` | Filing metadata (accession numbers, dates, forms) per company |
| `filings/{CIK}/{accession}/{document}.txt` | Parsed filing content (HTML to text) |
| `tickers.json` | SEC ticker-to-CIK mapping |

Prefetch writes `tickers.json` and `filings_metadata/` only. Filing bodies under `filings/` are written by the resource server on the first `parse_html_page` of an EDGAR URL when `use_cache: true`.

### Cache location

| Scenario | Location |
|----------|----------|
| `cache_dir` set to an absolute path | Uses that path directly |
| `cache_dir` set to a relative path | Resolved from the current working directory |
| `cache_dir` not set (null) | `~/.cache/nemo_gym/finance_sec_search/` |

**Important**: The default `~/.cache/...` path is only suitable for local
development on a workstation. In containerized or Slurm environments this path
is **ephemeral** (destroyed when the container exits) and **not shared** across
jobs -- each seed runs in its own container and cannot see another seed's cache.
For multi-seed rollouts or any production use, always set `cache_dir` to a
shared, persistent absolute path on a mounted filesystem (e.g.
`/workspace/cache/finance_sec_search`).

### Pre-warming the cache (prefetch)

The `prefetch_sec_metadata.py` script populates the metadata cache for a set of
companies **before** starting rollouts. Set `use_cache: true` and configure the
same `cache_dir` for the resource server so rollouts read the prefetched files.

**Requirements**: Python 3.10+, `aiohttp`, `pyyaml` (both are Gym
dependencies). Internet access to SEC.gov is required. No GPU, no model server,
and no running Gym server needed.

`--cache_dir` must be the same absolute path the resource server will use. On a
shared filesystem (for example Lustre), put it on that mount so every node sees
the files. `~/.cache/...` is not shared across Slurm jobs.

```bash
# Prefetch for specific tickers:
python resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py \
    --cache_dir /shared/cache/finance_sec_search \
    --tickers AAPL MSFT NVDA GOOG AMZN

# Or with a YAML ticker list (expects a 'tickers' key with a list):
python resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py \
    --cache_dir /shared/cache/finance_sec_search \
    --ticker_config /path/to/tickers.yaml

# Overlay extra ticker→CIK mappings (Vals v1 names dropped from live SEC).
# The overlay path is repo-root relative. Overlay CIKs are prefetched in
# addition to --tickers, so RDFN/SAVE/X are covered even if not listed.
python resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py \
    --cache_dir /shared/cache/finance_sec_search \
    --tickers AAPL MSFT \
    --supplementary_tickers benchmarks/finance_sec_search/data/supplementary_tickers.json

# Force refresh (re-fetch even if cached):
python resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py \
    --cache_dir /shared/cache/finance_sec_search \
    --tickers AAPL --force
```

Start the server against that directory (eval configs leave `use_cache` false
unless you override it):

```bash
gym env start \
    --model-type vllm_model \
    --benchmark finance_sec_search/config_no_web_search \
    +use_cache=true \
    +cache_dir=/shared/cache/finance_sec_search
```

The first rollout still downloads filing bodies and writes them under `filings/`.
A second identical run should log `SEC filing reads by source: cache=N ...`
instead of `live=N` for those same EDGAR URLs.

`cache_dir` also relocates Gym's `uv` cache to `{cache_dir}/uv`, so pointing it
at a fresh directory rebuilds every server virtualenv on startup. Reuse one
path across runs.

The script is **idempotent**: it skips companies whose cache file already exists
(unless `--force` is used).

### Without prefetch

With `use_cache: true`, an empty cache is populated lazily on first access. With
the default `use_cache: false`, every request fetches fresh data and SEC.gov
connectivity is required throughout the rollout.

### Shared cache

Multiple seeds or runs can share the same `cache_dir`. With prefetch, all GPU
jobs are read-only (no race conditions). Without prefetch, concurrent writes are
benign because all writers produce identical data for the same company.

## End-to-End Rollout

### 1. Prepare the dataset

#### Custom questions (`convert_questions.py`)

The input is a JSONL file with question/answer pairs. An example is provided at
`resources_servers/finance_sec_search/data/example_questions.jsonl`:

```json
{"question": "What is the number of shares of common stock outstanding as of November 14, 2025 for Nvidia?", "expected_answer": "24.3 billion"}
{"question": "As of September 24, 2022 how many full-time equivalent employees did Apple have?", "expected_answer": "164,000"}
```

Convert raw questions into Gym input format (adds tool definitions, system prompt, etc.):

```bash
python resources_servers/finance_sec_search/scripts/convert_questions.py \
  --input resources_servers/finance_sec_search/data/example_questions.jsonl \
  --output resources_servers/finance_sec_search/data/example.jsonl
```

Add `--include-web-search` / `-w` to include the optional `web_search` tool:

```bash
python resources_servers/finance_sec_search/scripts/convert_questions.py \
  --input resources_servers/finance_sec_search/data/example_questions.jsonl \
  --output resources_servers/finance_sec_search/data/example.jsonl \
  --include-web-search
```

Select the local full-text search contract with `--search-tool edgar_search`:

```bash
python resources_servers/finance_sec_search/scripts/convert_questions.py \
  --input resources_servers/finance_sec_search/data/example_questions.jsonl \
  --output resources_servers/finance_sec_search/data/example_edgar_search.jsonl \
  --search-tool edgar_search
```

This keeps the same user prompt and companion tools while replacing
`sec_filing_search` with the agent-facing `edgar_search` schema. The
`/edgar_search` route reads `local_edgar_index_path` in read-only immutable mode
and returns sec-api-compatible filing metadata. It does not call sec-api.io.
`parse_html_page` remains separate and may read the filing cache, filing dump,
or SEC.gov.

A pre-converted `example.jsonl` (without web search) is checked in and ready to
use — you only need to re-run `convert_questions.py` if you modify the raw
questions or want to change the tool set.

#### Vals AI public benchmark (`prepare.py`)

The [Vals AI finance-agent](https://github.com/vals-ai/finance-agent) 50-question
public benchmark lives in `benchmarks/finance_sec_search/`. It downloads the
`public.csv` dataset from GitHub and converts it to Gym format:

```bash
# Prepare via Gym CLI (recommended — used by gym env start with benchmark configs):
gym eval prepare --benchmark finance_sec_search/config_no_web_search

# Or run the script directly:
python benchmarks/finance_sec_search/prepare.py            # without web_search
python benchmarks/finance_sec_search/prepare.py --include-web-search  # with web_search
```

Output is written to `benchmarks/finance_sec_search/data/`:

| Config | Prepare script | Output file |
|--------|---------------|-------------|
| `config_no_web_search.yaml` | `prepare.py` | `finance_sec_search_benchmark.jsonl` |
| `config_web_search.yaml` | `prepare_web_search.py` | `finance_sec_search_benchmark_web_search.jsonl` |

> **Note:** `prepare.py` duplicates the prompt and tool definitions from
> `convert_questions.py`. They are functionally identical — `convert_questions.py`
> is the canonical source for custom questions, while `prepare.py` is specific to
> downloading and converting the Vals AI dataset.

#### SECQUE benchmark

The open-book [SECQUE benchmark](../../benchmarks/secque/README.md) is a
separate benchmark recipe that reuses the generic `equivalence_llm_judge`
resource server. It does not use the tool-calling server documented here.

### 2. Start the vLLM server

Launch a vLLM-compatible model server (e.g. Qwen3-30B-A3B) so the policy and judge endpoints are available.

### 3. Start the Gym servers

With a local vLLM model server:

```bash
gym env start \
  --model-type vllm_model \
  --resources-server finance_sec_search
```

Or with an OpenAI-compatible API (e.g. OpenAI, Azure, NIM):

```bash
gym env start \
  --model-type openai_model \
  --resources-server finance_sec_search
```

### 4. Collect rollouts

```bash
gym eval run --no-serve \
  --agent finance_agent \
  --input resources_servers/finance_sec_search/data/example.jsonl \
  --output results/finance_sec_search_rollouts.jsonl
```

Add `--limit 1` for a quick single-question test:

```bash
gym eval run --no-serve \
  --agent finance_agent \
  --input resources_servers/finance_sec_search/data/example.jsonl \
  --output results/finance_sec_search_rollouts.jsonl \
  --limit 1
```

### Run tests

```bash
gym env test --resources-server finance_sec_search
```

## Verification

Uses LLM-as-judge with a financial grading rubric (0/1/2 scale). `reward_mode: binary` gives reward 1.0 only to fully correct answers (`[[2]]`); `reward_mode: scaled` also gives partially correct answers (`[[1]]`) 0.5. The judge additionally requires grounding: an answer that restates general financial knowledge without filing-specific figures, dates, or disclosures scores `[[0]]` even when it is conceptually correct. The judge prompt and rubric are defined in /prompt_templates.

