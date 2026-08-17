# Finance SEC Search Resource Server

Financial information retrieval using SEC EDGAR filings with optional web search via Tavily.

**Only companies listed in the [SEC company tickers file](https://www.sec.gov/files/company_tickers.json) are supported.** Questions about companies not in this list will fail at the ticker lookup step.

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
```

The config uses `${oc.select:tavily_api_key,null}`, so `web_search` is disabled
when `tavily_api_key` is omitted or null.

## Cache Management

The resource server can cache SEC data locally to avoid redundant API calls and
enable offline operation after the first fetch. Caching is opt-in.

### Enabling / disabling the cache (`use_cache`)

| `use_cache` | Behavior |
|-------------|----------|
| `false` (default) | The on-disk cache is fully bypassed — no cache directories are created and **every request fetches fresh filings live**. |
| `true` | The on-disk cache under `cache_dir` is read and written: ticker mappings, filing metadata, and parsed filing content are cached and reused across requests/runs. |

Keep `use_cache: false` (the default) for **eval**

### What is cached

| Directory | Contents |
|-----------|----------|
| `filings_metadata/{CIK}.json` | Filing metadata (accession numbers, dates, forms) per company |
| `filings/{CIK}/{accession}.txt` | Parsed filing content (HTML to text) |
| `tickers.json` | SEC ticker-to-CIK mapping |

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

```bash
# Prefetch for specific tickers:
python resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py \
    --cache_dir /path/to/cache \
    --tickers AAPL MSFT NVDA GOOG AMZN

# Or with a YAML ticker list (expects a 'tickers' key with a list):
python resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py \
    --cache_dir /path/to/cache \
    --ticker_config /path/to/tickers.yaml

# Force refresh (re-fetch even if cached):
python resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py \
    --cache_dir /path/to/cache \
    --tickers AAPL --force
```

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

