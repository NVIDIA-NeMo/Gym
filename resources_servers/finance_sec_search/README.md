# Finance SEC Search Resource Server

Financial information retrieval using SEC EDGAR filings with optional web search via Tavily.

**Only companies listed in the [SEC company tickers file](https://www.sec.gov/files/company_tickers.json) are supported.** Questions about companies not in this list will fail at the ticker lookup step.

## Tools

| Tool | Description |
|------|-------------|
| `sec_filing_search` | Search SEC EDGAR for filing metadata by stock ticker symbol |
| `parse_html_page` | Fetch and parse any HTML page (SEC URLs use disk cache), store under a key |
| `retrieve_information` | Query stored documents via LLM prompt with `{{key}}` placeholders |
| `submit_final_result` | Submit the final answer (keeps model in tool-calling mode until ready) |
| `web_search` | Internet search via Tavily API (optional — requires `tavily_api_key`) |

If `tavily_api_key` is not configured, `web_search` returns an error directing the model to use SEC tools instead.

## Setup

### env.yaml

Create `env.yaml` in the Gym root:

```yaml
policy_base_url: http://localhost:5000/v1
policy_api_key: empty
policy_model_name: /hf_models/Qwen3-30B-A3B

search_judge_model_base_url: http://localhost:5000/v1
search_judge_model_api_key: ""
search_judge_model_name: /hf_models/Qwen3-30B-A3B

finance_sec_search_resources_server:
  resources_servers:
    finance_sec_search:
      cache_dir: /workspace/cache/finance_sec_search
      # tavily_api_key: <your-tavily-key>
```

## Cache Management

The resource server caches SEC data locally to avoid redundant API calls and to
enable offline operation after the first fetch.

### What is cached

| Directory | Contents |
|-----------|----------|
| `filings_metadata/{CIK}.json` | Filing metadata (accession numbers, dates, forms) per company |
| `filings/{hash}.txt` | Parsed filing content (HTML to text) |
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
companies **before** starting rollouts. This avoids SEC.gov API calls during
GPU-intensive rollout collection and eliminates race conditions when multiple
seeds share the same cache.

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

If the cache is empty, the resource server lazily fetches and caches metadata
from SEC.gov on first access. This works but is slower on the first run and
requires SEC.gov connectivity during rollout.

### Shared cache

Multiple seeds or runs can share the same `cache_dir`. With prefetch, all GPU
jobs are read-only (no race conditions). Without prefetch, concurrent writes are
benign because all writers produce identical data for the same company.

## End-to-End Rollout

### 1. Prepare the dataset

The input is a JSONL file with question/answer pairs. An example is provided at
`resources_servers/finance_sec_search/data/example_questions.jsonl`:

```json
{"question": "What is the number of shares of common stock outstanding as of November 14, 2025 for Nvidia?", "expected_answer": "24.3 billion"}
{"question": "As of September 24, 2022 how many full-time equivalent employees did Apple have?", "expected_answer": "164,000"}
```

Convert the questions into the Gym input format (adds tool definitions, system prompt, etc.)

Use the `--include-web-search` flag to include the optional `web_search` tool:

```bash
python resources_servers/finance_sec_search/scripts/convert_questions.py \
  --input resources_servers/finance_sec_search/data/example_questions.jsonl \
  --output resources_servers/finance_sec_search/data/example.jsonl \
  --include-web-search
```

#### SecQue benchmark

To prepare the [SecQue](https://huggingface.co/datasets/nogabenyoash/SecQue) dataset (filters to questions mentioning known companies and converts to Gym format):

```bash
cd resources_servers/finance_sec_search
python scripts/prepare_secque_questions.py
```

This writes `data/secque_questions.jsonl`. Use it as the `input_jsonl_fpath` in step 4 below.

**Note that this dataset is not used for training anywhere and is only used for eval/benchmark purposes.**


### 2. Start the vLLM server

Launch a vLLM-compatible model server (e.g. Qwen3-30B-A3B) so the policy and judge endpoints are available.

### 3. Start the Gym servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/finance_sec_search/configs/finance_sec_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

### 4. Collect rollouts

```bash
ng_collect_rollouts \
  +agent_name=finance_agent \
  +input_jsonl_fpath=resources_servers/finance_sec_search/data/example.jsonl \
  +output_jsonl_fpath=results/finance_sec_search_rollouts.jsonl
```

Add `+limit=1` for a quick single-question test:

```bash
ng_collect_rollouts \
  +agent_name=finance_agent \
  +input_jsonl_fpath=resources_servers/finance_sec_search/data/example.jsonl \
  +output_jsonl_fpath=results/finance_sec_search_rollouts.jsonl \
  +limit=1
```

### Run tests

```bash
ng_test +entrypoint=resources_servers/finance_sec_search
```

## Verification

Uses LLM-as-judge with a financial grading rubric (0/1/2 scale). Only fully correct answers ([[2]]) receive reward 1.0. The judge prompt and rubric are defined in /prompt_templates.

