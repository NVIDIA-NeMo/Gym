# BigFinanceBench resource server

Tools-only wrapper around the Apache-2.0 `big-finance-harness` tools. The
resource-server requirements pin the tools-only fork
`wowowoxuan1994/big-finance-benchmark` to an immutable commit. Install its
isolated environment with the repository's normal resource-server setup, or
directly:

```bash
cd resources_servers/big_finance
uv venv --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The fork keeps LiteLLM, Google provider, and CLI dependencies in its optional
`eval` extra. Gym installs only the base tool dependencies, avoiding the
upstream harness's model-provider dependency conflicts.

Routes are `/web_search`, `/edgar_search`, `/fetch_url`, `/python_exec`, and
`/final_answer`; each accepts the upstream JSON arguments and returns the
upstream tool string verbatim. `/seed_session`, `/verify`,
`/aggregate_metrics`, and `/reverify_mode` follow the standard Gym protocol.
Tool replies carry `x-nemo-gym-tool-error: false` for a successful return and
`true` for an exception or an unconfigured tool. Failure bodies contain the error
message. The shared agent uses this status and preserves it in rollout metadata
for reverification, so successful JSON content with an `error` field remains a
successful observation.

`configs/big_finance.yaml` also ships a `big_finance` agent instance backed by
the shared finance loop. Its five-row `data/example.jsonl` fixture is converted
from the first five rows of the pinned public BigFinanceBench subset and retains
the evaluation-only, do-not-train, canary, prompt, license, and provenance
fields. The required five-row `data/example_rollouts.jsonl` and matching
`data/example_metrics.json` are committed alongside it. Validate all three with
`gym env test --resources-server big_finance`.

`big_finance_web_search: false` turns web search off: no search backend is built
and `/web_search` returns the "not configured" error. Pair it with rows prepared
by `benchmarks/big_finance/prepare.py --no-web-search`, which removes the tool
and its mentions from the prompt and the `fetch_url` description (see
`benchmarks/big_finance/README.md`).

## Offline SEC corpus

Set `big_finance_sec_corpus_metadata_path` to the `sec_metadata.parquet` of a
downloaded SEC corpus (the format `finance_task_tree_sdg` reads; see its README)
to serve `edgar_search` and `fetch_url` from it:

- Neither tool sends any request to SEC. `edgar_search` lists the company's
  filings in the corpus newest first, with the upstream fields only. `fetch_url`
  returns SEC documents from the corpus through the upstream parsing, retrieval,
  and truncation; an SEC URL outside the corpus is an error observation, never a
  live request. Non-SEC URLs are still fetched live.
- A row may carry `as_of_date` (ISO date). Filings made after it are left out of
  `edgar_search` and refused by `fetch_url`. Rows without it are not clamped.
- Tool names, schemas, and descriptions, the prompt, and grading are unchanged.
  `sec_edgar_user_agent` is still required.

For OpenAI-backed rollouts, also pass
`--config resources_servers/big_finance/configs/openai_model.yaml`. This applies
BigFinance's retry and timeout settings to its own `big_finance_policy_model`
instance without changing the shared `policy_model`. Omit this overlay when
selecting another model type.

The independent judge defaults to `gpt-5-mini` for convenience. Judge model
choice affects scores. To compare with published results, explicitly override
`big_finance_judge_model_name` and match their judge model snapshot and grading
settings, including the reward mode. Configure `big_finance_judge_base_url` and
`big_finance_judge_api_key` as needed for that endpoint.

## Optional SEC cache

To reuse SEC downloads across rollouts and server restarts, add these settings
to the Gym configuration used when starting the environment:

```yaml
big_finance_use_cache: true
big_finance_cache_dir: /absolute/path/to/big-finance-sec-cache
```

Caching defaults to `false`. If enabled without an explicit directory, it uses
`~/.cache/nemo_gym/big_finance/`. These BigFinance-specific settings do not
change the cache settings of other benchmarks in a mixed run.

The cache stores successful SEC HTTP responses: ticker mappings, company
submissions metadata, and SEC documents fetched through `fetch_url`. It stores
the downloaded bytes and text encoding, not Vals-parsed text or a final tool
answer. The upstream tools still apply `form_type`, `limit`, HTML/PDF parsing,
`query` retrieval, and `max_tokens` on every call. Other websites and
`web_search`, Python execution, and judging are not cached.

Entries are snapshots with no automatic expiry. Use a new cache directory to
refresh them, or set `big_finance_use_cache: false` to bypass disk reads and
writes. The upstream ticker mapping still has its existing process-local memory
cache; restart the resources server to refresh that mapping as well. Concurrent
requests for the same uncached URL share one download within a server process.
Unreadable cache entries fall back to a download; HTTP failures are not stored.
The cache does not pre-download all filing bodies.

This option preserves the upstream search coverage: `edgar_search` still reads
only `filings.recent`, with a default limit of 20 matching records. It does not
load historical `filings.files` or add date filtering. Use a separate directory
from Vals caches, whose metadata and parsed-body formats are different.

See `benchmarks/big_finance/README.md` for credentials, scoring, safety,
preparation, and standalone parity notes.

The tool package is Apache-2.0 licensed. The public five-row fixture is derived
from the separately licensed CC BY 4.0 BigFinanceBench public subset.
