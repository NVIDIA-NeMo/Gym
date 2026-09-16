# Description
RL environment which allows access to web search (Search Provider: [You.com](https://you.com/platform))

A tool-for-tool counterpart to `resources_servers/tavily_search`: same three tools
(`web_search`, `find_in_page`, `scroll_page`), same judge, same output shape, same task
rows. Only the retrieval backend differs, so running both over one dataset gives a clean
provider A/B.

## Key config parameters

- You will need a You.com API key from <https://you.com/platform>.
- `search_mode` selects how much of each page comes back. This is the only knob that
  separates the arms, and it dominates both accuracy and token spend:

  | `search_mode` | What each result carries | Notes |
  | --- | --- | --- |
  | `snippets` (default) | keyword excerpts + description | cheapest; fewest tokens per call |
  | `highlights` | query-relevant passages extracted per page | middle ground on tokens |
  | `full_page` | the whole page as markdown | billed per page crawled; raise `max_result_chars` or the crawl is truncated away |
  | `eco` | titles, snippets, descriptions from `/v1/eco_search` | smallest corpus of any arm; no extraction, no news section |

  `eco` is a separate, lighter endpoint rather than an extraction level, so it takes
  only `query` and `count` — `crawl_timeout` and the API-side `exclude_domains` list do
  not apply. Client-side domain exclusion still runs, so an opt-out registry is honoured
  either way. Measured on one query: 2762 chars over 8 results, against 12823 chars over
  10 for `snippets`.

- `include_news` (default `false`) adds You.com's news section ahead of the web results.
  Turn it on for recency-sensitive benchmarks (LiveBench-style), where web-only retrieval
  systematically misses fresh sources. `/v1/search` returns both sections unconditionally,
  so this is an output-side filter — no request parameter opts into news.
- `crawl_timeout` (default `10`) bounds every `/v1/contents` fetch (`find_in_page`,
  `scroll_page`) in all modes, and `/v1/search` in `full_page` mode.
- `page_cache_max_entries` (default `512`) bounds the per-process page cache. Only
  successful fetches are cached, so one crawl timeout does not disable a URL for the run.
- `exclude_domains_file_path` is **optional**. Point it at a domain opt-out registry if
  your deployment has legal exclusions to honour; leave it unset otherwise. Entries are
  sent to the API (capped at its 500-domain limit) *and* enforced client-side, so
  registries larger than the cap are still fully applied.
- This environment uses LLM-as-judge to gauge the correctness of answers. Recommended
  judge model is Qwen3-235B-A22B-Instruct-2507.

Required to add to `env.yaml` / your config:

```yaml
search_judge_model_base_url: <YOUR_JUDGE_MODEL_URL>
search_judge_model_api_key: ""
search_judge_model_name: Qwen/Qwen3-235B-A22B-Instruct-2507

you_search_resources_server:
  resources_servers:
    you_search:
      ydc_api_key: <YOUR_KEY>
      search_mode: snippets
```

`ydc_api_key` also accepts a list, in which case calls round-robin across the keys.

## Commands to Run

```bash
gym env start \
    --resources-server you_search/you_search_judge_vllm_model \
    --model-type vllm_model
```

The `example` dataset ships in-repo. The benchmark sets are shared verbatim with
`tavily_search` — task rows carry only `question`, `ground_truth`, and the tool schema,
nothing provider-specific — so download them under that dataset name:

```bash
gym dataset download --storage gitlab \
    --name tavily_search \
    --revision 0.0.1 \
    --artifact browsecomp_test_set.jsonl \
    --output resources_servers/you_search/benchmark/browsecomp/browsecomp_test_set.jsonl

gym dataset download --storage gitlab \
    --name tavily_search \
    --revision 0.0.1 \
    --artifact simple_qa_test_set.jsonl \
    --output resources_servers/you_search/benchmark/simpleqa/simpleqa_test_set.jsonl
```

## Implementation notes

- **No SDK, no httpx.** You.com is a plain REST API, so every call goes straight through
  NeMo Gym's global aiohttp client. There is no vendored httpx transport to adapt — the
  reason `tavily_search` needs `TavilySearchAIOHTTPClient` does not apply here.
- **Two endpoints back the three tools.** `web_search` calls `POST /v1/search`;
  `find_in_page` and `scroll_page` both call `POST /v1/contents` and share a bounded
  per-process page cache, so scrolling a long page costs one crawl regardless of how many
  windows the agent reads.
- **`find_in_page` selects its window client-side.** `/v1/contents` takes no query —
  passing one returns an identical payload — so the server scores candidate windows by
  query-term coverage and returns the best-matching region, snapped to a line boundary,
  rather than the head of every page.
- **Result bodies are deduplicated.** You.com's `description` is frequently a truncated
  copy of `snippets`; emitting both cost ~16% more tokens per call for no added
  information. Whichever is subsumed by the other is dropped.
- **Extraction degrades gracefully.** A crawl that times out or a page with no
  query-relevant span still yields a usable result — the snippet and description are used
  as fallback rather than returning a blank entry.

### Performance Metrics

Not yet baselined. `verified: false` until a full run lands.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
