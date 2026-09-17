# Description
RL enviroment which allows access to web search (Search Provider: Tavily)

## Prerequisites and setup

Follow [Tavily search access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/tavily-search)
for the required exclusion policy, service credentials, and datasets.

NVIDIA users can start with @rgala. External users can contact the maintainers
to discuss policy and dataset access options before running this recipe.


### Performance Metrics
100*16 samples:
- Acc: 0.3212
- Time in `gym eval run`: 44 mins


# Licensing information
Code: ?
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0

## Search policy and runtime limits

The same server supports regular Gym agents and native MCP clients such as
OpenCode. Set `expose_tools_over_mcp: true` to expose `web_search`, `find_in_page`,
and `scroll_page`. MCP sessions expose only those three tools; grading remains
on the benchmark’s resource server when this is used as a separate tool service.

`tavily_api_key` accepts a key, a list of keys, or a comma-separated key pool.
Calls rotate through the keys, and retries rotate to the next key. Both rate
limits and transient HTTP failures count toward `max_http_attempts` (default 3).
`http_timeout_s` caps each attempt (default 60 seconds; shorter SDK timeouts apply).
Provider error bodies are not forwarded to models or printed.

Set `exclude_domains_file_path` to your deployment’s exclusion policy. Keep
internal policies in private configuration repositories.

Exclusion JSON uses `notices[].properties[]` entries of type `domain` or
`url_substring`. Domain matches include subdomains. Requested and returned URLs
are checked, including reported extraction redirects, trailing-dot hosts and
percent encoding. Author/publisher metadata is descriptive, not enforced.
Search requests disable aggregate answers, and any returned aggregate answer is
discarded because its sources cannot be checked against the policy. A finite
list cannot prevent all online answer leakage.

Search defaults remain 10 results, advanced search, and 2,000-character snippets.
`max_results`, `search_depth`, and `max_result_chars` make these configurable.
The URL-keyed page cache evicts the least recently used page after `max_cached_pages`
(default 128); zero disables caching. `max_cached_page_chars` and `max_scroll_words`
are optional caps. Output formatting and tool schemas remain unchanged.
Tool transcripts and Gym observability provide the audit trail; there is no
separate browser log format or per-rollout result-index state.
