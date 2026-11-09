# Description
RL environment which allows access to web search (Search Provider: Tavily)

## Prerequisites and setup

Follow [Tavily search access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/tavily-search)
for the required exclusion policy, service credentials, and datasets.

Use your own authorized service credentials and exclusion policy. The setup
guide includes a small run with the committed example dataset and explains how
to configure your own training or evaluation data.


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

The server supports regular Gym agents and native MCP clients, including OpenCode
and Pi. See [tool-service behavior and compatibility](../../fern/versions/latest/pages/infrastructure/tavily-search.mdx#tool-service-behavior-and-compatibility)
for the exclusion policy, key pools, bounded retries, and cache limits. These
policies apply to all Tavily configurations, including removal of the aggregate
`Search Answer` section. Tool transcripts and Gym observability record tool use.

Tavily's `/search` accepts at most 150 `exclude_domains` (HTTP 400 beyond). A
longer policy is pre-filtered provider-side with a deterministic subset of
`tavily_exclude_domains_limit` domains (default 150, also the largest value the
config accepts; `0` or `null` sends none): the hosts matching
`tavily_exclude_domains_priority` first (default: dataset hubs, benchmark
publishers, code hosts and archive mirrors; patterns are matched
case-insensitively against the normalised hostnames), then file order. Only
the set of domains is preserved for a policy within the limit: priority hosts
move to the front and a duplicate rule is sent once. The server logs one
warning per start with the number of domains not sent. Every `domain` and
`url_substring` rule is still enforced locally on requested and returned URLs,
so the cap never widens what an agent can see.

## Exa provider

Set `search_provider: exa` and `exa_api_key` to serve the same three tools from
[Exa](https://exa.ai) instead of Tavily; `tavily_api_key` is then not needed.
`web_search` returns Exa's query-matched highlights as each result's summary,
`find_in_page` returns the page passages that match its query (the page text
when Exa returns none), and `scroll_page` reads the page text. Both answer a page
that could not be fetched with `No content found.` The exclusion
policy, the provider-side domain subset (sent as Exa's `excludeDomains`), output
limits, page cache, key pool and bounded retries work as they do for Tavily; a
401 (invalid key) or 402 (credits or budget exhausted) tries the next Exa key.
`search_depth` applies to Tavily only, and `tavily_exclude_domains_limit` also
caps Exa's `excludeDomains`. With caching on, Exa fetches at most
`max_cached_page_chars` of a page's text when that is set. `exa_base_url` defaults to `https://api.exa.ai`.
Tavily stays the default. The DeepSearchQA and WideSearch session configs run
this as their `exa_search` tool server. With either provider, closing a session drops
its call metrics unless `dump_session_id_to_metrics_on_exit` is set.
