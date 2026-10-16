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
`tavily_exclude_domains_limit` domains (default 150; `0` or `null` sends none):
the hosts matching `tavily_exclude_domains_priority` first (default: dataset
hubs, benchmark publishers, code hosts and archive mirrors), then file order.
The server logs one warning per start with the number of domains not sent.
Every `domain` and `url_substring` rule is still enforced locally on requested
and returned URLs, so the cap never widens what an agent can see.
