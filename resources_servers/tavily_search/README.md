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

## Sessions

The search tools keep no task state; the server keeps only each session's Tavily call metrics, which `/verify` returns in `metrics`.

- An Environment Server, such as `single_agent_turn_legacy` in `configs/tavily_search_judge_*.yaml`, seeds with a typed `ResourcesSeedSessionRequest` and closes with `/close_session`. The metrics are keyed by the caller's `resources_session_id`: a repeated seed returns the same session, `/verify` leaves its metrics for the close, and the close releases them even without the seed's cookie. A seed after its session closed gets 409, a seed whose `task_data` lacks `question` or `ground_truth` gets 422, and a tool call after the close gets 404 with `failure_kind: session_lost`.
- Any other caller, such as an Agent's `/run` or a native MCP client, needs no typed seed: a session's metrics start with its first tool call, and its `/verify` releases them.

The score reads only the response, so a `/verify` for a session the server no longer holds still scores it, with empty metrics. With `dump_session_id_to_metrics_on_exit: true`, released metrics stay in memory until shutdown, so the dump still lists every session.
