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

The server supports regular Gym agents and native MCP clients, including OpenCode
and Pi. See [tool-service behavior and compatibility](../../fern/versions/latest/pages/infrastructure/tavily-search.mdx#tool-service-behavior-and-compatibility)
for the exclusion policy, key pools, bounded retries, and cache limits. These
policies apply to all Tavily configurations, including removal of the aggregate
`Search Answer` section. Tool transcripts and Gym observability record tool use.
