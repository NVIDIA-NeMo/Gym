# DeepSearchQA Resources Server

Grades DeepSearchQA single-answer and set-answer responses with the official
LLM judge prompt. The reward is answer-set F1: missing answers reduce recall,
and unsupported extra answers reduce precision.

Session harnesses get Exa search from a
[`tavily_search`](../tavily_search/README.md) tool server with
`search_provider: exa`; the remaining legacy configs use [`harness_exa_search`](../../responses_api_agents/harness_exa_search/README.md).
See [the benchmark](../../benchmarks/deepsearchqa/README.md) for which is which.
