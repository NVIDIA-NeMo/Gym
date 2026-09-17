# DeepSearchQA Resources Server

Grades DeepSearchQA single-answer and set-answer responses with the official
LLM judge prompt. The reward is answer-set F1: missing answers reduce recall,
and unsupported extra answers reduce precision.

The reusable sandbox and Exa integration live in
[`harness_exa_search`](../../responses_api_agents/harness_exa_search/README.md).
