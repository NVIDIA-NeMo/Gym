# DeepSeek model

An OpenAI-compatible model server for the DeepSeek Harness agent. It inherits
`openai_model` backend calls, retries, and buffered SSE handling, with a local
request schema that also accepts DeepSeek's `reasoning_effort: max` and validates
`thinking.type` as `enabled` or `disabled`. Streaming requests retain `thinking`.

Compose `configs/deepseek_model.yaml` with the agent and resources configuration.
Set `policy_base_url`, `policy_api_key`, and `policy_model_name` through Gym's
configuration system. See the [agent setup](../../responses_api_agents/deepseek_harness_agent/README.md).
