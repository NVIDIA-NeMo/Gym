# vllm_model_with_compaction

Dedicated vLLM model server for context-compacted rollouts. It keeps the
standard Gym model interface (`/v1/responses` and `/v1/chat/completions`) while
accepting `required_prefix_token_ids` only on this server's `/v1/responses`.

Use `configs/vllm_model_for_compaction.yaml` together with
`simple_agent_with_compaction`. Other agents should continue to use
`responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`.
