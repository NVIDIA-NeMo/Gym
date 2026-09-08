# vllm_model_with_compaction

Dedicated vLLM model server for context-compacted rollouts. It keeps the
standard Gym model interface (`/v1/responses` and `/v1/chat/completions`) while
accepting `required_prefix_token_ids` only on this server's `/v1/responses`.

Use `configs/vllm_model_for_compaction.yaml` together with
`simple_agent_with_compaction`. Other agents should continue to use
`responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`.

The dedicated `/tokenize` endpoint is only for estimating the prospective
context length before generation. Training token IDs must come from the same
vLLM request that sampled the response. If vLLM omits those requested inline
IDs, this adapter fails closed instead of reconstructing them with a later
`/tokenize` request.
