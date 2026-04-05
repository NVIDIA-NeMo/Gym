# Megatron Inference Model

Model backend that talks to a Megatron-Core inference server via the OpenAI-compatible
chat completions API. Extends `responses_api_models/vllm_model` to capture the extra
per-token metadata that Megatron emits for training (policy epoch, KV-cache epoch, and
the number of evictions per token).

Use this backend in place of `vllm_model` when you are collecting rollouts from a
Megatron inference server and need the additional token-level training information on
each response.

## Licensing information

- **Code**: Apache 2.0
- **Data**: N/A

## Dependencies

- `nemo_gym`: Apache 2.0
- `responses_api_models/vllm_model`: Apache 2.0
