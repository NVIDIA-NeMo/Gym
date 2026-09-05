# Model backends

Model backends define how NeMo Gym reaches policy and judge models. They adapt hosted APIs, local serving,
and provider-specific behavior to Gym's model-server contract while keeping the environment and harness
independent of where inference runs.

Use `gym list models` to see every selectable backend and config flavor, then pass one with
`--model-type <name>`.

| Backend | Purpose |
| --- | --- |
| `openai_model` | Native OpenAI-compatible Responses and Chat Completions APIs |
| `azure_openai_model` | Azure OpenAI Chat Completions with Responses conversion |
| `inference_provider` | Hosted OpenAI-compatible Chat Completions providers |
| `litellm_model` | LiteLLM proxy integration |
| `vllm_model` | Existing vLLM endpoint, including training token metadata |
| `local_vllm_model` | Gym-managed local or multi-node vLLM deployment |
| `local_vllm_model_proxy` | Proxy to a Gym-managed local vLLM backend |
| `vllm_model_with_compaction` | vLLM Responses compaction behavior |
| `genrm_model` | Generative reward-model behavior on vLLM |
| `switchyard_model` | Switchyard routing integration |

The top-level directory was named `responses_api_models/` before MB-1553. Legacy Python imports and config
paths remain supported through a compatibility namespace and path alias. The YAML key
`responses_api_models` is the internal server-type wire key and has not changed.
