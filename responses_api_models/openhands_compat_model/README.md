# OpenHands compat model

`vllm_model` server for the OpenHands 0.62 agent that
`responses_api_agents/partner_if_harbor_agent` runs. It drops the `name` field from tool
messages before forwarding a `/v1/chat/completions` request, and it needs a Chat
Completions backend (`is_responses_native: false`): with a native Responses backend it
rejects those requests.

`configs/openhands_compat_model.yaml` configures it as the policy model with the same
settings as `vllm_model`; select it with `--model-type openhands_compat_model`.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
