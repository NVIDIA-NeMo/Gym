# Description

`claude_model` is a native Anthropic Messages API model server behind NeMo Gym's `/v1/responses` interface. It translates NeMo Gym Responses API requests to Anthropic `/v1/messages` payloads and maps Anthropic responses back to NeMo Gym Responses API objects.

It supports text messages, system/developer prompt extraction, function tools, previous tool calls/results, thinking blocks, usage mapping, and optional request concurrency limiting. It uses `nemo_gym.server_utils.request()` for raw aiohttp transport instead of the Anthropic Python SDK.

# Usage

Start with a resources server config and the Claude model config:

```bash
ng_run "+config_paths=[resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,responses_api_models/claude_model/configs/claude_model.yaml]" \
  +policy_base_url=https://inference-api.nvidia.com/v1 \
  '+policy_api_key=${anthropic_api_key}' \
  +policy_model_name=us/aws/anthropic/eccn-claude-opus-4-8
```

`anthropic_base_url` accepts either host-only or `/v1` style URLs. Both `https://api.anthropic.com` and `https://api.anthropic.com/v1` resolve to `/v1/messages`.

Minimal direct smoke test once the model server is running:

```bash
curl -s <POLICY_MODEL_URL>/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Say hello in one short sentence.",
    "max_output_tokens": 64
  }' | python -m json.tool
```

Collect one rollout through the simple agent:

```bash
mkdir -p results

ng_collect_rollouts \
  +agent_name=example_single_tool_call_simple_agent \
  +input_jsonl_fpath=resources_servers/example_single_tool_call/data/example.jsonl \
  +output_jsonl_fpath=results/claude_example_single_tool_call_rollouts.jsonl \
  +limit=1 \
  +num_repeats=1
```

# Notes

Provider-specific Anthropic fields that are not in NeMo Gym's shared Responses schema can be passed through `extra_body`. Some options are mutually constrained by Anthropic; for example, adaptive/extended thinking is not compatible with `top_k`, and `top_p` must stay in the supported thinking range.

# Licensing information

Code: Apache 2.0

Data: N/A

Dependencies:
- nemo_gym: Apache 2.0
