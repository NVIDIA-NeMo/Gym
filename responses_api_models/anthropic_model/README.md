# Description

`anthropic_model` is a native Anthropic Messages API model server behind NeMo Gym's `/v1/responses` interface. It translates NeMo Gym Responses API requests to Anthropic `/v1/messages` payloads and maps Anthropic responses back to NeMo Gym Responses API objects.

It supports text messages, base64 image inputs, system/developer prompt extraction, function tools, previous tool calls/results, thinking blocks, usage mapping, and optional request concurrency limiting. It uses `nemo_gym.server_utils.request()` for raw aiohttp transport instead of the Anthropic Python SDK.

# Usage

Start with a resources server config and the Anthropic model config:

```bash
ng_run "+config_paths=[resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,responses_api_models/anthropic_model/configs/anthropic_model.yaml]" \
  +policy_base_url="$ANTHROPIC_BASE_URL" \
  +policy_api_key="$ANTHROPIC_API_KEY" \
  +policy_model_name="$ANTHROPIC_MODEL_NAME"
```

`anthropic_base_url` accepts either host-only or `/v1` style URLs. Both `https://api.anthropic.com` and `https://api.anthropic.com/v1` resolve to `/v1/messages`.

This example uses the simple agent harness because it exercises `anthropic_model` as the policy model server through NeMo Gym's `/v1/responses` interface. `claude_code_agent` is a separate agent harness that invokes Claude Code/Anthropic directly, so it is useful for testing Claude Code workflows but does not validate this model server.

For modern Claude models, prefer adaptive thinking with the typed `thinking` config:

```yaml
thinking:
  type: adaptive
```

`thinking_budget_tokens` remains available for older models that require manual `thinking: {type: enabled, budget_tokens: ...}`.

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

Provider-specific Anthropic fields that are not modeled as typed config can be passed through `extra_body`. Some options are model-specific: Claude Opus 4.7 and 4.8 reject configurable sampling parameters (`temperature`, `top_p`, `top_k`), so omit them and use prompting or adaptive thinking/effort controls instead.

Responses `input_image` parts are supported when `image_url` is a base64 data URL. Supported media types are `image/jpeg`, `image/png`, `image/gif`, and `image/webp`; remote image URLs are rejected with a 400.

Anthropic `stop_reason` values are mapped to Responses-compatible `incomplete_details` when possible. `max_tokens` and `model_context_window_exceeded` map to `max_output_tokens`; `refusal` maps to `content_filter`. Other stop reasons such as `end_turn`, `tool_use`, and `pause_turn` remain complete responses.

# Licensing information

Code: Apache 2.0

Data: N/A

Dependencies:
- nemo_gym: Apache 2.0
