# Synthetic Tool-Use Scenario Generation

This package owns the third stage of the [synthetic conversational tool-use pipeline](../synthetic_tool_use/README.md).
It consumes each accepted `policy.md`, renders system and user messages, assigns inside/outside-policy scope, validates
the structured response, removes duplicate scenarios, and writes scenario JSONL artifacts.

## Ownership

- implementation: [`stage.py`](stage.py)
- scenario models and schema loading: [`schema.py`](schema.py)
- system prompt, user prompt, and response schema: [`prompts`](prompts)
- prompt loading and hashing: [`assets.py`](assets.py)

The checked-in response schema is regenerated in tests from the local Pydantic models.

Run this stage after policy/tool artifacts exist in the configured output directory:

```bash
python resources_servers/synthetic_tool_use/scripts/generate_synthetic_tool_use_seeds.py \
  --config resources_servers/synthetic_tool_use/configs/proactive.yaml \
  --resume scenarios
```

The resulting raw bundle is materialized for the
[runtime simulation server](../synthetic_tool_use_simulation/README.md) by the pipeline CLI.
