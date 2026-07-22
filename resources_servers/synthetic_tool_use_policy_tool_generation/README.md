# Synthetic Tool-Use Policy And Tool Generation

This package owns the second stage of the [synthetic conversational tool-use pipeline](../synthetic_tool_use/README.md).
For each accepted domain it generates and optionally refines `policy.md` and `tools.jsonl`, performs deterministic
schema and leakage validation, and optionally runs cohesion and golden-reference judgments.

## Ownership

- implementation: [`stage.py`](stage.py)
- prompt input formatting: [`rendering.py`](rendering.py)
- profile loading: [`profiles.py`](profiles.py)
- profiles: [`general.yaml`](profiles/general.yaml) and [`proactive.yaml`](profiles/proactive.yaml)
- prompts: [`prompts`](prompts)
- 16 golden policy/tool reference files: [`references/golden_policies`](references/golden_policies)

The profile names describe behavior, not source domains. `general` is the baseline flow. `proactive` adds the source
policy instruction requiring proactive confirmation. Shared judge and refinement prompts are stored once.

Run this stage after domains exist in the configured output directory:

```bash
python resources_servers/synthetic_tool_use/scripts/generate_synthetic_tool_use_seeds.py \
  --config resources_servers/synthetic_tool_use/configs/proactive.yaml \
  --resume policies
```

The next stage is [scenario generation](../synthetic_tool_use_scenario_generation/README.md).
