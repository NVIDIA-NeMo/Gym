# Synthetic Tool-Use Domain Generation

This package owns the first stage of the [synthetic conversational tool-use pipeline](../synthetic_tool_use/README.md).
It sends an initial domain prompt and a follow-up request to an OpenAI-compatible model, validates candidate
structure, removes normalized duplicate names, and registers accepted domains in the shared run manifest.

## Ownership

- implementation: [`stage.py`](stage.py)
- follow-up prompt rendering: [`rendering.py`](rendering.py)
- prompt loading and hashing: [`prompts/domain_generation.txt`](prompts/domain_generation.txt) and
  [`assets.py`](assets.py)

Input is the domain stage block from `synthetic_tool_use/configs/{general,proactive}.yaml`. Output is
`domains.raw.jsonl`, `domains.accepted.jsonl`, and one `domain.json` per accepted source index. Run this stage with:

```bash
python resources_servers/synthetic_tool_use/scripts/generate_synthetic_tool_use_seeds.py \
  --config resources_servers/synthetic_tool_use/configs/general.yaml \
  --resume domains
```

The next stage is [policy/tool generation](../synthetic_tool_use_policy_tool_generation/README.md).
