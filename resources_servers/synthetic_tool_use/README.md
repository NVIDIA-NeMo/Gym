# Synthetic Conversational Tool-Use Pipeline

This directory is the index and shared integration layer for the synthetic conversational tool-use pipeline. The
stage implementations are sibling packages with the same `synthetic_tool_use_` prefix, so each stage has an explicit
owner while the complete workflow remains easy to find.

## Components

| Component | Path | Owns | Produces |
| --- | --- | --- | --- |
| Domain generation | [`synthetic_tool_use_domain_generation`](../synthetic_tool_use_domain_generation/README.md) | Domain prompt and candidate generation | Accepted domain JSONL and per-domain `domain.json` |
| Policy/tool generation | [`synthetic_tool_use_policy_tool_generation`](../synthetic_tool_use_policy_tool_generation/README.md) | General/proactive profiles, policy/tool prompts, references, refinement, and judging | `policy.md`, `tools.jsonl`, and quality reports |
| Scenario generation | [`synthetic_tool_use_scenario_generation`](../synthetic_tool_use_scenario_generation/README.md) | Scenario prompts, response schema, scope assignment, and scenario validation | `scenarios/**/*.jsonl` |
| Dataset materialization and rollout simulation | [`synthetic_tool_use_simulation`](../synthetic_tool_use_simulation/README.md) | Gym row materialization, session state, user/tool simulation, and verification | Gym input rows and scored rollouts |
| Policy loop | [`synthetic_tool_use_agent`](../../responses_api_agents/synthetic_tool_use_agent/README.md) | Policy-model turns and tool-call orchestration | Responses API trajectory |

Only `synthetic_tool_use_simulation` is a long-running Gym resource server. The three generation packages are
independently owned offline stages that call OpenAI-compatible model endpoints and exchange durable artifacts. This
keeps generation dependencies and prompts out of the runtime server without introducing HTTP services around a
filesystem batch pipeline.

## Artifact Flow

```text
domain prompt
    |
    v
domains.raw.jsonl + domains.accepted.jsonl
    |
    v
<domain>/domain.json
    |
    v
<domain>/policy.md + <domain>/tools.jsonl
    |
    v
<domain>/scenarios/<model>/scenarios_*.jsonl
    |
    v
build_synthetic_tool_use_dataset.py
    |
    v
Gym JSONL row
    |
    +--> synthetic_tool_use_agent
    |
    +--> synthetic_tool_use_simulation
```

The shared code under [`common`](common) defines only the cross-stage artifact models, manifest store, provider client,
parsing, and deterministic validation. Stage-specific prompts and renderers do not live in `common` and are not
imported from another stage.

## Prompt Assets

Each generation component owns the prompts and references it uses:

- domain assets: `synthetic_tool_use_domain_generation/prompts`
- policy/tool assets: `synthetic_tool_use_policy_tool_generation/prompts` and `references`
- scenario assets: `synthetic_tool_use_scenario_generation/prompts`

Asset hashes are recorded in each run manifest so a run identifies the exact prompt, schema, and reference set it
used. [`test_seed_generation.py`](tests/test_seed_generation.py) checks asset loading, schema generation, prompt
rendering, and ownership boundaries between stages.

## Running

There are exactly two checked-in generation configs:

- [`general.yaml`](configs/general.yaml)
- [`proactive.yaml`](configs/proactive.yaml)

Run the complete resumable workflow:

```bash
python resources_servers/synthetic_tool_use/scripts/generate_synthetic_tool_use_seeds.py \
  --config resources_servers/synthetic_tool_use/configs/proactive.yaml \
  --resume \
  all \
  --dataset-name synthetic_tool_use_proactive \
  --output-path /tmp/synthetic_tool_use_proactive.jsonl
```

Run one stage against the same manifest by replacing `all` with `domains`, `policies`, or `scenarios`. The `validate`
and `materialize` commands do not call generation models. `--domain-start` is inclusive and `--domain-end` is
exclusive for partitioned policy/tool and scenario workers.

## Design Documents

- [Generation workflow](docs/generation.md)
- [Rollout behavior](docs/rollout.md)
