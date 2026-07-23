# Synthetic Conversational Tool-Use Pipeline

This package connects the resource servers that generate and consume synthetic conversational tool-use data.

## Components

| Resource server | Path | Responsibility |
| --- | --- | --- |
| Pipeline | `synthetic_tool_use` | Orchestrate generation, validate artifacts, and materialize Gym rows |
| Domain generation | `synthetic_tool_use_domain_generation` | Generate and deduplicate domain candidates |
| Policy/tool generation | `synthetic_tool_use_policy_tool_generation` | Generate, refine, validate, and judge policies and tools |
| Scenario generation | `synthetic_tool_use_scenario_generation` | Generate and validate inside/outside-policy customer scenarios |
| Rollout simulation | `synthetic_tool_use_simulation` | Simulate users and tools and verify completed trajectories |

Each component is a Gym `app.py` server. Generation models are configured as `responses_api_models` and referenced
with `ModelServerRef`; resource servers communicate through `ServerClient`. The generation servers do not construct
provider clients or read endpoint credentials.

## Artifact Flow

```text
domain generation
  -> domains.accepted.jsonl and domains/<index>/domain.json
policy/tool generation
  -> domains/<index>/policy.md and tools.jsonl
scenario generation
  -> domains/<index>/scenarios/<model>/scenarios_*.jsonl
pipeline materialization
  -> Gym JSONL rows
synthetic_tool_use_agent + synthetic_tool_use_simulation
  -> scored rollout trajectories
```

All generation stages share a manifest-backed output directory. Completed artifacts are checked before a resumed
stage is skipped, and every provider response or validation failure is retained under the run's `attempts` paths.
`seed_generation.output_dir` must be an absolute path visible to every server process.

## Configuration

The two complete Gym server graphs are:

- [`general.yaml`](configs/general.yaml)
- [`proactive.yaml`](configs/proactive.yaml)

Set `NVI_KEY_PROD`, or set the role-specific `DOMAIN_MODEL_API_KEY`, `POLICY_MODEL_API_KEY`, `JUDGE_MODEL_API_KEY`,
and `SCENARIO_MODEL_API_KEY` variables. The model servers default to `https://inference-api.nvidia.com/v1`; the
corresponding `*_MODEL_BASE_URL` variables override those endpoints.

Start the proactive graph from the repository root:

```bash
gym env start "+config_paths=[resources_servers/synthetic_tool_use/configs/proactive.yaml]"
```

Call `synthetic_tool_use_pipeline` through `ServerClient`:

```python
import asyncio

from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


async def main():
    client = ServerClient.load_from_global_config()
    response = await client.post(
        server_name="synthetic_tool_use_pipeline",
        url_path="/generate",
        json={"stages": ["domains", "policy_tools", "scenarios"], "resume": True},
    )
    await raise_for_status(response)
    print(await get_response_json(response))


asyncio.run(main())
```

The pipeline also exposes `POST /validate` and `POST /materialize`. Each stage server exposes its own typed
`POST /generate` route for partitioned or stage-only work; `domain_start` is inclusive and `domain_end` is exclusive.

## Documentation

- [Generation workflow](docs/generation.md)
- [Rollout behavior](docs/rollout.md)
