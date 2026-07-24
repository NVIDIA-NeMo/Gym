# Conversational Tool-Use Generation

Conversational tool-use data is produced by three independent Gym resource servers. Each server owns one generation
stage and communicates with configured model servers through Gym's `ServerClient`.

| Stage | Runtime name | Implementation | Model dependencies |
| --- | --- | --- | --- |
| Domain generation | `conversational_tool_use_domain_generation` | `domain/app.py` | Domain model |
| Policy/tool generation | `conversational_tool_use_policy_tool_generation` | `policy_tools/app.py` | Policy/tool and judge models |
| Scenario generation | `conversational_tool_use_scenario_generation` | `scenarios/app.py` | Scenario model |

There is no parent generation server. Call the stages in order and pass the same absolute `seed_generation.output_dir`
to every server:

```text
domain generation
  -> domains.accepted.jsonl and domains/<index>/domain.json
policy/tool generation
  -> domains/<index>/policy.md and tools.jsonl
scenario generation
  -> domains/<index>/scenarios/<model>/scenarios_*.jsonl
dataset materialization
  -> Gym JSONL rows
```

All stages share `run_manifest.json`. With `resume: true`, a completed stage is skipped only when its expected
artifacts still load and validate. Policy/tool and scenario requests support an inclusive `domain_start` and exclusive
`domain_end`.

## Configuration

The complete server/model graphs are:

- [`general.yaml`](configs/general.yaml)
- [`proactive.yaml`](configs/proactive.yaml)

Set `NVI_KEY_PROD`, or provide the role-specific `DOMAIN_MODEL_API_KEY`, `POLICY_MODEL_API_KEY`,
`JUDGE_MODEL_API_KEY`, and `SCENARIO_MODEL_API_KEY` variables. The corresponding `*_MODEL_BASE_URL` variables
override the default `https://inference-api.nvidia.com/v1` endpoint.

Start a graph from the repository root:

```bash
gym env start \
  "+config_paths=[resources_servers/conversational_tool_use_simulation/generation/configs/general.yaml]"
```

Invoke each stage explicitly:

```python
import asyncio

from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


async def generate() -> None:
    client = ServerClient.load_from_global_config()
    request = {"resume": True, "domain_start": None, "domain_end": None}
    for server_name in (
        "conversational_tool_use_domain_generation",
        "conversational_tool_use_policy_tool_generation",
        "conversational_tool_use_scenario_generation",
    ):
        response = await client.post(server_name=server_name, url_path="/generate", json=request)
        await raise_for_status(response)
        print(server_name, await get_response_json(response))


asyncio.run(generate())
```

## Materialization

Materialization is an explicit offline step rather than a resource-server route. The builder validates every selected
domain bundle before writing Gym rows:

```bash
uv run python \
  resources_servers/conversational_tool_use_simulation/scripts/build_conversational_tool_use_dataset.py \
  --source-dir /tmp/conversational-tool-use-general/domains \
  --source-name synthetic_tool_use_general \
  --dataset-name synthetic_tool_use_general \
  --output-path /tmp/synthetic_tool_use_general.jsonl \
  --max-rows 0 \
  --max-rows-per-domain 0 \
  --scan-domains-per-source 0
```

The checked-in prompts, parsing, retries, quality gates, and artifact ownership are documented with each stage:

- [Domain generation](domain/README.md)
- [Policy/tool generation](policy_tools/README.md)
- [Scenario generation](scenarios/README.md)
