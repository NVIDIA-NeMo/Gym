# Description

This is an example environment with simple tool calls. The example data can be found in `example_single_tool_call/data/example.jsonl`.

## Configs

- `configs/example_single_tool_call.yaml` runs the example through Simple Agent's `/run`.
- `configs/example_single_tool_call_single_agent_turn.yaml` runs it through the single-agent-turn Environment Server, which seeds the Resources Server and Simple Agent sessions, grants the agent direct HTTP access to the tools, and verifies the response.
- `configs/example_single_tool_call_tool_sidecar.yaml` runs `get_weather` in a sidecar of each attempt's sandbox group on OpenSandbox, with Simple Agent's loop in the group's agent container. The Resources Server starts its own class there as the tool host, and no tool call reaches the Resources host. See "Tools in a sidecar" in the single-agent-turn Environment Server's README.

## Tutorial

For a hands-on walkthrough of building a single-step environment from scratch, see the [Single-Step Environment](https://docs.nvidia.com/nemo/gym/main/environment-tutorials/single-step-environment) tutorial.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
