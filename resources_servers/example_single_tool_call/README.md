# Description

This is an example environment with simple tool calls. The example data can be found in `example_single_tool_call/data/example.jsonl`.

## Configs

- `configs/example_single_tool_call.yaml` runs the example through Simple Agent's `/run`.
- `configs/example_single_tool_call_single_agent_turn.yaml` runs it through the single-agent-turn Environment Server, which seeds the Resources Server and Simple Agent sessions, grants the agent direct HTTP access to the tools, and verifies the response.

## Tutorial

For a hands-on walkthrough of building a single-step environment from scratch, see the [Single-Step Environment](https://docs.nvidia.com/nemo/gym/main/environment-tutorials/single-step-environment) tutorial.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
