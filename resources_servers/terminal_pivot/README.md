# Description

This is a resource server for verifying terminal-based agent actions. It evaluates agent responses that represent terminal command sequences against expected answers. The server supports two different schema formats (`terminus_1` and `terminus_2`) for terminal interaction tasks.

For each verification request, the agent's JSON output is validated through multiple checks:
1. **JSON Parsing**: The model output must be valid JSON
2. **Schema Validation**: The response must conform to the specified harness schema (`terminus_1` or `terminus_2`)
3. **Task Completion**: If the expected answer requires task completion, the agent must also indicate completion
4. **Command Correctness**: The predicted keystrokes must exactly match the expected keystrokes in order

The server returns a binary reward: `1.0` for correct responses that pass all checks, `0.0` otherwise.

## Supported Schemas

### terminus_1
- `state_analysis`: Description of the current terminal state
- `explanation`: Brief explanation of what the commands will do
- `commands`: List of command objects with `keystrokes`, `is_blocking`, and `timeout_sec`
- `is_task_complete`: Boolean indicating if the task is complete

### terminus_2
- `analysis`: Analysis of the current state based on terminal output
- `plan`: Description of the plan for next steps
- `commands`: List of command objects with `keystrokes` and optional `duration`
- `task_complete`: Boolean indicating if the task is complete (optional)

Data links: ?

# Example usage

## Running servers

The following command can be used to run this resource server, along with the simple agent and a policy model:

```bash
config_paths="resources_servers/terminal_pivot/configs/terminal_pivot.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

Then, rollouts can be collected using a command such as the following:

```bash
ng_collect_rollouts \
    +agent_name=terminal_pivot_simple_agent \
    +input_jsonl_fpath=resources_servers/terminal_pivot/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/terminal_pivot/data/example_rollouts.jsonl
```

## Expected Data Format

Each data sample should include:
- `expected_answer`: A JSON string containing the expected terminal commands
- `metadata.harness`: Either `"terminus_1"` or `"terminus_2"` to specify the schema format

# Licensing information

Code: Apache 2.0<br>
Data: ?

## Dependencies

- nemo_gym: Apache 2.0
- openapi-schema-validator: BSD-3-Clause

