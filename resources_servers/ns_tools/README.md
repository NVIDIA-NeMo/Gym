# Description
This is a resource server for executing NeMo Skills tools (e.g., stateful Python code execution) with math verification via math_with_judge.

It integrates with the NeMo Skills ToolManager to dynamically load and execute tools, maintaining stateful sessions across multiple tool calls within a rollout.

# Example usage

## Running servers
The following are example commands for running this resource server with the simple agent and a vLLM model:
```bash
config_paths="resources_servers/ns_tools/configs/ns_tools.yaml, \
resources_servers/math_with_judge/configs/math_with_judge.yaml, \
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]" \
    +policy_base_url="http://localhost:8000/v1" \
    +policy_model_name="Qwen/Qwen3-8B" \
    ++math_with_judge.resources_servers.math_with_judge.should_use_judge=false \
    ++math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

Then, rollouts can be collected using a command such as the following:
```bash
ng_collect_rollouts \
    +agent_name=ns_tools_simple_agent \
    +input_jsonl_fpath=resources_servers/ns_tools/data/example.jsonl \
    +output_jsonl_fpath=data/ns_tools_rollouts.jsonl \
    +limit=5
```

## Sample data format
Each sample requires:
- `question`: The math question being asked
- `expected_answer`: The expected answer (verified by math-verify)
- `responses_create_params`: The request parameters for the model, including tool definitions

Example with Python tool:
```json
{
  "question": "What is 2 + 2?",
  "expected_answer": "4",
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a helpful assistant. Put your final answer in \\boxed{}."},
      {"role": "user", "content": "What is 2 + 2? Use Python to calculate this."}
    ],
    "tools": [{
      "type": "function",
      "name": "stateful_python_code_exec",
      "description": "Execute Python code in a stateful environment.",
      "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"]
      }
    }]
  }
}
```

## Preparing datasets

The `prepare_dataset.py` script transforms source datasets into the JSONL format required by nemo-gym.

### Regenerating compmath_prepared.jsonl

The `data/compmath_prepared.jsonl` file is generated from the nemo-skills comp-math-24-25 dataset. To regenerate it:

```bash
python prepare_dataset.py \
    --input /path/to/nemo_skills/dataset/comp-math-24-25/test.txt \
    --output data/compmath_prepared.jsonl \
    --prompt_config generic/math \
    --tools nemo_skills.mcp.servers.python_tool.PythonTool \
    --verifier_type math_with_judge
```

### Validating example data

To validate the example data and regenerate metrics:
```bash
ng_prepare_data "+config_paths=[resources_servers/ns_tools/configs/ns_tools.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
    +output_dirpath=data/ns_tools \
    +mode=example_validation
```

# Licensing information
Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- nemo_skills: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
