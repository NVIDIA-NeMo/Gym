# NeMo Skills Tools Resources Server

This resources server provides integration with NeMo Skills ToolManager for tool execution (e.g., stateful Python code execution) with math verification via math_with_judge.

## Features

- **NeMo Skills Tools**: Dynamically loads tools from nemo_skills (e.g., PythonTool for stateful code execution)
- **Math Verification**: Uses math_with_judge (math-verify library) to verify model answers
- **Session Management**: Maintains stateful tool sessions across multiple tool calls within a rollout

## Prerequisites

- vLLM server running with tool calling enabled:
  ```bash
  vllm serve MODEL --enable-auto-tool-choice --tool-call-parser hermes
  ```
- nemo_skills sandbox server running on localhost:6000

## Quick Start

Run the demo script:

```bash
./resources_servers/ns_tools/run_demo.sh
```

Or manually:

### 1. Start Servers

```bash
config_paths="resources_servers/ns_tools/configs/ns_tools.yaml,\
resources_servers/ns_tools/configs/math_with_judge_no_judge.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[${config_paths}]" \
  +policy_base_url="http://localhost:8000/v1" \
  +policy_api_key="EMPTY" \
  +policy_model_name="Qwen/Qwen3-8B"
```

### 2. Collect Rollouts

```bash
ng_collect_rollouts \
    +agent_name=ns_tools_simple_agent \
    +input_jsonl_fpath=data/ns_tools_example.jsonl \
    +output_jsonl_fpath=data/ns_tools_rollouts.jsonl
```

## Sample Data Format

Each sample requires:
- `question`: The math question being asked
- `expected_answer`: The expected answer (verified by math-verify)
- `responses_create_params`: The request parameters for the model, including tool definitions

### Example with Python tool

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

### Example without tools (pure math)

```json
{
  "question": "What is 7 times 8?",
  "expected_answer": "56",
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "Show your work and put your final answer in \\boxed{}."},
      {"role": "user", "content": "What is 7 times 8?"}
    ]
  }
}
```

## Architecture

```
Sample with question + expected_answer
           │
           ▼
   NSToolsResourcesServer
           │
           ├── Tool calls: /{tool_name} → nemo_skills ToolManager
           │
           └── /verify → delegates to math_with_judge
```

## Configuration

See `configs/ns_tools.yaml` for configuration options including:
- `nemo_skills_tools`: List of tool modules to load
- `nemo_skills_tool_overrides`: Per-tool configuration
- `sandbox_host`/`sandbox_port`: Code execution sandbox connection
- `verifier`: Reference to the math_with_judge server
