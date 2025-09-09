# Aviary Wrapper Resource Server

Provides a NeMo Gym resource server that wraps the HotPotQA environment from Future-House Aviary for multi-hop question answering tasks.

## Description

This resource server wraps the HotPotQA environment to provide simple tool functions for NeMo Gym agents. HotPotQA is a multi-hop question answering environment where agents use search and lookup tools to find answers to complex questions. The wrapper handles all session management automatically while presenting a clean interface that exactly matches the training data.

## Features

- **Simple tool interface**: Tools match training data  - `search(entity)`, `lookup(keyword)`, `submit_answer(answer)`
- **Stateful session management**: Uses `simple_agent_stateful` to track session IDs across tool calls
- **HotPotQA integration**: multi-hop reasoning with Wikipedia search and lookup  
- **Reward calculation**: Leverages HotPotQA's built-in answer evaluation with proper session-based verification

## API Endpoints

### Tool Endpoints (Match Training Data)
- `POST /search` - Search for an entity on Wikipedia, sets current page in state for lookup to work for looking up terms on that
  - Request: `{"entity": "Barack Obama"}`
  - Response: `{"result": "Barack Obama is the 44th president..."}`
  
- `POST /lookup` - Look up keywords in current search results  
  - Request: `{"keyword": "birth year"}`
  - Response: `{"result": "(Result 1/3) Barack Obama was born in 1961..."}`
  
- `POST /submit_answer` - Submit final answer and get reward and store in state
  - Request: `{"answer": "1961"}`
  - Response: `{"result": "Finished.", "reward": 1.0, "done": true}`

- `POST /verify` - Extract reward from session state (requires session_id from stateful agent)

## Configuration

```yaml
max_steps: 10
correct_reward: 1.0
incorrect_reward: 0.0
```

## Usage

See `configs/run_aviary.txt` for complete setup instructions.

**Important**: This resource server requires the `simple_agent_stateful` agent to properly track session IDs across tool calls for correct reward verification.

### Quick Start
```bash
# Start servers
aviary_config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/aviary_wrapper/configs/aviary_wrapper.yaml"
ng_run "+config_paths=[$aviary_config_paths]"

# Collect rollouts  
ng_collect_rollouts +agent_name=aviary_wrapper_simple_agent_stateful \
    +input_jsonl_fpath=resources_servers/aviary_wrapper/data/hotpotqa_5_test.jsonl \
    +output_jsonl_fpath=resources_servers/aviary_wrapper/data/test_rollouts.jsonl
```

## Dependencies
- **fhaviary[hotpotqa]**: Future-House Aviary with HotPotQA environment
- **simple_agent_stateful**: Required for session state tracking
