(contribute-new-environment)=

# Add a New Environment

An environment consists of three components: Agents, Models, and Resources. Most contributors will create new resource servers while using existing agent server and model server implementations. If you need to create custom agents or models, you can reference the implementations in `responses_api_agents/` and `responses_api_models/`.

This guide focuses on the resource server contribution process.

:::{tip}
For a guide to building your first resource server, see {doc}`/tutorials/creating-resource-server`.
:::

## Required Files

Your resource server must include these files:

```{list-table}
:header-rows: 1
:widths: 30 70

* - File
  - Description
* - `app.py`
  - Main server implementation with `verify` function
* - `configs/*.yaml`
  - Configuration with valid `domain` field
* - `tests/test_app.py`
  - At least one unit test
* - `data/example.jsonl`
  - At least five example inputs
* - `data/example_rollouts.jsonl`
  - Pre-generated rollouts from example data (generate before submitting PR)
* - `requirements.txt`
  - Python dependencies
* - `README.md`
  - Documentation with licensing information
```


## Contribution Workflow

Contributing a resource server follows this sequence:

```{list-table}
:header-rows: 1
:widths: 10 30 60

* - Step
  - Phase
  - Description
* - 1
  - Curate Tasks
  - Collect or generate training tasks and create example data
* - 2
  - Implementation
  - Build resource server with verification logic
* - 3
  - Testing
  - Write and run unit tests
* - 4
  - Example Rollouts
  - Generate example rollouts to verify functionality
* - 5
  - Reward Profiling
  - Validate reward distribution with inference runs
* - 6
  - Training Validation
  - Train with GRPO to ensure meaningful training signal
* - 7
  - Submit PR
  - Submit pull request with all required information
* - 8
  - Review
  - Address reviewer feedback and verify reproducibility
```

## Detailed Steps

### 1. Curate Training Tasks

Prepare the dataset for your environment:

- Collect or generate prompts/tasks for your environment
- Create `data/example.jsonl` with at least 5 representative task examples

### 2. Resource Server Implementation

Build your resource server:

- Run `ng_init_resources_server +entrypoint=resources_servers/my_server` to scaffold the new resource server
- Follow the {doc}`/tutorials/creating-resource-server` guide to implement your specific logic
- Implement verification logic for your tasks by defining the `verify()` function. See {doc}`/about/concepts/task-verification` for common verification patterns.
- Set the `domain` field in your resource server configuration (see {py:class}`~nemo_gym.config_types.Domain`).
- Complete the auto-generated `README.md` with licensing information

### 3. Testing

Write and run tests for your resource server:

- At least one test per server is required for PR approval
- You are responsible for ensuring your tests adequately cover your server's functionality

### 4. Generate Example Rollouts

Verify basic functionality and generate example rollouts:

- Document the command used to start your server, for example, `ng_run +entrypoint=resources_servers/my_server`
- Generate rollouts and save 5 example outputs to `data/example_rollouts.jsonl` to demonstrate correct reward signals

### 5. Reward Profiling

Run inference to validate reward distribution:

- Use a ~500 sample subset (minimum)
- Use Qwen3-4B, Qwen3 30B A3B, or equivalent model
- Generate 16 responses per prompt
- Report reward distribution
- **For tool calling**: Provide tool call metrics and correlation with rewards

### 6. Training-Based Validation

Validate with actual training:

- Train with GRPO on Qwen3-4B, Qwen 30B A3B Instruct, or equivalent model
- Include training accuracy curve
- Include test benchmark accuracy curve (if applicable)

### 7. Submit PR

Include the following in your pull request description:

- Description of the environment
- Description of the verification logic
- Description of the prompts/tasks: What is the source? Which domain does it cover?
- Provide relevant license information for data and software. If models were used for synthetic data generation, note this in your PR description

### 8. PR Review Process

After submitting your PR:

1. A team member will be assigned to review and reproduce your environment
2. The reviewer will verify all steps and check correctness of the 5 example rollouts
3. The reviewer will re-run the procedure to ensure reproducibility
4. Address any feedback from reviewers
5. Once approved, maintainers will merge your contribution

## Recommended Technical Design

For optimal performance and scalability, we recommend following these design patterns:

### Async-First Design

Endpoint handlers should be asynchronous to handle concurrent requests efficiently during training:

```python
# Recommended: async function
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

Avoid spawning additional threads or processes unless necessary. A single Gym instance can handle tens of thousands of concurrent requests when properly implemented.

### NeMo Gym OpenAI Client

We recommend using the NeMo Gym OpenAI client from `nemo_gym.openai_utils`:

```python
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
```

The NeMo Gym client is optimized for scale and provides consistent behavior. External clients like LiteLLM often preprocess or postprocess inputs and outputs in ways that can interfere with training data collection.

### Pydantic Models

Consider using Pydantic models for request and response validation by extending base classes from `nemo_gym.base_resources_server`:

```python
from pydantic import BaseModel
from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse

class MyVerifyRequest(BaseVerifyRequest):
    expected_result: str
    difficulty: int
```

### Error Handling

Tool execution errors should be propagated back to the model rather than crashing the server, enabling the model to learn from mistakes:

```python
async def execute_tool(self, path: str, body: ToolRequest) -> ToolResponse:
    try:
        result = self.tool_functions[path](**body.model_dump())
        return ToolResponse(output=result)
    except Exception as e:
        # Return error to model so it can correct itself
        return ToolResponse(output=f"Error executing tool '{path}': {str(e)}")
```

### Configuration

Pass configuration through NeMo Gym config files rather than environment variables for better reproducibility:

```yaml
# configs/my_server.yaml
host: 0.0.0.0
port: 8000
domain: agent
```

### Multi-Step Rollouts

For multi-step scenarios, the model returns training information on response messages (`prompt_token_ids`, `generation_token_ids`, `generation_log_probs`). When constructing messages for subsequent model calls, propagate this information from previous responses to maintain the training data chain.


## Reference

- {doc}`/tutorials/creating-resource-server` - Introductory tutorial for creating your first resource server
- {doc}`/about/concepts/task-verification` - Verification and reward concepts
- {doc}`/about/concepts/core-components` - Resource server architecture
