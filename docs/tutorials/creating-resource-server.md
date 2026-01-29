---
orphan: true
---

(creating-resource-server)=

# Creating a Resource Server

Learn how to create a custom resource server to implement tools, verifiers, and business logic for your training environment.

:::{card}

**Goal**: Build a custom resource server with tools and verification logic.

^^^

**In this tutorial, you will**:

1. Initialize a resource server from template
2. Implement tool endpoints
3. Add verification logic for rewards
4. Test with rollout collection

:::

:::{button-ref} ../get-started/rollout-collection
:color: secondary
:outline:
:ref-type: doc

← Previous: Rollout Collection
:::

:::{important}
Run all commands from the **repository root** directory (where `pyproject.toml` is located).
:::

## What You'll Build

By the end of this tutorial, you'll have:

- [ ] A runnable resource server with `ng_run`
- [ ] Unit tests in `tests/test_app.py`
- [ ] Configuration with required `domain` field
- [ ] Example data in `data/example.jsonl` (5 examples)
- [ ] Example rollouts in `data/example_rollouts.jsonl`
- [ ] Documentation with licensing information

---

## What is a Resource Server?

Resource servers are the backbone of tool-based interactions in NeMo Gym. They provide:

- **Tool implementations**: APIs that models can call to perform actions or retrieve information
- **Verification logic**: Functions to evaluate model performance and compute rewards
- **Business logic abstraction**: Clean separation between model logic and domain-specific functionality

Each resource server must implement a `verify` function that evaluates the model's interactions and returns a reward signal for reinforcement learning.

**Key term**: A **rollout** is a complete interaction trace—the model's inputs, tool calls, and final outputs—used for training and evaluation.

---

## 1. Initialize the Resource Server

Resource servers live in the `resources_servers/` directory. Create a weather server that provides weather information to models.

Run the initialization command from the repository root:

```bash
ng_init_resources_server +entrypoint=resources_servers/my_weather_tool
```

This command creates a new directory structure with template files:

```text
resources_servers/my_weather_tool/
├── app.py                      # Main server implementation
├── configs/
│   └── my_weather_tool.yaml       # Configuration files
├── data/
│   └── .gitignore              # Data directory for examples/datasets
├── tests/
│   └── test_app.py             # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

:::{tip}
The initialization command also creates a paired simple agent configuration that references your resource server, making it easy to test end-to-end.
:::

---

## 2. Configure the Domain

Open `resources_servers/my_weather_tool/configs/my_weather_tool.yaml` and update the `domain` field:

```yaml
my_weather_tool_resources_server:
  resources_servers:
    my_weather_tool:
      entrypoint: app.py
      domain: agent  # Change from 'other' to 'agent' for this use case
```

The `domain` field categorizes your resource server and is **required**. Common categories include:

- `math` — Mathematical problem-solving
- `coding` — Code generation and programming
- `agent` — Agent-based interactions and tool calling
- `knowledge` — Knowledge-based question answering
- `instruction_following` — Instruction following benchmarks
- `long_context` — Long context handling
- `safety` — Safety and alignment
- `games` — Game-playing scenarios
- `e2e` — End-to-end workflows
- `other` — General purpose

:::{tip}
The domain is used for metrics grouping and dataset naming. Choose the category that best describes your task.
:::

---

## 3. Implement the Server

Open `resources_servers/my_weather_tool/app.py` and add the complete implementation:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# 1. Define the server configuration
class MyWeatherResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the weather resource server."""
    pass


# 2. Define request and response schemas for your tools
class GetWeatherRequest(BaseModel):
    """Request schema for getting weather information."""
    city: str


class GetWeatherResponse(BaseModel):
    """Response schema for weather information."""
    city: str
    weather_description: str


# 3. Implement the resource server
class MyWeatherResourcesServer(SimpleResourcesServer):
    config: MyWeatherResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        """Register API routes."""
        app = super().setup_webserver()
        
        # Register your tool endpoints
        app.post("/get_weather")(self.get_weather)
        
        return app

    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        """
        Tool implementation: Get weather for a city.
        
        In a production implementation, this would call a weather API.
        For this example, we return a simple static response.
        """
        return GetWeatherResponse(
            city=body.city,
            weather_description=f"The weather in {body.city} is cold."
        )

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """
        Verification function: Evaluate rollout performance.
        
        This function is called after a rollout completes.
        Return a reward between 0.0 and 1.0.
        """
        # Check if the model called the get_weather tool
        used_tool = False
        for output in body.response.output:
            if output.type == "function_call" and output.name == "get_weather":
                used_tool = True
                break
        
        # Return higher reward if the tool was used correctly
        reward = 1.0 if used_tool else 0.0
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    MyWeatherResourcesServer.run_webserver()
```

### Key Components

1. **Configuration Class**: Extends `BaseResourcesServerConfig` and holds server-specific settings
2. **Request/Response Schemas**: Pydantic models defining the API contract
3. **Server Class**: Extends `SimpleResourcesServer` and implements tools and verification
4. **`setup_webserver()`**: Registers FastAPI routes for your tools
5. **Tool Methods**: Async functions that implement the actual tool logic
6. **`verify()`**: **Required** method that evaluates task performance and returns a reward

---

## 4. Add Dependencies (Optional)

If your server needs external packages, add them to `requirements.txt`:

```text
-e nemo-gym[dev] @ ../../
# Add any other dependencies here
```

---

## 5. Write Tests

Update `resources_servers/my_weather_tool/tests/test_app.py` to test your implementation:

```python
import pytest
from unittest.mock import MagicMock
from nemo_gym.server_utils import ServerClient
from resources_servers.my_weather_tool.app import (
    MyWeatherResourcesServer,
    MyWeatherResourcesServerConfig,
    GetWeatherRequest,
)


@pytest.fixture
def server():
    """Create a server instance for testing."""
    config = MyWeatherResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="my_weather_tool",
    )
    return MyWeatherResourcesServer(
        config=config, server_client=MagicMock(spec=ServerClient)
    )


@pytest.mark.asyncio
async def test_get_weather(server):
    """Test the get_weather tool."""
    request = GetWeatherRequest(city="San Francisco")
    response = await server.get_weather(request)
    
    assert response.city == "San Francisco"
    assert "cold" in response.weather_description.lower()


@pytest.mark.asyncio
async def test_verify(server):
    """Test the verify function."""
    from nemo_gym.base_resources_server import BaseVerifyRequest
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    
    # Create a proper BaseVerifyRequest with required fields
    verify_request = BaseVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"type": "text", "text": "What's the weather?"}]
        ),
        response=NeMoGymResponse(
            output=[{"type": "text", "text": "It's cold."}]
        )
    )
    
    response = await server.verify(verify_request)
    assert response.reward >= 0.0
    assert response.reward <= 1.0
```

Run the tests:

```bash
ng_test +entrypoint=resources_servers/my_weather_tool
```

For detailed test output:

```bash
cd resources_servers/my_weather_tool
source .venv/bin/activate
pytest -v
```

---

## 6. Run with an Agent

The initialization command created a paired simple agent configuration in the same YAML file. Start the servers:

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/my_weather_tool/configs/my_weather_tool.yaml"

ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=my_weather_tool_resources_server
```

This starts three servers:

1. The simple agent server (coordinates interactions)
2. The OpenAI model server (provides LLM responses)
3. Your weather resource server (provides the `get_weather` tool)

Configure your OpenAI API key in `env.yaml` (located in the repository root):

```yaml
openai_api_key: ${oc.env:OPENAI_API_KEY}  # Reads from environment variable
policy_api_key: ${openai_api_key}
policy_base_url: https://api.openai.com/v1
policy_model_name: gpt-4o-mini
```

:::{tip}
Set your API key as an environment variable before running:

```bash
export OPENAI_API_KEY="sk-your-key-here"  # pragma: allowlist secret
```

Never commit API keys directly in YAML files.
:::

### Test the resources server

After the servers start, test your resources server in a new terminal:

```bash
python responses_api_agents/simple_agent/client.py
```

The model should be able to use your `get_weather` tool to answer questions about weather!

---

## 7. Create Example Data

Your resource server needs example data for testing and validation. Create `resources_servers/my_weather_tool/data/example.jsonl` with at least five example inputs.

:::{note}
JSONL (JSON Lines) format: one JSON object per line, no wrapping array or trailing commas.
:::

```json
{"input": [{"type": "text", "text": "What's the weather in San Francisco?"}]}
{"input": [{"type": "text", "text": "Tell me the weather in New York"}]}
{"input": [{"type": "text", "text": "How's the weather in Seattle?"}]}
{"input": [{"type": "text", "text": "What is the current weather in Boston?"}]}
{"input": [{"type": "text", "text": "Can you check the weather in Chicago?"}]}
```

### Generate Example Rollouts

Collect rollouts by running against your example inputs. This generates interaction traces showing how models use your tools:

```bash
ng_collect_rollouts +agent_name=my_weather_tool_simple_agent \
    +input_jsonl_fpath=resources_servers/my_weather_tool/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/my_weather_tool/data/example_rollouts.jsonl \
    +limit=null \
    +num_repeats=null \
    +num_samples_in_parallel=null
```

:::{note}
Ensure your servers are running (from step 6) before collecting rollouts. The command processes each input example, runs it through the servers, and saves the complete interaction including tool calls and verification rewards to `example_rollouts.jsonl`.
:::

---

## 8. Update Documentation

Update `resources_servers/my_weather_tool/README.md` with licensing and usage information:

```markdown
# My Weather Tool Resource Server

A simple weather information resource server demonstrating tool calling.

## Description

This resource server provides a `get_weather` tool that returns weather information for cities.

## Data

- Example data: Five synthetic weather queries

## Licensing Information

**Code**: Apache 2.0

**Data**: Apache 2.0 (synthetic examples)

## Dependencies

- nemo_gym: Apache 2.0
```

:::{important}
Your PR will not be merged unless licensing information is present and accurate!
:::

---

## Advanced: Verification Patterns

:::{dropdown} Multi-step verification with output parsing
:icon: code

For tasks requiring multiple tool calls, parse the final output to compute accuracy:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    """Extract and validate multi-step results."""
    expected = body.expected_values  # From request
    
    # Parse the final tool call output
    actual = []
    for output in reversed(body.response.output):
        if output.type == "function_call" and output.name == "submit_answer":
            import json
            actual = json.loads(output.arguments).get("values", [])
            break
    
    # Compute accuracy metrics
    accuracy = expected == actual
    set_overlap = len(set(actual) & set(expected)) / len(expected) if expected else 0
    
    return BaseVerifyResponse(
        **body.model_dump(),
        reward=float(accuracy),
    )
```

See `resources_servers/example_multi_step/app.py` for a complete example.
:::

:::{dropdown} LLM-as-judge verification
:icon: sparkle-fill

For tasks with multiple valid answers, use an LLM to judge correctness:

```python
# See resources_servers/math_with_judge/app.py for the full pattern
```

See `resources_servers/math_with_judge/app.py` for implementation details.
:::

:::{dropdown} Unit test verification (code generation)
:icon: beaker

For code generation tasks, run unit tests against model output:

```python
# See resources_servers/code_gen/app.py for the full pattern
```

See `resources_servers/code_gen/app.py` for implementation details.
:::

---

## Next Steps

Now that you have a working resource server:

1. **Add training data**: Collect rollouts and prepare datasets for RL training
2. **Add complex verification**: Add reward shaping and detailed performance metrics
3. **Scale up**: Add more tools and more sophisticated business logic
4. **Integrate with RL**: Use {ref}`RL Training with NeMo RL using GRPO <training-nemo-rl-grpo-index>` to train models on your tasks

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Collect Rollouts
:link: offline-training-w-rollouts
:link-type: doc
Learn how to collect and process rollouts for training data.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Train models using your resource server with NeMo RL.
:::

::::

---

## Troubleshooting

### Domain validation error

If you encounter the error `"A domain is required for resource servers"`, ensure the `domain` field is set in your config YAML file.

### Import errors

Ensure you are running commands from the repository root directory and have installed dependencies:

```bash
uv sync
```

### Server does not start

Check that:

- Port is not already in use
- Configuration file syntax is valid YAML
- All imports in `app.py` are correct

### Tests fail

Ensure:

- You are in the correct Python environment
- All dependencies are installed
- Test file imports match your actual file structure

### Debugging server behavior

Check server status and logs:

```bash
# View running servers
ng_status

# For detailed logs, run the server directly:
cd resources_servers/my_weather_tool
source .venv/bin/activate
python app.py
```

Server logs appear in the terminal where `ng_run` was executed.

---

## Summary

You've learned how to:

✅ Initialize a resource server with `ng_init_resources_server`  
✅ Configure the required `domain` field  
✅ Add tools and verification logic  
✅ Write and run tests  
✅ Run your server with an model
✅ Create required data artifacts  

Resource servers are the foundation for building custom RL environments in NeMo Gym. Experiment with different tool implementations and verification strategies to create engaging tasks for your models!
