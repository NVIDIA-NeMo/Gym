(environment-tutorials-index)=
(env-creating-training-environment)=

# Building Environments

Building an RL environment in NeMo Gym involves three pillars: **Task Preparation** (what the model will practice), **Environment Design** (tools and verification), and **Model Training** (collecting rollouts and running RL). This guide walks through all three using a single-step weather tool as the running example.

:::{tip}
Looking to use an existing environment rather than build your own? See the [Available Environments](https://github.com/NVIDIA-NeMo/Gym#-available-environments) in the README.
:::

:::{admonition} Key Concepts
:class: seealso

Before diving in, review these foundational pages:

- {ref}`core-components` --- Model, Resources, and Agent servers
- {ref}`architecture` --- How components interact during startup and execution
- {ref}`task-verification` --- Reward computation and verification patterns
- {ref}`configuration-concepts` --- YAML configuration system
:::

## Prerequisites

Complete **both** of the following before starting:

1. **{doc}`/get-started/detailed-setup`** --- Clone the repository, install dependencies, configure your API key, and verify servers start correctly.
2. **{doc}`/get-started/rollout-collection`** --- Collect and view your first batch of rollouts.

:::{tip}
If you followed the {doc}`Quickstart </get-started/index>`, you've already completed both. You're ready to proceed.
:::

:::{important}
Run all commands from the **repository root** directory (where `pyproject.toml` is located).
:::

---

## How It Works

NeMo Gym uses a decoupled three-component architecture: the Agent Server orchestrates the loop, the Model Server runs inference, and the Resources Server provides tools and verification.

```text
                        Environment Components in NeMo Gym

                    +-----------------------------------------+
                    |            Agent Server                 |
                    |                                         |
                    |  run():                                 |
                    |    1. resources.seed_session()          |
                    |    2. multi-step/multi-turn agent loop: |
                    |         model.responses()               |
                    |         resources.my_tool()             |
                    |    3. resources.verify()                |
                    +-----------+-------------+---------------+
                                |             |
                                v             v
            +----------------------+    +------------------------+
            |    Model Server      |    |   Resources Server     |
            |                      |    |                        |
            |  responses():        |    |  seed_session():       |
            |    conversation      |    |    # init env state    |
            |    -> text, tool     |    |  my_tool():            |
            |    calls, code, etc. |    |    # execute action    |
            |                      |    |  verify():             |
            |                      |    |    # evaluate -> reward|
            +----------------------+    +------------------------+
```

The **Agent Server** orchestrates the run loop for each episode: it loads the prompt, calls the model, dispatches tool calls to the Resources Server, and triggers verification.

The **Model Server** exposes `responses()`: given a conversation, it produces text, tool calls, code, etc.

The **Resources Server** exposes three types of endpoints:
- `seed_session()` --- initialize environment state for each episode
- `my_tool()` --- execute actions (your custom tool endpoints)
- `verify()` --- evaluate the rollout and return a reward

The built-in `simple_agent` handles multi-step tool calling (repeated model-tool loops within a single episode) and is a good starting point. The `ng_init_resources_server` command automatically creates a paired simple agent configuration. For a taxonomy of rollout structures (multi-step, multi-turn, etc.) and core capabilities, see [Environment Properties](#environment-properties).

---

The rest of this guide walks through building a simple single-step environment: a weather assistant that calls one tool (`get_weather`) and verifies the model used it correctly.

## 1. Scaffolding

Scaffold a new resource server and its paired agent configuration:

```bash
ng_init_resources_server +entrypoint=resources_servers/my_weather_tool
```

This creates the resource server directory and a paired simple agent configuration:

```text
resources_servers/my_weather_tool/
+-- app.py                      # Main server implementation
+-- configs/
|   +-- my_weather_tool.yaml    # Configuration files
+-- data/
|   +-- .gitignore              # Data directory for examples/datasets
+-- tests/
|   +-- test_app.py             # Unit tests
+-- requirements.txt            # Python dependencies
+-- README.md                   # Documentation
```

---

## 2. Task Preparation

Understanding the task is the first step in designing the environment itself.

Every environment starts with **task data** --- the scenarios your model will practice on. Task data is stored in JSONL format (one JSON object per line), where each line represents a single training example. To get started, it's not atypical for a domain-expert hand-craft a few examples from scratch. Once the environment is developed and tested with these examples, you can scale up by collecting more data or using or synthetic generation with [NeMo Data Designer](https://github.com/NVIDIA-NeMo/Data-Designer).

### Create Data

Create `resources_servers/my_weather_tool/data/example.jsonl` with five weather examples. You may start with just the user inputs --- the remaining fields (tool definitions, system prompts, etc.) are populated automatically during [rollout collection](#collect-rollouts):

```json
{"input": [{"type": "text", "text": "What's the weather in San Francisco?"}]}
{"input": [{"type": "text", "text": "Tell me the weather in New York"}]}
{"input": [{"type": "text", "text": "How's the weather in Seattle?"}]}
{"input": [{"type": "text", "text": "What is the current weather in Boston?"}]}
{"input": [{"type": "text", "text": "Can you check the weather in Chicago?"}]}
```

### Complete JSONL Format

After rollout collection (which we will get to after building the environment), each line is expanded into the full schema with tool definitions, system prompts, and ground-truth metadata:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a helpful weather assistant."},
      {"role": "user", "content": "What's the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "City name"}},
          "required": ["city"],
          "additionalProperties": false
        },
        "strict": true
      }
    ],
    "parallel_tool_calls": false
  }
}
```

| Field | Description |
|---|---|
| `responses_create_params` | OpenAI Responses API-compatible input |
| `responses_create_params.input` | Conversation messages (system, user, assistant) |
| `responses_create_params.tools` | Available tools/functions for the agent |
| `responses_create_params.parallel_tool_calls` | Whether the model may call multiple tools simultaneously. Set to `false` to force sequential tool calls --- useful when tool outputs depend on each other. |
| `expected_*` (custom fields) | Ground truth fields passed through to `verify()` for reward computation |

---

## 3. Environment Design

This section covers the key aspects of building the environment itself: creating the Resources Server, writing verification logic, and testing.

### 3.1 Agent Server

The built-in `simple_agent` handles multi-step tool calling out of the box --- no custom agent code is needed. Its core logic (simplified from [`responses_api_agents/simple_agent/app.py`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/simple_agent)):

```python
# run() — episode lifecycle
async def run(self, request, body):
    await resources_server.seed_session(body)      # initialize env state
    response = await self.responses(body)           # multi-step agent loop
    return await resources_server.verify(response)  # compute reward

# responses() — multi-step tool loop
async def responses(self, body):
    while True:
        model_response = await model_server.responses(conversation)
        tool_calls = [o for o in model_response.output if o.type == "function_call"]

        if not tool_calls:  # model produced a final text response
            break

        for call in tool_calls:
            result = await resources_server.post(f"/{call.name}", call.arguments)
            conversation.append(result)

    return model_response
```

For most environments, `simple_agent` is all you need. For more advanced patterns (multi-turn correction, custom orchestration), see the other agents in [`responses_api_agents/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents), or build your own by extending `SimpleResponsesAPIAgent`.

### 3.2 Resources Server

While the agent handles orchestration, the **Resources Server** is where you define what makes your environment unique. It provides:
- **Tool implementations** --- APIs that models can call
- **Verification logic** --- reward computation for RL
- **Session state** --- per-episode state management (for stateful environments)


Open `resources_servers/my_weather_tool/app.py` and implement:

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
        return GetWeatherResponse(city=body.city, weather_description=f"The weather in {body.city} is cold.")

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Evaluate rollout and return a reward. See Verification Logic below."""
        ...


if __name__ == "__main__":
    MyWeatherResourcesServer.run_webserver()
```

#### Key Components

| Component | Purpose |
|---|---|
| **Configuration Class** | Extends `BaseResourcesServerConfig`; holds server-specific settings |
| **Request/Response Schemas** | Pydantic models defining the API contract |
| **`setup_webserver()`** | Registers FastAPI routes for your tools |
| **Tool Methods** | Async functions implementing tool logic |
| **`verify()`** | **Required** --- evaluates task performance and returns a reward |



### 3.3 Verification Logic

The `verify()` function is the heart of your RL environment --- it computes the reward signal that drives model training. In this example, verification is simple: return `1.0` if the model called the `get_weather` tool, `0.0` otherwise. Real environments will have more sophisticated logic, but the principle is the same --- inspect the model's output and score it.

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    # Check if the model called the get_weather tool
    used_tool = False
    for output in body.response.output:
        if output.type == "function_call" and output.name == "get_weather":
            used_tool = True
            break

    # Return higher reward if the tool was used correctly
    reward = 1.0 if used_tool else 0.0
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

See {ref}`task-verification` for the full verification patterns and best practices.

#### Configure - Wiring the pieces together

Open `resources_servers/my_weather_tool/configs/my_weather_tool.yaml`. This file contains both the resource server and its paired simple agent configuration. 

Update the `domain` field from `other` to `agent`:

```yaml
my_weather_tool:
  resources_servers:
    my_weather_tool:
      entrypoint: app.py
      domain: agent  # Change from 'other' to match your use case
my_weather_tool_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_weather_tool
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_weather_tool/data/example.jsonl
```

The `domain` field categorizes your resource server and is **required**. Common values: `math`, `coding`, `agent`, `knowledge`, `instruction_following`, `long_context`, `safety`, `games`, `e2e`, `other`.

:::{tip}
The domain is used for metrics grouping and dataset naming. Choose the category that best describes your task.
:::

The agent entry references the resource server and model server by name, wiring all three components together.


### 3.4 Testing

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
    return MyWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


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
            input=[{"role": "user", "content": "What's the weather?"}]
        ),
        response=NeMoGymResponse(
            id="",
            object="response",
            created_at=0.0,
            model="",
            output=[
                {
                    "role": "assistant",
                    "id": "",
                    "content": [{"type": "output_text", "annotations": [], "text": "It's cold."}],
                }
            ],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        ),
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

Return to the root directory and re-activate the root environment:

```bash
cd ../..
deactivate
source .venv/bin/activate
```

---

## 4. Model Training

### Run the Servers

Start the servers:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/my_weather_tool/configs/my_weather_tool.yaml"

ng_run "+config_paths=[$config_paths]"
```

`ng_run` reads the config files and starts all three components from the architecture diagram:

1. **Agent Server** (`my_weather_tool_simple_agent`) --- the `simple_agent` that orchestrates the seed → model → tool → verify loop
2. **Model Server** (`openai_model`) --- proxies LLM inference requests to the OpenAI API
3. **Resources Server** (`my_weather_tool`) --- serves your `get_weather` tool endpoint and `verify()` logic

The agent config is included automatically because it's defined in the same YAML file as the resource server.

### Configure API Keys

Configure your OpenAI API key in `env.yaml` (located in the repository root). The `env.yaml` is never committed to Git and is designed to hold secrets like API keys:

```yaml
openai_api_key: ???
policy_api_key: ${openai_api_key}
policy_base_url: https://api.openai.com/v1
policy_model_name: gpt-4o-mini
```

:::{tip}
If you don't want to use the OpenAI API, you can try using a local vLLM server (requires GPU access) instead! See {ref}`model-server-vllm`.
:::

### Test with Client

Inside `responses_api_agents/simple_agent/client.py`, change the server name from `example_single_tool_call_simple_agent` to `my_weather_tool_simple_agent`. Then, in a new terminal:

```bash
python responses_api_agents/simple_agent/client.py
```

The model should use your `get_weather` tool to answer questions about weather.

### Collect Rollouts

Before training, you collect rollouts to validate that your environment works end-to-end and to profile and establish a baseline. Each rollout runs a task through the full agent loop (prompt → model → tool calls → verification) and records the complete interaction along with the reward. This serves two purposes:

1. **Validation** --- confirm your tools, verification logic, and data produce sensible rewards. If a strong model scores near zero, something is likely wrong with your environment.
2. **Baselining** --- measure pass rates across models to understand task difficulty before training begins.

With your servers still running, collect rollouts against your example inputs:

```bash
ng_collect_rollouts +agent_name=my_weather_tool_simple_agent \
    +input_jsonl_fpath=resources_servers/my_weather_tool/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/my_weather_tool/data/example_rollouts.jsonl \
    +limit=null \
    +num_repeats=null \
    +num_samples_in_parallel=null
```

:::{note}
Ensure your servers are running before collecting rollouts. The command processes each input example, runs it through the servers, and saves the complete interaction including tool calls and verification rewards.
:::

### Train with RL

Once you've collected rollouts and validated your environment, run training with your preferred RL framework:

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo RL (GRPO)
:link: training-nemo-rl-grpo-index
:link-type: ref
Train models using GRPO with NeMo RL.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` TRL
:link: /training-tutorials/trl
:link-type: doc
Train with Hugging Face TRL.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Unsloth
:link: /training-tutorials/unsloth
:link-type: doc
Train with Unsloth for fast fine-tuning.
:::

::::

---

## Next Steps: Progressive Tutorials

These tutorials build on the concepts above with increasingly complex environments:

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Multi-Step Environment
:link: multi-step-environment
:link-type: doc
Multiple sequential tool calls with ground-truth verification.
+++
{bdg-primary}`intermediate`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Stateful Environment
:link: stateful-environment
:link-type: doc
Per-episode session state with `SESSION_ID_KEY`.
+++
{bdg-primary}`intermediate`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Real-World Environment
:link: real-world-environment
:link-type: doc
Production environment with dynamic routing and state-based verification.
+++
{bdg-primary}`advanced`
:::

::::

## Best Practices

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} {octicon}`repo;1.5em;sd-mr-1` Integrate external libraries
:link: integrate-external-environments
:link-type: doc
Best practices for integrating external training environments, benchmarks, or agents into NeMo Gym.
+++
{bdg-secondary}`benchmark`
:::

:::{grid-item-card} {octicon}`repo;1.5em;sd-mr-1` Add a benchmark
:link: adding-a-benchmark
:link-type: doc
Best practices for integrating benchmarks into NeMo Gym.
+++
{bdg-secondary}`benchmark`
:::

:::{grid-item-card} {octicon}`repo;1.5em;sd-mr-1` Design a customer evaluation
:link: designing-customer-evaluation
:link-type: doc
Best practices for designing a customer evaluation benchmark.
+++
{bdg-secondary}`benchmark`
:::

::::

## Appendix

### Environment Properties

Training environments can be broadly characterized along five dimensions:
1. **Rollout structure**: The interaction pattern between the model, environment, and user.
2. **Core capabilities**: The behaviors or skills that a model needs in order to succeed in a given use case.
3. **Knowledge domain**: What subject area, area of expertise, or field of study is involved.
4. **Task type**: The high-level use case that is represented in the training environment.
5. **Verification method**: How the environment computes rewards from model responses. See {doc}`/about/concepts/task-verification` for details.

Below are a subset of rollout structures and core capabilities found across NeMo Gym environments. We plan to add these as structured metadata to environments in the future. If you have ideas for additional properties, please let us know by [opening an issue](https://github.com/NVIDIA-NeMo/Gym/issues).

#### Rollout Structure
| Rollout structure | Description |
|---|---|
| Multi-step | Interleaved assistant and tool messages |
| Multi-turn | Interleaved user and assistant messages |
| Multi-modal | Interleaved text, image, video, and/or audio messages |
| Long context | Message content is very large or the number of messages is very large |

#### Core Capabilities
| Core capability | Developer/User need | Rollout Structures Required |
|---|---|---|
| Information dependency | The model receives environment responses that may require changes to subsequent actions. | Multi-step |
| Proactive asking | Developers put the model in a situation where user context is missing. The model needs to recognize user context is missing and ask the user for the missing context. | Multi-turn |
| Schema adherence | Users need more than one piece of information delivered by the model at one time in a specified delivery format. | |
| Meta data instruction following | User constrains the meta-properties of the model response e.g. "respond in 5 words". | |
| Counterintuitive instruction following | User provides instructions that are against conventional wisdom, typically making sense in the specific context in which the model is being used | |
| Information relevance | Given a large volume of inputs, the model needs to ignore content irrelevant to the task at hand. | Long context |
| Multiple intent synthesis | Users provide multiple tasks for the model to accomplish. | Multi-step, Multi-turn |

### Troubleshooting

#### Domain validation error

If you encounter the error `"A domain is required for resource servers"`, ensure the `domain` field is set in your config YAML file.

#### Import errors

Ensure you are running commands from the repository root directory and have installed dependencies:

```bash
uv sync
```

#### Server does not start

Check that:

- Port is not already in use
- Configuration file syntax is valid YAML
- All imports in `app.py` are correct

#### Tests fail

Ensure:

- You are in the correct Python environment
- All dependencies are installed
- Test file imports match your actual file structure

#### Debugging server behavior

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
