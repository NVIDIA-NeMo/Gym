# Your First Agent

**Goal**: Understand how your weather agent works and learn to interact with it

## What Just Happened?

In the setup tutorial, you ran this command and saw JSON output:
```bash
python responses_api_agents/simple_agent/client.py
```

Let's break down exactly what happened behind the scenes.

## The Agent Workflow

When you ran the client script, here's the complete flow:

### 1. User Request
```python
# From the client script
{"role": "user", "content": "going out in sf tn"}
```

Your agent received a casual message about going out in San Francisco tonight.

### 2. Agent Analyzes Request  
The agent (configured in `responses_api_agents/simple_agent/client.py`) sent this to GPT-4:
- **System message**: "You are a helpful personal assistant..." (defined in the client script)
- **User message**: "going out in sf tn" 
- **Available tools**: `get_weather` function definition
- **GPT-4's decision**: Recognizes "sf" as San Francisco and "going out" implies needing weather info

### 3. Tool Call
```json
{
    "type": "function_call",
    "name": "get_weather", 
    "arguments": "{\"city\":\"San Francisco\"}",
    "status": "completed"
}
```

The agent decided to call the weather tool with San Francisco as the city. The argument structure `{"city": "..."}` matches the tool definition provided in the client script's `parameters.properties.city` field.

### 4. Tool Execution
```json
{
    "type": "function_call_output",
    "output": "{\"city\": \"San Francisco\", \"weather_description\": \"The weather in San Francisco is cold.\"}"
}
```

The weather resource server (defined in `resources_servers/example_simple_weather/app.py`) returned this response; it always says "cold" regardless of the actual city.

### 5. Final Response
```json
{
    "type": "message",
    "content": [{"type": "output_text", "text": "The weather in San Francisco tonight is cold. You might want to wear layers or bring a jacket to stay comfortable while you're out!"}]
}
```

The agent used the weather data to give helpful advice.

## Understanding the Output Format

The JSON output uses OpenAI's [**Responses API**](https://platform.openai.com/docs/api-reference/responses/object#responses/object-output).

The output list may contain multiple item types, such as:

- **ResponseOutputMessage:** user-facing message content returned by the model.
- **ResponseOutputItemReasoning:** internal reasoning or "thinking" traces that explain the model’s thought process.
- **ResponseFunctionToolCall:** a request from the model to invoke an external function or tool.

## Modifying the Agent Request

Let's try different inputs to see how the agent behaves. Create a new file at `responses_api_agents/simple_agent/custom_client.py`:

```python
# custom_client.py
import json
from asyncio import run
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

server_client = ServerClient.load_from_global_config()

# Try different messages
test_messages = [
    "What's the weather like in New York?",
    "Should I bring an umbrella to Chicago?", 
    "Tell me a joke",  # No weather needed
    "I'm planning a picnic in Seattle tomorrow"
]

async def test_agent():
    for message in test_messages:
        print(f"\n Testing: '{message}'")
        print("-" * 50)
        
        task = server_client.post(
            server_name="simple_weather_simple_agent",
            url_path="/v1/responses", 
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {
                        "role": "developer",
                        "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
                    },
                    {"role": "user", "content": message},
                ],
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather information for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name"},
                            },
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                ],
            ),
        )
        
        result = await task
        response_data = await result.json()
        
        print(json.dumps(response_data["output"], indent=4))
        print()  # Extra line for readability between tests

run(test_agent())
```

Specify the config and run NeMo Gym
```bash
# Define which servers to start
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_simple_weather/configs/simple_weather.yaml"

# Start all servers
ng_run "+config_paths=[${config_paths}]"
```

Keep your servers running, and run your new custom client in new terminal:
```bash
# Navigate to project directory
cd /path/to/Gym

# Activate virtual environment
source .venv/bin/activate

# Run your new custom client
python responses_api_agents/simple_agent/custom_client.py
```

## What You'll Observe

**Weather questions** → Agent calls the tool:
- "What's the weather like in New York?" → Calls `get_weather`
- "Should I bring an umbrella to Chicago?" → Calls `get_weather` 

**Non-weather questions** → Agent responds directly:
- "Tell me a joke" → No tool call, just responds with humor

**Ambiguous questions** → Agent makes intelligent decisions:
- "I'm planning a picnic in Seattle tomorrow" → Likely calls weather tool


## About This Implementation

In this weather agent example, both the tool and verification functions are implemented directly within NeMo Gym:

**Weather Tool** (`resources_servers/example_simple_weather/app.py`):
```python
async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
    return GetWeatherResponse(
        city=body.city, 
        weather_description=f"The weather in {body.city} is cold."
    )
```

**Verification Function** (same file):
```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

This is a simplified example to help you understand the agent workflow. In production scenarios, you would typically:

- **Connect to external APIs** (real weather services, databases, etc.)
- **Implement sophisticated verification** (checking accuracy, measuring performance)
- **Handle error cases** (API failures, invalid inputs)

> [!TIP]
> A later tutorial will cover integrating with external services and building more realistic resource servers.



## What You've Learned

This weather agent demonstrates patterns you'll see throughout NeMo Gym:
- **Agent workflow**: Request → Analysis → Tool calls → Integration → Response  
- **Models** handle the reasoning and decision-making
- **Resource servers** provide tools and verification  
- **Agents** orchestrate between models and resources
- Everything is configurable via YAML files

→ **[Next: Verifying Agent Results]**
