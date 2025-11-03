(gs-first-agent)=

# Your First Agent

In the setup tutorial, you ran a command and saw JSON output from a weather agent. But what actually happened? Let's break down the agent workflow and learn to interact with it directly.

:::{card}

**Goal**: Understand how your weather agent works and learn to interact with it.

^^^

**In this tutorial, you will**:

1. Trace the complete agent workflow from request to response
2. Understand the role of models, tools, and resource servers
3. Modify the agent request to test different behaviors
4. Explore the Responses API output format

:::

:::{button-ref} setup-installation
:color: secondary
:outline:
:ref-type: doc

← Previous: Setup and Installation
:::

:::{tip}
**Going deeper**: For conceptual understanding of how **agents**, **models**, and **resource servers** work together, refer to [Core Abstractions (Concepts)](../about/concepts/core-abstractions.md).
:::

---

## The Agent Workflow

In the setup tutorial, you ran this command and saw JSON output:

```bash
python responses_api_agents/simple_agent/client.py
```

Let's break down exactly what happened behind the scenes. When you ran the client script, here's the complete flow:

### 1. Client Script Sends Request
```python
# From the client script
{"role": "user", "content": "going out in sf tn"}
```

**Who**: Your client script (`client.py`)  
**What**: Sends a casual user message to the agent server  
**Result**: Agent receives the request and prepares to process it

### 2. Agent Forwards to GPT-4
**Who**: The agent server (`simple_weather_simple_agent`)  
**What**: Packages the request with context and available tools, then sends to GPT-4:
- System message: "You are a helpful personal assistant..."
- User message: "going out in sf tn"
- Available tools: `get_weather` function definition

**Result**: GPT-4 receives everything it needs to make decisions

### 3. GPT-4 Decides to Call Tool
```json
{
    "type": "function_call",
    "name": "get_weather", 
    "arguments": "{\"city\":\"San Francisco\"}",
    "status": "completed"
}
```

**Who**: GPT-4 (the language model)  
**What**: Analyzes the request, recognizes "sf" as San Francisco, determines weather info is needed  
**Result**: Returns a tool call instruction to the agent

### 4. Resource Server Executes Tool
```json
{
    "type": "function_call_output",
    "output": "{\"city\": \"San Francisco\", \"weather_description\": \"The weather in San Francisco is cold.\"}"
}
```

**Who**: Weather resource server (`simple_weather`)  
**What**: Receives the tool call, executes `get_weather("San Francisco")`, returns weather data  
**Result**: Tool output is sent back to GPT-4

### 5. GPT-4 Generates Final Response
```json
{
    "type": "message",
    "content": [{"type": "output_text", "text": "The weather in San Francisco tonight is cold. You might want to wear layers or bring a jacket to stay comfortable while you're out!"}]
}
```

**Who**: GPT-4 (the language model)  
**What**: Takes the weather data and crafts a helpful response for the user  
**Result**: Complete response returned to your client script

---

## Understanding the Output Format

The JSON output uses OpenAI's [**Responses API**](https://platform.openai.com/docs/api-reference/responses/object#responses/object-output).

The output list may contain multiple item types, such as:

- **ResponseOutputMessage:** user-facing message content returned by the model.
- **ResponseOutputItemReasoning:** internal reasoning or "thinking" traces that explain the model's thought process.
- **ResponseFunctionToolCall:** a request from the model to invoke an external function or tool.

---

## Modifying the Agent Request

Let's try different inputs to see how the agent behaves. 

1. Create a new file at `responses_api_agents/simple_agent/custom_client.py`:

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

2. Specify the config and run NeMo Gym.

   ```bash
   # Define which servers to start
   config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
   resources_servers/simple_weather/configs/simple_weather.yaml"

   # Start all servers
   ng_run "+config_paths=[${config_paths}]"
   ```

3. Keep your servers running, and run your new custom client in new terminal:

   ```bash
   # Navigate to project directory
   cd /path/to/Gym

   # Activate virtual environment
   source .venv/bin/activate

   # Run your new custom client
   python responses_api_agents/simple_agent/custom_client.py
   ```

## What You'll Observe

You might notice that running the same query multiple times can produce different behaviors:

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Question Type
  - Example
  - Agent Behavior
* - **Weather questions**
  - "What's the weather like in New York?"
  - Calls `get_weather` tool
* - 
  - "Should I bring an umbrella to Chicago?"
  - Calls `get_weather` tool
* - **Non-weather questions**
  - "Tell me a joke"
  - Responds directly, no tool call
* - **Ambiguous questions**
  - "I'm planning a picnic in Seattle tomorrow"
  - May or may not call weather tool
```

This non-deterministic behavior is normal for language models—and it's exactly why we need **verification and scoring** (covered in the next tutorial) to measure and improve agent quality consistently.


## About This Implementation

In this weather agent example, both the tool and verification functions are implemented directly within NeMo Gym:

**Weather Tool** (`resources_servers/simple_weather/app.py`):
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

- **Connect to external APIs** (real weather services, databases, and so on)
- **Implement sophisticated verification** (checking accuracy, measuring performance)
- **Handle error cases** (API failures, invalid inputs)

:::{tip}
A later tutorial will cover integrating with external services and building more realistic resource servers.
:::



## What You've Learned

This weather agent demonstrates patterns you'll see throughout NeMo Gym:

- **Agent workflow**: Request → Analysis → Tool calls → Integration → Response
- **Models** handle the reasoning and decision-making
- **Resource servers** provide tools and verification
- **Agents** orchestrate between models and resources
- Everything is configurable using YAML files

---

## Next Steps

Now that you understand how agents work, the next question is: how do you measure if an agent's response is *good*? That's where verification comes in.

:::{button-ref} verifying-agent-results
:color: primary
:outline:
:ref-type: doc

Next: Verifying Agent Results →
:::

Learn how to score agent performance and create the reward signals that drive reinforcement learning.

