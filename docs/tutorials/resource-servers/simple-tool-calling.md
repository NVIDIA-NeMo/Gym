(tutorial-weather-api-server)=

# Build a Weather API Server

In {doc}`Your First Resource Server </tutorials/creating-resource-server>`, you learned the structure of a resource server. Now you'll build a complete weather API server that agents can query to answer weather questions.

:::{card}

**Goal**: Build a working weather tool server with deterministic verification.

^^^

**In this tutorial, you will**:

1. Create a `get_weather` tool that returns city weather data
2. Implement verification that checks if agents extract information correctly
3. Create test prompts and run end-to-end
4. Walk away with a server you can modify for your own API integrations

:::

:::{button-ref} /tutorials/resource-servers/index
:color: secondary
:outline:
:ref-type: doc

← Resource Server Patterns
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`/tutorials/creating-resource-server`
- ✅ NeMo Gym installed and working
- ✅ Basic Python and FastAPI knowledge

**What you'll build**: A weather information server where agents call `get_weather("Paris")` and you verify they correctly report the temperature and conditions.

:::{tip}
**Reference implementation**: `resources_servers/example_simple_weather/`
:::

---

## 1. Understand What We're Building

The weather server handles a simple task:

1. User asks: "What's the weather in Paris?"
2. Agent calls `get_weather(city="Paris")`
3. Server returns: `{"temperature": 22, "conditions": "sunny"}`
4. Agent responds: "It's 22°C and sunny in Paris"
5. Verification checks: Did the agent correctly extract and report the data?

```{mermaid}
sequenceDiagram
    participant User
    participant Agent
    participant WeatherServer
    User->>Agent: What's the weather in Paris?
    Agent->>WeatherServer: get_weather(city="Paris")
    WeatherServer-->>Agent: {temperature: 22, conditions: "sunny"}
    Agent->>User: It's 22°C and sunny in Paris
    WeatherServer-->>Agent: ✅ reward=1.0
```

**✅ Success Check**: You understand the request → tool → response → verify flow.

---

## 2. Create the Weather Tool

<!-- SME: Extract and adapt from example_simple_weather/app.py -->

```python
# app.py
from nemo_gym.base_resources_server import SimpleResourcesServer, BaseResourcesServerConfig
from pydantic import BaseModel

class WeatherServerConfig(BaseResourcesServerConfig):
    pass

class WeatherServer(SimpleResourcesServer):
    
    def __init__(self, config: WeatherServerConfig):
        super().__init__(config)
        
        # Mock weather data - replace with real API for production
        self.weather_data = {
            "Paris": {"temperature": 22, "conditions": "sunny"},
            "London": {"temperature": 15, "conditions": "cloudy"},
            "Tokyo": {"temperature": 28, "conditions": "humid"},
            "New York": {"temperature": 18, "conditions": "partly cloudy"},
        }
    
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
    
    def call_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "get_weather":
            city = tool_args.get("city", "")
            weather = self.weather_data.get(city, {"temperature": "unknown", "conditions": "unknown"})
            return f"Temperature: {weather['temperature']}°C, Conditions: {weather['conditions']}"
        return "Unknown tool"
```

**✅ Success Check**: You have a `get_tools()` returning the tool schema and `call_tool()` handling requests.

---

## 3. Implement Verification

The verification checks if the agent correctly reported the weather information:

```python
# Add to app.py

def verify(self, responses_create_params: dict, output: list) -> float:
    """Verify the agent correctly reported the weather."""
    
    # Extract the city from the original prompt
    user_message = responses_create_params["input"][0]["content"]
    
    # Find which city was asked about
    asked_city = None
    for city in self.weather_data:
        if city.lower() in user_message.lower():
            asked_city = city
            break
    
    if not asked_city:
        return 0.0  # City not in our data
    
    # Get expected values
    expected = self.weather_data[asked_city]
    expected_temp = str(expected["temperature"])
    expected_conditions = expected["conditions"]
    
    # Check the final assistant response
    final_response = ""
    for msg in reversed(output):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_response = msg["content"].lower()
            break
    
    # Verify both temperature and conditions are mentioned
    has_temp = expected_temp in final_response
    has_conditions = expected_conditions.lower() in final_response
    
    if has_temp and has_conditions:
        return 1.0  # Full credit
    elif has_temp or has_conditions:
        return 0.5  # Partial credit
    else:
        return 0.0  # Incorrect
```

**✅ Success Check**: Verification returns 1.0 for correct answers, 0.5 for partial, 0.0 for wrong.

---

## 4. Create Test Data

Create `data/example.jsonl` with test prompts:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What's the weather like in Paris today?"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "Tell me the current weather in London"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "How's the weather in Tokyo?"}], "model": "gpt-4"}}
```

---

## 5. Create Configuration

Create `configs/weather_server.yaml`:

```yaml
weather_resources_server:
  resources_servers:
    weather:
      entrypoint: app.py
      domain: other

weather_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: weather_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/weather/data/example.jsonl
```

---

## 6. Run and Test

```bash
# Start the servers
config_paths="resources_servers/weather/configs/weather_server.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$config_paths]"

# In another terminal, collect rollouts
ng_collect_rollouts +agent_name=weather_simple_agent \
    +input_jsonl_fpath=resources_servers/weather/data/example.jsonl \
    +output_jsonl_fpath=results/weather_rollouts.jsonl \
    +limit=3
```

**✅ Success Check**: Rollouts show reward=1.0 for correct weather reports.

---

## Adapt for Your Use Case

This pattern works for any single-step API integration:

- **Stock prices**: `get_stock_price(symbol)` → verify price reported correctly
- **Database lookup**: `query_user(id)` → verify user info extracted
- **Calculator**: `calculate(expression)` → verify math result reported

Replace the mock data with real API calls and adjust verification logic.

---

## Learn More

- {doc}`/training/verification/index` — Verification patterns and reward design
- {doc}`data-extraction-server` — When you need multiple tool calls

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Extraction
:link: data-extraction-server
:link-type: doc

Build a server requiring multiple tool calls to complete tasks.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Stateful Counter
:link: counter-game-server
:link-type: doc

Build a server with session state management.
:::

::::
