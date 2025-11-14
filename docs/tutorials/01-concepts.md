# Understanding Core Concepts

Before diving into code, let's understand the three core abstractions in NeMo Gym.

> [!TIP]
> If you are new to reinforcement learning for LLMs, we recommend you review **[Key Terminology]** first.

## Overview

<div align="center">
  <img src="product_overview.png" alt="NeMo Gym Architecture" width="800">
</div>

## 1. Models

The LLM that performs text inference - stateless, single-call text generation without conversation memory or orchestration

**Available Implementations:**
- `openai_model`: Direct integration with OpenAI's Responses API  
- `vllm_model`: Middleware converting local models (via vLLM) to Responses API format

**Configuration:** Models are configured with API endpoints and credentials via YAML files in `responses_api_models/*/configs/`


## 2. Resources

Servers that provide tools for agents to use and verifiers that score agent performance on tasks. NeMo Gym contains a variety of NVIDIA and community contributed resources that you may wish to utilize. We also have tutorials on how to add your own Resource server.

**Resources Provide**
- **Tools**: Functions agents can call (e.g., `get_weather`, `search_web`)
- **Verification Logic**: Scoring systems that evaluate agent responses for training/evaluation

**Examples:**
- `simple_weather`: Mock weather API for testing and tutorials
- `google_search`: Web search capabilities via Google Search API  
- `math_with_code`: Python code execution environment for mathematical reasoning
- `math_with_judge`: Mathematical problem verification using symbolic computation
- `mcqa`: Multiple choice question answering evaluation
- `instruction_following`: General instruction compliance scoring


**Configuration**: See resource-specific config files in `resources_servers/*/configs/`


## 3. Agents

The orchestration layer that connects models to resources.

- Route requests to the right model
- Provide tools to the model
- Handle multi-turn conversations
- Format responses consistently

**Examples:**
- `simple_agent`: Basic agent that coordinates model calls with resource tools

**Configuration Pattern**:
```yaml
your_agent_name:                     # server ID
  responses_api_agents:              # server type. corresponds to the folder name in the project root
    your_agent_name:                 # agent type. name of the folder inside the server type folder 
      entrypoint: app.py             # server entrypoint path, relative to the agent type folder 
      resources_server:              # which resource server to use
        name: simple_weather         
      model_server:                  # which model server to use
        name: policy_model           
```



## How They Work Together

Let's trace through a weather request:

1. **User** → **Agent**: "What's the weather in NYC?"

2. **Agent** → **Model**: 
   ```json
   {
     "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
     "tools": [{"name": "get_weather", "parameters": "..."}]
   }
   ```

3. **Model** → **Agent**: "I should call the weather tool"
   ```json
   {"tool_calls": [{"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}]}
   ```

4. **Agent** → **Resource**: Calls the weather server

5. **Resource** → **Agent**: Returns weather data
   ```json
   {"city": "NYC", "weather_description": "cold"}
   ```

6. **Agent** → **Model**: "Here's the weather data, respond to user"

7. **Model** → **Agent** → **User**: "It's cold in NYC, bring a jacket!"

## Key Benefits of This Architecture

- **Reusable**: Same model can work with different tools
- **Testable**: Each component can be tested independently  
- **Scalable**: Components can run on different servers
- **Flexible**: Mix and match any combination


## What's Next?

Now that you understand the concepts, let's get everything installed and running.

→ **Next: [Setup and Installation]**
