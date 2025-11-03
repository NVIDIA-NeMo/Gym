(concepts-core-abstractions)=
# Core Abstractions

NeMo Gym is built around three simple but powerful ideas: **Models** (LLMs that generate text), **Resources** (tools and scoring systems), and **Agents** (orchestrators that connect them together).

Think of it like building a research assistant:
- The **Model** is the brain that understands language and generates responses
- The **Resources** are the reference books, calculators, and evaluation rubrics
- The **Agent** is the coordinator who knows when to consult which resource and how to assemble the final answer

This separation makes your system modular, testable, and easy to customize.

![NeMo Gym architecture showing the relationship between Models, Resources, and Agents, with downstream outputs to RL frameworks, offline training, and evaluation](../../_images/product_overview.png)

---

## The Three Abstractions

### Models — The Brain

**What it does**: Generates text based on prompts, without maintaining conversation history or deciding which tools to use.

**Think of it as**: A stateless function that takes text in, produces text out.

**Examples you can use**:
- OpenAI's GPT-4 or GPT-4o
- Azure OpenAI endpoints
- Local models via vLLM (Llama, Mistral, etc.)

Models don't orchestrate conversations or call tools directly—they just respond to what they're given. That's the Agent's job.

### Resources — The Tools and Scorekeepers

**What it does**: Provides two key capabilities:
1. **Tools** that agents can call (search the web, execute code, query databases)
2. **Verification** that scores how well the agent performed (for RL training)

**Think of it as**: A toolbox with built-in quality control.

**Examples**:

```{list-table}
:header-rows: 0
:widths: 20 80

* - **Math**
  - Provides calculator tool and verifies answers are mathematically correct
* - **Coding**
  - Executes Python code and verifies it passes test cases
* - **Search**
  - Provides web search and verifies the agent found relevant information
```

Each resource server knows how to score agent performance in its domain, generating the reward signals needed for reinforcement learning. NeMo Gym includes several production-ready resource servers for you to get started with.

### Agents — The Coordinators

**What it does**: Connects models to resources, manages the tool-calling loop, and handles multi-turn conversations.

**Think of it as**: The project manager who:
- Routes user requests to the model
- Provides available tools to the model
- Executes tool calls when the model requests them
- Manages conversation history across turns
- Ensures responses are properly formatted

**Example**: When a user asks "What's the weather in NYC?", the agent:
1. Sends the question to the model with available tools
2. Receives "I should call get_weather for NYC"
3. Calls the weather resource server
4. Sends the weather data back to the model
5. Returns the model's friendly response to the user

---

## How They Work Together

Let's trace a simple weather query through the system:

1. **User asks**: "What's the weather in NYC?"

2. **Agent → Model** (with available tools):
   > "Here are the tools you can use: get_weather(city). The user asks: What's the weather in NYC?"

3. **Model → Agent** (decides to use tool):
   > "I'll call get_weather with city='NYC'"

4. **Agent → Resource Server** (executes the tool call)

5. **Resource → Agent** (returns data):
   > "NYC: Cold, 35°F"

6. **Agent → Model** (provides tool result):
   > "The weather tool returned: Cold, 35°F. Respond to the user."

7. **Model → Agent → User** (final response):
   > "It's cold in NYC—35°F. You'll want to bring a warm jacket!"

The Agent coordinates everything, but the Model makes decisions and generates text.

---

## Why This Design?

This modular architecture provides four key benefits:

```{list-table}
:header-rows: 1
:widths: 25 75

* - Benefit
  - What It Enables
* - **Reusable Components**
  - The same model can work with different tools. The same tools can work with different models. Mix and match without rewriting code.
* - **Test in Isolation**
  - Debug your math verifier without touching the model. Test your agent logic without deploying infrastructure. Each piece works independently.
* - **Deploy Flexibly**
  - Run everything locally on your laptop, or distribute across multiple servers. Each component is an independent HTTP service.
* - **Swap via Configuration**
  - Want to try a different model? Change one line in your YAML config. Need a different set of tools? Update the resource server reference. No code changes required.
```

---

## Technical Details

:::{dropdown} Implementation Architecture
Each abstraction is implemented as a FastAPI HTTP server:

| Abstraction | Base Class | Key Endpoints |
|-------------|-----------|---------------|
| **Models** | `BaseResponsesAPIModel` | `POST /v1/responses`, `POST /v1/chat/completions` |
| **Resources** | `BaseResourcesServer` | `POST /verify`, `POST /seed_session` |
| **Agents** | `BaseResponsesAPIAgent` | `POST /v1/responses`, `POST /run` |

All components communicate via HTTP, enabling deployment flexibility.
:::

:::{dropdown} Model Integration Details
**Available Model Implementations**:

- **openai_model**: Direct OpenAI API integration
- **azure_openai_model**: Azure OpenAI endpoints
- **vllm_model**: Local model serving with vLLM

**Configuration Example**:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      api_key: ${oc.env:OPENAI_API_KEY}
      model_name: gpt-4o-2024-08-06
```

**API Format**: Standardized on OpenAI's Responses API for tool-calling support and broad compatibility.
:::

:::{dropdown} Resource Server Details
**Core Responsibilities**:

1. **Provide Tools**: Define functions in OpenAI format that agents can call
2. **Execute Logic**: Implement the actual tool behavior (search, calculate, etc.)
3. **Score Performance**: Return reward signals (0.0-1.0) for RL training

**Reward Signal**:
The `verify()` endpoint returns a `BaseVerifyResponse` with a `reward` field—the critical output for reinforcement learning training pipelines.

**Stateful Sessions**:
Resources can maintain state across turns using the `seed_session()` endpoint for complex multi-turn workflows.
:::

:::{dropdown} Agent Coordination Details
**Agent Configuration**:

```yaml
simple_agent_weather:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      max_steps: 10
      resources_server:
        name: simple_weather
      model_server:
        name: policy_model
```

**Key Parameters**:
- `max_steps`: Prevent infinite tool-calling loops
- `resources_server`: Which resource server to connect to
- `model_server`: Which model to use for generation

**Tool-Calling Loop**:
The agent manages the iterative process of: model generates tool call → agent executes → model receives results → repeat until done.
:::

