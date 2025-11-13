(concepts-core-abstractions)=
# Core Abstractions

NeMo Gym is built around three simple but powerful ideas: **Agents** (orchestrators that connect components together), **Models** (LLMs that generate text), and **Resources** (tools and scoring systems).

Think of it like building a research assistant:
- The **Agent** is the coordinator who knows when to consult which resource and how to assemble the final answer
- The **Model** is the brain that understands language and generates responses
- The **Resources** are the reference books, calculators, and evaluation rubrics

This separation makes your system modular, testable, and easy to customize.

![NeMo Gym architecture showing the relationship between Agents, Models, and Resources, with downstream outputs to RL frameworks, offline training, and evaluation](../../_images/product_overview.png)

---

## The Three Abstractions

::::{tab-set}

:::{tab-item} Agents
**The Coordinators**

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
:::

:::{tab-item} Models
**The Brain**

**What it does**: Generates text based on prompts, without maintaining conversation history or deciding which tools to use.

**Think of it as**: A stateless function that takes text in, produces text out.

**Examples you can use**:
- OpenAI's GPT-4 or GPT-4o
- Azure OpenAI endpoints
- Local models via vLLM (Llama, Mistral, etc.)

Models don't orchestrate conversations or call tools directly—they just respond to what they're given. That's the Agent's job.
:::

:::{tab-item} Resources
**The Tools and Scorekeepers**

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
:::

::::

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
