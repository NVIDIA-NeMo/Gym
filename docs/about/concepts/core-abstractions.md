(core-abstractions)=

# Core Abstractions

Before diving into code, let's understand the three core abstractions in NeMo Gym.

> If you are new to reinforcement learning for LLMs, we recommend you refer to **[Key Terminology](./key-terminology)** first.

```{image} ../../_images/product_overview.svg
:alt: NeMo Gym Architecture
:width: 800px
:align: center
```

::::{tab-set}

:::{tab-item} Model

Responses API Model servers are model endpoints that perform text inference - stateless, single-call text generation without conversation memory or orchestration. You will always have at least one Response API Model server active during training, typically known as the "policy" model.

**Available Implementations:**

- `openai_model`: Direct integration with OpenAI's Responses API  
- `vllm_model`: Middleware converting local models (using vLLM) to Responses API format

**Configuration:** Models are configured with API endpoints and credentials using YAML files in `responses_api_models/*/configs/`

:::

:::{tab-item} Resources

Resource servers provide tool implementations that can be invoked through tool calling and verification logic that measures task performance. NeMo Gym contains a variety of NVIDIA and community contributed resource servers that you can use during training. We also have tutorials on how to add your own resource server.

**What Resources Provide**

Each resource server combines both tools and verification logic:

- **Tools**: Functions agents can call during task execution
- **Verification logic**: Scoring logic that evaluates performance (returns reward signals for training)

**Example Resource Servers**

Each example shows what **tools** the agent can use and what **verification logic** measures success:

- **[`google_search`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/google_search)**: Web search with verification
  - **Tools**: `search()` queries Google API; `browse()` extracts webpage content
  - **Verification logic**: Checks if final answer matches expected result for MCQA questions

- **[`math_with_code`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/math_with_code)**: Mathematical reasoning with code execution
  - **Tool**: `execute_python()` runs Python code with numpy, scipy, pandas
  - **Verification logic**: Extracts boxed answer and checks mathematical correctness

- **[`code_gen`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/code_gen)**: Competitive programming problems
  - **Tools**: None (agent generates code directly)
  - **Verification logic**: Executes generated code against unit test inputs/outputs

- **[`math_with_judge`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/math_with_judge)**: Mathematical problem solving
  - **Tools**: None (or can be combined with `math_with_code`)
  - **Verification logic**: Uses math library + LLM judge to verify answer equivalence

- **[`mcqa`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/mcqa)**: Multiple choice question answering
  - **Tools**: None (knowledge-based reasoning)
  - **Verification logic**: Checks if selected option matches ground truth

- **[`instruction_following`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/instruction_following)**: Instruction compliance evaluation
  - **Tools**: None (evaluates response format/content)
  - **Verification logic**: Checks if response follows all specified instructions

- **[`simple_weather`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/example_simple_weather)**: Mock weather API
  - **Tool**: `get_weather()` returns mock weather data
  - **Verification logic**: Checks if weather tool was called correctly

**Configuration**: Refer to resource-specific config files in `resources_servers/*/configs/`

:::

:::{tab-item} Agents

Responses API Agent servers orchestrate the interaction between models and resources.

- Route requests to the right model
- Provide tools to the model
- Handle multi-turn conversations
- Format responses consistently

An agent can also be referred to as a "training environment." NeMo Gym contains several training environment patterns that cover a variety of scenarios including multi-step, multi-turn, or user modeling scenarios.

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

:::
::::
