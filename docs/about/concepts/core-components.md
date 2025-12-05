(core-components)=

# Core Components

Before diving into code, let's understand the three server components that make up a training environment in NeMo Gym.

> If you are new to reinforcement learning for LLMs, we recommend you refer to **[Key Terminology](./key-terminology)** first.

```{image} ../../_images/product_overview.svg
:alt: NeMo Gym Architecture
:width: 800px
:align: center
```

::::{tab-set}

:::{tab-item} Model

Responses API Model servers are stateless model endpoints that perform single-call text generation without conversation memory or orchestration. During training, you will always have at least one active Responses API Model server, typically called the "policy" model.

**Available Implementations:**

- `openai_model`: Integration with OpenAI's Responses API  
- `azure_openai_model`: Integration with Azure OpenAI API
- `vllm_model`: Middleware converting local models (using vLLM) to Responses API format

**Configuration:** Models are configured with API endpoints and credentials using YAML files in `responses_api_models/*/configs/`

:::

:::{tab-item} Resources

Resource servers host the components and logic of environments including multi-step state persistence, tool and reward function implementations. Resource servers are responsible for returning observations, such as tool results or updated game state, and rewards as a result of actions taken by the policy model. Actions can be moves in a game, tool calls, text generation, or anything an agent can do. NeMo Gym contains a variety of NVIDIA and community contributed resource servers that you can use during training. We also have tutorials on how to add your own resource server.

**Examples of Resources**

A resource server usually provides tasks, possible actions, and {term}`verification <Verifier>` logic:

- **Tasks**: Problems or prompts that agents solve during rollouts
- **Actions**: Actions agents can take during rollouts
- **Verification logic**: Scoring logic that evaluates performance (returns {term}`reward signals <Reward / Reward Signal>` for training)

**Example Resource Servers**

Each example shows what **task** the agent solves, what **actions** are available, and what **verification logic** measures success:

- **[`google_search`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/google_search)**: Web search with verification
  - **Task**: Answer knowledge questions using web search
  - **Actions**: `search()` queries Google API; `browse()` extracts webpage content
  - **Verification logic**: Checks if final answer matches expected result for MCQA questions

- **[`math_with_code`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/math_with_code)**: Mathematical reasoning with code execution
  - **Task**: Solve math problems using Python
  - **Actions**: `execute_python()` runs Python code with numpy, scipy, pandas
  - **Verification logic**: Extracts boxed answer and checks mathematical correctness

- **[`code_gen`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/code_gen)**: Competitive programming problems
  - **Task**: Implement solutions to coding problems
  - **Actions**: None (agent generates code directly)
  - **Verification logic**: Executes generated code against unit test inputs/outputs

- **[`math_with_judge`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/math_with_judge)**: Mathematical problem solving
  - **Task**: Solve math problems
  - **Actions**: None (or can be combined with `math_with_code`)
  - **Verification logic**: Uses math library + LLM judge to verify answer equivalence

- **[`mcqa`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/mcqa)**: Multiple choice question answering
  - **Task**: Answer multiple choice questions
  - **Actions**: None (knowledge-based reasoning)
  - **Verification logic**: Checks if selected option matches ground truth

- **[`instruction_following`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/instruction_following)**: Instruction compliance evaluation
  - **Task**: Follow specified instructions
  - **Actions**: None (evaluates response format/content)
  - **Verification logic**: Checks if response follows all specified instructions

- **[`simple_weather`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/example_simple_weather)**: Mock weather API
  - **Task**: Report weather information
  - **Actions**: `get_weather()` returns mock weather data
  - **Verification logic**: Checks if weather tool was called correctly

**Configuration**: Refer to resource-specific config files in `resources_servers/*/configs/`

:::

:::{tab-item} Agents

Responses API Agent servers {term}`orchestrate <Orchestration>` the rollout lifecycle—the full cycle of task execution and verification.

- Execute tool calls by routing model requests to resources
- Run multi-step and multi-turn agentic loops (model → tool execution → repeat)
- Manage session state across servers

NeMo Gym provides several agent patterns covering multi-step, multi-turn, and user modeling scenarios.

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
