(core-components)=

# Environment Components

> New to reinforcement learning for LLMs? Start with {ref}`training-approaches` for context on SFT, RL, and RLVR, or refer to {doc}`key-terminology` for a quick glossary.

## What is an Environment?

An environment is defined by the task for the agent to accomplish, the actions the agent can take, and the state of the world the agent observes and acts upon. The environment also determines how the agent's performance is evaluated: what constitutes success and how reward is assigned.

In NeMo Gym, these concepts map to three server components:

- **Agent** servers orchestrate each rollout: calling the model to generate outputs, routing tool calls through resources, and collecting the final reward.
- **Model** servers handle stateless text generation, producing the model's output each turn (e.g. text, tool calls, or code).
- **Resources** servers provide tasks, tools, and external state (sandboxes, databases, APIs) the agent interacts with, as well as the verification logic that scores performance.

```{image} ../../_images/product_overview.svg
:alt: NeMo Gym Architecture
:width: 800px
:align: center
```

::::{tab-set}

:::{tab-item} Agents

The Agent server defines whether a rollout is single-step or multi-step, single-turn or multi-turn, and orchestrates the full rollout lifecycle: calling the model, routing tool calls to resources, and collecting the final reward. It does not run an LLM itself, it delegates all text generation to the Model server.

You can use an existing agent, integrate an external one, or build your own from scratch. See {doc}`/agent-server/index` for details.

:::

:::{tab-item} Model

Responses API Model servers are stateless model endpoints that perform single-call text generation without conversation memory or orchestration. During training, you will always have at least one active Responses API Model server, typically called the "policy" model.

**Available Implementations:**

- `openai_model`: Integration with OpenAI's Responses API  
- `azure_openai_model`: Integration with Azure OpenAI API
- `vllm_model`: Middleware converting local models (using vLLM) to Responses API format

**Configuration:** Models are configured with API endpoints and credentials using YAML files in `responses_api_models/*/configs/`

:::

:::{tab-item} Resources

Resource servers host the components and logic of environments including multi-step state persistence, tool and reward function implementations. Resource servers are responsible for returning observations, such as tool results or updated environment state, and rewards as a result of actions taken by the policy model. Actions can be moves in a game, tool calls, or anything an agent can do. NeMo Gym contains a variety of NVIDIA and community contributed resource servers that you can use during training. We also have tutorials on how to add your own resource server.

**Examples of Resources**

A resource server usually provides tasks, possible actions, and {term}`verification <Verifier>` logic:

- **Tasks**: Problems or prompts that agents solve during rollouts
- **Actions**: Actions agents can take during rollouts, including tool calling
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

- **[`example_single_tool_call`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/example_single_tool_call)**: Mock weather API
  - **Task**: Report weather information
  - **Actions**: `get_weather()` returns mock weather data
  - **Verification logic**: Checks if weather tool was called correctly

**Configuration**: Refer to resource-specific config files in `resources_servers/*/configs/`

:::
::::
