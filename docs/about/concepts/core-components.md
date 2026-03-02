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

Model servers are stateless LLM inference endpoints. They receive a conversation and return the model's next output (text, tool calls, or code) with no memory or orchestration logic. During training, you will always have at least one active Model server, the "policy" model being trained.

See {doc}`/model-server/index` for available implementations and configuration.

:::

:::{tab-item} Resources

Resources servers define the environment: the tasks agents solve, the tools and external state they interact with, and the verification logic that scores performance and returns reward signals for training. Each resource server manages isolated per-rollout state via session IDs.

See {doc}`/resources-server/index` for details.

:::
::::
