(core-components)=

# Environment Components

> New to reinforcement learning for LLMs? Start with {ref}`training-approaches` for context on SFT, RL, and RLVR, or refer to {doc}`key-terminology` for a quick glossary.

## What is an Environment?

An environment is defined by the task for the agent to accomplish, the actions the agent can take, and the state of the world the agent observes and acts upon. The environment also determines how the agent's performance is evaluated: what constitutes success and how reward is assigned.

In NeMo Gym, these concepts map to three server components:

- **{doc}`Agent </agent-server/index>`** servers define whether a rollout is single-step or multi-step, single-turn or multi-turn, and orchestrate the full rollout lifecycle: calling the model, routing tool calls to resources, and collecting the final reward. The Agent server does not run an LLM itself — it delegates all text generation to the Model server.
- **{doc}`Model </model-server/index>`** servers are stateless LLM inference endpoints. They receive a conversation and return the model's next output (text, tool calls, or code) with no memory or orchestration logic.
- **{doc}`Resources </resources-server/index>`** servers provide the tasks that agents solve, the tools and external state they interact with, and the verification logic that scores performance and returns reward signals for training. Each resources server manages isolated per-rollout state via session IDs.

```
┌──────────────────────────────────────────┐
│              Agent Server                │
│                                          │
│  run():                                  │
│    1. resources.seed_session()  ─────────────►  Resources Server
│    2. multi-step/multi-turn loop:        │
│         model.responses()       ─────────────►  Model Server
│         resources.my_tool()     ─────────────►  Resources Server
│    3. resources.verify()        ─────────────►  Resources Server
│         → reward                         │
└──────────────────────────────────────────┘

┌───────────────────────────┐   ┌───────────────────────────────────┐
│       Model Server        │   │        Resources Server           │
│                           │   │                                   │
│  responses():             │   │  seed_session(): init env state   │
│    conversation           │   │  my_tool():      execute action   │
│    → text, tool calls,    │   │  verify():       evaluate → reward│
│      or code              │   │                                   │
└───────────────────────────┘   └───────────────────────────────────┘
```