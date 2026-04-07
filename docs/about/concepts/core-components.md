(core-components)=

# Environment Components

> New to reinforcement learning for LLMs? Start with {ref}`training-approaches` for context on SFT, RL, and RLVR, or refer to {doc}`key-terminology` for a quick glossary.

## What is an Environment?

An environment in reinforcement learning is defined by a Markov Decision Process: at each step, the agent takes an action, the environment transitions to a new state, and returns an observation and a reward.

For a language model, the action at each step is a single token. The observation is the conversation context: the full text the model can see at that point, including the system prompt, prior turns, and any tool results that have been injected. The reward is sparse: zero at every step except the end of the episode, where a verifier assigns a score.

Because the model can only observe the conversation and not the full state of the environment (the contents of a database, the filesystem, the execution environment), most LLM training setups are technically partially observable MDPs (POMDPs). The model infers hidden state from what it observes through tool calls.

The environment does not step between tokens. While the model is generating a response, the environment waits. It steps only at response boundaries: when the model finishes generating, the agent decides what happens next. Tools may be called, results may be injected into the context, a new user turn may be added, or the episode may end and return a reward.

The episode is the full sequence from the initial prompt to the terminal state, potentially spanning many response boundaries. The policy gradient is computed across every token in the episode.

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