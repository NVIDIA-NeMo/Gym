# AgentDojo family agent

Shared implementation for AgentDojo-family agent servers. It adapts the upstream AgentDojo execution pipeline to
NeMo Gym's `/run` contract and routes every policy-model request through the configured Gym model server.

This module is not a standalone benchmark entrypoint. Use [`agentdojo_agent`](../agentdojo_agent/README.md) and its
documentation for the official AgentDojo suites.

The shared configuration supports benchmark version selection, bounded concurrency, rollout timeouts, model-system
role compatibility, optional defenses, and trajectory accounting. The model bridge in `model_bridge.py` translates
between AgentDojo's Chat Completions messages and NeMo Gym response objects.

The benchmark-specific test suites exercise this module through their configured entrypoints:

```bash
gym env test +entrypoint=responses_api_agents/agentdojo_agent
```
