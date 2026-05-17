(agent-server)=

# Agent Server

The Agent server is the central component of environment design. It defines whether a rollout is single-step or multi-step, single-turn or multi-turn, and orchestrates all interaction logic — calling the model, executing tool calls through resources, and collecting the final reward. The Agent server does not run an LLM itself, it is orchestration code that delegates all text generation to the Model server.

## Rollout Lifecycle

Every agent follows a three-phase lifecycle:

1. **Seed:**  initialize the Resources server session with task data
2. **Agent loop:** send the conversation to the Model server, route any tool calls to the Resources server and feed results back to the Model, optionally interact with a user or another agent. Repeat until stop criteria are met (e.g. max steps, max turns, context length exceeded).
3. **Verify** — send the full conversation to the Resources server to compute a reward

```python
class Agent:
    async def run(self, task_data):
        # 1. Initialize episode
        resources_server.seed_session(task_data)

        # 2. Run the agent loop (may be multiple steps and/or multiple turns)
        response = self.responses(task_data.prompt, task_data.tools)

        # 3. Grade the result
        reward = resources_server.verify(response, task_data.ground_truth)
        return response, reward
```

How the agent loop works depends on the agent implementation, e.g. single turn vs multi-turn.


## Reference Agents
- **[Multi-Step Agent](multi-step-agent):** (`SimpleAgent`) single-turn, multi-step. One user message triggers a model response that may loop through tool calls before verification.
- **[Multi-Turn Agent](multi-turn-agent):** multi-turn, multi-step. Alternates between a policy model and an LLM user model over several dialogue turns. Each turn can include tool-call steps. Used for games, conversations, and iterative refinement.

## Design Conventions
- **Cookie propagation:** cookies are propagated through all hops (model server, resources server) so stateful environments can track session state.
- **Token propagation:** policy model token IDs and log probs are included in the output trajectory for RL training.
- **Tool error handling:** tool call errors from the resources server are returned to the model as normal tool results, not raised as exceptions. Invalid calls become part of the training trajectory.

## Integrating External Agents

You can also integrate external agents that bring their own tools and interaction patterns. For example, [`MiniSWEAgent`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/mini_swe_agent) wraps a coding harness running in Docker containers and converts its output back into the NeMo Gym format.

## Tools in Agent vs. Resources Server

Existing agents may come with predefined tools, allowing you to leverage them directly and use the Resources server to supplement with any additional external tools. When building a new environment, prefer defining tools in the Resources server rather than the Agent server. This separation of concerns allows different agents to share the same Resources server without duplicating tool logic.

## Server Configuration
:::{seealso}
[Agent Server Fields](../reference/configuration.md#agent-server-fields) for server configuration syntax and fields.
:::
