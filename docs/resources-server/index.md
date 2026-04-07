(resources-server)=

# Resources Server

In reinforcement learning, "environment" has two related meanings. The first is the world the agent acts in: something that accepts actions, maintains state, and returns observations and rewards. The second, used more loosely in LLM training, refers to the complete setup used to train a policy model.

A resources server is a flexible building block for environments in the first sense. It can expose tools the model can call, maintain state across the turns of a rollout, verify outputs and return rewards, and orchestrate secondary models such as reward models or judges. Not all of these are required; a resources server can be as minimal as a single verifier or as complex as a full simulated environment.

The training environment in the second sense is defined by the agent server. In multi-environment training, each environment is identified by an agent server reference (`agent_ref`). You have N environments for N agent refs. `simple_agent` is reused across many environments, each paired with a different resources server; other environments use a custom agent that owns more of the episode logic. Some agent servers contain all of the environment logic themselves and use the resources server only for verification or not at all; SWE-RL and verifier-style agents are examples of this.

```python
# Resources Server - pseudocode
class MyResourceServer(SimpleResourcesServer):
    
    # Initialize the "sandbox" for this specific rollout
    async def seed_session(self, session_id, task_data):
        self.state[session_id] = initialize_environment(task_data)

    # Define tool implementations
    async def my_tool(self, session_id, tool_args):
        result = execute_action(self.state[session_id], tool_args)
        return result

    # Define verification logic
    async def verify(self, session_id, response, ground_truth):
        # 1. Extract what the agent actually did
        actual_outcome = self.state[session_id].get_final_state()
        
        # 2. Reward if the actual outcome matches expected outcome
        if actual_outcome == ground_truth:
            return reward(1.0)
        return reward(0.0)
```

## Session Management

NeMo Gym uses a `session_id` to maintain isolated state for every parallel rollout. This ensures that concurrent rollouts never interfere with each other, and for multi-step environments, preserves state across steps within a single rollout.

## Tool Implementations

Tools are exposed as HTTP endpoints that the Agent server calls during a rollout. Each tool receives the `session_id` to access the correct rollout state, executes an action, and returns the result as an observation back to the model. Tools may also mutate the session state (e.g., updating a database), which the verifier can later inspect to evaluate performance.

## Verification Logic

Every Resources server implements a `verify()` function that evaluates the result of a rollout and returns a reward signal for training. See {doc}`/about/concepts/task-verification` for verification approaches, patterns, and best practices.

## Example Resources Servers

**[`workplace_assistant`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/workplace_assistant)** — Multi-step tool calling in a workplace setting.
- **Task**: Execute business activities such as sending emails, scheduling meetings, and managing projects.
- **Actions**: 26 tools across 5 databases (email, calendar, analytics, project management, CRM). Each tool can read and mutate the database state.
- **Verification**: State matching: executes both the agent's actions and the ground truth actions against fresh databases, then compares the resulting states.

**[`math_with_code`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/math_with_code)** — Mathematical reasoning with code execution.
- **Task**: Solve math problems using Python as a reasoning tool.
- **Actions**: `execute_python()` runs code in an isolated per-session process with numpy, scipy, and pandas available. State persists across steps so the agent can build on previous computations.
- **Verification**: Answer correctness: extracts the boxed answer from the model's final response and compares it against the expected result.

## Env API

For new environments, consider using the {doc}`Env API </resources-server/env-api>` instead of `SimpleResourcesServer`. It follows the Gymnasium interface and requires less boilerplate.

## Server Configuration
:::{seealso}
[Resources Server Fields](../reference/configuration.md#resources-server-fields) for server configuration syntax and fields.
:::
