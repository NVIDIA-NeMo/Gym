(resources-server)=

# Resources Server

The Resources server is the "world" the agent interacts with. It defines the task, the tools and actions available to the agent, and the verification logic that evaluates performance and returns reward signals for training.

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

For semantic or rubric-based scoring, `verify()` may call a **second language model** (LLM-as-a-judge); the concept is outlined in {ref}`task-verification` under *What is LLM-as-a-judge?*. For configuration, deployment, and implementation patterns, see {doc}`/environment-tutorials/llm-as-judge-verification`.

## Example Resources Servers

**[`workplace_assistant`](https://github.com/NVIDIA-NeMo/Gym/tree/main/nemo_gym/resources_servers/workplace_assistant)** — Multi-step tool calling in a workplace setting.
- **Task**: Execute business activities such as sending emails, scheduling meetings, and managing projects.
- **Actions**: 26 tools across 5 databases (email, calendar, analytics, project management, CRM). Each tool can read and mutate the database state.
- **Verification**: State matching: executes both the agent's actions and the ground truth actions against fresh databases, then compares the resulting states.

**[`math_with_code`](https://github.com/NVIDIA-NeMo/Gym/tree/main/nemo_gym/resources_servers/math_with_code)** — Mathematical reasoning with code execution.
- **Task**: Solve math problems using Python as a reasoning tool.
- **Actions**: `execute_python()` runs code in an isolated per-session process with numpy, scipy, and pandas available. State persists across steps so the agent can build on previous computations.
- **Verification**: Answer correctness: extracts the boxed answer from the model's final response and compares it against the expected result.

## Server Configuration
:::{seealso}
[Resources Server Fields](../reference/configuration.md#resources-server-fields) for server configuration syntax and fields.
:::
