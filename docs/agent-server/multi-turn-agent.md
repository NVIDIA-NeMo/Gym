# Multi-Turn Agent

The [`MultiTurnAgent`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/multi_turn_agent) orchestrates dialogue between a **policy model** (being trained/evaluated) and a **user model** (an LLM simulating the human user). It wraps the [Multi-Step Agent's](multi-step-agent) tool-call loop with an outer turn-based loop.

This agent assumes the user side of the conversation is always driven by an LLM. If your use case requires a non-LLM user, create a custom agent that subclasses `MultiTurnAgent` and overrides `_generate_user_response`.

## How It Works

The agent has two nested loops:

- **Conversation loop (`run`)** — alternates between policy and user model turns. Controlled by `max_turns`.
- **Single-turn tool-call loop (`responses`)** — within a single turn, the model may make multiple tool calls. Same as the Multi-Step Agent's loop. Controlled by `max_steps_per_turn`.

```python
class MultiTurnAgent:
    async def run(self, task_data):
        # 1. Initialize episode
        resources_server.seed_session(task_data)
        conversation = task_data.prompt

        # 2. Conversation loop
        for turn in range(max_turns):
            # Policy turn — calls responses() which runs the single-turn tool-call loop
            policy_output = self.responses(conversation, task_data.tools)
            conversation.append(policy_output)

            if is_terminal(policy_output):
                break

            if turn < max_turns - 1:
                # User model turn
                user_message = user_model.responses(
                    user_system_prompt + conversation
                )
                conversation.append(user_message)

                if user_signals_done(user_message):
                    break

        # 3. Grade the result
        reward = resources_server.verify(conversation, task_data.ground_truth)
        return conversation, reward
```

## Configuration

```yaml
my_multi_turn_agent:
  responses_api_agents:
    multi_turn_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      user_model_server:
        type: responses_api_models
        name: user_model
      max_turns: 5
      max_steps_per_turn: null
      user_model_system_prompt: >-
        You are simulating a user in a conversation...
      user_model_stop_token: null
      user_model_tool_choice: null
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resources_server` | ref | required | Resources server for tools and verification |
| `model_server` | ref | required | Policy model being trained/evaluated |
| `user_model_server` | ref | required | LLM simulating the user |
| `max_turns` | int | required | Max policy turns per episode |
| `max_steps_per_turn` | int | null | Max tool-call steps per turn (both policy and user model). Null = unbounded for policy, safety limit of 10 for user model |
| `user_model_system_prompt` | str | required | System prompt for the user model persona. Can be overridden per task in JSONL |
| `user_model_stop_token` | str | null | Substring that terminates the conversation when detected in the user model's message |
| `user_model_tool_choice` | str | null | `"required"` forces tool use; null uses API default (`"auto"`) |

### Model Server Setup

The agent requires two model servers. Convenience configs that define both in one file:

- `responses_api_models/openai_model/configs/openai_model_with_user.yaml`
- `responses_api_models/vllm_model/configs/vllm_model_for_training_with_user.yaml`

Add user model credentials to `env.yaml`:

```yaml
# Policy model
policy_base_url: your-policy-endpoint
policy_api_key: your-policy-key
policy_model_name: your-policy-model

# User model
user_base_url: your-user-model-endpoint
user_api_key: your-user-key
user_model_name: your-user-model
```

Both can point to the same endpoint if you want both sides to use the same LLM.

## Stop Conditions

**Conversation** (across turns) stops when:

- `max_turns` reached
- `user_model_stop_token` detected in the user model's message
- Policy model hits `max_output_tokens`
- User model produces no output

**Single turn** (tool-call loop) stops on the same conditions as the [Multi-Step Agent](multi-step-agent):

- Text with no tool calls
- `max_steps_per_turn` reached
- `max_output_tokens` hit

For single-action-per-turn environments (e.g. board games), set `max_steps_per_turn: 1`

## User Model Tool Calls

The user model can call tools on the same resources server as the policy. Tools from the JSONL data are passed to both models. Only the user model's final text appears in the trajectory — its tool calls are internal. If the user model makes tool calls but produces no text (common with `user_model_tool_choice: required`), the last tool result is used as the user message.

## Output Format

The final `output` list contains the full interleaved trajectory across all turns. Each policy turn uses the same output types as the [Multi-Step Agent](multi-step-agent). User messages appear as `{role: "user", type: "message"}` items between turns.

A typical multi-turn sequence:

```
[
  # Policy turn 1
  reasoning, function_call, function_call_output, message,
  # User turn 1
  user_message,
  # Policy turn 2
  reasoning, function_call, function_call_output, message,
  # User turn 2
  user_message,
  # Policy turn 3
  reasoning, message,
]
```

Only policy model tokens are used for RL training; user model tokens are not included.

## Per-Task System Prompt

The `user_model_system_prompt` can be overridden per task in JSONL:

```text
{
  "responses_create_params": { "input": [...], "tools": [...] },
  "user_model_system_prompt": "You are a customer asking about returns..."
}
```

This allows different tasks within the same dataset to use different user personas without changing the YAML config.