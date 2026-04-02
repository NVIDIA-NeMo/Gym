# Multi-Turn Agent

A reference agent server that orchestrates multi-turn dialogue between a **policy model** (being trained/evaluated) and a **user model** (an LLM simulating the human user).

This agent assumes the user side of the conversation is always driven by an LLM. The `user_model_server` is a required config field. There is no built-in support for scripted, template-based, or environment-driven user messages. If your use case requires a non-LLM user (e.g., deterministic scripts or resources-server-generated prompts), create a custom agent that subclasses `MultiTurnAgent` and overrides `_generate_user_message`, or build a separate agent tailored to your pattern.

## How It Works

The agent runs a turn-based loop:

1. **Seed** the resources server session with task data.
2. **Policy turn** — call the policy model with the full conversation. The policy may execute tool calls within a turn (same inner loop as `SimpleAgent`).
3. **User turn** — call the user model with a system prompt and the conversation so far to generate the next user message.
4. **Repeat** until a stop condition is met.
5. **Verify** the full conversation with the resources server to produce a reward.

### Stop Conditions

The outer conversation loop stops when:

- `max_turns` reached
- User model emits the configured `user_model_stop_token`
- Policy model hits `max_output_tokens` (context length exceeded)

Within each policy turn, the inner tool-call loop stops when:

- The policy model produces a text response with no tool calls
- `max_steps_per_turn` reached (same behavior as `SimpleAgent.max_steps`)
- `max_output_tokens` hit on the model response

## Configuration

The base template (`configs/multi_turn_agent.yaml`) defines the agent's interface. Each environment provides its own config that sets environment-specific values:

```yaml
# Base template (multi_turn_agent.yaml) — documents all parameters
multi_turn_agent:
  responses_api_agents:
    multi_turn_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: ???                     # required, set by resources server config
      model_server:
        type: responses_api_models
        name: policy_model
      user_model_server:
        type: responses_api_models
        name: user_model
      max_turns: ???                  # required, set by resources server config
      max_steps_per_turn: null        # no limit by default; inner loop self-terminates
      user_model_system_prompt: ???   # required; can be overridden per-task in JSONL
      user_model_stop_token: null     # e.g. "[END]" to signal conversation end
```

```yaml
# Environment config (e.g. tic_tac_toe.yaml) — sets concrete values
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
      user_model_stop_token: "[END]"
```

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resources_server` | ResourcesServerRef | required | Resources server for tools and verification |
| `model_server` | ModelServerRef | required | Policy model being trained/evaluated |
| `user_model_server` | ModelServerRef | required | LLM simulating the user |
| `max_turns` | int | required | Maximum number of policy turns (no default; must be set per environment) |
| `max_steps_per_turn` | int | null | Max tool-call steps per policy turn (null = unbounded; inner loop self-terminates) |
| `user_model_system_prompt` | str | required | System prompt for the user model (can be overridden per-task in JSONL) |
| `user_model_stop_token` | str | null | Token that signals conversation end (e.g. "[END]") |

### Per-Task Override

The `user_model_system_prompt` can be overridden per task by including it in the JSONL data alongside `responses_create_params`:

```json
{
  "responses_create_params": { "input": [...], "tools": [...] },
  "user_model_system_prompt": "You are a customer asking about returns..."
}
```

## User Model Tool Calls

The user model can make tool calls against the same resources server as the policy model. If the JSONL data includes `tools`, those are passed to the user model too. This enables scenarios where the user model also interacts with the environment.

Only the user model's final text message appears in the conversation trajectory — its tool calls are internal and not visible to the policy model. The policy observes the effects of user tool calls through environment state on its next turn.

If the environment doesn't need user model tools, simply omit `tools` from the JSONL data.

## Output Format

The final `NeMoGymResponse.output` contains the full interleaved trajectory:

```
[policy_turn_1_outputs..., user_message_1, policy_turn_2_outputs..., user_message_2, ...]
```

Policy outputs include assistant messages, tool calls, and tool results. User messages appear as standard `{role: "user"}` input messages.

Only policy model tokens are relevant for RL training — user model tokens are not included in training metadata.
