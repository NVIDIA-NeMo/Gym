(multi-turn-agent)=

# Multi-Turn Agent

```{note}
This agent is under active development. The design below describes the target architecture.
```

## Background

NeMo Gym currently provides `SimpleAgent` as the default reference agent. It handles **single-turn, multi-step** interactions: one user message triggers a model response that may loop through tool calls, then verification produces a reward. There is also a demonstrated need for **multi-turn training**, where the agent interacts with a simulated user over several dialogue turns before completing its objective and receiving a reward.

## Description

The Multi-Turn Agent is a new reference agent server that orchestrates a multi-turn dialogue between a **policy model** (being trained or evaluated) and a **user model** (an LLM simulating the human user).

## Design

### Configuration

The agent accepts a second model server reference (`user_model_server`) in its config alongside the existing `model_server` (policy) and `resources_server`:

```yaml
multi_turn_agent:
  responses_api_agents:
    multi_turn_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: ???
      model_server:
        type: responses_api_models
        name: policy_model
      user_model_server:
        type: responses_api_models
        name: user_model
      max_turns: 5
      max_steps_per_turn: null
```

### Turn-Based Loop

The `run()` method implements a turn-based loop:

1. **Seed** the resources server session.
2. **Start** with the initial user message from the input data.
3. **Alternate** between:
   - **Policy turn** — call the policy `model_server` with the full conversation so far. The policy may also execute tool calls within a turn, following the same pattern as `SimpleAgent`.
   - **User turn** — call the `user_model_server` with a user-model system prompt and the full conversation so far to generate the next user message.
4. **Stop** when any of the following criteria are met:
   - `max_turns` reached
   - User model emits a configurable stop signal
   - Max context length for the policy model is reached
   - The policy model indicates completion (modeled after `simple_agent/app.py` stop criteria)
5. **Verify** the full conversation via the resources server.

```python
# Multi-Turn Agent - pseudocode
class MultiTurnAgent:
    async def run(self, task_data):
        # 1. Initialize episode
        resources_server.seed_session(task_data)
        conversation = task_data.prompt

        # 2. Multi-turn loop
        for turn in range(max_turns):
            # Policy turn (may include tool-call steps)
            policy_output = self.responses(conversation, task_data.tools)
            conversation.append(policy_output)

            if is_terminal(policy_output):
                break

            if turn < max_turns - 1:
                # User model turn
                user_message = user_model_server.responses(
                    user_system_prompt + conversation
                )
                conversation.append(user_message)

                if user_signals_done(user_message):
                    break

        # 3. Grade the result
        reward = resources_server.verify(conversation, task_data.ground_truth)
        return conversation, reward
```

### Cookie and Token Propagation

- **Cookies** are propagated through all hops (policy model, user model, resources server) per NeMo Gym conventions.
- **Token IDs and log probs** from policy model responses are propagated into the output trajectory for RL training. User model tokens are **not** needed for training — only policy model tokens are included.

## Open Questions

- Should the user model have access to tools?
- Where should the user system prompt be specified (YAML config, `verifier_metadata`, or both)?

## Out of Scope

- **User model fine-tuning / co-training** — this agent trains and evaluates only the policy model; the user model is treated as a fixed simulator.
- **Async/parallel turns** — turns are strictly sequential (user then policy); no concurrent generation.
- **Reward shaping per turn** — verification happens once after all turns complete; per-turn intermediate rewards are not in scope.
- **Changes to `SimpleAgent`** — the existing single-turn agent remains unchanged; code may be extracted into shared utilities but `SimpleAgent` behavior must not change.

## Acceptance Criteria

- [ ] New agent server at `responses_api_agents/multi_turn_agent/` with standard directory structure (`app.py`, `configs/`, `tests/`, `requirements.txt`, `README.md`)
- [ ] `responses()` handles a single policy turn (model + tool-call loop), propagating cookies and accumulating outputs
- [ ] `run()` orchestrates the full multi-turn dialogue: seed, alternating policy/user turns, verify
- [ ] Cookies are propagated through all hops (policy model, user model, resources server) per Gym conventions
- [ ] Policy model token IDs and log probs are propagated in the output trajectory for RL training
- [ ] User model messages appear in the output as standard input messages (no training metadata needed)
- [ ] Configurable stop criteria: `max_turns` limit, policy completion signal, context length exceeded, and user model stop signal
- [ ] Unit tests using mocked model and resources server endpoints
- [ ] Agent `README.md` documents usage, config options, and the turn-based interaction pattern
- [ ] Pre-commit hooks pass
