(env-api)=

# Env API

<!-- TODO: framing on the two definitions of "environment" and when to use Env vs custom agent -->

`GymnasiumServer` is a [Gymnasium](https://gymnasium.farama.org/)-style base class for resources servers. Implement `step()`, optionally `reset()`, pair with `gymnasium_agent`.

## Interface

```python
from resources_servers.gymnasium import GymnasiumServer
from nemo_gym.openai_utils import NeMoGymResponse

class MyEnv(GymnasiumServer):
    async def reset(self, metadata: dict, session_id=None) -> tuple[str | None, dict]:
        # Called once at the start of each episode.
        # Return (observation, info).
        # None observation = use responses_create_params.input as-is.
        return None, {}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None) -> tuple[str | None, float, bool, bool, dict]:
        # Called after each model response.
        # Return (observation, reward, terminated, truncated, info).
        # observation: next prompt for the model, or None if episode is over.
        # reward is only used when terminated=True.
        # truncated=True means the episode hit the step limit.
        ...
```

`metadata` contains the extra fields from `verifier_metadata` in your JSONL input. Access them via `metadata.get("field_name")`.

`session_id` is a unique string per rollout, used to store and retrieve per-episode state. Stateless environments can ignore it.

`reset()` is optional. The default implementation returns `(None, {})`.

## Single-step

Single-step environments are the common non agentic use case: one model call, then grade the output. Implement `step()` so it always returns `terminated=True`.

```python
class MySingleStepEnv(GymnasiumServer):
    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None):
        response_text = _extract_text(action)
        reward = 1.0 if metadata.get("answer") in response_text else 0.0
        return None, reward, True, False, {}
```

## Multi-step with action tags

Multiple model calls per episode without tool calling. The model uses `<action>` tags in its output, `step()` parses them and returns the next observation or terminates.

```python
import re
from pydantic import Field

class BlackjackEnv(GymnasiumServer):
    session_state: dict = Field(default_factory=dict)

    async def reset(self, metadata, session_id=None):
        hand = deal_hand()
        self.session_state[session_id] = hand
        return f"Your hand: {hand}. <action>hit</action> or <action>stand</action>?", {}

    async def step(self, action, metadata, session_id=None):
        text = _extract_text(action)
        m = re.search(r"<action>\s*(hit|stand)\s*</action>", text, re.IGNORECASE)
        decision = m.group(1).lower() if m else "stand"

        hand = self.session_state[session_id]
        if decision == "hit":
            hand = hit(hand)
            if bust(hand):
                return None, -1.0, True, False, {}
            return f"Your hand: {hand}. <action>hit</action> or <action>stand</action>?", 0.0, False, False, {}

        reward = score_against_dealer(hand)
        return None, reward, True, False, {}
```

## Tool-use

`action` is the full `NeMoGymResponse` from the model server, including any function calls already parsed by the model server's tool call parser. `step()` checks `action.output` for items with `type == "function_call"`, executes them, and returns the results as the next observation. Tool schemas go in `responses_create_params.tools` in your JSONL data so the model knows what tools are available.

```python
import json
from pydantic import Field

class MyToolEnv(GymnasiumServer):
    session_state: dict = Field(default_factory=dict)

    async def reset(self, metadata, session_id=None):
        self.session_state[session_id] = initialize(metadata)
        return None, {}

    async def step(self, action, metadata, session_id=None):
        tool_calls = [o for o in action.output if o.type == "function_call"]

        if tool_calls:
            state = self.session_state[session_id]
            results = []
            for call in tool_calls:
                args = json.loads(call.arguments)
                result = state["functions"][call.name](**args)
                results.append(f"{call.name} -> {result}")
            return "\n".join(results), 0.0, False, False, {}

        reward = self._grade(action, metadata)
        return None, reward, True, False, {}
```

## Multi-turn

`step()` returns the next user message as the observation. The `gymnasium_agent` appends it to the conversation and calls the model again. Return `None` to end.

```python
class MyMultiTurnEnv(GymnasiumServer):
    session_turns: dict = Field(default_factory=dict)

    async def reset(self, metadata, session_id=None):
        self.session_turns[session_id] = 0
        return None, {}

    async def step(self, action, metadata, session_id=None):
        follow_ups = metadata.get("follow_ups", [])
        turn = self.session_turns.get(session_id, 0)

        if turn < len(follow_ups):
            self.session_turns[session_id] = turn + 1
            return follow_ups[turn], 0.0, False, False, {}

        reward = self._grade(action, metadata)
        return None, reward, True, False, {}
```

## LLM-as-judge

Use `step()` to call a judge model via `self.server_client` and score the output. The judge model must be configured as a separate model server.

```python
class MyJudgeEnv(GymnasiumServer):
    judge_server: str = "judge_model"  # name of the model server in YAML

    async def step(self, action, metadata, session_id=None):
        response_text = _extract_text(action)
        judge_input = f"Question: {metadata.get('question')}\nAnswer: {response_text}\nIs this correct? Say YES or NO."
        judge_resp = await self.server_client.post(
            server_name=self.judge_server,
            url_path="/v1/responses",
            json={"input": [{"role": "user", "content": judge_input}]},
        )
        judgment = await judge_resp.json()
        reward = 1.0 if "YES" in str(judgment.get("output_text", "")).upper() else 0.0
        return None, reward, True, False, {}
```

## Reward model

Same pattern. Call a reward model endpoint and use its score directly.

```python
class MyRewardModelEnv(GymnasiumServer):
    rm_server: str = "reward_model"

    async def step(self, action, metadata, session_id=None):
        resp = await self.server_client.post(
            server_name=self.rm_server,
            url_path="/v1/score",
            json={"input": metadata.get("prompt"), "response": _extract_text(action)},
        )
        score = (await resp.json()).get("score", 0.0)
        return None, score, True, False, {}
```

## YAML configuration

`GymnasiumServer` pairs with `gymnasium_agent` instead of `simple_agent`. The agent config references the env server via `env_server`.

```yaml
my_env_instance:
  resources_servers:
    my_env:
      entrypoint: app.py
      domain: knowledge

my_gymnasium_agent_instance:
  responses_api_agents:
    gymnasium_agent:
      entrypoint: app.py
      env_server:
        type: resources_servers
        name: my_env
      model_server:
        type: responses_api_models
        name: policy_model
      max_steps: 10
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_env/data/example.jsonl
```

## Examples

Working implementations are in the repository:

- `resources_servers/reasoning_gym_env/` -- single-step grading
- `resources_servers/workplace_assistant_env/` -- stateful tool dispatch
- `resources_servers/example_multi_turn_env/` -- scripted multi-turn
- `resources_servers/blackjack_env/` -- multi-step game with action tags
- `resources_servers/tictactoe_env/` -- adversarial game
