# Gymnasium

`GymnasiumServer` is a [Gymnasium](https://gymnasium.farama.org/)-style base class for resources servers. Implement `step()`, optionally `reset()`, and use with `gymnasium_agent`. Not a standalone server.

```python
from resources_servers.base_gymnasium import GymnasiumServer
```

## Interface

```python
from resources_servers.base_gymnasium import GymnasiumServer
from nemo_gym.openai_utils import NeMoGymResponse

class MyEnv(GymnasiumServer):
    async def reset(self, metadata: dict, session_id=None) -> tuple[str | None, dict]:
        return None, {}  # (observation, info); observation (if set) is appended to input

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None) -> tuple[str | None, float, bool, bool, dict]:
        ...  # (observation, reward, terminated, truncated, info)
```

`reset()` runs once per episode. `step()` runs after each model response and returns the 5-tuple:

- **observation**: next message to the model, or `None` to end the episode.
- **reward**: per-step reward; the agent sums across steps by default.
- **terminated**: episode ended naturally (task solved, game over).
- **truncated**: episode cut short (step limit, timeout).
- **info**: extra metadata the env wants to return (debug info, scores, stats). Also how the env sends `tool_outputs` to the agent (see Tool Use).

Only `step()` is required. The default `reset()` returns `(None, {})`, meaning the prompt from the dataset is used as-is. A non-`None` observation from `reset()` is appended to the input as a user message before the first model call.

The three arguments shown in the signatures above:

- **`metadata`**: any extra fields from the JSONL row (e.g. `ground_truth`, `category`). Use for task config or scoring references. Access via `metadata.get("field")`.
- **`session_id`**: unique string per rollout. Use as a key into `self.session_state` for per-episode state (game boards, conversation history, etc.). Stateless envs can ignore it.
- **`action`**: the model's `NeMoGymResponse` for the current turn. Use `extract_text(action)` for text or iterate `action.output` for structured items like `function_call`.

## Single-Step

Single-step environments are the common non-agentic case: one model call, then grade the output. Implement `step()` so it always returns `terminated=True`.

```python
class MySingleStepEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        response_text = extract_text(action)
        reward = 1.0 if metadata.get("answer") in response_text else 0.0
        return None, reward, True, False, {}
```

## Multi-Step with Action Tags

Multiple model calls per episode without native tool calling. The model uses `<action>` tags in its output; `step()` parses them and returns the next observation or terminates.

```python
import re

class BlackjackEnv(GymnasiumServer):
    async def reset(self, metadata, session_id=None):
        hand = deal_hand()
        self.session_state[session_id] = hand
        return f"Your hand: {hand}. <action>hit</action> or <action>stand</action>?", {}

    async def step(self, action, metadata, session_id=None):
        text = extract_text(action)
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

## Tool Use

For tool-calling environments, `step()` inspects `action.output` for items with `type == "function_call"`, executes them, and returns per-call outputs in `info["tool_outputs"]`. The agent synthesizes proper `function_call_output` items tied to each `call_id`, so the model sees the tool_call/response structure it was trained on. Tool schemas go in `responses_create_params.tools` in your JSONL so the model knows what tools are available.

```python
import json

class MyToolEnv(GymnasiumServer):
    async def reset(self, metadata, session_id=None):
        self.session_state[session_id] = initialize(metadata)
        return None, {}

    async def step(self, action, metadata, session_id=None):
        tool_calls = [o for o in action.output if o.type == "function_call"]

        if tool_calls:
            fns = self.session_state[session_id]["functions"]
            outputs = [self.tool_output(c, fns[c.name](**json.loads(c.arguments))) for c in tool_calls]
            return None, 0.0, False, False, {"tool_outputs": outputs}

        reward = self._grade(action, metadata)
        return None, reward, True, False, {}
```

`self.tool_output(call, result)` is a helper on `GymnasiumServer` that builds the `{"call_id", "output"}` dict the agent expects (JSON-serializes the result for you).

## Multi-Turn

`step()` returns the next user message as the observation. The `gymnasium_agent` appends it to the conversation and calls the model again. Return `None` as the observation to end.

```python
class MyMultiTurnEnv(GymnasiumServer):
    async def reset(self, metadata, session_id=None):
        self.session_state[session_id] = {"turn": 0}
        return None, {}

    async def step(self, action, metadata, session_id=None):
        follow_ups = metadata.get("follow_ups", [])
        state = self.session_state[session_id]

        if state["turn"] < len(follow_ups):
            msg = follow_ups[state["turn"]]
            state["turn"] += 1
            return msg, 0.0, False, False, {}

        reward = self._grade(action, metadata)
        return None, reward, True, False, {}
```

## LLM-as-Judge

Use `step()` to call a judge model through `self.server_client` and score the output. The judge model must be configured as a separate model server.

```python
class MyJudgeEnv(GymnasiumServer):
    judge_server: str = "judge_model"  # name of the model server in YAML

    async def step(self, action, metadata, session_id=None):
        response_text = extract_text(action)
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

## Reward Model

Same pattern. Call a reward model endpoint and use its score directly.

```python
class MyRewardModelEnv(GymnasiumServer):
    rm_server: str = "reward_model"

    async def step(self, action, metadata, session_id=None):
        resp = await self.server_client.post(
            server_name=self.rm_server,
            url_path="/v1/score",
            json={"input": metadata.get("prompt"), "response": extract_text(action)},
        )
        score = (await resp.json()).get("score", 0.0)
        return None, score, True, False, {}
```

## YAML Configuration

`GymnasiumServer` pairs with `gymnasium_agent` instead of `simple_agent`. Same shape as the `simple_agent` config, with the agent referencing the environment through `resources_server`.

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
      resources_server:
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

- [`blackjack`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/blackjack): multi-step game with action tags.
