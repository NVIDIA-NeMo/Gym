(gymnasium-api)=


# Gymnasium API

`GymnasiumServer` is a [Gymnasium](https://gymnasium.farama.org/)-style base class for resources servers. Implement `step()`, optionally `reset()`, and pair with `gymnasium_agent`.

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

< Back to Resources Server
:::

---

## Interface

```python
from resources_servers.base_gymnasium import GymnasiumServer
from nemo_gym.openai_utils import NeMoGymResponse

class MyEnv(GymnasiumServer):
    async def reset(self, metadata: dict, session_id=None) -> tuple[str | None, dict]:
        # Called once at the start of each episode to initialize the environment
        # Return (observation, info)
        # None observation = use responses_create_params.input as-is.
        return None, {}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None) -> tuple[str | None, float, bool, bool, dict]:
        # Called after each model response, executes any actions taken or implements other env logic (<action> tags, tool calls, user turns, or just single-step verification), computes rewards, determines if done.
        # Return (observation, reward, terminated, truncated, info)
        # observation: next message to model (tool result, rendering of state, user message, and so on), or None if episode is over.
        # reward: per-step reward, accumulated across all steps by the agent.
        # terminated: True when the episode ends naturally (task solved, game over, etc). Tells the agent to stop.
        # truncated: True when the episode is cut short (step limit, timeout, etc).
        # info: arbitrary dict passed through to the final response. Use for diagnostics, scores, metadata.
        ...
```

The main method to implement is `step()`. Implementing a custom `reset()` is optional — use it to initialize environment state or return an opening message from the environment. The default implementation returns `(None, {})`.

`metadata` is the `verifier_metadata` dict from your input JSONL, passed through unchanged. This is where you put task-specific data (expected answers, test cases, board configurations, and so on) that the environment needs for initialization or scoring. Access fields with `metadata.get("field_name")`.

`session_id` is a unique string per rollout. Use it as a key to store per-episode state (such as game boards or conversation history) in a dict on your server. Stateless environments can ignore it.

`action` is the full `NeMoGymResponse` from the model server. Your `step()` parses whatever it needs from it: text content, `<action>` tags, function calls, and so on. Use `extract_text()` for text, or inspect `action.output` directly for structured output like tool calls.

:::{tip}
For the full single-step and multi-step patterns using the standard `SimpleResourcesServer` API, see {doc}`/environment-tutorials/single-step-environment` and {doc}`/environment-tutorials/multi-step-environment`.
:::

---

## Single-Step

Single-step environments are the common non-agentic use case: one model call, then grade the output. Implement `step()` so it always returns `terminated=True`.

```python
class MySingleStepEnv(GymnasiumServer):
    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None):
        response_text = extract_text(action)
        reward = 1.0 if metadata.get("answer") in response_text else 0.0
        return None, reward, True, False, {}
```

---

## Multi-Step with Action Tags

Multiple model calls per episode without native tool calling. The model uses `<action>` tags in its output, `step()` parses them and returns the next observation or terminates.

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

---

## Tool Use

For tool-calling environments, `step()` checks `action.output` for items with `type == "function_call"`, executes them, and returns the results as the next observation. Tool schemas go in `responses_create_params.tools` in your JSONL data so the model knows what tools are available.

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

---

## Multi-Turn

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

---

## LLM-as-Judge

Use `step()` to call a judge model through `self.server_client` and score the output. The judge model must be configured as a separate model server. See {doc}`/environment-tutorials/llm-as-judge-verification` for the full pattern.

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

---

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

---

## YAML Configuration

`GymnasiumServer` pairs with `gymnasium_agent` instead of `simple_agent`. The agent config references the env server through `env_server`.

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

---

## Examples

- [`blackjack`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/blackjack) — Multi-step game with action tags
