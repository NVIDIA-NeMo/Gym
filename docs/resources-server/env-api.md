(env-api)=

# Env API

<!-- TODO: framing on the two definitions of "environment" and when to use Env vs custom agent -->

`Env` is a [Gymnasium](https://gymnasium.farama.org/)-style base class for resources servers. Implement `step()`, optionally `reset()`, pair with `env_agent`.

## Interface

```python
from nemo_gym.envs import Env
from nemo_gym.openai_utils import NeMoGymResponse

class MyEnv(Env):
    async def reset(self, metadata: dict, session_id=None) -> tuple[str | None, dict]:
        # Called once at the start of each episode.
        # Return (initial_observation, info).
        # Return None as the observation to use responses_create_params.input as-is.
        return None, {}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None) -> tuple[str | None, float, bool, bool, dict]:
        # Called after each model response.
        # Return (next_observation, reward, terminated, truncated, info).
        # next_observation: the next prompt to send to the model, or None if the episode is over.
        # reward is only used when terminated=True.
        # truncated=True means the episode hit the step limit.
        ...
```

`metadata` contains the extra fields from `verifier_metadata` in your JSONL input. Access them via `metadata.get("field_name")`.

`session_id` is a unique string per rollout, used to store and retrieve per-episode state. Stateless environments can ignore it.

`reset()` is optional. The default implementation returns `(None, {})`.

## Single-step

Single-step environments are the common case: one model call, then grade the output. Implement `step()` so it always returns `terminated=True`.

```python
class MySingleStepEnv(Env):
    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None):
        response_text = _extract_text(action)
        reward = 1.0 if metadata.get("answer") in response_text else 0.0
        return None, reward, True, False, {}
```

## Tool-using environments

For environments where the model calls tools, the original `SimpleResourcesServer` path is usually a better fit. Tools need to be declared to the model via function schemas in `responses_create_params.tools` (from your JSONL data), and `simple_agent` handles dispatching function calls to named endpoints on the resources server.

`Env` can handle tool dispatch inside `step()` if you need it — parse `action.output` for function calls, execute them, and return results as the next observation — but you still need to provide the tool schemas to the model separately. For most tool-using benchmarks, `SimpleResourcesServer` + `simple_agent` is the simpler path.

## Stateful environments

Use `session_id` to key per-rollout state. Initialize in `reset()`, read in `step()`.

```python
from pydantic import Field

class MyStatefulEnv(Env):
    session_turns: dict = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id=None) -> tuple:
        self.session_turns[session_id] = 0
        return None, {}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None) -> tuple:
        follow_ups = metadata.get("follow_ups", [])
        turn = self.session_turns.get(session_id, 0)

        if turn < len(follow_ups):
            self.session_turns[session_id] = turn + 1
            return follow_ups[turn], 0.0, False, False, {}

        reward = self._grade(action, metadata)
        return None, reward, True, False, {}
```

## YAML configuration

`Env` pairs with `env_agent` instead of `simple_agent`. The agent config references the env server via `env_server`.

```yaml
my_env_instance:
  resources_servers:
    my_env:
      entrypoint: app.py
      domain: knowledge

my_env_agent_instance:
  responses_api_agents:
    env_agent:
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

- `resources_servers/reasoning_gym_env/` -- single-step grading via the reasoning_gym library
- `resources_servers/example_tool_env/` -- tool-using with internal dispatch
- `resources_servers/workplace_assistant_env/` -- stateful, session-scoped tool environments
- `resources_servers/example_multi_turn_env/` -- scripted user simulator with per-session turn tracking
