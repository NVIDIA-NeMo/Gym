# Env Abstraction Design

**Status**: Implemented (draft)
**Author**: cmunley

---

## Problem

Adding a new benchmark to NeMo-Gym currently requires understanding the full microservice architecture: resources server, agent server, YAML wiring, HTTP endpoints, cookie propagation. This is too much overhead for the common case. Users reinvent the same patterns (single-step grading, multi-turn conversations, game environments) over and over.

---

## Goals

- One `Env` base class following the Gymnasium API
- Users implement `step()` and optionally `reset()`. One class, one YAML stanza.
- Philosophically correct: `Env` = the RL environment; the LLM is the RL agent
- Non-goal: replace `SimpleResourcesServer` / `SimpleResponsesAPIAgent`. Advanced users and tool-using environments keep using those directly.

---

## Design

### Single Env class, Gymnasium interface

```python
class Env(SimpleResourcesServer):

    async def reset(self, metadata: dict, session_id=None) -> tuple[str | None, dict]:
        """Return (initial_observation, info). None = use responses_create_params as-is."""
        return None, {}

    @abstractmethod
    async def step(self, action: NeMoGymResponse, metadata: dict, session_id=None) -> tuple[str | None, float, bool, bool, dict]:
        """Return (observation, reward, terminated, truncated, info)."""
        ...
```

Exposes two HTTP endpoints: `POST /reset` and `POST /step`. No `/verify` or `/seed_session`.

`session_id` is extracted from the request by the base class and passed through. Stateless envs ignore it. Stateful envs (games, tool environments) use it to key per-rollout state.

### EnvAgent

A single generic agent (`responses_api_agents/env_agent`) drives the Gymnasium loop:

1. Call `/reset` on env server
2. Call model with observation
3. Call `/step` with model response
4. If `terminated` or `truncated`: return reward
5. Else: append observation to conversation, go to 2

Accumulates token IDs, usage, and outputs across all turns. Returns `EnvRunResponse(BaseVerifyResponse)` compatible with the training framework.

### When to use Env vs SimpleResourcesServer

Use `Env` when you are writing a new environment and the episode logic fits the step/reset pattern: single-step grading, multi-turn conversations, games, user simulators.

Use `SimpleResourcesServer` + `simple_agent` when you need named tool endpoints that the agent dispatches to (the model calls tools via function calling and the agent routes the calls). Tool-using benchmarks are better served by this path because the model needs function schemas, and `simple_agent` handles the dispatch.

Use a custom agent server when the agent IS the environment (SWE-RL, verifier-style agents with complex execution loops).

---

## Architecture

```
Old path:  SimpleResourcesServer (/seed_session + /verify)  <-->  simple_agent / langgraph_agent / ...
New path:  Env (/reset + /step)                             <-->  EnvAgent
```

Both paths coexist. Existing servers are unchanged.

YAML config for the new path:
```yaml
my_env:
  resources_servers:
    my_env:
      entrypoint: app.py
      domain: games
my_env_agent:
  responses_api_agents:
    env_agent:
      entrypoint: app.py
      env_server: {type: resources_servers, name: my_env}
      model_server: {type: responses_api_models, name: policy_model}
      max_steps: 10
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_env/data/example.jsonl
```

---

## What was considered but not built

**Multiple Env subclasses** (MultiStepEnv, ToolEnv, MultiTurnEnv, JudgeEnv, RewardModelEnv). Early iterations split envs into separate base classes by execution pattern. Collapsed into a single `Env` class after adopting the Gymnasium API, since `step()` is general enough to cover all patterns. Judge and reward model patterns are just implementations of `step()` that call another model.

**Config expansion** (`envs:` stanza auto-expanding to resources server + agent pair). Deferred. Currently users write two YAML stanzas. The expansion can be added to the head server later.

**`environments/` folder** combining agent + resources server as a unit. Deferred pending the env location decision.

---

## Examples

| Server | Pattern |
|---|---|
| `reasoning_gym_env` | Single-step grading |
| `workplace_assistant_env` | Stateful tool dispatch (session-scoped) |
| `example_multi_turn_env` | Scripted user simulator |
| `blackjack_env` | Multi-step game with action tags |
| `tictactoe_env` | Multi-turn adversarial game |

---

## Open Questions

1. **Env location** — should Env-based servers live in `resources_servers/` (current), in a new `environments/` folder, or as `env.py` alongside `app.py` in existing server folders?
2. **Config expansion** — should `envs:` in YAML auto-generate the agent pairing, or keep the explicit two-stanza approach?
3. **Observation injection** — EnvAgent currently injects `step()` observations as user messages. For tool-using envs, tool results should be formatted as function_call_output items. Worth adding a response format field to `EnvStepResponse`?
