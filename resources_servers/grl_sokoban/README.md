# GRL Sokoban Resource Server

Single-box Sokoban puzzle environment adapted to the `GymnasiumServer` interface introduced in PR `#1072`. The environment is implemented under `resources_servers/grl_sokoban/sokoban_env`, mirroring the Sokoban implementation in the GRL repo (https://github.com/lmgame-org/GRL) and based on code from https://github.com/lmgame-org/lmenv, developed in collaboration with NVIDIA. The implementation uses `gym-sokoban` (https://github.com/mpSchrader/gym-sokoban).

## Why it exists
- **Domain**: Deterministic Sokoban puzzles.
- **Interaction style**: The environment returns a board observation and expects exactly one move per turn.
- **Evaluation**: Reward is accumulated directly through `/step`, so this server should be paired with `responses_api_agents/gymnasium_agent`.

## Setup

Please follow the setup instructions in the repo root README or [detailed setup docs](../../docs/get-started/detailed-setup.md).

## Running
Spin up the server alongside a compatible agent:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
responses_api_agents/gymnasium_agent/configs/gymnasium_agent.yaml,\
resources_servers/grl_sokoban/configs/grl_sokoban.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:

```bash
ng_collect_rollouts +agent_name=grl_sokoban_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/example_rollouts.jsonl \
    +limit=5
```

Launch the rollout viewer:

```bash
ng_viewer +jsonl_fpath=resources_servers/grl_sokoban/data/example_rollouts.jsonl
```

## Prompt format

The example dataset assumes the model will:

1. Read the initial board returned by `/reset`.
2. Emit exactly one move per turn using an action tag such as `<action>Up</action>`.
3. Continue until the environment terminates or the agent reaches `max_steps`.

## Tests

```bash
pytest resources_servers/grl_sokoban/tests responses_api_agents/gymnasium_agent/tests
```
