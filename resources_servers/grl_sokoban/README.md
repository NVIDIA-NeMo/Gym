# GRL Sokoban Resource Server

Single-box Sokoban puzzle environment. The environment is implemented under `resources_servers/grl_sokoban/sokoban_env`, mirroring the Sokoban implementation in the GRL repo (https://github.com/lmgame-org/GRL) and based on code from https://github.com/lmgame-org/lmenv, developed in collaboration with NVIDIA. The implementation uses `gym-sokoban` (https://github.com/mpSchrader/gym-sokoban).

## Why it exists
- **Domain**: Deterministic Sokoban puzzles.
- **Interaction style**: The environment returns a board observation and expects exactly one move per turn.
- **Evaluation**: Reward is accumulated directly through `/step`, so this server should be used with `responses_api_agents/gymnasium_agent`.

## Running
Spin up the server alongside a compatible agent:

```bash
gym env start \
    --model-type openai_model \
    --config responses_api_agents/gymnasium_agent/configs/gymnasium_agent.yaml \
    --resources-server grl_sokoban
```

Collect trajectories:

```bash
gym eval run --no-serve \
    --agent grl_sokoban_gymnasium_agent \
    --input resources_servers/grl_sokoban/data/example.jsonl \
    --output resources_servers/grl_sokoban/data/example_rollouts.jsonl \
    --limit 5
```

## Data
See `generate_test_examples.py` to generate example data.

## Prompt format

The example dataset assumes the model will:

1. Read the initial board returned by `/reset`.
2. Emit exactly one move per turn using an action tag such as `<action>Up</action>`.
3. Continue until the environment terminates or the agent reaches `max_steps`.
