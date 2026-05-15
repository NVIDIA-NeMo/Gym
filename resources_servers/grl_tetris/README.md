# GRL Tetris Resource Server

GRL Tetris environment in Gymnasium style. The model emits one or more `<action>Left|Right|Down</action>` tags per turn, the env applies them sequentially, breaking on game-over. Inner environment logic lives under `resources_servers/grl_tetris/tetris_env` and is a standalone adaptation of the upstream GRL implementation.

## Why it exists
- **Domain**: Classic falling-block Tetris on a configurable grid.
- **Interaction style**: Gymnasium API (`reset` + `step` returning `(obs, reward, terminated, truncated, info)`). Multiple actions can be batched into a single model turn.
- **Evaluation**: Reward accumulates per game step; `terminated=True` when the game ends. Pair with `responses_api_agents/gymnasium_agent`.
- **Independence**: No runtime dependency on the GRL repository. The environment is vendored and self-contained.

## Running
Start NeMo Gym servers

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
responses_api_agents/gymnasium_agent/configs/gymnasium_agent.yaml,\
resources_servers/grl_tetris/configs/grl_tetris.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```bash
ng_collect_rollouts +agent_name=grl_tetris_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/grl_tetris/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/grl_tetris/data/example_rollouts.jsonl \
    +limit=5
```


## Licensing
- Code: Apache 2.0
- Data: Apache 2.0
