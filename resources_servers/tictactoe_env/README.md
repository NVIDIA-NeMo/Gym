# TicTacToe Env

TicTacToe implemented as a Gymnasium-style `Env`. The model plays as X against a rule-based opponent that takes winning moves when available.

Reward: +1 win, 0 draw, -1 loss.

Move format: `row R col C` (R and C are 1-3).

## Run

```bash
ng_run "+config_paths=[resources_servers/tictactoe_env/configs/tictactoe_env.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=tictactoe_env_agent \
    +input_jsonl_fpath=resources_servers/tictactoe_env/data/example.jsonl \
    +output_jsonl_fpath=results/tictactoe_rollouts.jsonl
```
