# TicTacToe Env

Multi-turn gymnasium-style  Model plays as X, opponent (O) is an LLM via user_model_server. Falls back to first empty cell on invalid opponent moves. Game state managed per session.

Example data provided in `data/example.jsonl` (system prompt only, no verifier_metadata needed). No train/validation data.

## Run

```bash
ng_run "+config_paths=[resources_servers/tictactoe_env/configs/tictactoe_env.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=tictactoe_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/tictactoe_env/data/example.jsonl \
    +output_jsonl_fpath=results/tictactoe_rollouts.jsonl
```
