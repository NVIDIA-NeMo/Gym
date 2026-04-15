# Blackjack Env

Multi-step gymnasium-style environment. 

Model hits or stands using `<action>` tags until the hand ends. Game state managed per session.

Example data provided in `data/example.jsonl` (system prompt only, no verifier_metadata needed). No train/validation data.

## Run

```bash
ng_run "+config_paths=[resources_servers/blackjack/configs/blackjack.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=blackjack_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/blackjack/data/example.jsonl \
    +output_jsonl_fpath=results/blackjack_rollouts.jsonl
```
