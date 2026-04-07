# Blackjack Env

Blackjack implemented as a Gymnasium-style `Env`. The model hits or stands until the hand ends.

Reward: +1 win, 0 draw, -1 loss or bust.

## Run

```bash
ng_run "+config_paths=[resources_servers/blackjack_env/configs/blackjack_env.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=blackjack_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/blackjack_env/data/example.jsonl \
    +output_jsonl_fpath=results/blackjack_rollouts.jsonl
```
