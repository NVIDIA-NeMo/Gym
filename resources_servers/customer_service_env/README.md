# Customer Service Env

Multi-turn gymnasium-style server with asymmetric tool access. The policy model is a support agent that sees only conversation text. The user model is a simulated customer with private tools (lookup_order, check_account, get_policy) that the policy never sees.

All scenario data is in the JSONL. Generate with:

```bash
python resources_servers/customer_service_env/scripts/generate_data.py --num 500 --seed 42 --output resources_servers/customer_service_env/data/train.jsonl
```

Example data provided in `data/example.jsonl` (5 entries).

## Run

```bash
ng_run "+config_paths=[resources_servers/customer_service_env/configs/customer_service_env.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=customer_service_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/customer_service_env/data/example.jsonl \
    +output_jsonl_fpath=results/customer_service_rollouts.jsonl
```
