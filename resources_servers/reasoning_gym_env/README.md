# Reasoning Gym Env

Single-step GymnasiumServer. Reimplements `reasoning_gym` using the Gymnasium API.

Uses example data from `resources_servers/reasoning_gym/data/`. No additional data provided.

## Run

```bash
ng_run "+config_paths=[resources_servers/reasoning_gym_env/configs/reasoning_gym_env.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=results/reasoning_gym_env_rollouts.jsonl
```
