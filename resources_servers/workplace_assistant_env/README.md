# Workplace Assistant Env

Multi-step GymnasiumServer with tool dispatch. Reimplements `workplace_assistant` using the Gymnasium API. Tools are loaded per session and executed inside `step()`.

Uses example data from `resources_servers/workplace_assistant/data/` (symlinked). No additional data provided.

## Run

```bash
ng_run "+config_paths=[resources_servers/workplace_assistant_env/configs/workplace_assistant_env.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=workplace_assistant_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/workplace_assistant_env/data/example.jsonl \
    +output_jsonl_fpath=results/workplace_assistant_env_rollouts.jsonl
```
