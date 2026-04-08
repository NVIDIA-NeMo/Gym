# Example Multi-Turn Env

Multi-turn gymnasium-style  Replays scripted follow-up questions from verifier_metadata. Turn index tracked per session.

Example data provided in `data/example.jsonl`. No train/validation data.

## Run

```bash
ng_run "+config_paths=[resources_servers/example_multi_turn_env/configs/example_multi_turn_env.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=example_multi_turn_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/example_multi_turn_env/data/example.jsonl \
    +output_jsonl_fpath=results/example_multi_turn_env_rollouts.jsonl
```
