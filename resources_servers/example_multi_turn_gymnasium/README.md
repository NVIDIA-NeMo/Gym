# Example Multi-Turn Env

Reference example for multi-turn environments using Gymnasium API standard. 

`reset()` runs once at the start of an episode to initialize state and return an inital observation, which in the current implementation would be appended to any input messages in the dataset. `step()` runs after each model response and returns `(observation, reward, terminated, truncated, info)`. If `observation` is non-None, the agent appends it as a user message and calls the model again. When `terminated` is True or max steps is reached the episode ends.

This example replays scripted follow-up questions from `verifier_metadata`, then checks the final answer. Each `step()` either returns the next follow-up (reward 0, not done) or checks whether `expected_answer` appears in the final response and terminates. In the example data, getting the final answer correct requires answering all intermediate questions correctly.

`verifier_metadata` fields:
- `follow_ups` - list of follow-up messages sent after each model turn
- `expected_answer` - substring expected in the final model response

Example data provided in `data/example.jsonl`.

## Run

```bash
ng_run "+config_paths=[resources_servers/example_multi_turn_gymnasium/configs/example_multi_turn_gymnasium.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=example_multi_turn_gymnasium_agent \
    +input_jsonl_fpath=resources_servers/example_multi_turn_gymnasium/data/example.jsonl \
    +output_jsonl_fpath=results/example_multi_turn_rollouts.jsonl
```
