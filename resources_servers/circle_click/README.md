# Circle Click

Environment for training VLMs to click images accurately. Uses images with colored circles on a white background and verifies that the model clicks the correct one. Image size, circle size, number of circles is configurable. Binary success reward.

# Running
Set `env.yaml`: 
```
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-VL-2B-Instruct
```

There is a bug in hermes tool parser causing crashes with many concurrent calls, so this particular vllm command may fail sporadically. TODO: switch example model 
```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct -tp 8 --enable-auto-tool-choice --tool-call-parser hermes &
ng_run "+config_paths=[resources_servers/circle_click/configs/circle_click.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" &
ng_collect_rollouts +agent_name=circle_click_simple_agent +input_jsonl_fpath=resources_servers/circle_click/data/example.jsonl +output_jsonl_fpath=resources_servers/circle_click/data/example_rollouts.jsonl
```

# Generating Data
The generate data script can be modified to arbitrarily control the task complexity and curriculum.
```bash
python3 resources_servers/circle_click/generate_data.py --n 1000 --out resources_servers/circle_click/data/train.jsonl