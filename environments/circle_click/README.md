# Circle Click

Environment for training VLMs to click images accurately. Uses images with colored circles on a white background and verifies that the model clicks the correct one. Image size, circle size, number of circles is configurable. Binary success reward.

# Running
Set `env.yaml`: 
```
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-VL-8B-Instruct
```

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct -tp 8 --enable-auto-tool-choice --tool-call-parser hermes &
gym env start --environment circle_click --model-type vllm_model &
gym eval run --no-serve --agent circle_click_simple_agent --input environments/circle_click/data/example.jsonl --output environments/circle_click/data/example_rollouts.jsonl --limit 1
```

# Generating Data
All data is synthetically generated using `prepare.py`.

The data generation script can be modified to arbitrarily control the task complexity and curriculum, including number and size of circles, size of images, or other modifications.
```bash
python3 environments/circle_click/prepare.py --n 1000 --out environments/circle_click/data/train.jsonl
```