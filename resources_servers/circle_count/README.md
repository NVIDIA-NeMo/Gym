<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Circle Count

Environment for training VLMs to count in images. Uses images with colored circles on a white background and verifies that the model reports the correct count for the target color. Image size, circle size, number of circles, and color distribution are configurable. Binary success reward.

Expects `\boxed{}` output format. No tools.

# Running
Set `env.yaml`:
```
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-VL-8B-Instruct
```

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct -tp 8 --enable-auto-tool-choice --tool-call-parser hermes &
ng_run "+config_paths=[resources_servers/circle_count/configs/circle_count.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" &
ng_collect_rollouts +agent_name=circle_count_simple_agent +input_jsonl_fpath=resources_servers/circle_count/data/example.jsonl +output_jsonl_fpath=resources_servers/circle_count/data/example_rollouts.jsonl +limit=1
```

# Generating Data
All data is synthetically generated using `generate_data.py`.

The script controls task complexity including number of circles, their size, image size, and color distribution.

```bash
python3 resources_servers/circle_count/generate_data.py --n 5 --out resources_servers/circle_count/data/example.jsonl
python3 resources_servers/circle_count/generate_data.py --n 1000 --out resources_servers/circle_count/data/train.jsonl
```

Key parameters:
- `--num-circles-min` / `--num-circles-max`: total circles per image (default 5–20)
- `--radius-min` / `--radius-max`: circle radius in pixels (default 30–60)
- `--img-size-min` / `--img-size-max`: image dimensions (default 1000×1000)
