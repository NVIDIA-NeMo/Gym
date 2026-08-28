<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisGym

VisGym is a stateful visual training environment. A policy receives an image,
emits one text action, and repeats until the task terminates. The canonical
environment config is `environments/visgym/config.yaml`; the implementation is
split between `resources_servers/visgym` and
`responses_api_agents/visgym_agent`.

## Prepare data

Generated training manifests and rendered assets are intentionally excluded
from Git. They are deterministic and can be recreated through one entry point:

```bash
# Ordered 5x5, 7x7, 9x9, and 11x11 maze curriculum
python environments/visgym/prepare.py maze --samples-per-stage 1280

# Multi-environment manifests and their small deterministic fixture assets
python environments/visgym/prepare.py multienv \
  --combine-envs maze_2d_7x7,maze_3d,jigsaw

# Seed manifests derived from the public VisGym Hugging Face dataset
python environments/visgym/prepare.py hf \
  --output-dir resources_servers/visgym/data/hf_manifests
```

Use `python environments/visgym/prepare.py <dataset> --help` for all generator
options. Minimal example and smoke fixtures remain committed so a clean clone
can validate the config and run focused tests without downloading training
data.

## Run

```bash
gym env start --environment visgym --model-type vllm_model
```

See `resources_servers/visgym/README.md` for the lifecycle, action schemas,
reward shaping, optional dependencies, and larger multi-environment recipes.

## Why VisGym has a specialized server and agent

The base Gymnasium server and agent are useful for environments with string
observations and function-call actions. VisGym additionally requires
multimodal image messages, boxed text-action parsing, an explicit final reward
drain, and preservation of prompt/generation token metadata across every turn
for on-policy RL. Adapting the generic pair would change those public schemas,
so VisGym shares the same NeMo Gym base classes while keeping its specialized
transport layer.
