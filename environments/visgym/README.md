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

## Task data

The canonical config uses the committed five-row Maze2D example; no preparation
command or external data download is required. Supply inline rows or JSONL files
for other tasks using `resources_servers/visgym/task_data.py`. Rows specify the
provider environment ID, seed, native kwargs, assets and a finite horizon.

Curriculum builders, synthetic assets and HF reset conversion are separate
experiments. The reusable adapter uses native rewards and rejects experimental
reward shaping. Maze3D and Fetch require separate renderer qualification and
are not enabled by this integration.

## Run

```bash
gym env start --environment visgym --model-type vllm_model
```

See `resources_servers/visgym/README.md` for the lifecycle, action schemas,
native rewards, pinned dependencies and qualification limits.

## Why VisGym has a specialized server and agent

The base Gymnasium server and agent are useful for environments with string
observations and function-call actions. VisGym additionally requires
multimodal image messages, boxed text-action parsing, an explicit final reward
drain, and preservation of prompt/generation token metadata across every turn
for on-policy RL. Adapting the generic pair would change those public schemas,
so VisGym shares the same NeMo Gym base classes while keeping its specialized
transport layer.
