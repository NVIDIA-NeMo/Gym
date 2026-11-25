#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Print the context limit of the SynClaw release pinned in resources_servers/synclaw/pyproject.toml,
# for serving the model with the same limit the agent falls back to:
#
#   vllm serve ... --max-model-len "$(bash environments/synclaw/max_model_len.sh)"
set -euo pipefail

pyproject="$(dirname "$0")/../../resources_servers/synclaw/pyproject.toml"
exec uv run --quiet --no-project --python 3.13 --with "$(grep -o 'synclaw @ [^"]*' "$pyproject")" \
    python -c 'from synclaw.rollout.policy import DEFAULT_MAX_MODEL_LEN; print(DEFAULT_MAX_MODEL_LEN)'
