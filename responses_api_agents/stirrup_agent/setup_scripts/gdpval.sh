#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Download the GDPVal dataset from HuggingFace and convert it to the JSONL
# format expected by the Stirrup agent's rollout collection.
#
# Output: responses_api_agents/stirrup_agent/data/gdpval.jsonl (220 tasks).
#
# Requires:
#   - Active nemo-gym venv (uv sync --extra dev)
#   - HF_TOKEN env var if the dataset requires authentication

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="${OUTPUT:-responses_api_agents/stirrup_agent/data/gdpval.jsonl}"
DATASET="${DATASET:-openai/gdpval}"
SPLIT="${SPLIT:-train}"

cd "${REPO_ROOT}"
python3 scripts/prepare_gdpval_dataset.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}"

echo "Done. Output: ${OUTPUT}"
