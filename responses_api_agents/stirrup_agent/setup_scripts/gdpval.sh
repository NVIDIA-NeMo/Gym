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
# Download the GDPVal dataset from HuggingFace and convert it to the benchmark
# JSONL format consumed by ``ng_e2e_collect_rollouts``.
#
# Output: benchmarks/gdpval/data/gdpval_benchmark.jsonl (220 tasks).
#
# Requires:
#   - Active nemo-gym venv (uv sync --extra dev)
#   - HF_TOKEN env var if the dataset requires authentication

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

ng_prepare_benchmark "+config_paths=[benchmarks/gdpval/config.yaml]"

echo "Done. Output: benchmarks/gdpval/data/gdpval_benchmark.jsonl"
