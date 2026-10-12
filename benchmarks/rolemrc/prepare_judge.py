# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare the RoleMRC gym-native benchmark dataset (LLM-as-judge mode).

The judge split is the subset of RoleMRC rows whose ``task`` has a 5-aspect judge
config (see ``_EVALUATION_CONFIG`` in the resources server's ``app.py``). Both
splits come out of the same ``prepare_rolemrc.py`` run; see ``prepare.py`` for
the shared helpers.
"""

from pathlib import Path

from benchmarks.rolemrc.prepare import _SRC_JUDGE, DATA_DIR, build


OUTPUT_FPATH = DATA_DIR / "rolemrc_judge_benchmark.jsonl"

# Agent that runs this benchmark (see config_judge.yaml).
_BENCHMARK_AGENT = "rolemrc_judge_benchmark_simple_agent"


def prepare() -> Path:
    """Build the gym-native RoleMRC judge benchmark JSONL (judge subset, tagged)."""
    return build(_SRC_JUDGE, OUTPUT_FPATH, _BENCHMARK_AGENT, "RoleMRC (judge)")


if __name__ == "__main__":
    prepare()
