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

"""Prepare the CombiBench "with solution" split.

The published answers are already substituted into each statement and only the
proof is missing (the paper's second setting). Gym allows one benchmark dataset
per workload, so this setting is its own benchmark; everything else is shared
with ``benchmarks/combibench/prepare.py``.
"""

from pathlib import Path

from benchmarks.combibench import prepare as base


OUTPUT_FPATH = Path(__file__).parent / "data" / "combibench_test_with_solution.jsonl"


def prepare(**kwargs) -> Path:
    kwargs.setdefault("split", "test_with_solution")
    kwargs.setdefault("output", OUTPUT_FPATH)
    return base.prepare(**kwargs)


if __name__ == "__main__":
    prepare()
