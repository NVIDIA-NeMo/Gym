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
"""Prepare the SWE-bench Multimodal test split for NeMo Gym.

Writes the test split: 480 JavaScript and TypeScript tasks from eleven repositories, in the row format of
benchmarks/swebench/multimodal/prepare.py (issue images attached as ``input_image`` parts, the test patch's
binary files stored in the row). Each row keeps the dataset's ``image``, ``eval_script``, ``log_parser`` and
``eval_type``, which the benchmark's resources server evaluates it with (``evaluation_spec_source: row``).

The dataset revision is pinned because the grading follows those row fields. A few issue images of the split are
on hosts that no longer serve them; such an image is left out and its URL stays in the issue text.
"""

from pathlib import Path

from benchmarks.swebench.multimodal.prepare import DATA_DIR
from benchmarks.swebench.multimodal.prepare import prepare as prepare_split


OUTPUT_FPATH = DATA_DIR / "swebench_multimodal_test_benchmark.jsonl"
SPLIT = "test"
# SWE-bench/SWE-bench_Multimodal at 2026-08-18: the rows carry swebench 5's evaluation fields.
REVISION = "4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5"


def prepare() -> Path:
    return prepare_split(SPLIT, OUTPUT_FPATH, REVISION, skip_unreachable_issue_images=True)


if __name__ == "__main__":
    prepare()
