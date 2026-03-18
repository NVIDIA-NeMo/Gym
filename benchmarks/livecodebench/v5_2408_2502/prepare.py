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
"""LiveCodeBench v5 (Aug 2024 – Feb 2025, 279 problems).

Matches Skills' test_v5_2408_2502 split. Uses the pre-prepared code_gen validation
dataset from HuggingFace (test cases from the official LCB runner).
"""

from pathlib import Path

from benchmarks.livecodebench.prepare_utils import prepare_from_hf_validation


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "livecodebench_v5_validation.jsonl"


def prepare() -> Path:
    return prepare_from_hf_validation(OUTPUT_FPATH, date_from="2024-08-01", date_to="2025-03-01")


if __name__ == "__main__":
    prepare()
