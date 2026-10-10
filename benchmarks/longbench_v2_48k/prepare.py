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
"""Prepare the LongBench v2 benchmark dataset (every row of the split).

Row construction lives in the resources server's ``prepare_longbench.py``. One
pass writes both the full file and the 48k subset under ``benchmarks/longbench/data``;
this script returns the full one and ``prepare_48k.py`` returns the subset.
"""

from pathlib import Path
from typing import Optional

from resources_servers.longbench_v2_48k.prepare_longbench import DEFAULT_MAX_PROMPT_TOKENS
from resources_servers.longbench_v2_48k.prepare_longbench import prepare as prepare_longbench


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
FULL_FPATH = DATA_DIR / "longbench_benchmark.jsonl"
BUDGET_FPATH = DATA_DIR / "longbench_48k_benchmark.jsonl"


def build(truncate_tokenizer: Optional[str] = None, max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS) -> None:
    """Write both benchmark files."""
    prepare_longbench(
        truncate_tokenizer=truncate_tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        output_full=str(FULL_FPATH),
        output_48k=str(BUDGET_FPATH),
    )


def prepare(truncate_tokenizer: Optional[str] = None, max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS) -> Path:
    """Build the benchmark files and return the path of the full one."""
    build(truncate_tokenizer=truncate_tokenizer, max_prompt_tokens=max_prompt_tokens)
    return FULL_FPATH


if __name__ == "__main__":
    prepare()
