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
"""Prepare the BrowseComp rows for harnesses that get the tools from the Resources Server.

The rows match ``prepare.py``'s, the same tasks and prompts, but carry no tool list, which session harnesses such as
OpenCode reject in the request. They reach ``search``, ``browse`` and ``bash_command`` through the Resources Server's
MCP grant instead (``opencode.yaml``).
"""

from pathlib import Path

from benchmarks.browsecomp.prepare import DATA_DIR, write_benchmark


OUTPUT_FPATH = DATA_DIR / "browsecomp_benchmark_harness_neutral.jsonl"


def prepare() -> Path:
    return write_benchmark(OUTPUT_FPATH, tools=None)


if __name__ == "__main__":
    prepare()
