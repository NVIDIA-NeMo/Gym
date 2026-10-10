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
"""Prepare the LongBench v2 48k benchmark dataset.

Same pass as ``prepare.py``; returns the subset whose untruncated prompt is
under 48000 tokens.
"""

from pathlib import Path

from benchmarks.longbench_v2_48k.prepare import BUDGET_FPATH, build


def prepare() -> Path:
    """Build the benchmark files and return the path of the 48k subset."""
    build()
    return BUDGET_FPATH


if __name__ == "__main__":
    prepare()
