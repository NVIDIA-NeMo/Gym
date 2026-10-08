# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare the vision (multimodal) variant of HLE.

Thin wrapper around ``benchmarks.hle.prepare.prepare`` with ``include_vision=True``.
The full HLE split (text + image questions) is downloaded and every row is fully
materialized (image questions carry an ``input_image`` block), written to
``benchmarks/hle/data/hle_benchmark_vision.jsonl``.
"""

from pathlib import Path

from benchmarks.hle.prepare import prepare as _prepare_hle


def prepare() -> Path:
    """Prepare the HLE vision dataset. Returns the written JSONL path."""
    return _prepare_hle(include_vision=True)


if __name__ == "__main__":
    prepare()
