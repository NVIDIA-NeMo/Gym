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
"""Prepare the vision (multimodal) variant of HLE-Verified.

Thin wrapper around ``benchmarks.hle_verified.prepare.prepare`` with
``include_vision=True``, mirroring ``benchmarks/hle/prepare_vision.py``. The Gold +
Revision rows are kept including image questions, and every row is fully materialized
(image questions carry an ``input_image`` block), written to
``benchmarks/hle_verified/data/hle_verified_benchmark_vision.jsonl``.

Evaluating the result needs a vision-capable policy model; the judge stays text-only.
"""

from pathlib import Path

from benchmarks.hle_verified.prepare import prepare as _prepare_hle_verified


def prepare() -> Path:
    """Prepare the HLE-Verified vision dataset. Returns the written JSONL path."""
    return _prepare_hle_verified(include_vision=True)


if __name__ == "__main__":
    prepare()
