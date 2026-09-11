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
"""Prepare HLE-Verified with text and image questions.

Writes materialized inputs to data/hle_verified_benchmark_vision.jsonl.
Requires a vision-capable policy model for evaluation.
"""

from pathlib import Path

from benchmarks.hle_verified.prepare import prepare as _prepare_hle_verified


def prepare() -> Path:
    """Prepare the HLE-Verified vision dataset. Returns the written JSONL path."""
    return _prepare_hle_verified(include_vision=True)


if __name__ == "__main__":
    prepare()
