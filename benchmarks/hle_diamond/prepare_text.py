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
"""Prepare the text-only variant of HLE-Diamond.

Thin wrapper around ``benchmarks.hle_diamond.prepare.prepare`` with ``text_only=True``:
the 728 questions without an image are written as raw fields to
``benchmarks/hle_diamond/data/hle_diamond_benchmark_text.jsonl`` for rollout-time
templating via ``prompt_config`` (``text.yaml`` and the sandboxed presets).
"""

import argparse
from pathlib import Path
from typing import Optional

from benchmarks.hle_diamond.prepare import add_limit_argument
from benchmarks.hle_diamond.prepare import prepare as _prepare_hle_diamond


def prepare(limit: Optional[int] = None) -> Path:
    """Prepare the HLE-Diamond text-only dataset. Returns the written JSONL path."""
    return _prepare_hle_diamond(text_only=True, limit=limit)


def main(argv: Optional[list[str]] = None) -> Path:
    """Script entry point: only ``--limit N`` is accepted; anything else is an error, never ignored."""
    parser = argparse.ArgumentParser(description="Prepare the text-only HLE-Diamond benchmark data.")
    add_limit_argument(parser)
    args = parser.parse_args(argv)
    return prepare(limit=args.limit)


if __name__ == "__main__":
    main()
