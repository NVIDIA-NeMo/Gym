#!/usr/bin/env python3
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
"""Count tokens in rollout JSONL files under a directory."""

import argparse
from pathlib import Path

from resources_servers.arena.scripts.count_rollout_tokens import report_rollout_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tokenizer", default="gpt-4o")
    parser.add_argument("--filename-suffix", default="evaluator_rollouts.jsonl")
    args = parser.parse_args()
    paths = sorted(args.directory.expanduser().rglob(f"*{args.filename_suffix}"))
    if not paths:
        parser.error("No matching JSONL files found")
    for path in paths:
        report_rollout_tokens(path, args.tokenizer)


if __name__ == "__main__":
    main()
