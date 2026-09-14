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
"""Summarize an LMArena benchmark prompt blend using the scoring taxonomy."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken
from omegaconf import OmegaConf

from resources_servers.arena.taxonomy import PROMPT_CATEGORY_ORDER, get_prompt_categories


PROMPT_PATHS = {
    "lmarena_v2": Path("benchmarks/lmarena_v2/data/lmarena_v2_validation.jsonl"),
    "lmarena_v3": Path("benchmarks/lmarena_v3/data/lmarena_v3_validation.jsonl"),
}
CONFIG_PATHS = {name: Path(f"resources_servers/arena/configs/{name}.yaml") for name in PROMPT_PATHS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=PROMPT_PATHS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    with PROMPT_PATHS[args.benchmark].open(encoding="utf-8") as f:
        lines = (json.loads(line) for line in f if line.strip())
        prompts = list(itertools.islice(lines, args.limit)) if args.limit else list(lines)
    if not prompts:
        raise ValueError("No prompts found")

    config = OmegaConf.load(CONFIG_PATHS[args.benchmark])[args.benchmark]["resources_servers"]["arena"]
    encoding = tiktoken.encoding_for_model(config.tokenizer_model)
    category_counts = Counter()
    prompt_tokens = []
    user_turns = []
    for prompt in prompts:
        messages = prompt["responses_create_params"]["input"]
        # tiktoken accepts text, so join all message content in the conversation.
        text = "\n".join(message["content"] for message in messages)
        prompt_tokens.append(len(encoding.encode(text)))
        user_turns.append(sum(message["role"] == "user" for message in messages))
        category_counts.update(get_prompt_categories(prompt))  # Categories may overlap.

    tokens = np.asarray(prompt_tokens)
    turns = np.asarray(user_turns)
    print(f"prompts: {len(prompts)}")
    print(
        f"prompt_tokens: median={round(np.median(tokens))} mean={round(np.mean(tokens))} "
        f"p05={round(np.percentile(tokens, 5))} p95={round(np.percentile(tokens, 95))}"
    )
    print(
        f"user_turns: single_turn={np.mean(turns == 1):.1%} multi_turn={np.mean(turns > 1):.1%} "
        f"p95={round(np.percentile(turns, 95))}"
    )
    print()
    print("Taxonomy categories overlap; percentages do not sum to 100%.")
    print("| Category | Prompts | Share |")
    print("|---|---:|---:|")
    for category in PROMPT_CATEGORY_ORDER:
        if category not in category_counts:
            continue
        count = category_counts[category]
        print(f"| {category} | {count} | {count / len(prompts):.1%} |")


if __name__ == "__main__":
    main()
