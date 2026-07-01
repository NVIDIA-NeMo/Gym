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
"""Prepare LongBench-v2 data for NeMo Gym.

Mirrors `nemo_skills/dataset/longbench-v2/prepare.py`: loads the same
HuggingFace dataset (`THUDM/LongBench-v2`, single "train" split that
holds all 503 evaluation questions), preserves every Skills field
(`index`, `context`, `question`, `choice_A..D`, `expected_answer`,
`domain`, `sub_domain`, `difficulty`, `length`, `context_tokens`),
and additionally emits the `options` list and `grading_mode` that the
existing `mcqa` resource server consumes for grading.

LongBench v2 covers 6 long-context domains (8k-2M words):
single-doc QA, multi-doc QA, long in-context learning, long-dialogue
history, code-repo understanding, long structured data.

Dataset: https://huggingface.co/datasets/THUDM/LongBench-v2
Paper:   https://arxiv.org/abs/2412.15204

Defaults: tokenizer ``o200k_base`` (tiktoken) for the
``context_tokens`` field, with no length filter. For an N3 1M-context
variant that filters to fit, see ``prepare_n3_1m.py`` and
``config_n3_1m.yaml``.

Invocation
----------

``gym eval prepare`` calls ``prepare()`` with no arguments, using
the defaults below. To build a custom variant, run this script
directly::

    python benchmarks/longbench_v2/prepare.py \\
        --tokenizer_name cl100k_base \\
        --max_context_tokens 131072
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import tiktoken
from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_FPATH = DATA_DIR / "longbench_v2_benchmark.jsonl"

DEFAULT_TOKENIZER_NAME = "o200k_base"
DEFAULT_MAX_CONTEXT_TOKENS: Optional[int] = None  # no filter by default


def _build_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    """Return a ``text -> token_count`` function.

    Tries ``tiktoken.get_encoding`` first; if the name isn't a tiktoken
    encoding, falls back to ``transformers.AutoTokenizer``. The tiktoken
    path uses ``disallowed_special=()`` because LongBench-v2 contexts
    sometimes contain raw ``<|endoftext|>`` strings that tiktoken would
    otherwise refuse to encode.
    """
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except ValueError:
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return lambda text: len(hf_tokenizer.encode(text, add_special_tokens=False))


def prepare(
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    max_context_tokens: Optional[int] = DEFAULT_MAX_CONTEXT_TOKENS,
    output_fpath: Path = DEFAULT_OUTPUT_FPATH,
) -> Path:
    """Download LongBench-v2, convert to Gym JSONL, return the output file path."""
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading THUDM/LongBench-v2 (split='train', tokenizer='{tokenizer_name}') ...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    count_tokens = _build_token_counter(tokenizer_name)

    kept = 0
    skipped = 0
    with open(output_fpath, "w", encoding="utf-8") as out:
        for entry in tqdm(dataset, desc=f"Writing {output_fpath.name}"):
            context_tokens = count_tokens(entry["context"])
            if max_context_tokens is not None and context_tokens > max_context_tokens:
                skipped += 1
                continue

            record = {
                # Fields preserved verbatim from Skills' prepare.py
                "index": entry["_id"],
                "context": entry["context"],
                "question": entry["question"],
                "choice_A": entry["choice_A"],
                "choice_B": entry["choice_B"],
                "choice_C": entry["choice_C"],
                "choice_D": entry["choice_D"],
                "expected_answer": entry["answer"],
                "domain": entry["domain"],
                "sub_domain": entry["sub_domain"],
                "difficulty": entry["difficulty"],
                "length": entry["length"],
                "context_tokens": context_tokens,
                # Gym-side additions consumed by the `mcqa` resource server.
                # mcqa's verify() reads `options`, `expected_answer`, `grading_mode`.
                "options": [
                    {"A": entry["choice_A"]},
                    {"B": entry["choice_B"]},
                    {"C": entry["choice_C"]},
                    {"D": entry["choice_D"]},
                ],
                "grading_mode": "strict_single_letter_boxed",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    cap_str = "none" if max_context_tokens is None else str(max_context_tokens)
    print(
        f"Wrote {kept} problems to {output_fpath} "
        f"(tokenizer={tokenizer_name}, cap={cap_str}; dropped {skipped} over cap)"
    )
    return output_fpath


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer_name",
        default=DEFAULT_TOKENIZER_NAME,
        help=(
            "Tokenizer used for the context_tokens count and length filter. "
            "Accepts a tiktoken encoding name (e.g. 'cl100k_base', 'o200k_base') "
            "or a HuggingFace model id (e.g. 'meta-llama/Llama-3.1-8B-Instruct'). "
            f"Default: {DEFAULT_TOKENIZER_NAME}"
        ),
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help=(
            "Drop samples whose tokenized context exceeds this many tokens. "
            "Omit (or pass a negative number) for no filter. "
            f"Default: {DEFAULT_MAX_CONTEXT_TOKENS}"
        ),
    )
    parser.add_argument(
        "--output_fpath",
        type=Path,
        default=DEFAULT_OUTPUT_FPATH,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT_FPATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cap = args.max_context_tokens if (args.max_context_tokens is None or args.max_context_tokens >= 0) else None
    prepare(tokenizer_name=args.tokenizer_name, max_context_tokens=cap, output_fpath=args.output_fpath)
