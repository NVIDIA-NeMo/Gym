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

"""Prepare LongCodeBench (LongCodeQA) evaluation data for NeMo Gym.

LongCodeBench is a multi-choice QA benchmark over long code contexts, ported
1-to-1 from the NeMo Skills `longcodebench` dataset. Each row's `question`
field is the long code prompt plus the postfix that instructs the model to
emit `Answer: \\boxed{X}`. The shared `benchmarks/prompts/generic/default.yaml`
template (`user: "{question}"`) wraps it as a single user message, mirroring
Skills' `prompt_format=openai` behaviour.

The resulting Gym JSONL is consumed by the `mcqa` resource server with
`grading_mode=strict_single_letter_boxed`. We provide empty-text option dicts
purely to populate the server's `allowed_letters` set; the option text is not
used for grading because the postfix forces a `\\boxed{X}` answer.

Defaults: tokenizer ``o200k_base`` (tiktoken) for the ``n_tokens``
field, with no length filter. For an N3 1M-context
variant that filters to fit, see ``prepare_n3_1m.py`` and
``config_n3_1m.yaml``.

Invocation
----------

``gym eval prepare`` calls ``prepare()`` with no arguments, using
the defaults below. To build a custom variant, run this script
directly::

    python benchmarks/longcodebench/prepare.py \\
        --tokenizer_name cl100k_base \\
        --max_context_tokens 131072
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Callable, Optional

import tiktoken


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_FPATH = DATA_DIR / "longcodebench_benchmark.jsonl"
OPTION_LETTERS = ("A", "B", "C", "D")

POSTFIX = (
    "\n\nThe last line of your response should be in the following format: "
    "'Answer: \\boxed{A/B/C/D}' (e.g. 'Answer: \\boxed{A}')."
)

DEFAULT_TOKENIZER_NAME = "o200k_base"
DEFAULT_MAX_CONTEXT_TOKENS: Optional[int] = None  # no filter by default


def _build_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    """Return a ``text -> token_count`` function.

    Tries ``tiktoken.get_encoding`` first; falls back to
    ``transformers.AutoTokenizer`` for HuggingFace model ids.
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
    """Download LongCodeBench LongCodeQA from HuggingFace and write Gym JSONL."""
    from datasets import load_dataset

    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading LongCodeBench LongCodeQA (tokenizer='{tokenizer_name}') ...")
    ds = load_dataset("json", data_files="hf://datasets/Steefano/LCB/LongCodeQA.zip")
    data = ds["train"]

    count_tokens = _build_token_counter(tokenizer_name)

    # Empty-text option dicts: the mcqa server only consumes the option *keys*
    # for `strict_single_letter_boxed` grading; option text is irrelevant since
    # the prompt postfix forces the model to emit `\boxed{<letter>}`.
    options = [{letter: ""} for letter in OPTION_LETTERS]

    kept = 0
    skipped = 0
    rows = []
    for entry in data:
        question = entry["prompt"].strip() + POSTFIX
        n_tokens = count_tokens(question)
        if max_context_tokens is not None and n_tokens > max_context_tokens:
            skipped += 1
            continue

        row = {
            "question": question,
            "options": options,
            "expected_answer": entry["correct_letter"],
            "grading_mode": "strict_single_letter_boxed",
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, question)),
            "repo": entry["repo"],
            "prompt_goal": entry["prompt_goal"],
            "is_hard": entry["is_hard"],
            "n_tokens": n_tokens,
        }
        rows.append(json.dumps(row) + "\n")
        kept += 1

    with open(output_fpath, "w", encoding="utf-8") as f:
        f.writelines(rows)

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
            "Tokenizer used for the n_tokens count and length filter. "
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
            "Drop samples whose tokenized prompt exceeds this many tokens. "
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
