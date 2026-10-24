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
"""Prepare the GraphWalks benchmark data.

Source: https://huggingface.co/datasets/openai/graphwalks

Ported from:
    https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/graphwalks/prepare.py

Two upstream-prompt corrections from Skills are preserved here verbatim:

  1. The BFS prompt is rewritten to disambiguate "depth N" — without
     this rewrite, models often return nodes at intermediate depths.
  2. The parents prompt sometimes includes the target node inside its
     own answer set; we strip it.

Defaults: tokenizer ``o200k_base`` (tiktoken) for the ``n_tokens``
field, with no length filter. For an N3 1M-context
variant that filters to fit, see ``prepare_n3_1m.py`` and
``config_n3_1m.yaml``.

Invocation
----------

``gym eval prepare`` calls ``prepare()`` with no arguments, using
the defaults below. To build a custom variant, run this script
directly::

    python benchmarks/graphwalks/prepare.py \\
        --tokenizer_name meta-llama/Llama-3.1-8B-Instruct \\
        --max_context_tokens 131072
"""

import argparse
import json
import re
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Optional

import tiktoken
from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_FPATH = DATA_DIR / "graphwalks_benchmark.jsonl"

DEFAULT_TOKENIZER_NAME = "o200k_base"
DEFAULT_MAX_CONTEXT_TOKENS: Optional[int] = None  # no filter by default
_worker_token_counter: Optional[Callable[[str], int]] = None
_worker_tokenizer_name: Optional[str] = None

_BFS_PATTERN = re.compile(r"Perform a BFS from node (\S+) with depth (\d+)")
_BFS_REPLACEMENT = (
    r"Perform a BFS from node \1 and return only the nodes at exactly depth \2 "
    r"(not nodes at intermediate depths)"
)
_PARENTS_PATTERN = re.compile(r"Find the parents of node ([^\s.]+)\.")


def _build_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    """Return a ``text -> token_count`` function.

    Tries ``tiktoken.get_encoding`` first; if the name isn't a tiktoken
    encoding, falls back to ``transformers.AutoTokenizer``.
    """
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except ValueError:
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return lambda text: len(hf_tokenizer.encode(text, add_special_tokens=False))


def _initialize_worker(tokenizer_name: str) -> None:
    global _worker_token_counter, _worker_tokenizer_name
    _worker_token_counter = None
    _worker_tokenizer_name = tokenizer_name


def _prepare_entry(entry: dict) -> tuple[dict, bool]:
    global _worker_token_counter
    if _worker_token_counter is None:
        if _worker_tokenizer_name is None:
            raise RuntimeError("GraphWalks preparation worker has no tokenizer name")
        # Task initialization propagates failures instead of respawning failed workers.
        _worker_token_counter = _build_token_counter(_worker_tokenizer_name)
    prompt_text = entry["prompt"]
    answer_nodes = list(entry["answer_nodes"])

    # Skills fix #1: disambiguate BFS depth before counting tokens.
    prompt_text = _BFS_PATTERN.sub(_BFS_REPLACEMENT, prompt_text)

    # Skills fix #2: strip the queried node from its own parents answer.
    match = _PARENTS_PATTERN.search(prompt_text)
    target = match.group(1) if match else None
    cleaned_self_parent = target is not None and target in answer_nodes
    if cleaned_self_parent:
        answer_nodes.remove(target)

    return {
        "responses_create_params": {"input": [{"role": "user", "content": prompt_text}]},
        "expected_answer": json.dumps(sorted(answer_nodes)),
        "problem_type": entry["problem_type"],
        "n_tokens": _worker_token_counter(prompt_text),
        "prompt_chars": entry["prompt_chars"],
    }, cleaned_self_parent


def prepare(
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    max_context_tokens: Optional[int] = DEFAULT_MAX_CONTEXT_TOKENS,
    output_fpath: Path = DEFAULT_OUTPUT_FPATH,
    *,
    num_workers: int = 1,
) -> Path:
    """Prepare rows in source order with one tokenizer per worker process."""
    if type(num_workers) is not int or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openai/graphwalks", split="train")
    if num_workers == 1:
        _initialize_worker(tokenizer_name)
    # Spawn avoids inheriting initialized tokenizer thread pools from a caller.
    workers = (
        get_context("spawn").Pool(num_workers, initializer=_initialize_worker, initargs=(tokenizer_name,))
        if num_workers > 1
        else nullcontext()
    )

    kept = 0
    skipped_tokens = 0
    skipped_self_parent = 0
    with workers as pool, output_fpath.open("w", encoding="utf-8") as fout:
        samples = pool.imap(_prepare_entry, dataset, chunksize=1) if pool else map(_prepare_entry, dataset)
        for sample, cleaned_self_parent in tqdm(samples, total=len(dataset), desc="Preparing GraphWalks"):
            skipped_self_parent += int(cleaned_self_parent)
            if max_context_tokens is not None and sample["n_tokens"] > max_context_tokens:
                skipped_tokens += 1
                continue

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

    cap_str = "none" if max_context_tokens is None else str(max_context_tokens)
    print(
        f"Wrote {kept} samples to {output_fpath} "
        f"(tokenizer={tokenizer_name}, cap={cap_str}; "
        f"dropped {skipped_tokens} over cap; cleaned {skipped_self_parent} self-parent answers)"
    )
    return output_fpath


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer_name",
        default=DEFAULT_TOKENIZER_NAME,
        help=(
            "Tokenizer used for token counting. Accepts a tiktoken encoding name "
            "(e.g. 'cl100k_base', 'o200k_base') or a HuggingFace model id "
            "(e.g. 'meta-llama/Llama-3.1-8B-Instruct'). "
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
        "--num_workers",
        type=int,
        default=1,
        help="Tokenizer worker processes; output retains source order. Default: 1",
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
    prepare(
        tokenizer_name=args.tokenizer_name,
        max_context_tokens=cap,
        output_fpath=args.output_fpath,
        num_workers=args.num_workers,
    )
