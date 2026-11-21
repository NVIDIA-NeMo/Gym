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
"""Build the LongBench v2 Gym dataset from ``THUDM/LongBench-v2``.

Dataset: https://huggingface.co/datasets/THUDM/LongBench-v2

One invocation writes two JSONL files: every row, and the subset whose rendered
prompt fits a token budget.

Two tokenizers do two different jobs, and they are set separately:

* ``--tokenizer`` measures prompt length to pick the budgeted subset. It must be
  the same for every model, or the models are not scored on the same rows. The
  default keeps 151 of the 503 rows.
* ``--truncate-tokenizer`` measures prompt length for truncation, and must be
  the tokenizer of the model that will answer. It defaults to ``--tokenizer``.

Every prompt longer than ``--max-prompt-tokens`` keeps the first and last half
of that budget and drops the middle. Contexts here reach 5.1M tokens, so
without this roughly half of the full split would exceed any current context
window. The subset is chosen from untruncated lengths, so ``--truncate-tokenizer``
never changes which rows land in the budgeted file.

Requires the ``datasets`` and ``transformers`` packages at prep time only.

Usage::

    python resources_servers/longbench/prepare_longbench.py --split train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional


HF_DATASET = "THUDM/LongBench-v2"
DEFAULT_SPLIT = "train"
DEFAULT_TOKENIZER = "google/gemma-4-E4B-it"
DEFAULT_MAX_TOKENS = 48000
# The answer budget (11144) plus chat-template overhead has to fit alongside the
# prompt in a 131072-token window.
DEFAULT_MAX_PROMPT_TOKENS = 119800

# The template has no trailing newline; the rendered prompt ends at the period.
PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

REQUIRED_COLUMNS = (
    "_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
    "context",
)


def render_prompt(row: dict[str, Any]) -> str:
    """Substitute a row's stripped context, question and choices into the template."""
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("$DOC$", str(row["context"]).strip())
    prompt = prompt.replace("$Q$", str(row["question"]).strip())
    for placeholder, column in (
        ("$C_A$", "choice_A"),
        ("$C_B$", "choice_B"),
        ("$C_C$", "choice_C"),
        ("$C_D$", "choice_D"),
    ):
        prompt = prompt.replace(placeholder, str(row[column]).strip())
    return prompt


def truncate_token_ids(token_ids: list[int], max_len: int) -> list[int]:
    """Keep the head and tail halves of ``max_len`` tokens, dropping the middle.

    The tail slice negates before dividing, so an odd ``max_len`` puts the extra
    token in the tail half and the result is still exactly ``max_len`` long.
    """
    if len(token_ids) <= max_len:
        return token_ids
    return token_ids[: max_len // 2] + token_ids[-max_len // 2 :]


def to_task(row: dict[str, Any], prompt: Optional[str] = None) -> dict[str, Any]:
    """Build one dataset row. Only top-level scalars reach verify(), so the
    choice list rides along under ``verifier_metadata``. The gold letter is
    written in both places, because a caller may forward ``verifier_metadata``
    but drop the top-level ``expected_answer``."""
    gold = str(row["answer"]).strip().upper()
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": render_prompt(row) if prompt is None else prompt}]
        },
        "expected_answer": gold,
        "_id": str(row["_id"]),
        "domain": str(row["domain"]),
        "sub_domain": str(row["sub_domain"]),
        "difficulty": str(row["difficulty"]),
        "length": str(row["length"]),
        "verifier_metadata": {
            "expected_answer": gold,
            "choices": [
                {"A": str(row["choice_A"]).strip()},
                {"B": str(row["choice_B"]).strip()},
                {"C": str(row["choice_C"]).strip()},
                {"D": str(row["choice_D"]).strip()},
            ],
        },
    }


def check_columns(column_names: Iterable[str]) -> None:
    """Abort on schema drift rather than emitting rows with missing fields."""
    missing = [name for name in REQUIRED_COLUMNS if name not in set(column_names)]
    if missing:
        raise SystemExit(f"{HF_DATASET} is missing expected columns: {', '.join(missing)}")


def load_split(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            f"The 'datasets' package is required to build {HF_DATASET}. "
            "Install it (pip install datasets) or stage a prepared JSONL."
        ) from exc
    return load_dataset(HF_DATASET, split=split)


def load_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "The 'transformers' package is required to measure prompt length. Install it (pip install transformers)."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    server_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Build the LongBench v2 Gym dataset.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split (default: train)")
    parser.add_argument(
        "--output-full",
        default=str(server_dir / "data" / "longbench_full.jsonl"),
        help="Output JSONL path for every row",
    )
    parser.add_argument(
        "--output-48k",
        dest="output_48k",
        default=str(server_dir / "data" / "longbench_48k.jsonl"),
        help="Output JSONL path for rows within the token budget",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer that selects the budgeted subset (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--truncate-tokenizer",
        default=None,
        help="Tokenizer of the answering model, used for truncation (default: --tokenizer)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Token budget for the budgeted file (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=f"Longest prompt written to either file (default: {DEFAULT_MAX_PROMPT_TOKENS})",
    )
    return parser


def prepare(
    split: str = DEFAULT_SPLIT,
    tokenizer: str = DEFAULT_TOKENIZER,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    truncate_tokenizer: Optional[str] = None,
    output_full: Optional[str] = None,
    output_48k: Optional[str] = None,
) -> Path:
    """Write both dataset files and return the path of the full one.

    One pass writes every row to the full file and the rows under the token
    budget to the budgeted file. Callers that declare a dataset path must match
    the returned path, so the full file is the one returned.
    """
    data_dir = Path(__file__).parent / "data"
    full_path = Path(output_full) if output_full else data_dir / "longbench_full.jsonl"
    budget_path = Path(output_48k) if output_48k else data_dir / "longbench_48k.jsonl"

    dataset = load_split(split)
    check_columns(dataset.column_names)

    budget_tok = load_tokenizer(tokenizer)
    truncate_id = truncate_tokenizer or tokenizer
    # One encode per row when both jobs use the same tokenizer; contexts reach
    # 5.1M tokens, so the second pass is worth avoiding.
    same_tokenizer = truncate_id == tokenizer
    truncate_tok = budget_tok if same_tokenizer else load_tokenizer(truncate_id)

    for path in (full_path, budget_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    full_count = 0
    budget_count = 0
    truncated_count = 0
    with full_path.open("w", encoding="utf-8") as full_fh, budget_path.open("w", encoding="utf-8") as budget_fh:
        for row in dataset:
            row = dict(row)
            prompt = render_prompt(row)

            # The subset is chosen from the untruncated length, so the answering
            # model's tokenizer cannot change which rows it holds.
            budget_ids = budget_tok.encode(prompt)
            within_budget = len(budget_ids) < max_tokens

            truncate_ids = budget_ids if same_tokenizer else truncate_tok.encode(prompt)
            if len(truncate_ids) > max_prompt_tokens:
                prompt = truncate_tok.decode(
                    truncate_token_ids(truncate_ids, max_prompt_tokens), skip_special_tokens=True
                )
                truncated_count += 1

            line = json.dumps(to_task(row, prompt), ensure_ascii=False) + "\n"
            full_fh.write(line)
            full_count += 1
            if within_budget:
                budget_fh.write(line)
                budget_count += 1

    print(
        f"LongBench v2: subset tokenizer {budget_tok.name_or_path}, truncation tokenizer {truncate_tok.name_or_path}"
    )
    print(f"LongBench v2: wrote {full_count} rows -> {full_path}")
    print(f"LongBench v2: kept {budget_count} of {full_count} rows (< {max_tokens} tokens) -> {budget_path}")
    print(f"LongBench v2: truncated {truncated_count} of {full_count} prompts to {max_prompt_tokens} tokens")
    return full_path


def main() -> None:
    args = build_parser().parse_args()
    prepare(
        split=args.split,
        tokenizer=args.tokenizer,
        max_tokens=args.max_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        truncate_tokenizer=args.truncate_tokenizer,
        output_full=args.output_full,
        output_48k=args.output_48k,
    )


if __name__ == "__main__":
    main()
