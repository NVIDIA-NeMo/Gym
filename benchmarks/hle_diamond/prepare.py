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
"""Prepare HLE-Diamond evaluation data for NeMo Gym.

Downloads ``cais/hle-diamond`` (a pinned revision of the 1000-question HLE-Diamond
set: 500 ``reasoning`` and 500 ``knowledge`` questions, 272 of them with an image)
and converts it to Gym JSONL for the ``equivalence_llm_judge`` resources server.

By default every question is kept and each row is fully materialized: the HLE
prompt template is baked into ``responses_create_params.input`` and image
questions carry an ``input_image`` block, so the dataset needs no
``prompt_config`` at rollout time (``config.yaml``). ``--text-only`` keeps the 728
questions without an image as raw fields to be templated at rollout time via
``prompt_config`` (``text.yaml`` and the sandboxed presets).

The corpus is validated before anything is written: a wrong row count, partition
split, image count or duplicated id aborts preparation. ``--limit N`` is the one
supported way to prepare a deliberate subset and bypasses that check loudly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple, Optional

from nemo_gym.vision_input import build_image_input


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "hle_diamond_benchmark.jsonl"
OUTPUT_TEXT_FPATH = DATA_DIR / "hle_diamond_benchmark_text.jsonl"

# Gym's shared HLE prompt (system + user): a paraphrase of the upstream SYSTEM_PROMPT, not a verbatim
# copy (see README > Prompting). The text variant applies the same file at rollout time via prompt_config.
PROMPT_CONFIG_FPATH = BENCHMARK_DIR.parent / "hle" / "prompts" / "default.yaml"

REPO_ID = "cais/hle-diamond"
SPLIT = "test"
REVISION = "04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23"  # pragma: allowlist secret

# Dataset columns read; rationale, author_name, canary and the image previews are never emitted.
SOURCE_COLUMNS = ("id", "question", "image", "answer", "answer_type", "partition", "category", "raw_subject")


class CorpusExpectation(NamedTuple):
    """What a complete download of the pinned revision must contain."""

    rows: int
    partition_counts: dict[str, int]
    image_rows: int


EXPECTED_CORPUS = CorpusExpectation(rows=1000, partition_counts={"reasoning": 500, "knowledge": 500}, image_rows=272)


def validate_limit(limit: Optional[int]) -> None:
    """Reject anything but a positive integer limit; ``None`` means the full set."""
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"--limit must be a positive integer, got {limit!r}")


def validate_corpus(rows: list[dict[str, Any]], expected: Optional[CorpusExpectation] = None) -> None:
    """Fail closed when the loaded corpus is not the complete pinned set (``EXPECTED_CORPUS`` by default)."""
    expected = expected or EXPECTED_CORPUS
    problems = []
    if len(rows) != expected.rows:
        problems.append(f"{len(rows)} rows loaded, expected {expected.rows}")
    partition_counts: dict[str, int] = {}
    for row in rows:
        partition_counts[row["partition"]] = partition_counts.get(row["partition"], 0) + 1
    if partition_counts != expected.partition_counts:
        problems.append(f"partition split {partition_counts} differs from {expected.partition_counts}")
    image_rows = sum(1 for row in rows if row["image"])
    if image_rows != expected.image_rows:
        problems.append(f"{image_rows} image rows, expected {expected.image_rows}")
    unique_ids = len({row["id"] for row in rows})
    if unique_ids != len(rows):
        problems.append(f"{unique_ids} unique ids across {len(rows)} rows")
    if problems:
        raise ValueError(
            f"Refusing to write an incomplete {REPO_ID}@{REVISION[:12]} corpus: "
            + "; ".join(problems)
            + ". Pass --limit N to prepare a deliberate subset."
        )


def load_rows(hf_token: Optional[str]) -> list[dict[str, Any]]:
    """Download the pinned split and return the source columns of every row.

    Arrow iteration keeps the Image columns (``image_preview``, ``rationale_image``)
    undecoded; ``image`` itself is a plain string column holding a data URI or "".
    """
    from datasets import load_dataset

    ds = load_dataset(REPO_ID, split=SPLIT, revision=REVISION, token=hf_token)
    rows: list[dict[str, Any]] = []
    for batch in ds.with_format("arrow").iter(batch_size=500):
        columns = {name: batch.column(name) for name in SOURCE_COLUMNS}
        for i in range(batch.num_rows):
            rows.append({name: column[i].as_py() for name, column in columns.items()})
    return rows


def format_row(row: dict[str, Any], prompt_config: Any = None) -> dict[str, Any]:
    """Convert one source row to Gym JSONL; materialize the prompt when ``prompt_config`` is given."""
    image = row["image"] or ""
    entry = {
        "question": row["question"],
        "expected_answer": row["answer"],
        # Metadata for analysis and slicing; not used for grading.
        "answer_type": row["answer_type"],
        "uuid": row["id"],
        "partition": row["partition"],
        "category": row["category"],
        "raw_subject": row["raw_subject"],
        "has_image": bool(image),
    }
    if prompt_config is not None:
        entry["responses_create_params"] = {"input": build_image_input(prompt_config, row["question"], image)}
    return entry


def prepare(text_only: bool = False, limit: Optional[int] = None) -> Path:
    """Download HLE-Diamond and write the Gym JSONL file.

    Args:
        text_only: Keep only questions without an image and write raw fields for
            ``prompt_config`` templating (``hle_diamond_benchmark_text.jsonl``).
            Otherwise every question is kept with a materialized
            ``responses_create_params.input`` (``hle_diamond_benchmark.jsonl``).
        limit: Keep only the first ``limit`` emitted rows. The only supported subset
            mechanism; it bypasses the completeness check.

    Returns:
        Path to the written JSONL file.
    """
    validate_limit(limit)

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, maybe_get_global_config_dict

    global_config_dict = maybe_get_global_config_dict()
    hf_token = global_config_dict.get(HF_TOKEN_KEY_NAME) if global_config_dict is not None else None

    print(f"Downloading {REPO_ID} (split {SPLIT}, revision {REVISION[:12]}) from HuggingFace...")
    source_rows = load_rows(hf_token)
    if limit is None:
        validate_corpus(source_rows)
    else:
        print(f"WARNING: --limit {limit} given; skipping the corpus completeness check and writing a SUBSET.")

    rows = [row for row in source_rows if not row["image"]] if text_only else source_rows
    if limit is not None:
        rows = rows[:limit]

    prompt_config = None
    if not text_only:
        from nemo_gym.prompt import load_prompt_config

        prompt_config = load_prompt_config(str(PROMPT_CONFIG_FPATH))

    output_fpath = OUTPUT_TEXT_FPATH if text_only else OUTPUT_FPATH
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(format_row(row, prompt_config), ensure_ascii=False) + "\n")

    if text_only:
        # Count from the source rows: after --limit, len(source_rows) - len(rows) would also include the
        # text rows the limit trimmed.
        detail = f"skipped {sum(1 for row in source_rows if row['image'])} image questions"
    else:
        detail = f"{sum(1 for row in rows if row['image'])} with an image"
    if limit is not None:
        detail += f", limited to {limit}"
    print(f"Wrote {len(rows)} problems to {output_fpath} ({detail})")
    return output_fpath


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def add_limit_argument(parser: argparse.ArgumentParser) -> None:
    """``--limit N``, shared with ``prepare_text.py`` so the script form of the text variant honours it too."""
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Prepare only the first N emitted rows (bypasses the corpus completeness check).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HLE-Diamond benchmark data.")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Keep only questions without an image, as raw fields for prompt_config (writes the _text file).",
    )
    add_limit_argument(parser)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    prepare(text_only=args.text_only, limit=args.limit)
