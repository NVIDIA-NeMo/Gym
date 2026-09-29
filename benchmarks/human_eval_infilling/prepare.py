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

"""Prepare HumanEval-Infilling (FIM) benchmark data for NeMo Gym.

Mirrors NeMo-Skills' ``nemo_skills/dataset/human-eval-infilling/prepare.py``:
  - Source: the same gzipped JSONL files served by
    `openai/human-eval-infilling <https://github.com/openai/human-eval-infilling>`_,
    fetched via ``human_eval_infilling.data.read_problems(split)`` (which
    Skills also uses at evaluation time). The HF mirror at
    ``loubnabnl/humaneval_infilling`` is a script-based loader and is no
    longer compatible with modern ``datasets`` versions; the upstream
    raw .jsonl.gz files are byte-identical content-wise.
  - Per-row transform: rename ``prompt`` -> ``prefix``, drop ``entry_point``
    and ``test`` (re-fetched at verify time by the
    ``human_eval_infilling`` library — keeping them here just bloats
    the JSONL), add ``language="python"``, ``split=<subset>``,
    ``comment_delimiter="#"``.

Per-row schema written (per subset):
  - task_id            : str — e.g. ``RandomSpanInfilling/HumanEval/0/2``
  - prefix             : str — code before the missing span (was ``prompt``)
  - suffix             : str — code after the missing span
  - canonical_solution : str — the held-out reference span (informational)
  - language           : "python"
  - split              : "single_line" | "multi_line" | "random_span"
  - comment_delimiter  : "#"
  - verifier_metadata.task_id : str — duplicate of ``task_id`` so the
    ``code_fim`` resource server can look up ``entry_point`` + ``test``
    via ``human_eval_infilling.data.read_problems(split)`` at runtime.

The default benchmark variant is ``random_span`` (the broadest of the
three difficulty levels and Skills' ``EVAL_SPLIT``). ``single_line`` and
``multi_line`` are also produced.
"""

import argparse
import json
from pathlib import Path
from typing import List


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

# Mirrors Skills' `benchmark_file_names` mapping (informational; the
# upstream `human_eval_infilling.data.read_problems` accepts the same
# split keys).
SUPPORTED_SPLITS = (
    "single_line",
    "multi_line",
    "random_span",
    "random_span_light",
)
DEFAULT_SPLITS = ("single_line", "multi_line", "random_span")


def _output_path(split: str) -> Path:
    return DATA_DIR / f"{split}.jsonl"


def _prepare_split(split: str) -> Path:
    """Download one split and write a Gym-format JSONL."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SUPPORTED_SPLITS}")

    from human_eval_infilling.data import read_problems

    print(f"Fetching HumanEval-Infilling :: {split} via human_eval_infilling.data.read_problems...")
    problems = read_problems(split)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _output_path(split)
    rows: List[str] = []
    # Iterate in the upstream `read_problems` order (= insertion order from
    # the .jsonl.gz file). This matches the HF dataset row order Skills'
    # prepare.py inherits, so the produced JSONL is byte-equivalent to
    # Skills' output (modulo Gym's added `verifier_metadata` field).
    for problem in problems.values():
        # Skills' clean_data: prefix = pop("prompt"); add language/split/comment_delimiter;
        # drop entry_point + test (the human_eval_infilling library re-fetches them at
        # eval time using task_id).
        out_row = {
            "task_id": problem["task_id"],
            "prefix": problem["prompt"],
            "suffix": problem["suffix"],
            "canonical_solution": problem["canonical_solution"],
            "language": "python",
            "split": split,
            "comment_delimiter": "#",
            # Required by the code_fim resource server's verify() call.
            "verifier_metadata": {"task_id": problem["task_id"]},
        }
        rows.append(json.dumps(out_row) + "\n")

    with open(out_path, "w") as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} problems to {out_path}")
    return out_path


def prepare(splits: List[str] = list(DEFAULT_SPLITS)) -> Path:
    """Download HumanEval-Infilling subsets and write Gym-format JSONLs.

    Returns the path to the default-variant (``random_span``) JSONL so the
    benchmark framework's post-prepare assertion against ``jsonl_fpath`` in
    ``config.yaml`` can match. The other requested splits are also written
    to disk under ``benchmarks/human_eval_infilling/data/`` and can be
    selected via the resource server's ``split:`` config field.
    """
    paths: List[Path] = []
    for split in splits:
        paths.append(_prepare_split(split))
    # The benchmark framework asserts that prepare() returns the path
    # matching the config's `jsonl_fpath`. Return the default-variant path.
    default_split = "random_span"
    return _output_path(default_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(SUPPORTED_SPLITS),
        help="Which subset(s) to prepare. Default: single_line, multi_line, random_span.",
    )
    args = parser.parse_args()
    prepare(args.splits)
