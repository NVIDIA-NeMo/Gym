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

"""Prepare the LeanCat dataset for NeMo Gym.

LeanCat is a 100-task benchmark of formal 1-category-theory statements in Lean 4.
Each task ships as a self-contained file (imports + ``open``/``variable`` preamble
+ a target theorem whose proof is ``sorry``). A task is solved when the model
returns a file that compiles under the pinned toolchain with the target statement
unchanged and no ``sorry``/``admit``/``axiom``/``unsafe`` declarations.

Upstream: https://github.com/sciencraft/LeanCat (paper: arXiv:2512.24796).
Dataset contents are CC BY 4.0; upstream evaluation code is MIT.

The conversion script lives here rather than in the source repo because the source
repo is third-party and not ours to modify -- the same exception under which
``math_formal_lean/prepare_minif2f.py`` is kept in-tree.

Usage:
    python prepare_leancat.py                      # fetch pinned revision, write data/
    python prepare_leancat.py --records local.jsonl --prompt prompts/static-passk.md
"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence


# Pinned upstream revision (commit dated 2026-06-19). Bump deliberately, not
# incidentally: another revision can change statements, difficulty labels, or the
# prompt, none of which are detectable from the JSONL alone.
LEANCAT_COMMIT = "4e136a13e5d0b94829c813e6f612fd991e670096"
RAW_BASE = f"https://raw.githubusercontent.com/sciencraft/LeanCat/{LEANCAT_COMMIT}"

RECORDS_URL = f"{RAW_BASE}/data/leancat_records.jsonl"
PROMPT_URL = f"{RAW_BASE}/prompts/static_passk.md"

# From configs/evaluation_protocol.json at the same commit. Recorded in each row so
# a rollout carries the toolchain it is only meaningful under.
LEAN_TOOLCHAIN = "leanprover/lean4:v4.19.0"
MATHLIB_VERSION = "v4.19.0"

EXPECTED_RECORDS = 100
NUM_EXAMPLE_ROWS = 5


def fetch_text(url: str) -> str:
    print(f"Fetching {url}")
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def load_records(raw: str) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} LeanCat records, got {len(records)}")
    return records


def render_prompt(template: str, formal_statement: str) -> str:
    """Fill the upstream template's single ``{formal_statement}`` placeholder.

    Upstream renders with ``str.format`` (``scripts/eval_common.py:render_prompt``),
    so we do too. Note the asymmetry: the *template* is formatted, never the
    statement, so braces inside Lean code can never be interpreted as fields.
    """
    return template.format(formal_statement=formal_statement)


def to_gym_row(record: Dict[str, Any], template: str) -> Dict[str, Any]:
    formal_statement = record["formal_statement"]

    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": render_prompt(template, formal_statement)}],
        },
        "verifier_metadata": {
            "problem_id": record["problem_id"],
            "level": record["level"],
            "tag": record["tag"],
            "domain": record["domain"],
            # The reference file, verbatim. The verifier needs it to confirm the
            # model did not weaken, rename, or drop hypotheses from the target.
            "formal_statement": formal_statement,
            # Unused by the static pass@k protocol (which is formal-input only),
            # but required by the natural-language and LeanBridge variants.
            "natural_language_statement": record["natural_language_statement"],
            "lean_toolchain": LEAN_TOOLCHAIN,
            "mathlib_version": MATHLIB_VERSION,
        },
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):3d} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        help=f"Local copy of leancat_records.jsonl. Defaults to fetching {RECORDS_URL}",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help=f"Local copy of the static pass@k prompt. Defaults to fetching {PROMPT_URL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).absolute().parent / "data",
        help="Directory to write train.jsonl and example.jsonl into.",
    )
    args = parser.parse_args()

    template = (args.prompt.read_text(encoding="utf-8") if args.prompt else fetch_text(PROMPT_URL)).strip()
    if "{formal_statement}" not in template:
        raise ValueError("Prompt template does not contain the {formal_statement} placeholder")

    raw_records = args.records.read_text(encoding="utf-8") if args.records else fetch_text(RECORDS_URL)
    records = load_records(raw_records)
    rows = [to_gym_row(record, template) for record in records]

    # Keep a copy of the exact prompt we rendered with, so a reviewer can diff it
    # against upstream without re-running the fetch.
    # Hyphenated, unlike the upstream filename: Gym's `no-underscore-md` pre-commit hook
    # rejects underscores in Markdown names. Contents are byte-identical to upstream.
    prompt_path = Path(__file__).absolute().parent / "prompts" / "static-passk.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(template + "\n", encoding="utf-8")
    print(f"Wrote prompt template to {prompt_path}")

    write_jsonl(args.output_dir / "train.jsonl", rows)
    write_jsonl(args.output_dir / "example.jsonl", rows[:NUM_EXAMPLE_ROWS])

    levels: Dict[str, int] = {}
    for row in rows:
        level = row["verifier_metadata"]["level"]
        levels[level] = levels.get(level, 0) + 1
    print(f"Levels: {levels}")
    print(f"Verify under Lean {LEAN_TOOLCHAIN} / Mathlib {MATHLIB_VERSION}.")


if __name__ == "__main__":
    main()
