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

"""Prepare the LeanCat dataset for NeMo Gym.

LeanCat is a 100-task benchmark of formal 1-category-theory statements in Lean 4.
Each task ships as a self-contained file (imports + ``open``/``variable`` preamble
+ a target theorem whose proof is ``sorry``). A task is solved when the model
returns a file that compiles under the pinned toolchain with the target statement
unchanged and no ``sorry``/``admit``/``axiom``/``unsafe`` declarations.

Upstream: https://github.com/sciencraft/LeanCat (paper: arXiv:2512.24796).
Dataset contents are CC BY 4.0; upstream evaluation code is MIT.

Rows are flat (one field per upstream column, no ``responses_create_params``), the same
shape ``benchmarks/minif2f`` uses. The prompt is applied at run time via ``prompt_config``,
so one dataset serves both shipped templates:

    benchmarks/prompts/eval/leancat/paper.yaml           the paper's Appendix D.1 template
                                                         (benchmark default)
    benchmarks/prompts/eval/leancat/upstream-repo.yaml   the upstream repo's own template

LeanCat has no train split. This script writes only the 5-row ``data/example.jsonl``;
``benchmarks/leancat/prepare.py`` writes the 100-row benchmark JSONL from the same
``build_rows``.

Usage:
    python prepare.py                         # -> data/example.jsonl
    python prepare.py --records local.jsonl   # skip the records fetch
"""

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).absolute().parents[2]

# Pinned upstream revision (commit dated 2026-06-19).
LEANCAT_COMMIT = "4e136a13e5"
RAW_BASE = f"https://raw.githubusercontent.com/sciencraft/LeanCat/{LEANCAT_COMMIT}"

RECORDS_URL = f"{RAW_BASE}/data/leancat_records.jsonl"
TARBALL_URL = f"https://codeload.github.com/sciencraft/LeanCat/tar.gz/{LEANCAT_COMMIT}"
# Upstream's own template. `upstream-repo.yaml` transcribes it; the tests refetch this URL
# to check the two still agree.
UPSTREAM_PROMPT_URL = f"{RAW_BASE}/prompts/static_passk.md"

# Both shipped prompts, applied at run time via `prompt_config`.
PROMPT_DIR = Path("benchmarks/prompts/eval/leancat")
PROMPT_CONFIG_PATH = PROMPT_DIR / "paper.yaml"
UPSTREAM_PROMPT_CONFIG_PATH = PROMPT_DIR / "upstream-repo.yaml"

# From configs/evaluation_protocol.json at the same commit; recorded in each row.
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


def load_statement_files(tar_bytes: bytes) -> Dict[str, str]:
    """Read ``CAT_statement/S_<id>.lean`` out of the pinned tarball, bytes untouched.

    Upstream prompts from these files, not from the JSONL, and the two differ in trailing
    whitespace. Sourcing from the file keeps the rendered prompt byte-identical to upstream's.
    """
    statements: Dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not member.isfile() or "/CAT_statement/" not in member.name or not name.endswith(".lean"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            statements[name.removeprefix("S_").removesuffix(".lean")] = handle.read().decode("utf-8")
    if len(statements) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} CAT_statement files, got {len(statements)}")
    return statements


def to_gym_row(record: Dict[str, Any], statements: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Render one upstream record as a flat Gym row.

    ``formal_statement`` is top-level because both the prompt template and the verifier
    read it from there.
    """
    problem_id = record["problem_id"]
    formal_statement = record["formal_statement"]

    if statements is not None:
        from_file = statements[problem_id]
        # The two upstream sources must agree on content; drift fails loudly.
        if from_file.strip() != formal_statement.strip():
            raise ValueError(
                f"Problem {problem_id}: CAT_statement/S_{problem_id}.lean disagrees with the JSONL record"
            )
        formal_statement = from_file

    return {
        "problem_id": problem_id,
        "level": record["level"],
        "tag": record["tag"],
        "domain": record["domain"],
        # The reference file, verbatim: filled into the prompt and used by the verifier.
        "formal_statement": formal_statement,
        # Unused by the static pass@k protocol; needed by upstream's natural-language variants.
        "natural_language_statement": record["natural_language_statement"],
        "lean_toolchain": LEAN_TOOLCHAIN,
        "mathlib_version": MATHLIB_VERSION,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):3d} rows to {path}")


def build_rows(records_raw: Optional[str] = None, use_statement_files: bool = True) -> List[Dict[str, Any]]:
    """Fetch the pinned upstream revision and build all 100 rows.

    Shared by this script and ``benchmarks/leancat/prepare.py``.
    """
    records = load_records(records_raw if records_raw is not None else fetch_text(RECORDS_URL))

    statements = None
    if use_statement_files:
        print(f"Fetching {TARBALL_URL}")
        with urllib.request.urlopen(TARBALL_URL) as response:
            statements = load_statement_files(response.read())
        print(f"Read {len(statements)} CAT_statement/*.lean files")

    return [to_gym_row(record, statements) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        help=f"Local copy of leancat_records.jsonl. Defaults to fetching {RECORDS_URL}",
    )
    parser.add_argument(
        "--no-statement-files",
        action="store_true",
        help=(
            "Take formal_statement from the JSONL instead of the CAT_statement/*.lean files. "
            "Faster, but some prompts then differ from upstream's by a trailing newline."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).absolute().parent / "data",
        help="Directory to write example.jsonl into.",
    )
    args = parser.parse_args()

    rows = build_rows(
        records_raw=args.records.read_text(encoding="utf-8") if args.records else None,
        use_statement_files=not args.no_statement_files,
    )

    # Only the 5-row example set; the benchmark JSONL is written by benchmarks/leancat/prepare.py.
    write_jsonl(args.output_dir / "example.jsonl", rows[:NUM_EXAMPLE_ROWS])

    levels: Dict[str, int] = {}
    for row in rows:
        levels[row["level"]] = levels.get(row["level"], 0) + 1
    print(f"Levels: {levels}")
    print(f"Prompt applied at run time from {PROMPT_CONFIG_PATH} (or {UPSTREAM_PROMPT_CONFIG_PATH}).")
    print(f"Verify under Lean {LEAN_TOOLCHAIN} / Mathlib {MATHLIB_VERSION}.")


if __name__ == "__main__":
    main()
