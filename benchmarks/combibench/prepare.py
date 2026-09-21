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

"""Prepare CombiBench for NeMo Gym.

Two upstream sources publish the same 100 problems and they disagree:

* ``hf`` — the Hugging Face dataset ``AI-MO/CombiBench`` that upstream's own
  harness loads. Pinned to its last revision (2025-07-13).
* ``github`` — the ``lean/CombiBench/*.lean`` files in the upstream repository,
  which received statement and answer corrections after the dataset was last
  updated and were bumped to Lean v4.24.0. Pinned to the current master.

Twelve ``test`` statements and thirteen ``test_with_solution`` statements differ
between the two once comments are ignored, and the answer lists differ for
``brualdi_ch1_5`` and ``brualdi_ch2_36``; see the benchmark README. Measured
against Mathlib v4.24.0 (the toolchain upstream pins), all 100 GitHub statements
compile while 6 Hugging Face statements do not, so ``github`` is the default.
The source is a preparation argument so a run records which one it measured.

Rows are written without prompts. ``prompt.yaml`` templates ``{formal_statement}``
at rollout time, so the model sees exactly what upstream's harness shows it.
"""

import argparse
import csv
import io
import json
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

HF_DATASET = "AI-MO/CombiBench"
HF_REVISION = "882ba08befd0856f5364db1e53d58c7e2cf704f9"  # pragma: allowlist secret

GITHUB_REPO = "MoonshotAI/CombiBench"
GITHUB_REVISION = "c67e4213597b1477351d9ef5ca37fb622084cc78"  # pragma: allowlist secret
GITHUB_TARBALL_URL = f"https://codeload.github.com/{GITHUB_REPO}/tar.gz/{GITHUB_REVISION}"
DOWNLOAD_TIMEOUT_SECONDS = 120

SPLITS = ("test", "test_with_solution")
SOURCES = ("hf", "github")
# Both splits hold every problem; a shorter file silently changes the
# denominator of every score, so preparation fails closed against this.
EXPECTED_ROWS = {"test": 100, "test_with_solution": 100}
OUTPUT_FPATHS = {split: DATA_DIR / f"combibench_{split}.jsonl" for split in SPLITS}

_ABBREV_SOLUTION_RE = re.compile(r"\babbrev\s+\S+_solution\b")
_BLOCK_COMMENT_RE = re.compile(r"/-[\s\S]*?-/")
_LINE_COMMENT_RE = re.compile(r"^\s*--.*\n", re.MULTILINE)


def positive_int(value: str) -> int:
    """A subset size below one is a mistake, not a request for everything."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"--limit must be a positive integer, got {number}")
    return number


# --------------------------------------------------------------------------- hf


def load_hf_rows(split: str) -> list[dict]:
    """Rows of one split at the pinned revision, as plain dicts."""
    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET, split=split, revision=HF_REVISION)
    rows = []
    for example in dataset:
        answer = example.get("answer")
        rows.append(
            {
                "theorem_name": example["theorem_name"],
                "natural_language": example.get("natural_language"),
                "answer": list(answer) if answer is not None else None,
                "source": example.get("source"),
                "tag": example.get("tag"),
                "formal_statement": example["formal_statement"],
            }
        )
    return rows


# ----------------------------------------------------------------------- github


def fetch_github_tree(cache_dir: Path) -> Path:
    """Download and unpack the pinned repository tarball; return ``lean/CombiBench``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lean_dir = cache_dir / f"CombiBench-{GITHUB_REVISION}" / "lean" / "CombiBench"
    if lean_dir.is_dir():
        return lean_dir

    print(f"Downloading {GITHUB_TARBALL_URL}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        with urllib.request.urlopen(GITHUB_TARBALL_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            tmp.write(response.read())
        tmp.flush()
        with tarfile.open(tmp.name) as tar:
            tar.extractall(cache_dir, filter="data")

    if not lean_dir.is_dir():
        raise SystemExit(f"Expected {lean_dir} in the downloaded archive")
    return lean_dir


def parse_answer_cell(cell: str) -> Optional[list[str]]:
    """``metadata.csv`` stores one answer bare and several as ``[a, b, c]``.

    The only multi-answer problem (``hackmath_6``) has comma-free members, so
    splitting a bracketed cell on ``", "`` is exact for the pinned data. A bare
    cell is never split: function answers contain commas.
    """
    cell = (cell or "").strip()
    if not cell:
        return None
    if cell.startswith("[") and cell.endswith("]"):
        return [part.strip() for part in cell[1:-1].split(", ")]
    return [cell]


def strip_comments(text: str) -> str:
    """Remove doc comments and comment lines the way upstream's HF export did.

    The published dataset carries the statements without their ``/-- ... -/``
    informal docstrings and inline ``--`` notes; the GitHub files keep them.
    Removing them here keeps the prompt shape identical across sources. Runs of
    blank lines left behind are collapsed to one so paragraph splitting in the
    verifier sees the same structure.
    """
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def load_github_rows(split: str, cache_dir: Path) -> list[dict]:
    lean_dir = fetch_github_tree(cache_dir)
    metadata_text = (lean_dir / "metadata.csv").read_text(encoding="utf-8")
    rows = []
    for record in csv.DictReader(io.StringIO(metadata_text)):
        name = record["theorem_name"].strip()
        if split == "test":
            path = lean_dir / f"{name}.lean"
        else:
            path = lean_dir / "with_solution" / f"{name}_sol.lean"
        if not path.is_file():
            raise SystemExit(f"Incomplete upstream checkout: missing {path}")
        rows.append(
            {
                "theorem_name": name,
                "natural_language": record.get("natural_language"),
                "answer": parse_answer_cell(record.get("answer", "")),
                "source": record.get("source") or None,
                "tag": record.get("tag"),
                "formal_statement": strip_comments(path.read_text(encoding="utf-8")),
            }
        )
    return rows


# ------------------------------------------------------------------- formatting


def format_row(record: dict, split: str, source: str) -> dict:
    revision = {"hf": HF_REVISION, "github": GITHUB_REVISION}.get(source)
    return {
        "theorem_name": record["theorem_name"],
        "formal_statement": record["formal_statement"],
        "answers": record.get("answer"),
        "natural_language": record.get("natural_language"),
        "tag": record.get("tag"),
        "source": record.get("source"),
        "split": split,
        "dataset_source": source,
        "dataset_revision": revision,
    }


def validate_rows(rows: list[dict], split: str) -> None:
    """Fail before writing anything if a row cannot be scored.

    Every statement must still contain a ``sorry`` for the model to fill, and a
    ``test`` row must publish exactly one answer per ``abbrev ..._solution`` so
    the verifier's positional zip pairs them correctly.
    """
    problems = []
    seen = set()
    for row in rows:
        name = row["theorem_name"]
        if not isinstance(name, str) or not name:
            problems.append("row without theorem_name")
            continue
        if name in seen:
            problems.append(f"{name}: duplicated")
        seen.add(name)
        statement = row["formal_statement"]
        if not isinstance(statement, str) or "sorry" not in statement:
            problems.append(f"{name}: formal_statement has no sorry to fill")
            continue
        answers = row["answers"]
        if answers is not None and (not isinstance(answers, list) or not all(isinstance(a, str) for a in answers)):
            problems.append(f"{name}: answers must be null or a list of strings")
            continue
        if split == "test":
            n_tags = len(_ABBREV_SOLUTION_RE.findall(statement))
            if n_tags != len(answers or []):
                problems.append(f"{name}: {n_tags} solution abbrev(s) but {len(answers or [])} answer(s)")
    if problems:
        raise SystemExit("Invalid benchmark rows; refusing to write.\n  " + "\n  ".join(problems))


def check_corpus_complete(rows: list[dict], split: str) -> None:
    expected = EXPECTED_ROWS[split]
    if len(rows) != expected:
        raise SystemExit(
            f"Incomplete benchmark corpus for split {split!r}: loaded {len(rows)}, expected {expected}. "
            "Refusing to write a short split; pass --limit for an intentional subset."
        )


def prepare(
    split: str = "test",
    source: str = "github",
    limit: Optional[int] = None,
    output: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    source_file: Optional[Path] = None,
) -> Path:
    """Write one split as Gym task rows and return the output path.

    ``source_file`` bypasses both upstream sources and reads Hugging-Face-shaped
    records from a local JSON list; it exists for synthetic example data and
    tests, and is exempt from the row-count manifest.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}")

    if source_file is not None:
        records = json.loads(Path(source_file).read_text(encoding="utf-8"))
    elif source == "hf":
        records = load_hf_rows(split)
    else:
        records = load_github_rows(split, cache_dir or Path(tempfile.gettempdir()) / "combibench")

    # Synthetic rows are labelled as such so they can never be mistaken for the benchmark.
    rows = [format_row(record, split, "synthetic" if source_file is not None else source) for record in records]
    if limit is not None:
        rows = rows[:limit]
    validate_rows(rows, split)
    if limit is None and source_file is None:
        check_corpus_complete(rows, split)

    output_path = Path(output) if output is not None else OUTPUT_FPATHS[split]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path}", file=sys.stderr)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare CombiBench for NeMo Gym")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="github",
        help="github: pinned .lean files (compile on Mathlib v4.24.0); hf: pinned dataset upstream's harness loads",
    )
    parser.add_argument("--limit", type=positive_int, default=None, help="Max rows to output (positive integer)")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output JSONL path (default: data/combibench_<split>.jsonl)"
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Where the GitHub tarball is unpacked")
    parser.add_argument("--source-file", type=Path, default=None, help="Local JSON list of HF-shaped records")
    args = parser.parse_args()
    prepare(
        split=args.split,
        source=args.source,
        limit=args.limit,
        output=args.output,
        cache_dir=args.cache_dir,
        source_file=args.source_file,
    )


if __name__ == "__main__":
    main()
