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
"""Prepare PutnamBench test split for the `math_formal_lean` resources server.

Clones `trishullab/PutnamBench` at a pinned commit, runs its own
`rewrite_solutions.py` script (which emits 660 `.lean` files with the upstream's
`sorry`-swapping already applied), then regex-parses each theorem into the
schema expected by the `math_formal_lean` server.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "putnam_bench_benchmark.jsonl"

REPO_URL = "https://github.com/trishullab/PutnamBench.git"
# Pinned commit; drift in the upstream `rewrite_solutions.py` script is detected
# by the `EXPECTED_TEST_ROWS` assertion below.
COMMIT_HASH = "64cedd86ef523f3d5f5dc7a21c10e3f69564c7d4"  # pragma: allowlist secret
EXPECTED_TEST_ROWS = 660

_DOC_COMMENT_RE = re.compile(r"/--[\s\S]*?-/", re.MULTILINE)
_LINE_COMMENT_RE = re.compile(r"^\s*--.*$", re.MULTILINE)
_THEOREM_RE = re.compile(
    r"theorem\s+(putnam_[0-9]{4}_[ab][0-6])([\s\S]*?):=\s*(?:sorry|by)\b",
    re.MULTILINE,
)


def _parse_lean_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    theorem_match = _THEOREM_RE.search(text)
    if not theorem_match:
        raise ValueError(f"No theorem found in {path}")

    theorem_name = theorem_match.group(1)
    theorem_body = theorem_match.group(2)
    theorem_text = f"theorem {theorem_name}{theorem_body}:= by\n"

    pre_theorem = text[: theorem_match.start()]
    doc_match = _DOC_COMMENT_RE.search(pre_theorem)
    line_match = _LINE_COMMENT_RE.search(pre_theorem)

    first_comment_start = None
    if doc_match:
        first_comment_start = doc_match.start()
    if line_match:
        first_comment_start = (
            line_match.start() if first_comment_start is None else min(first_comment_start, line_match.start())
        )

    if first_comment_start is None:
        header = pre_theorem.rstrip()
        informal_prefix = ""
    else:
        header = pre_theorem[:first_comment_start].rstrip()
        informal_prefix = pre_theorem[first_comment_start:].strip("\n")

    if informal_prefix:
        informal_prefix += "\n"
    header = (header.strip() + "\n") if header else ""

    return {
        "name": theorem_name,
        "split": "test",
        "formal_statement": theorem_text,
        "informal_prefix": informal_prefix,
        "header": header,
    }


def _clone_and_rewrite(target_dir: Path) -> None:
    """Clone the upstream repo at the pinned commit and run its rewrite script."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "putnambench"
        print(f"Cloning {REPO_URL} at {COMMIT_HASH}...")
        subprocess.run(["git", "clone", REPO_URL, str(repo_path)], check=True)
        subprocess.run(["git", "-C", str(repo_path), "checkout", COMMIT_HASH], check=True)

        rewrite_script = repo_path / "lean4" / "scripts" / "rewrite_solutions.py"
        if not rewrite_script.exists():
            raise FileNotFoundError(f"rewrite_solutions.py not found at {rewrite_script}")

        print("Running upstream rewrite_solutions.py...")
        subprocess.run([sys.executable, str(rewrite_script)], check=True, cwd=str(rewrite_script.parent))

        solutions_dir = repo_path / "lean4" / "solutions_replaced_new"
        if not solutions_dir.exists():
            raise FileNotFoundError(f"solutions_replaced_new not found at {solutions_dir}")

        for lean_file in solutions_dir.glob("*.lean"):
            shutil.copy(lean_file, target_dir / lean_file.name)


def prepare() -> Path:
    """Clone + rewrite + parse PutnamBench. Returns the output JSONL path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as workdir:
        lean_dir = Path(workdir) / "lean"
        lean_dir.mkdir()
        _clone_and_rewrite(lean_dir)

        lean_files = sorted(lean_dir.glob("*.lean"))
        assert len(lean_files) == EXPECTED_TEST_ROWS, (
            f"Expected {EXPECTED_TEST_ROWS} Lean files, found {len(lean_files)}; upstream may have drifted."
        )

        entries = [_parse_lean_file(f) for f in lean_files]

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
