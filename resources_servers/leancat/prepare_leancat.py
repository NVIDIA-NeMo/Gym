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

The conversion script lives here rather than in the source repo because the source
repo is third-party and not ours to modify -- the same exception under which
``math_formal_lean/prepare_minif2f.py`` is kept in-tree.

Two prompt variants are supported, because the repo and the paper do not agree:

    repo      prompts/static_passk.md at the pinned commit, byte-exact. This is what
              upstream's scripts/passk.py actually reads, so it is the default.
    paper-d1  The template printed in the paper's Appendix D.1 (arXiv v2). Same first
              three lines, then it asks for step-by-step reasoning inside a lean4 fence
              instead of permitting auxiliary declarations and naming the banned tokens.

Which one produced the published numbers is unknown: the GitHub repo is a release mirror
synced from a private development repo, so its prompt may post-date the paper. Run both if
the difference matters to you.

Usage:
    python prepare_leancat.py                          # repo prompt -> data/train.jsonl
    python prepare_leancat.py --prompt-variant paper-d1  # -> data/paper_d1_train.jsonl
    python prepare_leancat.py --records local.jsonl --prompt some/other/template.md
"""

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# Pinned upstream revision (commit dated 2026-06-19). Bump deliberately, not
# incidentally: another revision can change statements, difficulty labels, or the
# prompt, none of which are detectable from the JSONL alone.
LEANCAT_COMMIT = "4e136a13e5d0b94829c813e6f612fd991e670096"
RAW_BASE = f"https://raw.githubusercontent.com/sciencraft/LeanCat/{LEANCAT_COMMIT}"

RECORDS_URL = f"{RAW_BASE}/data/leancat_records.jsonl"
PROMPT_URL = f"{RAW_BASE}/prompts/static_passk.md"
TARBALL_URL = f"https://codeload.github.com/sciencraft/LeanCat/tar.gz/{LEANCAT_COMMIT}"

# From configs/evaluation_protocol.json at the same commit. Recorded in each row so
# a rollout carries the toolchain it is only meaningful under.
LEAN_TOOLCHAIN = "leanprover/lean4:v4.19.0"
MATHLIB_VERSION = "v4.19.0"

EXPECTED_RECORDS = 100
NUM_EXAMPLE_ROWS = 5

# variant -> (local template filename, output filename prefix). The repo variant is fetched
# and its local copy refreshed; paper-d1 is a hand transcription of a typeset listing and is
# only ever read, never overwritten.
PROMPT_VARIANTS = {
    "repo": ("static-passk.md", ""),
    "paper-d1": ("paper-d1.md", "paper_d1_"),
}


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


def load_statement_files(tar_bytes: bytes) -> Dict[str, str]:
    """Read ``CAT_statement/S_<id>.lean`` out of the pinned tarball, bytes untouched.

    This, not the JSONL, is what upstream prompts from: ``eval_common.load_problem`` does
    ``lean_path.read_text()`` with no ``strip()``. The two sources agree on content but not
    on trailing whitespace -- 60 of the 100 ``.lean`` files end in a newline that
    ``leancat_records.jsonl`` has stripped -- and that newline lands inside the prompt's
    code fence. Sourcing from the file is what makes the rendered prompt byte-identical to
    the reference harness's.

    One tarball rather than 100 raw fetches: same pin, no rate-limit exposure.
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


def to_gym_row(record: Dict[str, Any], template: str, statements: Optional[Dict[str, str]]) -> Dict[str, Any]:
    problem_id = record["problem_id"]
    formal_statement = record["formal_statement"]

    if statements is not None:
        from_file = statements[problem_id]
        # Content drift between the two upstream sources would silently change what is
        # asked and what is checked, so it fails the run rather than getting normalised away.
        if from_file.strip() != formal_statement.strip():
            raise ValueError(
                f"Problem {problem_id}: CAT_statement/S_{problem_id}.lean disagrees with the JSONL record"
            )
        formal_statement = from_file

    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": render_prompt(template, formal_statement)}],
        },
        "verifier_metadata": {
            "problem_id": problem_id,
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
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS),
        default="repo",
        help="Which prompt template to render with. See the module docstring for the difference.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help=f"Explicit template file, overriding --prompt-variant. Defaults to fetching {PROMPT_URL}",
    )
    parser.add_argument(
        "--no-statement-files",
        action="store_true",
        help=(
            "Render prompts from the JSONL's formal_statement instead of the CAT_statement/*.lean "
            "files. Faster and offline-friendly, but 60 of the 100 prompts then differ from the "
            "reference harness's by the file's trailing newline."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).absolute().parent / "data",
        help="Directory to write train.jsonl and example.jsonl into.",
    )
    args = parser.parse_args()

    prompts_dir = Path(__file__).absolute().parent / "prompts"
    template_name, output_prefix = PROMPT_VARIANTS[args.prompt_variant]

    if args.prompt is not None:
        template = args.prompt.read_text(encoding="utf-8")
        refresh_local_copy = False
    elif args.prompt_variant == "repo":
        template = fetch_text(PROMPT_URL)
        refresh_local_copy = True
    else:
        # Transcribed from the paper, not fetchable; read it and leave it alone.
        template = (prompts_dir / template_name).read_text(encoding="utf-8")
        refresh_local_copy = False

    # `.strip()` matches eval_common.load_prompt, which upstream applies to every template.
    template = template.strip()
    if "{formal_statement}" not in template:
        raise ValueError("Prompt template does not contain the {formal_statement} placeholder")

    raw_records = args.records.read_text(encoding="utf-8") if args.records else fetch_text(RECORDS_URL)
    records = load_records(raw_records)

    statements = None
    if not args.no_statement_files:
        print(f"Fetching {TARBALL_URL}")
        with urllib.request.urlopen(TARBALL_URL) as response:
            statements = load_statement_files(response.read())
        print(f"Read {len(statements)} CAT_statement/*.lean files")

    rows = [to_gym_row(record, template, statements) for record in records]

    if refresh_local_copy:
        # Keep a copy of the exact prompt we rendered with, so a reviewer can diff it against
        # upstream without re-running the fetch. Only for the fetched variant: writing a
        # `--prompt` file's contents over static-passk.md would silently corrupt the pinned copy.
        # Hyphenated, unlike the upstream filename: Gym's `no-underscore-md` pre-commit hook
        # rejects underscores in Markdown names. Contents are byte-identical to upstream.
        prompt_path = prompts_dir / template_name
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(template + "\n", encoding="utf-8")
        print(f"Wrote prompt template to {prompt_path}")

    write_jsonl(args.output_dir / f"{output_prefix}train.jsonl", rows)
    write_jsonl(args.output_dir / f"{output_prefix}example.jsonl", rows[:NUM_EXAMPLE_ROWS])

    levels: Dict[str, int] = {}
    for row in rows:
        level = row["verifier_metadata"]["level"]
        levels[level] = levels.get(level, 0) + 1
    print(f"Levels: {levels}")
    print(f"Verify under Lean {LEAN_TOOLCHAIN} / Mathlib {MATHLIB_VERSION}.")


if __name__ == "__main__":
    main()
