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

"""Build the Formal Conjectures dataset from a pinned upstream revision.

The conversion lives here rather than in the source repo because upstream is third party and
not ours to modify -- the same exception under which ``math_formal_lean/prepare_minif2f.py``
is kept in tree.

``extract.py`` can pull ~1600 candidate tasks out of the repo, but candidacy is not the same
as answerability: a statement can reference a definition that does not survive the rewrite to
plain Mathlib, and some upstream "proofs" transitively depend on a ``sorry`` elsewhere in the
file. Neither is detectable by reading the text. So the benchmark is defined by
``verified_tasks.json`` -- the task ids whose **reference version was observed to compile
clean, with the target free of ``sorryAx``**, inside a Mathlib v4.33.1 sandbox.

That list is the benchmark's definition and is committed. Regenerating it means re-running the
validation sweep (see the README); prepare only re-derives the rows for those ids, so the
dataset is reproducible from the pin without a Lean install.
"""

import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

from resources_servers.formal_conjectures.extract import extract_file


HERE = Path(__file__).absolute().parent

# Pinned upstream revision. Bump deliberately: another revision changes statements, proofs and
# category labels, none of which is detectable from the JSONL alone -- and invalidates the
# verified-task list, which was measured against this tree.
# Abbreviated deliberately: the full 40-character SHA is a "Hex High Entropy String" to
# detect-secrets and fails the secrets-detector CI job. 10 characters clears it and GitHub
# resolves the abbreviated ref for the codeload URL below. Expand with `git rev-parse b82b08faa9`.
FC_COMMIT = "b82b08faa9"
TARBALL_URL = f"https://codeload.github.com/google-deepmind/formal-conjectures/tar.gz/{FC_COMMIT}"

# Formal Conjectures pins this toolchain; the sandbox must match or the numbers are noise.
LEAN_TOOLCHAIN = "leanprover/lean4:v4.33.1"
MATHLIB_VERSION = "v4.33.1"

VERIFIED_TASKS_FPATH = HERE / "verified_tasks.json"
NUM_EXAMPLE_ROWS = 5


def fetch_sources() -> Dict[str, str]:
    """Return ``{relative path: file text}`` for the pinned revision.

    One tarball rather than a clone: same pin, no git dependency, no rate-limit exposure.
    """
    print(f"Fetching {TARBALL_URL}")
    with urllib.request.urlopen(TARBALL_URL) as response:
        blob = response.read()

    sources: Dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".lean"):
                continue
            # strip the leading `formal-conjectures-<sha>/` component
            rel = member.name.split("/", 1)[1]
            handle = tar.extractfile(member)
            if handle is not None:
                sources[rel] = handle.read().decode("utf-8", errors="replace")
    print(f"Read {len(sources)} .lean files")
    return sources


def build_rows() -> List[Dict[str, Any]]:
    """Re-derive the verified tasks as Gym rows."""
    sources = fetch_sources()
    wanted = set(json.loads(VERIFIED_TASKS_FPATH.read_text(encoding="utf-8")))

    by_id = {}
    for path, text in sources.items():
        if not path.startswith("FormalConjectures/"):
            continue
        # No FC-only-identifier screen here. That heuristic exists to cut the cost of the
        # validation sweep by skipping statements unlikely to compile against a stock Mathlib;
        # it needs Mathlib's name list to be accurate, which is not available from the tarball,
        # and an over-broad version silently drops verified tasks. Membership is already
        # decided by `verified_tasks.json`, so re-screening here is redundant and harmful.
        for task in extract_file(path, text, fc_only_names=set()):
            by_id[task.task_id] = task

    missing = wanted - by_id.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} verified task(s) no longer extractable at {FC_COMMIT}, e.g. "
            f"{sorted(missing)[:3]}. The pin and the verified list have drifted apart."
        )

    rows = []
    for task_id in sorted(wanted):
        t = by_id[task_id]
        rows.append(
            {
                "task_id": t.task_id,
                "source_path": t.source_path,
                "declaration": t.declaration,
                "full_name": t.full_name,
                "category": t.category,
                "ams": t.ams,
                "target_statement": t.target_statement,
                "task_file": t.task_file,
                "reference_proof": t.reference_proof,
                "lean_toolchain": LEAN_TOOLCHAIN,
                "mathlib_version": MATHLIB_VERSION,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):3d} rows to {path}")


def main() -> None:
    rows = build_rows()
    write_jsonl(HERE / "data" / "example.jsonl", rows[:NUM_EXAMPLE_ROWS])
    import collections

    print("Categories:", dict(collections.Counter(r["category"] for r in rows)))
    print(f"Verify under Lean {LEAN_TOOLCHAIN} / Mathlib {MATHLIB_VERSION}.")


if __name__ == "__main__":
    main()
