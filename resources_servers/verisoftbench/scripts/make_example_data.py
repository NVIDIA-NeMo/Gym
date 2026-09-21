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
"""Write ``data/example.jsonl``: five synthetic rows in the exact shape the preparer emits.

The benchmark's tasks are third-party theorems from 23 Lean repositories and are not
redistributed here. The five theorems below are written for this file: elementary facts
about ``Nat``, ``Bool`` and ``List`` that need no imports, each placed at the root of a
repository the Lean image contains so that a smoke run against them compiles inside
that repository's own toolchain. Prompts are built with upstream's ``PromptBuilder``
from the pinned checkout, exactly as for the real corpus.
"""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_verisoftbench import DEFAULT_UPSTREAM_DIR, PromptSource, prepare  # noqa: E402
from upstream import ensure_checkout  # noqa: E402


EXAMPLES = [
    ("IntroEffects", "add_zero_right", "(n : Nat) : n + 0 = n", ":= by\n  simp", "Semantics"),
    (
        "LeanExprEvaluator",
        "bool_and_comm",
        "(a b : Bool) : (a && b) = (b && a)",
        ":= by\n  cases a <;> cases b <;> rfl",
        "Compiler",
    ),
    (
        "Lentil",
        "length_append",
        "(xs ys : List Nat) : (xs ++ ys).length = xs.length + ys.length",
        ":= by\n  simp",
        "Framework",
    ),
    ("lean-hoare", "two_mul_eq_add", "(n : Nat) : 2 * n = n + n", ":= by\n  omega", "Type systems"),
    (
        "LeroyCompilerVerificationCourse",
        "reverse_reverse",
        "(xs : List Nat) : xs.reverse.reverse = xs",
        ":= by\n  simp",
        "Applied verif.",
    ),
]


def example_rows() -> list[dict]:
    rows = []
    for i, (repo, name, signature, proof, category) in enumerate(EXAMPLES, start=1):
        thm_name = f"GymExample.{name}"
        stmt = f"theorem {thm_name} {signature}"
        rows.append(
            {
                "id": 100_000 + i,
                "thm_name": thm_name,
                "thm_stmt": stmt,
                "lean_root": repo,
                "rel_path": "GymExample.lean",
                "imports": [],
                "used_lib_defs": [],
                "used_repo_defs": [],
                "lib_lemmas": [],
                "repo_lemmas": [],
                "used_local_defs": [],
                "used_local_lemmas": [],
                "local_ctx": "",
                "target_theorem": stmt + " :=",
                "ground_truth_proof": proof,
                "nesting_depth": 1,
                "transitive_dep_count": 0,
                "subset_aristotle": False,
                "category": category,
            }
        )
    return rows


def main() -> None:
    upstream_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_UPSTREAM_DIR
    ensure_checkout(upstream_dir)
    rows = prepare(example_rows(), PromptSource(upstream_dir), limit=None)
    out = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
