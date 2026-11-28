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
"""Download Verina and write the Gym task rows.

Verina (https://github.com/sunblaze-ucb/verina, Apache-2.0) has 189 tasks, each a Lean file with the function's
signature, precondition, postcondition, a reference solution, and test cases. The tasks are read from the GitHub
repository at a pinned commit rather than the Hugging Face copy, because the repository's test files keep the
JSON type of every value (the Hugging Face copy flattens list-valued expected outputs into separate strings).

Outputs, in `data/`:
- `train.jsonl` and `validation.jsonl`: every task, split about 80/20 by a hash of its id;
- `reference_answers.jsonl`: each task's reference solution written in the answer format the model is asked
  for, used by `validate_references.py` to check the verifier against known-good solutions.
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

from resources_servers.verina.lean_file import SOLUTION_REGIONS, prompt_view, region_contents, strip_solution


VERINA_COMMIT = "66b3ce5014edafb20cb26b87c681e7f31847249f"
VERINA_TARBALL = f"https://codeload.github.com/sunblaze-ucb/verina/tar.gz/{VERINA_COMMIT}"
VALIDATION_PERCENT = 20

PROMPT = """Write a Lean 4 function that meets a formal specification, and prove that it does.

## Task

{description}

## Lean file

The file below is checked with Lean 4 (v4.18.0) and Mathlib. Your answer fills its four placeholders:

- `{{{{code_aux}}}}`: optional helper definitions used by the function.
- `{{{{code}}}}`: the body of `{function_name}`, the expression after `:=`. Do not repeat the `def` line.
- `{{{{proof_aux}}}}`: optional helper lemmas used by the proof.
- `{{{{proof}}}}`: the tactics that prove `{theorem_name}`, the part after `by`.

```lean
{lean_view}
```

## Answer format

Write each part inside its tag. Leave out a tag you do not need, but always include `<code>` and `<proof>`:

<code_aux>
...
</code_aux>
<code>
...
</code>
<proof_aux>
...
</proof_aux>
<proof>
...
</proof>

A correct function earns partial credit; a correct function with a complete proof earns full credit. A proof that
uses `sorry`, `admit`, `native_decide`, or a new axiom does not count. Your answer may not declare instances, new
syntax or notation, or run commands such as `#eval`.
"""


def lean_string(text: str) -> str:
    escapes = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}
    out = []
    for char in text:
        if char in escapes:
            out.append(escapes[char])
        elif ord(char) < 0x20:
            out.append(f"\\x{ord(char):02x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def lean_char(text: str) -> str:
    if len(text) != 1:
        raise ValueError(f"a Char value must be one character, got {text!r}")
    escapes = {"\\": "\\\\", "'": "\\'", "\n": "\\n", "\t": "\\t", "\r": "\\r"}
    return "'" + escapes.get(text, text) + "'"


def lean_literal(value: object, lean_type: str) -> str:
    """Render one test value as Lean source for a term of `lean_type`.

    Numbers and booleans arrive as JSON values; strings and characters arrive as raw text; every other type arrives
    as a string that is already Lean syntax (`#[1, 2]`, `some (0, 1)`, `fun x => x > 5`).
    """
    lean_type = lean_type.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        text = repr(value)
        return f"({text})" if value < 0 else text
    if isinstance(value, list):
        element = re.fullmatch(r"(?:List|Array)\s+(.+)", lean_type)
        if element is None:
            raise ValueError(f"a JSON list cannot be a {lean_type}")
        items = ", ".join(lean_literal(item, element.group(1).strip("() ")) for item in value)
        return ("#[" if lean_type.startswith("Array") else "[") + items + "]"
    if not isinstance(value, str):
        raise ValueError(f"unsupported test value {value!r} for {lean_type}")
    if lean_type == "String":
        return lean_string(value)
    if lean_type == "Char":
        return lean_char(value)
    if lean_type == "Bool":
        return value.strip().lower()
    return value


def typed(value: object, lean_type: str) -> str:
    return f"({lean_literal(value, lean_type)} : {lean_type})"


def reference_answer(template: str) -> str:
    """The task's reference solution in the tagged format the prompt asks for."""
    regions = region_contents(template)
    return "\n".join(f"<{name}>\n{regions.get(name, '').strip(chr(10))}\n</{name}>" for name in SOLUTION_REGIONS)


def build_row(task_dir: Path) -> tuple[dict, dict]:
    """Return the task row and its reference answer row."""
    task = json.loads((task_dir / "task.json").read_text())
    tests = json.loads((task_dir / "test.json").read_text())
    template = (task_dir / "task.lean").read_text()
    signature = task["signature"]
    function_name = signature["name"]
    theorem_name = f"{function_name}_spec_satisfied"
    if not re.search(rf"^theorem {re.escape(theorem_name)}\b", template, re.MULTILINE):
        raise ValueError(f"{task['id']}: no theorem named {theorem_name}")
    param_types = [(param["param_name"], param["param_type"]) for param in signature["parameters"]]

    rendered_tests = []
    for test in tests:
        args = [typed(test["input"][name], lean_type) for name, lean_type in param_types]
        rendered_tests.append({"args": args, "expected": typed(test["expected"], signature["return_type"])})

    lean_template = strip_solution(template)
    prompt = PROMPT.format(
        description=(task_dir / "description.txt").read_text().strip(),
        function_name=function_name,
        theorem_name=theorem_name,
        lean_view=prompt_view(lean_template).strip(),
    )
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "task_id": task["id"],
        "lean_template": lean_template,
        "function_name": function_name,
        "theorem_name": theorem_name,
        "tests": rendered_tests,
        "difficulty": "advanced" if task["id"].startswith("verina_advanced") else "basic",
    }
    return row, {"task_id": task["id"], "answer": reference_answer(template)}


def split_of(task_id: str) -> str:
    bucket = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


def task_dirs(source: Path) -> list[Path]:
    dirs = [path for path in (source / "datasets" / "verina").iterdir() if (path / "task.json").exists()]
    return sorted(dirs, key=lambda path: (path.name.rsplit("_", 1)[0], int(path.name.rsplit("_", 1)[1])))


def download(dest: Path) -> Path:
    """Fetch the pinned Verina commit into `dest` and return the repository root."""
    root = dest / f"verina-{VERINA_COMMIT}"
    if root.exists():
        return root
    with urllib.request.urlopen(VERINA_TARBALL) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if "/datasets/verina/" in member.name]
        archive.extractall(dest, members=members, filter="data")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="An existing Verina checkout to read instead of downloading.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source or download(args.out_dir / "upstream")
    rows: dict[str, list[dict]] = {"train": [], "validation": []}
    references = []
    for task_dir in task_dirs(source):
        row, reference = build_row(task_dir)
        rows[split_of(row["task_id"])].append(row)
        references.append(reference)

    for split, split_rows in rows.items():
        with (args.out_dir / f"{split}.jsonl").open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "reference_answers.jsonl").open("w") as f:
        for reference in references:
            f.write(json.dumps(reference, ensure_ascii=False) + "\n")
    print(", ".join(f"{split}: {len(split_rows)}" for split, split_rows in rows.items()))


if __name__ == "__main__":
    main()
