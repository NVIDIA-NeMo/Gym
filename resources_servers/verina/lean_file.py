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
"""Build the Lean file that checks one Verina solution.

A Verina task file marks its regions with comment lines such as `-- !benchmark @start code` and
`-- !benchmark @end code`. The model writes four of those regions (the helper definitions and body of the function,
and the helper lemmas and tactic proof of the specification theorem); every other region, including the pre- and
postconditions, comes from the task. After the task text the checker appends one `#eval` per test case and a
`#print axioms` for the specification theorem. Each appended command prints a marker that `lean_runner` matches to the
command's own line, so output that the model's code prints can never be mistaken for a passing test.
"""

import re
import textwrap
from dataclasses import dataclass


# The regions a solution fills, in the order the model writes them.
SOLUTION_REGIONS = ("code_aux", "code", "proof_aux", "proof")

_START = re.compile(r"^\s*-- !benchmark @start (\w+)")
_END = re.compile(r"^\s*-- !benchmark @end (\w+)")
_TAG = "{tag}"

# A test command prints exactly one of these, followed by the test index.
PASS_MARKER = "VERINA_TEST_PASS"
FAIL_MARKER = "VERINA_TEST_FAIL"

# Text a solution may not contain. Each pattern closes a way to score without solving the task:
# - new syntax (`macro`, `notation`, `elab`, ...) can change what the task's pre- and postconditions mean;
# - a new `instance` can redefine the `==` that the tests compare with;
# - `#eval`, `run_cmd` and similar commands run arbitrary code while the file is checked;
# - `implemented_by`, `extern` and `unsafe` let the compiled code differ from the definition the proof is about;
# - `debug.skipKernelTC` turns off the kernel's check of proofs.
# `sorry`, `native_decide`, and new axioms are caught by the axiom check instead. None of these appear in the
# reference solutions.
_COMMAND_PREFIX = r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|local|scoped|noncomputable|partial)\s+)*"
FORBIDDEN_PATTERNS = {
    # These keywords are also common variable names (`let prefix := ...`), so a match also needs what follows the
    # command: a precedence (`infixl:65`), a string literal, `(`, `|`, or `:`, possibly after one identifier.
    "new syntax": re.compile(
        _COMMAND_PREFIX
        + r"(?:macro|macro_rules|syntax|notation\d?|infix|infixl|infixr|prefix|postfix|elab|elab_rules)"
        + r"(?::\w+)?\s*(?:\w+\s+)?(?:\"|\(|\||:(?!=))",
        re.MULTILINE,
    ),
    "instance": re.compile(
        _COMMAND_PREFIX + r"(?:instance|deriving\s+instance)\b|\battribute\s*\[[^\]]*\binstance\b", re.MULTILINE
    ),
    "code execution": re.compile(r"#eval\b|#exit\b|\brun_(?:cmd|tac|elab|meta)\b|\b(?:builtin_)?initialize\b"),
    "unsafe": re.compile(r"\bunsafe\b|\bimplemented_by\b|@\[\s*extern\b"),
    "skipKernelTC": re.compile(r"\bskipKernelTC\b"),
    "benchmark marker": re.compile(r"!benchmark"),
}


@dataclass(frozen=True)
class Region:
    name: str
    content: str


def parse_template(text: str) -> list[str | Region]:
    """Split a Verina task file into plain text and named regions, dropping the marker lines."""
    parts: list[str | Region] = []
    plain: list[str] = []
    region_name: str | None = None
    region_lines: list[str] = []
    for line in text.splitlines():
        if region_name is None:
            start = _START.match(line)
            if start:
                if plain:
                    parts.append("\n".join(plain))
                    plain = []
                region_name = start.group(1)
                region_lines = []
            else:
                plain.append(line)
            continue
        end = _END.match(line)
        if end:
            if end.group(1) != region_name:
                raise ValueError(f"region {region_name!r} is closed by {end.group(1)!r}")
            parts.append(Region(region_name, "\n".join(region_lines)))
            region_name = None
        else:
            region_lines.append(line)
    if region_name is not None:
        raise ValueError(f"region {region_name!r} is never closed")
    if plain:
        parts.append("\n".join(plain))
    return parts


def region_names(parts: list[str | Region]) -> list[str]:
    return [part.name for part in parts if isinstance(part, Region)]


def strip_solution(text: str) -> str:
    """Return a task file with the contents of the solution regions removed and the markers kept."""
    lines: list[str] = []
    inside: str | None = None
    for line in text.splitlines():
        start = _START.match(line)
        end = _END.match(line)
        if inside is None:
            lines.append(line)
            if start and start.group(1) in SOLUTION_REGIONS:
                inside = start.group(1)
        elif end and end.group(1) == inside:
            lines.append(line)
            inside = None
    return "\n".join(lines) + "\n"


def region_contents(text: str) -> dict[str, str]:
    """Return the contents of every region of a task file, keyed by region name."""
    return {part.name: part.content for part in parse_template(text) if isinstance(part, Region)}


def render(parts: list[str | Region], fills: dict[str, str]) -> str:
    """Join the parts back into Lean source, replacing each region named in `fills`.

    A region's indentation is taken from its original content (or two spaces for the function body and proof, which
    sit inside a `def ... :=` and a `by`), and applied to every line of the replacement.
    """
    out: list[str] = []
    for part in parts:
        if isinstance(part, str):
            out.append(part)
            continue
        if part.name not in fills:
            out.append(part.content)
            continue
        indent = "  " if part.name in ("code", "proof") else ""
        body = textwrap.dedent(fills[part.name]).strip("\n")
        out.append("\n".join(indent + line if line.strip() else "" for line in body.splitlines()))
    return "\n".join(out) + "\n"


def prompt_view(template: str) -> str:
    """Return the task file as the model sees it: each solution region becomes a `{{region}}` placeholder."""
    parts = parse_template(template)
    view = render(parts, {name: "{{" + name + "}}" for name in SOLUTION_REGIONS})
    return re.sub(r"\n{3,}", "\n\n", view)


@dataclass(frozen=True)
class CheckFile:
    """A Lean file plus the 1-based lines whose output the checker reads."""

    source: str
    test_lines: tuple[int, ...]
    axioms_line: int


def build_check_file(
    template: str,
    solution: dict[str, str],
    *,
    function_name: str,
    theorem_name: str,
    tests: list[dict],
    import_mathlib: bool,
) -> CheckFile:
    """Assemble the task, the solution, the test commands, and the axiom check into one Lean file.

    `tests` holds rows of `{"args": [lean term, ...], "expected": lean term}`, already rendered by the data preparer.
    The function's final argument is a proof of its precondition; tests supply it with an axiom declared after the
    specification theorem, so the theorem cannot depend on it, and only test evaluation does.
    """
    parts = parse_template(template)
    missing = [name for name in SOLUTION_REGIONS if name not in region_names(parts)]
    if missing:
        raise ValueError(f"task template has no {', '.join(missing)} region")
    body = render(parts, {name: solution.get(name, "") for name in SOLUTION_REGIONS})
    if import_mathlib and not re.search(r"^import Mathlib\s*$", body, re.MULTILINE):
        body = "import Mathlib\n" + body

    # A few tasks have no precondition, so their function takes no `h_precond` argument.
    signature = re.search(rf"^def {re.escape(function_name)}\b[^\n]*", template, re.MULTILINE)
    if signature is None:
        raise ValueError(f"task template does not define {function_name}")
    precondition_arg = ["verina_test_precondition"] if "h_precond" in signature.group(0) else []

    lines = body.rstrip("\n").split("\n")
    lines += ["", "axiom verina_test_precondition {p : Prop} : p", ""]
    test_lines: list[int] = []
    for index, test in enumerate(tests):
        call = " ".join([function_name, *test["args"], *precondition_arg])
        lines.append(
            f'#eval if ({call}) == {test["expected"]} then "{PASS_MARKER} {index}" else "{FAIL_MARKER} {index}"'
        )
        test_lines.append(len(lines))
    lines.append("")
    lines.append(f"#print axioms {theorem_name}")
    return CheckFile(source="\n".join(lines) + "\n", test_lines=tuple(test_lines), axioms_line=len(lines))


def forbidden_constructs(solution: dict[str, str]) -> list[str]:
    """Return the names of forbidden constructs that appear anywhere in the solution."""
    text = "\n".join(solution.values())
    return [name for name, pattern in FORBIDDEN_PATTERNS.items() if pattern.search(text)]


# Only the rest of the opening tag's line is skipped: the first content line keeps its indentation, which
# `render` needs to dedent the section as a whole.
_SECTION = re.compile(r"<(code_aux|code|proof_aux|proof)>[^\S\n]*\n?(.*?)</\1>", re.DOTALL)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def extract_solution(text: str) -> dict[str, str]:
    """Read the model's tagged answer sections, keeping the last occurrence of each tag.

    A section wrapped in a Markdown code fence is unwrapped. A leading `by` in the proof is dropped, since the
    template already opens the tactic block.
    """
    sections: dict[str, str] = {}
    for name, content in _SECTION.findall(text):
        fenced = _FENCE.match(content)
        sections[name] = (fenced.group(1) if fenced else content).strip("\n")
    proof = sections.get("proof")
    if proof is not None:
        sections["proof"] = re.sub(r"^\s*by\b[ \t]*\n?", "", proof, count=1)
    return sections
