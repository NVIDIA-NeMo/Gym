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

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.swebench.pro.prepare import OUTPUT_FPATH as PRO_OUTPUT_FPATH
from benchmarks.swebench.pro.prepare import prepare as prepare_pro


BENCHMARK_DIR = Path(__file__).parent.parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "swebench_pro_multiturn_benchmark.jsonl"
DECOMPOSITIONS_FPATH = Path(__file__).parent / "decompositions.json"
PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8")
NON_SPEC_LINE_PREFIXES = ("The golden patch introduces", "New public interfaces are introduced")
SPEC_FIELDS = ("requirements", "interface")
IDENTIFIER_PATTERNS = (
    re.compile(r"`([^`\n]+)`"),
    re.compile(r"(?<![\w`])((?:[\w-]+)?(?:[./:][\w-]+)+/?)"),
    re.compile(r"\b([a-z]+[A-Z]\w*|[A-Z][a-z0-9]+[A-Z]\w*)\b"),
    re.compile(r"\b(\w+_\w+)\b"),
)
WORD_PATTERN = re.compile(r"\w+")
COPY_WORDS = 8
BANNED_REQUEST_PATTERNS = {
    "mentions tests or grading": re.compile(
        r"\b(tests?|testing|pytest|jest|mocha|grader|grading|graded|hidden|f2p|p2p|fail[_ -]to[_ -]pass|"
        r"pass[_ -]to[_ -]pass|reward|benchmark)\b",
        re.IGNORECASE,
    ),
    "gives feedback on earlier work": re.compile(
        r"\b(looks? good|that works|works now|well done|great job|nice work|is correct|was correct|all good)\b",
        re.IGNORECASE,
    ),
    "previews later requests": re.compile(
        r"\b(later on|in a later|next turn|future turn|follow[- ]up request|we will also|we'll also)\b", re.IGNORECASE
    ),
}


def spec_text(row: Mapping[str, Any], field: str) -> str:
    text = str(row.get(field) or "")
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, str):
            return decoded
        if "\n" not in stripped:
            return stripped[1:-1].replace("\\n", "\n").replace('\\"', '"')
    return text


def _spec_line_numbers(text: str) -> set[int]:
    return {
        number
        for number, line in enumerate(text.splitlines(), 1)
        if line.strip() and line.strip() != "---" and not line.strip().startswith(NON_SPEC_LINE_PREFIXES)
    }


def _excerpt(text: str, ranges: Sequence[Sequence[int]]) -> str:
    lines = text.splitlines()
    return "\n\n".join("\n".join(lines[start - 1 : end]).strip("\n") for start, end in ranges)


def validate_turns(instance_id: str, row: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> None:
    if len(turns) < 2:
        raise ValueError(f"{instance_id}: a multi-turn task needs at least two turns")
    for field in SPEC_FIELDS:
        text = spec_text(row, field)
        line_count = len(text.splitlines())
        covered: Counter[int] = Counter()
        for turn in turns:
            for start, end in turn[field]:
                if not 1 <= start <= end <= line_count:
                    raise ValueError(f"{instance_id}: {field} range {[start, end]} is outside 1..{line_count}")
                covered.update(range(start, end + 1))
        expected = _spec_line_numbers(text)
        missing = sorted(expected - covered.keys())
        repeated = sorted(number for number in expected if covered[number] > 1)
        if missing or repeated:
            raise ValueError(f"{instance_id}: {field} lines missing={missing} repeated={repeated}")


def _copied_span(request: str, source: str) -> str | None:
    words = WORD_PATTERN.findall(request.lower())
    source_words = WORD_PATTERN.findall(source.lower())
    spans = {tuple(source_words[i : i + COPY_WORDS]) for i in range(len(source_words) - COPY_WORDS + 1)}
    for i in range(len(words) - COPY_WORDS + 1):
        if tuple(words[i : i + COPY_WORDS]) in spans:
            return " ".join(words[i : i + COPY_WORDS])
    return None


def request_issues(row: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> list[str]:
    issues = []
    problem = spec_text(row, "problem_statement")
    spec = ""
    for index, turn in enumerate(turns, 1):
        spec += "\n" + "\n".join(_excerpt(spec_text(row, field), turn[field]) for field in SPEC_FIELDS)
        known = problem + spec
        request = turn["request"]
        for reason, pattern in BANNED_REQUEST_PATTERNS.items():
            if match := pattern.search(request):
                issues.append(f"turn {index} request {reason}: {match.group(0)!r}")
        for pattern in IDENTIFIER_PATTERNS:
            for token in pattern.findall(request):
                token = token.strip().rstrip(".,;:)").removesuffix("()")
                if token and token not in known:
                    issues.append(
                        f"turn {index} request introduces {token!r}, which the task text up to this turn lacks"
                    )
        if span := _copied_span(request, problem):
            issues.append(f"turn {index} request copies the problem statement: {span!r}")
        if span := _copied_span(request, spec):
            issues.append(f"turn {index} request repeats requirement or interface text: {span!r}")
    return issues


def render_turn(row: Mapping[str, Any], turn: Mapping[str, Any]) -> str:
    sections = [turn["request"].strip()]
    if turn["requirements"]:
        sections.append("Requirements:\n" + _excerpt(spec_text(row, "requirements"), turn["requirements"]))
    if turn["interface"]:
        sections.append("New interfaces introduced:\n" + _excerpt(spec_text(row, "interface"), turn["interface"]))
    return "\n\n".join(sections)


def build_row(row: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_turns(str(row["instance_id"]), row, turns)
    if issues := request_issues(row, turns):
        raise ValueError(f"{row['instance_id']}: " + "; ".join(issues))
    messages = [render_turn(row, turn) for turn in turns]
    return dict(row) | {
        "responses_create_params": {
            "input": [{"role": "user", "content": PROMPT_TEMPLATE.format(request=messages[0])}],
        },
        "followup_user_messages": messages[1:],
    }


def prepare(
    source_fpath: Path = PRO_OUTPUT_FPATH,
    decompositions_fpath: Path = DECOMPOSITIONS_FPATH,
    output_fpath: Path = OUTPUT_FPATH,
) -> Path:
    if not source_fpath.is_file():
        prepare_pro(output_fpath=source_fpath)
    decompositions = json.loads(decompositions_fpath.read_text(encoding="utf-8"))

    rows = []
    with source_fpath.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["instance_id"] in decompositions:
                rows.append(build_row(row, decompositions[row["instance_id"]]))
    missing = sorted(decompositions.keys() - {row["instance_id"] for row in rows})
    if missing:
        raise ValueError(f"Decomposed instances missing from {source_fpath}: {missing}")

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} SWE-bench Pro multi-turn problems to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    prepare()
