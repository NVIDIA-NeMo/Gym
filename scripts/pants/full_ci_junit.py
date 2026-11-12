# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete CI testcase outcomes without changing the tested commands."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict
from xml.etree import ElementTree


class Case(TypedDict):
    file: str
    classname: str
    name: str
    outcome: str
    count: int


class Report(TypedDict):
    valid: bool
    errors: list[str]
    reports: list[str]
    testcase_count: int
    outcomes: dict[str, int]
    cases: list[Case]


class Comparison(TypedDict):
    equal: bool
    baseline_valid: bool
    pants_valid: bool
    missing: list[Case]
    unexpected: list[Case]


CaseKey = tuple[str, str, str, str]


def _rows(cases: Counter[CaseKey]) -> list[Case]:
    return [
        {"file": key[0], "classname": key[1], "name": key[2], "outcome": key[3], "count": count}
        for key, count in sorted(cases.items())
    ]


def summarize_reports(directory: Path, *, expected_reports: list[str] | None = None) -> Report:
    """Require all expected XML files and retain testcase identities, multiplicity and outcomes.

    An explicit empty manifest is the only way to accept no testcases. It is appropriate
    for lint/coverage or work excluded by routing, never for a test command that failed
    during setup. Command exit status and non-pytest checks must be checked separately.
    """
    reports = sorted(directory.rglob("*.xml"))
    names = [path.relative_to(directory).as_posix() for path in reports]
    errors = []
    if expected_reports is not None:
        missing = sorted(set(expected_reports) - set(names))
        extra = sorted(set(names) - set(expected_reports))
        if missing:
            errors.append(f"Missing JUnit reports: {missing}")
        if extra:
            errors.append(f"Unexpected JUnit reports: {extra}")
        if len(expected_reports) != len(set(expected_reports)):
            errors.append("Expected report manifest contains duplicates")
    elif not reports:
        errors.append("No JUnit reports; setup may have failed before tests ran")
    cases: Counter[CaseKey] = Counter()
    for path, name in zip(reports, names, strict=True):
        try:
            tree = ElementTree.parse(path)
        except (ElementTree.ParseError, OSError) as error:
            errors.append(f"Invalid JUnit report {name}: {error}")
            continue
        if tree.getroot().tag not in {"testsuite", "testsuites"}:
            errors.append(f"Invalid JUnit root in {name}: {tree.getroot().tag}")
            continue
        testcases = list(tree.iter("testcase"))
        if not testcases:
            errors.append(f"JUnit report has no testcases: {name}")
        for testcase in testcases:
            if not testcase.get("name"):
                errors.append(f"Unnamed testcase in {name}")
            outcomes = {
                "xfailed" if child.tag == "skipped" and child.get("type") == "pytest.xfail" else child.tag
                for child in testcase
                if child.tag in {"failure", "error", "skipped"}
            }
            # Setup and teardown can both emit errors; retain all distinct result types.
            outcome = "+".join(sorted(outcomes)) if outcomes else "passed"
            cases[(testcase.get("file", ""), testcase.get("classname", ""), testcase.get("name", ""), outcome)] += 1
    if not cases and expected_reports != []:
        errors.append("No testcase outcomes were recorded")
    outcomes: Counter[str] = Counter()
    for key, count in cases.items():
        outcomes[key[3]] += count
    return {
        "valid": not errors,
        "errors": errors,
        "reports": names,
        "testcase_count": sum(cases.values()),
        "outcomes": dict(sorted(outcomes.items())),
        "cases": _rows(cases),
    }


def compare_reports(baseline: Report, pants: Report) -> Comparison:
    """Require testcase-by-testcase parity, including skips and duplicate occurrences."""

    def counter(report: Report) -> Counter[CaseKey]:
        result: Counter[CaseKey] = Counter()
        for case in report["cases"]:
            result[(case["file"], case["classname"], case["name"], case["outcome"])] += case["count"]
        return result

    before = counter(baseline)
    after = counter(pants)
    return {
        "equal": baseline["valid"] and pants["valid"] and before == after,
        "baseline_valid": baseline["valid"],
        "pants_valid": pants["valid"],
        "missing": _rows(before - after),
        "unexpected": _rows(after - before),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("pants", nargs="?", type=Path)
    parser.add_argument("--manifest", type=Path, help="JSON list of expected relative XML paths; [] permits no tests")
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text()) if args.manifest else None
    if expected is not None and (not isinstance(expected, list) or any(not isinstance(x, str) for x in expected)):
        parser.error("--manifest must contain a JSON list of report filenames")
    baseline = summarize_reports(args.baseline, expected_reports=expected)
    if args.pants is None:
        print(json.dumps(baseline, indent=2))
        raise SystemExit(0 if baseline["valid"] else 1)
    pants = summarize_reports(args.pants, expected_reports=expected)
    comparison = compare_reports(baseline, pants)
    print(json.dumps({"baseline": baseline, "pants": pants, "comparison": comparison}, indent=2))
    raise SystemExit(0 if comparison["equal"] else 1)


if __name__ == "__main__":
    main()
