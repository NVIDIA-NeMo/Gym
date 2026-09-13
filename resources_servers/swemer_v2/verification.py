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
"""Run one Swemer Agentic-v2 task's tests in a sandbox and grade the result.

Every row ships its own hand-authored ``test_command`` (there is no per-language template like
Scale-SWE/SWE-rebench have): the dataset authors already ran the actual test suite and recorded
FAIL_TO_PASS/PASS_TO_PASS test ids in whatever format that specific invocation prints. That
means grading is framework-specific, not just language-specific, and the id FORMAT itself is
part of the contract:

  - pytest: dotted module path, e.g. ``tests.pkg.test_mod::test_name`` (note: dots, not
    slashes -- NOT a real pytest node id, which uses ``tests/pkg/test_mod.py::test_name``).
  - go: ``<package import path>::<Test name>``, matching ``go test -json``'s own
    ``Package``/``Test`` event fields directly.
  - jest / vitest: ``<absolute spec file>::<fullName>`` (describe/it chain, space-joined),
    matching the JSON reporter's ``testResults[].name`` / ``assertionResults[].fullName``.
  - mocha: ``<fullTitle>``, optionally prefixed with a literal empty ``::`` (some tasks' ids
    have it, some don't -- both normalise to the same ``fullTitle`` mocha's own JSON reporter
    reports).

``test_command`` frequently does not already request structured output (only ~half of jest/go
rows do, mocha and vitest are mostly plain-text) — see ``inject_output_flags``, which forces it
on rather than trusting each task's author to have set it.

Only ``SUPPORTED_FRAMEWORKS`` are handled. The rest of the ~20 frameworks in the dataset (maven/
junit's opaque surrogate ids, ad hoc per-task C/C++ verification scripts, and a long tail of
one-off runners) are not in ``data/swemer_v2_training.jsonl`` at all. Grading a framework this
file cannot actually parse would produce a plausible-looking empty result, which reads as
"nothing passed": indistinguishable from a real failure.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


TEST_OUTPUT_BEGIN = "___NEMO_GYM_SWEMER_V2_TEST_BEGIN___"
TEST_OUTPUT_END = "___NEMO_GYM_SWEMER_V2_TEST_END___"

SUPPORTED_FRAMEWORKS = frozenset({"pytest", "go", "jest", "mocha", "vitest"})

PASSED = "PASSED"
FAILED = "FAILED"


_FRAMEWORKS = ("pytest", "go", "jest", "mocha", "vitest")

# Two shell shapes confirmed to break when a flag is appended after them, both found for real on
# a live full-set run:
#   - a bare `exit` builtin call as the command's final statement, which takes at most one
#     argument, so appending anything turns `exit $EXIT` into `exit $EXIT -rA` -- invalid syntax
#     for the builtin itself, which fails the whole command instead of just being ignored.
#   - a command whose last statement is a `(...)` subshell group (a `;`/`&&`-chained sequence of
#     `(cd dir && ...)` steps, common in multi-package jest/go commands): a bare word directly
#     after the closing `)` with no separator is a syntax error regardless of what preceded it,
#     e.g. `(cd pkg && npx jest ...) --json` fails with "syntax error near unexpected token".
_TRAILING_UNSAFE_RE = re.compile(r"(?:exit\s+(?:\$\??|\$[A-Za-z_][A-Za-z0-9_]*|-?[0-9]+)|\))\s*$")

# jest/vitest rows sometimes already request `--outputFile=<path>`, writing the JSON report to a
# file instead of stdout -- the only channel this harness captures. Observed for real: 5 of 8
# vitest rows using `--outputFile=` had no JSON anywhere in captured output because of exactly
# this, even though the run itself passed. `cat`-ing the file back onto stdout after the test
# command runs is safe to add unconditionally, including for rows whose own script already reads
# the file back some other way (e.g. via `node -e`/`jq`): a second, redundant `cat` of the same
# content changes nothing for `_extract_json_object`, which accepts the first valid match.
_OUTPUT_FILE_RE = re.compile(r"--outputFile[= ](\S+)")


def inject_output_flags(framework: str, test_command: str) -> str:
    """Force machine-parseable output, regardless of what the row's own command requests.

    Appending at the end (rather than trusting where in a ``&&`` chain the runner sits) is safe
    for the overwhelming majority of rows, including ones whose ``test_command`` is a custom
    wrapper script rather than the labeled framework's own CLI: all four CLIs here take the LAST
    occurrence of a repeated flag, and an unrecognised trailing flag is normally either ignored
    or (for a case like pytest's old ``-rA``, see ``_parse_pytest``) simply produces no extra
    output rather than an error.

    The shapes that are NOT safe, confirmed for real, are a command whose last statement is a
    bare ``exit $VAR`` or a closing ``)`` from a subshell group -- see ``_TRAILING_UNSAFE_RE``.
    An earlier version of this function tried to avoid the ``exit`` case by skipping injection
    whenever the framework's own invocation word (``pytest``, ``jest``, ...) was missing from the
    command, on the theory that a missing tool name meant a wrapper script. That was the wrong
    signal: on a live full-set run it cost 44 jest/mocha/vitest rows their (previously correct)
    resolution, because most such wrapper commands tolerate the appended flag just fine -- only
    the specific trailing shapes above actually corrupt anything.

    A command chaining multiple invocations of the framework's own CLI (e.g. two ``go test ...``
    runs separated by ``;``, one per package) needs the flag on ALL of them, not just the first:
    an earlier version's ``str.replace(..., count=1)`` left later invocations un-jsonified,
    silently losing any target id that only that later invocation's package covered.
    """
    if framework not in _FRAMEWORKS:
        raise ValueError(f"no output-flag strategy for framework {framework!r}")
    output_file_match = _OUTPUT_FILE_RE.search(test_command)
    cat_suffix = f"; cat {output_file_match.group(1)} 2>/dev/null || true" if output_file_match else ""
    if _TRAILING_UNSAFE_RE.search(test_command.rstrip()):
        return test_command + cat_suffix
    if framework == "pytest":
        command = test_command if " -rA" in test_command else test_command + " -rA"
    elif framework == "go":
        command = test_command if "-json" in test_command else test_command.replace("go test", "go test -json")
    elif framework == "jest":
        command = test_command + " --json"
    elif framework == "mocha":
        command = test_command + " --reporter json"
    else:
        command = test_command + " --reporter=json"  # vitest
    return command + cat_suffix


@dataclass
class VerificationInputs:
    instance_id: str
    workdir: str
    patch: str
    test_patch: str
    test_framework: str
    test_command: str
    fail_to_pass: Sequence[str] = field(default_factory=tuple)
    pass_to_pass: Sequence[str] = field(default_factory=tuple)


@dataclass
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    test_output: str
    error: str | None = None


def patch_section_path(section: str) -> str | None:
    """Return the repository-relative path one ``diff --git`` section targets."""
    a_path: str | None = None
    b_path: str | None = None
    for line in section.splitlines():
        if line.startswith("@@"):
            break
        if line.startswith("--- ") and a_path is None:
            value = line[4:].strip()
            a_path = None if value == "/dev/null" else value.removeprefix("a/")
        elif line.startswith("+++ ") and b_path is None:
            value = line[4:].strip()
            b_path = None if value == "/dev/null" else value.removeprefix("b/")

    if b_path or a_path:
        return b_path or a_path

    header = re.match(r"^diff --git a/(.+?) b/(.+)$", section.splitlines()[0] if section else "")
    return header.group(2) if header else None


def drop_patch_sections(patch: str, paths: Iterable[str]) -> str:
    """Drop the diff sections targeting ``paths``.

    ``git add -N . && git diff`` picks up every untracked file, including ones that were already
    untracked before the agent touched anything. Excluding those paths keeps the extracted patch
    to what the agent actually changed.
    """
    dropped = set(paths)
    if not patch or not dropped:
        return patch

    kept: list[str] = []
    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        if not section.strip():
            continue
        path = patch_section_path(section)
        if path is not None and path in dropped:
            continue
        kept.append(section)

    return "".join(kept)


def build_eval_script(inputs: VerificationInputs) -> str:
    """The script run inside the sandbox.

    Patch order mirrors Scale-SWE: the candidate patch (implementation) first, then the
    dataset's held-out ``test_patch``. The two touch disjoint files in every sample inspected
    (implementation vs. test tree), so order has not been observed to matter, but this keeps the
    convention consistent across the SWE resources servers.
    """
    apply_patch = (
        "git apply --reject --recount --ignore-space-change --whitespace=nowarn /tmp/nemo_gym_patch.diff || true"
        if inputs.patch.strip()
        else ""
    )
    apply_test_patch = (
        "git apply --reject --recount --ignore-space-change --whitespace=nowarn /tmp/nemo_gym_test_patch.diff || true"
        if inputs.test_patch.strip()
        else ""
    )
    injected_command = inject_output_flags(inputs.test_framework, inputs.test_command)

    return f"""#!/bin/bash
cd {shlex.quote(inputs.workdir)} || exit 97
set +e
{apply_patch}
{apply_test_patch}

echo "{TEST_OUTPUT_BEGIN}"
{injected_command}
__test_exit=$?
echo "{TEST_OUTPUT_END}"
exit $__test_exit
"""


def slice_test_output(log: str) -> str:
    start = log.find(TEST_OUTPUT_BEGIN)
    if start == -1:
        return log
    start += len(TEST_OUTPUT_BEGIN)
    end = log.find(TEST_OUTPUT_END, start)
    return log[start:end] if end != -1 else log[start:]


def verification_files(inputs: VerificationInputs) -> dict[str, str]:
    files = {"/tmp/nemo_gym_eval.sh": build_eval_script(inputs)}
    if inputs.patch.strip():
        files["/tmp/nemo_gym_patch.diff"] = inputs.patch
    if inputs.test_patch.strip():
        files["/tmp/nemo_gym_test_patch.diff"] = inputs.test_patch
    return files


def grade(statuses: dict[str, str], fail_to_pass: Iterable[str], pass_to_pass: Iterable[str]) -> dict[str, Any]:
    """Resolved only when every required test is observed AND passing.

    A test absent from the output counts as not passing: treating "absent" as success is how a
    test command that never ran scores as a resolved instance.
    """

    def split(names: Iterable[str]) -> tuple[list[str], list[str]]:
        passed, failed = [], []
        for name in names:
            (passed if statuses.get(name) == PASSED else failed).append(name)
        return passed, failed

    f2p_passed, f2p_failed = split(fail_to_pass)
    p2p_passed, p2p_failed = split(pass_to_pass)
    return {
        "FAIL_TO_PASS": {"success": f2p_passed, "failure": f2p_failed},
        "PASS_TO_PASS": {"success": p2p_passed, "failure": p2p_failed},
        "tests_observed": len(statuses),
        "resolved": not f2p_failed and not p2p_failed,
    }


async def run_verification(
    sandbox: Any,
    inputs: VerificationInputs,
    timeout_s: float | None = None,
    log_dir: Path | None = None,
) -> VerificationResult:
    import asyncio

    result = await sandbox.exec("bash /tmp/nemo_gym_eval.sh", timeout_s=timeout_s)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

    if log_dir is not None:

        def _persist() -> None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "test_output.log").write_text(output, errors="replace")
            except OSError:
                pass

        await asyncio.to_thread(_persist)

    if result.return_code == 97:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output=output,
            error=f"workdir {inputs.workdir} not present in the image",
        )

    target_ids = list(inputs.fail_to_pass) + list(inputs.pass_to_pass)
    try:
        statuses = await asyncio.to_thread(
            parse_statuses, inputs.test_framework, slice_test_output(output), target_ids
        )
    except Exception as exc:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=True,
            test_results=None,
            test_output=output,
            error=f"parse failure ({inputs.test_framework}): {exc}",
        )
    report = grade(statuses, inputs.fail_to_pass, inputs.pass_to_pass)
    return VerificationResult(
        completed=True,
        resolved=bool(report["resolved"]),
        patch_applied=True,
        test_results=report,
        test_output=output,
    )


def parse_statuses(framework: str, output: str, target_ids: Iterable[str]) -> dict[str, str]:
    if framework == "pytest":
        return _parse_pytest(output, target_ids)
    if framework == "go":
        return _parse_go(output, target_ids)
    if framework in ("jest", "vitest"):
        return _parse_jest_like(output, target_ids)
    if framework == "mocha":
        return _parse_mocha(output, target_ids)
    raise ValueError(f"no parser for framework {framework!r}")


# --- Framework parsers -------------------------------------------------------------------
# Each was written against real captured sandbox output (one task per framework, golden patch
# applied, `inject_output_flags` command run for real), not against assumptions about a tool's
# JSON schema -- see the module docstring for why that distinction mattered here.

_PYTEST_STATUS_WORDS = ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS")
# `-v` per-test lines pad with spaces up to a right-aligned "[ NN%]"; the status word itself is
# never adjacent to that padding, so this only needs to find "<status><space-or-end>".
_PYTEST_TRAILING_STATUS_RE = re.compile(
    r"^(?P<rest>.*?)\s+(?P<status>" + "|".join(_PYTEST_STATUS_WORDS) + r")\s*(?:\[\s*\d+%\])?\s*$"
)


def _match_pytest_rest(rest: str, by_prefix: dict[str, list[str]]) -> tuple[str, str] | None:
    """Given the node-id portion of one status line, return ``(dotted_id, matched_name)`` or
    ``None``. Shared by both line shapes below -- once the status word is stripped off, a node
    id's own structure (and the ambiguity in its trailing test name) is identical either way.

    Matching is by common SUFFIX of dotted path components, not exact equality or a fixed
    direction, for reasons found for real on live full-set runs, not the single-sample fixtures
    this parser was first built against:

    - A path element outside the checkout root prints as ``../test_x.py`` (rootdir above cwd);
      naively dotting that adds leading empty segments a dataset id never has.
    - The dataset's id can be relative to a NARROWER root than what pytest displays -- e.g. a
      real node id ``canvas_api_client/tests/test_v1_client.py::TestFoo::test_x`` against a
      dataset id ``tests.test_v1_client.TestFoo::test_x``, missing the package directory
      entirely (dataset id shorter than the real path).
    - The dataset's id can also be relative to a WIDER root -- e.g. ``test_command`` does
      ``cd /workspace/repo/master && pytest ...``, so the real node id
      ``buildbot/test/.../test_x.py::...`` never contains the ``master`` directory pytest was
      invoked from, but the dataset's id is ``master.buildbot.test....test_x::...`` (dataset id
      LONGER than the real path). Requiring the dataset prefix to fit inside the real path (as an
      earlier version of this function did) misses this direction entirely -- confirmed for real
      on a live full-set run, not a hypothetical, and not fixable by picking one direction over
      the other, since both shapes occur across different rows of the same dataset.

    Neither side is trusted to be the "real" root, so a candidate only needs its KNOWN suffix
    (whichever is shorter) to line up; the candidate with the longest matching common suffix
    wins, which prefers a more specific match over a coincidentally-short generic one.
    """
    py_idx = rest.find(".py::")
    if py_idx == -1:
        return None
    file_parts = [part for part in rest[:py_idx].split("/") if part not in ("", ".", "..")]
    after_parts = rest[py_idx + len(".py::") :].split("::")
    full_parts = file_parts + after_parts[:-1]
    tail = after_parts[-1]

    best: tuple[str, list[str]] | None = None
    best_common = 0
    for prefix, candidates in by_prefix.items():
        prefix_parts = prefix.split(".")
        common = min(len(prefix_parts), len(full_parts))
        if common == 0 or prefix_parts[-common:] != full_parts[-common:]:
            continue
        if common > best_common:
            best = (prefix, candidates)
            best_common = common
    if best is None:
        return None
    prefix, candidates = best
    for name in sorted(candidates, key=len, reverse=True):
        if tail.startswith(name) and (len(tail) == len(name) or tail[len(name)] in " \t-"):
            return f"{prefix}::{name}", name
    return None


def _parse_pytest(output: str, target_ids: Iterable[str]) -> dict[str, str]:
    """Handles pytest's two verdict-line shapes, since which one appears depends on the pytest
    version baked into the row's (often years-old, pinned) image:

    - ``-rA`` short summary: ``STATUS <node id>``. This is what ``inject_output_flags`` asks
      for, but ``-rA``'s ``A`` (all) category was only added in pytest 3.6 (2018); an older
      pytest silently accepts the unrecognised flag and never prints the summary section at all
      -- observed for real on a pytest-3.3.2 image (198/198 real passes, 0 observed by a
      summary-only parser).
    - plain ``-v``: ``<node id> STATUS   [ NN%]``, printed inline as each test runs. Present in
      every pytest version and always on regardless of ``-rA``, so this is checked whenever the
      summary-line shape does not match, not only as an explicit fallback mode.

    The dataset's ids (``tests.pkg.test_mod::test_name``, or ``tests.pkg.test_mod.TestFoo::
    test_name`` for a class-qualified test) cannot be turned into a real pytest node id by a
    fixed rule: a dotted id has no way to say where the module path ends and a test class name
    begins (both are just more dots) -- ``tests.test_requirements.RequirementTreeTests`` could
    as easily be a 3-level package path as a module plus a class. (First cut of this parser
    guessed "convert every dot to a slash", which is only correct for class-less tests; it
    silently scored 12 real class-based rows as fully unresolved, caught by a live pilot run
    before this shipped.)

    Going the other way is unambiguous, so this matches real node ids OUT of the output against
    the dataset's convention instead: pytest always spells a node id ``<path>.py::<optional
    Class::...::>test_name``, so splitting on the first ``.py::`` isolates the file path (always
    a clean identifier chain, safe to dot-join), and splitting what remains on ``::`` isolates
    any class qualifiers from the test name -- exactly the dataset's own encoding (path dots +
    joined qualifiers, single ``::``, test name last). The test name itself can still contain
    spaces or `` - `` (parametrize brackets), so it is matched against the known candidates for
    that exact prefix (longest first, boundary-checked), the same technique Scale-SWE uses for
    the same reason.

    A node id can legitimately appear on more than one status line, and which one should win
    depends on WHY: a rerun plugin genuinely re-executing a flaky test and getting a different
    real outcome the second time should have its later line win (the ``-rA`` summary line is
    printed after the inline ``-v`` run, so plain last-line-wins is correct there -- see
    ``test_pytest_parser_prefers_rA_summary_when_both_shapes_present``). But pytest also reports
    a test's call phase (``PASSED``/``FAILED``) independently of a setup/teardown-phase ``ERROR``
    for the SAME single execution, and both land in the same ``-rA`` summary. Observed for real
    across 5 rows in one dataset delivery (all ``getsentry-sentry-*``): every target id printed
    as both ``PASSED <id>`` (the test's own body ran correctly) AND, later in the same summary,
    ``ERROR <id>`` from an unrelated teardown-phase fixture (a file-descriptor-count assertion,
    consistent with a sandbox artifact rather than anything the golden patch touched). Unlike a
    rerun's ``FAILED``, ``ERROR`` is not a verdict on the test itself, so it is the one status
    that must never downgrade an already-recorded ``PASSED`` for the same id.
    """
    by_prefix: dict[str, list[str]] = {}
    for target_id in target_ids:
        prefix, sep, name = target_id.partition("::")
        if sep:
            by_prefix.setdefault(prefix, []).append(name)

    statuses: dict[str, str] = {}

    def _record(node_id: str, status_word: str) -> None:
        if status_word == "ERROR" and statuses.get(node_id) == PASSED:
            return
        statuses[node_id] = PASSED if status_word == "PASSED" else FAILED

    for line in output.splitlines():
        matched = False
        for status in _PYTEST_STATUS_WORDS:
            status_prefix = status + " "
            if not line.startswith(status_prefix):
                continue
            result = _match_pytest_rest(line[len(status_prefix) :], by_prefix)
            if result is not None:
                _record(result[0], status)
            matched = True
            break
        if matched:
            continue

        trailing = _PYTEST_TRAILING_STATUS_RE.match(line)
        if trailing is None:
            continue
        result = _match_pytest_rest(trailing.group("rest"), by_prefix)
        if result is not None:
            _record(result[0], trailing.group("status"))
    return statuses


def _parse_go(output: str, target_ids: Iterable[str]) -> dict[str, str]:
    """``go test -json``: one JSON object per line. A test's terminal event is exactly one of
    ``pass``/``fail``/``skip`` on a line that names both ``Package`` and ``Test`` -- the
    ``run``/``output``/``cont``/``pause`` events and the package-level summary (no ``Test``
    field) are not verdicts and are skipped.
    """
    statuses: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        test = event.get("Test")
        package = event.get("Package")
        action = event.get("Action")
        if not test or not package or action not in ("pass", "fail", "skip"):
            continue
        statuses[f"{package}::{test}"] = PASSED if action == "pass" else FAILED
    return statuses


def _extract_json_object(text: str, required_keys: tuple[str, ...]) -> Any:
    """Locate and parse the reporter's JSON object in ``text``, among the surrounding noise.

    Two kinds of noise bracket it in practice: jest/vitest print their normal human-readable
    console report BEFORE the ``--json`` blob (not instead of it), and that report routinely
    contains a bare ``{`` from source code inside a test title (observed for real: a test named
    after a JS object literal, e.g. ``{ validKey: 'msg' }``) -- so the FIRST ``{`` in the text is
    often not the start of anything parseable, and a naive "first `{` that parses" can also match
    a small, well-formed but IRRELEVANT nested fragment before reaching the real blob.
    ``npx``/``npm`` separately print an update-notice banner AFTER the real JSON (also observed
    for real). ``required_keys`` disambiguates: only a parsed object carrying one of the report's
    own top-level keys (e.g. ``testResults`` for jest/vitest) is accepted.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(key in obj for key in required_keys):
            return obj
    raise ValueError(f"no JSON object with any of {required_keys} found in output")


def _parse_jest_like(output: str, target_ids: Iterable[str]) -> dict[str, str]:
    """jest ``--json`` and vitest ``--reporter=json`` share a schema: ``testResults[].name``
    (the spec file) and ``testResults[].assertionResults[].fullName``/``status``. The dataset's
    ids are ``<file>::<fullName>``, but some carry a stray leading space after ``::`` (cause
    unconfirmed -- plausibly an empty top-level describe when the id was generated); both sides
    are stripped before comparing so it does not matter.
    """
    data = _extract_json_object(output, required_keys=("testResults",))
    statuses: dict[str, str] = {}
    parsed: dict[tuple[str, str], str] = {}
    for file_result in data.get("testResults", []):
        name = file_result.get("name", "")
        for assertion in file_result.get("assertionResults", []):
            status = assertion.get("status")
            if status not in ("passed", "failed"):
                continue  # pending/todo/skipped are not a target verdict either way
            parsed[(name, assertion.get("fullName", "").strip())] = PASSED if status == "passed" else FAILED

    for target_id in target_ids:
        file_part, sep, title_part = target_id.partition("::")
        if not sep:
            continue
        status = parsed.get((file_part, title_part.strip()))
        if status is not None:
            statuses[target_id] = status
    return statuses


# mocha's default `spec` reporter (checkmarked, indented-by-describe text), used as a fallback
# when `--reporter json` never reached the real mocha invocation -- see `_parse_mocha_spec_text`.
_MOCHA_SPEC_PASS_RE = re.compile(r"^\s*[✔✓]\s+(?P<title>.+?)(?:\s*\(\d+m?s\))?\s*$")


def _parse_mocha_spec_text(output: str, target_ids: Iterable[str]) -> dict[str, str] | None:
    """Fallback for mocha's default ``spec`` reporter when no JSON reporter output exists at all.

    Confirmed for real on a live full-set run: ``inject_output_flags`` appends ``--reporter
    json`` to the end of ``test_command``, but roughly a third of mocha rows' commands wrap
    mocha inside the project's own test-runner script (``ts-node``, ``jake``, a custom ``npm``
    script) rather than invoking the mocha CLI directly -- the appended flag lands on the outer
    script's argv, not mocha's, so it never takes effect and the run falls back to mocha's
    default text reporter. 29 of 49 non-karma mocha rows without JSON output in one full-set run
    were exactly this: a real, fully-passing run with unambiguous per-test checkmarks.

    Only passing tests are extracted. A target id absent from this text already grades as failing
    (`grade` treats "no verdict" as failing), which is correct for a title mocha never printed,
    so there is no need to parse mocha's numbered failure blocks to get a correct verdict either
    way -- and no real captured failing-block example was available to pin a parser against, so
    none is attempted.
    """
    titles: set[str] = set()
    for line in output.splitlines():
        match = _MOCHA_SPEC_PASS_RE.match(line)
        if match:
            titles.add(match.group("title").strip())
    if not titles:
        return None

    statuses: dict[str, str] = {}
    for target_id in target_ids:
        _, sep, title = target_id.partition("::")
        if not sep:
            title = target_id
        if title.strip() in titles:
            statuses[target_id] = PASSED
    return statuses


def _parse_mocha(output: str, target_ids: Iterable[str]) -> dict[str, str]:
    """mocha's ``--reporter json``: a test's status is which of ``passes``/``failures``/
    ``pending`` it appears in (entries have no explicit status field), keyed by ``fullTitle``
    (mocha's reporter carries no file path at all, unlike jest/vitest). The dataset's ids are
    inconsistent about a file-path prefix before ``::`` -- observed all three shapes across
    different rows, not a parser choice: a real path (``/workspace/repo/.../Swap.js::title``),
    an empty one (``::title``), and no separator at all (bare ``title``). All three carry the
    same ``fullTitle`` as their last (and only reliable) component, so the id is always taken
    from after the FIRST ``::`` when one is present, discarding whatever precedes it either way.

    Falls back to ``_parse_mocha_spec_text`` when no JSON reporter output exists at all -- see
    that function for why that happens on real data despite ``inject_output_flags`` requesting
    JSON for every mocha row.
    """
    target_ids = list(target_ids)
    try:
        data = _extract_json_object(output, required_keys=("passes", "failures", "pending"))
    except ValueError:
        fallback = _parse_mocha_spec_text(output, target_ids)
        if fallback is not None:
            return fallback
        raise
    by_title: dict[str, str] = {}
    for entry in data.get("passes", []):
        by_title[entry.get("fullTitle", "").strip()] = PASSED
    for entry in data.get("failures", []):
        by_title[entry.get("fullTitle", "").strip()] = FAILED
    # pending (skipped) tests are deliberately absent: `grade` treats "no verdict" as failing,
    # which is correct for a test that never ran.

    statuses: dict[str, str] = {}
    for target_id in target_ids:
        _, sep, title = target_id.partition("::")
        if not sep:
            title = target_id
        status = by_title.get(title.strip())
        if status is not None:
            statuses[target_id] = status
    return statuses
