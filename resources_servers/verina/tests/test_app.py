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
import asyncio
import json
import re
from pathlib import Path

import pytest

from nemo_gym.verification import exercise_verifier_fixture

from ..app import (
    VERIFIER_FIXTURE,
    VerinaResourcesServerConfig,
    VerinaVerifyRequest,
    _fixture_verifier,
    build_runner,
)
from ..lean_file import (
    Region,
    build_check_file,
    extract_solution,
    forbidden_constructs,
    parse_template,
    prompt_view,
    strip_solution,
)
from ..lean_runner import CheckOutcome, LeanMessage, read_outcome
from ..prepare_verina import lean_literal, typed


TEMPLATE = """-- !benchmark @start import type=solution
-- !benchmark @end import
@[reducible, simp]
def double_precond (n : Nat) : Prop :=
  -- !benchmark @start precond
  True
  -- !benchmark @end precond
-- !benchmark @start code_aux
def helper := 1
-- !benchmark @end code_aux
def double (n : Nat) (h_precond : double_precond (n)) : Nat :=
  -- !benchmark @start code
  n + n
  -- !benchmark @end code
-- !benchmark @start proof_aux
-- !benchmark @end proof_aux
theorem double_spec_satisfied (n : Nat) (h_precond : double_precond (n)) :
    double n h_precond = 2 * n := by
  -- !benchmark @start proof
  unfold double
  omega
  -- !benchmark @end proof
"""

CASES_PATH = Path(__file__).parent / "verifier_cases.jsonl"
LEAN_INSTALLED = (Path(VerinaResourcesServerConfig.model_fields["lean_workspace"].default) / ".installed").exists()
needs_lean = pytest.mark.skipif(
    not LEAN_INSTALLED, reason="Lean workspace not installed; run `python resources_servers/verina/setup_lean.py`"
)


def test_parse_template_reads_regions_and_drops_markers() -> None:
    parts = parse_template(TEMPLATE)
    regions = {part.name: part.content for part in parts if isinstance(part, Region)}
    assert regions["code"] == "  n + n"
    assert regions["precond"] == "  True"
    assert not any("!benchmark" in part for part in parts if isinstance(part, str))


def test_parse_template_rejects_unclosed_region() -> None:
    with pytest.raises(ValueError, match="never closed"):
        parse_template("-- !benchmark @start code\n  n\n")


def test_strip_solution_removes_only_solution_regions() -> None:
    stripped = strip_solution(TEMPLATE)
    assert "n + n" not in stripped and "omega" not in stripped and "helper" not in stripped
    assert "  True" in stripped
    assert stripped.count("!benchmark @start") == TEMPLATE.count("!benchmark @start")


def test_prompt_view_shows_placeholders_not_the_reference() -> None:
    view = prompt_view(strip_solution(TEMPLATE))
    assert "  {{code}}" in view and "  {{proof}}" in view and "{{code_aux}}" in view
    assert "n + n" not in view


def test_build_check_file_indents_solution_and_appends_checks() -> None:
    check = build_check_file(
        strip_solution(TEMPLATE),
        {"code": "n * 2", "proof": "unfold double\nomega"},
        function_name="double",
        theorem_name="double_spec_satisfied",
        tests=[{"args": ["(3 : Nat)"], "expected": "(6 : Nat)"}],
        import_mathlib=True,
    )
    lines = check.source.splitlines()
    assert lines[0] == "import Mathlib"
    assert "  n * 2" in lines and "  omega" in lines
    assert lines[check.test_lines[0] - 1].startswith("#eval if (double (3 : Nat) verina_test_precondition) ==")
    assert lines[check.axioms_line - 1] == "#print axioms double_spec_satisfied"
    # The axiom that supplies preconditions to tests comes after the theorem, so the theorem cannot use it.
    assert check.source.index("axiom verina_test_precondition") > check.source.index("theorem double_spec")


def test_build_check_file_omits_precondition_when_function_has_none() -> None:
    template = strip_solution(TEMPLATE).replace(" (h_precond : double_precond (n)) : Nat", " : Nat", 1)
    check = build_check_file(
        template,
        {"code": "n * 2"},
        function_name="double",
        theorem_name="double_spec_satisfied",
        tests=[{"args": ["(3 : Nat)"], "expected": "(6 : Nat)"}],
        import_mathlib=False,
    )
    assert "#eval if (double (3 : Nat)) == (6 : Nat)" in check.source


def test_extract_solution_keeps_first_line_indentation() -> None:
    # Dedenting needs the first line's indentation; losing it nests the following lines under the first tactic.
    solution = extract_solution("<proof>\n  unfold f\n  constructor\n</proof>")
    assert solution["proof"] == "  unfold f\n  constructor"


def test_extract_solution_unwraps_fences_and_leading_by() -> None:
    text = "Plan...\n<code>\n```lean\nn + n\n```\n</code>\n<proof>\nby\n  omega\n</proof>\n<proof>\nsimp\n</proof>"
    solution = extract_solution(text)
    assert solution["code"] == "n + n"
    assert solution["proof"] == "simp"  # the last occurrence of a tag wins
    assert extract_solution("<proof>by omega</proof>")["proof"] == "omega"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("instance : BEq Nat := ⟨fun _ _ => true⟩", ["instance"]),
        ("attribute [local instance] foo", ["instance"]),
        ('macro "omega" : tactic => `(tactic| sorry)', ["new syntax"]),
        ('local notation "∧" => Or', ["new syntax"]),
        ("#eval IO.println 1", ["code execution"]),
        ("@[implemented_by fast] def slow := 0", ["unsafe"]),
        ("set_option debug.skipKernelTC true", ["skipKernelTC"]),
        # `prefix` used as a variable name is fine; only the command is forbidden.
        ("let prefix := l.take 2\nprefix ++ l", []),
        ("termination_by l.length", []),
    ],
)
def test_forbidden_constructs(text: str, expected: list[str]) -> None:
    assert forbidden_constructs({"code_aux": text}) == expected


def _check():
    return build_check_file(
        strip_solution(TEMPLATE),
        {"code": "n + n", "proof": "omega"},
        function_name="double",
        theorem_name="double_spec_satisfied",
        tests=[{"args": ["(1 : Nat)"], "expected": "(2 : Nat)"}, {"args": ["(2 : Nat)"], "expected": "(4 : Nat)"}],
        import_mathlib=False,
    )


def test_read_outcome_counts_only_markers_on_test_lines() -> None:
    check = _check()
    first, second = check.test_lines
    messages = [
        LeanMessage("information", first, '"VERINA_TEST_PASS 0"'),
        # A pass marker printed from the solution's own line, or for another test's index, does not count.
        LeanMessage("information", 3, '"VERINA_TEST_PASS 1"'),
        LeanMessage("information", second, '"VERINA_TEST_PASS 0"'),
        LeanMessage("information", check.axioms_line, "'double_spec_satisfied' depends on axioms: [propext]"),
    ]
    outcome = read_outcome(check, messages)
    assert (outcome.tests_passed, outcome.tests_total) == (1, 2)
    assert outcome.proof_verified and not outcome.all_tests_passed


def test_read_outcome_rejects_sorry_and_errors() -> None:
    check = _check()
    sorry = LeanMessage("information", check.axioms_line, "'double_spec_satisfied' depends on axioms: [sorryAx]")
    assert not read_outcome(check, [sorry]).proof_verified
    clean = LeanMessage("information", check.axioms_line, "'double_spec_satisfied' does not depend on any axioms")
    assert read_outcome(check, [clean]).proof_verified
    error = LeanMessage("error", 5, "already declared")
    assert not read_outcome(check, [clean, error]).proof_verified
    assert not read_outcome(check, []).proof_verified  # no axiom report at all


def test_timed_out_outcome_passes_nothing() -> None:
    outcome = CheckOutcome(timed_out=True, tests_total=3)
    assert not outcome.all_tests_passed and not outcome.proof_verified


@pytest.mark.parametrize(
    ("value", "lean_type", "expected"),
    [
        (-5, "Int", "(-5)"),
        (True, "Bool", "true"),
        ("False", "Bool", "false"),
        ('say "hi"\n', "String", '"say \\"hi\\"\\n"'),
        ("'", "Char", "'\\''"),
        ([4, -1], "List Int", "[4, (-1)]"),
        ("#[1, 2]", "Array Int", "#[1, 2]"),
        ("some (0, 1)", "Option (Nat × Nat)", "some (0, 1)"),
    ],
)
def test_lean_literal(value: object, lean_type: str, expected: str) -> None:
    assert lean_literal(value, lean_type) == expected


def test_typed_adds_the_type_ascription() -> None:
    assert typed(3, "Nat") == "(3 : Nat)"


def _fixture_request(name: str) -> dict:
    for line in CASES_PATH.read_text().splitlines():
        case = json.loads(line)
        if case["name"] == name:
            return case["request"]
    raise KeyError(name)


def _with_answer(request: dict, answer: str) -> VerinaVerifyRequest:
    response = json.loads(json.dumps(request["response"]))
    response["output"][0]["content"][0]["text"] = answer
    return VerinaVerifyRequest(**{**request, "response": response})


@needs_lean
def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
        )
    )


@needs_lean
def test_attacks_on_the_proof_do_not_earn_full_reward() -> None:
    request = _fixture_request("reference function and proof")
    reference = request["response"]["output"][0]["content"][0]["text"]
    code = re.search(r"<code>\n(.*?)\n</code>", reference, re.S).group(1)
    attacks = {
        # The function is right, so these still earn the tests-only reward, but never the proof reward.
        "sorry": f"<code>\n{code}\n</code>\n<proof>\nsorry\n</proof>",
        "native_decide": f"<code>\n{code}\n</code>\n<proof>\nnative_decide\n</proof>",
        "own axiom": (
            f"<code>\n{code}\n</code>\n<proof_aux>\naxiom cheat : False\n</proof_aux>\n<proof>\nexact cheat.elim\n</proof>"
        ),
        # Declaring the theorem first, with an easier statement, makes the task's theorem a duplicate.
        "theorem declared early": (
            "<code_aux>\ntheorem hasOppositeSign_spec_satisfied : True := trivial\n</code_aux>\n"
            f"<code>\n{code}\n</code>\n<proof>\nsimp\n</proof>"
        ),
    }
    verifier = _fixture_verifier()
    for name, answer in attacks.items():
        result = asyncio.run(verifier.verify(_with_answer(request, answer)))
        assert not result.proof_verified, name
        assert result.reward == verifier.config.tests_only_reward, name


@needs_lean
@pytest.mark.parametrize(
    ("code_aux", "code", "expect_timeout"),
    [
        # Lean compiles a loop with no exit to `unreachable`, so the loop needs a reachable base case.
        ("partial def spin (n : Nat) : Bool := if n == 0 then false else spin (n + 1)", "spin 1", True),
        # Doubling a list exhausts the memory limit long before the timeout.
        ("partial def grow (l : List Nat) : Bool := if l.isEmpty then false else grow (l ++ l)", "grow [1]", False),
    ],
)
def test_runaway_solutions_are_stopped(code_aux: str, code: str, expect_timeout: bool) -> None:
    request = _fixture_request("reference function and proof")
    answer = f"<code_aux>\n{code_aux}\n</code_aux>\n<code>\n{code}\n</code>\n<proof>\nsorry\n</proof>"
    verifier = _fixture_verifier()
    # A 1 GB heap still imports Mathlib, and the doubling list reaches it well within the timeout on any machine.
    verifier.config = verifier.config.model_copy(update={"timeout_s": 60.0, "memory_limit_gb": 1.0})
    verifier._runner = build_runner(verifier.config)
    result = asyncio.run(verifier.verify(_with_answer(request, answer)))
    assert result.reward == 0.0 and result.tests_passed == 0
    assert result.timed_out == expect_timeout
