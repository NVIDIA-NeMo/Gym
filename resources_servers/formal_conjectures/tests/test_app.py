# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercises ``FormalConjecturesVerifier.verify`` against real request/response shapes.

``VERIFIER_FIXTURE`` as declared in ``app.py`` uses the bare ``FormalConjecturesVerifier``
class as its ``server_factory``. That class is deliberately abstract: it gets ``config`` and a
working ``_run_lean`` only from ``FormalConjecturesResourcesServer``, which a
fixture-constructed instance never builds. So this module supplies its own ``server_factory``
with those two filled in; the fixture's ``request_model``/``cases_path`` are reused unchanged.

The fake sandbox is driven entirely by markers embedded in the submitted Lean code (the
``response`` text each JSONL case declares) rather than a mock configured per test function,
because ``VerifierFixture`` calls ``server_factory()`` fresh per case with no per-case hook
available for ``full_reward``/``zero_reward``/``malformed`` cases (only ``determinism`` cases
get ``reseed``, and this fixture declares none).
"""

import asyncio
from dataclasses import replace
from typing import Optional

import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult

from ..app import (
    STATUS_UNPROVED,
    VERIFIER_FIXTURE,
    FormalConjecturesResourcesServerConfig,
    FormalConjecturesVerifier,
    FormalConjecturesVerifyRequest,
    target_is_proved,
)


class _FixtureServer(FormalConjecturesVerifier):
    """A ``FormalConjecturesVerifier`` whose Lean compiles are canned.

    The canned results mirror what a real ``lake env lean`` run would say for each situation,
    so the fixture cases exercise the exact ``verify()`` branches a real compile would hit.
    """

    def __init__(self, config: Optional[FormalConjecturesResourcesServerConfig] = None) -> None:
        self.config = config or FormalConjecturesResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="formal_conjectures",
            check_lean_version=False,
        )

    async def _check_toolchain_once(self, expected) -> None:
        return None

    async def _run_lean(self, code: str, timeout_s: Optional[float] = None) -> SandboxExecResult:
        if "MARKER_TIMEOUT" in code:
            return SandboxExecResult(stdout="", stderr="", return_code=124)
        if "MARKER_COMPILE_ERROR" in code:
            return SandboxExecResult(stdout="", stderr="error: unknown identifier 'bogus'", return_code=1)
        if "MARKER_SANDBOX_ERROR" in code:
            return SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="exec_failed")
        if "sorry" in code:
            # A real `#print axioms` on an unfilled proof reports `sorryAx` in the list, and
            # `lake env lean` still exits 0 with only a "declaration uses 'sorry'" warning.
            return SandboxExecResult(
                stdout="warning: declaration uses 'sorry'\n'thm' depends on axioms: [sorryAx]",
                stderr="",
                return_code=0,
            )
        return SandboxExecResult(
            stdout="'thm' depends on axioms: [propext, Classical.choice]",
            stderr="",
            return_code=0,
        )


FIXTURE = replace(VERIFIER_FIXTURE, server_factory=_FixtureServer)


def test_verifier_fixture() -> None:
    from nemo_gym.verifier_fixture import exercise_verifier_fixture

    asyncio.run(
        exercise_verifier_fixture(
            FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


def _request(code: str, target_statement: str = "theorem thm : True") -> FormalConjecturesVerifyRequest:
    return FormalConjecturesVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Prove thm"}]},
            "response": {
                "id": "r",
                "created_at": 0,
                "model": "m",
                "object": "response",
                "output": [
                    {
                        "id": "m1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": f"```lean4\n{code}\n```", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "task_file": "import Mathlib\n\ntheorem thm : True := by sorry\n",
            "full_name": "thm",
            "target_statement": target_statement,
        }
    )


def test_other_sorry_in_file_is_not_a_failure() -> None:
    """An FC file keeps the open conjecture it sanity-checks; that hole must not fail the task.

    This is the branch that makes FC differ from `leancat`: the shared status mapping would
    call the "declaration uses 'sorry'" warning a compile error, and every real FC task would
    score 0. Only `#print axioms` on the *target* decides.
    """
    code = "import Mathlib\n\ntheorem open_conj : True := by sorry\n\ntheorem thm : True := by trivial\n"
    server = _FixtureServer()
    # The fake keys off "sorry" appearing in the submitted file, which is exactly this case.
    result = asyncio.run(server.verify(_request(code)))
    assert result.proof_status == STATUS_UNPROVED
    assert result.reward == 0.0

    # ...and with no `sorry` anywhere, the same file proves out.
    proved = asyncio.run(server.verify(_request("import Mathlib\n\ntheorem thm : True := by trivial\n")))
    assert proved.proof_status == "completed"
    assert proved.reward == 1.0
    assert proved.statement_preserved is True


def test_lemma_and_theorem_are_interchangeable() -> None:
    """`lemma` is notation for `theorem`; writing one for the other is not weakening anything."""
    server = _FixtureServer()
    result = asyncio.run(
        server.verify(
            _request("import Mathlib\n\nlemma thm : True := by trivial\n", target_statement="theorem thm : True")
        )
    )
    assert result.statement_preserved is True
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "submission",
    [
        # Conclusion replaced.
        "import Mathlib\n\ntheorem thm : False := by trivial\n",
        # Hypothesis added.
        "import Mathlib\n\ntheorem thm (h : False) : True := by trivial\n",
        # Renamed, so `#print axioms thm` would not even refer to the model's declaration.
        "import Mathlib\n\ntheorem other : True := by trivial\n",
        # Target dropped entirely.
        "import Mathlib\n\ntheorem unrelated : 1 = 1 := by rfl\n",
    ],
)
def test_altered_statement_is_rejected(submission: str) -> None:
    server = _FixtureServer()
    result = asyncio.run(server.verify(_request(submission)))
    assert result.proof_status == "statement_modified"
    assert result.reward == 0.0
    assert result.statement_preserved is False


def test_extending_the_conclusion_is_a_known_gap() -> None:
    """The check is containment, so text *appended* to the conclusion slips past it.

    Asserting the current behaviour rather than hiding it: `check_target_statement_preserved`
    asks whether the recorded signature is still present, which a submission that adds a
    disjunct keeps true. Tightening it (anchoring on the `:=` that follows) risks the false
    rejections the substring rule was tuned to avoid over a 12k-rollout run, so the limitation
    is documented in the README alongside `#check` / `#print axioms` as the stricter tools.
    """
    server = _FixtureServer()
    result = asyncio.run(server.verify(_request("import Mathlib\n\ntheorem thm : True ∨ False := by simp\n")))
    assert result.statement_preserved is True


def test_added_axiom_is_rejected_but_axiom_in_a_name_is_not() -> None:
    server = _FixtureServer()
    cheated = asyncio.run(
        server.verify(_request("import Mathlib\n\naxiom cheat : True\n\ntheorem thm : True := by exact cheat\n"))
    )
    assert cheated.proof_status == "banned_tokens"

    # `axiom` inside an identifier is an ordinary Mathlib reference, not a declaration.
    honest = asyncio.run(
        server.verify(_request("import Mathlib\n\ntheorem thm : True := by exact Classical.axiom_placeholder\n"))
    )
    assert honest.proof_status != "banned_tokens"


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ("'thm' depends on axioms: [propext]", True),
        ("'thm' depends on axioms: [sorryAx, propext]", False),
        ("no axiom line here", None),
    ],
)
def test_target_is_proved(stdout: str, expected: Optional[bool]) -> None:
    assert target_is_proved({"stdout": stdout, "stderr": ""}) is expected
