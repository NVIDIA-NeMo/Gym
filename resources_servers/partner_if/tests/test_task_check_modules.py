# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-specific checkers come from host-declared modules; contracts still name them by verifier ID."""

import sys
from types import ModuleType

import pytest

from resources_servers.partner_if import checks
from resources_servers.partner_if.checks import CheckContext, fixed, register_modules, resolve
from resources_servers.partner_if.engine import evaluate
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import GradingContract, RawJudgment, VerifierSpec
from resources_servers.partner_if.tests.test_app import server


SAYS_HELLO = "synthetic.says_hello.v1"


def says_hello(context: CheckContext) -> tuple[bool, str]:
    return "HELLO" in context.evidence.answer, "Checked the answer for HELLO"


def install(monkeypatch, name, checkers):
    module = ModuleType(name)
    if checkers is not None:
        module.CHECKERS = checkers
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def plugin(monkeypatch):
    """An importable task-check module; the registry is restored after each test."""
    monkeypatch.setattr(checks, "REGISTRY", dict(checks.REGISTRY))
    return install(monkeypatch, "synthetic_task_checks", {SAYS_HELLO: fixed(says_hello)})


def test_declared_modules_register_their_checkers_idempotently(plugin):
    with pytest.raises(ValueError, match="task_check_modules"):
        resolve(VerifierSpec(id=SAYS_HELLO))
    register_modules([plugin.__name__])
    register_modules([plugin.__name__])  # Every server of a benchmark registers the same module.
    checker, params = resolve(VerifierSpec(id=SAYS_HELLO))
    assert checker.run(params, CheckContext(EvidenceBundle(answer="HELLO")))[0] is True
    assert checker.run(params, CheckContext(EvidenceBundle(answer="BYE")))[0] is False


@pytest.mark.parametrize("identifier", [SAYS_HELLO, "text_no_uuid_v1"])
def test_modules_cannot_redefine_a_registered_verifier(plugin, monkeypatch, identifier):
    register_modules([plugin.__name__])
    other = install(monkeypatch, "other_task_checks", {"synthetic.other.v1": fixed(says_hello)})
    other.CHECKERS[identifier] = fixed(says_hello)
    with pytest.raises(ValueError, match=f"redefines registered verifiers: {identifier}"):
        register_modules([other.__name__])
    assert "synthetic.other.v1" not in checks.REGISTRY


@pytest.mark.parametrize("exported", [None, [], {"synthetic.bare.v1": says_hello}])
def test_modules_must_export_a_checker_mapping(plugin, monkeypatch, exported):
    install(monkeypatch, "malformed_task_checks", exported)
    with pytest.raises(ValueError, match="must define CHECKERS"):
        register_modules(["malformed_task_checks"])


def test_fixed_checks_do_not_accept_dataset_overrides(plugin):
    register_modules([plugin.__name__])
    with pytest.raises(ValueError, match="Extra inputs"):
        resolve(VerifierSpec(id=SAYS_HELLO, params={"literal": "BYE"}))


async def test_registered_checks_share_the_engine_and_never_reach_the_judge(plugin):
    register_modules([plugin.__name__])
    contract = GradingContract(
        task_id="synthetic",
        origin="test",
        source="test",
        rubric_version="test",
        audit_signoff=False,
        data_kind="synthetic_smoke",
        rubrics=[
            {"id": "format", "constraint": "Says HELLO", "verifier": {"id": SAYS_HELLO}},
            {"id": "meaning", "constraint": "Meaning", "verifier": {"id": "llm_judge_v1"}},
        ],
    )

    async def judge(items, evidence):
        assert [item.id for item in items] == ["meaning"]
        return [RawJudgment(id="meaning", passed=False, reason="Missing action", evidence="No action")]

    result = await evaluate(contract, EvidenceBundle(answer="HELLO"), judge=judge)
    assert result.dimensions["if"].points_earned == 1
    assert result.dimensions["if"].points_possible == 2
    assert result.reward == 0


def test_servers_register_their_configured_modules_at_startup(plugin):
    server(task_check_modules=[plugin.__name__])
    assert checks.REGISTRY[SAYS_HELLO] is plugin.CHECKERS[SAYS_HELLO]
    with pytest.raises(ModuleNotFoundError):
        server(task_check_modules=["missing_task_checks"])
