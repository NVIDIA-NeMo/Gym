# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for task-independent checks, evidence and scoring."""

import pytest

from resources_servers.partner_if.engine import evaluate, reward_checks
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import GradingContract, RawJudgment


def item(identifier, *, verifier="text_contains_v1", params=None, **extra):
    return {
        "id": identifier,
        "constraint": "The required literal is present.",
        "verifier": {
            "id": verifier,
            "params": params if params is not None else {"source": "answer", "literal": "HELLO"},
        },
        **extra,
    }


def task(items):
    return GradingContract(
        task_id="synthetic-mixed",
        origin="synthetic",
        source="authored test",
        rubric_version="test-v4",
        audit_signoff=False,
        data_kind="synthetic_smoke",
        rubrics=items,
    )


def evidence(**overrides):
    return EvidenceBundle(answer="HELLO", **overrides)


async def no_judge(items, bundle):
    raise AssertionError("Deterministic checks must not call an LLM")


async def test_one_engine_mixes_check_types_and_separates_completion():
    async def judge(items, bundle):
        assert [r.id for r in items] == ["llm"]
        assert bundle.answer == "HELLO"
        return [RawJudgment(id="llm", passed=False, reason="Required action missing", evidence="No plan")]

    result = await evaluate(
        task(
            [
                item("literal"),
                item("llm", verifier="llm_judge_v1", params={}),
                item("completion", dimension="completion"),
            ]
        ),
        evidence(),
        judge=judge,
    )
    assert result.dimensions["if"].points_earned == 1
    assert result.dimensions["if"].points_possible == 2
    assert result.dimensions["if"].partial_score == 0.5
    assert result.dimensions["if"].binary_score == 0
    assert result.dimensions["completion"].partial_score == 1
    assert result.dimensions["completion"].binary_score == 1
    assert result.reward == 0


async def test_if_only_task_does_not_invent_completion_score():
    result = await evaluate(task([item("one")]), evidence(), judge=no_judge)
    assert set(result.dimensions) == {"if"}
    assert result.reward == 1


def test_completion_cannot_gate_an_if_requirement():
    with pytest.raises(ValueError, match="IF.*completion"):
        task([item("content", dimension="completion"), item("format", depends_on=["content"])])


async def test_failed_completion_does_not_lower_if_reward():
    result = await evaluate(
        task(
            [
                item("format"),
                item("content", dimension="completion", params={"source": "answer", "literal": "MISSING"}),
            ]
        ),
        evidence(),
        judge=no_judge,
    )
    assert result.reward == 1
    assert result.dimensions["if"].binary_score == 1
    assert result.dimensions["completion"].binary_score == 0


async def test_uncaptured_completion_artifact_does_not_mask_valid_deterministic_if():
    result = await evaluate(
        task(
            [
                item("format"),
                item("content", dimension="completion", params={"source": "artifact:missing", "literal": "HELLO"}),
            ]
        ),
        evidence(),
        judge=no_judge,
    )
    assert result.status == "scored" and result.reward == 1
    assert result.dimensions["completion"].binary_score is None


@pytest.mark.parametrize("capture", [None, {"status": "error", "error": "Disconnected"}])
async def test_completion_capture_error_does_not_mask_independent_llm_if(capture):
    async def judge(items, bundle):
        assert [r.id for r in items] == ["format"]
        return [RawJudgment(id="format", passed=True, reason="Present", evidence="HELLO")]

    result = await evaluate(
        task(
            [
                item("format", verifier="llm_judge_v1", params={"artifacts": []}),
                item(
                    "content", dimension="completion", verifier="llm_judge_v1", params={"artifacts": ["deliverable"]}
                ),
            ]
        ),
        evidence(artifacts={} if capture is None else {"deliverable": capture}),
        judge=judge,
    )
    assert result.status == "scored" and result.reward == 1
    assert result.items[0].raw_passed is True
    assert result.items[1].raw_passed is None
    assert result.dimensions["completion"].binary_score is None


@pytest.mark.parametrize("params", [{}, {"artifacts": None}, {"artifacts": ["deliverable"]}])
async def test_llm_capture_dependencies_remain_conservative(params):
    result = await evaluate(
        task([item("file", verifier="llm_judge_v1", params=params)]),
        evidence(artifacts={"deliverable": {"status": "error", "error": "Disconnected"}}),
        judge=no_judge,
    )
    assert result.reward is None and result.items[0].raw_passed is None


@pytest.mark.parametrize("status,passed", [("present", True), ("absent", False)])
async def test_llm_absent_deliverable_is_gradable_not_a_capture_failure(status, passed):
    async def judge(items, bundle):
        assert bundle.artifact_text("deliverable") == ("HELLO" if passed else None)
        return [RawJudgment(id="file", passed=passed, reason="Checked file", evidence=status)]

    result = await evaluate(
        task([item("file", verifier="llm_judge_v1", params={"artifacts": ["deliverable"]})]),
        evidence(
            artifacts={
                "deliverable": {"status": status, "text": "HELLO" if passed else ""},
                "unrelated": {"status": "error", "error": "Disconnected"},
            }
        ),
        judge=judge,
    )
    assert result.status == "scored" and result.reward == int(passed)


@pytest.mark.parametrize(
    "notes_params,sent",
    [({"artifacts": ["notes"]}, {"report", "notes"}), ({}, {"report", "notes", "build_log"})],
    ids=["listed", "unlisted"],
)
async def test_the_judge_is_sent_only_the_artifacts_its_items_need(notes_params, sent):
    async def judge(items, bundle):
        assert set(bundle.artifacts) == sent
        return [RawJudgment(id=r.id, passed=True, reason="Present", evidence="HELLO") for r in items]

    result = await evaluate(
        task(
            [
                item("report", verifier="llm_judge_v1", params={"artifacts": ["report"]}),
                item("notes", verifier="llm_judge_v1", params=notes_params),
            ]
        ),
        evidence(artifacts={name: {"status": "present", "text": name} for name in ("report", "notes", "build_log")}),
        judge=judge,
    )
    assert result.reward == 1


async def test_no_artifact_dependency_cannot_bypass_actor_capture_failure():
    result = await evaluate(
        task([item("format", verifier="llm_judge_v1", params={"artifacts": []})]),
        evidence(trajectory_error="Actor lost"),
        judge=no_judge,
    )
    assert result.reward is None and result.items[0].reason == "Actor lost"


async def test_uncaptured_llm_prerequisite_masks_dependent_if():
    contract = GradingContract.model_validate(
        task([item("format")]).model_dump()
        | {
            "rubrics": [item("format", depends_on=["file"])],
            "prerequisites": [item("file", verifier="llm_judge_v1", params={"artifacts": ["deliverable"]})],
        }
    )
    result = await evaluate(contract, evidence(), judge=no_judge)
    assert result.reward is None
    assert result.items[0].raw_passed is True and result.items[0].passed is None
    assert result.items[0].blocked_by == ["file"]


@pytest.mark.parametrize(
    "format_depends_on,status,reward,earned", [([], "scored", 1, 1), (["file"], "unscorable", None, None)]
)
async def test_only_prerequisites_of_if_rubrics_gate_availability(format_depends_on, status, reward, earned):
    contract = GradingContract.model_validate(
        task([item("format")]).model_dump()
        | {
            "rubrics": [
                item("format", depends_on=format_depends_on),
                item("content", dimension="completion", depends_on=["file"]),
            ],
            # Keeps the default dimension; its artifact was never captured.
            "prerequisites": [item("file", params={"source": "artifact:missing", "literal": "HELLO"})],
        }
    )
    result = await evaluate(contract, evidence(), judge=no_judge)
    assert result.status == status and result.reward == reward
    assert result.dimensions["if"].points_earned == earned and result.dimensions["if"].points_possible == 1
    assert result.dimensions["completion"].binary_score is None


def test_reward_checks_are_if_rubrics_and_their_transitive_prerequisites():
    contract = GradingContract.model_validate(
        task([item("if_a")]).model_dump()
        | {
            # `if_a` needs `p1`, which needs `p2`; only the completion rubric needs `p3`.
            "rubrics": [item("if_a", depends_on=["p1"]), item("content", dimension="completion", depends_on=["p3"])],
            "prerequisites": [item("p1", depends_on=["p2"]), item("p2"), item("p3")],
        }
    )
    assert reward_checks(contract) == {"if_a", "p1", "p2"}


@pytest.mark.parametrize("format_depends_on", [["file"], ["relay"]])
async def test_dependency_failure_does_not_hide_an_unavailable_prerequisite(format_depends_on):
    contract = GradingContract.model_validate(
        task([item("format")]).model_dump()
        | {
            # `format` needs `file` directly or through `relay`.
            "rubrics": [item("format", depends_on=format_depends_on)],
            "prerequisites": [
                item("absent", params={"source": "answer", "literal": "ABSENT"}),
                item("file", params={"source": "artifact:missing", "literal": "HELLO"}, depends_on=["absent"]),
                item("relay", depends_on=["file"]),
            ],
        }
    )
    result = await evaluate(contract, evidence(), judge=no_judge)
    assert result.prerequisites[1].raw_passed is None and result.prerequisites[1].passed is False
    assert result.status == "unscorable" and result.reward is None


@pytest.mark.parametrize("artifacts", ["deliverable", [""], ["deliverable", "deliverable"]])
async def test_llm_artifact_dependencies_are_validated(artifacts):
    with pytest.raises(ValueError):
        await evaluate(
            task([item("file", verifier="llm_judge_v1", params={"artifacts": artifacts})]),
            evidence(),
            judge=no_judge,
        )


@pytest.mark.parametrize("answer,earned", [("", 0), ("citation partial", 1), ("citation full", 2)])
async def test_unscored_prerequisite_gates_without_adding_points_or_using_partial_coverage(answer, earned):
    contract = GradingContract(
        **{
            "task_id": "citations",
            "origin": "synthetic",
            "source": "authored",
            "rubric_version": "v4",
            "audit_signoff": False,
            "data_kind": "synthetic_smoke",
            "rubrics": [
                item("coverage", params={"source": "answer", "literal": "full"}, depends_on=["has_citation"]),
                item("format", params={"source": "answer", "literal": "citation"}, depends_on=["has_citation"]),
            ],
        },
        prerequisites=[item("has_citation", params={"source": "answer", "literal": "citation"})],
    )
    result = await evaluate(contract, EvidenceBundle(answer=answer), judge=no_judge)
    assert result.dimensions["if"].points_possible == 2
    assert result.dimensions["if"].points_earned == earned
    assert len(result.items) == 2 and len(result.prerequisites) == 1
    assert result.items[1].blocked_by == ([] if answer else ["has_citation"])


async def test_dependencies_preserve_raw_verdict_and_fixed_denominator():
    result = await evaluate(
        task(
            [
                item("absent", params={"source": "answer", "literal": "NOT THERE"}),
                item("dependent", depends_on=["absent"]),
                item("independent"),
            ]
        ),
        evidence(),
        judge=no_judge,
    )
    dependent = result.items[1]
    assert dependent.raw_passed is True
    assert dependent.passed is False
    assert dependent.blocked_by == ["absent"]
    assert result.dimensions["if"].partial_score == pytest.approx(1 / 3)
    assert result.items[2].passed is True


@pytest.mark.parametrize(
    "bad",
    [
        [item("x", depends_on=["missing"])],
        [item("x", depends_on=["y"]), item("y", depends_on=["x"])],
        [item("x"), item("x")],
    ],
)
def test_invalid_dependency_graph_rejected(bad):
    with pytest.raises(ValueError):
        task(bad)


@pytest.mark.parametrize(
    "spec",
    [
        {"id": "some.module.eval", "params": {}},
        {"id": "text_contains_v1", "params": {"source": "answer", "literal": "HELLO", "execute": "bad"}},
        {"id": "llm_judge_v1", "params": {"arbitrary": True}},
    ],
)
async def test_invalid_checker_configuration_rejected_before_inference(spec):
    with pytest.raises(ValueError):
        await evaluate(task([{**item("one"), "verifier": spec}]), evidence(), judge=no_judge)


async def test_actor_missing_artifact_is_failure_capture_error_is_unscorable():
    rubric = task([item("file", params={"source": "artifact:deliverable", "literal": "HELLO"})])
    absent = evidence(artifacts={"deliverable": {"status": "absent"}})
    failed = await evaluate(rubric, absent, judge=no_judge)
    assert failed.status == "scored" and failed.reward == 0
    lost = evidence(artifacts={"deliverable": {"status": "error", "error": "capture disconnected"}})
    masked = await evaluate(rubric, lost, judge=no_judge)
    assert masked.status == "unscorable"
    assert masked.reward is None
    assert masked.dimensions["if"].partial_score is None


async def test_undeclared_artifact_is_infrastructure_gap_not_empty_submission():
    result = await evaluate(
        task([item("file", params={"source": "artifact:deliverable", "literal": "HELLO"})]),
        evidence(),
        judge=no_judge,
    )
    assert result.status == "unscorable" and result.reward is None


@pytest.mark.parametrize("raw", [[], [True, True], [True]])
async def test_invalid_judge_results_cannot_be_model_failures(raw):
    async def judge(items, bundle):
        return [RawJudgment(id="wrong", passed=passed, reason="x", evidence="x") for passed in raw]

    with pytest.raises(ValueError, match="exactly once"):
        await evaluate(task([item("llm", verifier="llm_judge_v1", params={})]), evidence(), judge=judge)


async def test_checker_exception_is_not_silently_scored_zero():
    async def judge(items, bundle):
        raise RuntimeError("judge offline")

    with pytest.raises(RuntimeError, match="judge offline"):
        await evaluate(task([item("llm", verifier="llm_judge_v1", params={})]), evidence(), judge=judge)


async def test_raw_and_effective_judgments_are_stable_in_original_order():
    async def judge(items, bundle):
        return [RawJudgment(id=r.id, passed=True, reason="ok", evidence="HELLO") for r in reversed(items)]

    result = await evaluate(
        task([item("first", verifier="llm_judge_v1", params={}), item("second", verifier="llm_judge_v1", params={})]),
        evidence(),
        judge=judge,
    )
    assert [r.id for r in result.items] == ["first", "second"]
    assert all(r.raw_passed is True and r.passed is True for r in result.items)


async def test_dependency_failure_does_not_hide_an_unavailable_raw_judgment():
    async def judge(items, bundle):
        return [RawJudgment(id="review", passed=None, reason="Needs review", evidence="Unclear intent")]

    result = await evaluate(
        task(
            [
                item("absent", params={"source": "answer", "literal": "ABSENT"}),
                item("review", verifier="llm_judge_v1", params={}, depends_on=["absent"]),
            ]
        ),
        evidence(),
        judge=judge,
    )
    assert result.status == "unscorable" and result.reward is None
    assert result.dimensions["if"].partial_score is None
