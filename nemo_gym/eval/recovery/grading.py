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
"""Saved grading artifacts: whether a saved row's grade is recorded, and why a saved row is re-verified."""

import math
from collections.abc import Mapping
from typing import Any

from nemo_gym.core.failure_kinds import (
    JUDGE_FAILED,
    JUDGE_UNPARSEABLE,
    VERIFIER_ERROR,
    FailureClassificationConfig,
)
from nemo_gym.eval.recovery.decision import (
    FAILURE_CLASS,
    FAILURE_EVENTS,
)
from nemo_gym.eval.recovery.evidence import (
    COLLECT_SCRIPT,
)
from nemo_gym.eval.recovery.policy import (
    _saved_answer,
)


def _grading_facts(row: Mapping[str, Any]) -> Mapping[str, Any]:
    facts = row.get("_ng_collected_grading_facts")
    return facts if isinstance(facts, dict) else {}


def _reward(row: Mapping[str, Any]) -> float | None:
    reward = row.get("reward")
    return reward if type(reward) in (int, float) and math.isfinite(reward) else None


def _unparsed_verdict(facts: Mapping[str, Any]) -> bool:
    """Saved judge evaluations of which one recorded no verdict: a null label, or a judge or parse error label."""
    return facts.get("judge_evaluations") is True and (
        facts.get("judge_unparsed") is True or facts.get("judge_error") is True
    )


def _scored_unmasked(row: Mapping[str, Any]) -> bool:
    """A scored row: unmasked, with a reward and no failure kind."""
    return (
        row.get(FAILURE_CLASS) is None
        and row.get("failure_kind") is None
        and row.get("mask_sample") is not True
        and _reward(row) is not None
    )


def _unparsed_judge_verdict(row: Mapping[str, Any]) -> bool:
    """Whether a scored row's grading is ``judge_unparseable`` for a verdict the judge did not give: a saved judge
    evaluation recorded no verdict (``_unparsed_verdict``), or a pairwise game has none (``games_unjudged``). Its
    reward is no grade."""
    facts = _grading_facts(row)
    return _scored_unmasked(row) and (_unparsed_verdict(facts) or facts.get("games_unjudged") is True)


def _unknown_grading_outcome(row: Mapping[str, Any]) -> bool:
    """Whether a scored row's verifier record reports a judge grading operation with outcome ``unknown``: the grading
    did not resolve, so its reward is no grade. Read from the verifier invocations the row already saves."""
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, Mapping) else None
    return (
        isinstance(verifiers, list)
        and _scored_unmasked(row)
        and any(
            isinstance(verifier, Mapping)
            and isinstance(verifier.get("grading_operations"), list)
            and any(
                isinstance(op, Mapping) and op.get("kind") == "llm_judge" and op.get("outcome") == "unknown"
                for op in verifier["grading_operations"]
            )
            for verifier in verifiers
        )
    )


def _judge_unparseable(row: Mapping[str, Any]) -> bool:
    """Whether a scored row is ``judge_unparseable``: its judge gave no verdict (``_unparsed_judge_verdict``) or its
    verifier's grading did not resolve (``_unknown_grading_outcome``). Judges are stateless: it is re-judged."""
    return _unparsed_judge_verdict(row) or _unknown_grading_outcome(row)


def _judge_evaluations(facts: Mapping[str, Any], reward: float | None) -> bool:
    """A non-empty ``judge_evaluations`` list without a judge error; when every evaluation's verdict label parses
    (``[[A=B]]``/``[[A!=B]]`` or ``Judgement: yes/no``), the reward is 1.0 exactly when every verdict is equal/yes."""
    verdicts = facts.get("judge_verdicts")
    if facts.get("judge_evaluations") is not True or facts.get("judge_error") is True or _unparsed_verdict(facts):
        return False
    if isinstance(verdicts, list) and verdicts and all(type(verdict) is bool for verdict in verdicts):
        return reward == (1.0 if all(verdicts) else 0.0)
    return True


def _judge_record(row: Mapping[str, Any]) -> bool:
    """A saved judge verdict: a ``judge_response``, a non-empty ``judgments`` list with a ``judge_score``, or
    non-empty ``judge_evaluations`` whose parsed verdicts agree with the reward (``_judge_evaluations``)."""
    facts = _grading_facts(row)
    return bool(facts.get("judge_response") or (facts.get("judgments") and facts.get("judge_score"))) or (
        _judge_evaluations(facts, _reward(row))
    )


def _rubric_verdict(row: Mapping[str, Any]) -> bool:
    """A saved rubric judge's verdict: ``verdict`` YES or NO, a finite ``rubric_score`` and ``scoring_threshold``, no
    ``invalid_judge_response``, and the reward 1.0 exactly when the verdict is YES and the score reaches the threshold
    (0.0 otherwise). The saved answer and score are bound."""
    record = _grading_facts(row).get("rubric")
    if not isinstance(record, dict):
        return False
    score, threshold, verdict = record.get("rubric_score"), record.get("scoring_threshold"), record.get("verdict")
    return (
        type(score) in (int, float)
        and type(threshold) in (int, float)
        and type(verdict) is bool
        and record.get("invalid_judge_response") is False
        and _reward(row) == (1.0 if verdict and score >= threshold else 0.0)
        and _bound_answer(row)
    )


def _symbolic_match(row: Mapping[str, Any]) -> bool:
    """A saved symbolic answer check: a ``library_reward`` the row's reward equals, or one beside a judge verdict
    (the judge fallback)."""
    library = _grading_facts(row).get("library_reward")
    return library is not None and (_reward(row) == library or _judge_record(row))


def _metric_record(row: Mapping[str, Any]) -> bool:
    """Saved in-process metric scores: numeric ``sentence_*`` fields (``sentence_chrf``, ``sentence_spbleu``)."""
    return bool(_grading_facts(row).get("sentence_metrics"))


def _test_execution(row: Mapping[str, Any]) -> bool:
    """A saved test run: ``evaluation_completed`` true with an integer ``verifier_exit_code`` and no
    ``verifier_error``."""
    facts = _grading_facts(row)
    return (
        row.get("evaluation_completed") is True
        and facts.get("verifier_exit_code") is not None
        and facts.get("verifier_error") is False
    )


def _bound_answer(row: Mapping[str, Any]) -> bool:
    """Whether the collector bound the saved answer and score together (``_ng_collected_grading_binding``)."""
    binding = row.get("_ng_collected_grading_binding")
    return isinstance(binding, str) and len(binding) == 64


def _repository_test_run(row: Mapping[str, Any]) -> bool:
    """A saved repository test run of the answer's patch: ``evaluation_completed`` true, the run's ``resolved``
    verdict with the row's reward equal to it (1.0 resolved, 0.0 not), saved ``test_results``, no ``error``, no
    evaluation timeout, and a resolved verdict only for a patch that applied. The saved answer and score are bound."""
    run = _grading_facts(row).get("test_run")
    return (
        isinstance(run, dict)
        and row.get("evaluation_completed") is True
        and type(run.get("resolved")) is bool
        and _reward(row) == float(run["resolved"])
        and run.get("test_results") is True
        and run.get("error") is False
        and run.get("timed_out") is False
        and (not run["resolved"] or run.get("patch_applied") is True)
        and _bound_answer(row)
    )


def _substep_tests(row: Mapping[str, Any]) -> bool:
    """Saved per-sub-step test results: a non-empty ``step_results`` list of booleans, one environment result per
    step, ``num_steps_total`` and ``num_steps_passed`` equal to its length and its passes, ``problem_accuracy`` true
    exactly when every step passed, and the row's reward equal to it (1.0 or 0.0). The saved answer and score are
    bound."""
    steps = _grading_facts(row).get("substep_tests")
    return (
        isinstance(steps, dict)
        and type(steps.get("steps")) is int
        and steps["steps"] >= 1
        and steps.get("num_steps_total") == steps["steps"]
        and steps.get("num_steps_passed") == steps.get("passed")
        and steps.get("environment_results") == steps["steps"]
        and type(steps.get("problem_accuracy")) is bool
        and steps["problem_accuracy"] is (steps.get("passed") == steps["steps"])
        and _reward(row) == float(steps["problem_accuracy"])
        and _bound_answer(row)
    )


def _unit_test_results(row: Mapping[str, Any]) -> bool:
    """Saved unit test results of the answer's extracted code: at least one test, and the row's reward 1.0 exactly
    when every test passed (0.0 otherwise), or 0.0, the default penalty, for an answer whose reasoning format
    violated the rule (``reasoning_format_violation_rate`` 1.0). A configured non-zero penalty is not saved on the row,
    so such a row stays unobserved. The saved answer and score are bound."""
    tests = _grading_facts(row).get("unit_tests")
    if not (
        isinstance(tests, dict)
        and type(tests.get("tests")) is int
        and tests["tests"] >= 1
        and type(tests.get("all_passed")) is bool
        and tests.get("code") is True
    ):
        return False
    violation = tests.get("reasoning_format_violation_rate")
    if violation not in (None, 0.0, 1.0):
        return False
    expected = 0.0 if violation == 1.0 else float(tests["all_passed"])
    return _reward(row) == expected and _bound_answer(row)


def _node_set_f1(row: Mapping[str, Any]) -> bool:
    """A saved node-set F1 re-graded from the answer: the collector re-parses the saved answer's final
    ``Final Answer: [...]`` line, reproduces the saved predicted nodes and parse outcome, and scores them against the
    saved expected node list as the verifier does; the row's reward equals that re-graded F1 and the saved ``f1``.
    The saved answer and score are bound."""
    f1 = _grading_facts(row).get("node_f1")
    return (
        isinstance(f1, dict)
        and f1.get("parse_matches") is True
        and type(f1.get("regraded")) is float
        and _reward(row) == f1["regraded"] == f1.get("f1")
        and _bound_answer(row)
    )


def _session_reward_file(row: Mapping[str, Any]) -> bool:
    """The verifier's own reward file of the row's session, read by the collector beside the artifacts
    (``resources/<session_id>/verifier/reward.json``): the evaluation completed with no infrastructure error and the
    row's reward equals the file's ``reward``. Test counts and partial credit are metadata, never the grade. The saved
    answer and score are bound."""
    recorded = _grading_facts(row).get("session_reward")
    return (
        recorded is not None
        and row.get("evaluation_completed") is True
        and _reward(row) == recorded
        and _bound_answer(row)
    )


def _guarded_objective_score(row: Mapping[str, Any]) -> bool:
    """A guardrail-gated objective score the agent's own rubric saved (``response.metrics``): the row's reward equals
    the saved ``aa_headline``, and that headline follows from the saved counts: 0.0 when a guardrail was violated,
    else the objectives passed over the objectives (0.0 without objectives). A row without any counted assertion (no
    world or no assertions also scores 0.0) is no evidence and stays unobserved. The counts are the rubric's own, not
    a re-run of its assertions. The saved answer and score are bound."""
    score = _grading_facts(row).get("guarded_objectives")
    if not isinstance(score, dict) or any(
        type(score.get(key)) not in (int, float) or score[key] < 0
        for key in ("aa_headline", "objectives_passed", "objectives_total", "guardrails_violated", "guardrails_total")
    ):
        return False
    passed, total = score["objectives_passed"], score["objectives_total"]
    if (
        passed > total
        or score["guardrails_violated"] > score["guardrails_total"]
        or total + score["guardrails_total"] == 0
    ):
        return False
    expected = 0.0 if score["guardrails_violated"] > 0 else (passed / total if total else 0.0)
    return _reward(row) == score["aa_headline"] == expected and _bound_answer(row)


def _native_scorer_score(row: Mapping[str, Any]) -> bool:
    """A native scorer's saved score (``native_metrics``): the row's reward equals the saved ``weighted_score`` in
    [0, 1], the scorer saved its binary ``execution`` flag and its diagnostics. The score is the scorer's own, not a
    re-run of it. The saved answer and score are bound."""
    score = _grading_facts(row).get("native_score")
    if not isinstance(score, dict):
        return False
    weighted = score.get("weighted_score")
    return (
        type(weighted) in (int, float)
        and 0.0 <= weighted <= 1.0
        and score.get("execution") in (0.0, 1.0)
        and score.get("diagnostics") is True
        and _reward(row) == weighted
        and _bound_answer(row)
    )


def _environment_reward(row: Mapping[str, Any]) -> bool:
    """The environment's own reward record: ``result.reward_info.reward`` equal to the row's reward."""
    recorded = _grading_facts(row).get("environment_reward")
    return recorded is not None and _reward(row) == recorded


# The saved grading artifacts that record how a trial was graded when its verifier recorded no grading operation and
# no grading record of its own. Each reads the scalar facts the collector projects (``_ng_collected_grading_facts``);
# a bare reward is never one.
GRADING_ARTIFACTS = (
    ("judge_record", _judge_record),
    ("rubric_verdict", _rubric_verdict),
    ("symbolic_match", _symbolic_match),
    ("metric_record", _metric_record),
    ("test_execution", _test_execution),
    ("environment_reward", _environment_reward),
    ("repository_test_run", _repository_test_run),
    ("substep_tests", _substep_tests),
    ("unit_test_results", _unit_test_results),
    ("node_set_f1", _node_set_f1),
    ("session_reward_file", _session_reward_file),
    ("guarded_objective_score", _guarded_objective_score),
    ("native_scorer_score", _native_scorer_score),
)


def grading_artifact(row: Mapping[str, Any]) -> str | None:
    """The name of the first saved grading artifact (``GRADING_ARTIFACTS``) that records the row's grading."""
    return next((name for name, holds in GRADING_ARTIFACTS if holds(row)), None)


def _empty_answer(row: Mapping[str, Any]) -> bool:
    """Whether the saved answer holds no assistant message with text: none at all, or only ones the collector saw
    without non-whitespace text (``has_text`` false). A message whose content was not saved counts as an answer."""
    answer = row.get("response")
    output = answer.get("output") if isinstance(answer, dict) else None
    return isinstance(output, list) and not any(
        isinstance(item, dict) and item.get("role") == "assistant" and item.get("has_text") is not False
        for item in output
    )


def _model_failure(row: Mapping[str, Any]) -> bool:
    """The benchmark's rule scored the model's own failed answer without grading it: the verifier marked the answer
    invalid (``invalid_model_response``), or the model left it incomplete, or it is empty and scored zero."""
    return (
        _grading_facts(row).get("invalid_model_response") is True
        or _incomplete_answer(row)
        or (_empty_answer(row) and _reward(row) == 0)
    )


def _incomplete_answer(row: Mapping[str, Any]) -> bool:
    """Whether the row's saved answer is a response left incomplete, without an error. An incomplete status alone
    says nothing about why: an agent also marks an answer incomplete that holds no message."""
    response = row.get("response")
    return isinstance(response, dict) and response.get("status") == "incomplete" and response.get("error") is None


# --- Saved-answer re-verification --------------------------------------------------------------------------------- #
# A saved result whose grading failed for a judge or transport cause, or whose grading was never recorded, is graded
# again from its saved answer: never by running the policy again. Collection settles it into the run at completion
# (``rollout_reverification.bind_judge_failures``) and ``--recover`` decides REVERIFY for it. No other masked
# result is re-verified: one that infrastructure left unmeasured runs again (``masked_infrastructure_kind``).
REVERIFY_JUDGE = "judge"
REVERIFY_VERIFY_TRANSPORT = "verify_transport"
REVERIFY_UNGRADED = "ungraded"
REVERIFY_VERIFIER = "verifier"
_REVERIFY_JUDGE_KINDS = frozenset({JUDGE_FAILED, JUDGE_UNPARSEABLE})


def _transport_kind(kind: Any) -> bool:
    return isinstance(kind, str) and kind.startswith("transport_")


def _verify_transport_chain(events: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    """Whether ``event`` and every explicit cause below it name only a lost verify call: an agent route failure or a
    transport kind, with no reply status. A verifier's own failure or an HTTP reply anywhere in the chain is not one."""
    from nemo_gym.core.failure_kinds import AGENT_ROUTE_FAILED

    seen: set[str] = set()
    current: Any = event
    while isinstance(current, Mapping):
        kind = current.get("failure_kind")
        foreign = current.get("foreign_error")
        if not (kind == AGENT_ROUTE_FAILED or _transport_kind(kind)):
            return False
        if current.get("http_status") is not None or (
            isinstance(foreign, Mapping) and foreign.get("http_status") is not None
        ):
            return False
        cause = current.get("cause_id")
        if cause is None:
            return True
        if not isinstance(cause, str) or cause in seen:
            return False
        seen.add(cause)
        current = events.get(cause)
    return False


def _verify_transport_failure(row: Mapping[str, Any]) -> bool:
    """Whether the row was masked because its verify call was lost in transport: its kind is an agent route failure
    or a transport kind, and a saved failure event of the ``verify`` operation has a transport-only cause chain."""
    from nemo_gym.core.failure_kinds import AGENT_ROUTE_FAILED

    kind = row.get("failure_kind")
    events = row.get(FAILURE_EVENTS)
    if not (kind == AGENT_ROUTE_FAILED or _transport_kind(kind)) or not isinstance(events, Mapping):
        return False
    return any(
        isinstance(event, Mapping) and event.get("operation") == "verify" and _verify_transport_chain(events, event)
        for event in events.values()
    )


def _grading_unrecorded(row: Mapping[str, Any]) -> bool:
    """Whether the row's verifier invocations were observed but recorded no grading: no grading operation, no
    verifier grading record, no saved grading artifact (``grading_artifact``) and no model-failure rule
    (``_model_failure``). A row from a pin without verifier observations proves nothing and is not selected."""
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, Mapping) else None
    if not isinstance(verifiers, list) or not verifiers:
        return False
    if any(
        not isinstance(verifier, Mapping) or verifier.get("grading_operations") or verifier.get("grading_record")
        for verifier in verifiers
    ):
        return False
    return grading_artifact(row) is None and not _model_failure(row)


def _saved_model_patch(row: Mapping[str, Any]) -> bool:
    """Whether the row saved, intact, the workspace patch its verifier collected and graded: the collector's projected
    facts, else the same checks on a full saved row (``saved_model_patch``, ``collected_patch``). Only a verifier that
    collects the patch itself can be given it back; one that reads the agent session at verify time cannot."""
    facts = row.get("_ng_collected_grading_facts")
    if isinstance(facts, Mapping) and "saved_model_patch" in facts:
        return facts["saved_model_patch"] is True and facts.get("collected_patch") is True
    return (
        _collector_function("saved_model_patch")(row) is True and _collector_function("collected_patch")(row) is True
    )


def reverification_cause(row: Mapping[str, Any]) -> str | None:
    """Why a saved result row is re-verified from its saved answer, or None when it is not.

    ``REVERIFY_JUDGE``: masked because its judge failed (``judge_failed``, ``judge_unparseable``), or scored while
    its judge gave no verdict or its grading did not resolve (``_judge_unparseable``).
    ``REVERIFY_VERIFY_TRANSPORT``: masked because its verify call was lost in transport (``_verify_transport_failure``).
    ``REVERIFY_VERIFIER``: masked because its verifier produced no verdict (``verifier_error``), with the workspace
    patch it graded saved intact (``saved_model_patch``): the verifier runs again on that patch, never the agent.
    ``REVERIFY_UNGRADED``: scored, but no grading was recorded for it (``_grading_unrecorded``).
    Only a result with a saved answer qualifies; a failures-sidecar row, a result a re-verification already settled
    (its settlement is its grading) and every other masked result do not. ``row`` is a collector projection, or a
    saved row passed through ``saved_row_evidence``.
    """
    if (
        row.get(FAILURE_CLASS) is not None
        or row.get("_ng_omitted_rollout") is True
        or row.get("_ng_rejudge_operation_id") is not None
        or not _saved_answer(row)
    ):
        return None
    if row.get("mask_sample") is True:
        if row.get("failure_kind") in _REVERIFY_JUDGE_KINDS:
            return REVERIFY_JUDGE
        if row.get("failure_kind") == VERIFIER_ERROR and _saved_model_patch(row):
            return REVERIFY_VERIFIER
        return REVERIFY_VERIFY_TRANSPORT if _verify_transport_failure(row) else None
    if row.get("evaluation_completed") is False or _reward(row) is None:
        return None
    if _judge_unparseable(row):
        return REVERIFY_JUDGE
    return REVERIFY_UNGRADED if _grading_unrecorded(row) else None


def masked_without_artifact_kind(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> str | None:
    """The kind of a masked result that is never measured nor run again, though infrastructure left it unmeasured; else None.

    A masked result ``masked_infrastructure_kind`` rejects only because it kept a saved answer (so every other rule of
    that predicate applies: the kind must be registered, evaluation-side and not graded again), that has no recorded
    grade (``has_recorded_grade``) and no re-verification cause (``reverification_cause``): its saved answer has no
    gradeable artifact, a resume keeps it and re-verification cannot grade it. It runs again only with the explicit
    ``--rerun-masked-without-artifact``. ``row`` is a collector projection, or a saved row passed through
    ``saved_row_evidence``.
    """
    from nemo_gym.rollouts.rollout_recovery import masked_kept_answer_kind

    kind = masked_kept_answer_kind(row, classification=classification)
    return kind if kind is not None and reverification_cause(row) is None else None


_COLLECTOR_FUNCTIONS: dict[str, Any] = {}


def _collector_function(name: str):
    """``COLLECT_SCRIPT``'s own projection ``name``, so an in-process reader projects a saved row as the collector does."""
    if not _COLLECTOR_FUNCTIONS:
        start, end = COLLECT_SCRIPT.index("\ndef finite("), COLLECT_SCRIPT.index("\ndef grading_result_binding(")
        namespace: dict[str, Any] = {"math": math}
        exec(COLLECT_SCRIPT[start:end], namespace)  # noqa: S102 - this module's own constant source
        _COLLECTOR_FUNCTIONS.update(namespace)
    return _COLLECTOR_FUNCTIONS[name]


def saved_row_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """A full saved row with the evidence fields ``COLLECT_SCRIPT`` projects for the grading predicates: its scalar
    grading facts, whether each assistant message of its answer has text, and its verifier invocations."""
    evidence = dict(row)
    facts = _collector_function("grading_facts")(row)
    if any(value is not None and value is not False and value != [] for value in facts.values()):
        evidence["_ng_collected_grading_facts"] = facts
    answer = row.get("response")
    if isinstance(answer, Mapping) and isinstance(answer.get("output"), list):
        evidence["response"] = dict(answer) | {
            "output": [
                item | {"has_text": has_text}
                if isinstance(item, dict)
                and (has_text := _collector_function("answer_item")(item).get("has_text")) is not None
                else item
                for item in answer["output"]
            ]
        }
    trajectory = row.get("ng_trajectory")
    if isinstance(trajectory, Mapping):
        evidence["ng_trial_observations"] = {"verifier_invocations": trajectory.get("verifier_invocations")}
    return evidence
