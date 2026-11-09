# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, model-independent requests used by every harness preset."""

from dataclasses import asdict, dataclass, field
from typing import Literal

from nemo_gym.core import AGENT_RUN_ERROR, MODEL_REQUEST_ERROR, MODEL_REQUEST_REJECTED
from nemo_gym.rollouts import Verdict
from nemo_gym.rollouts.harness_capabilities import ExpectedOutcome, PriorityTier

from .registry import LACKS


SUITE = "harness-probes/v2"
ALL_EVIDENCE = tuple(f"TE-{i}" for i in range(1, 10))
# Evidence of a run that ends on a model or harness failure instead of a final answer.
OUTCOME_EVIDENCE = ("TE-1", "TE-2", "TE-4", "TE-7", "TE-8")
GRADED_INCOMPLETE = ExpectedOutcome("incomplete", "verify")
# Every model request fails, so the run ends on a failed model call, and its failure row has no usage to reconcile.
FAILED_MODEL_HEALTH: dict[str, Verdict] = {
    "rollout_ended_on_failed_model_call": "unhealthy",
    "rollout_token_count_mismatch": "unobserved",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    evidence: tuple[str, ...] = ALL_EVIDENCE
    tool_steps: int = 2
    tool_exit_code: int = 0
    usage: bool = True
    http_errors: tuple[int, ...] = ()
    terminal_error: bool = False
    expected_reward: float = 1.0
    steps: bool = True
    judge_verifier: bool = False
    health_expectations: dict[str, Verdict] = field(default_factory=dict)
    tier: PriorityTier = "P0"
    compaction: bool | None = None
    # What the provider does once the scripted tool steps are done: give the final answer, refuse every request with
    # vllm_model's propagated context-length 400, answer every request with vllm_model's empty length stop, or never
    # answer (hold each request until the harness disconnects).
    ending: Literal["answer", "context_overflow_error", "length_stop", "no_reply"] = "answer"
    # The last scripted tool call ends the harness's process tree, so the harness leaves no result.
    kill_harness: bool = False
    # The harness's own run deadline in seconds, set through its adapter's deadline key; None keeps the probe timeout.
    run_deadline_seconds: int | None = None
    # The longest a harness may retry a failing model before it replies, from its first model request.
    retry_budget_seconds: float | None = None
    # The responses the outcome ladder may report for this run, each with the Environment Server's action (P0).
    outcomes: tuple[ExpectedOutcome, ...] = ()
    # The ladder's report once its transitional rules are gone (P1).
    target_outcome: ExpectedOutcome | None = None
    # Harness capabilities the scenario needs (registry.LACKS); a harness without one does not run it.
    requires: tuple[str, ...] = ()

    @property
    def final_answer(self) -> bool:
        """Whether the script ends on the final answer."""
        return self.ending == "answer" and not self.kill_harness

    def task(self) -> dict:
        """Only the task reaches the harness; expectations stay in the runner."""
        return {
            "task_id": self.name,
            "responses_create_params": {
                "input": [
                    {"role": "user", "content": "Run the requested shell checks, then report CONFORMANCE_DONE."}
                ],
            },
        }


SCENARIOS = (
    Scenario("tool_success", "Two successful tool executions, changed history, reasoning and a final answer."),
    Scenario("tool_failure", "A shell command exits 7, followed by recovery and a final answer.", tool_exit_code=7),
    Scenario(
        "usage_omitted",
        "The provider omits usage; counts must remain unknown.",
        usage=False,
        health_expectations={
            "model_call_missing_token_counts": "unhealthy",
            # The fixture deliberately supplies no counts to reconcile.
            "rollout_token_count_mismatch": "unobserved",
        },
    ),
    Scenario(
        "retry_429",
        "Two identical requests receive 429 before recovery.",
        http_errors=(429, 429),
        # A failed attempt has no tokens to account for, and the retry recovers: healthy.
    ),
    Scenario(
        "retry_500",
        "A model HTTP 500 is followed by recovery.",
        http_errors=(500,),
        # A failed attempt has no tokens to account for, and the retry recovers: healthy.
    ),
    Scenario(
        "model_error",
        "A model HTTP 400 on every request is reported to the EnvironmentServer as model_request_rejected (the model "
        "never replied), with its error and retained evidence.",
        evidence=("TE-1", "TE-2", "TE-4", "TE-7", "TE-8"),
        tool_steps=0,
        http_errors=(400,),
        terminal_error=True,
        expected_reward=0.0,
        steps=False,  # The first request fails permanently; no model decision is returned.
        health_expectations={
            "rollout_ended_on_failed_model_call": "unhealthy",
            # Aggregate usage counts finished responses: an initial rejection must
            # retain a known zero total, even though the failed attempt has no usage.
            "rollout_token_count_mismatch": "healthy",
        },
    ),
    Scenario(
        "context_overflow",
        "After one shell step every request is refused with the HTTP 400 context-length error vllm_model returns "
        "with propagate_context_overflow_errors on. Graded: incomplete, or failed model_request_error 400 from a "
        "harness that reads only the status.",
        evidence=OUTCOME_EVIDENCE,
        tool_steps=1,
        ending="context_overflow_error",
        expected_reward=0.0,
        health_expectations={"rollout_ended_on_failed_model_call": "unhealthy"},
        run_deadline_seconds=45,
        outcomes=(GRADED_INCOMPLETE, ExpectedOutcome("failed", "verify", MODEL_REQUEST_ERROR, 400)),
        target_outcome=GRADED_INCOMPLETE,
    ),
    Scenario(
        "context_overflow_length",
        "After one shell step every request gets the empty completion vllm_model returns for a context overflow by "
        "default (Chat finish_reason length; Responses status incomplete). Graded: incomplete, or completed when "
        "the harness carries on.",
        evidence=OUTCOME_EVIDENCE,
        tool_steps=1,
        ending="length_stop",
        expected_reward=0.0,
        # The empty length stop is an empty turn with zero completion tokens that ended on the length limit.
        health_expectations={
            "agent_turn_hollow": "unhealthy",
            "model_call_zero_completion_tokens": "unhealthy",
            "model_call_runaway_generation": "unhealthy",
        },
        run_deadline_seconds=45,
        outcomes=(GRADED_INCOMPLETE, ExpectedOutcome("completed", "verify")),
    ),
    Scenario(
        "deadline_before_reply",
        "The provider never answers, so the harness's run deadline ends the run before any model reply. Incomplete "
        "and graded, or, until such deadlines are graded, failed unknown and retried.",
        evidence=OUTCOME_EVIDENCE,
        tool_steps=0,
        ending="no_reply",
        expected_reward=0.0,
        steps=False,  # No model decision is ever returned.
        run_deadline_seconds=20,
        outcomes=(GRADED_INCOMPLETE, ExpectedOutcome("failed", "retry", "unknown")),
        target_outcome=GRADED_INCOMPLETE,
        requires=("run_deadline",),
    ),
    Scenario(
        "deadline_after_reply",
        "One shell step, then the provider never answers, so the harness's run deadline ends the run after a model "
        "reply. Incomplete and graded.",
        evidence=OUTCOME_EVIDENCE,
        tool_steps=1,
        ending="no_reply",
        expected_reward=0.0,
        run_deadline_seconds=20,
        outcomes=(GRADED_INCOMPLETE,),
        requires=("run_deadline",),
    ),
    Scenario(
        "harness_no_result",
        "The scripted shell call kills every process of the harness's supervised tree, so the harness leaves no "
        "result. Failed agent_run_error and retried.",
        evidence=OUTCOME_EVIDENCE,
        tool_steps=1,
        kill_harness=True,
        expected_reward=0.0,
        outcomes=(ExpectedOutcome("failed", "retry", AGENT_RUN_ERROR),),
        requires=("harness_process",),
    ),
    Scenario(
        "model_401",
        "Every model request answers HTTP 401, so the model never replies. Failed model_request_rejected and not "
        "retried.",
        evidence=OUTCOME_EVIDENCE,
        tool_steps=0,
        http_errors=(401,),
        terminal_error=True,
        expected_reward=0.0,
        steps=False,  # The first request fails permanently; no model decision is returned.
        health_expectations=FAILED_MODEL_HEALTH,
        outcomes=(ExpectedOutcome("failed", "fail", MODEL_REQUEST_REJECTED),),
    ),
    Scenario(
        "model_503",
        "Every model request answers HTTP 503. Failed model_request_error 503 and retried, after the harness's own "
        "retries end within a bounded time.",
        evidence=OUTCOME_EVIDENCE,
        tool_steps=0,
        http_errors=(503,),
        terminal_error=True,
        expected_reward=0.0,
        steps=False,
        health_expectations=FAILED_MODEL_HEALTH,
        run_deadline_seconds=45,
        retry_budget_seconds=30,
        outcomes=(ExpectedOutcome("failed", "retry", MODEL_REQUEST_ERROR, 503),),
    ),
    Scenario(
        "judge_verifier",
        "A model-backed verifier grades the final answer; judge calls have verifier ownership, not policy steps.",
        tool_steps=0,
        judge_verifier=True,
    ),
    Scenario("verifier_failure", "A completed trajectory receives a known zero reward.", expected_reward=0.0),
    Scenario(
        "context_compaction",
        "Automatic compaction during shell actions; retain the helper without creating a policy turn.",
        # Hermes protects the first three and last twenty messages. Build enough
        # real tool history for its automatic summarizer to have an eligible span.
        tool_steps=14,
        tier="P1",
        compaction=True,
    ),
)


def scenarios_for(harness: str, scenarios: tuple[Scenario, ...] = SCENARIOS) -> tuple[Scenario, ...]:
    """The scenarios a harness runs: those whose requirements it meets."""
    lacks = set(LACKS.get(harness, ()))
    return tuple(scenario for scenario in scenarios if not lacks.intersection(scenario.requires))


def suite_manifest(scenarios: tuple[Scenario, ...]) -> dict:
    return {"version": SUITE, "scenarios": [asdict(scenario) for scenario in scenarios]}
