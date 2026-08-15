from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifierResult:
    passed: bool
    violation: str | None = None
    needs_llm_judge: bool = False


class BaseVerifier:
    """Interface for all constraint verifiers.

    check() takes the text of a single trajectory step and an optional context dict.
    Returns VerifierResult(passed=True) if the constraint is satisfied.

    Standard context keys (all optional):
        step_index      int              0-based position in the trajectory
        step_type       str              "thinking" | "tool_call" | "observation" | "final_answer"
        tool_name       str | None       name of the tool called (tool_call steps only)
        prior_steps     list             prior TrajectoryStep objects (for monotonicity checks)
        allowed_files   list[str] | None file paths mentioned in the user request
        is_first_step   bool             True when step_index == 0
        is_final_step   bool             True for final_answer steps
        constraint_params dict           parameters from the AgenticConstraint / ConversationalConstraint
    """

    def check(self, text: str, context: dict | None = None) -> VerifierResult:
        raise NotImplementedError
