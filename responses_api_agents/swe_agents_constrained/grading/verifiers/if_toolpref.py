"""Verifier for ToolPreference constraints.

Tool preference verification operates on a ToolCallTrace (structured sequence of tool
calls), not on raw text.  The interface is different from BaseVerifier intentionally:
the input is typed, not a raw string.

Tier semantics
--------------
HARD      : only the preferred_tool may be called — any other tool is a violation.
AVOIDANCE : the forbidden_tool must never appear in the trace.
SOFT      : when both the preferred and alternative tools were available and applicable,
            the preferred tool should have been chosen.  This requires contextual
            applicability judgment and is handled by the LLM judge path
            (check_trace returns needs_llm_judge=True for soft tier).
"""
from __future__ import annotations

from .base import VerifierResult
from ..if_toolpref.constraints import PreferenceTier, ToolPreference
from ..if_toolpref.schema import ToolCallTrace


class ToolPreferenceVerifier:
    """Verify that a ToolCallTrace complies with a ToolPreference instruction."""

    def check_trace(
        self,
        trace: ToolCallTrace,
        preference: ToolPreference,
    ) -> VerifierResult:
        tool_names = [step.tool for step in trace.steps]

        if preference.tier == PreferenceTier.HARD:
            return self._check_hard(tool_names, preference)

        if preference.tier == PreferenceTier.AVOIDANCE:
            return self._check_avoidance(tool_names, preference)

        # SOFT — rule-based check is unreliable; flag for LLM judge
        return VerifierResult(
            passed=True,
            needs_llm_judge=True,
        )

    def _check_hard(self, tool_names: list[str], preference: ToolPreference) -> VerifierResult:
        if not preference.preferred_tool:
            return VerifierResult(passed=True)
        violations = [t for t in tool_names if t != preference.preferred_tool]
        if violations:
            return VerifierResult(
                passed=False,
                violation=(
                    f"HARD preference violated: called {violations!r} "
                    f"instead of exclusively {preference.preferred_tool!r}"
                ),
            )
        if not tool_names:
            return VerifierResult(
                passed=False,
                violation=f"HARD preference: expected {preference.preferred_tool!r} to be called but trace is empty",
            )
        return VerifierResult(passed=True)

    def _check_avoidance(self, tool_names: list[str], preference: ToolPreference) -> VerifierResult:
        if not preference.forbidden_tool:
            return VerifierResult(passed=True)
        if preference.forbidden_tool in tool_names:
            count = tool_names.count(preference.forbidden_tool)
            return VerifierResult(
                passed=False,
                violation=(
                    f"AVOIDANCE violated: {preference.forbidden_tool!r} "
                    f"called {count} time(s)"
                ),
            )
        return VerifierResult(passed=True)
