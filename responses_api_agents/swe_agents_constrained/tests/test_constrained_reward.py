# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from responses_api_agents.swe_agents_constrained.constrained_reward import (
    coerce_constraint_declarations,
    grade_and_shape,
)


class TestCoerceConstraintDeclarations:
    def test_new_schema_passthrough(self):
        raw = [{"type": "unified_diff", "params": {"strict": True}}]
        assert coerce_constraint_declarations(raw) == [{"type": "unified_diff", "params": {"strict": True}}]

    def test_legacy_bare_string(self):
        assert coerce_constraint_declarations(["no_secret_literals_in_code"]) == [
            {"type": "no_secret_literals_in_code", "params": {}}
        ]

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            coerce_constraint_declarations([{"params": {}}])


def _msg(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _tool_call(name: str, call_id: str, args: str = "{}") -> dict:
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": args}


def _tool_output(call_id: str, output: str = "ok") -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


# A small multi-turn agentic trajectory: message -> tool call -> observation ->
# message -> tool call -> observation -> final answer. Messages carry fenced
# code blocks so CODE_STEPS-scoped constraints are applicable, and (when
# intent_tagged) the canonical [INTENT:<VERB>] tag before each tool call.
def _trajectory(intent_tagged: bool) -> list[dict]:
    run_prefix = "[INTENT:RUN] run the failing test\n" if intent_tagged else ""
    edit_prefix = "[INTENT:EDIT] fix the bug\n" if intent_tagged else ""
    return [
        _msg(f"{run_prefix}Let me look at the failing test first.\n```bash\npytest -x tests/\n```"),
        _tool_call("execute_bash", "call_1", json.dumps({"command": "pytest -x tests/"})),
        _tool_output("call_1", "1 failed"),
        _msg(f"{edit_prefix}Now I will fix the bug.\n```python\nraise ImportError(msg)\n```"),
        _tool_call("str_replace_editor", "call_2", json.dumps({"path": "foo.py"})),
        _tool_output("call_2", "edited"),
        _msg("Done. The failing test now passes."),
    ]


def _metadata(constraints: list, **extra: str) -> dict[str, str]:
    return {"constraints": json.dumps(constraints), **extra}


class TestGradeAndShape:
    def test_no_constraints_passthrough(self):
        fields = grade_and_shape(_trajectory(True), {}, task_reward=1.0, default_alpha=1.0)
        assert fields["reward"] == 1.0
        assert fields["constraint_graded"] is False
        assert fields["constraint_reward"] is None
        assert fields["reward_components"] == {"task": 1.0}

    def test_compliant_trajectory_doubles_reward(self):
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        fields = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        assert fields["constraint_graded"] is True
        assert fields["constraint_reward"] == 1.0
        assert fields["reward"] == 2.0  # task * (1 + 1.0 * 1.0)

    def test_zero_task_reward_blocks_constraint_reward(self):
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        fields = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=0.0, default_alpha=1.0)
        assert fields["constraint_reward"] == 1.0
        assert fields["reward"] == 0.0  # no constraint reward hacking

    def test_violated_constraint_keeps_task_gradient(self):
        # tool_call_intent_tag: every tool call must be preceded by an
        # [INTENT:<VERB>] tagged intent line.
        constraints = [{"type": "tool_call_intent_tag", "params": {}}]
        compliant = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        violating = grade_and_shape(_trajectory(False), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        assert compliant["constraint_reward"] > violating["constraint_reward"]
        assert violating["reward"] >= 1.0  # task reward survives constraint violation
        assert compliant["reward"] > violating["reward"]

    def test_alpha_override_from_metadata(self):
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        fields = grade_and_shape(
            _trajectory(True),
            _metadata(constraints, constraint_alpha="0.5"),
            task_reward=1.0,
            default_alpha=1.0,
        )
        assert fields["constraint_alpha"] == 0.5
        assert fields["reward"] == 1.5

    def test_list_typed_constraints_tolerated(self):
        # Older generated files carry constraints as a native list rather than
        # a JSON string; grading must accept both.
        fields = grade_and_shape(
            _trajectory(True),
            {"constraints": [{"type": "no_force_git_commands", "params": {}}], "constraint_alpha": "1.0"},
            task_reward=1.0,
            default_alpha=1.0,
        )
        assert fields["constraint_graded"] is True
        assert fields["reward"] == 2.0

    def test_grading_error_passes_task_reward_through(self):
        fields = grade_and_shape(
            _trajectory(True),
            {"constraints": "not-valid-json"},
            task_reward=1.0,
            default_alpha=1.0,
        )
        assert fields["reward"] == 1.0
        assert fields["constraint_graded"] is False
        assert any("constraint grading error" in v for v in fields["violations"])

    def test_reward_components_include_per_constraint_scores(self):
        constraints = [
            {"type": "no_force_git_commands", "params": {}},
            {"type": "no_secret_literals_in_code", "params": {}},
        ]
        fields = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        assert "constraint_no_force_git_commands" in fields["reward_components"]
        assert "constraint_no_secret_literals_in_code" in fields["reward_components"]
