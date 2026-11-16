# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.trajectory import grading_result_binding
from tests.unit_tests.orchestration.test_environment_grading_health import _collect_session, _gate


@pytest.mark.parametrize("base", [0.0, 1.0])
async def test_math_tool_grade_recovery_uses_final_score_and_real_judge_dependencies(tmp_path, monkeypatch, base):
    saved, collected = await _collect_session(tmp_path, monkeypatch, reward=base, verifier="math_tool_rewards")
    assert saved["reward"] == pytest.approx(base + (0.1 if base else 0.02))
    assert "ng_grading_transform" not in saved
    [verifier] = saved["ng_trajectory"]["verifier_invocations"]
    [grade] = verifier["grading_operations"]
    assert grade["outcome"] == "completed"
    assert grade["result_binding_sha256"] == grading_result_binding(saved)
    assert len(grade["dependencies"]) == (2 if base else 1)
    passed, reason, report, counts = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert counts["observed_healthy"] == 1 and counts["unobserved"] == 0
