# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The saved verification states whether the judge measured the patch (gym-p0/v1 TE-6)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from environment_servers.swe_together import app
from environment_servers.swe_together.schemas import SWETResult
from environment_servers.swe_together.tests.test_environment import activation, server, state
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


LONG_PATCH = "diff --git a/x b/x\n" + "+line\n" * 40


async def verify(tmp_path, monkeypatch, *, patch=LONG_PATCH, judge=None):
    environment = server(tmp_path)
    environment.server_client = SimpleNamespace()
    episode, _ = state(tmp_path, [])
    episode.snapshots.final_patch = patch
    episode.snapshots.baseline = {}
    episode.sandbox = SimpleNamespace()
    episode.task_metadata = {"judge_timeout": 10}
    monkeypatch.setattr(app.SWETEnvironmentServer, "_new_sandbox", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(app, "replicate_submission", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "run_judge", judge or AsyncMock(side_effect=AssertionError("judge must not run")))
    monkeypatch.setattr(app, "interaction_metrics", AsyncMock(return_value={}))
    turns = [activation()]
    initial = NeMoGymResponseCreateParamsNonStreaming(input="task")
    result = await environment._verify(episode, initial, turns)
    saved = json.loads((tmp_path / "verification.json").read_text())
    return SWETResult.model_validate(result).model_dump(mode="json"), saved


@pytest.mark.parametrize("score,reward", [(0.9, 1.0), (0.0, 0.0)])
async def test_judged_patch_is_a_completed_measurement(tmp_path, monkeypatch, score, reward):
    record, saved = await verify(tmp_path, monkeypatch, judge=AsyncMock(return_value={"judge_score": score}))
    for value in (record, saved):
        assert value["evaluation_completed"] is True and value["mask_sample"] is False
        assert value["judge_score"] == score and value["reward"] == reward
        assert value["failure_kind"] is None


async def test_short_patch_zero_is_a_completed_measurement(tmp_path, monkeypatch):
    record, saved = await verify(tmp_path, monkeypatch, patch="tiny")
    for value in (record, saved):
        assert value["evaluation_completed"] is True and value["mask_sample"] is False
        assert value["judge_score"] == 0.0 and value["reward"] == 0.0
        assert value["verdict"]["reason"] == "short_patch"


async def test_failed_judge_is_an_incomplete_masked_measurement(tmp_path, monkeypatch):
    judge = AsyncMock(side_effect=RuntimeError("no readable verdict"))
    record, saved = await verify(tmp_path, monkeypatch, judge=judge)
    for value in (record, saved):
        assert value["evaluation_completed"] is False and value["mask_sample"] is True
        assert value["judge_score"] is None
        assert value["failure_kind"] == "judge_failed"
        assert "no readable verdict" in value["failure_reason"]


def test_result_without_completion_status_is_rejected():
    response = activation().response
    with pytest.raises(ValidationError, match="evaluation_completed"):
        SWETResult.model_validate(
            {"responses_create_params": {"input": "x"}, "response": response, "reward": 0.0, "task_id": "task"}
        )
