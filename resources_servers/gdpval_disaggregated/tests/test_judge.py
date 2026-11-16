# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared rubric scorer must produce a valid measurement before a rollout is published."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from resources_servers.gdpval.scoring import SCORING_ERROR_KEY
from resources_servers.gdpval_disaggregated import gym_bridge
from resources_servers.gdpval_disaggregated.gym_bridge import JudgeConfig, judge, read_artifacts, valid_reward
from resources_servers.gdpval_disaggregated.tests.test_app import body


@pytest.fixture
def task(tmp_path: Path) -> Path:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/greeting.txt").write_text("Hello")
    (tmp_path / "trajectory.json").write_text(body().model_dump_json())
    (tmp_path / "judge.json").write_text(
        json.dumps(
            {
                "task_id": "sample",
                "prompt": "Write a greeting",
                "rubric_pretty": "A greeting exists",
            }
        )
    )
    (tmp_path / "artifact-manifest.json").write_text('[{"path": "greeting.txt", "bytes": 5}]')
    return tmp_path


def config(**updates: object) -> JudgeConfig:
    return JudgeConfig(judge_url="http://judge/v1", judge_model="judge", judge_timeout=10, **updates)


def judgement(scores: list[float], maxima: list[float]) -> dict[str, object]:
    return {
        "scoring_method": "structured_rubric",
        "num_trials_completed": len(scores),
        "scores": scores,
        "max_possible_scores": maxima,
    }


@pytest.mark.parametrize("reward", [0.0, 0.5, 1.0])
def test_valid_structured_score_includes_zero(reward):
    assert valid_reward(reward, judgement([reward * 2], [2]))


@pytest.mark.parametrize(
    "reward,metadata",
    [
        (float("nan"), judgement([1], [2])),
        (True, judgement([1], [2])),
        (0, judgement([], [])),
        # "0 out of 0": a malformed judgement, not a measured zero.
        (0.0, judgement([0], [0])),
        # A score above its maximum is clipped to 1.0 by the scorer; it is not a valid measurement.
        (1.0, judgement([5], [2])),
        (0.0, judgement([-1], [2])),
        (0.5, {"scoring_method": "structured_rubric", "num_trials_completed": 1, "scores": [1]}),
        (0, {SCORING_ERROR_KEY: "no_valid_scores"}),
        (0, None),
    ],
)
def test_invalid_judgement_is_not_a_policy_zero(reward, metadata):
    assert not valid_reward(reward, metadata)


async def test_real_artifacts_use_one_structured_trial(task, monkeypatch):
    scorer = AsyncMock(return_value=(0.5, judgement([1], [2])))
    monkeypatch.setattr(gym_bridge, "score_with_rubric_structured", scorer)
    await judge(task, config())
    kwargs = scorer.await_args.kwargs
    assert kwargs["num_trials"] == 1 and kwargs["formatting_retries"] == 2
    assert "Hello" in kwargs["deliverable_text"]
    assert kwargs["deliverable_content_blocks"]
    result = json.loads((task / "rollout.json").read_text())
    assert result["reward"] == 0.5 and not result["mask_sample"]


async def test_invalid_judge_does_not_publish_rollout(task, monkeypatch):
    monkeypatch.setattr(
        gym_bridge, "score_with_rubric_structured", AsyncMock(return_value=(0, {SCORING_ERROR_KEY: "no_valid_scores"}))
    )
    with pytest.raises(ValueError, match="Invalid GDPVal judgement"):
        await judge(task, config())
    assert not (task / "rollout.json").exists()
    assert (task / "judge-response.json").exists()


async def test_nonempty_deliverables_cannot_be_declared_empty(task):
    with pytest.raises(ValueError, match="disagrees"):
        await judge(task, config(empty_submission_zero_reward=True))
    assert not (task / "rollout.json").exists()


def test_nested_deliverables_are_read_with_directory_labels(task):
    (task / "artifacts/nested").mkdir()
    (task / "artifacts/nested/greeting.txt").write_text("Bonjour")
    text, blocks = read_artifacts(task / "artifacts", config())
    assert "Hello" in text and "Bonjour" in text
    assert "=== nested__greeting.txt ===" in text and "=== greeting.txt ===" in text
    serialized = json.dumps(blocks)
    assert "Hello" in serialized and "Bonjour" in serialized and "nested__greeting.txt" in serialized


def test_deliverables_with_run_state_names_are_graded(task):
    # This directory holds only deliverables, so names the shared reader reserves for run state are work product.
    (task / "artifacts/metadata.json").write_text('{"marker": "metadata-marker"}')
    (task / "artifacts/log.txt").write_text("log-marker")
    text, blocks = read_artifacts(task / "artifacts", config())
    serialized = json.dumps(blocks)
    for marker in ("metadata-marker", "log-marker", "Hello"):
        assert marker in text and marker in serialized


def test_nested_text_shares_one_submission_budget(task, monkeypatch):
    from resources_servers.gdpval import file_reader

    monkeypatch.setattr(file_reader, "MAX_TOTAL_CHARS", 1_000)
    for index in range(4):
        (task / f"artifacts/part{index}").mkdir()
        (task / f"artifacts/part{index}/notes.txt").write_text("x" * 900)
    text, _ = read_artifacts(task / "artifacts", config())
    assert len(text) <= 1_000


def test_label_collision_refuses_partial_judgement(task):
    (task / "artifacts/nested").mkdir()
    (task / "artifacts/nested/a.txt").write_text("nested")
    (task / "artifacts/nested__a.txt").write_text("flat")
    with pytest.raises(ValueError, match="same label"):
        read_artifacts(task / "artifacts", config())


def test_media_conversion_error_refuses_partial_judgement(task, monkeypatch):
    monkeypatch.setattr(
        gym_bridge,
        "convert_deliverables_to_content_blocks",
        lambda *a, **kw: [
            {"type": "text", "text": "\nreport.pdf: [Error: cannot render]"},
        ],
    )
    with pytest.raises(ValueError, match="Unreadable GDPVal deliverable"):
        read_artifacts(task / "artifacts", config())


async def test_artifact_failure_leaves_typed_diagnostic(task, monkeypatch):
    def fail(*args):
        raise ValueError("unreadable file")

    monkeypatch.setattr(gym_bridge, "read_artifacts", fail)
    with pytest.raises(ValueError, match="unreadable file"):
        await judge(task, config())
    assert json.loads((task / "judge-failure.json").read_text()) == {
        "kind": "artifact_processing",
        "error_type": "ValueError",
    }
    assert not (task / "rollout.json").exists()


def test_text_submission_remains_readable_without_office_runtime(task):
    text, blocks = read_artifacts(task / "artifacts", config(office_rendering_available=False))
    assert "Hello" in text
    assert "Hello" in json.dumps(blocks)
