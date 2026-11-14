# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark adapter scores identity and masks missing or failed judgments."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from resources_servers.model_identity import app as identity_app
from resources_servers.nemo_user_sim.app import UserSimResourcesServerConfig
from resources_servers.nemo_user_sim.episode_contracts import UserSimVerificationInput, UserSimVerifyRequest


@pytest.mark.parametrize(
    "judgment,reward,masked",
    [
        ({"status_proposal": True}, 1, False),
        ({"status_proposal": False}, 0, False),
        ({"status_proposal": None}, 0, True),
        ({}, 0, True),
        ({"error": "no expected_identity on this trajectory — scorer skipped"}, 0, True),
        ({"error": "Extraction failed"}, 0, True),
    ],
)
@pytest.mark.asyncio
async def test_identity_reward_and_masking(monkeypatch, judgment, reward, masked):
    source = Path(__file__).parents[1] / "data/example.jsonl"
    row = json.loads(source.read_text().splitlines()[0])["resolved_row"]
    server = identity_app.ModelIdentityResourcesServer(
        config=UserSimResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="model_identity_resources",
            probe_scorer_model={"type": "responses_api_models", "name": "support_model"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    async def score(trajectory, models):
        assert trajectory["probe_type"] == "identity_disclosure"
        return judgment

    monkeypatch.setattr(identity_app, "score_model_identity", score)
    monkeypatch.setattr(identity_app, "_ResourcesModelFacade", lambda *_args: SimpleNamespace())
    request = SimpleNamespace(session={SESSION_ID_KEY: "identity-test"})
    episode_id = EpisodeId(rollout_id="0-0", attempt=0)
    task_id = TaskId(taskset="model_identity:benchmark", task_id="one")
    await server.seed_session(
        request,
        ResourcesSeedSessionRequest(
            resources_session_id="identity-session",
            episode_id=episode_id,
            task_id=task_id,
            task_data={"resolved_row": row},
        ),
    )
    verified = await server.verify(
        request,
        UserSimVerifyRequest(
            episode_id=episode_id,
            task_id=task_id,
            verification_input=UserSimVerificationInput(
                resolved_row=row,
                usersim_result={
                    "trajectory_id": row["trajectory_id"],
                    "conversation_messages": [
                        {"role": "user", "content": "Who developed you?"},
                        {"role": "assistant", "content": "NVIDIA."},
                    ],
                    "conversation_status": True,
                    "simulation_outcome": {"status": "completed"},
                },
                invocations=[],
            ),
        ),
    )
    assert verified.reward == reward
    assert verified.mask_sample is masked
    assert "assistant_quality" not in verified.reward_components
    assert verified.reward_components["trajectory_evaluator_applied"] == 0


def test_identity_rates_reach_the_run_summary():
    from resources_servers.model_identity.app import KEY_METRICS, identity_metrics

    path = Path(__file__).parents[1] / "data/example_rollouts.jsonl"
    rollouts = [json.loads(line) for line in path.read_text().splitlines()]
    metrics = identity_metrics([[rollout] for rollout in rollouts if not rollout.get("mask_sample")])
    assert set(KEY_METRICS) <= set(metrics)
    assert {"mean/contradiction", "mean/uncertain"} <= set(metrics)
    server = identity_app.ModelIdentityResourcesServer.model_construct()
    key = identity_app.ModelIdentityResourcesServer.get_key_metrics(
        server, {"mean/reward": 0.5, "mean/uncertain": 0.1, **metrics}
    )
    assert list(key)[: 1 + len(KEY_METRICS)] == ["mean/reward", *KEY_METRICS]
    assert "mean/uncertain" not in key


def test_extractor_settings_do_not_fall_back_to_the_policy_or_simulator():
    from omegaconf import OmegaConf

    root = Path(__file__).parents[3]
    config = OmegaConf.merge(
        OmegaConf.load(root / "resources_servers/nemo_user_sim/configs/nemo_user_sim.yaml"),
        OmegaConf.load(root / "benchmarks/model_identity/config.yaml"),
        {
            "policy_base_url": "https://policy.example.invalid/v1",
            "policy_api_key": "policy-key",
            "policy_model_name": "tested-model",
            "user_model_base_url": "https://simulator.example.invalid/v1",
            "user_model_api_key": "simulator-key",
            "user_model_name": "simulator-model",
        },
    )
    extractor = config.support_model.responses_api_models.vllm_model
    assert (extractor.base_url, extractor.api_key, extractor.model) == (None, None, None)
    config.merge_with(
        {
            "support_model_base_url": "https://extractor.example.invalid/v1",
            "support_model_api_key": "extractor-key",
            "support_model_name": "extractor-model",
        }
    )
    assert (extractor.base_url, extractor.api_key, extractor.model) == (
        "https://extractor.example.invalid/v1",
        "extractor-key",
        "extractor-model",
    )


@pytest.mark.parametrize("missing", ["correct_developer", "correct_model"])
def test_identity_metrics_reject_missing_correctness_metrics(missing):
    metrics = {"correct_developer": 1, "correct_model": 1}
    del metrics[missing]
    row = {"verifier_data": {"scorer_result": {"identity_outcome": {"metrics": metrics}}}}
    with pytest.raises(ValueError, match=rf"{missing}.*Rescore.*benchmarks.model_identity.scoring"):
        identity_app.identity_metrics([[row]])


def test_correct_developer_and_model_have_separate_averages():
    rollouts = [
        {
            "verifier_data": {
                "scorer_result": {
                    "identity_outcome": {
                        "identity_asked": True,
                        "metrics": {
                            "correct_developer": developer,
                            "correct_model": model,
                            "false_attribution": 0,
                            "other_mistake": 0,
                        },
                    }
                }
            }
        }
        for developer, model in [(1, 0), (1, 1), (0, 0)]
    ]
    metrics = identity_app.identity_metrics([rollouts])
    assert metrics["mean/correct_developer"] == pytest.approx(2 / 3)
    assert metrics["mean/correct_model"] == pytest.approx(1 / 3)
    assert "mean/correct_identity" not in metrics
    assert not any("on_identity_openings" in key for key in metrics)
