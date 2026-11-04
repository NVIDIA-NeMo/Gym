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
    source = Path(__file__).parents[3] / "resources_servers/nemo_user_sim/data/example.jsonl"
    row = json.loads(source.read_text().splitlines()[0])["resolved_row"]
    row["probe_type"] = "identity_disclosure"
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
