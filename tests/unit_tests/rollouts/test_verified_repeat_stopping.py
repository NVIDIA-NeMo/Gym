# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollouts import rollout_collection as collection
from nemo_gym.rollouts.repeat_stopping import STOP_ON_SUCCESS_REASON, is_verified_success
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, install_fake_server_client


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"execute_only": True},
        {"reward": True},
        {"reward": float("nan")},
        {"reward": float("inf")},
        {"reward": 1, "mask_sample": True},
        {"reward": 1, "_ng_failure_class": "timeout"},
        {"reward": 1, "_ng_no_persist": True},
        {"reward": 0.01},
    ],
)
def test_only_a_measured_reward_can_stop_a_repeat(result):
    assert not is_verified_success(result, 0.5)
    assert is_verified_success({"reward": 0.5}, 0.5)


@pytest.mark.parametrize("threshold", [True, float("inf"), float("nan")])
def test_threshold_must_be_a_finite_number(threshold):
    with pytest.raises(ValidationError):
        collection.RolloutCollectionConfig(
            input_jsonl_fpath="in", output_jsonl_fpath="out", stop_on_success_threshold=threshold
        )


async def test_resume_after_success_precedes_first_omission(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n")
    config = collection.RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        num_repeats=4,
        stop_on_success_threshold=0.75,
        disable_health_check=True,
        disable_aggregation=True,
    )
    posted = []

    async def post(*, url_path, json, **kwargs):
        assert url_path == "/run"
        posted.append((json[TASK_INDEX_KEY_NAME], json[ROLLOUT_INDEX_KEY_NAME]))
        return FakeResponse(200, {"response": {}, "reward": [0.25, 0.75][json[ROLLOUT_INDEX_KEY_NAME]]})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})

    def interrupt(*args):
        raise RuntimeError("interrupted before recording skip")

    with monkeypatch.context() as crash:
        crash.setattr(RolloutStore, "record_omission", interrupt)
        with pytest.raises(RuntimeError, match="interrupted before recording skip"):
            await collection.RolloutCollectionHelper().run_from_config(config)
    assert [r["reward"] for r in RolloutStore.read(output).selected("success")] == [0.25, 0.75]
    resume = config.model_copy(update={"resume_from_cache": True})
    for _ in range(2):
        await collection.RolloutCollectionHelper().run_from_config(resume)
        store = RolloutStore.read(output)
        assert not store.pending()
        skips = store.selected("omitted")
        assert [r[ROLLOUT_INDEX_KEY_NAME] for r in skips] == [2, 3]
        assert all(r["_ng_failure_message"] == STOP_ON_SUCCESS_REASON and "reward" not in r for r in skips)
        assert posted == [(0, 0), (0, 1)]
