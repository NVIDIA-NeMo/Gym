# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise saved visual results through Gym's persistent resume owner."""

import copy
import json

import pytest

from nemo_gym.config import ConfigError
from nemo_gym.rollouts import RolloutStore, RunManifest
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request, server  # noqa: F401
from resources_servers.opencode_multimodal_benchmarks.tests.test_grading_evidence import observed


@pytest.mark.asyncio
async def test_cut_resume_preserves_settled_rows_and_rejects_material_drift(tmp_path, server):
    rows = [
        request("VStarBench", "A").model_dump(mode="json")
        | {
            "_ng_task_index": index,
            "_ng_rollout_index": 0,
            "agent_ref": {"name": "visual_agent", "type": "responses_api_agents"},
        }
        for index in range(3)
    ]
    source, output = tmp_path / "input.jsonl", tmp_path / "rollouts.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    global_config = {
        "visual_agent": {
            "responses_api_agents": {
                "opencode_agent": {
                    "model": "vision-fixture",
                    "build_system_prompt": "Answer the question",
                    "resources_server": {"name": "visual", "type": "resources_servers"},
                }
            }
        },
        "visual": {"resources_servers": {"opencode_multimodal_benchmarks": {"judge_model": "judge-a"}}},
    }
    collection = {"num_repeats": 1}

    def prepare():
        return copy.deepcopy(rows), RunManifest.create(source, rows, collection, global_config)

    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, *_ = store.pending(3)
        store.record_dispatch(first)
        scored = await observed(server, request("VStarBench", "A"))
        store.record_outcome(first | scored, sync=True)
    saved = output.read_bytes()
    # A newly introduced operational default and a changed placement port must
    # not replay the already settled trial or change the run identity.
    collection["dispatch_connect_grace_s"] = 60.0
    global_config["visual_agent"]["responses_api_agents"]["opencode_agent"]["port"] = 8123
    with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
        assert [row["_ng_task_index"] for row in resumed.pending(3)] == [1, 2]
        assert output.read_bytes() == saved
    for key, value in [("model", "other-model"), ("build_system_prompt", "Different prompt")]:
        agent = global_config["visual_agent"]["responses_api_agents"]["opencode_agent"]
        original = agent[key]
        agent[key] = value
        with pytest.raises(ConfigError):
            RolloutStore.start_or_resume(output, prepare, resume=True)
        agent[key] = original
        assert output.read_bytes() == saved
    global_config["visual"]["resources_servers"]["opencode_multimodal_benchmarks"]["judge_model"] = "judge-b"
    with pytest.raises(ConfigError):
        RolloutStore.start_or_resume(output, prepare, resume=True)
    assert output.read_bytes() == saved
