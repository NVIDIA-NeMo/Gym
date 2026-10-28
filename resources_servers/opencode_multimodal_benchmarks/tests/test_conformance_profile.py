# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from scripts.harness_conformance.episode import HARNESSES, _config

from responses_api_agents.opencode_multimodal_agent.eval import OpenCodeMultimodalEvalConfig


def test_conformance_uses_eval_profile_and_explicit_sandbox(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "command": "/installed/opencode",
                "sandbox": {
                    "bwrap_command": ["/installed/bwrap"],
                    "python_env": "/installed/python",
                    "plugin_dependencies": "/installed/plugins",
                },
            }
        )
    )
    monkeypatch.setenv("GYM_MULTIMODAL_SANDBOX_CONFIG", str(runtime))
    config = _config("opencode_multimodal", tmp_path, [10001, 10002, 10003, 10004], 60)
    agent = OpenCodeMultimodalEvalConfig.model_validate(
        config["probe_agent"]["responses_api_agents"]["opencode_multimodal_agent"]
    )
    assert HARNESSES["opencode_multimodal"] == "OpenCodeMultimodalEvalAgent"
    assert agent.entrypoint == "eval.py" and agent.command == "/installed/opencode"
    assert agent.max_turns == 65 and agent.turn_budget.regular_turns == 55
    assert agent.final_answer_reserve is None and agent.system_prompt is None
    assert agent.max_images is None and not agent.compaction_auto
    assert agent.resources_server.name == "probe_resources"
    assert agent.build_system_prompt.startswith("You are opencode")
