# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from responses_api_agents.opencode_multimodal_agent.media import final_answer
from responses_api_agents.opencode_multimodal_agent.tests.test_app import agent
from responses_api_agents.opencode_multimodal_agent.tests.test_eval import evaluation


def test_training_final_plugin_and_eval_budget_remain_separate(monkeypatch, tmp_path):
    training = agent(monkeypatch)
    training_dir = tmp_path / "train"
    training_dir.mkdir()
    training._write_opencode_config(training_dir)
    train_config = json.loads((training_dir / "opencode.json").read_text())
    assert any(".visual-final-answer.mjs" in path for path in train_config["plugin"])
    assert training.config.final_answer_reserve == 2048
    assert train_config["agent"]["build"]["steps"] == 8
    evaluation_agent = evaluation(monkeypatch, tmp_path)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    evaluation_agent._write_opencode_config(eval_dir)
    eval_config = json.loads((eval_dir / "opencode.json").read_text())
    assert evaluation_agent.config.final_answer_reserve is None
    assert not (eval_dir / ".visual-final-answer.mjs").exists()
    assert eval_config["agent"]["build"]["steps"] == 18
    assert all(".visual-final-answer.mjs" not in path for path in eval_config["plugin"])


def test_scoring_cleanup_is_explicit_and_does_not_edit_history():
    text = "orphan reasoning</think>42"
    output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
    assert final_answer(output) == text  # Eval verifier owns its cleanup policy.
    assert final_answer(output, strip_reasoning=True) == "42"
    assert output[0]["content"][0]["text"] == text


def test_advisory_headroom_is_configurable(monkeypatch, tmp_path):
    runner = evaluation(monkeypatch, tmp_path)
    assert runner._build_opencode_config()["agent"]["build"]["steps"] == 18
    runner.config.final_answer_turn_headroom = 0
    assert runner._build_opencode_config()["agent"]["build"]["steps"] == 20
