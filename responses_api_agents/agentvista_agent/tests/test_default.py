# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the original execution loop with a policy/tool/judge fixture."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from responses_api_agents.agentvista_agent.runtime.data import inventory
from responses_api_agents.agentvista_agent.runtime.default_worker import run


@pytest.mark.parametrize("judge", ["0", "1", "provider-error"])
def test_original_loop_tool_feedback_and_judge_status(tmp_path: Path, monkeypatch, judge: str) -> None:
    source = os.environ.get("AGENTVISTA_ROOT")
    if not source:
        pytest.skip("Set AGENTVISTA_ROOT to the pinned upstream source")
    from PIL import Image

    sys.path.insert(0, source)
    # This unit fixture avoids importing Torch just to load a small local image.
    image_loader = ModuleType("qwen_vl_utils")
    image_loader.fetch_image = lambda spec: Image.open(spec["image"]).convert("RGB")
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", image_loader)
    from engine.api import api_model_caller, api_processors
    from tools import get_tool
    from utils.general_qa_tool import GPT4VisionClient

    image = tmp_path / "input.png"
    Image.new("RGB", (320, 200), "red").save(image)
    dataset = tmp_path / "tasks.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "fixture",
                    "problem": "Read the image and search for 42.",
                    "images": [image.name],
                    "solution": "42",
                }
            ]
        )
    )
    calls = []

    def policy(model, messages, sampling, tools):
        calls.append(json.loads(json.dumps(messages)))
        assert os.environ["REASONING_END_POINT"] == "http://gym/rollout/fixture/v1/chat/completions"
        assert "REASONING_API_KEY_2" not in os.environ
        assert any(part.get("type") == "image_url" for part in messages[1]["content"])
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "<think>Draft reasoning must not enter history.</think>",
                "tool_calls": [
                    {
                        "id": "search-1",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "index": 0,
                            "name": "web_search",
                            "arguments": json.dumps({"query": "42"}),
                        },
                    }
                ],
            }
        for message in messages:
            for call in message.get("tool_calls") or []:
                assert "index" not in call
                assert "index" not in call["function"]
        assert any(row.get("role") == "tool" and "Evidence: 42" in str(row.get("content")) for row in messages)
        assert "Draft reasoning" not in json.dumps(messages)
        return "<think>Considering <answer>draft</answer>.</think><answer>42</answer>"

    monkeypatch.setattr(api_model_caller, "call_vision_api", policy)
    monkeypatch.setattr(get_tool("web_search"), "call", lambda *args, **kwargs: "Evidence: 42")
    monkeypatch.setattr(
        GPT4VisionClient,
        "query",
        lambda *args, **kwargs: ("" if judge == "provider-error" else judge, "Explanation: fixture"),
    )
    monkeypatch.setattr(api_processors, "compute_score", api_processors.compute_score)
    for key in (
        "VERIFIER_API_KEY",
        "VERIFIER_END_POINT",
        "VERIFIER_MODEL_NAME",
        "AGENTVISTA_VISIT_API_KEY",
        "AGENTVISTA_VISIT_API_ENDPOINT",
        "AGENTVISTA_VISIT_MODEL_NAME",
    ):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.setenv("SERPAPI_KEY", "fixture")
    monkeypatch.setenv("REASONING_API_KEY_2", "must-be-removed")
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    mounted = tmp_path / "mounted"
    mounted.symlink_to(persistent, target_is_directory=True)
    config = {
        "output": str(mounted / "episode"),
        "source_root": source,
        "dataset": str(dataset),
        "image_folder": str(tmp_path),
        "task_id": "fixture",
        "source_sha256": inventory(dataset, tmp_path)["fixture"]["source_sha256"],
        "endpoint": "http://gym/rollout/fixture/v1",
        "model": "fixture",
        "default_max_turns": 4,
    }
    result = run(config)
    assert result["artifacts"] == str(mounted / "episode")
    assert (persistent / "episode/result.json").is_file()
    assert len(calls) == 2, result
    assert result["native_trajectory"] and result["harness"] == "default"
    assert result["agent_output"] == "42"
    if judge == "provider-error":
        assert result["mask_sample"] and result["verify"]["status"] == "ERROR"
    else:
        assert not result["mask_sample"], result
        assert "failure_kind" not in result
        assert result["reward"] == float(judge)
        assert result["verify"]["status"] == ("PASS" if judge == "1" else "FAIL")
