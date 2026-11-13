# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise original tools through the adapter; external providers are local fixtures."""

import json
import os
import sys
from types import SimpleNamespace

import pytest


if not os.environ.get("AGENTVISTA_ROOT"):
    pytest.skip("Set AGENTVISTA_ROOT to the pinned upstream source", allow_module_level=True)
sys.path.insert(0, os.environ["AGENTVISTA_ROOT"])

from PIL import Image

from responses_api_agents.agentvista_agent.runtime.data import dataset_rows, inventory
from responses_api_agents.agentvista_agent.runtime.mcp import ToolSession
from responses_api_agents.agentvista_agent.runtime.scoring import score
from responses_api_agents.agentvista_agent.runtime.worker import _execution_health


@pytest.mark.parametrize(
    "returncode,timed_out,native_error,termination",
    [(1, False, False, "process_exit"), (-15, True, False, "timeout"), (0, False, True, "native_error")],
)
def test_budget_finish_does_not_hide_execution_failures(returncode, timed_out, native_error, termination):
    events = [{"type": "step_finish", "part": {"reason": "length", "tokens": {"output": 100}}}]
    if native_error:
        events.append({"type": "error", "error": {"name": "APIError"}})
    health = _execution_health(events, returncode, timed_out, 100)
    assert health["termination"] == termination
    assert not health["execution_healthy"]


@pytest.fixture
def episode(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (320, 200), "red").save(image)
    dataset = tmp_path / "tasks.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "fixture",
                    "problem": "Read the image",
                    "images": [image.name],
                    "solution": "red",
                }
            ]
        )
    )
    return dataset, image


def test_task_image_binding_and_no_answer_in_gym_row(episode):
    dataset, image = episode
    tasks = inventory(dataset, image.parent)
    row = dataset_rows(tasks)[0]
    assert "solution" not in json.dumps(row) and "red" not in json.dumps(row)
    initial = tasks["fixture"]["source_sha256"]
    Image.new("RGB", (320, 200), "blue").save(image)
    assert inventory(dataset, image.parent)["fixture"]["source_sha256"] != initial
    with pytest.raises(ValueError, match="unknown"):
        inventory(dataset, image.parent, "missing")


def test_original_tool_schemas_and_text_search_outputs(monkeypatch, tmp_path, episode):
    import requests
    from tools import get_tool_info

    monkeypatch.setenv("SERPAPI_KEY", "fixture-key")
    dataset, image = episode
    target = tmp_path / "persistent"
    target.mkdir()
    mounted = tmp_path / "mounted"
    mounted.symlink_to(target, target_is_directory=True)
    session = ToolSession({"output": str(mounted), "images": [str(image)], "web_search_max_calls": 1})
    assert session.output == mounted
    for schema in session.schema():
        assert schema == get_tool_info(schema["name"])
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        payload = {
            "organic": [{"title": "Fixture", "link": "https://example.com", "snippet": "Evidence"}],
            "images": [
                {
                    "title": "Red",
                    "imageUrl": "https://example.com/red.jpg",
                    "link": "https://example.com",
                }
            ],
            "organic_results": [{"title": "Match", "link": "https://example.com", "snippet": "match"}],
        }
        return SimpleNamespace(status_code=200, json=lambda: payload, text=json.dumps(payload))

    monkeypatch.setattr(requests, "post", post)
    web = session.execute("web_search", {"query": "fixture"})
    assert "Evidence" in web[0]["text"]
    assert "Maximum number" in session.execute("web_search", {"query": "again"})[0]["text"]
    text = session.execute("image_search", {"query": "red"})
    assert len(text) == 1 and "red.jpg" in text[0]["text"]
    from tools.image_search import ImageSearch

    uploads = []

    def upload(self, path, **kwargs):
        uploads.append(path)
        return "https://example.com/upload.jpg"

    monkeypatch.setattr(ImageSearch, "_upload_local_image", upload)
    reverse = session.execute("image_search", {"search_type": "reverse", "image_url": "original_image"})
    assert uploads and all(item["type"] == "text" for item in reverse)
    assert sent[-1][0].endswith("/lens")
    assert json.loads(sent[-1][1]["data"])["url"] == "https://example.com/upload.jpg"
    Image.new("RGB", (10, 10), "red").save(mounted / "saved.jpg")
    session.execute("image_search", {"search_type": "reverse", "image_url": "saved.jpg"})
    assert uploads[-1] == str(mounted / "saved.jpg")
    (mounted / "escape.jpg").symlink_to(image)
    with pytest.raises(ValueError, match="outside episode"):
        session.execute("image_search", {"search_type": "reverse", "image_url": "escape.jpg"})


def test_visit_preserves_auxiliary_summarizer(monkeypatch, tmp_path):
    import tools.visit as visit
    from tools.visit import Visit

    monkeypatch.setenv("AGENTVISTA_VISIT_API_KEY", "fixture-key")
    monkeypatch.setenv("AGENTVISTA_VISIT_API_ENDPOINT", "https://example.com/v1/chat/completions")
    monkeypatch.setenv("AGENTVISTA_VISIT_MODEL_NAME", "fixed-helper")
    monkeypatch.setattr(visit, "JINA_AVAILABLE", True)
    monkeypatch.setattr(Visit, "_jina_readpage", lambda *args: "Original page evidence")
    calls = []

    def summarize(self, content, goal, url):
        calls.append((self.model_name, content, goal, url))
        return "Summary with evidence"

    monkeypatch.setattr(Visit, "_summarize_with_api", summarize)
    session = ToolSession({"output": str(tmp_path), "images": []})
    result = session.execute("visit", {"url": "https://example.com", "goal": "find evidence"})
    assert calls == [("fixed-helper", "Original page evidence", "find evidence", "https://example.com")]
    assert len(result) == 1 and "Summary with evidence" in result[0]["text"]


def test_real_kernel_state_image_output_and_reverse_reference(monkeypatch, tmp_path, episode):
    from tools.code_interpreter import _kill_kernels_and_subprocesses

    dataset, image = episode
    monkeypatch.setenv("SERPAPI_KEY", "fixture-key")
    monkeypatch.setenv("VERIFIER_API_KEY", "HOST_JUDGE_CANARY")
    session = ToolSession({"output": str(tmp_path / "tools"), "images": [str(image)]})
    try:
        first = session.execute(
            "code_interpreter",
            {"code": "counter = 40\nprint(original_image.size)\noriginal_image.crop((0, 0, 100, 100))"},
        )
        assert any(item["type"] == "image" for item in first)
        assert "tool_image_1" in session.image_map
        second = session.execute("code_interpreter", {"code": "print(counter + 2)\nprint(tool_image_1.size)"})
        assert "42" in second[0]["text"] and "100" in second[0]["text"]
        # Attack the complete MCP -> proxy -> Gym -> namespace -> real kernel path.
        # This would expose all three on the original same-user host kernel.
        reference = tmp_path / "private-reference.json"
        reference.write_text("HOST_REFERENCE_CANARY")
        code = (
            "import os,json\nfrom pathlib import Path\n"
            + f"print(json.dumps({{'reference_visible': Path({str(reference)!r}).exists(), "
            + "'judge_visible': 'VERIFIER_API_KEY' in os.environ, "
            + f"'host_process_visible': Path('/proc/{os.getpid()}/environ').exists()}}))"
        )
        evidence = session.execute("code_interpreter", {"code": code})
        text = json.loads(evidence[0]["text"])["result"]
        assert '"reference_visible": false' in text
        assert '"judge_visible": false' in text
        assert '"host_process_visible": false' in text
        assert "HOST_REFERENCE_CANARY" not in text and "HOST_JUDGE_CANARY" not in text
        other = ToolSession({"output": str(tmp_path / "other"), "images": [str(image)]})
        isolated = other.execute("code_interpreter", {"code": "print('counter' in globals())"})
        assert "False" in isolated[0]["text"]
    finally:
        session.close()
        if "other" in locals():
            other.close()
        _kill_kernels_and_subprocesses()


@pytest.mark.parametrize("part", [""])
def test_verifier_failure_is_unscored(monkeypatch, part):
    from utils.general_qa_tool import GPT4VisionClient

    for key in ("VERIFIER_API_KEY", "VERIFIER_END_POINT", "VERIFIER_MODEL_NAME"):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.setattr(GPT4VisionClient, "__init__", lambda self: None)
    monkeypatch.setattr(GPT4VisionClient, "query", lambda *args, **kwargs: (part, "provider failure"))
    with pytest.raises(RuntimeError):
        score("question", "answer", ["prediction"])


@pytest.mark.parametrize("value", ["0", "1"])
def test_judged_zero_and_one_preserve_original_answer_extraction(monkeypatch, value):
    from utils.general_qa_tool import GPT4VisionClient

    for key in ("VERIFIER_API_KEY", "VERIFIER_END_POINT", "VERIFIER_MODEL_NAME"):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.setattr(GPT4VisionClient, "__init__", lambda self: None)
    monkeypatch.setattr(GPT4VisionClient, "query", lambda *args, **kwargs: (value, "Explanation: fixture"))
    judged = score("question", "answer", ["thinking <answer>final</answer>"])
    assert judged["prediction"] == "final" and judged["reward"] == float(value)
    assert judged["deterministically_verified"] is False
