# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance import runner
from scripts.harness_conformance.feedback import failure_details
from scripts.harness_conformance.media_fixture import PROMPT_BYTES, TOOL_BYTES, image_part, prompt_content
from scripts.harness_conformance.provider import Probe
from scripts.harness_conformance.rendering import render_table
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.harness_capabilities.cli import json_rows
from nemo_gym.harness_capabilities.media import image_payloads, inspect_media
from nemo_gym.responses_converter import ResponsesConverter


MEDIA = next(s for s in SCENARIOS if s.media)
READ = {
    "type": "function",
    "name": "read",
    "strict": False,
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}


def chat_content(parts):
    return [
        {"type": "image_url", "image_url": {"url": part["image_url"], "detail": part.get("detail")}}
        if part["type"] == "input_image"
        else {"type": "text", "text": part["text"]}
        for part in parts
    ]


def test_media_probe_uses_native_tool_and_captures_wire_payloads(tmp_path):
    probe = Probe(MEDIA, tmp_path)
    body = {
        "model": "conformance-model",
        "tools": [{"type": "function", "function": {k: v for k, v in READ.items() if k != "type"}}],
        "messages": [{"role": "user", "content": chat_content(prompt_content())}],
    }
    with TestClient(probe.model_app()) as client:
        first = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
        assert first.status_code == 200
        message = first.json()["choices"][0]["message"]
        call = message["tool_calls"][0]
        path = Path(json.loads(call["function"]["arguments"])["path"])
        assert path.read_bytes() == TOOL_BYTES
        body["messages"].extend(
            [
                message,
                {"role": "tool", "tool_call_id": call["id"], "content": "Read image file [image/png]"},
                {"role": "user", "content": chat_content([image_part(path.read_bytes())])},
            ]
        )
        second = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
        assert second.json()["choices"][0]["message"]["content"] == "CONFORMANCE_DONE"
    witness = json.loads((tmp_path / "witness.json").read_text())
    assert witness["media_tool"]["id"] == call["id"] and not witness["tool_calls"]
    captures = [row for _, row in json_rows(tmp_path / "capture/0-0.capture.jsonl")]
    assert [row["request"] for row in captures] == [a["request"] for a in witness["attempts"]]
    checks = {
        c["id"]: c
        for c in inspect_media(
            witness,
            None,
            applies=True,
            prompt_images=tuple(image_payloads([image_part(b) for b in PROMPT_BYTES])),
            tool_image=image_payloads(image_part(TOOL_BYTES))[0],
        )
    }
    for key in ("media.prompt_delivery", "media.tool_delivery", "media.backend_delivery"):
        assert checks[key]["status"] == "pass"


@pytest.mark.parametrize("conversion", ["real", "drop_image", "reject"])
def test_responses_media_exercises_converter_and_witnesses_backend(tmp_path, monkeypatch, conversion):
    original = ResponsesConverter.responses_to_chat_completion_create_params

    def convert(self, *args, **kwargs):
        if conversion == "reject":
            raise NotImplementedError("controlled conversion rejection")
        result = original(self, *args, **kwargs)
        if conversion == "drop_image":
            result.messages[0]["content"].pop(1)
        return result

    monkeypatch.setattr(ResponsesConverter, "responses_to_chat_completion_create_params", convert)
    probe = Probe(MEDIA, tmp_path)
    body = {"model": "conformance-model", "input": [{"role": "user", "content": prompt_content()}], "tools": [READ]}
    with TestClient(probe.model_app()) as client:
        result = client.post("/ng-rollout/0-0/v1/responses", json=body)
    assert result.status_code == (409 if conversion == "reject" else 200)
    assert probe.attempts[0]["request"] == body
    witness = json.loads((tmp_path / "witness.json").read_text())
    checks = {
        c["id"]: c
        for c in inspect_media(
            witness,
            None,
            applies=True,
            prompt_images=tuple(image_payloads([image_part(b) for b in PROMPT_BYTES])),
            tool_image=image_payloads(image_part(TOOL_BYTES))[0],
        )
    }
    assert checks["media.prompt_delivery"]["status"] == "pass"
    assert checks["media.backend_delivery"]["status"] == ("pass" if conversion == "real" else "fail")
    if conversion == "real":
        assert "messages" in witness["backend_requests"][0]
        assert result.json()["object"] == "response"
        assert any(item["type"] == "function_call" for item in result.json()["output"])


@pytest.mark.parametrize("failure", ["evidence", "execution", "checker"])
@pytest.mark.parametrize("p1_only", [False, True])
def test_p1_failures_are_visible_without_changing_p0_gate(tmp_path, monkeypatch, failure, p1_only):
    def execute(command, **kwargs):
        media = command[command.index("--scenario") + 1] == MEDIA.name
        return {"returncode": int(media and failure == "execution"), "timed_out": False}

    def inspect(scenario, directory, execution):
        if scenario.media and failure == "checker":
            raise ValueError("controlled checker failure")
        return {
            "scenario": scenario.name,
            "tier": scenario.tier,
            "verdict": "not_fulfilled" if scenario.media else "fulfilled",
            "exercised": not scenario.media,
            "execution": execution,
            "evidence": {},
            "checks": [],
        }

    monkeypatch.setattr(runner, "run_process", execute)
    monkeypatch.setattr(runner, "inspect_episode", inspect)
    selected = (MEDIA,) if p1_only else (SCENARIOS[0], MEDIA)
    summary, code = runner.run_suite(harnesses=["pi"], scenarios=selected, output=tmp_path / "run", timeout=1)
    expected_code = (2 if failure in ("execution", "checker") else 1) if p1_only else 0
    assert code == expected_code
    row = summary["harnesses"]["pi"]
    assert row["verdict"] == ("not_fulfilled" if p1_only else "fulfilled")
    assert row["scenarios"][-1]["verdict"] == "not_fulfilled"
    assert failure_details("pi", tmp_path / "run")[0] == ("1 / 1" if p1_only else "0 / 1")
    if not p1_only:
        table = render_table({**summary, "gym_commit": "a" * 40})
        assert "| `pi` | Passed | 1/1 | 0/1 |" in table


def test_text_only_artifacts_cannot_pass_media_scenario(tmp_path):
    result = runner.inspect_episode(MEDIA, tmp_path, {"returncode": 0, "timed_out": False})
    assert result["tier"] == "P1" and result["verdict"] == "not_fulfilled"
    assert all(c["status"] == "fail" for c in result["checks"] if c["id"].startswith("media."))
