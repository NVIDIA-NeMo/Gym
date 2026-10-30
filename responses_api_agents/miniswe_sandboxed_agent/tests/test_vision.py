# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import shlex
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import VLLMConverter
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig


HELPER = Path(__file__).parents[1] / "runtime" / "view_image.py"


def bash_response(commands: list[str]) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0,
        object="response",
        model="test",
        parallel_tool_calls=len(commands) > 1,
        tool_choice="auto",
        tools=[],
        output=[
            {
                "type": "function_call",
                "call_id": f"call_{index}",
                "name": "bash",
                "arguments": json.dumps({"command": command}),
            }
            for index, command in enumerate(commands, 1)
        ],
    )


def png_bytes() -> bytes:
    """A valid PNG larger than the native observation text limit."""

    def chunk(name: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))

    pixels = b"".join(b"\0" + hashlib.shake_256(str(row).encode()).digest(256 * 3) for row in range(64))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("enabled", [False, True])
async def test_native_image_command_is_opt_in_and_reaches_model(tmp_path, runner_factory, enabled):
    requests = []

    async def query(params):
        requests.append(params)
        commands = (
            [
                "command -v python3; printf before; view_image 'reference image.png'; "
                "echo after; view_image 'reference image.png'",
                "printf 'second result'",
            ]
            if len(requests) == 1
            else ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
        )
        return bash_response(commands)

    harness = await runner_factory(
        context=HarnessContext(session_id="vision-test", instruction="Inspect the reference image"),
        config=MiniSWEConfig(enable_view_image=enabled),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="test",
        directory=tmp_path / "artifacts",
        observability_enabled=True,
    )
    data = png_bytes()
    Path(harness.context.workdir, "reference image.png").write_bytes(data)
    response, outcome, extra = await harness.execute(30)
    assert outcome.reason == "completed", outcome
    assert extra.get("experimental_features", []) == (["view_image"] if enabled else [])
    assert len(requests) == 2
    config = json.loads((harness.directory / "config.yaml").read_text())
    assert config["model"]["model_class"] == ("vision_model.VisionResponseModel" if enabled else "litellm_response")
    assert (Path(harness.remote_directory) / "image-tools/view_image").exists() == enabled
    assert ("Vision is available through bash" in requests[0]["input"][1]["content"]) == enabled
    images = [
        part
        for item in requests[1]["input"]
        for part in item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    if enabled:
        assert len(images) == 2
        assert base64.b64decode(images[0]["image_url"].split(",", 1)[1]) == data
        tool = next(item for item in requests[1]["input"] if item.get("type") == "function_call_output")
        assert tool["call_id"] == "call_1"
        assert hashlib.sha256(data).hexdigest() in tool["output"]
        assert "before" in tool["output"] and "after" in tool["output"]
        assert json.loads(tool["output"])["output"].splitlines()[0] == shutil.which("python3")
        assert data[:12].hex() not in tool["output"]
        assert base64.b64encode(data).decode("ascii") not in tool["output"]
        results = requests[1]["input"]
        tool_indices = [index for index, item in enumerate(results) if item.get("type") == "function_call_output"]
        image_indices = [
            index
            for index, item in enumerate(results)
            if item.get("role") == "user" and isinstance(item.get("content"), list)
        ]
        assert len(tool_indices) == 2
        assert "second result" in results[tool_indices[1]]["output"]
        assert max(tool_indices) < min(image_indices)
        # Exercise the real Responses -> Chat Completions boundary used by vLLM.
        chat = (
            VLLMConverter(return_token_id_information=False)
            .responses_to_chat_completion_create_params(
                NeMoGymResponseCreateParamsNonStreaming.model_validate(requests[1])
            )
            .model_dump()
        )
        assert any(
            part.get("type") == "image_url"
            for item in chat["messages"]
            if item["role"] == "user"
            for part in item["content"]
            if isinstance(part, dict)
        )
        assert any(
            m.get("role") == "user" and isinstance(m.get("content"), list)
            for m in extra["mini_swe_trajectory"]["messages"]
        )
    else:
        assert images == []
        tool = next(item for item in requests[1]["input"] if item.get("type") == "function_call_output")
        assert json.loads(tool["output"])["returncode"] == 127
    assert response.output


async def test_malformed_image_envelopes_remain_text_and_allow_completion(tmp_path, runner_factory):
    valid_shape = {"_nemo_gym_view_image": 1, "path": "image.png", "sha256": "0" * 64}
    records = [
        {"ordinary": "json"},
        {"_nemo_gym_view_image": 1},
        {**valid_shape, "content": None},
        {**valid_shape, "content": []},
        {**valid_shape, "content": "not a list"},
        {**valid_shape, "content": [{}]},
        {**valid_shape, "content": [None]},
        {**valid_shape, "content": [{"type": "image", "mimeType": "image/png", "data": None}]},
        {**valid_shape, "content": [{"type": "image", "mimeType": "text/plain", "data": "bytes"}]},
        {**valid_shape, "path": None, "content": [{"type": "image", "mimeType": "image/png", "data": "bytes"}]},
        {**valid_shape, "sha256": None, "content": [{"type": "image", "mimeType": "image/png", "data": "bytes"}]},
    ]
    text = "\n".join(json.dumps(record) for record in records)
    requests = []

    async def query(params):
        requests.append(params)
        command = (
            f"printf '%s\\n' {shlex.quote(text)}"
            if len(requests) == 1
            else "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        )
        return bash_response([command])

    harness = await runner_factory(
        context=HarnessContext(session_id="malformed-images", instruction="Read the command output"),
        config=MiniSWEConfig(enable_view_image=True),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="test",
        directory=tmp_path / "artifacts",
    )
    response, outcome, _ = await harness.execute(30)
    assert outcome.reason == "completed", outcome
    assert len(requests) == 2
    tool = next(item for item in requests[1]["input"] if item.get("type") == "function_call_output")
    assert json.loads(tool["output"])["output"].strip() == text
    assert not any(
        part.get("type") == "input_image"
        for item in requests[1]["input"]
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if isinstance(part, dict)
    )
    assert response.output


@pytest.mark.parametrize("failure", ["missing", "text", "oversize"])
def test_image_command_reports_unusable_input(tmp_path, failure):
    path = tmp_path / "image.png"
    if failure == "text":
        path.write_text("This is not an image")
    elif failure == "oversize":
        with path.open("wb") as handle:
            handle.truncate(20 * 1024 * 1024 + 1)
    result = subprocess.run([sys.executable, str(HELPER), str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "view_image:" in result.stderr
