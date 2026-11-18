# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Images through the in-sandbox runner: the pixels each model call carries, and what the trajectory keeps.

The stub model server records every request, so each test checks the exact image parts the model was sent (decoded,
compared pixel for pixel with the file or the row's image) and that the trajectory holds the same data URL, in the
order the model got it.
"""

import asyncio
import base64
import io
import json
from pathlib import Path

import pytest
import stirrup.core.agent as stirrup_agent
from PIL import Image

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse
from responses_api_agents.stirrup_agent.app import _observations
from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent
from responses_api_agents.stirrup_agent.sandbox_runner import Observer, run
from responses_api_agents.stirrup_agent.tests.fakes import _call, _ModelServer


@pytest.fixture(autouse=True)
def restore_run_tool(monkeypatch):
    """The runner replaces ``run_tool`` on Stirrup's agent classes for its process; undo that after each test."""
    for cls in (stirrup_agent.Agent, NeMoAgent):
        monkeypatch.setattr(cls, "run_tool", cls.run_tool)


def _png(color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> bytes:
    image = Image.new("RGB", size, color)
    # A second color in one corner, so a transposed or mirrored image does not compare equal.
    image.paste((255 - color[0], 255 - color[1], 255 - color[2]), (0, 0, size[0] // 4, size[1] // 4))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _pixels(data: bytes) -> tuple[tuple[int, int], bytes]:
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        return rgb.size, rgb.tobytes()


def _decoded(url: str) -> bytes:
    assert url.startswith("data:image/png;base64,"), url[:40]
    return base64.b64decode(url.split(",", 1)[1])


def _sent_images(request: dict) -> list[tuple[int, str, str]]:
    """(message index, role, data URL) of every image part in one chat request, in order."""
    return [
        (index, message["role"], part["image_url"]["url"])
        for index, message in enumerate(request["messages"])
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


def _kept_images(items: list[dict]) -> list[str]:
    """The data URL of every input_image part in trajectory items, in order."""
    urls = []
    for item in items:
        content = item.get("output") if item.get("type") == "function_call_output" else item.get("content")
        if isinstance(content, list):
            urls += [part["image_url"] for part in content if part.get("type") == "input_image"]
    return urls


def _payload(model_url: str, workdir: Path, content, *, view_image: bool = True) -> dict:
    return {
        "input": [{"role": "user", "content": content}],
        "instructions": None,
        "tools": [],
        "finish_tool_names": None,
        "code_exec": True,
        "view_image": view_image,
        "tool_access": None,
        "workdir": str(workdir),
        "max_turns": 10,
        "min_compaction_summary_words": 1,
        "client": {"model": "policy", "base_url": model_url, "max_tokens": 32768, "temperature": 0.0},
    }


def _run(tmp_path: Path, content, answers: list, *, view_image: bool = True) -> tuple[dict, _ModelServer]:
    workdir = tmp_path / "root"
    workdir.mkdir(exist_ok=True)
    model = _ModelServer(answers)
    try:
        output = asyncio.run(
            run(_payload(model.base_url, workdir, content, view_image=view_image), tmp_path, Observer())
        )
    finally:
        model.close()
    return output, model


FINISH = _call("finish", json.dumps({"reason": "done", "paths": []}))


def _two_calls(*paths: str) -> dict:
    calls = [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {"name": "view_image", "arguments": json.dumps({"path": p})},
        }
        for i, p in enumerate(paths)
    ]
    return {"role": "assistant", "content": None, "tool_calls": calls}


def test_an_initial_image_reaches_the_first_model_call_and_the_trajectory(tmp_path):
    red = _png((200, 30, 30))
    content = [
        {"type": "input_image", "image_url": _data_url(red)},
        {"type": "input_text", "text": "What color is it?"},
    ]

    output, model = _run(tmp_path, content, [FINISH])

    [(index, role, url)] = _sent_images(model.requests[0])
    assert role == "user"
    assert _pixels(_decoded(url)) == _pixels(red)
    assert _kept_images(output["input_items"]) == [url]
    [user] = [item for item in output["input_items"] if item["role"] == "user"]
    assert [part["type"] for part in user["content"]] == ["input_image", "input_text"]


def test_an_image_viewed_after_a_tool_call_reaches_the_next_model_call_and_the_trajectory(tmp_path):
    (tmp_path / "root").mkdir()
    receipt = _png((30, 120, 200))
    (tmp_path / "root" / "receipt.png").write_bytes(receipt)

    output, model = _run(
        tmp_path, "Check the receipt.", [_call("view_image", json.dumps({"path": "receipt.png"})), FINISH]
    )

    assert "view_image" in [tool["function"]["name"] for tool in model.requests[0]["tools"]]
    assert _sent_images(model.requests[0]) == []
    [(index, role, url)] = _sent_images(model.requests[1])
    messages = model.requests[1]["messages"]
    # The tool's text answer first, then the image in a user message, as Stirrup's text_only_tool_responses sends it.
    assert role == "user" and messages[index - 1]["role"] == "tool"
    assert messages[index - 1]["tool_call_id"] == "call-view_image"
    assert _pixels(_decoded(url)) == _pixels(receipt)

    kinds = [(item["type"], item.get("role") or item.get("name")) for item in output["output_items"]]
    assert kinds[:3] == [("function_call", "view_image"), ("function_call_output", None), ("message", "user")]
    assert _kept_images(output["output_items"]) == [url]
    assert output["observations"]["tool_calls"][0]["status"] == "completed"


@pytest.mark.parametrize("one_turn", [False, True], ids=["two-turns", "one-turn"])
def test_two_images_keep_their_order(tmp_path, one_turn):
    (tmp_path / "root").mkdir()
    first, second = _png((250, 200, 0)), _png((0, 160, 80), size=(48, 64))
    (tmp_path / "root" / "a.png").write_bytes(first)
    (tmp_path / "root" / "b.png").write_bytes(second)
    if one_turn:
        answers = [_two_calls("a.png", "b.png"), FINISH]
    else:
        answers = [
            _call("view_image", json.dumps({"path": "a.png"})),
            _call("view_image", json.dumps({"path": "b.png"})),
            FINISH,
        ]

    output, model = _run(tmp_path, "Compare the two.", answers)

    sent = [url for _, _, url in _sent_images(model.requests[-1])]
    assert [_pixels(_decoded(url)) for url in sent] == [_pixels(first), _pixels(second)]
    assert _kept_images(output["output_items"]) == sent


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [("missing.png", None, "not found"), ("notes.png", b"total: 12 boxes\n", "Unsupported")],
    ids=["missing", "not-an-image"],
)
def test_a_missing_or_invalid_image_is_a_failed_call_and_no_pixels_reach_the_model(tmp_path, name, data, message):
    (tmp_path / "root").mkdir()
    if data is not None:
        (tmp_path / "root" / name).write_bytes(data)

    output, model = _run(tmp_path, "Check it.", [_call("view_image", json.dumps({"path": name})), FINISH])

    assert all(_sent_images(request) == [] for request in model.requests)
    [result] = [item for item in output["output_items"] if item["type"] == "function_call_output"][:1]
    assert message in result["output"]
    assert _kept_images(output["output_items"]) == []
    assert output["observations"]["tool_calls"][0]["status"] == "failed"


def test_a_large_image_reaches_the_model_downscaled_to_one_megapixel_and_the_trajectory_keeps_what_it_got(tmp_path):
    (tmp_path / "root").mkdir()
    scan = _png((240, 240, 230), size=(1600, 1200))
    (tmp_path / "root" / "scan.png").write_bytes(scan)

    output, model = _run(tmp_path, "Read it.", [_call("view_image", json.dumps({"path": "scan.png"})), FINISH])

    [(_, _, url)] = _sent_images(model.requests[1])
    (width, height), _ = _pixels(_decoded(url))
    assert width * height <= 1_000_000 and width / height == pytest.approx(1600 / 1200, rel=0.01)
    assert _kept_images(output["output_items"]) == [url]


def test_without_the_option_the_model_has_no_view_image_tool(tmp_path):
    output, model = _run(tmp_path, "Hello.", [FINISH], view_image=False)

    assert "view_image" not in [tool["function"]["name"] for tool in model.requests[0]["tools"]]


def test_a_trajectory_with_images_is_a_valid_response_and_observation(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "receipt.png").write_bytes(_png((10, 10, 10)))
    content = [{"type": "input_image", "image_url": _data_url(_png((1, 2, 3)))}, {"type": "input_text", "text": "Go."}]

    output, _ = _run(tmp_path, content, [_call("view_image", json.dumps({"path": "receipt.png"})), FINISH])

    response = NeMoGymResponse(
        id="r",
        created_at=0,
        model="policy",
        object="response",
        output=output["input_items"] + output["output_items"],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    assert len(_kept_images(response.model_dump(mode="json")["output"])) == 2
    bundle = _observations(output, ModelServerRef(type="responses_api_models", name="policy_model"))
    assert bundle.gaps == []
