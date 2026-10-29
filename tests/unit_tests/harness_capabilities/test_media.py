# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import copy

import pytest

from nemo_gym.harness_capabilities.media import image_payloads, inspect_media


def image(data: bytes) -> dict:
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(data).decode()}


@pytest.fixture
def media_episode():
    a, b, c = [image(data) for data in (b"first", b"second", b"tool")]
    prompt = {"role": "user", "content": [{"type": "input_text", "text": "before"}, a, b]}
    call = {"type": "function_call", "call_id": "tool-1", "name": "read", "arguments": '{"path":"image.png"}'}
    output = {"type": "function_call_output", "call_id": "tool-1", "output": [c]}
    requests = [{"input": [prompt]}, {"input": [prompt, call, output]}]
    witness = {
        "attempts": [{"request": request} for request in requests],
        "backend_requests": copy.deepcopy(requests),
        "media_tool": {"id": "tool-1", "name": "read", "arguments": {"path": "image.png"}},
    }
    record = {
        "ng_trajectory": {
            "model_calls": [{"request": copy.deepcopy(request)} for request in requests],
            "tool_calls": [
                {
                    "tool_call_id": "tool-1",
                    "tool_name": "read",
                    "invocation_id": "agent",
                    "status": "completed",
                    "output": [copy.deepcopy(c)],
                }
            ],
            "invocations": [{"invocation_id": "agent", "conversation": [copy.deepcopy(call), copy.deepcopy(output)]}],
        }
    }
    return (
        witness,
        record,
        {"applies": True, "prompt_images": tuple(image_payloads([a, b])), "tool_image": image_payloads(c)[0]},
    )


def test_media_matches_independent_witness_and_is_p1(media_episode):
    witness, record, options = media_episode
    checks = inspect_media(witness, record, **options)
    assert len(checks) == 6
    assert all(c["tier"] == "P1" and c["status"] == "pass" for c in checks)


@pytest.mark.parametrize(
    "mutation", ["missing", "corrupt", "substitute", "reorder", "duplicate", "text", "role", "detail"]
)
def test_saved_media_rejects_lost_or_changed_payload(media_episode, mutation):
    witness, record, options = media_episode
    message = record["ng_trajectory"]["model_calls"][0]["request"]["input"][0]
    content = message["content"]
    if mutation == "missing":
        content.pop()
    elif mutation == "corrupt":
        content[1]["image_url"] = "data:image/png;base64,invalid!"
    elif mutation == "substitute":
        content[1] = image(b"other")
    elif mutation == "reorder":
        content[1:] = reversed(content[1:])
    elif mutation == "duplicate":
        content.append(content[1])
    elif mutation == "text":
        content[0]["text"] = "changed"
    elif mutation == "role":
        message["role"] = "assistant"
    else:
        content[1]["detail"] = "low"
    checks = {c["id"]: c for c in inspect_media(witness, record, **options)}
    assert checks["media.prompt_delivery"]["status"] == "pass"
    assert checks["media.saved_requests"]["status"] == "fail"


@pytest.mark.parametrize("source", ["both", "witness", "record"])
def test_no_vacuous_pass_or_fallback(media_episode, source):
    witness, record, options = media_episode
    if source in ("both", "witness"):
        for attempt in witness["attempts"]:
            attempt["request"] = {"input": []}
    if source in ("both", "record"):
        record["ng_trajectory"]["model_calls"] = []
    checks = {c["id"]: c for c in inspect_media(witness, record, **options)}
    assert checks["media.saved_requests"]["status"] == "fail"
    if source != "record":
        assert checks["media.prompt_delivery"]["status"] == "fail"
        assert checks["media.tool_delivery"]["status"] == "fail"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "id", "owner", "arguments", "name", "status", "payload"])
def test_tool_media_requires_canonical_output_and_association(media_episode, mutation):
    witness, record, options = media_episode
    trajectory = record["ng_trajectory"]
    tool = trajectory["tool_calls"][0]
    if mutation == "missing":
        trajectory["tool_calls"] = []
    elif mutation == "duplicate":
        trajectory["tool_calls"].append(copy.deepcopy(tool))
    elif mutation == "id":
        tool["tool_call_id"] = "other"
    elif mutation == "owner":
        tool["invocation_id"] = "other"
    elif mutation == "arguments":
        trajectory["invocations"][0]["conversation"][0]["arguments"] = '{"path":"other.png"}'
    elif mutation == "name":
        tool["tool_name"] = "other"
    elif mutation == "status":
        tool["status"] = "failed"
    else:
        tool["output"] = "Image read successfully"
    checks = {c["id"]: c for c in inspect_media(witness, record, **options)}
    assert checks["media.tool_output" if mutation == "payload" else "media.tool_join"]["status"] == "fail"
    assert checks["media.tool_delivery"]["status"] == "pass"


def test_explicit_applicability_and_missing_evidence(media_episode):
    _, _, options = media_episode
    assert all(c["status"] == "fail" for c in inspect_media(None, None, **options))
    assert all(c["status"] == "not_applicable" for c in inspect_media(None, None))


def test_chat_images_and_detail_are_retained():
    part = image(b"image")
    chat = {"type": "image_url", "image_url": {"url": part["image_url"], "detail": "high"}}
    assert image_payloads(chat) == [{**image_payloads(part)[0], "detail": "high"}]
    assert image_payloads({"type": "text", "text": part["image_url"]}) == []
    for unavailable in ("https://example.com/a.png", "data:image/png;base64,", "data:image/png;base64,!"):
        with pytest.raises(ValueError):
            image_payloads({"type": "input_image", "image_url": unavailable})
