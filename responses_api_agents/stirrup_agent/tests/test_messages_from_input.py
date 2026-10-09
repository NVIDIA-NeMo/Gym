# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.stirrup_agent.stirrup_utils import messages_from_input


pytest.importorskip("stirrup")


def _body(**kwargs) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(**kwargs)


def _messages_from_input(body: NeMoGymResponseCreateParamsNonStreaming):
    """Convert the request as the sandbox runner receives it, serialized to JSON."""
    params = body.model_dump(mode="json")
    return messages_from_input(params["input"], params["instructions"])


def test_single_user_message_becomes_the_task_without_a_system_prompt():
    system_prompt, messages = _messages_from_input(_body(input=[{"role": "user", "content": "Do the task."}]))

    assert system_prompt is None
    assert [(type(m).__name__, m.content) for m in messages] == [("UserMessage", "Do the task.")]


def test_instructions_and_leading_system_messages_form_the_system_prompt():
    body = _body(
        instructions="Be brief.",
        input=[
            {"role": "system", "content": "Work in /root."},
            {"role": "user", "content": [{"type": "input_text", "text": "Do the task."}]},
        ],
    )

    system_prompt, messages = _messages_from_input(body)

    assert system_prompt == "Be brief.\n\nWork in /root."
    assert messages[0].content == ["Do the task."]


def test_assistant_messages_are_not_supported():
    body = _body(input=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])

    with pytest.raises(NotImplementedError):
        _messages_from_input(body)


def test_input_without_a_user_message_is_rejected():
    with pytest.raises(ValueError, match="requires a user message"):
        _messages_from_input(_body(input=[]))


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


@pytest.fixture(scope="module")
def png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def mp4(tmp_path_factory) -> bytes:
    moviepy = pytest.importorskip("moviepy")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    moviepy.ColorClip((16, 16), color=(255, 0, 0), duration=0.2).write_videofile(str(path), fps=5, logger=None)
    return path.read_bytes()


def test_base64_image_becomes_an_image_block_with_the_decoded_bytes(png):
    from stirrup.core.models import ImageContentBlock

    content = [
        {"type": "input_text", "text": "Look:"},
        {"type": "input_image", "detail": "auto", "image_url": _data_url("image/png", png)},
    ]

    _, messages = _messages_from_input(_body(input=[{"role": "user", "content": content}]))

    text, image = messages[0].content
    assert text == "Look:"
    assert isinstance(image, ImageContentBlock) and image.data == png


@pytest.mark.parametrize("field", ["video_url", "video"])
def test_base64_video_becomes_a_video_block_with_the_decoded_bytes(mp4, field):
    from stirrup.core.models import VideoContentBlock

    url = _data_url("video/mp4", mp4)
    part = {"type": "input_video", field: {"url": url} if field == "video_url" else url}

    _, messages = _messages_from_input(_body(input=[{"role": "user", "content": [part]}]))

    (video,) = messages[0].content
    assert isinstance(video, VideoContentBlock) and video.data == mp4


def test_media_must_be_a_base64_data_url():
    part = {"type": "input_image", "detail": "auto", "image_url": "https://example.com/cat.png"}

    with pytest.raises(ValueError, match="base64 data URL"):
        _messages_from_input(_body(input=[{"role": "user", "content": [part]}]))


def test_developer_messages_join_the_system_prompt():
    body = _body(input=[{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Do the task."}])

    assert _messages_from_input(body)[0] == "Be brief."


def test_system_messages_after_a_user_message_are_rejected():
    body = _body(input=[{"role": "user", "content": "Do the task."}, {"role": "system", "content": "Be brief."}])

    with pytest.raises(ValueError, match="before the first user message"):
        _messages_from_input(body)


def test_system_messages_must_be_text_only(png):
    image = {"type": "input_image", "detail": "auto", "image_url": _data_url("image/png", png)}
    body = _body(input=[{"role": "system", "content": [image]}, {"role": "user", "content": "Do the task."}])

    with pytest.raises(ValueError, match="text-only system messages"):
        _messages_from_input(body)


def test_non_message_items_are_rejected():
    call = {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}
    body = _body(input=[{"role": "user", "content": "Do the task."}, call])

    with pytest.raises(ValueError, match="expects message items"):
        _messages_from_input(body)
