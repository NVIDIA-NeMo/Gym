# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.spatialclaw_agent.app import (
    _configure_video_role_preprocessing,
    _extract_request_input,
    _frame_cache_dir,
    _session_id,
)


def test_extract_request_input_supports_video_side_channel() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Track the red object."},
                        {"type": "input_image", "image_url": "file:///data/reference.png", "detail": "auto"},
                        {"type": "input_video", "video_url": "file:///data/clip.mp4"},
                    ],
                }
            ]
        }
    )

    instruction, images, videos = _extract_request_input(body)

    assert instruction == "Track the red object."
    assert images == ["file:///data/reference.png"]
    assert videos == ["file:///data/clip.mp4"]


def test_session_id_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="filename-safe"):
        _session_id("../escape")


def test_explicit_video_preprocessing_overrides_apply_to_all_roles() -> None:
    role_names = (
        "main_params",
        "planning_params",
        "general_params",
        "vlm_params",
        "vlm_grounding_params",
        "reflection_params",
    )
    config = SimpleNamespace(
        **{name: SimpleNamespace(mm_processor_kwargs={"role": name, "max_num_tiles": 2}) for name in role_names}
    )

    _configure_video_role_preprocessing(
        config,
        {"max_num_tiles": 1, "video_as_images": True},
    )

    for name in role_names:
        assert getattr(config, name).mm_processor_kwargs == {
            "role": name,
            "max_num_tiles": 1,
            "video_as_images": True,
        }


def test_frame_cache_key_includes_sampling_protocol(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    first = _frame_cache_dir(
        str(video),
        str(tmp_path / "cache"),
        SimpleNamespace(video_max_fps=1, video_frame_resize_short_edge=None),
    )
    second = _frame_cache_dir(
        str(video),
        str(tmp_path / "cache"),
        SimpleNamespace(video_max_fps=2, video_frame_resize_short_edge=None),
    )

    assert first != second
    assert first.parent == second.parent == (tmp_path / "cache").resolve()
