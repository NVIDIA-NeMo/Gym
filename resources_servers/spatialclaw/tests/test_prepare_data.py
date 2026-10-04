# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.spatialclaw.prepare_data import (
    _instruction,
    _media_and_metadata,
    _portable_dataset_config,
)


def test_instruction_matches_spatialclaw_runner_choice_rendering() -> None:
    benchmark = SimpleNamespace(data_specific_prompt="Answer with one letter.")
    sample = SimpleNamespace(question="Where is the cup?", choices={"A": "Left", "B": "Right"})

    assert _instruction(benchmark, sample) == ("Where is the cup?\n\nAnswer with one letter.\nA. Left\nB. Right")


def test_media_keeps_video_as_agent_side_channel() -> None:
    sample = SimpleNamespace(
        video="/data/clip.mp4",
        video_sources_per_video=None,
        image_groups=None,
        images=[],
        frame_indices=[],
        fps=0.0,
        total_video_frames=0,
        duration_sec=0.0,
        ref_images=[],
    )

    media, metadata = _media_and_metadata(sample)

    assert media == [{"type": "input_video", "video_url": "file:///data/clip.mp4"}]
    assert metadata["video_sources_per_video"] == ["file:///data/clip.mp4"]
    assert "fps" not in metadata
    assert "total_video_frames" not in metadata


def test_dataset_config_is_portable_between_checkouts(tmp_path) -> None:
    config_path = tmp_path / "spatial_agent" / "config" / "dataset" / "videomme.json"

    assert _portable_dataset_config(tmp_path, config_path) == "videomme.json"


def test_prepared_images_validate_against_gym_schema() -> None:
    media, _ = _media_and_metadata(SimpleNamespace(images=["/data/image.png"]))
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Question"}, *media]}]}
    )
    assert body.model_dump()["input"][0]["content"][1]["image_url"] == "file:///data/image.png"
