# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox image's video tools (sandbox_image/video_tools): which frames, what size, the text, the program.

Decoding needs FFmpeg on PATH (the image has 7.0.2) and is skipped without it; the frame rules need only Pydantic.
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from resources_servers.video_mcqa.sandbox_image.video_tools.contract import (
    FrameProfile,
    MediaContractError,
    SourceVideoMetadata,
    ToolConfig,
    VideoSpec,
    frame_labels,
    resolve_frames,
    resolve_resize,
    sampling_receipt,
    sampling_timestamps,
    video_source_label,
    vision_tokens,
    watch_video_text,
)


SANDBOX_IMAGE = Path(__file__).resolve().parents[1] / "sandbox_image"
CLIPS = Path(__file__).resolve().parents[1] / "data" / "example_videos"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg is required to decode frames"
)


def spec(**updates: object) -> VideoSpec:
    return VideoSpec(
        **{"video_path": "videos/a.mp4", "start": 0.0, "end": 3.0, "num_frames": 6, "resize_long_edge": 448, **updates}
    )


def one_fps_source(**updates: object) -> SourceVideoMetadata:
    """Three source frames at 0, 1 and 2 s of a 3 s, 640x360 video."""
    values = {"width": 640, "height": 360, "duration_seconds": 3.0, "frame_timestamps_seconds": (0.0, 1.0, 2.0)}
    return SourceVideoMetadata(**{**values, **updates})


def test_samples_are_bin_centers() -> None:
    times = sampling_timestamps(spec(start=0.0, end=18.0, num_frames=12))
    assert times == tuple(0.75 + 1.5 * index for index in range(12))


def test_each_sample_is_the_frame_on_screen_and_held_frames_repeat() -> None:
    resolved = resolve_frames(spec(), source=one_fps_source(), profile=FrameProfile())
    # Samples at 0.25, 0.75, ..., 2.75 s show the frame that started last: each source frame twice.
    assert resolved.sampling_timestamps_seconds == (0.25, 0.75, 1.25, 1.75, 2.25, 2.75)
    assert resolved.frame_indices == (0, 0, 1, 1, 2, 2)
    assert resolved.timestamps_seconds == (0.0, 0.0, 1.0, 1.0, 2.0, 2.0)
    assert "Frames: 6 total, 3 unique source frames (50.0% repeated)" in sampling_receipt(resolved)


def test_a_frame_that_started_before_the_window_is_sampled_inside_it() -> None:
    resolved = resolve_frames(spec(start=1.5, end=1.9, num_frames=2), source=one_fps_source(), profile=FrameProfile())
    assert resolved.frame_indices == (1, 1)


@pytest.mark.parametrize(
    ("updates", "source", "code"),
    [
        (
            {"start": 0.0, "end": 1.0, "num_frames": 2},
            {"frame_timestamps_seconds": (0.6, 1.0, 2.0)},
            "source_video_not_started",
        ),
        ({"end": 3.5}, {}, "invalid_source_range"),
        ({"start": 0.0, "end": 1e-6, "num_frames": 4}, {}, "invalid_sampling_interval"),
    ],
)
def test_requests_the_source_cannot_serve_are_tool_errors(updates, source, code) -> None:
    with pytest.raises(MediaContractError) as error:
        resolve_frames(spec(**updates), source=one_fps_source(**source), profile=FrameProfile())
    assert error.value.code == code and error.value.to_dict()["code"] == code


def test_frames_round_down_to_the_model_grid() -> None:
    profile = FrameProfile()
    assert profile.grid_cell == 28
    assert resolve_resize(source_width=1920, source_height=1080, resize_long_edge=448, profile=profile) == (448, 252)
    assert resolve_resize(source_width=1280, source_height=720, resize_long_edge=500, profile=profile) == (476, 280)
    # Each side is floored from its floating-point product: 360 * 0.7 / 28 < 9.
    assert resolve_resize(source_width=640, source_height=360, resize_long_edge=448, profile=profile) == (448, 224)
    # Upscaling is allowed: the long edge is an explicit target.
    assert resolve_resize(source_width=320, source_height=240, resize_long_edge=448, profile=profile) == (448, 336)
    # The grid is a parameter: 16 px patches merged 2x2 make 32 px cells.
    wide = FrameProfile(patch_size=16)
    assert resolve_resize(source_width=1920, source_height=1080, resize_long_edge=448, profile=wide) == (448, 224)
    assert vision_tokens(num_frames=2, width=448, height=252, profile=profile) == 2 * 16 * 9


@pytest.mark.parametrize(
    ("size", "long_edge", "code"),
    [((4000, 100), 448, "invalid_resize"), ((1000, 1000), 2100, "video_frame_pixels_exceeded")],
)
def test_frames_too_narrow_or_too_large_are_refused_not_stretched(size, long_edge, code) -> None:
    with pytest.raises(MediaContractError) as error:
        resolve_resize(source_width=size[0], source_height=size[1], resize_long_edge=long_edge, profile=FrameProfile())
    assert error.value.code == code


def test_the_pixel_cap_refuses_a_call_instead_of_dropping_frames() -> None:
    source = SourceVideoMetadata(
        width=1920, height=1080, duration_seconds=100.0, frame_timestamps_seconds=tuple(i / 10 for i in range(1000))
    )
    resolved = resolve_frames(spec(end=100.0, num_frames=594), source=source, profile=FrameProfile())
    assert resolved.width * resolved.height * 594 <= 67_108_864
    with pytest.raises(MediaContractError) as error:
        resolve_frames(spec(end=100.0, num_frames=595), source=source, profile=FrameProfile())
    assert error.value.code == "video_pixels_exceeded"
    assert error.value.details["video_pixels"] == 595 * 448 * 252


def test_labels_and_receipt_read_as_a_video_prompt() -> None:
    resolved = resolve_frames(spec(), source=one_fps_source(), profile=FrameProfile())
    assert (
        frame_labels((0.25, 1.5))
        == "This is a video:\nFrame 1 sampled at 0.25 seconds:\nFrame 2 sampled at 1.50 seconds:"
    )
    assert watch_video_text(resolved) == (
        "Video: videos/a.mp4\n"
        "Source range: [0.00s, 3.00s)\n"
        "Temporal labels are sampling times; original source PTS is preserved separately.\n"
        "Frames: 6 total, 3 unique source frames (50.0% repeated)\n"
        "Resolution: 448×224\n"
        "Visual tokens: 768\n"
        "Sampling interval: 0.5 s\n"
        "Maximum gap between distinct source-frame PTS: 1 s\n"
        "This is a video:\n"
        "Frame 1 sampled at 0.25 seconds:\n"
        "Frame 2 sampled at 0.75 seconds:\n"
        "Frame 3 sampled at 1.25 seconds:\n"
        "Frame 4 sampled at 1.75 seconds:\n"
        "Frame 5 sampled at 2.25 seconds:\n"
        "Frame 6 sampled at 2.75 seconds:"
    )
    single = sampling_receipt(resolve_frames(spec(num_frames=1), source=one_fps_source(), profile=FrameProfile()))
    assert "Sampling interval: n/a (one sample)" in single
    assert "Maximum gap between distinct source-frame PTS: n/a (one unique source frame)" in single


def test_a_path_that_could_read_as_a_special_token_is_quoted() -> None:
    assert video_source_label(spec(video_path="videos/<img>.mp4")).startswith(
        'Video: "videos/\\u003cimg\\u003e.mp4"\n'
    )
    with pytest.raises(ValueError, match="line breaks"):
        video_source_label(spec(video_path="videos/a\n.mp4"))


@pytest.mark.parametrize(
    "updates",
    [
        {"start": "0"},
        {"num_frames": 2.0},
        {"num_frames": 0},
        {"end": 0.0},
        {"start": 3.0},
        {"fps": 2},
        {"video_path": ""},
    ],
)
def test_tool_arguments_are_strict(updates) -> None:
    with pytest.raises(ValidationError):
        spec(**updates)


def test_profile_and_tool_configuration_are_checked() -> None:
    with pytest.raises(ValidationError, match="reciprocal"):
        FrameProfile(downsample_ratio=0.3)
    with pytest.raises(ValidationError, match="<= 1"):
        FrameProfile(downsample_ratio=2.0)
    config = ToolConfig(workspace_root="/testbed", cache_dir="/tmp/video-mcqa/index")
    assert config.command == ["/usr/local/bin/python3", "-I", "-m", "video_tools.cli"]
    assert config.max_tool_media_bytes == 400 << 20 and config.profile == FrameProfile()
    with pytest.raises(ValidationError, match="outside workspace_root"):
        ToolConfig(workspace_root="/testbed", cache_dir="/testbed/index")
    with pytest.raises(ValidationError, match="absolute"):
        ToolConfig(workspace_root="testbed", cache_dir="/tmp/index")


@pytest.fixture
def media():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from resources_servers.video_mcqa.sandbox_image.video_tools import media

    return media


@pytest.fixture
def workspace(tmp_path) -> Path:
    root = tmp_path / "testbed"
    (root / "videos").mkdir(parents=True)
    shutil.copy(CLIPS / "synthetic-5.mp4", root / "videos" / "clip.mp4")
    return root


@needs_ffmpeg
def test_watch_video_decodes_the_resolved_frames_as_jpeg(media, workspace, tmp_path) -> None:
    from PIL import Image

    request = VideoSpec(
        video_path=str(workspace / "videos" / "clip.mp4"), start=0.5, end=2.5, num_frames=3, resize_long_edge=100
    )
    result = media.watch_video(request, workspace_root=workspace, cache_dir=tmp_path / "index", profile=FrameProfile())
    resolved = result["metadata"]["resolved_video"]
    # The clip has 8 frames per second: samples at 0.83, 1.5 and 2.17 s show frames 6, 12 and 17.
    assert resolved["frame_indices"] == [6, 12, 17]
    assert resolved["timestamps_seconds"] == [0.75, 1.5, 2.125]
    assert (resolved["width"], resolved["height"]) == (84, 56)
    assert result["output"].startswith("Video: videos/clip.mp4\nSource range: [0.50s, 2.50s)\n")
    assert result["output"].endswith("Frame 3 sampled at 2.17 seconds:")
    assert [attachment["mime"] for attachment in result["attachments"]] == ["image/jpeg"] * 3
    for attachment in result["attachments"]:
        data = base64.b64decode(attachment["url"].removeprefix("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(data)) as image:
            assert (image.format, image.size) == ("JPEG", (84, 56))
    assert result["metadata"]["processing_stats"]["source_index_cache_hit"] is False
    again = media.watch_video(request, workspace_root=workspace, cache_dir=tmp_path / "index", profile=FrameProfile())
    assert again["metadata"]["processing_stats"]["source_index_cache_hit"] is True
    assert again["metadata"]["frame_sha256"] == result["metadata"]["frame_sha256"]


@needs_ffmpeg
def test_the_seek_decode_matches_the_sequential_reference(media, workspace, tmp_path) -> None:
    source = workspace / "videos" / "clip.mp4"
    index, _ = media._get_source_index(source, cache_dir=tmp_path / "index", profile=FrameProfile())
    request = VideoSpec(video_path="videos/clip.mp4", start=0.0, end=4.0, num_frames=16, resize_long_edge=448)
    resolved = resolve_frames(request, source=index.source, profile=FrameProfile())
    decoded = media._decode_frames(source, index, resolved, profile=FrameProfile())
    reference = media._decode_sequential(source, resolved, timeout=60)
    assert decoded.backend == "ffmpeg_exact_seek" and decoded.frames.shape == (16, 336, 448, 3)
    assert (decoded.frames == reference).all()


@needs_ffmpeg
def test_a_source_that_changed_since_its_index_is_refused_and_indexed_again(media, workspace, tmp_path) -> None:
    source = workspace / "videos" / "clip.mp4"
    index, _ = media._get_source_index(source, cache_dir=tmp_path / "index", profile=FrameProfile())
    os.utime(source, ns=(1, 1))
    with pytest.raises(MediaContractError, match="changed"):
        media._check_unchanged(source, index)
    _, cached = media._get_source_index(source, cache_dir=tmp_path / "index", profile=FrameProfile())
    assert cached is False


def test_paths_outside_the_workspace_are_refused(media, workspace, tmp_path) -> None:
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"not an image")
    (workspace / "link.png").symlink_to(outside)
    for path in (str(outside), "link.png", "../secret.png"):
        with pytest.raises(MediaContractError) as error:
            media.workspace_file(path, workspace_root=workspace)
        assert error.value.code == "invalid_media_path"


def test_view_image_returns_the_file_unchanged(media, workspace) -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (30, 20), (200, 10, 10)).save(buffer, format="PNG")
    (workspace / "frame.png").write_bytes(buffer.getvalue())
    result = media.view_image("frame.png", workspace_root=workspace)
    assert result["output"] == "Image: frame.png"
    (attachment,) = result["attachments"]
    assert attachment["mime"] == "image/png"
    assert base64.b64decode(attachment["url"].removeprefix("data:image/png;base64,")) == buffer.getvalue()
    big = io.BytesIO()
    Image.new("L", (2049, 2048)).save(big, format="PNG")
    (workspace / "big.png").write_bytes(big.getvalue())
    with pytest.raises(MediaContractError) as error:
        media.view_image("big.png", workspace_root=workspace)
    assert error.value.code == "image_pixels_exceeded"


def run_cli(config: Path, request: bytes | str, *extra: str) -> subprocess.CompletedProcess[str]:
    """The program as the plugin runs it, from a copy of sandbox_image (as the image's .pth file puts it on the path)."""
    environment = {**os.environ, "PYTHONPATH": str(SANDBOX_IMAGE)}
    data = request.encode() if isinstance(request, str) else request
    return subprocess.run(
        [sys.executable, "-m", "video_tools.cli", "--config", str(config), *extra],
        input=data,
        capture_output=True,
        env=environment,
        timeout=120,
        check=False,
    )


@pytest.fixture
def config_file(workspace, tmp_path) -> Path:
    path = tmp_path / "tools.json"
    config = ToolConfig(workspace_root=str(workspace), cache_dir=str(tmp_path / "index"))
    path.write_text(config.model_dump_json())
    return path


@pytest.mark.parametrize(
    ("request_body", "code"),
    [
        (
            json.dumps({"tool": "watch_video", "arguments": {"video_path": "videos/clip.mp4", "start": "0"}}),
            "invalid_media_request",
        ),
        (json.dumps({"tool": "unknown", "arguments": {}}), "invalid_media_request"),
        (json.dumps({"tool": "view_image", "arguments": {"path": "missing.png"}}), "invalid_media_request"),
        (b"{" + b" " * (64 << 10) + b"}", "tool_input_too_large"),
    ],
)
def test_the_program_answers_tool_errors_as_json(media, config_file, request_body, code) -> None:
    result = run_cli(config_file, request_body)
    assert result.returncode == 0, result.stderr
    reply = json.loads(result.stdout)
    assert reply["error"]["code"] == code and isinstance(reply["output"], str)


@needs_ffmpeg
def test_the_program_returns_frames_and_checks_ffmpeg(media, config_file) -> None:
    arguments = {"video_path": "videos/clip.mp4", "start": 0.0, "end": 4.0, "num_frames": 8, "resize_long_edge": 448}
    result = run_cli(config_file, json.dumps({"tool": "watch_video", "arguments": arguments}))
    assert result.returncode == 0, result.stderr
    reply = json.loads(result.stdout)
    assert len(reply["attachments"]) == 8 and "Visual tokens: 1536" in reply["output"]
    check = run_cli(config_file, "", "--check")
    assert check.returncode == 0 and json.loads(check.stdout)["ok"] is True
