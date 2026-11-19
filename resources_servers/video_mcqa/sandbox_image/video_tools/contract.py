# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which source frames watch_video returns, at what size, and the text the model reads with them.

The frame rules:

- the model picks the source window ``[start, end)`` in seconds, the frame count and the long edge;
- the window is cut into ``num_frames`` equal bins, and each bin is sampled at its center, the sampling time;
- each sample is the latest source frame whose presentation time is not after its sampling time, so a frame the
  video holds can repeat;
- frames keep their aspect ratio: scaled to the requested long edge, then each side is rounded down to a multiple
  of the model's grid cell, ``patch_size / downsample_ratio`` pixels (by default 14 / 0.5 = 28);
- the text says what was sampled and that the frames are attached after the tool results, then gives each frame's
  sampling time, ``Frame i: t s``, in the order of the images.

Only the standard library and Pydantic are imported here, so the resources server validates the same configuration
that the tools read in the sandbox.
"""

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


# Area limit of one resized frame and of an image that view_image returns.
MAX_FRAME_PIXELS = 2048 * 2048
# view_image returns a file of at most this many bytes.
MAX_IMAGE_BYTES = 32 << 20
# Pillow settings of every frame: JPEG quality 95 without chroma subsampling (4:4:4).
JPEG_QUALITY = 95


class MediaContractError(ValueError):
    """A tool error the model can act on, as opposed to a programming error."""

    def __init__(self, code: str, message: str, *, details: dict[str, object]) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, object]:
        """The error as JSON the model reads, without media."""
        return {"code": self.code, "message": str(self), "details": self.details}


class FrameProfile(BaseModel):
    """The model's grid and the limits of one watch_video call (by default a 28-pixel grid: 14 / 0.5)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    # The vision encoder's patch edge in pixels.
    patch_size: PositiveInt = 14
    # The encoder merges (1 / downsample_ratio)^2 patches into one token.
    downsample_ratio: PositiveFloat = 0.5
    # Pixels of all frames of one call together (num_frames * width * height).
    video_max_pixels: PositiveInt = Field(default=67_108_864, le=2**53 - 1)
    # Deadline of each FFmpeg run, and of all frame decodes of one call together.
    decode_timeout_seconds: PositiveFloat = 120.0
    # Decoded RGB bytes one call may hold in memory.
    max_materialized_bytes: PositiveInt = 8_589_934_592

    @model_validator(mode="after")
    def merge_is_an_integer(self) -> "FrameProfile":
        if self.downsample_ratio > 1:
            raise ValueError("downsample_ratio must be <= 1")
        inverse = 1.0 / self.downsample_ratio
        if not math.isclose(inverse, round(inverse)):
            raise ValueError("downsample_ratio must be the reciprocal of an integer")
        return self

    @property
    def spatial_merge_size(self) -> int:
        """Patches merged along each side into one token."""
        return round(1.0 / self.downsample_ratio)

    @property
    def grid_cell(self) -> int:
        """The pixel multiple of each resized side."""
        return self.patch_size * self.spatial_merge_size


def _absolute_directory(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError("must be an absolute sandbox directory other than /")
    return value


class ToolSettings(BaseModel):
    """The frame profile and the plugin's limits, as a resources server config sets them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: FrameProfile = Field(default_factory=FrameProfile)
    # The program the plugin runs once per tool call, with "--config <tools.json>" added: the image's Python, which
    # has video_tools on its path (sandbox_image/Dockerfile).
    command: list[str] = Field(default=["/usr/local/bin/python3", "-I", "-m", "video_tools.cli"], min_length=1)
    # The plugin stops a tool call after this many seconds.
    timeout_seconds: PositiveFloat = 1260.0
    # The plugin stops a tool call whose JSON reply grows past this many bytes.
    max_output_bytes: PositiveInt = 258 << 20
    # Base64 bytes of tool images one episode sends to the model. From the tool result that would pass it on, the
    # plugin sends a notice in place of each result's images, the same in every later request.
    max_tool_media_bytes: PositiveInt = 400 << 20
    # Absolute sandbox directories outside the workspace whose files the tools also read: OpenCode's temporary
    # directory, where its bash tool keeps scratch files such as frames the model cut.
    scratch_roots: tuple[str, ...] = ("/tmp/opencode",)

    @field_validator("scratch_roots")
    @classmethod
    def absolute_scratch_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_absolute_directory(root) for root in value)


class ToolConfig(ToolSettings):
    """What the resources server writes to the sandbox at each seed (``tools.json`` beside the plugin)."""

    # Absolute sandbox directory the tools read files under; relative tool paths start there.
    workspace_root: str
    # Absolute sandbox directory of the tools' source indexes, outside workspace_root.
    cache_dir: str

    @field_validator("workspace_root", "cache_dir")
    @classmethod
    def absolute_directory(cls, value: str) -> str:
        return _absolute_directory(value)

    @model_validator(mode="after")
    def cache_outside_workspace(self) -> "ToolConfig":
        if any(
            PurePosixPath(self.cache_dir).is_relative_to(root) for root in (self.workspace_root, *self.scratch_roots)
        ):
            raise ValueError("cache_dir must be outside workspace_root and scratch_roots")
        return self


class VideoSpec(BaseModel):
    """The model's watch_video arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    video_path: str = Field(min_length=1)
    start: NonNegativeFloat
    end: PositiveFloat
    num_frames: PositiveInt
    resize_long_edge: PositiveInt

    @model_validator(mode="after")
    def validate_window(self) -> "VideoSpec":
        """Absolute source seconds, half-open window."""
        if self.end <= self.start:
            raise ValueError("source range must satisfy 0 <= start < end")
        return self


@dataclass(frozen=True)
class SourceVideoMetadata:
    """Decoded source timing; times are seconds from the start of the whole file."""

    width: int
    height: int
    duration_seconds: float
    frame_timestamps_seconds: tuple[float, ...]

    def validate(self) -> None:
        """Real, strictly increasing presentation times, variable frame rate included."""
        timestamps = self.frame_timestamps_seconds
        if self.width <= 0 or self.height <= 0 or not timestamps:
            raise ValueError("source must have positive dimensions and decoded frames")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("source duration must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in timestamps):
            raise ValueError("source frame timestamps must be finite and non-negative")
        if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("source frame timestamps must be strictly increasing")
        if timestamps[-1] >= self.duration_seconds:
            raise ValueError("source duration must extend past its last frame")


@dataclass(frozen=True)
class ResolvedFrames:
    """The source frames and the size of one watch_video call."""

    requested: VideoSpec
    source_width: int
    source_height: int
    source_duration_seconds: float
    # Index of each sample's source frame, in presentation order, and that frame's presentation time.
    frame_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    # The center of each bin: the time each label gives.
    sampling_timestamps_seconds: tuple[float, ...]
    width: int
    height: int
    vision_tokens: int

    def to_dict(self) -> dict[str, object]:
        """A JSON record of the call, without pixels."""
        return {
            "requested": self.requested.model_dump(),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_duration_seconds": self.source_duration_seconds,
            "frame_indices": list(self.frame_indices),
            "timestamps_seconds": list(self.timestamps_seconds),
            "sampling_timestamps_seconds": list(self.sampling_timestamps_seconds),
            "width": self.width,
            "height": self.height,
            "vision_tokens": self.vision_tokens,
        }


def timestamp_transport_microseconds(timestamps: tuple[float, ...]) -> list[int]:
    """Integer microseconds that keep each time's two-decimal label; distinct times must stay distinct."""
    result = []
    for timestamp in timestamps:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("transport requires finite non-negative PTS")
        representative = round(timestamp * 1_000_000)
        if f"{representative / 1_000_000:.2f}" != f"{timestamp:.2f}":
            representative += -1 if representative / 1_000_000 > timestamp else 1
        result.append(representative)
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError("source PTS precision exceeds integer microsecond transport")
    return result


def sampling_timestamps(requested: VideoSpec) -> tuple[float, ...]:
    """The center of each of num_frames equal bins of the window."""
    interval = (requested.end - requested.start) / requested.num_frames
    result = tuple(requested.start + (index + 0.5) * interval for index in range(requested.num_frames))
    details: dict[str, object] = {"start": requested.start, "end": requested.end, "num_frames": requested.num_frames}
    if any(not requested.start <= t < requested.end for t in result) or any(
        left >= right for left, right in zip(result, result[1:])
    ):
        raise MediaContractError(
            "invalid_sampling_interval",
            "the requested uniform sampling times cannot be represented distinctly",
            details=details,
        )
    try:
        timestamp_transport_microseconds(result)
    except ValueError as error:
        raise MediaContractError(
            "invalid_sampling_interval",
            "uniform sampling times must be distinct at integer-microsecond precision; "
            "choose fewer frames or a longer source range",
            details=details,
        ) from error
    return result


def resolve_resize(
    *, source_width: int, source_height: int, resize_long_edge: int, profile: FrameProfile
) -> tuple[int, int]:
    """Scale to the requested long edge, then round each side DOWN to the grid cell.

    Upscaling is allowed: the long edge is an explicit target. A frame too narrow for one grid cell is refused, not
    stretched, and so is a frame over the area limit.
    """
    if min(source_width, source_height, resize_long_edge) <= 0:
        raise ValueError("source dimensions and resize_long_edge must be positive")
    unit = profile.grid_cell
    scale = resize_long_edge / max(source_width, source_height)
    width = math.floor(source_width * scale / unit) * unit
    height = math.floor(source_height * scale / unit) * unit
    if min(width, height) < unit:
        raise MediaContractError(
            "invalid_resize",
            "resize_long_edge is too small for this aspect ratio and model grid",
            details={"grid_pixel_unit": unit, "resize_long_edge": resize_long_edge},
        )
    frame_pixels = width * height
    if frame_pixels > MAX_FRAME_PIXELS:
        raise MediaContractError(
            "video_frame_pixels_exceeded",
            "single resized video frame exceeds the 2048*2048 pixel limit; choose a smaller resize_long_edge",
            details={
                "width": width,
                "height": height,
                "frame_pixels": frame_pixels,
                "max_frame_pixels": MAX_FRAME_PIXELS,
                "resize_long_edge": resize_long_edge,
            },
        )
    return width, height


def vision_tokens(*, num_frames: int, width: int, height: int, profile: FrameProfile) -> int:
    """Tokens of the frames after the encoder merges patches; one frame per temporal group."""
    unit = profile.grid_cell
    if min(width, height) <= 0 or width % unit or height % unit:
        raise ValueError("resolved dimensions must align to the model spatial grid")
    merge = profile.spatial_merge_size
    return num_frames * (width // profile.patch_size // merge) * (height // profile.patch_size // merge)


def validate_video_pixels(*, num_frames: int, width: int, height: int, profile: FrameProfile) -> None:
    """The per-call pixel cap; frames and size are never reduced to fit it."""
    total = num_frames * width * height
    if total > profile.video_max_pixels:
        raise MediaContractError(
            "video_pixels_exceeded",
            "sampled video exceeds video_max_pixels; choose fewer frames or a smaller resize_long_edge",
            details={
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "video_pixels": total,
                "video_max_pixels": profile.video_max_pixels,
            },
        )


def resolve_frames(requested: VideoSpec, *, source: SourceVideoMetadata, profile: FrameProfile) -> ResolvedFrames:
    """Pick the source frame of each sample and the frame size; independent of any text."""
    source.validate()
    if requested.end > source.duration_seconds:
        raise MediaContractError(
            "invalid_source_range",
            "end exceeds the complete source video duration",
            details={"end": requested.end, "source_duration_seconds": source.duration_seconds},
        )
    width, height = resolve_resize(
        source_width=source.width,
        source_height=source.height,
        resize_long_edge=requested.resize_long_edge,
        profile=profile,
    )
    tokens = vision_tokens(num_frames=requested.num_frames, width=width, height=height, profile=profile)
    validate_video_pixels(num_frames=requested.num_frames, width=width, height=height, profile=profile)
    timestamps = source.frame_timestamps_seconds
    sampling = sampling_timestamps(requested)
    # The frame on screen at each sampling time: a source frame shows until the next one starts, so a frame that
    # starts before `start` can still be sampled inside the window. None is invented before the video starts.
    indices = tuple(bisect.bisect_right(timestamps, t) - 1 for t in sampling)
    if indices[0] < 0:
        raise MediaContractError(
            "source_video_not_started",
            "the first sampling time precedes the first source video frame; choose a later start",
            details={"first_source_pts": timestamps[0], "first_sampling_time": sampling[0]},
        )
    return ResolvedFrames(
        requested=requested,
        source_width=source.width,
        source_height=source.height,
        source_duration_seconds=source.duration_seconds,
        frame_indices=indices,
        timestamps_seconds=tuple(timestamps[index] for index in indices),
        sampling_timestamps_seconds=sampling,
        width=width,
        height=height,
        vision_tokens=tokens,
    )


def frame_labels(timestamps: tuple[float, ...]) -> str:
    """One line per frame, in the order of the images: the frame's sampling time."""
    if not timestamps or any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise ValueError("frame labels require finite, non-negative seconds")
    if any(left > right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("frame label times must be ordered")
    return "\n".join(f"Frame {index + 1}: {timestamp:.2f} s" for index, timestamp in enumerate(timestamps))


def video_source_label(spec: VideoSpec) -> str:
    """The video's path as the text names it, without characters a tokenizer could read as special tokens."""
    if any(character in spec.video_path for character in "\r\n\x00"):
        raise ValueError("model-visible media paths must not contain line breaks or NUL")
    visible_path = PurePosixPath(spec.video_path).as_posix()
    if "<" in visible_path or ">" in visible_path or "\\" in visible_path:
        visible_path = json.dumps(visible_path, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return visible_path


def sampling_receipt(resolved: ResolvedFrames) -> str:
    """What was sampled, from the resolved call only: the frame count (and how many differ when the video holds
    frames), the video, the window, the spacing of the sampling times, the frame size and the visual tokens."""
    requested = resolved.requested
    count = requested.num_frames
    unique = len(set(resolved.frame_indices))
    frames = f"{count} frames" if count > 1 else "1 frame"
    if unique < count:
        frames += f" ({unique} distinct)"
    # The sampling times are the bin centers, so consecutive ones are one bin apart.
    spacing = f", one every {(requested.end - requested.start) / count:.3g} s" if count > 1 else ""
    return (
        f"Sampled {frames} of {video_source_label(requested)} from {requested.start:.2f} s to {requested.end:.2f} s"
        f"{spacing}, at {resolved.width}×{resolved.height} ({resolved.vision_tokens} visual tokens)."
    )


def watch_video_text(resolved: ResolvedFrames) -> str:
    """The tool result's text. For an OpenAI-compatible provider, OpenCode sends a tool result's images after the tool
    results of the response (video_tools.mjs), so the text says what was sampled, that the frames follow the tool
    results, and then each frame's sampling time in the order of the images."""
    count = resolved.requested.num_frames
    where = (
        f"The {count} frames are attached after the tool results, in this order:"
        if count > 1
        else "The frame is attached after the tool results:"
    )
    return f"{sampling_receipt(resolved)}\n{where}\n{frame_labels(resolved.sampling_timestamps_seconds)}"
