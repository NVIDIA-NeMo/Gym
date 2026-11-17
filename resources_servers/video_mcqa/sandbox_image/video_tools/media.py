# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode the frames contract.py picks and return them as JPEG images; return an image file unchanged.

A source's presentation times and keyframes come from one ffprobe pass and are kept in a small index file per source,
so later calls on the same file skip the probe. A call decodes only the frames it needs: FFmpeg seeks to the keyframe
before each group of frames and checks that every decoded frame has the presentation time the index gives; any
mismatch decodes the window sequentially instead. Frames are resized by pixel area. Sources are never modified.
"""

import base64
import bisect
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .contract import (
    JPEG_QUALITY,
    MAX_FRAME_PIXELS,
    MAX_IMAGE_BYTES,
    FrameProfile,
    MediaContractError,
    ResolvedFrames,
    SourceVideoMetadata,
    VideoSpec,
    resolve_frames,
    watch_video_text,
)


_INDEX_SCHEMA = 1
# Index files are a few bytes per frame; this holds about 8M frames (3 days at 30 fps).
_INDEX_MAX_BYTES = 256 << 20
# The packet pass prints about 50 bytes per packet: 512 MiB reads about 10M packets.
_PACKET_PROBE_MAX_BYTES = 512 << 20
_STREAM_ENTRIES = (
    "stream=width,height,start_time,duration,time_base:stream_tags=rotate:"
    "stream_side_data=rotation:format=duration,start_time"
)
_STREAM_FIELDS = frozenset({"width", "height", "start_time", "duration", "time_base"})
_FORMAT_FIELDS = frozenset({"duration", "start_time"})
# Targets at most this many frames apart share one seek.
_SEEK_JOIN_GAP_FRAMES = 32
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SourceIndex:
    source: SourceVideoMetadata
    source_stat: tuple[int, ...]
    # Integer presentation times and keyframe positions, in presentation order; empty without a usable seek index.
    raw_pts: tuple[int, ...]
    key_frames: tuple[int, ...]
    time_base: Fraction | None


@dataclass(frozen=True)
class _DecodedFrames:
    frames: np.ndarray
    backend: str
    groups: int
    fallback_reason: str | None


class _SeekMismatch(RuntimeError):
    """A seek decode that does not reproduce the index; only this falls back to the sequential decode."""


def _source_stat(source: Path) -> tuple[int, ...]:
    stat = source.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def workspace_file(path: str, *, workspace_root: Path) -> Path:
    """A regular file under the workspace; a symlink that leads out of it is refused."""
    root = workspace_root.resolve(strict=True)
    requested = Path(path)
    source = (requested if requested.is_absolute() else root / requested).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise MediaContractError(
            "invalid_media_path",
            "media must be a regular file within the configured workspace",
            details={"path": path},
        )
    return source


def _run_process(
    command: list[str], *, timeout: float, maximum_bytes: int, maximum_stderr_bytes: int = 64 << 10
) -> tuple[bytes, bytes]:
    """Run a child with bounded time and output; both streams go to temporary files, not pipes."""
    if timeout <= 0:
        raise MediaContractError("decode_timeout", "media decode deadline expired", details={})
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        deadline = time.monotonic() + timeout
        with subprocess.Popen(command, stdout=output, stderr=errors) as process:
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise MediaContractError(
                            "decode_timeout",
                            "media subprocess exceeded its deadline",
                            details={"timeout_seconds": timeout},
                        )
                    if (
                        os.fstat(output.fileno()).st_size > maximum_bytes
                        or os.fstat(errors.fileno()).st_size > maximum_stderr_bytes
                    ):
                        raise MediaContractError(
                            "media_output_exceeded",
                            "media subprocess exceeds the output resource limit",
                            details={"maximum_bytes": maximum_bytes},
                        )
                    time.sleep(0.02)
            finally:
                if process.poll() is None:
                    process.kill()
                returncode = process.wait()
        size = os.fstat(output.fileno()).st_size
        if size > maximum_bytes or os.fstat(errors.fileno()).st_size > maximum_stderr_bytes:
            raise MediaContractError(
                "media_output_exceeded",
                "media subprocess output exceeds the resource limit",
                details={"actual_bytes": size, "maximum_bytes": maximum_bytes},
            )
        errors.seek(0)
        if returncode:
            raise MediaContractError(
                "decode_failed",
                "media subprocess failed",
                details={"returncode": returncode, "stderr": errors.read(4096).decode(errors="replace")},
            )
        output.seek(0)
        return output.read(), errors.read()


def _probe_json(source: Path, section: str, entries: str, *, profile: FrameProfile) -> dict:
    """One ffprobe pass over the first video stream, with its stream and container fields."""
    stdout, _ = _run_process(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            section,
            "-show_streams",
            "-show_format",
            "-show_entries",
            f"{entries}:{_STREAM_ENTRIES}",
            "-of",
            "json",
            str(source),
        ],
        timeout=profile.decode_timeout_seconds,
        maximum_bytes=64 << 20,
    )
    return json.loads(stdout)


@dataclass(frozen=True)
class _PacketPass:
    """Every packet in decode order, then the first video stream's and the container's fields.

    The fields are nested as ffprobe's JSON writer nests them. ``pts`` is None when some packet cannot stand in for
    its decoded frame (no presentation time, or marked discarded or corrupt), or when there is no packet.
    """

    pts: list[int] | None
    seconds: list[float]
    key_frames: list[bool]
    stream: dict
    container: dict


def _csv_section(line: bytes, name: bytes) -> dict | None:
    """The CSV line of the stream or the container as JSON's fields, or None for any other shape."""
    fields = line.split(b",")
    if fields[0] != name or b'"' in line:
        return None
    keys = _STREAM_FIELDS if name == b"stream" else _FORMAT_FIELDS
    section: dict = {}
    seen: set[str] = set()
    for field in fields[1:]:
        if not field:
            continue  # a side data entry without a shown field leaves an empty one
        key, separator, text = field.decode(errors="replace").partition("=")
        if not separator or key in seen:
            return None
        seen.add(key)
        if key in keys:
            if text == "N/A":
                if key not in ("start_time", "duration"):
                    return None
                continue  # the JSON writer leaves an unknown time out
            section[key] = text
        elif name == b"stream" and key == "tag:rotate":
            section["tags"] = {"rotate": text}
        elif name == b"stream" and key.startswith("side_datum/") and key.endswith(":rotation"):
            seen.discard(key)  # one per display matrix
            section.setdefault("side_data_list", []).append({"rotation": text})
        else:
            return None
    return section


def _csv_packet_pass(source: Path, *, profile: FrameProfile) -> _PacketPass | None:
    """The packet pass printed as CSV (about 50 bytes per packet, where JSON takes about 120).

    None when ffprobe fails, or for output this does not take apart exactly (sections besides the packets, one
    stream and the container, as MPEG-TS programs print, or an unexpected field); the JSON pass then reads it.
    """
    try:
        raw, _ = _run_process(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"packet=pts,pts_time,flags:{_STREAM_ENTRIES}",
                "-of",
                "csv=nk=0",
                str(source),
            ],
            timeout=profile.decode_timeout_seconds,
            maximum_bytes=_PACKET_PROBE_MAX_BYTES,
        )
    except MediaContractError as error:
        if error.code == "decode_failed":
            return None
        raise
    pts: list[int] = []
    seconds: list[float] = []
    key_frames: list[bool] = []
    usable = True
    position = 0
    while raw.startswith(b"packet,", position):
        end = raw.find(b"\n", position)
        if end < 0:
            return None
        fields = raw[position + 7 : end].split(b",")
        position = end + 1
        # pts, pts_time and flags; packet side data (no field shown) adds only empty fields.
        if (
            len(fields) < 3
            or any(fields[3:])
            or not fields[0].startswith(b"pts=")
            or not fields[1].startswith(b"pts_time=")
            or not fields[2].startswith(b"flags=")
        ):
            return None
        if not usable:
            continue
        value, moment, flags = fields[0][4:], fields[1][9:], fields[2][6:]
        if value == b"N/A" or moment == b"N/A" or b"D" in flags or b"C" in flags:
            usable = False  # the frames are decoded instead
            continue
        try:
            pts.append(int(value))
            seconds.append(float(moment))
        except ValueError:
            return None
        key_frames.append(b"K" in flags)
    lines = raw[position:].split(b"\n")
    if len(lines) != 3 or lines[2]:
        return None
    stream = _csv_section(lines[0], b"stream")
    container = _csv_section(lines[1], b"format")
    if stream is None or container is None:
        return None
    return _PacketPass(pts if usable and pts else None, seconds, key_frames, stream, container)


def _probe_source_index(source: Path, *, profile: FrameProfile) -> _SourceIndex:
    """Presentation times and keyframes of every frame, from one complete probe.

    Every demuxed packet carries its frame's presentation time and keyframe flag, so the index reads packets (well
    under a second for an hour of video) instead of decoding the stream. A stream whose packets cannot stand in for
    its frames (one without a presentation time, or one marked discarded or corrupt) is decoded instead.
    """
    before = _source_stat(source)
    packets = _csv_packet_pass(source, profile=profile)
    if packets is None:
        result = _probe_json(source, "-show_packets", "packet=pts,pts_time,flags", profile=profile)
        listed = result.get("packets", [])
        usable = bool(listed) and all(
            type(packet.get("pts")) is int and "pts_time" in packet and not {"D", "C"} & set(packet.get("flags", ""))
            for packet in listed
        )
    else:
        result = {"streams": [packets.stream], "format": packets.container}
        usable = packets.pts is not None
    frames: list[dict] = []
    if usable and packets is None:
        # Decode order -> presentation order.
        frames = [
            {
                "best_effort_timestamp": packet["pts"],
                "best_effort_timestamp_time": packet["pts_time"],
                "key_frame": int("K" in packet.get("flags", "")),
            }
            for packet in sorted(listed, key=lambda packet: packet["pts"])
        ]
    elif not usable:
        packets = None
        result = _probe_json(
            source,
            "-show_frames",
            "frame=best_effort_timestamp,best_effort_timestamp_time,key_frame",
            profile=profile,
        )
        frames = result["frames"]
    streams = result["streams"]
    if len(streams) != 1:
        raise ValueError("video must have one selected primary stream")
    stream = streams[0]
    if int(stream.get("tags", {}).get("rotate", 0)) or any(
        int(item.get("rotation", 0)) for item in stream.get("side_data_list", [])
    ):
        raise MediaContractError(
            "unsupported_rotation", "normalize rotated video into a new source before freezing it", details={}
        )
    # Times count from the start of the whole container, not of the first video frame: an audio stream may start
    # earlier, and the video's own start would shift every time against the source.
    container = result["format"]
    origin = float(container.get("start_time", stream.get("start_time", 0)))
    if packets is not None:
        # Decode order -> presentation order (stable, as sorted()).
        order = np.argsort(np.asarray(packets.pts, dtype=np.int64), kind="stable")
        timestamps = tuple((np.asarray(packets.seconds, dtype=np.float64)[order] - origin).tolist())
    else:
        timestamps = tuple(float(frame["best_effort_timestamp_time"]) - origin for frame in frames)
    if origin == 0:
        duration = float(container["duration"])
    elif "duration" in stream:
        # A shifted MP4 or MPEG-TS clock, without counting the time before playback starts as video.
        duration = float(stream.get("start_time", origin)) - origin + float(stream["duration"])
    else:
        duration = float(container["duration"]) - origin
    metadata = SourceVideoMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_seconds=duration,
        frame_timestamps_seconds=timestamps,
    )
    if before != _source_stat(source):
        raise MediaContractError("source_changed", "source changed while it was indexed", details={})
    metadata.validate()
    seek_available = "time_base" in stream
    if packets is not None:
        raw_pts = tuple(np.asarray(packets.pts, dtype=np.int64)[order].tolist()) if seek_available else ()
        key_frames = (
            tuple(np.flatnonzero(np.asarray(packets.key_frames, dtype=bool)[order]).tolist()) if seek_available else ()
        )
    else:
        # Some streams have frames without integer times: they keep the same timing but no verified seeks.
        seek_available = seek_available and all(
            type(frame.get("best_effort_timestamp")) is int and type(frame.get("key_frame")) is int for frame in frames
        )
        raw_pts = tuple(frame["best_effort_timestamp"] for frame in frames) if seek_available else ()
        key_frames = tuple(i for i, frame in enumerate(frames) if frame["key_frame"] == 1) if seek_available else ()
    time_base = Fraction(stream["time_base"]) if seek_available else None
    return _SourceIndex(metadata, before, raw_pts, key_frames, time_base)


def _index_record(index: _SourceIndex) -> bytes:
    return json.dumps(
        {
            "schema_version": _INDEX_SCHEMA,
            "source_stat": list(index.source_stat),
            "width": index.source.width,
            "height": index.source.height,
            "duration_seconds": index.source.duration_seconds,
            "frame_timestamps_seconds": list(index.source.frame_timestamps_seconds),
            "raw_pts": list(index.raw_pts),
            "key_frames": list(index.key_frames),
            "time_base": str(index.time_base) if index.time_base is not None else None,
        },
        separators=(",", ":"),
    ).encode()


def _load_index(path: Path, *, expected_stat: tuple[int, ...]) -> _SourceIndex:
    """An index file, checked as strictly as a new probe; a stale or damaged one raises."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _INDEX_MAX_BYTES:
        raise ValueError("invalid source index file")
    record = json.loads(path.read_bytes())
    fields = {
        "schema_version",
        "source_stat",
        "width",
        "height",
        "duration_seconds",
        "frame_timestamps_seconds",
        "raw_pts",
        "key_frames",
        "time_base",
    }
    if not isinstance(record, dict) or set(record) != fields or record["schema_version"] != _INDEX_SCHEMA:
        raise ValueError("source index schema mismatch")
    stat = tuple(record["source_stat"])
    if any(type(value) is not int for value in stat) or stat != expected_stat:
        raise ValueError("source index file identity mismatch")
    times, raw_pts, key_frames = record["frame_timestamps_seconds"], record["raw_pts"], record["key_frames"]
    if (
        type(record["width"]) is not int
        or type(record["height"]) is not int
        or type(record["duration_seconds"]) not in (int, float)
        or any(type(value) not in (int, float) for value in times)
        or any(type(value) is not int for value in (*raw_pts, *key_frames))
    ):
        raise ValueError("invalid source index values")
    metadata = SourceVideoMetadata(
        width=record["width"],
        height=record["height"],
        duration_seconds=float(record["duration_seconds"]),
        frame_timestamps_seconds=tuple(float(value) for value in times),
    )
    metadata.validate()
    if any(left >= right for left, right in zip(raw_pts, raw_pts[1:])) or any(
        left >= right for left, right in zip(key_frames, key_frames[1:])
    ):
        raise ValueError("source index presentation order is invalid")
    time_base = Fraction(record["time_base"]) if record["time_base"] is not None else None
    if raw_pts:
        if len(raw_pts) != len(times) or time_base is None or time_base <= 0:
            raise ValueError("source index PTS length/time base mismatch")
        if key_frames and (key_frames[0] < 0 or key_frames[-1] >= len(raw_pts)):
            raise ValueError("invalid source index keyframe position")
    elif key_frames or time_base is not None:
        raise ValueError("partial source seek index")
    return _SourceIndex(metadata, stat, tuple(raw_pts), tuple(key_frames), time_base)


@contextmanager
def _index_lock(path: Path, *, timeout: float) -> Iterator[None]:
    """Serialize the first probe of a source across tool processes, with a bounded wait."""
    deadline = time.monotonic() + timeout
    with path.open("a+b") as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise MediaContractError("decode_timeout", "source index lock deadline expired", details={})
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _check_unchanged(source: Path, index: _SourceIndex) -> None:
    """The source still has the file status it had when it was indexed (the resources server checked its bytes)."""
    if _source_stat(source) != index.source_stat:
        raise MediaContractError("source_changed", "source changed since indexing", details={})


def _get_source_index(source: Path, *, cache_dir: Path, profile: FrameProfile) -> tuple[_SourceIndex, bool]:
    """The source's index, read from its index file or probed once and written there; True when it was read."""
    identity = _source_stat(source)
    executable = shutil.which("ffprobe")
    if executable is None:
        raise FileNotFoundError("ffprobe is unavailable")
    probe = Path(executable).resolve(strict=True)
    key = hashlib.sha256(
        json.dumps(
            {
                "schema": _INDEX_SCHEMA,
                "path": str(source.resolve()),
                "stat": identity,
                "probe_path": str(probe),
                "probe_stat": _source_stat(probe),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    record_path = cache_dir / (key + ".json")
    if record_path.exists():
        return _load_index(record_path, expected_stat=identity), True
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _index_lock(cache_dir / (key + ".lock"), timeout=profile.decode_timeout_seconds):
        if record_path.exists():
            return _load_index(record_path, expected_stat=identity), True
        index = _probe_source_index(source, profile=profile)
        if index.source_stat != identity:
            raise MediaContractError("source_changed", "source changed before indexing", details={})
        data = _index_record(index)
        if len(data) > _INDEX_MAX_BYTES:
            raise ValueError("source index exceeds the bounded metadata limit")
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                os.link(temporary, record_path)
                record_path.chmod(0o400)
            finally:
                temporary.unlink(missing_ok=True)
        return index, False


def _decode_sequential(source: Path, resolved: ResolvedFrames, *, timeout: float) -> np.ndarray:
    """Decode the unique source frames in one pass, then repeat the held ones."""
    targets = tuple(dict.fromkeys(resolved.frame_indices))
    count = len(targets)
    select = "+".join(f"eq(n\\,{index})" for index in targets)
    raw, _ = _run_process(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-noautorotate",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"select={select},scale={resolved.width}:{resolved.height}:flags=area",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(count),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout=timeout,
        maximum_bytes=count * resolved.width * resolved.height * 3,
    )
    if len(raw) != count * resolved.width * resolved.height * 3:
        raise ValueError("decoder did not return the exact requested RGB frame count")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, resolved.height, resolved.width, 3)
    if count != resolved.requested.num_frames:
        frames = frames[np.searchsorted(targets, resolved.frame_indices)]
    return frames


def _seek_groups(index: _SourceIndex, resolved: ResolvedFrames) -> list[tuple[int, tuple[int, ...]]]:
    """Unique target frames grouped by the keyframe a seek starts from; near groups share one seek."""
    groups: list[tuple[int, tuple[int, ...]]] = []
    for target in dict.fromkeys(resolved.frame_indices):
        position = bisect.bisect_right(index.key_frames, target) - 1
        start = index.key_frames[position] if position >= 0 else 0
        if groups and start <= groups[-1][1][-1] + _SEEK_JOIN_GAP_FRAMES:
            groups[-1] = (groups[-1][0], groups[-1][1] + (target,))
        else:
            groups.append((start, (target,)))
    return groups


def _decode_seek_group(
    source: Path,
    index: _SourceIndex,
    resolved: ResolvedFrames,
    *,
    start_index: int,
    targets: tuple[int, ...],
    timeout: float,
) -> np.ndarray:
    if index.time_base is None:
        raise _SeekMismatch("source has no verified integer PTS time base")
    expected_pts = [index.raw_pts[target] for target in targets]
    # FFmpeg's input -ss counts from the start of the whole container; with -copyts the selection below compares
    # each frame's time in the stream's own clock.
    seek_seconds = index.source.frame_timestamps_seconds[start_index]
    command = ["ffmpeg", "-nostdin", "-nostats", "-v", "info", "-threads", "2", "-noautorotate", "-copyts"]
    if start_index > 0:
        command.extend(["-ss", f"{seek_seconds:.9f}", "-noaccurate_seek"])
    select = "+".join(f"eq(pts\\,{pts})" for pts in expected_pts)
    command.extend(
        [
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"select={select},scale={resolved.width}:{resolved.height}:flags=area,showinfo=checksum=0",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(len(targets)),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    expected_bytes = len(targets) * resolved.width * resolved.height * 3
    stdout, stderr = _run_process(
        command,
        timeout=timeout,
        maximum_bytes=expected_bytes,
        maximum_stderr_bytes=(64 << 10) + 2048 * len(targets),
    )
    observed_pts = [int(value) for value in re.findall(rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:", stderr)]
    time_bases = re.findall(rb"config in time_base:\s*(\d+)/(\d+)", stderr)
    if (
        observed_pts != expected_pts
        or len(stdout) != expected_bytes
        or not time_bases
        or any(Fraction(int(n), int(d)) != index.time_base for n, d in time_bases)
    ):
        raise _SeekMismatch("decoded PTS/time base/frame count differs from the fixed source index")
    return np.frombuffer(stdout, dtype=np.uint8).reshape(len(targets), resolved.height, resolved.width, 3)


def _decode_frames(
    source: Path, index: _SourceIndex, resolved: ResolvedFrames, *, profile: FrameProfile
) -> _DecodedFrames:
    """The resolved frames, area-resized (as the model's image processor resizes); errors other than a seek
    mismatch raise."""
    deadline = time.monotonic() + profile.decode_timeout_seconds
    if not index.raw_pts or not index.key_frames:
        frames = _decode_sequential(source, resolved, timeout=deadline - time.monotonic())
        return _DecodedFrames(frames, "ffmpeg_sequential", 1, "no_verified_seek_index")
    groups = _seek_groups(index, resolved)
    try:
        decoded = [
            _decode_seek_group(
                source,
                index,
                resolved,
                start_index=start,
                targets=targets,
                timeout=deadline - time.monotonic(),
            )
            for start, targets in groups
        ]
    except _SeekMismatch as error:
        _LOGGER.warning("exact_seek_fallback: %s", error)
        frames = _decode_sequential(source, resolved, timeout=deadline - time.monotonic())
        return _DecodedFrames(frames, "ffmpeg_sequential", len(groups), str(error))
    frames = np.concatenate(decoded, axis=0)
    if len(frames) != resolved.requested.num_frames:
        targets = tuple(dict.fromkeys(resolved.frame_indices))
        frames = frames[np.searchsorted(targets, resolved.frame_indices)]
    return _DecodedFrames(frames, "ffmpeg_exact_seek", len(groups), None)


def frame_jpeg(frame: np.ndarray) -> bytes:
    """One RGB frame as JPEG, quality 95 without chroma subsampling (4:4:4)."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame)).save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    return buffer.getvalue()


def watch_video(
    requested: VideoSpec, *, workspace_root: Path, cache_dir: Path, profile: FrameProfile
) -> dict[str, object]:
    """The requested frames as JPEG data URLs, in order, with the text that labels them."""
    source = workspace_file(requested.video_path, workspace_root=workspace_root)
    # The text names the workspace-relative path, never a machine-specific absolute one.
    requested = requested.model_copy(update={"video_path": source.relative_to(workspace_root.resolve()).as_posix()})
    index_started = time.monotonic()
    index, index_cache_hit = _get_source_index(source, cache_dir=cache_dir, profile=profile)
    index_seconds = time.monotonic() - index_started
    resolved = resolve_frames(requested, source=index.source, profile=profile)
    byte_count = requested.num_frames * resolved.width * resolved.height * 3
    if byte_count > profile.max_materialized_bytes:
        raise MediaContractError(
            "media_memory_exceeded",
            "exact RGB frames exceed the materialization memory limit",
            details={"rgb_bytes": byte_count, "maximum_bytes": profile.max_materialized_bytes},
        )
    decode_started = time.monotonic()
    decoded = _decode_frames(source, index, resolved, profile=profile)
    decode_seconds = time.monotonic() - decode_started
    _check_unchanged(source, index)
    encode_started = time.monotonic()
    images = [frame_jpeg(frame) for frame in decoded.frames]
    return {
        "output": watch_video_text(resolved),
        "attachments": [
            {"mime": "image/jpeg", "url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}
            for image in images
        ],
        "metadata": {
            "resolved_video": resolved.to_dict(),
            "frame_sha256": [hashlib.sha256(image).hexdigest() for image in images],
            "video_pixels": requested.num_frames * resolved.width * resolved.height,
            "video_max_pixels": profile.video_max_pixels,
            "processing_stats": {
                "source_index_cache_hit": index_cache_hit,
                "index_seconds": index_seconds,
                "decode_seconds": decode_seconds,
                "encode_seconds": time.monotonic() - encode_started,
                "jpeg_bytes": sum(len(image) for image in images),
                "decode_backend": decoded.backend,
                "seek_groups": decoded.groups,
                "fallback_reason": decoded.fallback_reason,
            },
        },
    }


def view_image(path: str, *, workspace_root: Path) -> dict[str, object]:
    """An image file's own bytes, after the byte and area checks; the image is not resized."""
    source = workspace_file(path, workspace_root=workspace_root)
    if source.stat().st_size > MAX_IMAGE_BYTES:
        raise MediaContractError(
            "image_too_large", "image exceeds its byte limit", details={"maximum_bytes": MAX_IMAGE_BYTES}
        )
    with Image.open(source) as image:
        width, height = image.size
        pixels = width * height
        if pixels > MAX_FRAME_PIXELS:
            raise MediaContractError(
                "image_pixels_exceeded",
                "image exceeds the 2048*2048 pixel limit; create a smaller copy",
                details={
                    "width": width,
                    "height": height,
                    "image_pixels": pixels,
                    "max_image_pixels": MAX_FRAME_PIXELS,
                },
            )
        image.verify()
        image_format = image.format
        if image_format is None:
            raise ValueError("image has no recognized MIME format")
        mime = Image.MIME[image_format]
    data = source.read_bytes()
    return {
        "output": f"Image: {source.relative_to(workspace_root.resolve()).as_posix()}",
        "attachments": [{"mime": mime, "url": f"data:{mime};base64," + base64.b64encode(data).decode()}],
        "metadata": {"source_sha256": hashlib.sha256(data).hexdigest(), "width": width, "height": height},
    }
