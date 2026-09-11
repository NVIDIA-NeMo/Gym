# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a fresh media tree for GDPVal; originals are never modified.

Requires ffmpeg/ffprobe for conversion. Video proxies are lossy, not byte-budget
guarantees. PCM16/24 audio is replaced only when FLAC is smaller and decodes to
identical samples. Failed or unsupported conversions retain the original bytes.
Video proxies exclude extra tracks and alpha; videos below 8 MiB stay unchanged.
Nested ZIP members are copied unchanged. Other formats are copied unchanged; the judge checks
whether it has a supported representation. Office conversion remains Gym's
ordinary preconvert step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path

from nemo_gym.deliverables import IGNORE_FILES


MIN_VIDEO_BYTES = 8 * 1024 * 1024
VIDEO_EXTENSIONS = {".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
PCM_EXTENSIONS = {".wav", ".wave", ".aif", ".aiff"}
PCM_CODECS = {"pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be"}
PROFILE = "h264-720p-crf26-aac128+pcm16-24-flac-v1"
TIMEOUT = 1800


class IntegrityError(ValueError):
    """A generated artifact or source identity failed validation."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(arguments: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        timeout=TIMEOUT,
    )


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def _video_identity(path: Path) -> dict:
    data = _probe(path)
    streams = data["streams"]
    video = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) > 1 or len(streams) != len(video) + len(audio):
        raise ValueError(f"unsupported extra video/audio/subtitle/data tracks: {path}")
    pixel_format = video[0].get("pix_fmt", "")
    if (
        not pixel_format
        or pixel_format.startswith(("yuva", "gbrap", "rgba", "bgra", "argb", "abgr", "ya", "ayuv", "vuya"))
        or pixel_format == "pal8"
        or any(k.lower() == "alpha_mode" and str(v) != "0" for k, v in video[0].get("tags", {}).items())
    ):
        raise ValueError(f"unsupported alpha/unknown video pixel format: {path}: {pixel_format}")
    duration = float(data["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid video duration: {path}")
    return {"duration": duration, "video": video[0], "audio": audio}


def _video(source: Path, target: Path) -> dict:
    before = _video_identity(source)
    _run(
        [
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-vf",
            "scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-threads:v",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    try:
        after = _video_identity(target)
    except (ValueError, KeyError, subprocess.SubprocessError) as error:
        raise IntegrityError(f"video proxy validation failed: {source}: {error}") from error
    video = after["video"]
    if (
        video["codec_name"] != "h264"
        or video["pix_fmt"] != "yuv420p"
        or not 0 < video["width"] <= 1280
        or not 0 < video["height"] <= 720
        or bool(before["audio"]) != bool(after["audio"])
        or any(s["codec_name"] != "aac" for s in after["audio"])
        or abs(before["duration"] - after["duration"]) > max(0.5, before["duration"] * 0.01)
    ):
        raise IntegrityError(f"video proxy failed stream/duration validation: {source}")
    return {"source_duration": before["duration"], "output_duration": after["duration"]}


def _audio_identity(path: Path, workspace: Path) -> tuple:
    streams = _probe(path)["streams"]
    if len(streams) != 1 or streams[0]["codec_type"] != "audio":
        raise ValueError(f"unsupported extra audio tracks: {path}")
    audio = streams[0]
    decoded = workspace / "decoded.s32"
    try:
        _run(["-i", str(path), "-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "s32le", str(decoded)])
        return int(audio["sample_rate"]), int(audio["channels"]), decoded.stat().st_size, _hash(decoded)
    finally:
        decoded.unlink(missing_ok=True)


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)


def _file(
    source: Path,
    target: Path,
    prepare_zip: Callable[[Path], None] | None = None,
    *,
    bookkeeping: bool = False,
    inside_zip: bool = False,
) -> dict:
    before_hash = _hash(source)
    source_size = source.stat().st_size
    extension = source.suffix.lower()
    kind = "unchanged"
    extra = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    original_target = target
    try:
        if bookkeeping:
            kind = "bookkeeping"
            _copy(source, target)
        elif extension in VIDEO_EXTENSIONS and source_size >= MIN_VIDEO_BYTES:
            target = target.with_name(target.name + ".mp4")
            kind = "h264_video"
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            extra = _video(source, target)
        elif extension in PCM_EXTENSIONS:
            streams = _probe(source)["streams"]
            if len(streams) == 1 and streams[0].get("codec_name") in PCM_CODECS:
                with tempfile.TemporaryDirectory(prefix="gdpval-audio-") as temporary:
                    workspace = Path(temporary)
                    before = _audio_identity(source, workspace)
                    converted = workspace / "converted.flac"
                    _run(["-i", str(source), "-map", "0:a:0", "-c:a", "flac", str(converted)])
                    try:
                        after = _audio_identity(converted, workspace)
                    except (ValueError, KeyError, subprocess.SubprocessError) as error:
                        raise IntegrityError(f"FLAC output validation failed: {source}: {error}") from error
                    if after != before:
                        raise IntegrityError(f"FLAC decoded samples changed: {source}")
                    if converted.stat().st_size < source_size:
                        target = target.with_name(target.name + ".flac")
                        _copy(converted, target)
                        kind = "lossless_flac"
                        extra = {"sample_rate": before[0], "channels": before[1], "decoded_sha256": before[3]}
        elif extension == ".zip" and not inside_zip:
            extra = {"members": _zip(source, target, prepare_zip)}
            kind = "zip"
    except IntegrityError:
        raise
    except (ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile, zlib.error, EOFError) as error:
        target.unlink(missing_ok=True)
        target, kind, extra = original_target, "unchanged", {}
        print(f"Media conversion skipped for {source}: {type(error).__name__}: {error}", flush=True)
    if kind == "unchanged":
        _copy(source, target)
    if _hash(source) != before_hash:
        raise IntegrityError(f"source changed during preparation: {source}")
    output_hash = _hash(target)
    if kind in {"unchanged", "bookkeeping"} and output_hash != before_hash:
        raise IntegrityError(f"copied bytes changed: {source}")
    return {
        "source": str(source),
        "output": str(target),
        "kind": kind,
        "source_bytes": source_size,
        "output_bytes": target.stat().st_size,
        "source_sha256": before_hash,
        "output_sha256": output_hash,
        **extra,
    }


def _zip(source: Path, target: Path, prepare_zip: Callable[[Path], None] | None = None) -> list[dict]:
    records = []
    with tempfile.TemporaryDirectory(prefix="gdpval-zip-") as temporary, zipfile.ZipFile(source) as archive:
        workspace = Path(temporary)
        original, prepared = workspace / "source", workspace / "output"
        infos = archive.infolist()
        if len(infos) > 10000 or sum(i.file_size for i in infos) > 8 * 1024**3:
            raise ValueError(f"ZIP exceeds preparation member/expanded-byte limit: {source}")
        names = set()
        for info in infos:
            name = info.filename.replace("\\", "/")
            parts = name.rstrip("/").split("/")
            mode = stat.S_IFMT(info.external_attr >> 16)
            if (
                any(p in {"", ".", ".."} for p in parts)
                or ":" in parts[0]
                or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
                or info.file_size > 1024**3
                or name in names
            ):
                raise ValueError(f"unsupported/unsafe/duplicate ZIP member: {source}: {info.filename}")
            names.add(name)
            member = original / name
            if info.is_dir():
                member.mkdir(parents=True, exist_ok=True)
                continue
            member.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info) as src, member.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            except (RuntimeError, NotImplementedError) as error:
                raise ValueError(f"unreadable ZIP member: {info.filename}: {error}") from error
            if member.stat().st_size != info.file_size:
                raise ValueError(f"incomplete ZIP member: {source}: {name}")
        originals = {p.relative_to(original).as_posix(): _hash(p) for p in original.rglob("*") if p.is_file()}
        if prepare_zip is not None:
            prepare_zip(original)
            if any(
                not (original / name).is_file() or _hash(original / name) != sha for name, sha in originals.items()
            ):
                raise IntegrityError(f"ZIP preparation modified an original member: {source}")
        for path in sorted(original.rglob("*")):
            relative = path.relative_to(original)
            output = prepared / relative
            if path.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            record = _file(path, output, inside_zip=True)
            record["source"] = relative.as_posix()
            record["output"] = Path(record["output"]).relative_to(prepared).as_posix()
            if relative.as_posix() not in originals:
                record["generated"] = True
                record["kind"] = "office_pdf"
            records.append(record)
        if all(record["kind"] == "unchanged" for record in records):
            _copy(source, target)
        else:
            with target.open("xb") as stream, zipfile.ZipFile(stream, "w") as output_zip:
                for path in sorted(prepared.rglob("*")):
                    name = path.relative_to(prepared).as_posix() + ("/" if path.is_dir() else "")
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    if path.is_dir():
                        output_zip.writestr(info, b"")
                    else:
                        with path.open("rb") as src, output_zip.open(info, "w") as dst:
                            shutil.copyfileobj(src, dst, 1024 * 1024)
        try:
            with zipfile.ZipFile(target) as output_zip:
                for record in records:
                    with output_zip.open(record["output"]) as member:
                        digest = hashlib.file_digest(member, "sha256").hexdigest()
                    if digest != record["output_sha256"]:
                        raise IntegrityError(f"ZIP output member changed: {source}: {record['source']}")
        except (zipfile.BadZipFile, zlib.error, KeyError, EOFError) as error:
            raise IntegrityError(f"ZIP output validation failed: {source}: {error}") from error
    return records


def tree_paths(root: Path, top_level_names: set[str] | None = None) -> list[Path]:
    if top_level_names is None:
        return sorted(root.rglob("*"))
    paths = []
    for name in sorted(top_level_names):
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"invalid top-level selection: {name!r}")
        path = root / name
        if path.exists() or path.is_symlink():
            paths.append(path)
            if path.is_dir() and not path.is_symlink():
                paths.extend(path.rglob("*"))
    return sorted(paths)


def build(
    source: Path,
    destination: Path,
    *,
    prepare_zip: Callable[[Path], None] | None = None,
    top_level_names: set[str] | None = None,
) -> dict:
    """Publish a fresh tree and sibling ``<destination>.media.json`` receipt.

    Existing outputs fail; callers must choose a new destination after failure.
    The receipt is published last and is the completion marker.
    """
    source = source.resolve(strict=True)
    destination = destination.absolute()
    destination = destination.parent.resolve() / destination.name
    receipt = destination.with_name(destination.name + ".media.json")
    if destination.exists() or destination.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"media destination/receipt already exists: {destination}")
    if not source.is_dir() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and destination must be disjoint directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tempfile.TemporaryDirectory(prefix=".gdpval-media-", dir=destination.parent) as temporary:
        workspace = Path(temporary)
        output = workspace / "tree"
        output.mkdir()
        paths = tree_paths(source, top_level_names)
        for path in paths:
            relative = path.relative_to(source)
            target = output / relative
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError(f"unsupported symlink/special source: {path}")
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            bookkeeping = (
                len(relative.parts) == 3
                and relative.parts[0].startswith("task_")
                and relative.parts[1].startswith("repeat_")
                and relative.parts[1].removeprefix("repeat_").isdigit()
                and path.name in IGNORE_FILES
            )
            record = _file(path, target, prepare_zip, bookkeeping=bookkeeping)
            record["source"] = relative.as_posix()
            record["output"] = Path(record["output"]).relative_to(output).as_posix()
            entries.append(record)
        if tree_paths(source, top_level_names) != paths or any(
            _hash(source / entry["source"]) != entry["source_sha256"] for entry in entries
        ):
            raise ValueError("source tree changed during preparation")
        manifest = {"profile": PROFILE, "source": str(source), "entries": entries}
        if top_level_names is not None:
            manifest["top_level_names"] = sorted(top_level_names)
        pending = workspace / "manifest.json"
        pending.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.rename(output, destination)
        os.rename(pending, receipt)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    build(arguments.source, arguments.destination)
    print(arguments.destination.with_name(arguments.destination.name + ".media.json"))
