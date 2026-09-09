# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "hsg/aav2/media.py"
spec = importlib.util.spec_from_file_location("checkpoint_media", MODULE)
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)
requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe required"
)


def test_plain_files_and_zip_are_byte_preserved_and_existing_output_fails(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "notes.txt").write_text("Reference notes\n")
    (source / "empty").mkdir()
    with zipfile.ZipFile(source / "files.zip", "w") as archive:
        archive.writestr("nested/notes.txt", "All original content")
        archive.writestr("empty/", "")
    output = tmp_path / "output"
    manifest = media.build(source, output)
    assert json.loads((tmp_path / "output.media.json").read_text()) == manifest
    assert (output / "empty").is_dir()
    for entry in manifest["entries"]:
        assert (source / entry["source"]).read_bytes() == (output / entry["output"]).read_bytes()
        assert entry["source_sha256"] == entry["output_sha256"]
    with pytest.raises(FileExistsError):
        media.build(source, output)


def test_native_repeat_bookkeeping_is_distinct_from_ordinary_pickle_evidence(tmp_path):
    source = tmp_path / "source"
    repeat = source / "task_one/repeat_0"
    repeat.mkdir(parents=True)
    payload = b"\x80\x04\x95\x00native history"
    (repeat / "history.pkl").write_bytes(payload)
    (repeat / "deliverable.pkl").write_bytes(payload)
    finish = '{"paths":["final.mp4","excluded_intermediate.mp4"]}'
    (repeat / "finish_params.json").write_text(finish)
    output = tmp_path / "output"
    manifest = media.build(source, output)
    assert (output / "task_one/repeat_0/history.pkl").read_bytes() == payload
    assert (output / "task_one/repeat_0/finish_params.json").read_text() == finish
    assert (output / "task_one/repeat_0/deliverable.pkl").read_bytes() == payload
    kinds = {Path(entry["source"]).name: entry["kind"] for entry in manifest["entries"]}
    assert kinds == {"finish_params.json": "bookkeeping", "history.pkl": "bookkeeping", "deliverable.pkl": "unchanged"}


def test_nested_destination_and_source_symlink_fail_without_changing_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        media.build(source, source / "new" / "output")
    assert list(source.iterdir()) == []
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        media.build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "a/../b.txt", "nested.zip", "C:/file.txt"])
def test_unsafe_or_nested_zip_fails_without_publication(tmp_path, name):
    source = tmp_path / "source"
    source.mkdir()
    with zipfile.ZipFile(source / "files.zip", "w") as archive:
        archive.writestr(name, b"content")
    with pytest.raises(ValueError, match="ZIP member"):
        media.build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "output.media.json").exists()
    assert not list(tmp_path.glob(".gdpval-media-*"))


@pytest.mark.parametrize("case", ["duplicate", "symlink", "corrupt"])
def test_zip_rejected_members_never_disappear_silently(tmp_path, case):
    source = tmp_path / "source"
    source.mkdir()
    archive_path = source / "files.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("file.txt", b"unique payload")
        if case == "duplicate":
            with pytest.warns(UserWarning):
                archive.writestr("file.txt", b"other payload")
        elif case == "symlink":
            info = zipfile.ZipInfo("link.txt")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"file.txt")
    if case == "corrupt":
        archive_path.write_bytes(archive_path.read_bytes().replace(b"unique payload", b"broken payload"))
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        media.build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "name,payload",
    [("drawing.step", b"STEP text"), ("image.psd", b"8BPS"), ("data.bin", b"\0"), ("files.tar.gz", b"\x1f\x8b\0")],
)
def test_other_formats_are_byte_preserved_loose_and_inside_zip(tmp_path, name, payload):
    source = tmp_path / "source"
    source.mkdir()
    (source / name).write_bytes(payload)
    with zipfile.ZipFile(source / "evidence.zip", "w") as archive:
        archive.writestr(name, payload)
    output = tmp_path / "output"
    manifest = media.build(source, output)
    assert (output / name).read_bytes() == payload
    assert (output / "evidence.zip").read_bytes() == (source / "evidence.zip").read_bytes()
    assert all(entry["source_sha256"] == entry["output_sha256"] for entry in manifest["entries"])
    members = next(entry["members"] for entry in manifest["entries"] if entry["kind"] == "zip")
    assert members[0]["source"] == name
    assert members[0]["source_sha256"] == members[0]["output_sha256"] == hashlib.sha256(payload).hexdigest()


@requires_ffmpeg
@pytest.mark.parametrize("codec,extension", [("pcm_s16le", ".wav"), ("pcm_s24be", ".aiff"), ("pcm_f32le", ".wav")])
def test_real_pcm_conversion_and_float_passthrough(tmp_path, codec, extension):
    source = tmp_path / "source"
    source.mkdir()
    audio = source / ("sound" + extension)
    media._run(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "0.3", "-c:a", codec, str(audio)])
    original = audio.read_bytes()
    manifest = media.build(source, tmp_path / "output")
    entry = manifest["entries"][0]
    if codec == "pcm_f32le":
        assert entry["kind"] == "unchanged"
        assert (tmp_path / "output" / entry["output"]).read_bytes() == original
    else:
        assert entry["kind"] == "lossless_flac"
        assert entry["output_bytes"] < entry["source_bytes"]
        assert media._audio_identity(audio, tmp_path) == media._audio_identity(
            tmp_path / "output" / entry["output"], tmp_path
        )
    assert audio.read_bytes() == original


@requires_ffmpeg
def test_real_zip_video_proxy_keeps_every_member_and_source_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "MIN_VIDEO_BYTES", 0)
    clip = tmp_path / "clip.mp4"
    media._run(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1600x900:rate=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            "-t",
            "0.6",
            "-c:v",
            "libx264",
            "-threads:v",
            "1",
            "-c:a",
            "aac",
            str(clip),
        ]
    )
    source = tmp_path / "source"
    source.mkdir()
    archive_path = source / "assets.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(clip, "nested/clip.mp4")
        archive.writestr("notes.txt", "Every member retained")
    original = archive_path.read_bytes()
    manifest = media.build(source, tmp_path / "output")
    members = manifest["entries"][0]["members"]
    assert {m["source"] for m in members} == {"nested/clip.mp4", "notes.txt"}
    assert [m["kind"] for m in members] == ["h264_video", "unchanged"]
    with zipfile.ZipFile(tmp_path / "output/assets.zip") as archive:
        assert archive.read("notes.txt") == b"Every member retained"
        proxy = tmp_path / "proxy.mp4"
        proxy.write_bytes(archive.read("nested/clip.mp4.mp4"))
    video = media._video_identity(proxy)
    assert video["video"]["width"] <= 1280 and video["video"]["height"] <= 720
    assert video["audio"][0]["codec_name"] == "aac"
    assert archive_path.read_bytes() == original
    assert manifest["entries"][0]["source_sha256"] == hashlib.sha256(original).hexdigest()
    repeated = media.build(source, tmp_path / "output2")
    assert repeated == manifest


@pytest.mark.parametrize("unsupported", ["extra_audio", "subtitles", "alpha"])
def test_unsupported_video_features_fail_before_conversion(tmp_path, monkeypatch, unsupported):
    source = tmp_path / "source"
    source.mkdir()
    (source / "clip.mp4").write_bytes(b"fixture")
    monkeypatch.setattr(media, "MIN_VIDEO_BYTES", 0)
    video = {"codec_type": "video", "pix_fmt": "rgba" if unsupported == "alpha" else "yuv420p"}
    extra = [{"codec_type": "audio"}, {"codec_type": "audio"}] if unsupported == "extra_audio" else []
    if unsupported == "subtitles":
        extra.append({"codec_type": "subtitle"})
    monkeypatch.setattr(media, "_probe", lambda _: {"streams": [video, *extra], "format": {"duration": "1"}})
    with pytest.raises(ValueError, match="unsupported"):
        media.build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_ffmpeg_failure_does_not_publish_partial_tree(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "clip.mp4").write_bytes(b"original")
    monkeypatch.setattr(media, "MIN_VIDEO_BYTES", 0)

    def fail_conversion(original, target):
        target.write_bytes(b"partial")
        raise subprocess.TimeoutExpired("ffmpeg", 1800)

    monkeypatch.setattr(media, "_video", fail_conversion)
    with pytest.raises(subprocess.TimeoutExpired):
        media.build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "output.media.json").exists()
    assert (source / "clip.mp4").read_bytes() == b"original"
