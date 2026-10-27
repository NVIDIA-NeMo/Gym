# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib

import pytest
from PIL import Image

from nemo_gym.web.artifacts import WebArtifactStore, _png_bytes, _safe_component


def test_safe_component_normalizes_untrusted_names() -> None:
    assert _safe_component(" ../../session / one ") == "session_one"
    assert _safe_component("...") == "session"


def test_save_screenshot_persists_inline_bytes_and_sanitizes_path(tmp_path) -> None:
    payload = b"not-a-real-png-but-valid-runtime-bytes"
    store = WebArtifactStore(tmp_path, inline_screenshots=True)

    screenshot = store.save_screenshot("../../session one", 7, payload, suffix="after click")
    path = tmp_path / "session_one" / "step-0007-after_click.png"

    assert path.read_bytes() == payload
    assert screenshot.data_url == f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"
    assert screenshot.artifact is not None
    assert screenshot.artifact.uri == path.resolve().as_uri()
    assert screenshot.artifact.size_bytes == len(payload)
    assert screenshot.artifact.sha256 == hashlib.sha256(payload).hexdigest()


def test_save_screenshot_encodes_pillow_image_without_inlining(tmp_path) -> None:
    store = WebArtifactStore(tmp_path, inline_screenshots=False)
    screenshot = store.save_screenshot("session", 0, Image.new("RGB", (2, 2), "red"))

    assert screenshot.data_url is None
    assert (tmp_path / "session" / "step-0000.png").read_bytes().startswith(b"\x89PNG")


def test_png_bytes_normalizes_optional_encoder_failures() -> None:
    with pytest.raises(RuntimeError, match="failed to encode web screenshot"):
        _png_bytes(object())


def test_recording_artifacts_indexes_only_finalized_nonempty_files(tmp_path) -> None:
    store = WebArtifactStore(tmp_path)
    assert store.recording_artifacts("missing") == []

    video_dir = store.session_dir("session-a") / "video" / "page"
    video_dir.mkdir(parents=True)
    webm_payload = b"finalized-webm"
    binary_payload = b"opaque"
    webm = video_dir / "task.webm"
    binary = video_dir / "recording.unknown-extension"
    webm.write_bytes(webm_payload)
    binary.write_bytes(binary_payload)
    (video_dir / "active.webm").touch()
    (video_dir / "nested-directory").mkdir()

    artifacts = store.recording_artifacts("session-a")

    assert [artifact.mime_type for artifact in artifacts] == ["application/octet-stream", "video/webm"]
    assert [artifact.size_bytes for artifact in artifacts] == [len(binary_payload), len(webm_payload)]
    assert artifacts[0].sha256 == hashlib.sha256(binary_payload).hexdigest()
    assert artifacts[1].sha256 == hashlib.sha256(webm_payload).hexdigest()
