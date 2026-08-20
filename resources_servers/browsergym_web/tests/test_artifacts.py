# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

from resources_servers.browsergym_web.artifacts import WebArtifactStore


def test_recording_artifacts_indexes_finalized_nonempty_files(tmp_path):
    store = WebArtifactStore(tmp_path)
    video_dir = store.session_dir("session-a") / "video" / "page"
    video_dir.mkdir(parents=True)
    payload = b"finalized-webm"
    recording = video_dir / "task.webm"
    recording.write_bytes(payload)
    (video_dir / "active.webm").touch()

    artifacts = store.recording_artifacts("session-a")

    assert len(artifacts) == 1
    assert artifacts[0].uri == recording.resolve().as_uri()
    assert artifacts[0].mime_type == "video/webm"
    assert artifacts[0].size_bytes == len(payload)
    assert artifacts[0].sha256 == hashlib.sha256(payload).hexdigest()


def test_recording_artifacts_is_empty_when_video_is_disabled(tmp_path):
    store = WebArtifactStore(tmp_path)

    assert store.recording_artifacts("session-without-video") == []
