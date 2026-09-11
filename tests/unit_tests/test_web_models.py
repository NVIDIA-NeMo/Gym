# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.web.models import WebArtifactRef, WebBenchmark, WebImage, WebRuntimeProfile, WebTask


def test_task_ids_are_normalized_to_strings() -> None:
    task = WebTask(benchmark=WebBenchmark.WEBARENA, task_id=42)
    assert task.task_id == "42"


def test_task_id_is_required() -> None:
    with pytest.raises(ValidationError, match="task_id is required"):
        WebTask(benchmark=WebBenchmark.WEBARENA, task_id=None)


def test_removed_runtime_profiles_are_rejected_at_the_wire_boundary() -> None:
    with pytest.raises(ValidationError, match="visual_browser"):
        WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="1",
            runtime_profile="browsergym",
        )


def test_visual_browser_runtime_is_backend_neutral() -> None:
    task = WebTask(
        benchmark=WebBenchmark.VISUALWEBARENA,
        task_id="17",
        runtime_profile=WebRuntimeProfile.VISUAL_BROWSER,
    )

    assert task.runtime_profile == WebRuntimeProfile.VISUAL_BROWSER


def test_image_requires_inline_or_artifact_transport() -> None:
    with pytest.raises(ValidationError, match="requires data_url or artifact"):
        WebImage()


def test_image_accepts_inline_or_artifact_transport() -> None:
    assert WebImage(data_url="data:image/png;base64,AA==").data_url is not None
    artifact = WebArtifactRef(uri="file:///tmp/a.png", mime_type="image/png", size_bytes=1, sha256="0" * 64)
    assert WebImage(artifact=artifact).artifact == artifact
