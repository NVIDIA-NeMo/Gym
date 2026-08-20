# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.web.models import WebBenchmark, WebImage, WebRuntimeProfile, WebTask


def test_task_ids_are_normalized_to_strings() -> None:
    task = WebTask(benchmark=WebBenchmark.WEBARENA, task_id=42)
    assert task.task_id == "42"


def test_selenium_is_scoped_to_webvoyager() -> None:
    with pytest.raises(ValidationError, match="only defined for WebVoyager"):
        WebTask(
            benchmark=WebBenchmark.WEBARENA,
            task_id="1",
            runtime_profile=WebRuntimeProfile.SELENIUM,
        )


def test_image_requires_inline_or_artifact_transport() -> None:
    with pytest.raises(ValidationError, match="requires data_url or artifact"):
        WebImage()
