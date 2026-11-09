# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import runpy
from unittest.mock import patch

from responses_api_agents.opencode_agent.app import OpenCodeAgent
from responses_api_agents.opencode_vision_agent import app


def test_reuses_opencode_agent() -> None:
    assert app.OpenCodeAgent is OpenCodeAgent


def test_uvicorn_workers_import_a_module_level_app() -> None:
    """With num_workers > 1, each worker imports this entrypoint and serves its `app`."""
    served = object()
    with patch.object(OpenCodeAgent, "run_webserver", return_value=served) as run_webserver:
        with patch("nemo_gym.utils.is_nemo_gym_fastapi_entrypoint", return_value=True):
            worker = runpy.run_path(app.__file__)
        plain = runpy.run_path(app.__file__)  # an ordinary import starts nothing
    assert worker["app"] is served and run_webserver.call_count == 1
    assert "app" not in plain


def test_image_python_is_importable() -> None:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import scipy  # noqa: F401
