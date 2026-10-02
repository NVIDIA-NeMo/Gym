# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from responses_api_agents.opencode_agent.app import OpenCodeAgent
from responses_api_agents.opencode_vision_agent import app


def test_reuses_opencode_agent() -> None:
    assert app.OpenCodeAgent is OpenCodeAgent


def test_image_python_is_importable() -> None:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import scipy  # noqa: F401
