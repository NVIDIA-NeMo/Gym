# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode plugin of the video tools (sandbox_image/video_tools.mjs) under Node.

video_tools_plugin.test.mjs drives the plugin's hooks as OpenCode calls them, with a stand-in for the video_tools
program; this module stages the plugin in a temporary directory, as the image puts it in /opt/video-mcqa, runs the
script with a bounded timeout and checks its success line.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PLUGIN = HERE.parent / "sandbox_image" / "video_tools.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to run the OpenCode plugin")
def test_the_plugin_returns_frames_as_images_and_keeps_requests_under_the_media_budget(tmp_path) -> None:
    staged = tmp_path / "video-mcqa"
    staged.mkdir()
    shutil.copy(PLUGIN, staged / PLUGIN.name)
    result = subprocess.run(
        ["node", str(HERE / "video_tools_plugin.test.mjs")],
        cwd=tmp_path,
        env={**os.environ, "VIDEO_TOOLS_DIR": str(staged)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "video tools plugin: all checks passed" in result.stdout
