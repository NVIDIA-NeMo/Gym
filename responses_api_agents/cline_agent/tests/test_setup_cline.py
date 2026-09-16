# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from responses_api_agents.cline_agent.setup_cline import _cline_version, ensure_cline


def test_cline_version_parses_cli_output() -> None:
    result = SimpleNamespace(returncode=0, stdout="Cline CLI 3.0.55\n", stderr="")
    with patch("responses_api_agents.cline_agent.setup_cline.subprocess.run", return_value=result):
        assert _cline_version("/usr/bin/cline") == "3.0.55"


def test_exact_existing_version_is_reused() -> None:
    with (
        patch("responses_api_agents.cline_agent.setup_cline.shutil.which", return_value="/usr/bin/cline"),
        patch("responses_api_agents.cline_agent.setup_cline._cline_version", return_value="3.0.55"),
        patch("responses_api_agents.cline_agent.setup_cline._npm_install") as npm_install,
    ):
        ensure_cline("3.0.55")
    npm_install.assert_not_called()


def test_stale_existing_version_is_replaced(tmp_path: Path) -> None:
    installed = False

    def which(command: str) -> str | None:
        if command == "npm":
            return "/usr/bin/npm"
        if command == "cline":
            return "/opt/new/bin/cline" if installed else "/usr/bin/cline"
        return None

    def install(_npm: str, version: str | None) -> None:
        nonlocal installed
        assert version == "3.0.55"
        installed = True

    def version(binary: str) -> str:
        return "3.0.55" if binary == "/opt/new/bin/cline" else "3.0.13"

    with (
        patch("responses_api_agents.cline_agent.setup_cline.Path.home", return_value=tmp_path),
        patch("responses_api_agents.cline_agent.setup_cline.shutil.which", side_effect=which),
        patch("responses_api_agents.cline_agent.setup_cline._cline_version", side_effect=version),
        patch("responses_api_agents.cline_agent.setup_cline._npm_install", side_effect=install) as npm_install,
        patch("responses_api_agents.cline_agent.setup_cline._npm_global_bin", return_value=None),
    ):
        ensure_cline("3.0.55")
    npm_install.assert_called_once_with("/usr/bin/npm", "3.0.55")


def test_install_that_does_not_satisfy_pin_fails(tmp_path: Path) -> None:
    def which(command: str) -> str | None:
        return {"cline": "/usr/bin/cline", "npm": "/usr/bin/npm"}.get(command)

    with (
        patch("responses_api_agents.cline_agent.setup_cline.Path.home", return_value=tmp_path),
        patch("responses_api_agents.cline_agent.setup_cline.shutil.which", side_effect=which),
        patch("responses_api_agents.cline_agent.setup_cline._cline_version", return_value="3.0.13"),
        patch("responses_api_agents.cline_agent.setup_cline._npm_install"),
        patch("responses_api_agents.cline_agent.setup_cline._npm_global_bin", return_value=None),
        pytest.raises(RuntimeError, match="expected '3.0.55'"),
    ):
        ensure_cline("3.0.55")
