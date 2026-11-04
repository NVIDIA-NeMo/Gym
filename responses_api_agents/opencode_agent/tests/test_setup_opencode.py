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

import subprocess
from unittest.mock import patch

import pytest

from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode, installed_opencode_version


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "expected"),
    [
        ("1.17.11\n", "", 0, "1.17.11"),
        ("", "1.17.11\n", 0, "1.17.11"),
        ("startup notice\n1.17.11\n", "", 0, "1.17.11"),
        ("", "", 0, None),
        (" \n\t", "", 0, None),
        ("1.17.11", "", 1, None),
        ("1.17.10", "", 0, "1.17.10"),
    ],
)
def test_installed_version_handles_unavailable_output(stdout, stderr, returncode, expected) -> None:
    result = subprocess.CompletedProcess(["opencode", "--version"], returncode, stdout, stderr)
    with (
        patch("responses_api_agents.opencode_agent.setup_opencode.shutil.which", return_value="/bin/opencode"),
        patch("responses_api_agents.opencode_agent.setup_opencode.subprocess.run", return_value=result),
    ):
        assert installed_opencode_version() == expected
        if expected == "1.17.11":
            ensure_opencode("1.17.11")
        else:
            with pytest.raises(RuntimeError, match="Expected OpenCode 1.17.11"):
                ensure_opencode("1.17.11")


def test_installed_version_timeout_is_unavailable() -> None:
    with (
        patch("responses_api_agents.opencode_agent.setup_opencode.shutil.which", return_value="/bin/opencode"),
        patch(
            "responses_api_agents.opencode_agent.setup_opencode.subprocess.run",
            side_effect=subprocess.TimeoutExpired("opencode --version", 60),
        ),
    ):
        assert installed_opencode_version() is None


def test_opencode_install_uses_shared_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.harness_conformance.ci import install_runtime

    from responses_api_agents.opencode_agent import setup_opencode
    from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
    from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION
    from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgentConfig

    versions = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup_opencode, "_npm_install", lambda npm, version: versions.append(version))
    install_runtime("opencode")
    assert versions == [OPENCODE_VERSION]
    assert OpenCodeAgentConfig.model_fields["opencode_version"].default == OPENCODE_VERSION
    assert OpenCodeSandboxedAgentConfig.model_fields["opencode_version"].default == OPENCODE_VERSION
