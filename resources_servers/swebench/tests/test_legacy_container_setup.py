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
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from resources_servers.swebench.legacy_container_setup import (
    CHROME_WRAPPER_PATH,
    GRADLE_INIT_PATH,
    LEGACY_CONTAINER_ENV,
    LEGACY_CONTAINER_SETUP_COMMAND,
    MAVEN_SETTINGS_PATH,
    apply_legacy_container_setup,
    legacy_container_files,
)


class _FakeSandbox:
    def __init__(self, return_code: int = 0, error_type: str | None = None) -> None:
        self.commands: list[str] = []
        self._result = SimpleNamespace(return_code=return_code, error_type=error_type, stderr="boom")

    async def exec(self, command: str, timeout_s=None):
        self.commands.append(command)
        return self._result


class TestLegacyContainerSetup:
    def test_env_matches_the_apptainer_harness(self) -> None:
        assert LEGACY_CONTAINER_ENV == {
            "_JAVA_OPTIONS": (
                "-Djava.net.preferIPv6Addresses=false "
                "-Drobolectric.dependency.repo.url=https://maven-central.storage-download.googleapis.com/maven2/"
            ),
            "GRADLE_USER_HOME": "/root/.gradle",
            "DOTNET_GCHeapHardLimit": "0x200000000",
            "CHROME_BIN": CHROME_WRAPPER_PATH,
            "CHROMIUM_BIN": CHROME_WRAPPER_PATH,
        }

    def test_files_ship_the_mirror_and_the_chrome_wrapper(self) -> None:
        files = legacy_container_files()
        assert set(files) == {MAVEN_SETTINGS_PATH, GRADLE_INIT_PATH, CHROME_WRAPPER_PATH}
        assert "<mirrorOf>central</mirrorOf>" in files[MAVEN_SETTINGS_PATH]
        assert "maven-central.storage-download.googleapis.com" in files[GRADLE_INIT_PATH]
        assert GRADLE_INIT_PATH.startswith(LEGACY_CONTAINER_ENV["GRADLE_USER_HOME"] + "/init.d/")
        assert "--no-sandbox --disable-dev-shm-usage" in files[CHROME_WRAPPER_PATH]

    @pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
    def test_chrome_wrapper_execs_the_first_real_binary_with_container_flags(self, tmp_path) -> None:
        chrome = tmp_path / "google-chrome"
        chrome.write_text('#!/bin/sh\necho "$@"\n')
        chrome.chmod(0o755)
        wrapper = tmp_path / "chrome-wrapper.sh"
        wrapper.write_text(
            legacy_container_files()[CHROME_WRAPPER_PATH].replace("/opt/google/chrome/google-chrome ", f"{chrome} ", 1)
        )
        wrapper.chmod(0o755)
        result = subprocess.run([str(wrapper), "--headless"], capture_output=True, text=True, env=os.environ)
        assert result.returncode == 0
        assert result.stdout.split() == ["--no-sandbox", "--disable-dev-shm-usage", "--headless"]

    async def test_setup_makes_the_wrapper_executable_and_fans_out_the_init_script(self) -> None:
        sandbox = _FakeSandbox()
        await apply_legacy_container_setup(sandbox)
        assert sandbox.commands == [LEGACY_CONTAINER_SETUP_COMMAND]
        assert f"chmod +x {CHROME_WRAPPER_PATH}" in LEGACY_CONTAINER_SETUP_COMMAND
        assert f"cp {GRADLE_INIT_PATH}" in LEGACY_CONTAINER_SETUP_COMMAND

    @pytest.mark.parametrize(("return_code", "error_type"), [(1, None), (-1, "timeout")])
    async def test_a_failed_setup_raises_so_the_sandbox_is_stopped(self, return_code, error_type) -> None:
        with pytest.raises(RuntimeError, match="legacy container setup failed"):
            await apply_legacy_container_setup(_FakeSandbox(return_code, error_type))
