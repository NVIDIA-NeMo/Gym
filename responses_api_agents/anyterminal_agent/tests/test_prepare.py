# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from responses_api_agents.anyterminal_agent.prepare import IMAGE_BUILD_ATTEMPTS, _build_one_sif


def _staged_path(command: list[str]) -> Path:
    return Path(command[3])


def test_build_retries_then_atomically_installs_image(tmp_path: Path) -> None:
    attempts = 0

    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        output = _staged_path(command)
        output.write_text("partial" if attempts == 1 else "complete")
        return SimpleNamespace(returncode=1 if attempts == 1 else 0, stderr="truncated manifest", stdout="")

    with (
        patch("responses_api_agents.anyterminal_agent.prepare.subprocess.run", side_effect=run),
        patch("responses_api_agents.anyterminal_agent.prepare.time.sleep") as sleep,
    ):
        name, ok, detail = _build_one_sif("task", "example/image:tag", tmp_path, force=False)

    assert (name, ok, detail) == ("task", True, "built after 2 attempts")
    assert attempts == 2
    assert (tmp_path / "task.sif").read_text() == "complete"
    assert list(tmp_path.iterdir()) == [tmp_path / "task.sif"]
    sleep.assert_called_once_with(2)


def test_build_removes_intermediate_output_after_three_failures(tmp_path: Path) -> None:
    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        _staged_path(command).write_text("partial")
        return SimpleNamespace(returncode=1, stderr="unexpected end of JSON input", stdout="")

    with (
        patch("responses_api_agents.anyterminal_agent.prepare.subprocess.run", side_effect=run) as run_mock,
        patch("responses_api_agents.anyterminal_agent.prepare.time.sleep"),
    ):
        name, ok, detail = _build_one_sif("task", "example/image:tag", tmp_path, force=False)

    assert (name, ok) == ("task", False)
    assert run_mock.call_count == IMAGE_BUILD_ATTEMPTS
    assert f"attempt {IMAGE_BUILD_ATTEMPTS}/{IMAGE_BUILD_ATTEMPTS}" in detail
    assert "unexpected end of JSON input" in detail
    assert list(tmp_path.iterdir()) == []


def test_failed_forced_rebuild_preserves_existing_image(tmp_path: Path) -> None:
    image = tmp_path / "task.sif"
    image.write_text("known-good")

    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        _staged_path(command).write_text("partial")
        return SimpleNamespace(returncode=1, stderr="registry unavailable", stdout="")

    with (
        patch("responses_api_agents.anyterminal_agent.prepare.subprocess.run", side_effect=run),
        patch("responses_api_agents.anyterminal_agent.prepare.time.sleep"),
    ):
        _name, ok, _detail = _build_one_sif("task", "example/image:tag", tmp_path, force=True)

    assert not ok
    assert image.read_text() == "known-good"
    assert list(tmp_path.iterdir()) == [image]
