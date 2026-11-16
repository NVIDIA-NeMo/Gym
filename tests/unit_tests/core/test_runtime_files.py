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
import asyncio
import tempfile
from pathlib import Path

import pytest
from omegaconf import DictConfig

import nemo_gym.config.global_config as global_config
from nemo_gym.core.episode_context import rollout_context
from nemo_gym.core.runtime_files import async_scratch_dir, debug_artifact_dir, scratch_dir


@pytest.fixture
def set_global_config(monkeypatch: pytest.MonkeyPatch):
    def set_(values: dict | None) -> None:
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None if values is None else DictConfig(values))
        monkeypatch.delenv(global_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

    return set_


class TestDebugArtifactDir:
    def test_off_without_a_global_or_server_setting(self, set_global_config) -> None:
        set_global_config({"debug_artifacts_dir": None})
        assert debug_artifact_dir(server_name="swebench", kind="verification") is None

        set_global_config(None)
        assert debug_artifact_dir(server_name="swebench", kind="verification") is None

    def test_global_root_separates_server_kind_rollout_and_agent_instance(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"debug_artifacts_dir": str(tmp_path)})

        with rollout_context("4-2-a1"):
            path = debug_artifact_dir(server_name="swebench", kind="verification", label="astropy__astropy-1")

        assert path is not None
        assert path.parent == tmp_path / "swebench" / "verification" / "4-2-a1" / "unnamed"
        assert "-astropy__astropy-1-" in path.name
        # Nothing is created until the server writes into the folder.
        assert not path.exists()

    def test_outside_a_rollout_uses_a_fixed_segment(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_artifacts_dir": str(tmp_path)})

        path = debug_artifact_dir(server_name="vision2code", kind="render")

        assert path is not None
        assert path.parent == tmp_path / "vision2code" / "render" / "no-rollout" / "unnamed"

    def test_repeated_calls_in_one_rollout_get_separate_folders(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_artifacts_dir": str(tmp_path)})

        with rollout_context("0-0"):
            first = debug_artifact_dir(server_name="swebench", kind="verification", label="same")
            second = debug_artifact_dir(server_name="swebench", kind="verification", label="same")

        assert first != second
        assert first.parent == second.parent

    def test_label_with_slashes_stays_one_folder(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_artifacts_dir": str(tmp_path)})

        path = debug_artifact_dir(server_name="swebench", kind="verification", label="../../org/repo")

        assert path is not None
        assert path.parent == tmp_path / "swebench" / "verification" / "no-rollout" / "unnamed"
        assert "/" not in path.name

    def test_server_setting_overrides_the_global_root(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_artifacts_dir": str(tmp_path / "global")})

        path = debug_artifact_dir(server_name="swebench", kind="verification", override=tmp_path / "mine")

        assert path is not None
        assert path.parent == tmp_path / "mine" / "verification" / "no-rollout" / "unnamed"

    def test_server_setting_turns_output_on_without_the_global_root(self, set_global_config, tmp_path: Path) -> None:
        set_global_config(None)

        path = debug_artifact_dir(server_name="swebench", kind="verification", override=tmp_path)

        assert path is not None
        assert path.is_relative_to(tmp_path)

    def test_relative_server_setting_is_refused(self, set_global_config) -> None:
        set_global_config(None)

        with pytest.raises(ValueError, match="absolute path"):
            debug_artifact_dir(server_name="swebench", kind="verification", override=Path("logs"))

    def test_kind_must_be_one_safe_segment(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_artifacts_dir": str(tmp_path)})

        with pytest.raises(ValueError, match="kind"):
            debug_artifact_dir(server_name="swebench", kind="../escape")


class TestScratchDir:
    def test_folder_is_under_the_temporary_directory_and_removed_on_exit(self, set_global_config) -> None:
        set_global_config(None)

        with scratch_dir(kind="workspace") as path:
            (path / "file.txt").write_text("x")
            assert path.is_relative_to(tempfile.gettempdir())

        assert not path.exists()

    def test_folder_is_removed_when_the_block_raises(self, set_global_config) -> None:
        set_global_config(None)

        with pytest.raises(RuntimeError):
            with scratch_dir(kind="workspace") as path:
                (path / "file.txt").write_text("x")
                raise RuntimeError("boom")

        assert not path.exists()

    def test_shared_folder_is_under_shared_scratch_dir(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"shared_scratch_dir": str(tmp_path / "scratch")})

        with scratch_dir(kind="harbor-job", shared=True) as path:
            assert path.parent == tmp_path / "scratch"

        assert not path.exists()
        assert list((tmp_path / "scratch").iterdir()) == []

    def test_keep_as_moves_the_folder_to_debug_output(self, set_global_config, tmp_path: Path) -> None:
        set_global_config(None)
        keep_as = tmp_path / "debug" / "verification" / "leaf"

        with scratch_dir(kind="verification", keep_as=keep_as) as path:
            (path / "report.json").write_text("{}")

        assert not path.exists()
        assert (keep_as / "report.json").read_text() == "{}"

    def test_failing_to_keep_does_not_fail_the_block(self, set_global_config, tmp_path: Path) -> None:
        set_global_config(None)
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("")

        with scratch_dir(kind="verification", keep_as=blocker / "leaf") as path:
            (path / "report.json").write_text("{}")

        assert not path.exists()


class TestAsyncScratchDir:
    async def test_folder_is_removed_when_the_task_is_cancelled(self, set_global_config) -> None:
        set_global_config(None)
        created: list[Path] = []
        entered = asyncio.Event()

        async def work() -> None:
            async with async_scratch_dir(kind="workspace") as path:
                created.append(path)
                (path / "file.txt").write_text("x")
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(work())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert created and not created[0].exists()
