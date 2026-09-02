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
from typing import Any

import pytest

from nemo_gym.sandbox.providers.apptainer import build as apptainer_build
from responses_api_agents.anyterminal_agent import prepare


def _row(task_name: str, docker_image: str) -> dict:
    return {"responses_create_params": {"metadata": {"task_name": task_name, "docker_image": docker_image}}}


class FakeProc:
    def __init__(self, returncode: int, stderr: str = "") -> None:
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr.encode()

    def kill(self) -> None:  # pragma: no cover - not reached in these tests
        pass

    async def wait(self) -> int:  # pragma: no cover - not reached in these tests
        return self.returncode


@pytest.fixture
def fake_apptainer(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Pretend apptainer is installed and record the build command lines."""
    calls: list[list[str]] = []

    async def _exec(*argv: str, **_kwargs: Any) -> FakeProc:
        calls.append(list(argv))
        if "fails" in argv[4]:
            return FakeProc(1, "manifest unknown")
        Path(argv[3]).write_text("sif")
        return FakeProc(0)

    monkeypatch.setattr(apptainer_build.asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(apptainer_build, "_require_apptainer", lambda *a, **k: "/usr/bin/apptainer")
    monkeypatch.setattr(apptainer_build, "_apptainer_subprocess_env", lambda *a, **k: {})
    return calls


def test_build_images_writes_one_sif_per_task_name(
    fake_apptainer: list[list[str]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [_row("alpha", "ghcr.io/acme/alpha:1"), _row("beta", "ghcr.io/acme/beta:1")]

    prepare.build_images(rows, tmp_path, jobs=2, force=False)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["alpha.sif", "beta.sif"]
    assert [c[-1] for c in fake_apptainer] == ["docker://ghcr.io/acme/alpha:1", "docker://ghcr.io/acme/beta:1"]
    out = capsys.readouterr().out
    # The container_formatter hint is how a user wires these into the agent config.
    assert f"container_formatter='{tmp_path}/{{task_name}}.sif'" in out


def test_build_images_gives_tasks_sharing_an_image_their_own_sif(
    fake_apptainer: list[list[str]], tmp_path: Path
) -> None:
    rows = [_row("alpha", "ghcr.io/acme/shared:1"), _row("beta", "ghcr.io/acme/shared:1")]

    prepare.build_images(rows, tmp_path, jobs=1, force=False)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["alpha.sif", "beta.sif"]


def test_build_images_reports_failures_and_exits_nonzero(
    fake_apptainer: list[list[str]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [_row("good", "ghcr.io/acme/good:1"), _row("bad", "ghcr.io/acme/fails:1")]

    with pytest.raises(SystemExit) as excinfo:
        prepare.build_images(rows, tmp_path, jobs=1, force=False)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "[FAIL] bad" in out
    assert "[ok] good" in out
    # One bad image must not discard the batch.
    assert (tmp_path / "good.sif").is_file()


def test_build_images_skips_existing_unless_forced(fake_apptainer: list[list[str]], tmp_path: Path) -> None:
    (tmp_path / "alpha.sif").write_text("known-good")
    rows = [_row("alpha", "ghcr.io/acme/alpha:1")]

    prepare.build_images(rows, tmp_path, jobs=1, force=False)
    assert fake_apptainer == []
    assert (tmp_path / "alpha.sif").read_text() == "known-good"

    prepare.build_images(rows, tmp_path, jobs=1, force=True)
    assert len(fake_apptainer) == 1
    assert (tmp_path / "alpha.sif").read_text() == "sif"


def test_build_images_exits_cleanly_when_apptainer_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _missing(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("The 'apptainer' binary is required for the apptainer sandbox provider.")

    monkeypatch.setattr(
        "nemo_gym.sandbox.providers.apptainer.provider._require_apptainer",
        _missing,
    )

    with pytest.raises(SystemExit) as excinfo:
        prepare.build_images([_row("alpha", "ghcr.io/acme/alpha:1")], tmp_path, jobs=1, force=False)

    assert "Omit --build-image" in str(excinfo.value.code)
