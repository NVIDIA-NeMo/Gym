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
import os
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.providers.apptainer import build as apptainer_build
from nemo_gym.sandbox.providers.apptainer.build import ApptainerImageBuildError, build_sif, build_sifs, is_usable_sif


BINARY = "/usr/bin/apptainer"
ENV: dict[str, str] = {}


class FakeProc:
    def __init__(self, returncode: int = 0, stderr: str = "", hang: bool = False) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.hang = hang
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.hang:
            await asyncio.Event().wait()
        return b"", self.stderr.encode()

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def patch_exec(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[list[str]]:
    """Route create_subprocess_exec to ``handler(argv) -> FakeProc``, recording argv."""
    calls: list[list[str]] = []

    async def _exec(*argv: str, **_kwargs: Any) -> FakeProc:
        calls.append(list(argv))
        return handler(list(argv))

    monkeypatch.setattr(apptainer_build.asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(apptainer_build, "_require_apptainer", lambda *a, **k: BINARY)
    monkeypatch.setattr(apptainer_build, "_apptainer_subprocess_env", lambda *a, **k: ENV)
    return calls


def succeed(argv: list[str]) -> FakeProc:
    Path(argv[3]).write_text("complete")
    return FakeProc()


def fail(stderr: str = "manifest unknown", *, leave_partial: bool = True) -> Any:
    def _handler(argv: list[str]) -> FakeProc:
        if leave_partial:
            Path(argv[3]).write_text("partial")
        return FakeProc(1, stderr)

    return _handler


def test_is_usable_sif_rejects_empty_and_non_regular(tmp_path: Path) -> None:
    (tmp_path / "good.sif").write_text("payload")
    (tmp_path / "empty.sif").write_bytes(b"")
    (tmp_path / "dir.sif").mkdir()

    assert is_usable_sif(tmp_path / "good.sif") is True
    assert is_usable_sif(tmp_path / "empty.sif") is False
    assert is_usable_sif(tmp_path / "dir.sif") is False
    assert is_usable_sif(tmp_path / "missing.sif") is False


def test_build_sif_skips_a_usable_existing_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "task.sif"
    target.write_text("known-good")
    calls = patch_exec(monkeypatch, succeed)

    assert asyncio.run(build_sif("example/image:tag", target, binary=BINARY, subprocess_env=ENV)) == target
    assert calls == []
    assert target.read_text() == "known-good"


def test_build_sif_rebuilds_a_truncated_existing_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "task.sif"
    target.write_bytes(b"")  # interrupted earlier build
    calls = patch_exec(monkeypatch, succeed)

    asyncio.run(build_sif("example/image:tag", target, binary=BINARY, subprocess_env=ENV))

    assert len(calls) == 1
    assert target.read_text() == "complete"


def test_build_sif_retries_then_installs_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "task.sif"
    attempts = 0

    def flaky(argv: list[str]) -> FakeProc:
        nonlocal attempts
        attempts += 1
        return fail("truncated manifest")(argv) if attempts == 1 else succeed(argv)

    calls = patch_exec(monkeypatch, flaky)

    asyncio.run(build_sif("example/image:tag", target, binary=BINARY, subprocess_env=ENV, retry_delay_s=0))

    assert len(calls) == 2
    assert calls[0][:3] == [BINARY, "build", "--force"]
    assert calls[0][-1] == "docker://example/image:tag"
    assert Path(calls[0][3]).parent != target.parent  # staged, not built in place
    assert target.read_text() == "complete"
    assert list(tmp_path.iterdir()) == [target]  # no temp dirs survive


def test_build_sif_reports_every_attempt_and_leaves_no_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_exec(monkeypatch, fail("unexpected end of JSON input"))

    with pytest.raises(ApptainerImageBuildError) as excinfo:
        asyncio.run(
            build_sif("example/image:tag", tmp_path / "task.sif", binary=BINARY, subprocess_env=ENV, retry_delay_s=0)
        )

    assert "attempt 3/3" in str(excinfo.value)
    assert "unexpected end of JSON input" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


def test_build_sif_failed_forced_rebuild_preserves_existing_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "task.sif"
    target.write_text("known-good")
    patch_exec(monkeypatch, fail("registry unavailable"))

    with pytest.raises(ApptainerImageBuildError):
        asyncio.run(
            build_sif(
                "example/image:tag", target, binary=BINARY, subprocess_env=ENV, retry_delay_s=0, skip_existing=False
            )
        )

    assert target.read_text() == "known-good"
    assert list(tmp_path.iterdir()) == [target]


def test_build_sif_times_out_and_kills_the_stalled_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    procs: list[FakeProc] = []

    def hang(_argv: list[str]) -> FakeProc:
        procs.append(FakeProc(hang=True))
        return procs[-1]

    patch_exec(monkeypatch, hang)

    with pytest.raises(ApptainerImageBuildError, match="timed out after 0.05s"):
        asyncio.run(
            build_sif(
                "example/image:tag",
                tmp_path / "task.sif",
                binary=BINARY,
                subprocess_env=ENV,
                attempts=1,
                build_timeout_s=0.05,
            )
        )

    assert procs[0].killed is True
    assert list(tmp_path.iterdir()) == []


def test_build_sif_surfaces_filesystem_errors_as_build_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unwritable image dir must not escape past the caller's error handling."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    os.chmod(image_dir, 0o500)
    patch_exec(monkeypatch, succeed)
    try:
        with pytest.raises(ApptainerImageBuildError, match="Permission denied"):
            asyncio.run(
                build_sif(
                    "example/image:tag",
                    image_dir / "task.sif",
                    binary=BINARY,
                    subprocess_env=ENV,
                    attempts=2,
                    retry_delay_s=0,
                )
            )
    finally:
        os.chmod(image_dir, 0o700)


def test_build_sifs_rejects_names_that_escape_the_image_dir(tmp_path: Path) -> None:
    for bad in ("../escape", "a/b", "..", " "):
        with pytest.raises(ValueError, match="single path component"):
            asyncio.run(build_sifs({bad: "example/image:tag"}, tmp_path))


def test_build_sifs_continue_on_error_keeps_the_rest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_exec(monkeypatch, lambda argv: FakeProc(1, "manifest unknown") if "bad" in argv[4] else succeed(argv))

    built = asyncio.run(
        build_sifs(
            {"good-a": "example/good-a:1", "bad": "example/bad:1", "good-b": "example/good-b:1"},
            tmp_path,
            continue_on_error=True,
            attempts=1,
        )
    )

    assert set(built) == {"good-a", "good-b"}
    assert built["good-a"] == tmp_path / "good-a.sif"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["good-a.sif", "good-b.sif"]


def test_build_sifs_raises_without_continue_on_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_exec(monkeypatch, lambda argv: FakeProc(1, "manifest unknown") if "bad" in argv[4] else succeed(argv))

    with pytest.raises(ApptainerImageBuildError):
        asyncio.run(build_sifs({"good": "example/good:1", "bad": "example/bad:1"}, tmp_path, attempts=1))


def test_build_sifs_respects_the_concurrency_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    live = 0
    peak = 0

    async def _exec(*argv: str, **_kwargs: Any) -> FakeProc:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return succeed(list(argv))

    monkeypatch.setattr(apptainer_build.asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(apptainer_build, "_require_apptainer", lambda *a, **k: BINARY)
    monkeypatch.setattr(apptainer_build, "_apptainer_subprocess_env", lambda *a, **k: ENV)

    asyncio.run(build_sifs({f"task-{i}": f"example/image:{i}" for i in range(8)}, tmp_path, concurrency=2, attempts=1))

    assert peak <= 2


def test_build_sif_warns_but_succeeds_when_staging_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "task.sif"
    patch_exec(monkeypatch, succeed)

    def busy(*_a: Any, **_k: Any) -> None:
        raise OSError(16, "Device or resource busy")

    monkeypatch.setattr(apptainer_build.shutil, "rmtree", busy)

    with caplog.at_level("WARNING", logger=apptainer_build.LOGGER.name):
        result = asyncio.run(build_sif("example/image:tag", target, binary=BINARY, subprocess_env=ENV))

    # A cleanup failure must not turn a good build into a reported failure.
    assert result == target
    assert target.read_text() == "complete"
    assert "failed to clean staging dir" in caplog.text


def test_build_sifs_rejects_a_concurrency_that_would_deadlock(tmp_path: Path) -> None:
    # Semaphore(0) blocks forever, so this must fail loudly rather than hang.
    for bad in (0, -1):
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            asyncio.run(build_sifs({"a": "example/image:tag"}, tmp_path, concurrency=bad))


def test_build_sif_rejects_a_zero_exit_that_produced_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # apptainer can exit 0 without writing a usable image, which must not be
    # installed as if the build had worked.
    patch_exec(monkeypatch, lambda argv: FakeProc(0))

    with pytest.raises(ApptainerImageBuildError, match="without producing a usable"):
        asyncio.run(
            build_sif(
                "example/image:tag",
                tmp_path / "task.sif",
                binary=BINARY,
                subprocess_env=ENV,
                attempts=1,
                retry_delay_s=0,
            )
        )

    assert list(tmp_path.iterdir()) == []
