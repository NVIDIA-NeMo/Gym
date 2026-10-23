# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the build records.

The reuse decisions over recorded images and concurrent recording into one datasets folder.
test_harbor_build_inputs.py stays a separate file beside it.
"""

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from nemo_gym.tasks.harbor.build_plan import BuildResult
from nemo_gym.tasks.harbor.build_records import record_built_image
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE, read_built_images, read_image_configs


def _d2_writer(folder: str, wid: int, count: int, barrier, errors) -> None:
    """One recording process: ``count`` distinct contexts through the real record_built_image, released by the
    barrier together with the other writers so their read-modify-writes overlap."""
    barrier.wait()
    for i in range(count):
        digest = f"{wid:02d}{i:04d}".ljust(32, "0")
        result = BuildResult(
            digest=digest,
            tasks=(f"task-{wid}-{i}",),
            tag=f"hb__{digest}",
            image=f"registry.example/x/hb__{digest}",
            tool="buildah",
            pushed=True,
            record={"config": {"Env": [], "Cmd": ["bash"]}, "architecture": "arm64", "os": "linux"},
        )
        try:
            record_built_image(Path(folder), result, overwrite=True)
        except Exception as exc:  # noqa: BLE001 - every failure is the finding
            errors.append(f"{type(exc).__name__}: {exc}")


def _d2_reader(folder: str, barrier, stop, counts) -> None:
    """Reads both record files in a loop while the writers run, the way the server reads them at seed: a raw
    ``json.loads`` with no retry (``environment_servers/harbor/app.py`` ``_compose_image_configs``), so a torn
    file is counted rather than healed by ``read_json_record``'s second read. The retrying readers run once per
    loop as well; one that still fails after its retry counts too."""
    barrier.wait()
    reads = torn = 0
    while not stop.is_set():
        reads += 1
        for name in (BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE):
            try:
                json.loads((Path(folder) / name).read_text())
            except FileNotFoundError:
                pass
            except (ValueError, UnicodeDecodeError):
                torn += 1
        for read in (read_built_images, read_image_configs):
            try:
                read(Path(folder))
            except HarborTaskError:
                torn += 1
    counts["reads"] = reads
    counts["torn"] = torn


class TestConcurrentRecording:
    """D2: record files are merged and replaced under a sidecar lock, atomically."""

    WRITERS = 4
    PER_WRITER = 30

    def test_four_writers_lose_nothing_and_a_reader_sees_no_torn_file(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        ctx = multiprocessing.get_context("fork")
        with ctx.Manager() as manager:
            errors = manager.list()
            counts = manager.dict()
            barrier = ctx.Barrier(self.WRITERS + 1)
            stop = ctx.Event()
            reader = ctx.Process(target=_d2_reader, args=(str(folder), barrier, stop, counts))
            writers = [
                ctx.Process(target=_d2_writer, args=(str(folder), wid, self.PER_WRITER, barrier, errors))
                for wid in range(self.WRITERS)
            ]
            reader.start()
            for process in writers:
                process.start()
            try:
                for process in writers:
                    process.join(timeout=120)
                stop.set()
                reader.join(timeout=30)
                assert all(process.exitcode == 0 for process in writers), [p.exitcode for p in writers]
                assert reader.exitcode == 0
                assert list(errors) == []
                assert counts["reads"] > 0 and counts["torn"] == 0, dict(counts)
            finally:
                for process in [reader, *writers]:
                    if process.is_alive():
                        process.terminate()
        expected = self.WRITERS * self.PER_WRITER
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert len(built) == expected and len(configs) == expected
        assert {entry["image"] for entry in built.values()} == set(configs)
        # The locks sit next to the records; no temporary file is left behind.
        assert (folder / f"{BUILT_IMAGES_FILE}.lock").is_file() and (folder / f"{IMAGE_CONFIGS_FILE}.lock").is_file()
        assert sorted(p.name for p in folder.iterdir()) == sorted(
            [BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE, f"{BUILT_IMAGES_FILE}.lock", f"{IMAGE_CONFIGS_FILE}.lock"]
        )

    def test_record_files_are_written_atomically_with_the_umask_applied(self, tmp_path, monkeypatch):
        folder = tmp_path / "ds"
        folder.mkdir()
        previous = os.umask(0o022)
        try:
            result = BuildResult(
                digest="e" * 32,
                tasks=("a",),
                tag="t:1",
                image="t:1",
                tool="docker",
                pushed=False,
                record={"os": "linux"},
            )
            record_built_image(folder, result)
            assert (folder / BUILT_IMAGES_FILE).stat().st_mode & 0o777 == 0o644
            assert (folder / IMAGE_CONFIGS_FILE).stat().st_mode & 0o777 == 0o644
            # A replace that fails leaves the previous record in place and no temporary file next to it.
            before = (folder / BUILT_IMAGES_FILE).read_text()
            later = BuildResult(digest="f" * 32, tasks=("b",), tag="t:2", image="t:2", tool="docker", pushed=False)
            monkeypatch.setattr(
                "nemo_gym.tasks.harbor.records.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk"))
            )
            with pytest.raises(OSError, match="disk"):
                record_built_image(folder, later)
            assert (folder / BUILT_IMAGES_FILE).read_text() == before
            assert not [p for p in folder.iterdir() if p.name.endswith(".tmp")]
        finally:
            os.umask(previous)
