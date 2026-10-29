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

from nemo_gym.data import BUILT_IMAGES_FILE, HarborTaskError, load_task
from nemo_gym.data.tasks.harbor.build_plan import BuildResult
from nemo_gym.data.tasks.harbor.build_records import record_built_image
from nemo_gym.data.tasks.harbor.digest import context_digest
from nemo_gym.data.tasks.harbor.records import IMAGE_CONFIGS_FILE, read_built_images, read_image_configs
from nemo_gym.data.tasks.harbor.task import built_image, built_service_images
from tests.unit_tests.harbor_build_helpers import REPO, UBUNTU_RECORD, write_task


class TestBuildRecords:
    """``built-images.json``: one entry per context, one image per built service, readable by older readers."""

    def test_a_legacy_record_is_main_alone_and_a_services_record_without_main_has_no_built_image(self, tmp_path):
        from nemo_gym.data.tasks.harbor.build_records import recorded_image_mismatch

        folder = tmp_path / "ds"
        write_task(folder / "a")
        digest = context_digest(folder / "a" / "environment")
        legacy = {"image": "reg/hb__x@sha256:1", "tasks": ["a"], "tool": "docker", "pushed": True}
        configs = {"reg/hb__x@sha256:1": UBUNTU_RECORD, "reg/hb__x-world@sha256:2": UBUNTU_RECORD}
        (folder / IMAGE_CONFIGS_FILE).write_text(json.dumps(configs))
        (folder / BUILT_IMAGES_FILE).write_text(json.dumps({digest: legacy}))
        # A record written before sidecars were built reads as main alone, and the task loads from it.
        assert built_service_images(folder / "a") == {"main": "reg/hb__x@sha256:1"}
        assert built_image(folder / "a") == "reg/hb__x@sha256:1" == load_task(folder / "a").image
        assert recorded_image_mismatch(legacy, configs, repository="reg", platform="linux/amd64") is None
        assert recorded_image_mismatch(legacy, configs, repository="reg", platform="linux/amd64", service="world") == (
            "no image is recorded for service world"
        )
        # A services record without main: the sidecar is recorded, main is not built, and the loader says so.
        sidecar_only = {
            "tasks": ["a"],
            "tool": "docker",
            "pushed": True,
            "services": {"world": {"image": "reg/hb__x-world@sha256:2"}},
        }
        (folder / BUILT_IMAGES_FILE).write_text(json.dumps({digest: sidecar_only}))
        assert built_service_images(folder / "a") == {"world": "reg/hb__x-world@sha256:2"}
        assert built_image(folder / "a") is None
        with pytest.raises(HarborTaskError, match="needs a build"):
            load_task(folder / "a")
        assert (
            recorded_image_mismatch(sidecar_only, configs, repository="reg", platform="linux/amd64", service="world")
            is None
        )
        assert recorded_image_mismatch(
            sidecar_only, configs, repository="reg", platform="linux/arm64", service="world"
        ) == ("it is linux/amd64, not linux/arm64")
        assert built_service_images(tmp_path / "nowhere") == {}

    def test_record_built_image_merges_services_and_keeps_main_at_the_top_level(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        digest = "d" * 32
        world = BuildResult(digest, ("a",), "t-world", "t-world@sha256:w", "docker", False, {"os": "linux"}, "world")
        main = BuildResult(digest, ("a", "b"), "t", "t@sha256:m", "docker", False, {"os": "linux"})
        record_built_image(folder, world)
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[digest]
        # No main yet: no top-level image, so an older reader sees no built main rather than the sidecar.
        assert entry == {
            "tasks": ["a"],
            "tool": "docker",
            "pushed": False,
            "services": {"world": {"image": "t-world@sha256:w", "tool": "docker", "pushed": False}},
        }
        record_built_image(folder, main)
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[digest]
        assert entry["image"] == "t@sha256:m" and entry["tasks"] == ["a", "b"]
        assert entry["services"] == {
            "world": {"image": "t-world@sha256:w", "tool": "docker", "pushed": False},
            "main": {"image": "t@sha256:m", "tool": "docker", "pushed": False},
        }
        # Pinned per service: a second main is ignored, a rebuilt world replaces only world.
        record_built_image(folder, BuildResult(digest, ("a",), "t2", "t2@sha256:n", "docker", False, {}))
        again = BuildResult(digest, ("a",), "t-world", "t-world@sha256:w2", "podman", True, {"os": "linux"}, "world")
        record_built_image(folder, again, overwrite=True)
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[digest]
        assert entry["services"] == {
            "world": {"image": "t-world@sha256:w2", "tool": "podman", "pushed": True},
            "main": {"image": "t@sha256:m", "tool": "docker", "pushed": False},
        }
        # The top-level pair is main's, never the last build's: a pushed world does not vouch for a local main.
        assert entry["image"] == "t@sha256:m" and entry["tool"] == "docker" and entry["pushed"] is False
        assert {"t-world@sha256:w", "t-world@sha256:w2", "t@sha256:m"} <= set(
            json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        )

    def test_each_service_is_judged_by_its_own_pushed_flag(self, tmp_path):
        from nemo_gym.data.tasks.harbor.build_records import recorded_image_mismatch

        folder = tmp_path / "ds"
        folder.mkdir()
        digest = "d" * 32
        pushed = f"{REPO}/hb__{digest}@sha256:m1"
        record_built_image(folder, BuildResult(digest, ("a",), f"{REPO}/hb__{digest}", pushed, "buildah", True, {}))
        # A local docker run then builds the sidecar: its record is local while main's stays pushed.
        local = f"hb__{digest}-world"
        record_built_image(
            folder, BuildResult(digest, ("a",), local, local, "docker", False, {}, "world"), overwrite=True
        )
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[digest]
        configs = {pushed: UBUNTU_RECORD, local: UBUNTU_RECORD}
        assert entry["pushed"] is True and entry["tool"] == "buildah"
        assert entry["services"]["world"] == {"image": local, "tool": "docker", "pushed": False}
        assert recorded_image_mismatch(entry, configs, repository=REPO, platform="linux/amd64") is None
        assert recorded_image_mismatch(entry, configs, repository=REPO, platform="linux/amd64", service="world") == (
            "it was built locally and this run pushes to a repository"
        )
        assert (
            recorded_image_mismatch(entry, configs, repository=None, platform="linux/amd64", service="world") is None
        )
        # A record without per-service flags (written before they were kept) reads the entry's for every service.
        old = {"tasks": ["a"], "tool": "podman", "pushed": False, "services": {"world": {"image": local}}}
        assert recorded_image_mismatch(old, configs, repository=REPO, platform="linux/amd64", service="world") == (
            "it was built locally and this run pushes to a repository"
        )
        assert recorded_image_mismatch(old, configs, repository=None, platform="linux/amd64", service="world") is None
        assert recorded_image_mismatch(old, configs, repository=None, platform="linux/amd64") == (
            "no image is recorded for service main"
        )


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
                "nemo_gym.data.tasks.harbor.records.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk"))
            )
            with pytest.raises(OSError, match="disk"):
                record_built_image(folder, later)
            assert (folder / BUILT_IMAGES_FILE).read_text() == before
            assert not [p for p in folder.iterdir() if p.name.endswith(".tmp")]
        finally:
            os.umask(previous)
