# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise parallel preparation through Harbor's real planner and record writer."""

import threading
import time
from copy import deepcopy

import pytest

from environments.unified_video.prepare_images import build_task_images
from nemo_gym.data.tasks.harbor.build_plan import BuildError
from nemo_gym.data.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.data.tasks.harbor.records import read_built_images, read_image_configs
from nemo_gym.data.tasks.harbor.task import load_task
from tests.unit_tests.data.tasks.harbor.harbor_build_helpers import DOCKER_INSPECT, FakeTool, write_task


def test_parallel_builds_preserve_all_records_deduplicate_and_resume(tmp_path, monkeypatch):
    for index in range(8):
        write_task(tmp_path / f"task-{index}", extra={"tools/setup.sh": f"echo {index}\n"})
    write_task(tmp_path / "duplicate", extra={"tools/setup.sh": "echo 0\n"})
    lock = threading.Lock()
    active = peak = builds = 0

    def factory(**kwargs):
        inspection = deepcopy(DOCKER_INSPECT)
        inspection[0]["Architecture"] = "arm64"
        fake = FakeTool("docker", inspect=inspection)

        def run(argv, env=None):
            nonlocal active, peak, builds
            if "compose" in argv and "build" in argv:
                with lock:
                    active += 1
                    builds += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.05)
                    return fake(argv, env)
                finally:
                    with lock:
                        active -= 1
            return fake(argv, env)

        return ImageBuilder(**kwargs, run=run)

    monkeypatch.setattr("nemo_gym.data.tasks.harbor.build_tools.ImageBuilder", factory)
    build_task_images(tmp_path, workers=4)
    assert 1 < peak <= 4
    assert builds == 8
    assert len(read_built_images(tmp_path)) == 8
    assert len(read_image_configs(tmp_path)) == 8
    for index in range(8):
        assert load_task(tmp_path / f"task-{index}").image.startswith("hb__")
    assert load_task(tmp_path / "duplicate").image == load_task(tmp_path / "task-0").image
    build_task_images(tmp_path, workers=4)
    assert builds == 8, "A completed preparation must reuse valid records"


def test_build_failure_is_fatal_and_is_not_recorded(tmp_path, monkeypatch):
    write_task(tmp_path / "failed")
    monkeypatch.setattr(
        "nemo_gym.data.tasks.harbor.build_tools.ImageBuilder",
        lambda **kwargs: ImageBuilder(**kwargs, run=FakeTool("docker", fail="build")),
    )
    with pytest.raises(BuildError):
        build_task_images(tmp_path, workers=2)
    assert not read_built_images(tmp_path)


def test_conflicting_contexts_fail_before_any_build(tmp_path, monkeypatch):
    write_task(tmp_path / "a")
    write_task(tmp_path / "b")
    (tmp_path / "b/environment/tools/setup.sh").chmod(0o755)

    def unexpected_builder(**kwargs):
        raise AssertionError("Invalid plans must not run a build")

    monkeypatch.setattr("nemo_gym.data.tasks.harbor.build_tools.ImageBuilder", unexpected_builder)
    with pytest.raises(RuntimeError, match="Refusing invalid task image plan"):
        build_task_images(tmp_path)
