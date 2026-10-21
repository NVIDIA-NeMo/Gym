# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of build planning.

Context digests over task folders, and plan_builds over tasks, dataset.toml overrides, sidecars
with a build section and the separate verifier's tests entry.
"""

import hashlib
import json
import shutil

import pytest

from nemo_gym.tasks.harbor.build_plan import detect_tool, needs_build, plan_builds
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.harbor_compat import environment_content_hash
from nemo_gym.tasks.harbor.image_build import build_images
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE
from nemo_gym.tasks.harbor.task import load_task
from tests.unit_tests.harbor_build_helpers import (
    BUILD_DOCKERFILE,
    OVERLAY_DOCKERFILE,
    PULL_DOCKERFILE,
    REPO,
    TOML,
    UBUNTU_RECORD,
    harbor_hash,
    tagged_buildah,
    write_base_record,
    write_task,
)


class TestContextDigest:
    def test_the_executable_bit_is_not_part_of_the_context(self, tmp_path):
        a = write_task(tmp_path / "a")
        before = context_digest(a / "environment")
        script = a / "environment" / "tools" / "setup.sh"
        script.chmod(script.stat().st_mode | 0o111)
        # Harbor's environment id hashes paths and bytes only; naming parity with Harbor is the contract,
        # so a mode change keeps the image name (and the recorded build) it had.
        assert context_digest(a / "environment") == before

    def test_symlinks_are_skipped(self, tmp_path):
        a = write_task(tmp_path / "a")
        before = context_digest(a / "environment")
        (a / "environment" / "link").symlink_to("tools/setup.sh")
        assert context_digest(a / "environment") == before
        (a / "environment" / "link").unlink()
        (a / "environment" / "link").symlink_to("tools/elsewhere.sh")
        assert context_digest(a / "environment") == before

    def test_requires_a_folder(self, tmp_path):
        with pytest.raises(ValueError, match="Not a build context"):
            context_digest(tmp_path / "missing")

    def test_ignored_names_are_skipped_and_content_and_relative_names_count(self, tmp_path):
        context = write_task(tmp_path / "a") / "environment"
        before = context_digest(context)
        assert len(before) == 32
        # Same files in another folder: the path is not part of the id.
        assert context_digest(write_task(tmp_path / "b") / "environment") == before
        (context / "tools" / "__pycache__").mkdir()
        (context / "tools" / "__pycache__" / "setup.cpython-313.pyc").write_bytes(b"\x00")
        (context / ".DS_Store").write_bytes(b"junk")
        assert context_digest(context) == before
        (context / "tools" / "setup.sh").write_text("#!/bin/sh\necho changed\n")
        changed = context_digest(context)
        assert changed != before
        (context / "tools" / "setup.sh").rename(context / "tools" / "renamed.sh")
        assert context_digest(context) not in (before, changed)
        # An empty folder hashes to its name, the seed Harbor uses when no docker_image is declared.
        empty = tmp_path / "empty"
        empty.mkdir()
        assert context_digest(empty) == environment_content_hash(empty, docker_image=None)
        assert context_digest(empty) == hashlib.sha256(b"empty").hexdigest()[:32]


class TestPlan:
    def test_force_builds_the_dockerfile_of_a_task_that_declares_an_image(self, tmp_path):
        # Harbor's --force-build: SWE-Interact's declared images lack what its Dockerfiles COPY in.
        folder = tmp_path / "ds"
        write_task(folder / "e", toml=TOML + 'docker_image = "org/prebuilt:1"\n')
        assert plan_builds(folder) == []
        (entry,) = plan_builds(folder, force=True)
        assert (entry.tasks, entry.service) == (("e",), "main")

    def test_a_recorded_build_wins_over_the_declared_image(self, tmp_path):
        folder = tmp_path / "ds"
        task_dir = write_task(folder / "e", toml=TOML + 'docker_image = "org/prebuilt:1"\n')
        assert load_task(task_dir).image == "org/prebuilt:1"
        built = f"{REPO}/hb__{'2' * 32}@sha256:{'f' * 64}"
        (folder / BUILT_IMAGES_FILE).write_text(
            json.dumps({context_digest(task_dir / "environment"): {"image": built}})
        )
        assert load_task(task_dir).image == built

    def test_each_sidecar_build_context_is_planned_under_its_service(self, tmp_path):
        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  user-server:\n    build:\n      context: ./user-server\n"
            "    environment:\n      - LITELLM_API_KEY=${LITELLM_API_KEY}\n"
        )
        extra = {"docker-compose.yaml": compose, "user-server/Dockerfile": "FROM python:3.12-slim\n"}
        write_task(folder / "a", extra=extra)
        write_task(folder / "b", extra=extra)  # same sidecar context content: one build
        plan = plan_builds(folder)
        sidecar = next(entry for entry in plan if entry.service == "user-server")
        assert sidecar.tasks == ("a", "b")
        assert sidecar.context == (folder / "a" / "environment" / "user-server").resolve()
        assert sidecar.digest == harbor_hash(sidecar.context)

    def test_a_sidecar_context_builds_from_its_own_dockerfile(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ./db\n"
        write_task(
            folder / "a",
            dockerfile=PULL_DOCKERFILE,
            extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM x\n"},
        )
        write_base_record(folder)
        builder, tool = tagged_buildah(tmp_path)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        bud = next(call for call in tool.calls if "bud" in call)
        assert bud[-3:] == [
            "-f",
            str((folder / "a" / "environment" / "db" / "Dockerfile").resolve()),
            str((folder / "a" / "environment" / "db").resolve()),
        ]
        recorded = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert recorded[result.digest]["image"] == result.image

    def test_needs_build_classifies_the_three_modes(self):
        records = {"ubuntu:24.04": UBUNTU_RECORD}
        assert needs_build(BUILD_DOCKERFILE, records)
        assert not needs_build(PULL_DOCKERFILE)
        assert needs_build("FROM ubuntu\nRUN <<EOF\ntrue\nEOF\n")
        # An overlay file resolves only against its base image's record: with it, overlay; without, a build.
        assert not needs_build(OVERLAY_DOCKERFILE, records)
        assert needs_build(OVERLAY_DOCKERFILE)
        expanding = "FROM ubuntu:24.04\nRUN x\nWORKDIR $HOME/app\n"
        assert not needs_build(expanding, records) and needs_build(expanding)

    def test_groups_tasks_by_context_and_skips_prebuilt_and_overlay(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b")  # same context as a
        write_task(folder / "c", extra={"tools/setup.sh": "#!/bin/sh\necho other\n"})
        write_task(folder / "d", dockerfile=OVERLAY_DOCKERFILE)
        write_task(folder / "e", toml=TOML + 'docker_image = "org/prebuilt:1"\n')
        write_base_record(folder)
        plan = plan_builds(folder)
        assert [entry.tasks for entry in plan] == [("a", "b"), ("c",)]
        assert plan[0].context == (folder / "a" / "environment").resolve()
        assert plan[0].digest == harbor_hash(plan[0].context)

    def test_select_and_dataset_overrides_apply(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b")
        (folder / "dataset.toml").write_text('[gym.tasks."b".environment]\ndocker_image = "org/pinned:2"\n')
        assert [entry.tasks for entry in plan_builds(folder)] == [("a",)]
        assert plan_builds(folder, select=lambda name: name == "b") == []

    def test_single_task_folder(self, tmp_path):
        task = write_task(tmp_path / "solo")
        (plan,) = plan_builds(task)
        assert plan.tasks == ("solo",)

    def test_detect_tool_needs_docker_without_a_repository(self, monkeypatch):
        monkeypatch.setattr(
            shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("buildah", "podman") else None
        )
        assert detect_tool() is None and detect_tool(REPO) == "buildah"
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name == "docker" else None)
        assert detect_tool() == "docker" and detect_tool(REPO) == "docker"
