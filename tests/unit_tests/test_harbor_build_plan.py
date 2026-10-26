# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of build planning.

Context digests over task folders, and plan_builds over tasks, dataset.toml overrides, sidecars
with a build section and the separate verifier's tests entry.
"""

import hashlib
import shutil

import pytest

from nemo_gym.tasks.harbor.build_plan import BuildConflict, BuildError, detect_tool, needs_build, plan_builds
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.harbor_compat import environment_content_hash
from tests.unit_tests.harbor_build_helpers import (
    BUILD_DOCKERFILE,
    OVERLAY_DOCKERFILE,
    PULL_DOCKERFILE,
    REPO,
    TOML,
    UBUNTU_RECORD,
    harbor_hash,
    settled,
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

    def test_a_folder_with_the_hash_but_other_build_inputs_is_not_grouped(self, tmp_path):
        """Review (round 15): two tasks with identical file contents whose setup.sh modes were 0700 and 0755 became
        one entry built from the first folder, so both ran one image whose recorded inputs matched the first only."""
        folder = tmp_path / "ds"
        for name in "abcd":
            write_task(folder / name)
        script = lambda name: folder / name / "environment" / "tools" / "setup.sh"  # noqa: E731
        script("a").chmod(0o700)
        script("b").chmod(0o755)
        script("c").chmod(0o700)  # a's folder, entry for entry
        script("d").chmod(0o700)
        (folder / "d" / "environment" / "cache").mkdir()  # an empty folder, which Harbor's hash misses too
        digest = harbor_hash(folder / "a" / "environment")
        assert all(harbor_hash(folder / name / "environment") == digest for name in "bcd")
        conflicts: dict[str, BuildConflict] = {}
        plan = plan_builds(folder, conflicts=conflicts)
        assert [(entry.tasks, entry.digest, entry.context) for entry in plan] == [
            (("a", "c"), digest, (folder / "a" / "environment").resolve())
        ]
        assert conflicts == {
            "b": BuildConflict(
                "a",
                f"b: environment/ has the same Harbor environment id as a's, {digest} (the same file names and "
                f"contents, so the one record and the images named hb__{digest} would serve both), but not the same "
                "build inputs: tools/setup.sh is a file 0755 in b's and a file 0700 in a's; one image cannot be built "
                "from both folders, so the task is left out (make the two folders identical, or change a file's "
                "content so each gets its own image)",
            ),
            "d": BuildConflict(
                "a",
                f"d: environment/ has the same Harbor environment id as a's, {digest} (the same file names and "
                f"contents, so the one record and the images named hb__{digest} would serve both), but not the same "
                "build inputs: cache is only in d's; one image cannot be built from both folders, so the task is left "
                "out (make the two folders identical, or change a file's content so each gets its own image)",
            ),
        }
        # The comparison does not depend on the table being asked for, and the first task selected is the entry's.
        assert plan_builds(folder) == plan
        assert [entry.tasks for entry in plan_builds(folder, select=lambda name: name in "bd")] == [("b",)]
        assert [entry.tasks for entry in plan_builds(folder, select=lambda name: name in "bc")] == [("b",)]

    def test_a_task_left_out_for_its_environment_plans_no_verifier_image_either(self, tmp_path):
        folder = tmp_path / "ds"
        separate = TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
        for name in "ab":
            write_task(folder / name, toml=separate)
            (folder / name / "tests" / "Dockerfile").write_text("FROM x\nCOPY . /tests/\n")
        (folder / "b" / "environment" / "tools" / "setup.sh").chmod(0o755)
        conflicts: dict[str, BuildConflict] = {}
        plan = plan_builds(folder, conflicts=conflicts)
        # b's tests/ is a's byte for byte and would share that entry; b is left out whole, nothing is built for it.
        assert [(entry.service, entry.tasks, entry.verifier) for entry in plan] == [
            ("main", ("a",), False),
            ("verifier", ("a",), True),
        ]
        assert list(conflicts) == ["b"] and conflicts["b"].first == "a"
        assert "tools/setup.sh is a file 0755 in b's" in conflicts["b"].reason
        # A task whose own tests/ collides is in no entry either, its main included, though main would be first of
        # its own context: every folder is checked before any is added.
        write_task(folder / "c", toml=separate, extra={"tools/setup.sh": "#!/bin/sh\necho c\n"})
        (folder / "c" / "tests" / "Dockerfile").write_text("FROM x\nCOPY . /tests/\n")
        (folder / "c" / "tests" / "test.sh").chmod(0o700)
        # a's mode is set too, not left to the umask: the reason below names both.
        (folder / "a" / "tests" / "test.sh").chmod(0o644)
        conflicts.clear()
        plan = plan_builds(folder, conflicts=conflicts)
        assert [(entry.service, entry.tasks) for entry in plan] == [("main", ("a",)), ("verifier", ("a",))]
        assert sorted(conflicts) == ["b", "c"] and conflicts["c"].first == "a"
        assert conflicts["c"].reason.startswith("c: tests/ has the same Harbor")
        assert "test.sh is a file 0700 in c's and a file 0644 in a's" in conflicts["c"].reason
        # The first task left out of the selection, the next folder is the one held and the others are compared to it.
        conflicts.clear()
        plan = plan_builds(folder, select=lambda name: name != "a", conflicts=conflicts)
        assert [(entry.service, entry.tasks) for entry in plan] == [("main", ("b",)), ("verifier", ("b",))]
        assert list(conflicts) == ["c"] and conflicts["c"].first == "b"

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

    # G2: counselbench's and kumo's shapes, main with a build section, one sidecar without image: and one with.
    SIDECAR_COMPOSE = (
        "services:\n"
        "  main:\n"
        "    image: ${MAIN_IMAGE_NAME}\n"
        "    build:\n"
        "      context: .\n"
        "  cache:\n"
        "    image: redis:7\n"
        "  world:\n"
        "    build:\n"
        "      context: .\n"
        "      dockerfile: world/Dockerfile\n"
        "  verifier:\n"
        "    image: ${MAIN_IMAGE_NAME}_verifier\n"
        "    build:\n"
        "      context: ${CONTEXT_DIR}\n"
        "      dockerfile: Dockerfile_verifier\n"
    )
    SIDECAR_FILES = {
        "docker-compose.yaml": SIDECAR_COMPOSE,
        "world/Dockerfile": "FROM python:3.12-slim\nCOPY world/ /opt/world/\n",
        "Dockerfile_verifier": "FROM python:3.12-slim\nCOPY tests/ /tests/\n",
        "tools/setup.sh": "#!/bin/sh\necho ok\n",
    }

    def test_every_service_with_a_build_section_is_planned_main_first(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra=self.SIDECAR_FILES)
        write_task(folder / "b", extra=self.SIDECAR_FILES)  # same context: the same three services, shared
        context = (folder / "a" / "environment").resolve()
        plan = plan_builds(folder)
        # One entry per (context, service), main first and the sidecars in file order; a sidecar with image: and
        # no build: (cache) is pulled, not built.
        assert [(entry.service, entry.tasks) for entry in plan] == [
            ("main", ("a", "b")),
            ("world", ("a", "b")),
            ("verifier", ("a", "b")),
        ]
        assert {entry.digest for entry in plan} == {context_digest(context)} == {harbor_hash(context)}
        assert all(entry.context == context for entry in plan)
        # Planning reads only the build keys; the sections resolve at build time, each under environment/.
        assert all(entry.build_dir == context and entry.dockerfile == "Dockerfile" for entry in plan)
        main, world, verifier = (settled(entry) for entry in plan)
        assert main.dockerfile_path == context / "Dockerfile"
        assert world.build_dir == context and world.dockerfile_path == context / "world" / "Dockerfile"
        assert verifier.build_dir == context and verifier.dockerfile_path == context / "Dockerfile_verifier"

    def test_a_sidecar_is_planned_whatever_mains_mode(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  runtime:\n    build: ./runtime\n"
        files = {"docker-compose.yaml": compose, "runtime/Dockerfile": "FROM python:3.12\nCOPY . /app\n"}
        # Overlay-mode main (itsm-bench), pull-mode main, a prebuilt main, and a compose file without main.
        write_task(folder / "overlay", dockerfile=OVERLAY_DOCKERFILE, extra=files)
        write_task(folder / "pull", dockerfile=PULL_DOCKERFILE, extra=files)
        write_task(folder / "prebuilt", toml=TOML + 'docker_image = "org/prebuilt:1"\n', extra=files)
        write_task(
            folder / "no-main",
            dockerfile=PULL_DOCKERFILE,
            extra={**files, "docker-compose.yaml": "services:\n  runtime:\n    build: ./runtime\n"},
        )
        # A sidecar with image: only, and a Compose spelling Harbor never reads, plan nothing.
        write_task(
            folder / "pulled",
            dockerfile=PULL_DOCKERFILE,
            extra={"docker-compose.yaml": "services:\n  db:\n    image: redis:7\n"},
        )
        write_task(
            folder / "other-name",
            dockerfile=PULL_DOCKERFILE,
            extra={"compose.yaml": compose, "runtime/Dockerfile": "FROM x\n"},
        )
        write_base_record(folder)
        plan = plan_builds(folder)
        assert [(entry.service, entry.tasks) for entry in plan] == [
            ("runtime", ("no-main",)),
            ("runtime", ("overlay",)),
            ("runtime", ("prebuilt",)),
            ("runtime", ("pull",)),
        ]
        assert all(entry.context == (folder / entry.tasks[0] / "environment").resolve() for entry in plan)
        # A file that does not parse is left for the loader; the task's main still plans on its own.
        write_task(folder / "broken", extra={"docker-compose.yaml": "services: [\n"})
        assert [(e.service, e.tasks) for e in plan_builds(folder / "broken")] == [("main", ("broken",))]

    def test_tasks_sharing_a_context_plan_main_when_any_of_them_needs_it(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ./db\n"
        files = {"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"}
        write_task(folder / "a", extra=files)
        write_task(folder / "b", extra=files)
        # a is prebuilt by a dataset override and plans db alone; b shares the context and needs main built too.
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\ndocker_image = "org/pinned:2"\n')
        assert [(e.service, e.tasks) for e in plan_builds(folder)] == [("main", ("a", "b")), ("db", ("a", "b"))]

    def test_a_sidecar_context_outside_the_environment_is_refused_naming_the_service(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ../x\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        main, db = plan_builds(folder)
        assert (main.service, db.service) == ("main", "db")
        builder, fake = tagged_buildah(tmp_path)
        with pytest.raises(
            BuildError, match=r"services\.db\.build\.context is refused.*outside the task's environment"
        ):
            list(builder.build_many([main, db]))
        assert fake.calls == []

    def test_detect_tool_needs_docker_without_a_repository(self, monkeypatch):
        monkeypatch.setattr(
            shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("buildah", "podman") else None
        )
        assert detect_tool() is None and detect_tool(REPO) == "buildah"
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name == "docker" else None)
        assert detect_tool() == "docker" and detect_tool(REPO) == "docker"
