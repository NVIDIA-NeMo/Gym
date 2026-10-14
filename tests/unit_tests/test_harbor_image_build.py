# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dockerfile builds (image mode three): planning, building through a tool, recording, and the loader."""

import hashlib
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import dockerfile_has_heredoc
from nemo_gym.tasks.harbor.harbor_compat import environment_content_hash
from nemo_gym.tasks.harbor.image_build import (
    COMPOSE_BUILD_FILE,
    HTTP_403,
    BuildError,
    BuildPlanEntry,
    BuildResult,
    ImageBuilder,
    build_images,
    compose_build_env,
    detect_tool,
    harbor_compose_document,
    main_build_settings,
    needs_build,
    normalize_inspect,
    plan_builds,
    record_built_image,
    resolve_task_env,
    validate_task_compose,
)
from nemo_gym.tasks.harbor.image_configs import compose_variables
from nemo_gym.tasks.harbor.task import (
    BUILT_IMAGES_FILE,
    IMAGE_CONFIGS_FILE,
    HarborTaskError,
    load_task,
    read_built_images,
    read_image_configs,
)


TOML = """
schema_version = "1.4"

[task]
name = "org/hello"
version = "1.0.0"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
cpus = 1
memory_mb = 2048
"""

BUILD_DOCKERFILE = "FROM ubuntu:24.04\nCOPY tools/ /opt/tools/\nRUN /opt/tools/setup.sh\nWORKDIR /app\nUSER app\n"
OVERLAY_DOCKERFILE = "FROM ubuntu:24.04\nRUN apt-get install -y jq\nWORKDIR /app\n"
PULL_DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"
REPO = "registry.example/team/repo"


def write_task(root: Path, *, dockerfile: str = BUILD_DOCKERFILE, toml: str = TOML, extra: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Do it.\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    for name, text in (extra or {"tools/setup.sh": "#!/bin/sh\necho ok\n"}).items():
        target = root / "environment" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return root


UBUNTU_RECORD = {"os": "linux", "architecture": "amd64", "image": "ubuntu:24.04", "config": {"Env": ["HOME=/root"]}}


def write_base_record(folder: Path, image: str = "ubuntu:24.04") -> None:
    """Record a base image's configuration next to the task folders, as prepare does before planning builds."""
    path = folder / IMAGE_CONFIGS_FILE
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    recorded[image] = {**UBUNTU_RECORD, "image": image}
    path.write_text(json.dumps(recorded))


BUILDAH_INSPECT = {
    "FromImageID": "abc123",
    "OCIv1": {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "Env": ["PATH=/usr/bin:/bin", "APP_HOME=/app"],
            "WorkingDir": "/app",
            "User": "app",
            "Cmd": ["/bin/sh"],
            "Labels": {"ignored": "yes"},
        },
    },
}
ARM_BUILDAH_INSPECT = {"FromImageID": "arm001", "OCIv1": {"architecture": "arm64", "os": "linux", "config": {}}}
DOCKER_INSPECT = [
    {
        "Id": "sha256:def456",
        "Architecture": "amd64",
        "Os": "linux",
        "Config": {"Env": ["PATH=/usr/bin"], "WorkingDir": "/srv", "User": "", "Entrypoint": ["/entry"], "Cmd": None},
        "RepoDigests": ["registry.example/team/repo@sha256:feedface"],
    }
]


class FakeTool:
    """Records every command and answers build, push and inspect the way the real tool would."""

    def __init__(
        self,
        tool: str,
        *,
        fail: str | None = None,
        inspect: dict | list | None = None,
        fail_output: str | None = None,
    ) -> None:
        self.tool = tool
        self.fail = fail
        self.fail_output = fail_output
        self.inspect = inspect
        self.calls: list[list[str]] = []
        # The environment each command ran with: ``None`` when the builder let it inherit the process environment.
        self.envs: list[dict[str, str] | None] = []

    def __call__(self, argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        self.envs.append(env)
        verb = argv[1] if self.tool != "buildah" else next(a for a in argv[1:] if a in ("bud", "push", "inspect"))
        if self.fail and self.fail in argv:
            return subprocess.CompletedProcess(argv, 1, "", self.fail_output or f"{verb}: boom\n")
        if verb in ("inspect", "image"):
            if self.inspect is not None:
                data = self.inspect
            else:
                data = BUILDAH_INSPECT if self.tool == "buildah" else DOCKER_INSPECT
            return subprocess.CompletedProcess(argv, 0, json.dumps(data), "")
        if verb == "push":
            if "--digestfile" in argv:
                Path(argv[argv.index("--digestfile") + 1]).write_text("sha256:pushed01\n")
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "latest: digest: sha256:pushed02 size: 1234\n", "")
        return subprocess.CompletedProcess(argv, 0, "built\n", "")


def local_docker() -> tuple[ImageBuilder, FakeTool]:
    tool = FakeTool("docker")
    return ImageBuilder(tool="docker", run=tool), tool


def harbor_hash(context: Path) -> str:
    """The environment id Harbor would compute for the context, straight from the vendored function."""
    return environment_content_hash(context)


def tagged_buildah(tmp_path: Path, **kwargs) -> tuple[ImageBuilder, FakeTool]:
    """A buildah builder that tags for a repository but pushes nothing, so the image stays local."""
    tool = FakeTool("buildah", **kwargs)
    builder = ImageBuilder(tool="buildah", repository=REPO, push=False, storage_root=tmp_path / "store", run=tool)
    return builder, tool


def settled(entry: BuildPlanEntry, image: str = "hb__x") -> BuildPlanEntry:
    """``entry`` with ``main``'s build section applied, as ``ImageBuilder.build`` applies it once the task's Compose
    file is checked and before the build command."""
    env = compose_build_env(entry, image, None)
    document = validate_task_compose(entry.context, env, tasks=entry.tasks)
    return replace(entry, **main_build_settings(entry, document, env))


class TestContextDigest:
    def test_stable_and_sensitive_to_content_and_names(self, tmp_path):
        a = write_task(tmp_path / "a")
        b = write_task(tmp_path / "b")
        assert context_digest(a / "environment") == context_digest(b / "environment")
        (b / "environment" / "tools" / "setup.sh").write_text("#!/bin/sh\necho changed\n")
        assert context_digest(a / "environment") != context_digest(b / "environment")
        c = write_task(tmp_path / "c", extra={"tools/other.sh": "#!/bin/sh\necho ok\n"})
        assert context_digest(a / "environment") != context_digest(c / "environment")

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

    def test_is_harbors_environment_id_of_the_folder(self, tmp_path):
        context = write_task(tmp_path / "a") / "environment"
        assert context_digest(context) == environment_content_hash(context)
        assert len(context_digest(context)) == 32

    def test_ignored_names_are_skipped_and_content_and_relative_names_count(self, tmp_path):
        context = write_task(tmp_path / "a") / "environment"
        before = context_digest(context)
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


class TestImageBuilder:
    def test_docker_builds_inspects_and_records_locally(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("docker", inspect=[{**DOCKER_INSPECT[0], "Config": BUILDAH_INSPECT["OCIv1"]["config"]}])
        builder = ImageBuilder(tool="docker", run=tool)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.image == result.tag == f"hb__{result.digest}"
        assert result.pushed is False and result.tool == "docker"
        version_argv, build_argv, inspect_argv = tool.calls
        # docker is Harbor's build step: the Compose plugin is checked, then `docker compose ... build`.
        assert version_argv == ["docker", "compose", "version"]
        assert build_argv[:2] == ["docker", "compose"] and build_argv[-1] == "build"
        assert build_argv[build_argv.index("--project-directory") + 1] == str(folder / "a" / "environment")
        assert inspect_argv == ["docker", "image", "inspect", result.tag]
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {result.digest: {"image": result.tag, "tasks": ["a"], "tool": "docker", "pushed": False}}
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert (
            configs[result.tag]["config"]["User"] == "app" and configs[result.tag]["config_digest"] == "sha256:def456"
        )

    def test_tag_for_follows_harbor_naming(self, tmp_path):
        digest = "0123456789abcdef0123456789abcdef"  # pragma: allowlist secret
        entry = BuildPlanEntry(digest, tmp_path / "ctx", ("a", "b"))
        local = ImageBuilder(tool="docker", run=FakeTool("docker"))
        assert local.tag_for(entry) == f"hb__{digest}"
        # A repository is the registry path the Harbor name goes under; a trailing slash is dropped.
        pushed = ImageBuilder(tool="buildah", repository=REPO + "/", storage_root=tmp_path, run=FakeTool("buildah"))
        assert pushed.tag_for(entry) == f"{REPO}/hb__{digest}"

    def test_buildah_builds_and_pushes_with_its_storage_flags(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=tool)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.tag == f"{REPO}/hb__{result.digest}"
        assert result.image == f"{result.tag}@sha256:pushed01" and result.pushed and result.tool == "buildah"
        build_argv, inspect_argv, push_argv = tool.calls
        assert build_argv[:7] == [
            "buildah",
            "--storage-driver",
            "vfs",
            "--root",
            str(tmp_path / "store" / "storage"),
            "--runroot",
            str(tmp_path / "store" / "run"),
        ]
        assert build_argv[7:14] == ["bud", "--format", "docker", "--isolation", "chroot", "--platform", "linux/amd64"]
        assert build_argv[-3:] == [
            "-f",
            str(folder / "a" / "environment" / "Dockerfile"),
            str(folder / "a" / "environment"),
        ]
        assert inspect_argv[-3:] == ["--type", "image", result.tag]
        assert push_argv[-2:] == [result.tag, f"docker://{result.tag}"] and "--digestfile" in push_argv
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {result.digest: {"image": result.image, "tasks": ["a"], "tool": "buildah", "pushed": True}}
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert configs[result.image] == {
            "architecture": "amd64",
            "config": {
                "Cmd": ["/bin/sh"],
                "Env": ["PATH=/usr/bin:/bin", "APP_HOME=/app"],
                "User": "app",
                "WorkingDir": "/app",
            },
            "config_digest": "sha256:abc123",
            "image": result.image,
            "os": "linux",
        }

    def test_podman_pushes_with_a_digest_file_and_records_the_pinned_image(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("podman")
        builder = ImageBuilder(tool="podman", repository=REPO, run=tool)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.pushed and result.tag == f"{REPO}/hb__{result.digest}"
        assert result.image == f"{result.tag}@sha256:pushed01"
        assert [call[1] for call in tool.calls] == ["build", "image", "push"]
        assert tool.calls[0][:7] == ["podman", "build", "--format", "docker", "--platform", "linux/amd64", "-t"]
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built[result.digest]["image"] == result.image and built[result.digest]["pushed"] is True
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[result.image]["config"]["WorkingDir"] == "/srv"

    def test_docker_push_digest_comes_from_the_push_output(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("docker")
        builder = ImageBuilder(tool="docker", repository=REPO, platform="linux/amd64", run=tool)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.image == f"{REPO}/hb__{result.digest}@sha256:pushed02"
        assert tool.calls[1][:2] == ["docker", "compose"] and tool.calls[1][-1] == "build"
        assert tool.calls[2] == ["docker", "image", "inspect", result.tag] and tool.calls[3][:2] == ["docker", "push"]

    def test_an_image_of_another_platform_is_refused_before_push_and_record(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("buildah", inspect=ARM_BUILDAH_INSPECT)
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "s", run=tool)
        with pytest.raises(BuildError, match="came out as linux/arm64, not linux/amd64"):
            build_images(folder, builder=builder, log=lambda _: None)
        assert not any("push" in call for call in tool.calls)
        assert not (folder / BUILT_IMAGES_FILE).exists()
        # Asking for that platform accepts the image.
        builder = ImageBuilder(
            tool="buildah", repository=REPO, platform="linux/arm64", storage_root=tmp_path / "s", run=tool
        )
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[result.image]["architecture"] == "arm64"

    def test_a_recorded_image_is_rebuilt_when_it_does_not_fit_the_run(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        local, _ = local_docker()
        (first,) = build_images(folder, builder=local, log=lambda _: None)
        # A local image cannot serve a run that pushes: it is rebuilt and the record replaced.
        lines: list[str] = []
        pushing = ImageBuilder(tool="podman", repository=REPO, run=FakeTool("podman"))
        (second,) = build_images(folder, builder=pushing, log=lines.append)
        assert lines[0] == f"Rebuilding {first.image} for a: it was built locally and this run pushes to a repository"
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built[first.digest] == {"image": second.image, "tasks": ["a"], "tool": "podman", "pushed": True}
        # A pushed image serves a local run too.
        assert build_images(folder, builder=local, log=lines.append) == []
        assert lines[-1] == f"Built image recorded for a: {second.image}"
        # A recorded image of another platform is rebuilt as well, and its record replaced.
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        configs[second.image]["architecture"] = "arm64"
        (folder / IMAGE_CONFIGS_FILE).write_text(json.dumps(configs))
        build_images(folder, builder=pushing, log=lines.append)
        assert lines[-2].endswith(": it is linux/arm64, not linux/amd64")
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[second.image]["architecture"] == "amd64"

    def test_a_deleted_record_forces_a_rebuild_and_replaces_the_image_config(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, tool = local_docker()
        (first,) = build_images(folder, builder=builder, log=lambda _: None)
        # The documented way to force a rebuild of an unchanged context: delete its built-images.json entry.
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        del built[first.digest]
        (folder / BUILT_IMAGES_FILE).write_text(json.dumps(built))
        # The rebuilt image carries a different configuration under the same local tag.
        tool.inspect = [{**DOCKER_INSPECT[0], "Id": "sha256:rebuilt", "Config": {"WorkingDir": "/rebuilt"}}]
        lines: list[str] = []
        (again,) = build_images(folder, builder=builder, log=lines.append)
        assert again.digest == first.digest and len([c for c in tool.calls if c[-1] == "build"]) == 2
        assert again.image == first.image
        assert lines == [f"Built {again.image} (local, docker) for task(s) a"]
        assert json.loads((folder / BUILT_IMAGES_FILE).read_text())[again.digest]["image"] == again.image
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[again.image] == again.record
        assert again.record["config_digest"] == "sha256:rebuilt"

    def test_platform_variants_compare_on_os_and_arch(self, tmp_path):
        from nemo_gym.tasks.harbor.image_build import platform_key, recorded_image_mismatch

        assert platform_key("linux/arm64/v8") == "linux/arm64" and platform_key(None) == ""
        folder = tmp_path / "ds"
        write_task(folder / "a")
        tool = FakeTool("buildah", inspect=ARM_BUILDAH_INSPECT)
        builder = ImageBuilder(
            tool="buildah", repository=REPO, platform="linux/arm64/v8", storage_root=tmp_path / "s", run=tool
        )
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[result.digest]
        assert recorded_image_mismatch(entry, configs, repository=REPO, platform="linux/arm64/v8") is None
        assert recorded_image_mismatch(entry, configs, repository=REPO, platform="linux/amd64") is not None

    def test_pushing_needs_a_repository_and_unknown_tools_are_refused(self, tmp_path):
        with pytest.raises(BuildError, match="needs builds.repository"):
            ImageBuilder(tool="docker", push=True, run=FakeTool("docker"))
        with pytest.raises(BuildError, match="Unknown build tool"):
            ImageBuilder(tool="kaniko", run=FakeTool("docker"))
        # buildah and podman keep images where no sandbox provider looks; without a push target they are refused.
        for tool in ("buildah", "podman"):
            with pytest.raises(BuildError, match=f"{tool} keeps built images.*builds.repository"):
                ImageBuilder(tool=tool, run=FakeTool(tool))
        # A repository without a push is allowed: the image is tagged for it and kept local.
        assert ImageBuilder(tool="buildah", repository=REPO, push=False, run=FakeTool("buildah")).push is False

    def test_no_tool_on_path_is_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(BuildError, match=r"No image builder found: install docker \(buildah and podman need"):
            ImageBuilder()
        with pytest.raises(BuildError, match="No image builder found: install buildah, podman, docker"):
            ImageBuilder(repository=REPO)

    def test_a_failed_build_records_nothing_and_names_the_tasks(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b", extra={"tools/setup.sh": "#!/bin/sh\necho b\n"})
        tool = FakeTool("docker", fail=str(folder / "b" / "environment"))
        builder = ImageBuilder(tool="docker", run=tool)
        with pytest.raises(BuildError, match=r"Building .* for task\(s\) b failed \(exit 1\)") as info:
            build_images(folder, builder=builder, log=lambda _: None)
        assert "boom" in str(info.value)
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        # The first context was recorded before the second failed; the failed one left no trace.
        assert [entry["tasks"] for entry in built.values()] == [["a"]]

    def test_recorded_contexts_are_not_rebuilt(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, tool = local_docker()
        build_images(folder, builder=builder, log=lambda _: None)
        lines: list[str] = []
        assert build_images(folder, builder=builder, log=lines.append) == []
        expected = f"Built image recorded for a: hb__{plan_builds(folder)[0].digest}"
        # The Compose plugin check, the compose build and the inspect of the first call; nothing afterwards.
        assert len(tool.calls) == 3 and lines == [expected]

    def test_record_keeps_existing_entries(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        (folder / IMAGE_CONFIGS_FILE).write_text(json.dumps({"other:1": {"os": "linux"}}))
        result = BuildResult(
            digest="d" * 64, tasks=("a",), tag="t:1", image="t:1", tool="docker", pushed=False, record={"os": "linux"}
        )
        record_built_image(folder, result)
        later = BuildResult(
            digest="d" * 64, tasks=("z",), tag="t:9", image="t:9", tool="docker", pushed=False, record={}
        )
        record_built_image(folder, later)
        assert json.loads((folder / BUILT_IMAGES_FILE).read_text())["d" * 64]["image"] == "t:1"
        assert set(json.loads((folder / IMAGE_CONFIGS_FILE).read_text())) == {"other:1", "t:1"}
        # Overwriting replaces both the build entry and the image's record.
        record_built_image(folder, later, overwrite=True)
        assert json.loads((folder / BUILT_IMAGES_FILE).read_text())["d" * 64]["image"] == "t:9"
        assert "t:9" in json.loads((folder / IMAGE_CONFIGS_FILE).read_text())

    def test_normalize_inspect_rejects_unexpected_shapes_and_warns_off_platform(self, caplog):
        with pytest.raises(BuildError, match="no OCIv1"):
            normalize_inspect("buildah", {"nope": 1})
        with pytest.raises(BuildError, match="no Config"):
            normalize_inspect("docker", [])
        arm = [{"Id": "sha256:1", "Architecture": "arm64", "Os": "linux", "Config": {}}]
        with caplog.at_level("WARNING"):
            record = normalize_inspect("podman", arm)
        assert record["architecture"] == "arm64" and "not linux/amd64" in caplog.text

    def test_build_argv_for_buildah_uses_a_platform_when_given(self, tmp_path):
        builder = ImageBuilder(
            tool="buildah", repository=REPO, platform="linux/arm64", storage_root=tmp_path, run=FakeTool("buildah")
        )
        entry = BuildPlanEntry("a" * 64, tmp_path / "ctx", ("t",))
        argv = builder.build_argv(entry, "x:1")
        assert argv[argv.index("--platform") + 1] == "linux/arm64"


# Harbor v0.22.0's build Compose file, quoted verbatim from
# https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker-compose-build.yaml
# (Gym does not depend on the harbor package, so the file is carried here as the reference).
HARBOR_BUILD_TEMPLATE = """\
services:
  main:
    build:
      context: ${CONTEXT_DIR}
    pull_policy: build
    command: [ "sh", "-c", "sleep infinity" ]
"""

FAKE_DOCKER = """\
#!{python}
# A docker CLI stand-in: records every invocation (argv, Harbor's variables, the generated Compose file) and
# answers `compose version`, `compose ... build`, `image inspect` and `push` the way docker would.
import json
import os
import sys

argv = sys.argv[1:]
record = {{"argv": argv, "env": {{k: os.environ.get(k) for k in ("MAIN_IMAGE_NAME", "CONTEXT_DIR",
    "DOCKER_DEFAULT_PLATFORM", "HOST_VERIFIER_LOGS_PATH", "ENV_ARTIFACTS_PATH", "CPUS", "MEMORY",
    "PYTHON_VERSION", "FROM_HOST")}}}}
if "-f" in argv:
    files = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
    record["files"] = files
    record["compose"] = open(files[0]).read()
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
if argv[:2] == ["compose", "version"]:
    if os.environ.get("FAKE_DOCKER_NO_COMPOSE"):
        sys.stderr.write("docker: 'compose' is not a docker command.\\nSee 'docker --help'\\n")
        sys.exit(1)
    print("Docker Compose version v2.29.1")
elif argv[:2] == ["image", "inspect"]:
    print(json.dumps({inspect!r}))
elif argv[:1] == ["push"]:
    print("latest: digest: sha256:pushed03 size: 1234")
elif argv[0] == "compose" and argv[-1] == "build":
    print("built")
else:
    sys.stderr.write("unexpected: " + " ".join(argv) + "\\n")
    sys.exit(2)
"""


class TestDockerComposeBackend:
    """The docker backend is Harbor's build step: a fake ``docker`` on PATH records what prepare runs."""

    @pytest.fixture
    def fake_docker(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        script = bin_dir / "docker"
        script.write_text(FAKE_DOCKER.format(python=sys.executable, inspect=DOCKER_INSPECT))
        script.chmod(0o755)
        log = tmp_path / "docker-calls.jsonl"
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
        monkeypatch.delenv("FAKE_DOCKER_NO_COMPOSE", raising=False)

        def calls() -> list[dict]:
            return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

        return calls

    def test_generated_compose_matches_harbor_template_field_by_field(self, tmp_path):
        context = tmp_path / "environment"
        harbor = yaml.safe_load(HARBOR_BUILD_TEMPLATE.replace("${CONTEXT_DIR}", str(context)))
        gym = harbor_compose_document(context, "hb__abc")
        assert list(gym) == ["services"] and list(gym["services"]) == ["main"]
        harbor_main, gym_main = harbor["services"]["main"], gym["services"]["main"]
        # Every field Harbor's file has is present with Harbor's value, the build section identical to Harbor's
        # with CONTEXT_DIR resolved (no `dockerfile`: Harbor names none, and naming one would clash with a task
        # file's `dockerfile_inline`).
        for key, value in harbor_main.items():
            assert gym_main[key] == value, key
        assert gym_main["build"] == harbor_main["build"] == {"context": str(context)}
        # Gym's one addition: the MAIN_IMAGE_NAME value as `image` (Harbor's file sets none).
        assert set(gym_main) - set(harbor_main) == {"image"} and gym_main["image"] == "hb__abc"
        assert gym_main["pull_policy"] == "build" and gym_main["command"] == ["sh", "-c", "sleep infinity"]

    def test_docker_runs_compose_build_over_harbor_file_never_docker_build(self, tmp_path, fake_docker):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        context = folder / "a" / "environment"
        builder = ImageBuilder(tool="docker")
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        calls = fake_docker()
        assert [c["argv"][:2] for c in calls] == [
            ["compose", "version"],
            ["compose", "--project-name"],
            ["image", "inspect"],
        ]
        assert not any(c["argv"][:1] == ["build"] for c in calls)
        build = calls[1]
        assert build["argv"] == [
            "compose",
            "--project-name",
            f"hb__{result.digest}",
            "--project-directory",
            str(context),
            "-f",
            build["files"][0],
            "build",
        ]
        assert Path(build["files"][0]).name == COMPOSE_BUILD_FILE
        # The file docker compose read is Harbor's, resolved for this context, tagged with Harbor's name.
        assert yaml.safe_load(build["compose"]) == harbor_compose_document(context, f"hb__{result.digest}")
        assert build["env"]["MAIN_IMAGE_NAME"] == f"hb__{result.digest}" == result.tag
        assert build["env"]["CONTEXT_DIR"] == str(context)
        assert build["env"]["DOCKER_DEFAULT_PLATFORM"] == "linux/amd64"
        assert build["env"]["HOST_VERIFIER_LOGS_PATH"] == "/logs/verifier"
        assert build["env"]["ENV_ARTIFACTS_PATH"] == "/logs/artifacts"
        # The recorded output is unchanged: the loader and the pull-mode path read the same files.
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {result.digest: {"image": result.tag, "tasks": ["a"], "tool": "docker", "pushed": False}}
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[result.tag]["config"]["WorkingDir"] == "/srv"

    def test_task_compose_main_build_section_and_prebuilt_sidecar_build_by_one_compose_build(
        self, tmp_path, fake_docker
    ):
        folder = tmp_path / "ds"
        compose = (
            "services:\n"
            "  main:\n"
            "    image: ${MAIN_IMAGE_NAME}\n"
            "    build:\n"
            "      context: ${CONTEXT_DIR}\n"
            "      dockerfile: Dockerfile\n"
            "      target: prod\n"
            "      args:\n"
            "        PYTHON_VERSION: '3.12'\n"
            "      additional_contexts:\n"
            "        base: service:cache\n"
            "      cache_from:\n"
            "        - type=registry,ref=registry.example/cache\n"
            "  cache:\n"
            "    image: redis:7\n"
        )
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "tools/setup.sh": "echo ok\n"})
        context = folder / "a" / "environment"
        # The validator returns the file as docker compose reads it: interpolated against the build's variables.
        variables = {"MAIN_IMAGE_NAME": "hb__x", "CONTEXT_DIR": str(context)}
        interpolated = compose.replace("${MAIN_IMAGE_NAME}", "hb__x").replace("${CONTEXT_DIR}", str(context))
        assert validate_task_compose(context, variables) == yaml.safe_load(interpolated)
        lines: list[str] = []
        builder = ImageBuilder(tool="docker")
        (result,) = build_images(folder, builder=builder, log=lines.append)
        build = fake_docker()[1]
        # Harbor's order: its own build file first, the task's docker-compose.yaml after it, project directory
        # the environment folder so relative paths resolve there; only main has a build section to build.
        assert build["files"] == [build["files"][0], str(context / "docker-compose.yaml")]
        assert build["argv"][-1] == "build" and build["argv"][-3] == "-f"
        assert build["argv"][build["argv"].index("--project-directory") + 1] == str(context)
        assert build["env"]["MAIN_IMAGE_NAME"] == result.tag and build["env"]["CONTEXT_DIR"] == str(context)
        assert lines == [f"Built {result.image} (local, docker) for task(s) a"]

    @pytest.mark.parametrize(
        ("sidecar", "message"),
        [
            ("  db:\n    build: ./db\n", r"service 'db' has a build section; sidecars must name a prebuilt image:"),
            ("  db:\n    image: hb-db\n    build: ./db\n", r"service 'db' has a build section"),
        ],
    )
    def test_a_sidecar_with_a_build_section_is_refused_before_any_build(self, tmp_path, fake_docker, sidecar, message):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n" + sidecar
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        with pytest.raises(BuildError, match=message):
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert [c["argv"] for c in fake_docker()] == [["compose", "version"]]
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_a_task_compose_using_dockerfile_inline_builds_with_harbors_file(self, tmp_path, fake_docker):
        folder = tmp_path / "ds"
        compose = (
            "services:\n"
            "  main:\n"
            "    build:\n"
            "      context: ${CONTEXT_DIR}\n"
            "      dockerfile_inline: |\n"
            "        FROM python:3.12\n"
            "        RUN pip install x\n"
        )
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        context = folder / "a" / "environment"
        (result,) = build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        build = fake_docker()[1]
        # Compose refuses a service declaring both dockerfile and dockerfile_inline; Gym's file, like Harbor's,
        # names no dockerfile, so the task's inline file merges cleanly over it.
        generated = yaml.safe_load(build["compose"])["services"]["main"]
        assert "dockerfile" not in generated["build"] and generated["image"] == result.tag
        assert build["files"][1] == str(context / "docker-compose.yaml")

    REFUSED = [
        ("    build:\n      secrets: [tok]\n", r"services\.main\.build\.secrets is refused"),
        ("    build:\n      ssh: [default]\n", r"services\.main\.build\.ssh is refused"),
        ("    build:\n      extra_hosts: [host.docker.internal:host-gateway]\n", r"build\.extra_hosts is refused"),
        ("    build:\n      network: host\n", r"build\.network is refused.*host networking"),
        (
            "    build:\n      additional_contexts:\n        home: /home\n",
            r"build\.additional_contexts is refused.*'/home'",
        ),
        (
            "    build:\n      additional_contexts: [src=../..]\n",
            r"build\.additional_contexts is refused.*'\.\./\.\.'",
        ),
        ("    build:\n      cache_to: [type=gha,token=abc]\n", r"build\.cache_to is refused.*token="),
        (
            "    build:\n      cache_from:\n        - type: s3\n          secret_access_key: k\n",
            r"build\.cache_from is refused",
        ),
        ("    build: ../..\n", r"build\.context is refused.*outside the task's environment folder"),
        ("    build:\n      context: /\n", r"build\.context is refused.*outside"),
        ("    build:\n      context: https://github.com/x/y.git\n", r"build\.context is refused.*remote context"),
        ("    build:\n      context: ${HOME}\n", r"build\.context is refused.*outside"),
        ("    build:\n      args: {A: '${NEMO_GYM_NOT_SET:?set it}'}\n", r"interpolation is refused.*set it"),
        ("    build:\n      dockerfile: ../../task.toml\n", r"build\.dockerfile is refused.*outside"),
        ("    build:\n      dockerfile: link\n", r"build\.dockerfile is refused.*outside"),
        (
            "    extends:\n      file: ../other.yaml\n      service: main\n",
            r"services\.main\.extends\.file is refused",
        ),
    ]

    @pytest.mark.parametrize(("snippet", "message"), REFUSED, ids=[m[:28] for _, m in REFUSED])
    def test_host_reaching_build_keys_are_refused_before_any_build(
        self, tmp_path, fake_docker, monkeypatch, snippet, message
    ):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n" + snippet
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        context = folder / "a" / "environment"
        (context / "link").symlink_to(folder.parent / "elsewhere")
        # Compose interpolates ${HOME} from the build's environment (the process environment): a path outside.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NEMO_GYM_NOT_SET", raising=False)
        with pytest.raises(BuildError, match=message) as info:
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert str(info.value).startswith(str(context / "docker-compose.yaml"))
        # The file is read at build time, after the plugin check and before docker compose build: nothing is built.
        assert [c["argv"] for c in fake_docker()] == [["compose", "version"]]
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_a_top_level_include_is_refused(self, tmp_path):
        context = tmp_path / "environment"
        context.mkdir()
        (context / "docker-compose.yaml").write_text("include: [../other.yaml]\nservices:\n  main: {}\n")
        with pytest.raises(BuildError, match=r"include is refused"):
            validate_task_compose(context, {})

    # Values that pass a check on the raw text and resolve to the refused value when docker compose interpolates
    # the file against the build's environment. ``env`` is set in the process environment for the case.
    BYPASSES = [
        ("    build:\n      network: ${X:-host}\n", {}, r"build\.network is refused.*host networking"),
        ("    build:\n      network: ${X-host}\n", {}, r"build\.network is refused"),
        ("    build:\n      network: ${NET_MODE}\n", {"NET_MODE": "host"}, r"build\.network is refused"),
        ("    build:\n      network: $NET_MODE\n", {"NET_MODE": "host"}, r"build\.network is refused"),
        ("    build:\n      cache_to: ['type=gha,${X:-token}=abc']\n", {}, r"build\.cache_to is refused.*token="),
        ("    build:\n      cache_from: ['${CACHE}']\n", {"CACHE": "type=s3,secret_access_key=k"}, r"cache_from"),
        ("    build:\n      additional_contexts: {home: '${X:-/home}'}\n", {}, r"additional_contexts.*'/home'"),
        ("    build:\n      extra_hosts: ['${X:-a:1.2.3.4}']\n", {}, r"build\.extra_hosts is refused"),
    ]

    @pytest.mark.parametrize(("snippet", "env", "message"), BYPASSES, ids=[s.split()[1][:24] for s, _, _ in BYPASSES])
    def test_interpolation_cannot_hide_a_refused_value(
        self, tmp_path, fake_docker, monkeypatch, snippet, env, message
    ):
        for name in ("X", "NET_MODE", "CACHE"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n" + snippet
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        with pytest.raises(BuildError, match=message):
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert [c["argv"] for c in fake_docker()] == [["compose", "version"]]
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_the_project_directorys_dotenv_counts_and_the_environment_wins_over_it(self, tmp_path, monkeypatch):
        # docker compose reads interpolation variables from .env in --project-directory (Harbor's environment/)
        # when the environment does not set them, so the check reads it the same way.
        monkeypatch.delenv("X", raising=False)
        context = write_task(tmp_path / "a", extra={"docker-compose.yaml": "", ".env": "X=host\n"}) / "environment"
        compose = "services:\n  main:\n    build:\n      context: ${CONTEXT_DIR}\n      network: ${X:-bridge}\n"
        (context / "docker-compose.yaml").write_text(compose)
        variables = {"CONTEXT_DIR": str(context), "MAIN_IMAGE_NAME": "hb__x"}
        with pytest.raises(BuildError, match=r"build\.network is refused"):
            validate_task_compose(context, variables)
        document = validate_task_compose(context, {**variables, "X": "bridge"})
        assert document["services"]["main"]["build"] == {"context": str(context), "network": "bridge"}

    # ``.env`` values docker compose interpolates before the document reads them: an unquoted or double-quoted
    # value resolves ${VAR} against the command's environment and the lines above it, so a check that took the
    # file literally would pass a placeholder that Compose builds with the refused value. ``env`` is the build's
    # environment for the case; UNSET_ANYWHERE is never in it.
    DOTENV_BYPASSES = [
        ("NET=${UNSET_ANYWHERE:-host}\n", "network: ${NET}", {}, r"build\.network is refused.*host networking"),
        ('NET="${B}"\n', "network: ${NET}", {"B": "host"}, r"build\.network is refused"),
        ("H=host\nNET=${H}\n", "network: ${NET}", {}, r"build\.network is refused"),
        ("TOK=${UNSET_ANYWHERE:-token}\n", "cache_to: ['type=gha,${TOK}=abc']", {}, r"cache_to is refused.*token="),
        ("CTX=${HOME}\n", "context: ${CTX}", {"HOME": "/"}, r"build\.context is refused.*resolves outside"),
    ]

    @pytest.mark.parametrize(
        ("dotenv", "line", "env", "message"),
        DOTENV_BYPASSES,
        ids=["dotenv_default", "dotenv_env_ref", "dotenv_chain", "dotenv_cache_credential", "dotenv_context_escape"],
    )
    def test_dotenv_values_are_interpolated_as_compose_interpolates_them(self, tmp_path, dotenv, line, env, message):
        context = write_task(tmp_path / "a", extra={".env": dotenv}) / "environment"
        compose = f"services:\n  main:\n    image: ${{MAIN_IMAGE_NAME}}\n    build:\n      {line}\n"
        if "context:" not in line:
            compose += "      context: ${CONTEXT_DIR}\n"
        (context / "docker-compose.yaml").write_text(compose)
        variables = {"CONTEXT_DIR": str(context), "MAIN_IMAGE_NAME": "hb__x", **env}
        with pytest.raises(BuildError, match=message):
            validate_task_compose(context, variables)

    def test_single_quoted_dotenv_values_stay_literal_and_an_unset_reference_is_empty(self, tmp_path, caplog):
        # Compose takes a single-quoted .env value as written; ``$$`` and a double-quoted ``\$`` are one dollar.
        dotenv = "NET='${X:-host}'\nARG=\"$$HOME \\$HOME\"\n"
        context = write_task(tmp_path / "a", extra={".env": dotenv}) / "environment"
        compose = (
            "services:\n  main:\n    build:\n      context: ${CONTEXT_DIR}\n      network: ${NET}\n"
            "      args:\n        A: ${ARG}\n"
        )
        (context / "docker-compose.yaml").write_text(compose)
        variables = {"CONTEXT_DIR": str(context), "MAIN_IMAGE_NAME": "hb__x", "X": "host", "HOME": "/root"}
        document = validate_task_compose(context, variables)
        assert document["services"]["main"]["build"] == {
            "context": str(context),
            "network": "${X:-host}",
            "args": {"A": "$HOME $HOME"},
        }
        # A bare reference nobody sets is an empty string in .env as in the document, with one warning naming the
        # task(s): Compose's own rule, and the loader's for the same file (image_configs.substitute_with_empty).
        (context / ".env").write_text("NET=${NOT_SET_ANYWHERE}\nARG=x\n")
        with caplog.at_level("WARNING", logger="nemo_gym.tasks.harbor.image_build"):
            document = validate_task_compose(context, variables, tasks=("a",))
        assert document["services"]["main"]["build"]["network"] == ""
        assert [r.getMessage() for r in caplog.records] == [
            f"{context / 'docker-compose.yaml'}: Compose variable ${{NOT_SET_ANYWHERE}} has no value in the build's "
            "environment (the process environment, the task's [environment].env and Harbor's variables) or "
            "environment/.env for task(s) a; it is an empty string, as docker compose makes it"
        ]
        # A required reference in .env is still refused, as Compose refuses it, naming the file.
        (context / ".env").write_text("NET=${NOT_SET_ANYWHERE:?set it}\nARG=x\n")
        with pytest.raises(
            BuildError, match=r"\.env: \$\{NOT_SET_ANYWHERE\} is required: set it; a value docker compose"
        ):
            validate_task_compose(context, variables)
        with pytest.raises(BuildError, match=r"COMPOSE_ENV_FILES is set in the build's environment"):
            validate_task_compose(context, {**variables, "COMPOSE_ENV_FILES": str(tmp_path / "other.env")})

    def test_compose_env_files_in_the_environment_fails_the_build_before_docker_compose_runs(
        self, tmp_path, fake_docker, monkeypatch
    ):
        # COMPOSE_ENV_FILES points Compose at other files than the environment/.env the check reads.
        monkeypatch.setenv("COMPOSE_ENV_FILES", str(tmp_path / "elsewhere.env"))
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    build:\n      args: {A: '1'}\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        with pytest.raises(
            BuildError, match=r"COMPOSE_ENV_FILES is set in the build's environment.*environment/\.env only"
        ):
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        # Refused when the file is read for the build, after the plugin check: docker compose build never ran.
        assert [c["argv"] for c in fake_docker()] == [["compose", "version"]]
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_benign_interpolation_resolves_from_the_builds_environment(self, tmp_path, fake_docker, monkeypatch):
        monkeypatch.delenv("NET_MODE", raising=False)
        folder = tmp_path / "ds"
        compose = (
            "services:\n"
            "  main:\n"
            "    image: ${MAIN_IMAGE_NAME}\n"
            "    build:\n"
            "      context: ${CONTEXT_DIR}\n"
            "      dockerfile: ${CONTEXT_DIR}/Dockerfile\n"
            "      network: ${NET_MODE:-bridge}\n"
            "      args:\n"
            "        MEM: ${MEMORY}\n"
            "        FROM_DOTENV: ${DOTENV_ONLY}\n"
            "        CHAINED: ${CHAINED}\n"
            "        LITERAL: $$HOME\n"
        )
        # A double-quoted .env value is interpolated as Compose interpolates it, against the build's environment
        # (MEMORY is Harbor's) and the lines above it (DOTENV_ONLY).
        dotenv = 'DOTENV_ONLY="${NOT_SET_ANYWHERE:-yes}"\nCHAINED=${DOTENV_ONLY}-${MEMORY}\n'
        write_task(folder / "a", extra={"docker-compose.yaml": compose, ".env": dotenv})
        context = folder / "a" / "environment"
        (result,) = build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        build = fake_docker()[1]
        assert build["files"][1] == str(context / "docker-compose.yaml") and build["env"]["MEMORY"] == "2048M"
        (entry,) = plan_builds(folder)
        document = validate_task_compose(context, compose_build_env(entry, result.tag, None))
        assert document["services"]["main"] == {
            "image": result.tag,
            "build": {
                "context": str(context),
                "dockerfile": f"{context}/Dockerfile",
                "network": "bridge",
                "args": {"MEM": "2048M", "FROM_DOTENV": "yes", "CHAINED": "yes-2048M", "LITERAL": "$$HOME"},
            },
        }

    @pytest.mark.parametrize("name", ["docker-compose.yml", "compose.yaml", "compose.yml"])
    def test_compose_spellings_harbor_never_reads_are_refused(self, tmp_path, fake_docker, name):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    build:\n      target: slim\n      args:\n        FEATURE: 'off'\n"
        write_task(folder / "a", extra={name: compose})
        with pytest.raises(
            BuildError, match=rf"{name}: Harbor's Docker environment reads .*docker-compose\.yaml only"
        ):
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert [c["argv"] for c in fake_docker()] == [["compose", "version"]]

    def test_main_image_other_than_harbors_is_refused(self, tmp_path, fake_docker):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra={"docker-compose.yaml": "services:\n  main:\n    image: mine:1\n"})
        with pytest.raises(BuildError, match=r"services\.main\.image must be \$\{MAIN_IMAGE_NAME\}.*'mine:1'"):
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert [c["argv"] for c in fake_docker()] == [["compose", "version"]]
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_missing_compose_plugin_fails_before_the_build_and_names_it(self, tmp_path, fake_docker, monkeypatch):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        monkeypatch.setenv("FAKE_DOCKER_NO_COMPOSE", "1")
        with pytest.raises(
            BuildError, match=r"Compose plugin is not available.*Install docker-compose-plugin"
        ) as info:
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert "'compose' is not a docker command" in str(info.value)
        assert [c["argv"] for c in fake_docker()] == [["compose", "version"]]
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_compose_env_puts_harbors_variables_over_the_process_environment(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("MAIN_IMAGE_NAME", "shell-wins-in-plain-compose")
        monkeypatch.setenv("KEEP_ME", "1")
        monkeypatch.setenv("HOST_TOKEN_VALUE", "from-host")
        monkeypatch.setenv("TEST_DIR", "/shell-says")
        monkeypatch.delenv("UNSET_ON_HOST", raising=False)
        entry = BuildPlanEntry(
            "a" * 32,
            tmp_path,
            ("t",),
            # Harbor's resolve_env_vars: whole-value templates read the host, defaults stand in, the rest literal.
            {
                "LITERAL": "x",
                "FROM_HOST": "${HOST_TOKEN_VALUE}",
                "DEFAULTED": "${UNSET_ON_HOST:-fallback}",
                "CONTEXT_DIR": "/task/says",
                "MEMORY": "1",
            },
            cpus=2,
            memory_mb=4096,
        )
        with caplog.at_level("WARNING", logger="nemo_gym.tasks.harbor.image_build"):
            env = compose_build_env(entry, "hb__x", None)
        # os.environ < task env < infra, Harbor's order.
        assert env["MAIN_IMAGE_NAME"] == "hb__x" and env["KEEP_ME"] == "1" and "DOCKER_DEFAULT_PLATFORM" not in env
        assert env["LITERAL"] == "x" and env["FROM_HOST"] == "from-host" and env["DEFAULTED"] == "fallback"
        assert env["CONTEXT_DIR"] == str(tmp_path) and env["CPUS"] == "2" and env["MEMORY"] == "4096M"
        # The two the loader defines for the same file, so the build and the seed interpolate it from one set.
        assert env["TEST_DIR"] == "/tests" and env["COMPOSE_PROJECT_NAME"] == "hb__" + "a" * 32
        assert "reserved by Harbor's Compose infrastructure, ignored: CONTEXT_DIR, MEMORY" in caplog.text
        with pytest.raises(BuildError, match=r"\[environment\]\.env NEEDED of task\(s\) t reads \$\{UNSET_ON_HOST\}"):
            resolve_task_env({"NEEDED": "${UNSET_ON_HOST}"}, tasks=("t",))

    def test_task_env_and_resources_reach_the_compose_build_like_harbors(self, tmp_path, fake_docker, monkeypatch):
        monkeypatch.setenv("HOST_TOKEN_VALUE", "from-host")
        monkeypatch.setenv("MAIN_IMAGE_NAME", "shell-value")
        folder = tmp_path / "ds"
        toml = TOML + '\n[environment.env]\nPYTHON_VERSION = "3.12"\nFROM_HOST = "${HOST_TOKEN_VALUE}"\n'
        toml += 'MAIN_IMAGE_NAME = "task-says"\nCONTEXT_DIR = "/task/says"\n'
        compose = (
            "services:\n  main:\n    build:\n      args:\n        PYTHON_VERSION: ${PYTHON_VERSION}\n"
            "        NCPU: ${CPUS}\n        MEM: ${MEMORY}\n"
        )
        write_task(folder / "a", toml=toml, extra={"docker-compose.yaml": compose})
        context = folder / "a" / "environment"
        (entry,) = plan_builds(folder)
        assert (entry.env, entry.cpus, entry.memory_mb) == (
            {
                "PYTHON_VERSION": "3.12",
                "FROM_HOST": "${HOST_TOKEN_VALUE}",
                "MAIN_IMAGE_NAME": "task-says",
                "CONTEXT_DIR": "/task/says",
            },
            1,
            2048,
        )
        (result,) = build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        env = fake_docker()[1]["env"]
        assert env["PYTHON_VERSION"] == "3.12" and env["FROM_HOST"] == "from-host"
        assert env["CPUS"] == "1" and env["MEMORY"] == "2048M"
        # Harbor's infrastructure variables win over the task's and the shell's.
        assert env["MAIN_IMAGE_NAME"] == result.tag and env["CONTEXT_DIR"] == str(context)

    def test_task_env_is_left_out_without_a_compose_file_as_harbor_leaves_it(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", toml=TOML + '\n[environment.env]\nX = "${NOT_SET_ANYWHERE}"\n')
        (entry,) = plan_builds(folder)
        assert entry.env == {} and entry.cpus == 1 and entry.memory_mb == 2048
        # Without a Compose file nothing interpolates, so an unresolvable template does not fail the build.
        assert "NOT_SET_ANYWHERE" not in str(compose_build_env(entry, "hb__x", None).get("X"))

    def test_tasks_sharing_a_context_with_different_env_share_the_image_and_are_logged(self, tmp_path, caplog):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    build:\n      args:\n        V: ${V}\n"
        write_task(folder / "a", toml=TOML + '\n[environment.env]\nV = "1"\n', extra={"docker-compose.yaml": compose})
        write_task(folder / "b", toml=TOML + '\n[environment.env]\nV = "2"\n', extra={"docker-compose.yaml": compose})
        write_task(folder / "c", toml=TOML + '\n[environment.env]\nV = "1"\n', extra={"docker-compose.yaml": compose})
        with caplog.at_level("WARNING", logger="nemo_gym.tasks.harbor.image_build"):
            (entry,) = plan_builds(folder)
        assert entry.tasks == ("a", "b", "c") and entry.env == {"V": "1"}
        assert "Tasks a and b share build context" in caplog.text and "(V)" in caplog.text
        assert "2" not in caplog.text.split("(V)")[0].split("settings")[-1]  # values are not logged
        assert "Tasks a and c" not in caplog.text


class TestToolEnvironment:
    """What buildah and podman run with: a containers.conf override, a registries.conf, and a proxy note on a 403."""

    @pytest.mark.parametrize("tool", ["buildah", "podman"])
    def test_buildah_and_podman_run_with_a_no_volumes_containers_conf(self, tmp_path, monkeypatch, tool):
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        monkeypatch.delenv("CONTAINERS_REGISTRIES_CONF", raising=False)
        monkeypatch.setenv("NEMO_GYM_PASSTHROUGH", "kept")
        folder = tmp_path / "ds"
        write_task(folder / "a")
        fake = FakeTool(tool)
        builder = ImageBuilder(tool=tool, repository=REPO, storage_root=tmp_path / "store", run=fake)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        assert result.pushed
        conf = tmp_path / "store" / "containers.conf"
        # The host's containers.conf may bind-mount /etc/localtime into build steps; this layer removes the default
        # volumes so a Dockerfile that installs tzdata can replace the file.
        assert conf.read_text() == "[containers]\nvolumes = []\n"
        assert [c[0] for c in fake.calls] == [tool, tool, tool]  # build, inspect, push
        # Unqualified FROM names resolve to docker.io, as under docker compose build: Debian's and Ubuntu's stock
        # registries.conf names no unqualified-search registry and the tools then refuse `FROM oven/bun:1`.
        registries = tmp_path / "store" / "registries.conf"
        assert registries.read_text() == 'unqualified-search-registries = ["docker.io"]\n'
        for env in fake.envs:
            assert env is not None and env["CONTAINERS_CONF_OVERRIDE"] == str(conf)
            assert env["CONTAINERS_REGISTRIES_CONF"] == str(registries)
            assert env["NEMO_GYM_PASSTHROUGH"] == "kept"
        # Written once; a second build reuses both files.
        stamps = (conf.stat().st_mtime_ns, registries.stat().st_mtime_ns)
        builder.build(plan_builds(folder)[0])
        assert (conf.stat().st_mtime_ns, registries.stat().st_mtime_ns) == stamps
        assert fake.envs[-1]["CONTAINERS_CONF_OVERRIDE"] == str(conf)
        assert fake.envs[-1]["CONTAINERS_REGISTRIES_CONF"] == str(registries)

    def test_a_callers_containers_conf_override_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CONTAINERS_CONF_OVERRIDE", str(tmp_path / "mine.conf"))
        monkeypatch.delenv("CONTAINERS_REGISTRIES_CONF", raising=False)
        folder = tmp_path / "ds"
        write_task(folder / "a")
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        build_images(folder, builder=builder, log=lambda _: None)
        # The caller's file applies as it is, and no override file is written; the registries file is still the
        # builder's.
        assert len(fake.envs) == 3
        for env in fake.envs:
            assert env["CONTAINERS_CONF_OVERRIDE"] == str(tmp_path / "mine.conf")
            assert env["CONTAINERS_REGISTRIES_CONF"] == str(tmp_path / "store" / "registries.conf")
        assert not (tmp_path / "store" / "containers.conf").exists()

    @pytest.mark.parametrize("tool", ["buildah", "podman"])
    def test_a_callers_registries_conf_is_left_alone(self, tmp_path, monkeypatch, tool):
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        monkeypatch.setenv("CONTAINERS_REGISTRIES_CONF", "/etc/containers/site-registries.conf")
        folder = tmp_path / "ds"
        write_task(folder / "a")
        fake = FakeTool(tool)
        builder = ImageBuilder(tool=tool, repository=REPO, storage_root=tmp_path / "store", run=fake)
        build_images(folder, builder=builder, log=lambda _: None)
        # A site whose main registries.conf carries mirrors or blocks keeps its own file (the override would
        # replace it); the containers.conf layer still applies.
        assert len(fake.envs) == 3
        for env in fake.envs:
            assert env["CONTAINERS_REGISTRIES_CONF"] == "/etc/containers/site-registries.conf"
            assert env["CONTAINERS_CONF_OVERRIDE"] == str(tmp_path / "store" / "containers.conf")
        assert not (tmp_path / "store" / "registries.conf").exists()

    def test_docker_gets_no_containers_conf(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        monkeypatch.delenv("CONTAINERS_REGISTRIES_CONF", raising=False)
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, fake = local_docker()
        assert builder.tool_env() is None
        build_images(folder, builder=builder, log=lambda _: None)
        version_env, build_env, inspect_env = fake.envs
        assert version_env is None and inspect_env is None
        assert "CONTAINERS_CONF_OVERRIDE" not in build_env and "CONTAINERS_REGISTRIES_CONF" not in build_env
        assert "MAIN_IMAGE_NAME" in build_env

    def test_a_403_names_the_proxy_variables_of_the_tools_environment_without_values(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        for name in [name for name in os.environ if name.lower().endswith("_proxy")]:
            monkeypatch.delenv(name)
        monkeypatch.setenv("https_proxy", "http://user:hunter2@proxy.example:3128")
        monkeypatch.setenv("NO_PROXY", "localhost,.internal.example")
        folder = tmp_path / "ds"
        write_task(folder / "a")
        denied = "Error: trying to reuse blob: pinging container registry registry.example: StatusCode: 403\n"
        fake = FakeTool("podman", fail="push", fail_output=denied)
        builder = ImageBuilder(tool="podman", repository=REPO, storage_root=tmp_path / "store", run=fake)
        with caplog.at_level(logging.WARNING), pytest.raises(BuildError) as info:
            build_images(folder, builder=builder, log=lambda _: None)
        message = str(info.value)
        assert "StatusCode: 403" in message
        assert "proxy variables NO_PROXY, https_proxy" in message and "add the registry host to no_proxy" in message
        for secret in ("hunter2", "proxy.example", "internal.example"):
            assert secret not in message and secret not in caplog.text
        assert "proxy variables NO_PROXY, https_proxy" in caplog.text
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_a_403_without_proxy_variables_points_at_the_tools_own_configuration(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        for name in [name for name in os.environ if name.lower().endswith("_proxy")]:
            monkeypatch.delenv(name)
        folder = tmp_path / "ds"
        write_task(folder / "a")
        fake = FakeTool("buildah", fail="push", fail_output="error pinging docker registry: 403 Forbidden\n")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        with pytest.raises(BuildError, match=r"carries no proxy variables.*containers\.conf"):
            build_images(folder, builder=builder, log=lambda _: None)
        # A failure that is not a 403 says nothing about proxies.
        fake = FakeTool("buildah", fail="push")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        with pytest.raises(BuildError) as info:
            build_images(folder, builder=builder, log=lambda _: None)
        assert "registry answered 403" not in str(info.value) and "proxy variables" not in str(info.value)

    def test_a_403_inside_a_hex_digest_is_not_a_403(self, tmp_path, monkeypatch):
        digest = "sha256:7f403ab9c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"
        assert HTTP_403.search(f"copying blob {digest}: connection reset\n") is None
        assert HTTP_403.search("b403") is None and HTTP_403.search("4030") is None
        assert (
            HTTP_403.search("StatusCode: 403\n")
            and HTTP_403.search("error: 403 Forbidden")
            and HTTP_403.search("(403)")
        )
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        folder = tmp_path / "ds"
        write_task(folder / "a")
        fake = FakeTool("buildah", fail="push", fail_output=f"error copying blob {digest}: connection reset\n")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        with pytest.raises(BuildError) as info:
            build_images(folder, builder=builder, log=lambda _: None)
        assert "registry answered 403" not in str(info.value) and "proxy variables" not in str(info.value)


HEREDOC_DOCKERFILE = "FROM docker.io/library/alpine:3.20\nCOPY tools/ /opt/tools/\nRUN <<EOF\necho hello > /x\nEOF\n"
PROBE_TAG = "localhost/nemo-gym/heredoc-probe"


class HeredocTool:
    """A buildah or podman stand-in whose heredoc preflight passes, or fails as Ubuntu 24.04's packages fail it."""

    def __init__(self, tool: str, *, heredocs: bool) -> None:
        self.tool = tool
        self.heredocs = heredocs
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        # The Dockerfile text of every preflight, read while its temporary context still exists.
        self.probe_dockerfiles: list[str] = []

    def __call__(self, argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        self.envs.append(env)
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, f"{self.tool} version 1.33.7 (image-spec 1.1.0-rc.5)\n", "")
        if "-f" in argv:
            if argv[argv.index("-t") + 1] == PROBE_TAG:
                self.probe_dockerfiles.append(Path(argv[argv.index("-f") + 1]).read_text())
                if not self.heredocs:
                    stderr = 'Error: building at STEP "PROBE ": Build error: Unknown instruction: "PROBE"\n'
                    return subprocess.CompletedProcess(argv, 125, "", stderr)
            return subprocess.CompletedProcess(argv, 0, "built\n", "")
        return FakeTool(self.tool)(argv, env)


class TestDirectBuildPath:
    """buildah and podman build the Dockerfile themselves: the docker image format, the heredoc preflight and
    ``main``'s build section of the task's Compose file, which ``docker compose build`` applies on the docker path."""

    # -- G7: image format ------------------------------------------------------------------------

    def test_buildah_and_podman_build_docker_format_images_and_docker_compose_gets_no_format(self, tmp_path):
        entry = BuildPlanEntry("a" * 32, tmp_path / "ctx", ("t",))
        buildah = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path, run=FakeTool("buildah"))
        argv = buildah.build_argv(entry, "x:1")
        bud = argv.index("bud")
        # Under the tools' default OCI format SHELL is ignored by RUN and HEALTHCHECK and ONBUILD are dropped.
        assert argv[bud : bud + 5] == ["bud", "--format", "docker", "--isolation", "chroot"]
        assert argv[-5:] == ["-t", "x:1", "-f", str(tmp_path / "ctx" / "Dockerfile"), str(tmp_path / "ctx")]
        podman = ImageBuilder(tool="podman", repository=REPO, run=FakeTool("podman"))
        assert podman.build_argv(entry, "x:1") == [
            "podman",
            "build",
            "--format",
            "docker",
            "--platform",
            "linux/amd64",
            "-t",
            "x:1",
            "-f",
            str(tmp_path / "ctx" / "Dockerfile"),
            str(tmp_path / "ctx"),
        ]
        docker, _ = local_docker()
        assert "--format" not in docker.compose_build_argv(entry, tmp_path / COMPOSE_BUILD_FILE)

    def test_normalize_inspect_reads_the_docker_config_over_the_oci_one(self):
        healthcheck = {"Test": ["CMD-SHELL", "curl -f localhost || exit 1"], "Interval": 30000000000}
        data = {
            "FromImageID": "abc",
            "OCIv1": {
                "architecture": "amd64",
                "os": "linux",
                "config": {"Env": ["A=1"], "WorkingDir": "/app", "Cmd": ["sh"], "Labels": {"ignored": "yes"}},
            },
            "Docker": {
                "config": {
                    "Shell": ["/bin/bash", "-c"],
                    "Healthcheck": healthcheck,
                    "OnBuild": ["RUN echo child"],
                    "Cmd": ["bash"],
                    "Entrypoint": None,
                    "Env": ["A=1"],
                }
            },
        }
        record = normalize_inspect("buildah", data)
        # The docker-format fields the OCI configuration cannot hold, and the docker section's Cmd over OCI's.
        assert record["config"] == {
            "Cmd": ["bash"],
            "Env": ["A=1"],
            "Healthcheck": healthcheck,
            "OnBuild": ["RUN echo child"],
            "Shell": ["/bin/bash", "-c"],
            "WorkingDir": "/app",
        }
        assert record["config_digest"] == "sha256:abc" and record["architecture"] == "amd64"
        # docker reports the docker-format configuration as Config.
        docker = [
            {
                "Id": "1",
                "Architecture": "amd64",
                "Os": "linux",
                "Config": {
                    "Shell": ["/bin/bash", "-c"],
                    "OnBuild": ["RUN x"],
                    "Cmd": ["sh"],
                    "Healthcheck": healthcheck,
                },
            }
        ]
        assert normalize_inspect("docker", docker)["config"] == {
            "Cmd": ["sh"],
            "Healthcheck": healthcheck,
            "OnBuild": ["RUN x"],
            "Shell": ["/bin/bash", "-c"],
        }
        # podman 4.9.3's `image inspect` of a docker-format image: Config is the OCI image configuration (no Shell,
        # no OnBuild) and Healthcheck is a top-level key beside it.
        podman = [
            {
                "Id": "0cad",
                "Architecture": "amd64",
                "Os": "linux",
                "Config": {"Env": ["A=1"], "Cmd": ["sh"], "WorkingDir": "/app", "Labels": {"io.buildah.version": "1"}},
                "Healthcheck": {"Test": ["CMD", "true"], "Interval": 5000000000, "Timeout": 3000000000},
            }
        ]
        assert normalize_inspect("podman", podman)["config"] == {
            "Cmd": ["sh"],
            "Env": ["A=1"],
            "Healthcheck": {"Test": ["CMD", "true"], "Interval": 5000000000, "Timeout": 3000000000},
            "WorkingDir": "/app",
        }
        # Without a HEALTHCHECK podman reports no such key, and none is recorded.
        del podman[0]["Healthcheck"]
        assert normalize_inspect("podman", podman)["config"] == {"Cmd": ["sh"], "Env": ["A=1"], "WorkingDir": "/app"}
        # An inspect without a Docker section (an OCI-format image) records as before.
        assert normalize_inspect("buildah", BUILDAH_INSPECT)["config"] == {
            "Cmd": ["/bin/sh"],
            "Env": ["PATH=/usr/bin:/bin", "APP_HOME=/app"],
            "User": "app",
            "WorkingDir": "/app",
        }

    # -- G5: heredoc preflight -------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("FROM a\nRUN <<EOF\necho hi\nEOF\n", True),
            ("FROM a\nRUN python - <<'PY'\nimport os\nPY\n", True),
            ('FROM a\nCOPY <<-"EOF" /x\nhi\nEOF\n', True),
            ("FROM a\nADD <<EOF /x\nhi\nEOF\n", True),
            ("FROM a\nRUN apt-get install -y \\\n  curl <<EOF\nhi\nEOF\n", True),
            # A space before the delimiter: BuildKit does not read it as a heredoc either (a hub defect).
            ("FROM a\nRUN cat > /x << 'EOF'\nhi\nEOF\n", False),
            ("FROM a\nRUN echo '<<EOF' && true\n", False),
            ("# RUN <<EOF\nFROM a\nENV X=<<EOF\n", False),
            (BUILD_DOCKERFILE, False),
        ],
    )
    def test_dockerfile_has_heredoc_judges_run_copy_and_add_lines_word_by_word(self, text, expected):
        assert dockerfile_has_heredoc(text) is expected

    @pytest.mark.parametrize("tool", ["buildah", "podman"])
    def test_a_tool_without_heredoc_support_fails_once_naming_tool_version_file_and_remedy(
        self, tmp_path, monkeypatch, tool
    ):
        # tool_env passes a preset CONTAINERS_CONF_OVERRIDE / CONTAINERS_REGISTRIES_CONF through by design.
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        monkeypatch.delenv("CONTAINERS_REGISTRIES_CONF", raising=False)
        folder = tmp_path / "ds"
        write_task(folder / "a", dockerfile=HEREDOC_DOCKERFILE)
        write_task(folder / "b", dockerfile=HEREDOC_DOCKERFILE + "ENV X=1\n")
        runner = HeredocTool(tool, heredocs=False)
        builder = ImageBuilder(tool=tool, repository=REPO, storage_root=tmp_path / "store", run=runner)
        first, second = plan_builds(folder)
        with pytest.raises(BuildError) as info:
            builder.build(first)
        message = str(info.value)
        assert message.startswith(str(folder / "a" / "environment" / "Dockerfile"))
        assert f"this host's {tool} cannot build one" in message
        assert f"{tool} version 1.33.7" in message and 'Unknown instruction: "PROBE"' in message
        assert "imagebuilder 1.2.6 or later" in message and "builds.tool: docker" in message
        # The preflight: FROM scratch plus a COPY heredoc (no base image), built with the build's own flags and
        # environment in a temporary context.
        assert runner.probe_dockerfiles == ["FROM scratch\nCOPY <<EOF /probe\nprobe\nEOF\n"]
        (probe,) = [call for call in runner.calls if PROBE_TAG in call]
        assert probe[:2] == [tool, "--storage-driver" if tool == "buildah" else "build"]
        assert "--format" in probe and probe[-1] == str(Path(probe[probe.index("-f") + 1]).parent)
        probe_env = runner.envs[runner.calls.index(probe)]
        assert probe_env == builder.tool_env() and probe_env["CONTAINERS_CONF_OVERRIDE"].endswith("containers.conf")
        # The second context fails from the cached outcome: no new preflight, and no task image was ever built.
        with pytest.raises(BuildError, match=re.escape(str(folder / "b" / "environment" / "Dockerfile"))):
            builder.build(second)
        assert len(runner.probe_dockerfiles) == 1
        assert [call for call in runner.calls if any("hb__" in arg for arg in call)] == []

    def test_the_preflight_runs_once_when_it_passes_and_never_for_a_dockerfile_without_a_heredoc(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a_plain")  # COPY: a build, but no heredoc
        write_task(folder / "b_heredoc", dockerfile=HEREDOC_DOCKERFILE)
        write_task(folder / "c_heredoc", dockerfile=HEREDOC_DOCKERFILE + "ENV X=1\n")
        runner = HeredocTool("buildah", heredocs=True)
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=runner)
        results = build_images(folder, builder=builder, log=lambda _: None)
        assert [result.tasks for result in results] == [("a_plain",), ("b_heredoc",), ("c_heredoc",)]
        assert len(runner.probe_dockerfiles) == 1
        builds = [i for i, call in enumerate(runner.calls) if "bud" in call]
        probes = [i for i, call in enumerate(runner.calls) if PROBE_TAG in call]
        # Three task builds and one probe, run after the plain build and before the first heredoc build.
        assert len(builds) == 4 and probes == [builds[1]]
        assert "--version" not in [arg for call in runner.calls for arg in call]
        plain = HeredocTool("buildah", heredocs=False)
        build_images(
            tmp_path / "ds" / "a_plain",
            builder=ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "s2", run=plain),
            log=lambda _: None,
        )
        assert plain.probe_dockerfiles == []

    # -- G6: main's build section ----------------------------------------------------------------

    MAIN_BUILD_COMPOSE = (
        "services:\n"
        "  main:\n"
        "    image: ${MAIN_IMAGE_NAME}\n"
        "    build:\n"
        "      context: ./svc\n"
        "      dockerfile: Dockerfile.agent\n"
        "      target: t\n"
        "      args:\n"
        "        A: '1'\n"
        "        B: ${TASK_VAR}\n"
    )
    TASK_ENV_TOML = TOML + '\n[environment.env]\nTASK_VAR = "from-task"\n'

    def test_mains_build_section_reaches_the_direct_build(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(
            folder / "a",
            toml=self.TASK_ENV_TOML,
            extra={
                "docker-compose.yaml": self.MAIN_BUILD_COMPOSE,
                "svc/Dockerfile.agent": "FROM scratch\nCOPY note /note\n",
                "svc/note": "n\n",
            },
        )
        context = (folder / "a" / "environment").resolve()
        (planned,) = plan_builds(folder)
        # Planning reads no Compose file: the entry carries the context and its default Dockerfile.
        assert planned.build_dir == context and planned.build_args == {} and planned.target is None
        entry = settled(planned)
        # The hash stays Harbor's hash of environment/ (the Compose file is part of it); the build settings are
        # main's: context and Dockerfile under environment/, args interpolated from the task's env, the target.
        assert entry.digest == context_digest(context) == harbor_hash(context)
        assert entry.build_dir == context / "svc" and entry.dockerfile_path == context / "svc" / "Dockerfile.agent"
        assert entry.build_args == {"A": "1", "B": "from-task"} and entry.target == "t"
        assert entry.dockerfile_inline is False
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        argv = builder.build_argv(entry, "x:1")
        assert argv[argv.index("bud") :] == [
            "bud",
            "--format",
            "docker",
            "--isolation",
            "chroot",
            "--platform",
            "linux/amd64",
            "--build-arg",
            "A=1",
            "--build-arg",
            "B=from-task",
            "--target",
            "t",
            "-t",
            "x:1",
            "-f",
            str(context / "svc" / "Dockerfile.agent"),
            str(context / "svc"),
        ]
        podman = ImageBuilder(tool="podman", repository=REPO, run=FakeTool("podman"))
        assert podman.build_argv(entry, "x:1")[4:] == argv[argv.index("--platform") :]
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        # build_images plans the bare entry and the builder applies the section itself before its first command.
        assert fake.calls[0] == builder.build_argv(entry, result.tag) and result.pushed
        assert json.loads((folder / BUILT_IMAGES_FILE).read_text())[entry.digest]["tasks"] == ["a"]

    def test_string_form_list_args_yaml_scalars_and_the_prepare_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEMO_GYM_HOST_ARG", "from-host")
        monkeypatch.delenv("NEMO_GYM_UNSET_ARG", raising=False)
        folder = tmp_path / "ds"
        write_task(
            folder / "a",
            extra={
                "docker-compose.yaml": "services:\n  main:\n    build: ./svc\n",
                "svc/Dockerfile": "FROM scratch\n",
            },
        )
        compose = (
            "services:\n  main:\n    build:\n      context: ${CONTEXT_DIR}\n      args:\n"
            "        - A=1\n        - NEMO_GYM_HOST_ARG\n        - NEMO_GYM_UNSET_ARG\n"
            "        - FLAG=${NEMO_GYM_HOST_ARG:-x}\n"
        )
        write_task(folder / "b", dockerfile=BUILD_DOCKERFILE + "ENV B=1\n", extra={"docker-compose.yaml": compose})
        compose = "services:\n  main:\n    build:\n      args: {N: 3, T: true, E: null}\n      target: ''\n"
        write_task(folder / "c", dockerfile=BUILD_DOCKERFILE + "ENV C=1\n", extra={"docker-compose.yaml": compose})
        write_task(folder / "d", dockerfile=BUILD_DOCKERFILE + "ENV D=1\n")
        a, b, c, d = (settled(entry) for entry in plan_builds(folder))
        # The string form is a context; the default Dockerfile name under it.
        assert a.build_dir == (folder / "a" / "environment").resolve() / "svc" and a.dockerfile == "Dockerfile"
        assert a.build_args == {} and a.target is None
        # Compose's list form: KEY=value, a bare KEY from the build's environment (the process environment counts,
        # as it does for docker compose), and a bare unset KEY dropped; Harbor's CONTEXT_DIR resolves.
        assert b.build_dir == (folder / "b" / "environment").resolve()
        assert b.build_args == {"A": "1", "NEMO_GYM_HOST_ARG": "from-host", "FLAG": "from-host"}
        # YAML scalars as Compose passes them; a null value is read from the environment like a bare list KEY and
        # dropped when unset; an empty target is none.
        assert c.build_args == {"N": "3", "T": "true"} and c.target is None
        # Without a Compose file the context builds its own Dockerfile.
        assert d.build_dir == d.context and d.dockerfile_path == d.context / "Dockerfile" and d.build_args == {}

    @pytest.mark.parametrize("args", ["{PYTHON_VERSION: }", "[PYTHON_VERSION]"], ids=["mapping-null", "list-bare"])
    def test_a_null_mapping_arg_is_read_from_the_environment_like_a_bare_list_key(self, tmp_path, monkeypatch, args):
        # Compose loads `KEY:` and `- KEY` into the same mapping with a nil value, resolves it from the project
        # environment and drops it when unset, so the Dockerfile's `ARG PYTHON_VERSION=3.12` default stands; an
        # explicit `--build-arg PYTHON_VERSION=` would override that default with an empty string.
        folder = tmp_path / "ds"
        compose = f"services:\n  main:\n    build:\n      args: {args}\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        (entry,) = plan_builds(folder)
        builder, _ = tagged_buildah(tmp_path)
        monkeypatch.delenv("PYTHON_VERSION", raising=False)
        assert settled(entry).build_args == {} and "--build-arg" not in builder.build_argv(settled(entry), "x:1")
        monkeypatch.setenv("PYTHON_VERSION", "3.13")
        assert settled(entry).build_args == {"PYTHON_VERSION": "3.13"}
        argv = builder.build_argv(settled(entry), "x:1")
        assert argv[argv.index("--build-arg") + 1] == "PYTHON_VERSION=3.13"

    @pytest.mark.parametrize(
        ("snippet", "message"),
        [
            ("    build: ../..\n", r"build\.context is refused.*outside the task's environment folder"),
            ("    build:\n      dockerfile: ../../task.toml\n", r"build\.dockerfile is refused.*outside"),
            (
                "    build:\n      args: {A: '${NEMO_GYM_NOT_SET:?set it}'}\n",
                r"interpolation is refused for a Dockerfile build of task\(s\) a: \$\{NEMO_GYM_NOT_SET\} is required: set it",
            ),
            ("    build:\n      args: A=1\n", r"build\.args is refused.*mapping or a list"),
        ],
    )
    def test_main_build_refusals_happen_before_any_command_and_name_the_task(
        self, tmp_path, monkeypatch, snippet, message
    ):
        monkeypatch.delenv("NEMO_GYM_NOT_SET", raising=False)
        folder = tmp_path / "ds"
        write_task(folder / "a", extra={"docker-compose.yaml": "services:\n  main:\n" + snippet})
        write_task(folder / "b", dockerfile=BUILD_DOCKERFILE + "ENV B=1\n")
        # Planning reads no Compose file, so one task's section cannot stop the dataset; its build refuses it.
        bad, good = plan_builds(folder)
        builder, fake = tagged_buildah(tmp_path)
        with pytest.raises(BuildError, match=message) as info:
            builder.build(bad)
        assert str(info.value).startswith(str((folder / "a" / "environment").resolve() / "docker-compose.yaml"))
        assert fake.calls == []
        assert builder.build(good).tasks == ("b",)

    def test_an_unset_variable_is_empty_with_one_warning_on_every_tool_as_the_loader_makes_it(
        self, tmp_path, monkeypatch, caplog
    ):
        # G3: terminal_bench_aai's shape, `TEST_DIR=${TEST_DIR}` next to a Dockerfile that needs a build, with the
        # variable not exported, plus one nobody sets. docker compose builds it with a warning and an empty string;
        # the direct build and the loader resolve the same file the same way.
        for name in ("TEST_DIR", "FOO_UNSET", "COMPOSE_PROJECT_NAME"):
            monkeypatch.delenv(name, raising=False)
        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    environment:\n      - TEST_DIR=${TEST_DIR}\n"
            "      - PROJECT=${COMPOSE_PROJECT_NAME}\n    build:\n      args:\n        FOO: ${FOO_UNSET}\n"
            "        BAR: ${FOO_UNSET}\n"
        )
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        write_task(folder / "b", extra={"docker-compose.yaml": compose})
        (entry,) = plan_builds(folder)
        env = compose_build_env(entry, "hb__x", None)
        assert env["TEST_DIR"] == "/tests" and env["COMPOSE_PROJECT_NAME"] == f"hb__{entry.digest}"
        with caplog.at_level("WARNING", logger="nemo_gym.tasks.harbor.image_build"):
            document = validate_task_compose(entry.context, env, tasks=entry.tasks)
            built = settled(entry)
        main = document["services"]["main"]
        assert main["environment"] == ["TEST_DIR=/tests", f"PROJECT=hb__{entry.digest}"]
        assert main["build"]["args"] == {"FOO": "", "BAR": ""} and built.build_args == {"FOO": "", "BAR": ""}
        # One warning per context and variable, however often the file reads it or the build interpolates it.
        assert [r.getMessage() for r in caplog.records] == [
            f"{entry.context / 'docker-compose.yaml'}: Compose variable ${{FOO_UNSET}} has no value in the build's "
            "environment (the process environment, the task's [environment].env and Harbor's variables) or "
            "environment/.env for task(s) a, b; it is an empty string, as docker compose makes it"
        ]
        buildah, fake = tagged_buildah(tmp_path)
        assert buildah.build(entry).tasks == ("a", "b")
        bud = fake.calls[0]
        assert bud[bud.index("--build-arg") :][:4] == ["--build-arg", "FOO=", "--build-arg", "BAR="]
        docker, fake = local_docker()
        assert docker.build(entry).tool == "docker" and fake.calls[1][-1] == "build"
        assert fake.envs[1]["TEST_DIR"] == "/tests"
        # The same file with a required variable is still refused, naming the tasks.
        (entry.context / "docker-compose.yaml").write_text(compose.replace("${FOO_UNSET}", "${FOO_UNSET:?need it}"))
        with pytest.raises(BuildError, match=r"of task\(s\) a, b: \$\{FOO_UNSET\} is required: need it"):
            buildah.build(entry)

    def test_the_build_and_the_loader_define_the_same_variables_for_a_task(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
        write_task(
            folder / "a", toml=TOML + '\n[environment.env]\nMODE = "strict"\n', extra={"docker-compose.yaml": compose}
        )
        builder, _ = local_docker()
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        (entry,) = plan_builds(folder)
        build_env = compose_build_env(entry, result.tag, None)
        # The loader keeps a literal dollar doubled for the document's checks; the build hands values to a process.
        loader = {key: value.replace("$$", "$") for key, value in compose_variables(load_task(folder / "a")).items()}
        common = sorted(set(loader) & set(build_env))
        assert {"MAIN_IMAGE_NAME", "CONTEXT_DIR", "TEST_DIR", "COMPOSE_PROJECT_NAME", "CPUS", "MEMORY", "MODE"} <= set(
            common
        )
        assert {key: loader[key] for key in common} == {key: build_env[key] for key in common}

    def test_the_direct_build_validates_the_compose_file_before_any_command(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ./db\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        # A context the loader refuses is never built and pushed under the hb__ name.
        with pytest.raises(BuildError, match="service 'db' has a build section"):
            build_images(folder, builder=builder, log=lambda _: None)
        assert fake.calls == [] and not (folder / BUILT_IMAGES_FILE).exists()

    def test_dockerfile_inline_is_refused_on_the_direct_path_with_the_docker_remedy(self, tmp_path):
        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    build:\n      context: .\n      dockerfile_inline: |\n        FROM scratch\n"
        )
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        (entry,) = plan_builds(folder)
        assert settled(entry).dockerfile_inline is True
        fake = FakeTool("podman")
        builder = ImageBuilder(tool="podman", repository=REPO, run=fake)
        with pytest.raises(
            BuildError, match=r"dockerfile_inline is not supported by the direct podman build.*builds\.tool: docker"
        ):
            builder.build(entry)
        assert fake.calls == []

    def test_a_missing_main_build_dockerfile_is_a_clear_error(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    build:\n      dockerfile: Dockerfile.agent\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        (entry,) = plan_builds(folder)
        builder, fake = tagged_buildah(tmp_path)
        with pytest.raises(BuildError, match=r"environment has no Dockerfile\.agent"):
            builder.build(entry)
        assert fake.calls == []


class TestLoaderBuildMode:
    def test_a_built_image_is_picked_up_in_pull_mode(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, _ = tagged_buildah(tmp_path)
        (result,) = build_images(folder, builder=builder, log=lambda _: None)
        task = load_task(folder / "a")
        assert task.image == result.tag
        assert task.overlay == ()
        # WORKDIR, ENV and USER come from the built image's recorded configuration.
        assert task.workdir == "/app" and task.user == "app" and task.sandbox_user == "app"
        assert task.env == {"PATH": "/usr/bin:/bin", "APP_HOME": "/app"}
        assert task.needs_sandbox

    def test_task_toml_settings_win_over_the_built_image(self, tmp_path):
        folder = tmp_path / "ds"
        toml = TOML + 'workdir = "/work"\n\n[environment.env]\nAPP_HOME = "/elsewhere"\n'
        write_task(folder / "a", toml=toml)
        builder, _ = tagged_buildah(tmp_path)
        build_images(folder, builder=builder, log=lambda _: None)
        task = load_task(folder / "a")
        assert task.workdir == "/work" and task.env["APP_HOME"] == "/elsewhere"

    def test_without_a_recorded_build_the_error_says_how_to_build(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        with pytest.raises(HarborTaskError, match=r"needs a build \(COPY\).*built-images.json.*`builds:`") as info:
            load_task(folder / "a")
        assert "docker_image" in str(info.value)

    def test_a_changed_context_needs_a_new_build(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a")
        builder, _ = tagged_buildah(tmp_path)
        build_images(folder, builder=builder, log=lambda _: None)
        (folder / "a" / "environment" / "tools" / "setup.sh").write_text("#!/bin/sh\necho changed\n")
        with pytest.raises(HarborTaskError, match="needs a build"):
            load_task(folder / "a")


@pytest.mark.skipif(shutil.which("buildah") is None, reason="buildah is not installed")
def test_buildah_builds_a_real_image_rootless(tmp_path):
    folder = tmp_path / "ds"
    write_task(
        folder / "a",
        dockerfile="FROM docker.io/library/alpine:3.20\nCOPY note.txt /note.txt\nWORKDIR /work\nENV GREETING=hi\n",
        extra={"note.txt": "hello\n"},
    )
    # The host's own platform, so the build needs no emulation; the tagged repository is never pushed.
    host = "linux/arm64" if platform.machine() in ("aarch64", "arm64") else "linux/amd64"
    builder = ImageBuilder(
        tool="buildah",
        repository="localhost/nemo-gym-test",
        push=False,
        platform=host,
        storage_root=tmp_path / "store",
    )
    (result,) = build_images(folder, builder=builder, log=lambda _: None)
    assert result.tag == f"localhost/nemo-gym-test/hb__{result.digest}" and result.pushed is False
    record = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[result.image]
    assert f"{record['os']}/{record['architecture']}" == host
    assert record["config"]["WorkingDir"] == "/work"
    assert "GREETING=hi" in record["config"]["Env"]
    assert record["config_digest"].startswith("sha256:")
    task = load_task(folder / "a")
    assert task.image == result.image and task.workdir == "/work" and task.env["GREETING"] == "hi"


class TestPrepareBuilds:
    """``prepare_target`` builds what needs building before the tasks load."""

    @pytest.fixture
    def fake_builder(self, monkeypatch):
        import nemo_gym.tasks.harbor.cli as harbor_cli

        made: list[dict] = []

        def factory(**kwargs):
            made.append(dict(kwargs))
            tool = kwargs.pop("tool", None) or "docker"
            return ImageBuilder(tool=tool, run=FakeTool(tool), **kwargs)

        monkeypatch.setattr(harbor_cli, "ImageBuilder", factory)
        return made

    def test_prepare_builds_once_and_the_tasks_load_in_pull_mode(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.cli import BuildOptions, prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "b")
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="podman", repository="reg.example/r")
        )
        assert fake_builder == [{"tool": "podman", "repository": "reg.example/r", "platform": None}]
        assert [t.task_id for t in prepared.tasks] == ["a", "b", "c"]
        built = next(t for t in prepared.tasks if t.task_id == "a")
        # One build for the two identical contexts, under Harbor's name for it, pinned by its pushed digest.
        image = f"reg.example/r/hb__{harbor_hash(folder / 'a' / 'environment')}@sha256:pushed01"
        assert built.image == image and built.overlay == ()
        assert next(t for t in prepared.tasks if t.task_id == "b").image == image
        assert prepared.images == (image, "ubuntu:24.04")
        assert json.loads((tmp_path / "out" / "ds" / "images.json").read_text()) == list(prepared.images)
        out = capsys.readouterr().out
        assert f"Built {image} (pushed, podman) for task(s) a, b" in out
        assert "Built 1 image(s) with podman, pushed 1" in out
        assert "Images: 2 distinct" in out
        # A second prepare finds the build recorded and builds nothing.
        fake_builder.clear()
        prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="podman", repository="reg.example/r")
        )
        assert "Built image recorded for a, b" in capsys.readouterr().out

    def test_builds_disabled_leaves_the_tasks_out_with_the_loader_message(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.cli import BuildOptions, prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert fake_builder == []
        assert [t.task_id for t in prepared.tasks] == ["c"]
        assert "needs a build (COPY)" in prepared.skipped["a"]
        assert (
            "Skipping Dockerfile builds (builds.enabled is false); tasks that need one are left out: a"
            in capsys.readouterr().out
        )
        # Once the build is recorded, a disabled build says nothing and the task runs from the recorded image.
        build_images(folder, builder=local_docker()[0], log=lambda _: None)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["a", "c"]
        assert "Skipping Dockerfile builds" not in capsys.readouterr().out

    def test_a_failed_build_stops_prepare_naming_the_tasks(self, tmp_path, monkeypatch):
        import nemo_gym.tasks.harbor.cli as harbor_cli
        from nemo_gym.tasks.harbor.cli import prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        monkeypatch.setattr(
            harbor_cli,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="docker", run=FakeTool("docker", fail=str(folder / "a" / "environment"))),
        )
        with pytest.raises(ValueError, match=r"Could not build the Dockerfile image\(s\) for task\(s\) a: Building"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert not (folder / BUILT_IMAGES_FILE).exists()


class TestBuildOptions:
    """The ``builds:`` configuration block, read from a merged config dict and from the command line's sources."""

    def test_from_config_defaults_and_environment(self, monkeypatch):
        from nemo_gym.tasks.harbor.cli import BuildOptions

        monkeypatch.delenv("NEMO_GYM_BUILD_REPOSITORY", raising=False)
        assert BuildOptions.from_config(None) == BuildOptions()
        assert BuildOptions.from_config({}) == BuildOptions(enabled=True, tool=None, repository=None, platform=None)
        assert BuildOptions.from_config({"builds": None}) == BuildOptions()
        # The repository falls back to the environment variable; an explicit setting wins over it.
        monkeypatch.setenv("NEMO_GYM_BUILD_REPOSITORY", "env.example/repo")
        assert BuildOptions.from_config({"other": 1}).repository == "env.example/repo"
        assert BuildOptions.from_config({"builds": {"repository": "cfg.example/r"}}).repository == "cfg.example/r"
        assert BuildOptions.from_config({"builds": {"repository": ""}}).repository == "env.example/repo"

    def test_from_config_reads_an_env_yaml_style_block(self):
        from nemo_gym.tasks.harbor.cli import BuildOptions

        block = {
            "builds": {"tool": "buildah", "repository": "flag.example/r", "platform": "linux/amd64", "enabled": False}
        }
        assert BuildOptions.from_config(block) == BuildOptions(
            enabled=False, tool="buildah", repository="flag.example/r", platform="linux/amd64"
        )

    def test_from_config_refuses_unknown_settings_and_wrong_types(self):
        from nemo_gym.tasks.harbor.cli import BuildOptions

        with pytest.raises(ValueError, match="Unknown `builds` setting\\(s\\) rebuild; expected tool, repository"):
            BuildOptions.from_config({"builds": {"rebuild": True}})
        with pytest.raises(ValueError, match="`builds.tool` must be one of buildah, podman, docker; got 'kaniko'"):
            BuildOptions.from_config({"builds": {"tool": "kaniko"}})
        with pytest.raises(ValueError, match="`builds.enabled` must be true or false; got 'no'"):
            BuildOptions.from_config({"builds": {"enabled": "no"}})
        with pytest.raises(ValueError, match="`builds.platform` must be a string; got 1"):
            BuildOptions.from_config({"builds": {"platform": 1}})
        with pytest.raises(ValueError, match="`builds` must be a mapping of tool, repository, platform, enabled"):
            BuildOptions.from_config({"builds": "docker"})

    @pytest.fixture
    def sources(self, tmp_path, monkeypatch):
        """An env.yaml of its own (never the repository's) and a --config file, both with a ``builds:`` block."""
        import nemo_gym.tasks.harbor.cli as harbor_cli

        monkeypatch.delenv("NEMO_GYM_BUILD_REPOSITORY", raising=False)
        dotenv = tmp_path / "env.yaml"
        dotenv.write_text("builds:\n  tool: podman\n  repository: env.example/r\n")
        monkeypatch.setattr(harbor_cli, "_resolve_under_cwd_or_install", lambda rel: tmp_path / rel)
        config = tmp_path / "run.yaml"
        config.write_text("builds:\n  tool: buildah\n  platform: linux/arm64\n")
        return dotenv, config

    def test_overrides_merge_the_command_line_sources_in_the_global_config_order(self, sources):
        from nemo_gym.tasks.harbor.cli import BuildOptions, build_options_from_overrides

        dotenv, config = sources
        # env.yaml alone: the block applies over the defaults.
        assert build_options_from_overrides([]) == BuildOptions(tool="podman", repository="env.example/r")
        # A --config file sits below env.yaml, which sits below the ++ overrides.
        tokens = [f'+config_paths=["{config}"]', "+agent_name=a"]
        assert build_options_from_overrides(tokens) == BuildOptions(
            tool="podman", repository="env.example/r", platform="linux/arm64"
        )
        assert build_options_from_overrides([*tokens, "++builds.tool=docker", "++builds.enabled=false"]) == (
            BuildOptions(enabled=False, tool="docker", repository="env.example/r", platform="linux/arm64")
        )
        # A bad value is refused the same way, whichever source it comes from.
        with pytest.raises(ValueError, match="`builds.tool` must be one of"):
            build_options_from_overrides(["++builds.tool=kaniko"])

    def test_overrides_leave_argv_alone_and_need_no_block(self, sources, monkeypatch):
        import sys

        from nemo_gym.tasks.harbor.cli import BuildOptions, build_options_from_overrides

        dotenv, _ = sources
        dotenv.write_text("some_key: 1\n")
        monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "x"])
        assert build_options_from_overrides(["+split=validation"]) == BuildOptions()
        assert sys.argv == ["gym", "eval", "run", "x"]

    def test_run_and_validate_read_the_block_before_preparing(self, tmp_path, monkeypatch, sources):
        """Both entry points hand the block of the command line's configuration to prepare."""
        import argparse

        import nemo_gym.tasks.harbor.cli as harbor_cli

        seen: list[harbor_cli.BuildOptions] = []

        def fake_prepare(target, **kwargs):
            seen.append(kwargs["build"])
            raise harbor_cli.HubError("stop here")

        def fail(message):
            raise SystemExit(message)

        monkeypatch.setattr(harbor_cli, "prepare_target", fake_prepare)
        monkeypatch.setattr(harbor_cli, "resolve_agent", lambda name: None)
        args = argparse.Namespace(
            target=str(tmp_path), agent="a", sandbox=None, _parser=argparse.Namespace(error=fail)
        )
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.run_target(args, ["++builds.enabled=false"])
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.validate_target(args, ["++builds.platform=linux/arm64"])
        assert seen == [
            harbor_cli.BuildOptions(enabled=False, tool="podman", repository="env.example/r"),
            harbor_cli.BuildOptions(tool="podman", repository="env.example/r", platform="linux/arm64"),
        ]


# -- D2: concurrent recording ------------------------------------------------------------------


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
                "nemo_gym.tasks.harbor.task.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk"))
            )
            with pytest.raises(OSError, match="disk"):
                record_built_image(folder, later)
            assert (folder / BUILT_IMAGES_FILE).read_text() == before
            assert not [p for p in folder.iterdir() if p.name.endswith(".tmp")]
        finally:
            os.umask(previous)


class TestMainImageSpellings:
    """D4: main's image is accepted when it resolves to MAIN_IMAGE_NAME, whichever interpolation spells it."""

    fake_docker = TestDockerComposeBackend.fake_docker

    @pytest.mark.parametrize(
        "spelling",
        ["${MAIN_IMAGE_NAME}", "$MAIN_IMAGE_NAME", "${MAIN_IMAGE_NAME:-fallback}", "${MAIN_IMAGE_NAME-fallback}"],
    )
    def test_an_interpolation_resolving_to_the_main_image_passes(self, tmp_path, spelling):
        folder = tmp_path / "ds"
        compose = f"services:\n  main:\n    image: {spelling}\n    build:\n      target: slim\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        context = folder / "a" / "environment"
        document = validate_task_compose(context, {"MAIN_IMAGE_NAME": "hb__0123", "CONTEXT_DIR": str(context)})
        assert document["services"]["main"]["image"] == "hb__0123"

    @pytest.mark.parametrize(
        ("image", "tail"),
        [
            ("mine:1", r"not 'mine:1'$"),
            ("${OTHER:-mine:1}", r"not '\$\{OTHER:-mine:1\}' \(it resolves to 'mine:1'\)$"),
            ("${MAIN_IMAGE_NAME:+other}", r"\(it resolves to 'other'\)$"),
            ("${UNSET_VARIABLE}", r"not '\$\{UNSET_VARIABLE\}'$"),
        ],
    )
    def test_anything_else_is_refused_with_what_it_resolves_to(self, tmp_path, image, tail):
        folder = tmp_path / "ds"
        compose = f"services:\n  main:\n    image: {image}\n    build:\n      target: slim\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        context = folder / "a" / "environment"
        with pytest.raises(BuildError, match=r"services\.main\.image must be \$\{MAIN_IMAGE_NAME\}.*" + tail):
            validate_task_compose(context, {"MAIN_IMAGE_NAME": "hb__0123", "CONTEXT_DIR": str(context)})

    def test_a_default_spelling_builds_under_the_recorded_name(self, tmp_path, fake_docker):
        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME:-local-dev}\n    build:\n      context: ${CONTEXT_DIR}\n"
        )
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "tools/setup.sh": "echo ok\n"})
        (result,) = build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        build = fake_docker()[1]
        assert build["argv"][-1] == "build" and build["env"]["MAIN_IMAGE_NAME"] == result.tag
        assert read_built_images(folder)[result.digest]["image"] == result.tag
