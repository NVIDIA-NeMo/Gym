# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of ImageBuilder.

Per-tool argv, the generated containers.conf and registries.conf environment, the direct buildah
and podman path, the compose plugin and heredoc preflights, and the rootless buildah build that
runs when the tool is installed.
"""

import json
import logging
import os
import platform
import re
import shutil
from pathlib import Path

import pytest

from nemo_gym.data import BUILT_IMAGES_FILE, RECORD_VERSION, load_compose_document, load_task
from nemo_gym.data.tasks.harbor.build_compose import COMPOSE_BUILD_FILE, compose_build_env, validate_task_compose
from nemo_gym.data.tasks.harbor.build_plan import BuildError, BuildPlanEntry, BuildResult, plan_builds
from nemo_gym.data.tasks.harbor.build_records import record_built_image
from nemo_gym.data.tasks.harbor.build_tools import HTTP_403, ImageBuilder, normalize_inspect
from nemo_gym.data.tasks.harbor.compose_variables import compose_variables
from nemo_gym.data.tasks.harbor.digest import context_digest
from nemo_gym.data.tasks.harbor.dockerfile import dockerfile_has_heredoc
from nemo_gym.data.tasks.harbor.image_build import build_images
from nemo_gym.data.tasks.harbor.records import IMAGE_CONFIGS_FILE
from nemo_gym.data.tasks.harbor.task import built_service_images
from tests.unit_tests.harbor_build_helpers import (
    ARM_BUILDAH_INSPECT,
    BUILD_DOCKERFILE,
    BUILDAH_INSPECT,
    DOCKER_INSPECT,
    HEREDOC_DOCKERFILE,
    PROBE_TAG,
    REPO,
    TOML,
    FakeTool,
    HeredocTool,
    harbor_hash,
    local_docker,
    main_record,
    settled,
    tagged_buildah,
    task_inputs_digest,
    write_task,
)


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
        assert built == {result.digest: main_record(result.tag, ["a"], "docker", False)}
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

    def test_tag_for_names_a_sidecar_as_compose_names_it_under_the_project(self, tmp_path):
        digest = "0123456789abcdef0123456789abcdef"  # pragma: allowlist secret
        local = ImageBuilder(tool="docker", run=FakeTool("docker"))
        pushed = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path, run=FakeTool("buildah"))
        world = BuildPlanEntry(digest, tmp_path / "ctx", ("a",), service="world")
        # Compose's <project>-<service> under the project name Gym passes, hb__<hash>; one flat name per image.
        assert local.tag_for(world) == f"hb__{digest}-world"
        assert pushed.tag_for(world) == f"{REPO}/hb__{digest}-world" == pushed.tag_name(digest, "world")
        assert pushed.tag_name(digest) == f"{REPO}/hb__{digest}"
        # An odd service name is sanitised like Harbor sanitises image names.
        odd = BuildPlanEntry(digest, tmp_path / "ctx", ("a",), service="Legacy App@2")
        assert local.tag_for(odd) == f"hb__{digest}-legacy-app-2"

    @pytest.mark.parametrize("tool", ["docker", "buildah"])
    def test_services_that_differ_by_case_only_are_refused_before_any_build(self, tmp_path, tool):
        """Review (M24): ``Db`` and ``db`` are both valid Compose service names and both build as ``hb__<digest>-db``
        once the name is lower-cased, so the second build silently replaced the first and both records named one image
        (on the compose path the images override replaced their distinct explicit ``image:`` values too). The collision
        is refused before any command, naming both services and the shared tag, on either tool path."""
        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  Db:\n    image: org/primary:1\n    build: ./primary\n"
            "  db:\n    image: org/replica:1\n    build: ./replica\n"
        )
        write_task(
            folder / "a",
            extra={
                "docker-compose.yaml": compose,
                "tools/setup.sh": "#!/bin/sh\necho ok\n",
                "primary/Dockerfile": "FROM postgres:16\n",
                "replica/Dockerfile": "FROM postgres:16\nENV REPLICA=1\n",
            },
        )
        builder, runner = local_docker() if tool == "docker" else tagged_buildah(tmp_path)
        plan = plan_builds(folder)
        assert [entry.service for entry in plan] == ["main", "Db", "db"]
        tag = builder.tag_name(plan[0].digest, "db")
        assert builder.tag_for(plan[1]) == builder.tag_for(plan[2]) == tag
        with pytest.raises(BuildError) as info:
            list(builder.build_many(plan))
        message = str(info.value)
        assert message.startswith(f"{folder / 'a' / 'environment' / 'docker-compose.yaml'} of task(s) a declares ")
        assert "declares services 'Db' and 'db', which both build as image " + tag in message
        assert message.endswith("; image names are lower-case, so rename one")
        # Nothing was built or pushed under any hb__ name; the docker path ran only its compose plugin check.
        assert [call for call in runner.calls if any("hb__" in arg for arg in call)] == []
        assert not any("build" in call or "bud" in call or "push" in call for call in runner.calls)

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
        assert built == {result.digest: main_record(result.image, ["a"], "buildah", True)}
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
            "inputs_digest": task_inputs_digest(folder / "a", repository=REPO),
            "os": "linux",
            "record_version": RECORD_VERSION,
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
        assert built[first.digest] == main_record(second.image, ["a"], "podman", True)
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
        from nemo_gym.data.tasks.harbor.build_records import platform_key, recorded_image_mismatch

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
        monkeypatch.setenv("https_proxy", "http://user:hunter2@proxy.example:3128")  # pragma: allowlist secret
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

    def test_malformed_build_args_are_refused_before_any_command_and_the_next_task_still_builds(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    build:\n      args: A=1\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        write_task(folder / "b", dockerfile=BUILD_DOCKERFILE + "ENV B=1\n")
        # Planning reads no Compose file, so one task's section cannot stop the dataset; its build refuses it.
        bad, good = plan_builds(folder)
        builder, fake = tagged_buildah(tmp_path)
        with pytest.raises(BuildError, match=r"build\.args is refused.*mapping or a list") as info:
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
        with caplog.at_level("WARNING", logger="nemo_gym.data.tasks.harbor.build_compose"):
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
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build:\n      context: ./db\n"
            "      network: host\n"
        )
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        # A context the docker path refuses is refused here too, main included: nothing is built or pushed.
        with pytest.raises(BuildError, match=r"services\.db\.build\.network is refused"):
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

    # -- G2: sidecars, one build per service -------------------------------------------------------

    SIDECAR_FILES = {
        "docker-compose.yaml": (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  world:\n    build:\n      context: .\n      dockerfile: world/Dockerfile\n"
            "  db:\n    image: ${MAIN_IMAGE_NAME}_db\n    build:\n      context: ./db\n      args:\n        V: '1'\n"
            "      no_cache: true\n"
        ),
        "world/Dockerfile": "FROM python:3.12-slim\nCOPY world/ /opt/world/\n",
        "db/Dockerfile": "FROM postgres:16\nARG V\n",
        "tools/setup.sh": "#!/bin/sh\necho ok\n",
    }

    def test_sidecars_build_one_per_service_with_their_own_context_and_dockerfile(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra=self.SIDECAR_FILES)
        context = (folder / "a" / "environment").resolve()
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        main, world, db = build_images(folder, builder=builder, log=lambda _: None)
        assert [r.service for r in (main, world, db)] == ["main", "world", "db"]
        assert world.tag == f"{REPO}/hb__{main.digest}-world" and db.tag == f"{REPO}/hb__{main.digest}-db"
        assert all(r.image == f"{r.tag}@sha256:pushed01" and r.pushed for r in (main, world, db))
        # buildah: build, inspect, push per service, in order; each service's context and Dockerfile.
        verbs = [next(a for a in call if a in ("bud", "inspect", "push")) for call in fake.calls]
        assert verbs == ["bud", "inspect", "push"] * 3
        builds = [call for call in fake.calls if "bud" in call]
        assert builds[0][-3:] == ["-f", str(context / "Dockerfile"), str(context)]
        # counselbench's world: context `.`, dockerfile under a subfolder.
        assert builds[1][-3:] == ["-f", str(context / "world" / "Dockerfile"), str(context)]
        # A sidecar with its own context, its build args applied; no_cache is ignored on the direct path.
        assert builds[2][-3:] == ["-f", str(context / "db" / "Dockerfile"), str(context / "db")]
        assert builds[2][builds[2].index("--build-arg") + 1] == "V=1" and "--no-cache" not in builds[2]
        assert all("--format" in call and call[call.index("--format") + 1] == "docker" for call in builds)
        # One merged record per context; compose-images.json holds every image's configuration.
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {
            main.digest: {
                "image": main.image,
                "tasks": ["a"],
                "tool": "buildah",
                "pushed": True,
                "services": {
                    "main": {"image": main.image, "tool": "buildah", "pushed": True},
                    "world": {"image": world.image, "tool": "buildah", "pushed": True},
                    "db": {"image": db.image, "tool": "buildah", "pushed": True},
                },
            }
        }
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert {main.image, world.image, db.image} <= set(configs)
        assert configs[world.image]["config"]["Cmd"] == ["/bin/sh"] and configs[world.image]["image"] == world.image
        # The loader: main runs from its image, the sidecars are substituted by service name.
        task = load_task(folder / "a")
        assert task.image == main.image
        assert built_service_images(folder / "a") == {"main": main.image, "world": world.image, "db": db.image}
        # Nothing to do the second time, each service reported.
        lines: list[str] = []
        assert build_images(folder, builder=builder, log=lines.append) == []
        assert lines == [
            f"Built image recorded for a: {main.image}",
            f"Built image recorded for a (world): {world.image}",
            f"Built image recorded for a (db): {db.image}",
        ]

    def test_a_failure_on_the_second_service_leaves_the_first_recorded(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra=self.SIDECAR_FILES)
        context = (folder / "a" / "environment").resolve()
        # world's build fails: its bud line ends with the context, `.`, the same as main's, so fail on the Dockerfile.
        fake = FakeTool("buildah", fail=str(context / "world" / "Dockerfile"))
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        with pytest.raises(BuildError, match=rf"Building {REPO}/hb__[0-9a-f]+-world for task\(s\) a failed"):
            build_images(folder, builder=builder, log=lambda _: None)
        # main was built, pushed and recorded before world failed; db was never started.
        (entry,) = json.loads((folder / BUILT_IMAGES_FILE).read_text()).values()
        assert set(entry["services"]) == {"main"} and entry["image"] == entry["services"]["main"]["image"]
        assert not any(str(context / "db") in call for call in fake.calls)
        # The next run builds the two missing services only.
        good = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=good)
        lines: list[str] = []
        world, db = build_images(folder, builder=builder, log=lines.append)
        assert (world.service, db.service) == ("world", "db")
        assert lines[0].startswith("Built image recorded for a: ") and len([c for c in good.calls if "bud" in c]) == 2
        assert set(json.loads((folder / BUILT_IMAGES_FILE).read_text())[world.digest]["services"]) == {
            "main",
            "world",
            "db",
        }

    def test_a_sidecar_left_local_by_an_interrupted_pushing_run_is_rebuilt_not_vouched_for_by_main(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra=self.SIDECAR_FILES)
        context = (folder / "a" / "environment").resolve()
        # 1. A local docker prepare (iterating on a laptop): main, world and db recorded as local tags.
        main, world, db = build_images(folder, builder=local_docker()[0], log=lambda _: None)
        assert not main.pushed and world.image == f"hb__{main.digest}-world"
        # 2. The cluster prepare (buildah, repository) rebuilds everything, pushes main, then fails on world.
        failing = FakeTool("buildah", fail=str(context / "world" / "Dockerfile"))
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=failing)
        lines: list[str] = []
        with pytest.raises(BuildError, match=r"-world for task\(s\) a failed"):
            build_images(folder, builder=builder, log=lines.append)
        local = "it was built locally and this run pushes to a repository"
        assert lines[:3] == [
            f"Rebuilding {main.image} for a: {local}",
            f"Rebuilding {world.image} for a (world): {local}",
            f"Rebuilding {db.image} for a (db): {local}",
        ]
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[main.digest]
        pushed_main = entry["services"]["main"]
        assert pushed_main["pushed"] is True and pushed_main["image"].startswith(f"{REPO}/hb__{main.digest}@sha256:")
        # world's record is still the local docker tag, flagged as such: main's push does not vouch for it.
        assert entry["services"]["world"] == {"image": world.image, "tool": "docker", "pushed": False}
        assert entry["image"] == pushed_main["image"] and entry["pushed"] is True and entry["tool"] == "buildah"
        # 3. The retry: main's pushed record fits, the local world and db are rebuilt and pushed, not reused.
        good = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=good)
        lines.clear()
        assert [r.service for r in build_images(folder, builder=builder, log=lines.append)] == ["world", "db"]
        assert lines[0] == f"Built image recorded for a: {pushed_main['image']}"
        assert lines[1] == f"Rebuilding {world.image} for a (world): {local}"
        assert len([c for c in good.calls if "bud" in c]) == 2
        entry = json.loads((folder / BUILT_IMAGES_FILE).read_text())[main.digest]
        assert all(record["pushed"] and record["tool"] == "buildah" for record in entry["services"].values())
        # The seed document carries the pushed sidecar, the one OpenSandbox can pull.
        document = load_compose_document(load_task(folder / "a"))
        assert document["services"]["world"]["image"].startswith(f"{REPO}/hb__{main.digest}-world@sha256:")
        assert build_images(folder, builder=builder, log=lambda _: None) == []

    def test_a_platform_mismatch_on_a_sidecar_names_the_service(self, tmp_path):
        folder = tmp_path / "ds"
        write_task(folder / "a", extra=self.SIDECAR_FILES)

        class ArmWorld(FakeTool):
            def __call__(self, argv, env=None):
                self.inspect = ARM_BUILDAH_INSPECT if any(a.endswith("-world") for a in argv) else None
                return super().__call__(argv, env)

        fake = ArmWorld("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        with pytest.raises(
            BuildError, match=r"-world for task\(s\) a \(service world\): the image came out as linux/arm64"
        ):
            build_images(folder, builder=builder, log=lambda _: None)
        assert not any("push" in call and call[-1].endswith("-world") for call in fake.calls)
        assert set(json.loads((folder / BUILT_IMAGES_FILE).read_text()).popitem()[1]["services"]) == {"main"}

    @pytest.mark.parametrize(
        ("section", "message"),
        [
            (
                "      dockerfile_inline: |\n        FROM scratch\n",
                r"services\.db\.build\.dockerfile_inline is not supported by the direct podman build",
            ),
            (
                "      additional_contexts:\n        base: service:main\n",
                r"services\.db\.build\.additional_contexts is not supported by the direct podman build",
            ),
        ],
        ids=["dockerfile_inline", "additional_contexts"],
    )
    def test_compose_only_build_keys_on_a_sidecar_are_refused_on_the_direct_path(self, tmp_path, section, message):
        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build:\n      context: ./db\n" + section
        )
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        fake = FakeTool("podman")
        builder = ImageBuilder(tool="podman", repository=REPO, run=fake)
        with pytest.raises(BuildError, match=message + r".*builds\.tool: docker"):
            build_images(folder, builder=builder, log=lambda _: None)
        # Knowable from the checked file: nothing is built, pushed or recorded for the context, main included.
        assert fake.calls == []
        assert not (folder / BUILT_IMAGES_FILE).is_file()

    def test_a_missing_sidecar_dockerfile_is_a_clear_error_naming_the_service(self, tmp_path):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ./db\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose})
        builder, fake = tagged_buildah(tmp_path)
        with pytest.raises(BuildError, match=r"environment/db has no Dockerfile for services\.db\.build"):
            build_images(folder, builder=builder, log=lambda _: None)
        assert fake.calls == []


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
