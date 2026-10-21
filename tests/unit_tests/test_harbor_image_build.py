# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dockerfile builds (image mode three): planning, building through a tool, recording, and the loader."""

import json

import pytest

import nemo_gym.tasks.harbor.build_options as build_options
import nemo_gym.tasks.harbor.prepare_builds as prepare_builds
from nemo_gym.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_build import build_images
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE
from nemo_gym.tasks.harbor.task import load_task
from tests.unit_tests.harbor_build_helpers import (
    PULL_DOCKERFILE,
    TOML,
    FakeTool,
    harbor_hash,
    local_docker,
    tagged_buildah,
    task_inputs_digest,
    write_base_record,
    write_task,
)


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


class TestPrepareBuilds:
    """``prepare_target`` builds what needs building before the tasks load."""

    @pytest.fixture
    def fake_builder(self, monkeypatch):
        made: list[dict] = []

        def factory(**kwargs):
            made.append(dict(kwargs))
            tool = kwargs.pop("tool", None) or "docker"
            return ImageBuilder(tool=tool, run=FakeTool(tool), **kwargs)

        monkeypatch.setattr(prepare_builds, "ImageBuilder", factory)
        return made

    def test_prepare_builds_once_and_the_tasks_load_in_pull_mode(self, tmp_path, fake_builder, capsys):
        from nemo_gym.tasks.harbor.build_options import BuildOptions
        from nemo_gym.tasks.harbor.cli import prepare_target

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
        from nemo_gym.tasks.harbor.build_options import BuildOptions
        from nemo_gym.tasks.harbor.cli import prepare_target

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

    def test_builds_disabled_leaves_out_a_record_that_does_not_fit_the_run(self, tmp_path, fake_builder, capsys):
        """A record of another platform is no better than none: with builds disabled its tasks are left out with
        the reason, as the enabled path would rebuild it, and no row references the image."""
        from nemo_gym.tasks.harbor.build_options import BuildOptions
        from nemo_gym.tasks.harbor.cli import prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        arm_image = "reg.example/r/hb__arm@sha256:" + "a" * 64
        digest = harbor_hash(folder / "a" / "environment")
        (folder / BUILT_IMAGES_FILE).write_text(
            json.dumps({digest: {"image": arm_image, "tasks": ["a"], "tool": "buildah", "pushed": True}})
        )
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        configs[arm_image] = {
            "os": "linux",
            "architecture": "arm64",
            "image": arm_image,
            "config": {},
            "record_version": RECORD_VERSION,
            "inputs_digest": task_inputs_digest(folder / "a", platform="linux/amd64"),
        }
        (folder / IMAGE_CONFIGS_FILE).write_text(json.dumps(configs))

        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False, platform="linux/amd64")
        )
        assert fake_builder == []
        assert [t.task_id for t in prepared.tasks] == ["c"]
        assert prepared.skipped["a"] == (
            f"recorded image {arm_image} cannot be used: it is linux/arm64, not linux/amd64; builds are disabled"
        )
        assert arm_image not in (tmp_path / "out" / "ds" / "tasks.jsonl").read_text()
        assert arm_image not in prepared.images
        assert f"Skipping a: recorded image {arm_image} cannot be used: it is linux/arm64" in capsys.readouterr().out
        # The same record serves a run that asks for its platform.
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False, platform="linux/arm64")
        )
        assert [t.task_id for t in prepared.tasks] == ["a", "c"]

    def test_recorded_images_need_no_build_tool(self, tmp_path, monkeypatch, capsys):
        """A host with every context recorded prepares without docker, podman or buildah; one with a context left
        to build still needs one, and the error says how to run with recorded images only."""
        import nemo_gym.tasks.harbor.build_tools as image_build_module
        from nemo_gym.tasks.harbor.build_options import BuildOptions
        from nemo_gym.tasks.harbor.cli import prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        write_base_record(folder)
        build_images(folder, builder=local_docker()[0], log=lambda _: None)
        monkeypatch.setattr(image_build_module, "detect_tool", lambda repository=None: None)

        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions())
        assert [t.task_id for t in prepared.tasks] == ["a"] and prepared.skipped == {}
        assert "Built image recorded for a: " in capsys.readouterr().out

        write_task(folder / "b", extra={"tools/setup.sh": "#!/bin/sh\necho other\n"})
        with pytest.raises(ValueError, match=r"for task\(s\) b: No image builder found.*builds\.enabled to false"):
            prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions())

    def test_a_failed_build_stops_prepare_naming_the_tasks(self, tmp_path, monkeypatch):
        from nemo_gym.tasks.harbor.cli import prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "a")
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="docker", run=FakeTool("docker", fail=str(folder / "a" / "environment"))),
        )
        with pytest.raises(ValueError, match=r"Could not build the Dockerfile image\(s\) for task\(s\) a: Building"):
            prepare_target(str(folder), output_root=tmp_path / "out")
        assert not (folder / BUILT_IMAGES_FILE).exists()


class TestBuildOptions:
    """The ``builds:`` configuration block, read from a merged config dict and from the command line's sources."""

    def test_from_config_defaults_and_environment(self, monkeypatch):
        from nemo_gym.tasks.harbor.build_options import BuildOptions

        monkeypatch.delenv("NEMO_GYM_BUILD_REPOSITORY", raising=False)
        assert BuildOptions.from_config(None) == BuildOptions()
        assert BuildOptions.from_config({}) == BuildOptions(enabled=True, tool=None, repository=None, platform=None)
        assert BuildOptions.from_config({"builds": None}) == BuildOptions()
        # The repository falls back to the environment variable; an explicit setting wins over it.
        monkeypatch.setenv("NEMO_GYM_BUILD_REPOSITORY", "env.example/repo")
        assert BuildOptions.from_config({"other": 1}).repository == "env.example/repo"
        assert BuildOptions.from_config({"builds": {"repository": "cfg.example/r"}}).repository == "cfg.example/r"
        assert BuildOptions.from_config({"builds": {"repository": ""}}).repository == "env.example/repo"
        # An env.yaml-style block sets all four fields.
        block = {
            "builds": {"tool": "buildah", "repository": "flag.example/r", "platform": "linux/amd64", "enabled": False}
        }
        assert BuildOptions.from_config(block) == BuildOptions(
            enabled=False, tool="buildah", repository="flag.example/r", platform="linux/amd64"
        )

    def test_from_config_refuses_unknown_settings_and_wrong_types(self):
        from nemo_gym.tasks.harbor.build_options import BuildOptions

        with pytest.raises(ValueError, match="Unknown `builds` setting\\(s\\) rebuild; expected tool, repository"):
            BuildOptions.from_config({"builds": {"rebuild": True}})
        with pytest.raises(ValueError, match="`builds.tool` must be one of buildah, podman, docker; got 'kaniko'"):
            BuildOptions.from_config({"builds": {"tool": "kaniko"}})
        with pytest.raises(ValueError, match="`builds.enabled` must be true or false; got 'no'"):
            BuildOptions.from_config({"builds": {"enabled": "no"}})
        with pytest.raises(ValueError, match="`builds.force` must be true or false; got 'yes'"):
            BuildOptions.from_config({"builds": {"force": "yes"}})
        assert BuildOptions.from_config({"builds": {"force": True}}).force is True
        assert BuildOptions.from_config({}).force is False
        with pytest.raises(ValueError, match="`builds.platform` must be a string; got 1"):
            BuildOptions.from_config({"builds": {"platform": 1}})
        with pytest.raises(ValueError, match="`builds` must be a mapping of tool, repository, platform, enabled"):
            BuildOptions.from_config({"builds": "docker"})

    @pytest.fixture
    def sources(self, tmp_path, monkeypatch):
        """An env.yaml of its own (never the repository's) and a --config file, both with a ``builds:`` block."""

        monkeypatch.delenv("NEMO_GYM_BUILD_REPOSITORY", raising=False)
        dotenv = tmp_path / "env.yaml"
        dotenv.write_text("builds:\n  tool: podman\n  repository: env.example/r\n")
        monkeypatch.setattr(build_options, "_resolve_under_cwd_or_install", lambda rel: tmp_path / rel)
        config = tmp_path / "run.yaml"
        config.write_text("builds:\n  tool: buildah\n  platform: linux/arm64\n")
        return dotenv, config

    def test_overrides_merge_the_command_line_sources_in_the_global_config_order(self, sources):
        from nemo_gym.tasks.harbor.build_options import BuildOptions, build_options_from_overrides

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

        from nemo_gym.tasks.harbor.build_options import BuildOptions, build_options_from_overrides

        dotenv, _ = sources
        dotenv.write_text("some_key: 1\n")
        monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "x"])
        assert build_options_from_overrides(["+split=validation"]) == BuildOptions()
        assert sys.argv == ["gym", "eval", "run", "x"]

    def test_run_and_validate_read_the_block_before_preparing(self, tmp_path, monkeypatch, sources):
        """Both entry points hand the block of the command line's configuration to prepare."""
        import argparse

        import nemo_gym.tasks.harbor.cli as harbor_cli

        seen: list[build_options.BuildOptions] = []

        def fake_prepare(target, **kwargs):
            seen.append(kwargs["build"])
            raise harbor_cli.HubError("stop here")

        def fail(message):
            raise SystemExit(message)

        monkeypatch.setattr(harbor_cli, "prepare_target", fake_prepare)
        # The rows file is scoped to the agent, so the stub has to name one.
        monkeypatch.setattr(
            harbor_cli, "resolve_agent", lambda name: harbor_cli.AgentSelection(tmp_path / "a.yaml", name, name)
        )
        args = argparse.Namespace(
            target=str(tmp_path), agent="a", sandbox=None, _parser=argparse.Namespace(error=fail)
        )
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.run_target(args, ["++builds.enabled=false"])
        with pytest.raises(SystemExit, match="stop here"):
            harbor_cli.validate_target(args, ["++builds.platform=linux/arm64"])
        assert seen == [
            build_options.BuildOptions(enabled=False, tool="podman", repository="env.example/r"),
            build_options.BuildOptions(tool="podman", repository="env.example/r", platform="linux/arm64"),
        ]
