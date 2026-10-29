# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dockerfile builds (image mode three): planning, building through a tool, recording, and the loader."""

import json
from pathlib import Path

import pytest
import yaml

import nemo_gym.cli.eval as cli_eval
import nemo_gym.data.tasks.harbor.build_options as build_options
import nemo_gym.data.tasks.harbor.prepare_builds as prepare_builds
from nemo_gym.data import BUILT_IMAGES_FILE, RECORD_VERSION, HarborTaskError, load_task
from nemo_gym.data.tasks.harbor.build_compose import (
    compose_build_env,
    harbor_compose_document,
)
from nemo_gym.data.tasks.harbor.build_plan import BuildError, needs_build, plan_builds
from nemo_gym.data.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.data.tasks.harbor.digest import context_digest
from nemo_gym.data.tasks.harbor.image_build import build_images
from nemo_gym.data.tasks.harbor.records import IMAGE_CONFIGS_FILE
from nemo_gym.data.tasks.harbor.task import built_service_images
from nemo_gym.data.tasks.harbor.verifier_env import verifier_compose_main_overrides, verifier_compose_sidecars
from tests.unit_tests.harbor_build_helpers import (
    BUILD_DOCKERFILE,
    PULL_DOCKERFILE,
    REPO,
    TOML,
    UBUNTU_RECORD,
    FakeTool,
    HeredocTool,
    harbor_hash,
    local_docker,
    main_record,
    settled,
    tagged_buildah,
    task_inputs_digest,
    write_base_record,
    write_task,
)
from tests.unit_tests.test_harbor_build_plan import TestPlan as PlanCases


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
        assert "docker_image" in str(info.value) and str(folder / BUILT_IMAGES_FILE) in str(info.value)
        # A task discovered deeper is read against the dataset root, and the message names that record, not one
        # next to the group folder.
        write_task(folder / "group" / "b")
        with pytest.raises(HarborTaskError, match=r"needs a build \(COPY\)") as nested:
            load_task(folder / "group" / "b", root=folder)
        assert str(folder / BUILT_IMAGES_FILE) in str(nested.value)
        assert str(folder / "group" / BUILT_IMAGES_FILE) not in str(nested.value)

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
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

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

    def test_a_forced_task_is_recorded_when_every_image_of_its_context_is_already_recorded(
        self, tmp_path, fake_builder, monkeypatch
    ):
        """Review (CI bot, round 34): two tasks share a context, `d` builds its Dockerfile and `e` declares a
        `docker_image`. A prepare under `builds.force` after `main` is recorded plans `e` as forced, but the
        complete-record return skipped `build_images`, whose reuse path is where forced names were recorded, so `e`
        was absent from `forced_tasks` and the loader kept its declared image: the requested Dockerfile additions
        silently missing on every repeat preparation. The fast path records them too, and still needs no build tool."""
        from nemo_gym.data.tasks.harbor import image_configs
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_task(folder / "d")
        write_task(folder / "e", toml=TOML + 'docker_image = "org/prebuilt:1"\n')
        write_base_record(folder)
        write_base_record(folder, "org/prebuilt:1")
        digest = harbor_hash(folder / "d" / "environment")

        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions())

        assert len(fake_builder) == 1 and prepared.skipped == {}
        images = {task.task_id: task.image for task in prepared.tasks}
        built = images["d"]
        assert built.startswith(f"hb__{digest}") or f"/hb__{digest}" in built
        assert images["e"] == "org/prebuilt:1"
        recorded = json.loads((folder / BUILT_IMAGES_FILE).read_text())[digest]
        assert recorded["tasks"] == ["d"] and "forced_tasks" not in recorded

        # The repeat under force: every image is recorded, so no build tool (and no registry) may be needed.
        class NoRegistry:
            def __init__(self, *args, **kwargs):
                raise AssertionError("the registry was consulted")

        monkeypatch.setattr(image_configs, "RegistryClient", NoRegistry)
        monkeypatch.setattr(
            prepare_builds, "ImageBuilder", lambda **kwargs: pytest.fail("a build tool was looked for")
        )

        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(force=True))

        assert prepared.skipped == {}
        recorded = json.loads((folder / BUILT_IMAGES_FILE).read_text())[digest]
        assert (recorded["tasks"], recorded["forced_tasks"]) == (["d"], ["e"])
        assert {task.task_id: task.image for task in prepared.tasks} == {"d": built, "e": built}
        # And the loader, on its own, now takes the recorded build for e as Harbor's --force-build does.
        assert load_task(folder / "e").image == built

    def test_a_sidecar_named_verifier_is_built_like_any_service_and_not_refused_before_the_build(
        self, tmp_path, fake_builder, monkeypatch
    ):
        """kumo's shape (PlanCases.SIDECAR_FILES): main, world and an ordinary sidecar named ``verifier``, every one
        with a ``build:`` section, in shared verifier mode. The plan entry of that sidecar has ``verifier=False`` (the
        separate verifier's image is the only entry with the flag), so the preflight filter treats its ``build:`` as
        built like world's, nothing is refused before the builder runs, and the task prepares with all three recorded."""
        from nemo_gym.data.tasks.harbor import image_configs
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target
        from nemo_gym.data.tasks.harbor.prepare_builds import refused_before_build

        class NoRegistry:
            def __init__(self, *args, **kwargs):
                raise AssertionError("the registry was consulted")

        monkeypatch.setattr(image_configs, "RegistryClient", NoRegistry)
        folder = tmp_path / "ds"
        write_task(folder / "k", extra=PlanCases.SIDECAR_FILES)
        write_base_record(folder, "redis:7")  # the pulled sidecar `cache`, recorded as prepare would have
        plan = plan_builds(folder)
        assert [(entry.service, entry.verifier) for entry in plan] == [
            ("main", False),
            ("world", False),
            ("verifier", False),
        ]
        assert refused_before_build(plan) == {}

        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="podman", repository="reg.example/r")
        )

        assert prepared.skipped == {}
        (task,) = prepared.tasks
        digest = harbor_hash(folder / "k" / "environment")
        assert built_service_images(task.path) == {
            "main": f"reg.example/r/hb__{digest}@sha256:pushed01",
            "world": f"reg.example/r/hb__{digest}-world@sha256:pushed01",
            "verifier": f"reg.example/r/hb__{digest}-verifier@sha256:pushed01",
        }
        assert task.image == f"reg.example/r/hb__{digest}@sha256:pushed01" and task.verifier_image is None

    def test_a_verifier_context_shared_with_a_refused_task_is_still_built_for_the_eligible_one(
        self, tmp_path, fake_builder, monkeypatch
    ):
        """Review (M24): tasks a and b in separate verifier mode ship a byte-identical ``tests/`` (one plan entry with
        tasks (a, b)) and distinct ``environment/`` Dockerfiles; a's Compose file carries a key the adapter refuses.
        The preflight used to drop every entry a refused task was on, the shared verifier entry included, so b's main
        was built but its verifier image never recorded and b failed to load. The plan is now made again without the
        refused tasks: b's main and the shared verifier are built, the verifier entry narrowed to b, only a left out."""
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        refused_compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    privileged: true\n"
        write_separate_task(folder / "a", extra={"docker-compose.yaml": refused_compose})
        write_separate_task(folder / "b", dockerfile=BUILD_DOCKERFILE + "ENV B=1\n")
        write_base_record(folder)
        tests = (folder / "a" / "tests" / "Dockerfile").read_bytes()
        assert tests == (folder / "b" / "tests" / "Dockerfile").read_bytes()
        tests_digest = harbor_hash(folder / "b" / "tests")
        shared = next(entry for entry in plan_builds(folder) if entry.verifier)
        assert shared.tasks == ("a", "b") and shared.digest == tests_digest == harbor_hash(folder / "a" / "tests")
        handed: list = []
        build_many = ImageBuilder.build_many

        def recording(self, entries, **kwargs):
            handed.extend(entries)
            return build_many(self, entries, **kwargs)

        monkeypatch.setattr(ImageBuilder, "build_many", recording)
        build = BuildOptions(tool="podman", repository="reg.example/r")

        refused = prepare_builds._build_dockerfiles(folder, build, select=None)

        assert list(refused) == ["a"]
        assert "Service 'main': unsupported Compose fields ['privileged']" in refused["a"]
        # b's main and the shared verifier were handed to the builder, the verifier entry for b alone; not a's main.
        assert {(entry.digest, entry.service, entry.tasks, entry.verifier) for entry in handed} == {
            (harbor_hash(folder / "b" / "environment"), "main", ("b",), False),
            (tests_digest, "verifier", ("b",), True),
        }
        recorded = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert recorded[tests_digest]["tasks"] == ["b"]
        assert recorded[tests_digest]["services"]["verifier"]["image"] == (
            f"reg.example/r/hb__{tests_digest}-verifier@sha256:pushed01"
        )

        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=build)

        assert list(prepared.skipped) == ["a"] and "unsupported Compose fields ['privileged']" in prepared.skipped["a"]
        (b,) = prepared.tasks
        assert b.task_id == "b"
        assert b.image == f"reg.example/r/hb__{harbor_hash(folder / 'b' / 'environment')}@sha256:pushed01"
        assert b.verifier_image == f"reg.example/r/hb__{tests_digest}-verifier@sha256:pushed01"

    def test_prepare_records_a_sidecar_and_makes_no_registry_call_for_it(self, tmp_path, fake_builder, monkeypatch):
        from nemo_gym.data.tasks.harbor import image_configs
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.image_configs import compose_image_references
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        class NoRegistry:
            def __init__(self, *args, **kwargs):
                raise AssertionError("the registry was consulted for a built image")

        monkeypatch.setattr(image_configs, "RegistryClient", NoRegistry)
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ./db\n    expose: ['5432']\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="podman", repository="reg.example/r")
        )
        (task,) = prepared.tasks
        digest = harbor_hash(folder / "a" / "environment")
        main = f"reg.example/r/hb__{digest}@sha256:pushed01"
        db = f"reg.example/r/hb__{digest}-db@sha256:pushed01"
        assert task.image == main and built_service_images(task.path) == {"main": main, "db": db}
        # The loader hands the recorded sidecar image to the Compose readers; its configuration was recorded by the
        # build, so record_compose_images resolved nothing (NoRegistry would have raised).
        assert compose_image_references(task) == [main, db]
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[db]["config"]["WorkingDir"] == "/srv"
        assert prepared.skipped == {}

    def test_builds_disabled_leaves_the_tasks_out_with_the_loader_message(self, tmp_path, fake_builder, capsys):
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

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

    @pytest.mark.parametrize("shape", ["legacy", "main-only", "sidecar-only"])
    def test_builds_disabled_names_a_task_whose_context_is_only_partly_recorded(
        self, tmp_path, fake_builder, capsys, shape
    ):
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  world:\n    build: ./world\n"
        files = {"docker-compose.yaml": compose, "world/Dockerfile": "FROM python:3.12\n", "tools/setup.sh": "true\n"}
        write_task(folder / "a", extra=files)
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        digest = harbor_hash(folder / "a" / "environment")
        main_image = f"reg.example/r/hb__{digest}@sha256:aa"
        world_image = f"reg.example/r/hb__{digest}-world@sha256:bb"
        write_base_record(folder)
        # The built images' records carry the digest of their inputs as they are now, as the build writes them.
        write_base_record(folder, main_image, inputs_digest=task_inputs_digest(folder / "a"))
        write_base_record(folder, world_image, inputs_digest=task_inputs_digest(folder / "a", service="world"))
        # A record of a pre-sidecar Gym, a prepare interrupted after main, and one interrupted after the sidecar.
        entries = {
            "legacy": {"image": main_image, "tasks": ["a"], "tool": "buildah", "pushed": True},
            "main-only": main_record(main_image, ["a"], "buildah", True),
            "sidecar-only": {"tasks": ["a"], "tool": "buildah", "pushed": True, "services": {"world": world_image}},
        }
        (folder / BUILT_IMAGES_FILE).write_text(json.dumps({digest: entries[shape]}))
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        # The task is left out by the loader (no main) or by the Compose reader (no world), and the digest is
        # present, but a planned service of it has no record: the notice still names the task.
        reason = "needs a build" if shape == "sidecar-only" else "has a build section and prepare recorded no build"
        assert [t.task_id for t in prepared.tasks] == ["c"] and reason in prepared.skipped["a"]
        assert fake_builder == []
        assert (
            "Skipping Dockerfile builds (builds.enabled is false); tasks that need one are left out: a"
            in capsys.readouterr().out
        )
        # Every planned service recorded: the task loads from the records and nothing is said.
        complete = {**main_record(main_image, ["a"], "buildah", True)}
        complete["services"] = {**complete["services"], "world": {"image": world_image}}
        (folder / BUILT_IMAGES_FILE).write_text(json.dumps({digest: complete}))
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["a", "c"] and prepared.tasks[0].image == main_image
        assert "Skipping Dockerfile builds" not in capsys.readouterr().out

    def test_builds_disabled_refuses_a_stale_sidecar_record_beside_an_unrecorded_main(
        self, tmp_path, fake_builder, capsys, monkeypatch
    ):
        """Review (M24): a context whose ``main`` had no record and whose sidecar's record was stale was only named
        for the missing ``main``, which the loader refuses for a task that needs it; a task whose ``main`` is pinned
        by the dataset needs only the sidecar, so it loaded and ran the stale one. An unusable record is refused
        whatever else the context lacks."""
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  world:\n    build:\n      context: ./world\n      args:\n        - FOO\n"
        )
        files = {"docker-compose.yaml": compose, "world/Dockerfile": "FROM python:3.12\n", "tools/setup.sh": "true\n"}
        write_task(folder / "a", extra=files)
        write_task(folder / "b", extra=files)
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        # a's main is pinned by the dataset (to the recorded base image, so nothing is resolved from a registry), so
        # only world is planned for it; b needs the shared main built.
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\ndocker_image = "ubuntu:24.04"\n')
        write_base_record(folder)
        monkeypatch.setenv("FOO", "1")
        digest = harbor_hash(folder / "a" / "environment")
        world_image = f"reg.example/r/hb__{digest}-world@sha256:bb"
        write_base_record(folder, world_image, inputs_digest=task_inputs_digest(folder / "a", service="world"))
        sidecar_only = {
            "tasks": ["a", "b"],
            "tool": "buildah",
            "pushed": True,
            "services": {"world": {"image": world_image, "tool": "buildah", "pushed": True}},
        }
        (folder / BUILT_IMAGES_FILE).write_text(json.dumps({digest: sidecar_only}))
        # world was recorded with FOO=1; the environment now says 2, so its record is stale while main is missing.
        monkeypatch.setenv("FOO", "2")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert fake_builder == []
        reason = (
            f"recorded image {world_image} cannot be used: its build inputs changed since it was built (a file's "
            "content or mode, a symbolic link, a folder, or a build setting) while the context's hash did not; "
            "builds are disabled"
        )
        assert [t.task_id for t in prepared.tasks] == ["c"] and prepared.skipped == {"a": reason, "b": reason}
        assert world_image not in (tmp_path / "out" / "ds" / "tasks.jsonl").read_text()
        out = capsys.readouterr().out
        assert f"Skipping a, b: {reason}" in out and "tasks that need one are left out: a, b" in out
        # The record usable again: a runs its pinned main beside the recorded world, b still waits for its main.
        monkeypatch.setenv("FOO", "1")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["a", "c"] and list(prepared.skipped) == ["b"]
        assert "needs a build (COPY)" in prepared.skipped["b"]
        assert prepared.tasks[0].built_images == {"world": world_image}

    def test_builds_disabled_leaves_out_a_record_that_does_not_fit_the_run(self, tmp_path, fake_builder, capsys):
        """A record of another platform is no better than none: with builds disabled its tasks are left out with
        the reason, as the enabled path would rebuild it, and no row references the image."""
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

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

    def test_an_amd64_level_above_the_baseline_is_refused_before_any_build(self, tmp_path, monkeypatch):
        """A BuildOptions made in code (prepare_target's callers, not from_config) reaches the tool through
        ImageBuilder, which refuses builds.platform as the configuration does. The case the records cannot tell
        apart: a's main is pinned by the dataset (no base image to record, so record_base_images never asks
        platform_pair) and its sidecar needs a build; before the check the sidecar was built with --platform
        linux/amd64/v3, pushed and recorded, and a later linux/amd64 run reused that record."""
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        with pytest.raises(BuildError, match="builds.platform: Unsupported platform: 'linux/amd64/v3'"):
            ImageBuilder(tool="docker", platform="linux/amd64/v3", run=FakeTool("docker"))
        assert (
            ImageBuilder(tool="docker", platform="linux/arm64/v8", run=FakeTool("docker")).platform == "linux/arm64/v8"
        )
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  world:\n    build:\n      context: ./world\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "world/Dockerfile": "FROM python:3.12\n"})
        (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\ndocker_image = "ubuntu:24.04"\n')
        write_base_record(folder)
        tool = FakeTool("buildah")
        monkeypatch.setattr(
            prepare_builds, "ImageBuilder", lambda **kw: ImageBuilder(run=tool, storage_root=tmp_path / "s", **kw)
        )
        with pytest.raises(
            ValueError,
            match=r"Could not build the Dockerfile image\(s\) for task\(s\) a: builds.platform: Unsupported platform: "
            r"'linux/amd64/v3'",
        ):
            prepare_target(
                str(folder),
                output_root=tmp_path / "out",
                build=BuildOptions(tool="buildah", repository=REPO, platform="linux/amd64/v3"),
            )
        assert tool.calls == [] and not (folder / BUILT_IMAGES_FILE).exists()
        # The baseline run then builds the sidecar for its own platform, from nothing recorded.
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(tool="buildah", repository=REPO)
        )
        bud = [call for call in tool.calls if "bud" in call]
        assert len(bud) == 1 and bud[0][bud[0].index("--platform") + 1] == "linux/amd64"
        assert [t.task_id for t in prepared.tasks] == ["a"] and list(prepared.tasks[0].built_images) == ["world"]

    def test_recorded_images_need_no_build_tool(self, tmp_path, monkeypatch, capsys):
        """A host with every context recorded prepares without docker, podman or buildah; one with a context left
        to build still needs one, and the error says how to run with recorded images only."""
        import nemo_gym.data.tasks.harbor.build_tools as image_build_module
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

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
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

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
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions

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
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions

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

    def test_from_config_refuses_an_amd64_level_above_the_baseline(self):
        """builds.platform is checked where it enters, by platform_pair's rules: an amd64 micro-architecture level
        above the baseline would be built and recorded under os/arch alone and reused by a later linux/amd64 run.
        The baseline's spellings and the other architectures pass as written (the records normalise them)."""
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions

        with pytest.raises(
            ValueError, match="`builds.platform`: Unsupported platform: 'linux/amd64/v3' .*use linux/amd64"
        ):
            BuildOptions.from_config({"builds": {"platform": "linux/amd64/v3"}})
        with pytest.raises(ValueError, match="`builds.platform`: Not a platform: 'amd64'"):
            BuildOptions.from_config({"builds": {"platform": "amd64"}})
        for platform in ("linux/amd64", "linux/amd64/v1", "linux/x86_64", "linux/arm64/v8", "linux/aarch64"):
            assert BuildOptions.from_config({"builds": {"platform": platform}}).platform == platform

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
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions, build_options_from_overrides

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

        from nemo_gym.data.tasks.harbor.build_options import BuildOptions, build_options_from_overrides

        dotenv, _ = sources
        dotenv.write_text("some_key: 1\n")
        monkeypatch.setattr(sys, "argv", ["gym", "eval", "run", "x"])
        assert build_options_from_overrides(["+split=validation"]) == BuildOptions()
        assert sys.argv == ["gym", "eval", "run", "x"]

    def test_an_unresolved_builds_block_is_a_configuration_error_not_the_defaults(self, sources, monkeypatch):
        """Review (round 14): ``builds: ${missing_profile}`` was selected with resolution failures suppressed, so
        the node read as absent and prepare ran with default build settings until the full configuration parse
        reported the mistake. A configured node that does not resolve is the error it was in M28; a sibling
        block that does not resolve is still left alone, since only the builds node's own interpolations run."""
        from nemo_gym.config.global_config import ConfigInterpolationError
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions, build_options_from_overrides

        dotenv, _ = sources
        dotenv.write_text("builds: ${missing_profile}\n")
        with pytest.raises(ConfigInterpolationError, match="missing_profile"):
            build_options_from_overrides([])
        dotenv.write_text("some_key: 1\n")
        with pytest.raises(ConfigInterpolationError, match="missing_profile"):
            build_options_from_overrides(["++builds=${missing_profile}"])
        monkeypatch.delenv("NEMO_GYM_UNSET_VAR_FOR_THIS_TEST", raising=False)
        dotenv.write_text(
            "builds:\n  tool: docker\nsandbox:\n  opensandbox:\n    api_key: ${oc.env:NEMO_GYM_UNSET_VAR_FOR_THIS_TEST}\n"
        )
        assert build_options_from_overrides([]) == BuildOptions(tool="docker")

    def test_run_and_validate_read_the_block_before_preparing(self, tmp_path, monkeypatch, sources):
        """Both entry points hand the block of the command line's configuration to prepare."""
        import argparse

        import nemo_gym.data.tasks.harbor.prepare as harbor_cli

        seen: list[build_options.BuildOptions] = []

        def fake_prepare(target, **kwargs):
            seen.append(kwargs["build"])
            raise harbor_cli.HubError("stop here")

        def fail(message):
            raise SystemExit(message)

        monkeypatch.setattr(harbor_cli, "prepare_target", fake_prepare)
        monkeypatch.setattr(cli_eval, "prepare_target", fake_prepare)
        # The rows file is scoped to the agent, so the stub has to name one.
        monkeypatch.setattr(
            harbor_cli, "resolve_agent", lambda name: harbor_cli.AgentSelection(tmp_path / "a.yaml", name, name)
        )
        monkeypatch.setattr(
            cli_eval, "resolve_agent", lambda name: harbor_cli.AgentSelection(tmp_path / "a.yaml", name, name)
        )
        args = argparse.Namespace(
            target=str(tmp_path), agent="a", sandbox=None, _parser=argparse.Namespace(error=fail)
        )
        with pytest.raises(SystemExit, match="stop here"):
            cli_eval.run_target(args, ["++builds.enabled=false"])
        with pytest.raises(SystemExit, match="stop here"):
            cli_eval.validate_target(args, ["++builds.platform=linux/arm64"])
        assert seen == [
            build_options.BuildOptions(enabled=False, tool="podman", repository="env.example/r"),
            build_options.BuildOptions(tool="podman", repository="env.example/r", platform="linux/arm64"),
        ]


SEPARATE_TOML = TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
# tb3-preview-v2's shape: a FROM plus RUN file (overlay-kind on environment/), built here because Harbor builds it.
TESTS_DOCKERFILE = "FROM python:3.13-slim-bookworm\nRUN pip install pytest==8.4.1\nCOPY . /tests/\n"
TESTS_OVERLAY_DOCKERFILE = "FROM python:3.13-slim-bookworm\nRUN pip install pytest==8.4.1\n"


def write_separate_task(
    root: Path, *, toml: str = SEPARATE_TOML, tests_dockerfile: str | None = TESTS_DOCKERFILE, **kwargs
) -> Path:
    """A separate-mode task whose ``tests/`` folder holds a Dockerfile (``None`` for none), the verifier's build context."""
    write_task(root, toml=toml, **kwargs)
    if tests_dockerfile is not None:
        (root / "tests" / "Dockerfile").write_text(tests_dockerfile)
    return root


class TestSeparateVerifierImage:
    """D1: the separate verifier's image is built from ``tests/Dockerfile``, as Harbor builds it."""

    def test_plan_adds_a_verifier_entry_for_the_tests_context_after_the_tasks_own(self, tmp_path):
        folder = tmp_path / "ds"
        write_separate_task(folder / "a")
        context = (folder / "a" / "environment").resolve()
        tests = (folder / "a" / "tests").resolve()
        main, verifier = plan_builds(folder)
        assert (main.service, main.is_main, main.verifier) == ("main", True, False)
        assert main.context == context and main.digest == harbor_hash(context)
        # The verifier entry: service `verifier`, tests/ as context and hash root (Harbor's environment_id of its
        # verifier env hashes that dir), building Harbor's template service main of it.
        assert (verifier.service, verifier.is_main, verifier.verifier) == ("verifier", False, True)
        assert verifier.compose_service == "main" and verifier.tasks == ("a",)
        assert verifier.context == tests and verifier.digest == harbor_hash(tests) == context_digest(tests)
        assert verifier.digest != main.digest
        assert verifier.build_dir == tests and verifier.dockerfile_path == tests / "Dockerfile"
        # The effective verifier environment is a copy of [environment]: its resources, and no env without a
        # tests/docker-compose.yaml, as Harbor resolves env for Compose environments only.
        assert (verifier.cpus, verifier.memory_mb, verifier.env) == (1, 2048, {})
        builder, _ = tagged_buildah(tmp_path)
        assert builder.tag_for(verifier) == f"{REPO}/hb__{verifier.digest}-verifier"
        assert builder.tag_for(main) == f"{REPO}/hb__{main.digest}"
        # Harbor's environment for the build: MAIN_IMAGE_NAME is hb__<tests hash>, CONTEXT_DIR the tests folder.
        env = compose_build_env(verifier, builder.tag_name(verifier.digest), None)
        assert env["MAIN_IMAGE_NAME"] == f"{REPO}/hb__{verifier.digest}" and env["CONTEXT_DIR"] == str(tests)
        assert env["COMPOSE_PROJECT_NAME"] == f"hb__{verifier.digest}"

    def test_the_verifier_environment_block_gives_the_entry_its_settings_and_compose_env(self, tmp_path):
        folder = tmp_path / "ds"
        toml = (
            TOML.replace("[verifier]\n", "[verifier]\n")
            + '\n[verifier.environment]\ncpus = 4\nmemory_mb = 8192\n\n[verifier.environment.env]\nMODE = "grade"\n'
            + '\n[environment.env]\nMODE = "agent"\n'
        )
        write_separate_task(folder / "a", toml=toml, dockerfile=PULL_DOCKERFILE)
        (folder / "a" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
        )
        write_base_record(folder)
        # A pull-mode main plans nothing of its own; the verifier block (no docker_image) still plans the build.
        (verifier,) = plan_builds(folder)
        assert verifier.verifier and (verifier.cpus, verifier.memory_mb) == (4, 8192)
        assert verifier.env == {"MODE": "grade"}
        assert settled(verifier, f"hb__{verifier.digest}").dockerfile_path == verifier.context / "Dockerfile"

    def test_an_overlay_kind_tests_dockerfile_is_built_not_overlaid(self, tmp_path):
        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE, tests_dockerfile=TESTS_OVERLAY_DOCKERFILE)
        write_base_record(folder)
        assert not needs_build(TESTS_OVERLAY_DOCKERFILE, {"python:3.13-slim-bookworm": UBUNTU_RECORD})
        (verifier,) = plan_builds(folder)
        assert verifier.verifier and verifier.dockerfile_path.read_text() == TESTS_OVERLAY_DOCKERFILE

    @pytest.mark.parametrize(
        "shape",
        ["shared", "shared_explicit", "declared_verifier_image", "inherited_docker_image", "no_tests_dockerfile"],
    )
    def test_no_verifier_entry_when_harbor_builds_none(self, tmp_path, shape):
        folder = tmp_path / "ds"
        toml = {
            "shared": TOML,
            "shared_explicit": TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "shared"\n'),
            "declared_verifier_image": TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
            # No block and a prebuilt main: Harbor's copy of [environment] carries the image, so prebuilt verifier.
            "inherited_docker_image": SEPARATE_TOML + 'docker_image = "org/prebuilt:1"\n',
            "no_tests_dockerfile": SEPARATE_TOML,
        }[shape]
        tests_dockerfile = None if shape == "no_tests_dockerfile" else TESTS_DOCKERFILE
        write_separate_task(folder / "a", toml=toml, tests_dockerfile=tests_dockerfile)
        plan = plan_builds(folder)
        assert not any(entry.verifier for entry in plan), plan
        assert [entry.service for entry in plan] == ([] if shape == "inherited_docker_image" else ["main"])
        # A dataset override that pins the task's image is Harbor's prebuilt verifier too.
        if shape == "no_tests_dockerfile":
            (folder / "a" / "tests" / "Dockerfile").write_text(TESTS_DOCKERFILE)
            assert [entry.service for entry in plan_builds(folder)] == ["main", "verifier"]
            (folder / "dataset.toml").write_text('[gym.tasks."a".environment]\ndocker_image = "org/pinned:2"\n')
            assert plan_builds(folder) == []

    def test_two_tasks_sharing_tests_share_the_verifier_image_and_a_compose_verifier_sidecar_is_not_it(self, tmp_path):
        folder = tmp_path / "ds"
        write_separate_task(folder / "a")
        write_separate_task(folder / "b", extra={"tools/setup.sh": "#!/bin/sh\necho other\n"})
        # kumo's shape beside it: a sidecar named `verifier` in environment/docker-compose.yaml, shared mode.
        write_task(folder / "k", extra=PlanCases.SIDECAR_FILES)
        plan = plan_builds(folder)
        by_task = [(entry.service, entry.tasks, entry.verifier) for entry in plan]
        tests_digest = harbor_hash(folder / "a" / "tests")
        assert by_task == [
            ("main", ("a",), False),
            ("verifier", ("a", "b"), True),
            ("main", ("b",), False),
            ("main", ("k",), False),
            ("world", ("k",), False),
            ("verifier", ("k",), False),
        ]
        assert plan[1].digest == tests_digest and plan[5].digest == harbor_hash(folder / "k" / "environment")
        assert plan[5].compose_service == "verifier" and plan[1].compose_service == "main"

    def test_buildah_builds_records_and_the_loader_exposes_the_verifier_image(self, tmp_path):
        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        tests = (folder / "a" / "tests").resolve()
        # Before the build: the loader refuses the task the way it refuses an unrecorded environment/ build, naming
        # the file, the record and the two remedies (Harbor grades in that image; nothing else is a stand-in).
        with pytest.raises(HarborTaskError) as info:
            load_task(folder / "a")
        message = str(info.value)
        assert message.startswith(f"{folder / 'a' / 'tests' / 'Dockerfile'} is the separate verifier's image")
        assert str(folder / BUILT_IMAGES_FILE) in message and "`builds:` block" in message
        assert "declare [verifier.environment].docker_image" in message
        (folder / "a" / "environment" / "Dockerfile").write_text(BUILD_DOCKERFILE)
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        main, verifier = build_images(folder, builder=builder, log=lambda _: None)
        assert verifier.service == "verifier" and verifier.tag == f"{REPO}/hb__{verifier.digest}-verifier"
        assert verifier.digest == harbor_hash(tests) and verifier.image == f"{verifier.tag}@sha256:pushed01"
        bud = next(call for call in fake.calls if "bud" in call and verifier.tag in call)
        assert bud[-3:] == ["-f", str(tests / "Dockerfile"), str(tests)]
        # Recorded under the tests/ digest as service verifier, no main there, its configuration beside it.
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert set(built) == {main.digest, verifier.digest}
        entry = built[verifier.digest]
        assert "image" not in entry and entry["tasks"] == ["a"]
        assert entry["services"] == {"verifier": {"image": verifier.image, "tool": "buildah", "pushed": True}}
        assert json.loads((folder / IMAGE_CONFIGS_FILE).read_text())[verifier.image]["config"]["User"] == "app"
        assert built_service_images(folder / "a") == {"main": main.image}  # environment/'s record is main's alone
        task = load_task(folder / "a")
        assert task.image == main.image and task.built_images == {"main": main.image}
        assert task.verifier_image == verifier.image
        # A second run finds both recorded; a changed tests/ folder is a new verifier build and main is kept.
        lines: list[str] = []
        assert build_images(folder, builder=builder, log=lines.append) == []
        assert lines == [
            f"Built image recorded for a: {main.image}",
            f"Built image recorded for a (verifier): {verifier.image}",
        ]
        (folder / "a" / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")
        (rebuilt,) = build_images(folder, builder=builder, log=lambda _: None)
        assert rebuilt.service == "verifier" and rebuilt.digest != verifier.digest
        assert load_task(folder / "a").verifier_image == rebuilt.image

    def test_an_unbuilt_verifier_of_a_nested_task_names_the_record_at_the_dataset_root(self, tmp_path):
        folder = tmp_path / "ds"
        write_separate_task(folder / "group" / "a", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        with pytest.raises(HarborTaskError) as info:
            load_task(folder / "group" / "a", root=folder)
        message = str(info.value)
        assert message.startswith(
            f"{folder / 'group' / 'a' / 'tests' / 'Dockerfile'} is the separate verifier's image"
        )
        assert str(folder / BUILT_IMAGES_FILE) in message
        assert str(folder / "group" / BUILT_IMAGES_FILE) not in message

    def test_a_declared_verifier_image_is_the_loaders_verifier_image_without_a_build(self, tmp_path):
        folder = tmp_path / "ds"
        toml = TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n'
        write_separate_task(folder / "a", toml=toml, dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        assert plan_builds(folder) == []
        assert load_task(folder / "a").verifier_image == "org/verifier:1"
        # Harbor's prebuilt verifier by inheritance: no block, [environment].docker_image, no tests/Dockerfile. The
        # verifier runs in the task's own image, so there is nothing to expose and nothing to plan.
        write_separate_task(
            folder / "b", toml=SEPARATE_TOML + 'docker_image = "org/prebuilt:1"\n', tests_dockerfile=None
        )
        assert plan_builds(folder) == [] and load_task(folder / "b").verifier_image is None

    def test_the_pre_pull_manifest_names_the_built_verifier_image(self, tmp_path, monkeypatch):
        """Review (M24): ``images.json`` named the task images, the declared verifier images and the Compose services
        but not a verifier image built from ``tests/Dockerfile``, which the episode starts like the others, so a host
        pre-pulling from the manifest did not have it."""
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE)
        write_task(folder / "plain", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        fake = FakeTool("buildah")
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake),
        )
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        verifier_image = f"{REPO}/hb__{harbor_hash(folder / 'a' / 'tests')}-verifier@sha256:pushed01"
        assert [(t.task_id, t.image, t.verifier_image) for t in prepared.tasks] == [
            ("a", "ubuntu:24.04", verifier_image),
            ("plain", "ubuntu:24.04", None),  # shared mode: no verifier image
        ]
        # The built verifier image once, beside the task image the two tasks share (sorted, as the file is written).
        assert prepared.images == (verifier_image, "ubuntu:24.04")
        assert json.loads((tmp_path / "out" / "ds" / "images.json").read_text()) == list(prepared.images)

    @pytest.mark.parametrize("shape", ["no_block", "block_without_image"])
    def test_no_tests_dockerfile_and_no_image_is_refused_as_harbor_refuses_it(self, tmp_path, shape):
        """Harbor's verifier environment dir is always tests/ (no fallback to environment/): without a Dockerfile
        there and without a docker_image its environment validation fails, so the loader refuses the task with that
        wording, with or without a [verifier.environment] block."""
        folder = tmp_path / "ds"
        toml = SEPARATE_TOML if shape == "no_block" else TOML + "\n[verifier.environment]\ncpus = 2\n"
        write_separate_task(folder / "a", toml=toml, dockerfile=PULL_DOCKERFILE, tests_dockerfile=None)
        write_base_record(folder)
        assert plan_builds(folder) == []  # nothing to build: preparing with builds enabled changes nothing
        with pytest.raises(HarborTaskError) as info:
            load_task(folder / "a")
        message = str(info.value)
        assert message.startswith(f"Task environment directory {folder / 'a' / 'tests'} has no environment definition")
        assert "require_agent_environment_definition" in message
        assert "Set [verifier.environment].docker_image or add tests/Dockerfile" in message
        assert "prepare" not in message  # no remedy that cannot help
        # A tests/docker-compose.yaml alone is a Harbor Compose build this server does not plan: refused as such.
        write_separate_task(folder / "b", toml=toml, dockerfile=PULL_DOCKERFILE, tests_dockerfile=None)
        (folder / "b" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build:\n      dockerfile: Dockerfile.verifier\n"
        )
        assert plan_builds(folder) == []
        with pytest.raises(HarborTaskError, match="this server builds it from tests/Dockerfile only"):
            load_task(folder / "b")

    def test_builds_disabled_leaves_a_task_whose_verifier_image_is_unbuilt_out(self, tmp_path, capsys):
        """A pull-mode main needs no build; the verifier's tests/Dockerfile does. With builds disabled the task is
        left out, in skipped.json and in prepare's message, as a task with an unrecorded environment/ build is."""
        from nemo_gym.data.tasks.harbor.build_options import BuildOptions
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE)
        write_task(folder / "c", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["c"]
        assert list(prepared.skipped) == ["a"] and "is the separate verifier's image" in prepared.skipped["a"]
        assert json.loads((prepared.output_dir / "skipped.json").read_text()) == prepared.skipped
        assert (
            "Skipping Dockerfile builds (builds.enabled is false); tasks that need one are left out: a"
            in capsys.readouterr().out
        )
        # Built and recorded, the task prepares with its verifier image and a disabled build says nothing.
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        (verifier,) = build_images(folder, builder=builder, log=lambda _: None)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["a", "c"] and prepared.skipped == {}
        assert prepared.tasks[0].verifier_image == verifier.image
        assert "Skipping Dockerfile builds" not in capsys.readouterr().out

    def test_a_tests_compose_file_with_a_sidecar_is_refused_by_the_loader_and_before_any_build(
        self, tmp_path, monkeypatch, capsys
    ):
        """Harbor builds and starts every service of tests/docker-compose.yaml as the verifier's Compose project;
        this server starts one verifier sandbox, so such a task is refused, built image or declared one."""
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE)
        write_separate_task(folder / "b", dockerfile=PULL_DOCKERFILE, extra={"tools/setup.sh": "#!/bin/sh\necho b\n"})
        write_base_record(folder)
        (folder / "a" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    image: postgres:16\n"
        )
        # b: the main-only shape (cadir, arc-crm-6) is read for main's build section and nothing else.
        (folder / "b" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
        )
        assert verifier_compose_sidecars(folder / "a") == ["db"] and verifier_compose_sidecars(folder / "b") == []
        with pytest.raises(HarborTaskError, match=r"declares service\(s\) db beside main"):
            load_task(folder / "a")
        # A declared verifier image changes nothing: Harbor stacks the file on its prebuilt template too.
        write_separate_task(
            folder / "c",
            toml=TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
            dockerfile=PULL_DOCKERFILE,
            tests_dockerfile=None,
        )
        (folder / "c" / "tests" / "docker-compose.yaml").write_text("services:\n  db:\n    image: postgres:16\n")
        with pytest.raises(HarborTaskError, match=r"declares service\(s\) db beside main"):
            load_task(folder / "c")
        fake = FakeTool("buildah")
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake),
        )
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["b"] and sorted(prepared.skipped) == ["a", "c"]
        assert prepared.skipped["a"].startswith("Could not build the Dockerfile image(s) for task(s) a: ")
        assert "declares service(s) db beside main" in prepared.skipped["a"]
        assert "declares service(s) db beside main" in prepared.skipped["c"]  # the loader's refusal, no build planned
        # Nothing was built for a: no command named its tests/ context and no record was written for it.
        a_tests = str((folder / "a" / "tests").resolve())
        assert not any(a_tests in call for call in fake.calls)
        assert harbor_hash(folder / "a" / "tests") not in json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert (
            prepared.tasks[0].verifier_image
            == f"{REPO}/hb__{harbor_hash(folder / 'b' / 'tests')}-verifier@sha256:pushed01"
        )
        assert "Built 1 image(s) with buildah, pushed 1" in capsys.readouterr().out

    def test_a_tests_compose_file_with_runtime_settings_on_main_is_refused_by_the_loader_and_before_any_build(
        self, tmp_path, monkeypatch
    ):
        """Review (M24): Harbor stacks tests/docker-compose.yaml on its verifier template, so ``main.environment``
        reaches the verifier's container and a test.sh that reads ``CHECK_MODE`` there grades 1; this server starts
        the verifier from the TOML environment alone and used to accept the file silently, grading 0. A runtime key
        on main is refused like a sidecar; main's image, build section and pull_policy are still read for a verifier
        prepare builds from tests/Dockerfile."""
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE)
        write_separate_task(folder / "b", dockerfile=PULL_DOCKERFILE, extra={"tools/setup.sh": "#!/bin/sh\necho b\n"})
        write_base_record(folder)
        (folder / "a" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: ${CONTEXT_DIR}\n"
            "    environment:\n      CHECK_MODE: strict\n    env_file: .env\n"
        )
        # b: main's image, build section (its args apply to the verifier's build) and pull_policy, nothing else.
        (folder / "b" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: ${CONTEXT_DIR}\n"
            "      args:\n        PYTEST: '8.4.1'\n    pull_policy: build\n"
        )
        assert verifier_compose_main_overrides(folder / "a", prebuilt=False) == ["environment", "env_file"]
        assert verifier_compose_main_overrides(folder / "b", prebuilt=False) == []
        with pytest.raises(HarborTaskError, match=r"docker-compose.yaml sets environment, env_file on main"):
            load_task(folder / "a")
        # A declared verifier image changes nothing: Harbor stacks the file on its prebuilt template too (where
        # main's image is an override as well, see the next test).
        write_separate_task(
            folder / "c",
            toml=TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
            dockerfile=PULL_DOCKERFILE,
            tests_dockerfile=None,
        )
        (folder / "c" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    volumes:\n      - ./fixtures:/fixtures\n"
        )
        with pytest.raises(HarborTaskError, match=r"sets image, volumes on main"):
            load_task(folder / "c")
        fake = FakeTool("buildah")
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake),
        )
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["b"] and sorted(prepared.skipped) == ["a", "c"]
        assert prepared.skipped["a"].startswith("Could not build the Dockerfile image(s) for task(s) a: ")
        assert "sets environment, env_file on main" in prepared.skipped["a"]
        assert "sets image, volumes on main" in prepared.skipped["c"]  # the loader's refusal, no build planned
        # Nothing was built for a: no command named its tests/ context and no record was written for it.
        a_tests = str((folder / "a" / "tests").resolve())
        assert not any(a_tests in call for call in fake.calls)
        assert harbor_hash(folder / "a" / "tests") not in json.loads((folder / BUILT_IMAGES_FILE).read_text())
        # b's verifier was built with its build argument and the task runs it.
        bud = next(call for call in fake.calls if "bud" in call and str((folder / "b" / "tests").resolve()) in call)
        assert "PYTEST=8.4.1" in bud
        assert (
            prepared.tasks[0].verifier_image
            == f"{REPO}/hb__{harbor_hash(folder / 'b' / 'tests')}-verifier@sha256:pushed01"
        )

    @pytest.mark.parametrize("spelling", ["org/other:2", "${MAIN_IMAGE_NAME}"])
    def test_an_image_on_main_beside_a_prebuilt_verifier_is_refused_as_an_override(self, tmp_path, spelling):
        """Review (M24): the allow-list admitted main's image for every verifier, but a prebuilt one reads nothing of
        main: Harbor's prebuilt template is `main: image: ${PREBUILT_IMAGE_NAME}` and the task file comes last, so a
        literal runs in place of the declared image and `${MAIN_IMAGE_NAME}` (still exported in prebuilt mode) runs
        the hb__<tests hash> nobody built, while this server ran the declared image. Refused like any other override,
        declared block or inherited [environment].docker_image, and so is a build section there: Harbor's prebuilt
        start skips `compose build` but its `up` builds a main that has one (always under `pull_policy: build`, else
        when the declared image is neither cached nor pulled) and grades in the result under the declared name. A pull
        policy on the declared image alone and the built verifier's pinned image still load."""
        folder = tmp_path / "ds"
        compose = f"services:\n  main:\n    image: {spelling}\n"
        write_separate_task(
            folder / "declared",
            toml=TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
            dockerfile=PULL_DOCKERFILE,
            tests_dockerfile=None,
        )
        write_base_record(folder)
        (folder / "declared" / "tests" / "docker-compose.yaml").write_text(compose)
        assert verifier_compose_main_overrides(folder / "declared", prebuilt=True) == ["image"]
        assert verifier_compose_main_overrides(folder / "declared", prebuilt=False) == []
        with pytest.raises(HarborTaskError, match=r"docker-compose.yaml sets image on main; Harbor applies the file"):
            load_task(folder / "declared")
        # The effective environment is a copy of [environment] with its docker_image: Harbor's prebuilt mode too.
        write_separate_task(
            folder / "inherited",
            toml=SEPARATE_TOML + 'docker_image = "org/prebuilt:1"\n',
            dockerfile=PULL_DOCKERFILE,
            tests_dockerfile=None,
        )
        (folder / "inherited" / "tests" / "docker-compose.yaml").write_text(compose)
        with pytest.raises(HarborTaskError, match=r"sets image on main"):
            load_task(folder / "inherited")
        # A build section beside a prebuilt verifier is an override as well, with the pull_policy that names the build
        # (Harbor's up builds tests/Dockerfile and tags it org/verifier:1); the file order is kept in the message.
        (folder / "declared" / "tests" / "Dockerfile").write_text(
            "FROM alpine:3.20\nRUN echo built-by-harbor > /marker\n"
        )
        (folder / "declared" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    pull_policy: build\n    build:\n      context: .\n      args:\n        PYTEST: '8.4.1'\n"
        )
        assert verifier_compose_main_overrides(folder / "declared", prebuilt=True) == ["pull_policy", "build"]
        assert verifier_compose_main_overrides(folder / "declared", prebuilt=False) == []
        with pytest.raises(HarborTaskError, match=r"docker-compose.yaml sets pull_policy, build on main"):
            load_task(folder / "declared")
        assert [entry for entry in plan_builds(folder) if entry.verifier] == []  # nothing is planned for it either
        # The section alone (Compose builds it when the declared image is not to be had), declared or inherited.
        (folder / "declared" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build:\n      context: .\n      dockerfile: Dockerfile\n"
        )
        with pytest.raises(HarborTaskError, match=r"sets build on main"):
            load_task(folder / "declared")
        (folder / "inherited" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build:\n      context: .\n"
        )
        with pytest.raises(HarborTaskError, match=r"sets build on main"):
            load_task(folder / "inherited")
        # A pull policy on the declared image says when Harbor pulls it, which this server does itself: no override.
        (folder / "declared" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    pull_policy: always\n"
        )
        assert verifier_compose_main_overrides(folder / "declared", prebuilt=True) == []
        assert load_task(folder / "declared").verifier_image == "org/verifier:1"
        # A verifier built from tests/Dockerfile reads main's image: the file is no override there and the build is
        # planned (validate_task_compose then pins the spelling to MAIN_IMAGE_NAME, so the literal fails the build).
        write_separate_task(folder / "built", dockerfile=PULL_DOCKERFILE)
        (folder / "built" / "tests" / "docker-compose.yaml").write_text(compose)
        assert verifier_compose_main_overrides(folder / "built", prebuilt=False) == []
        (verifier,) = [entry for entry in plan_builds(folder) if entry.verifier]
        assert verifier.tasks == ("built",)

    def test_an_include_or_unreadable_tests_compose_file_beside_a_prebuilt_verifier_is_refused(
        self, tmp_path, monkeypatch
    ):
        """Review (M24): the override checks read the file's own ``services`` and nothing else, so a top-level
        ``include`` passed them with ``services: {}`` while Compose merges the included file's main (``environment``,
        ``build``) into Harbor's verifier project: CHECK_MODE reached Harbor's verifier and its ``up`` built
        tests/Dockerfile under the declared name, this server graded in the bare declared image. validate_task_compose
        refuses ``include`` for a Dockerfile build but never sees a prebuilt verifier's file; the loader refuses it
        now for every separate verifier, declared block or inherited docker_image, and a file that is not valid YAML
        or not a mapping (Harbor's ``compose up`` fails the trial on it, the readers here saw ``{}``) the same way.
        The same for every other top-level key: a ``networks`` block passed too, and ``default: {internal: true}``
        cuts the egress of Harbor's verifier (its templates give main no network key, so Compose attaches it to the
        project's default network) while this server's sandbox grades with it; ``volumes`` and the rest are refused
        as the class, so only ``services``, ``name``, ``version`` and ``x-`` extensions load. An empty file and a
        plain main-only one still load, and a built verifier's file is refused before any build as well."""
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_separate_task(
            folder / "declared",
            toml=TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
            dockerfile=PULL_DOCKERFILE,
            tests_dockerfile="FROM alpine:3.20\nRUN echo built-by-harbor > /marker\n",
        )
        write_separate_task(
            folder / "inherited",
            toml=SEPARATE_TOML + 'docker_image = "org/prebuilt:1"\n',
            dockerfile=PULL_DOCKERFILE,
            tests_dockerfile=None,
        )
        write_base_record(folder)
        included = "services:\n  main:\n    build:\n      context: .\n    environment:\n      CHECK_MODE: strict\n"
        for name in ("declared", "inherited"):
            tests = folder / name / "tests"
            (tests / "other.yaml").write_text(included)
            (tests / "docker-compose.yaml").write_text("include:\n  - other.yaml\nservices: {}\n")
            # The readers see nothing of the included main: the gap the refusal closes.
            assert verifier_compose_sidecars(folder / name) == []
            assert verifier_compose_main_overrides(folder / name, prebuilt=True) == []
            with pytest.raises(
                HarborTaskError,
                match=rf"{name}/tests/docker-compose.yaml: include is refused for the separate verifier: Compose merges",
            ):
                load_task(folder / name)
        assert [entry for entry in plan_builds(folder) if entry.verifier] == []  # nothing is planned either
        # Not valid YAML, not a mapping, services not a mapping: Harbor's compose up fails on each; refused here.
        tests = folder / "declared" / "tests"
        (tests / "docker-compose.yaml").write_text("services:\n  main:\n    environment: [CHECK_MODE=strict\n  :bad\n")
        with pytest.raises(HarborTaskError, match=r"declared/tests/docker-compose.yaml is not valid YAML: "):
            load_task(folder / "declared")
        (tests / "docker-compose.yaml").write_text("- main\n")
        with pytest.raises(HarborTaskError, match=r"declared/tests/docker-compose.yaml is not a mapping$"):
            load_task(folder / "declared")
        (tests / "docker-compose.yaml").write_text("services:\n  - main\n")
        with pytest.raises(HarborTaskError, match=r"declared/tests/docker-compose.yaml: services is not a mapping$"):
            load_task(folder / "declared")
        # A top-level networks block beside a loading main, and one alone: the readers see nothing of it, and
        # Harbor's compose up applies it to the verifier project's default network. Refused for both verifiers.
        networked = "services:\n  main:\n    pull_policy: missing\nnetworks:\n  default:\n    internal: true\n"
        for name in ("declared", "inherited"):
            for text in (networked, "networks:\n  default:\n    internal: true\n"):
                (folder / name / "tests" / "docker-compose.yaml").write_text(text)
                assert verifier_compose_sidecars(folder / name) == []
                assert verifier_compose_main_overrides(folder / name, prebuilt=True) == []
                with pytest.raises(
                    HarborTaskError,
                    match=rf"{name}/tests/docker-compose.yaml: top-level networks is refused for the separate verifier: "
                    r"Harbor's compose up applies it to the verifier project and this server reads services alone$",
                ):
                    load_task(folder / name)
        # The class is closed, not one key at a time: volumes (inert without a reference on main) and a key Compose
        # does not know are refused alike, named in file order.
        (tests / "docker-compose.yaml").write_text("volumes:\n  data: {}\nservices: {}\n")
        with pytest.raises(HarborTaskError, match=r"declared/tests/docker-compose.yaml: top-level volumes is refused"):
            load_task(folder / "declared")
        (tests / "docker-compose.yaml").write_text(
            "services: {}\nnetworks:\n  default: {}\nsecrets:\n  s:\n    file: ./s\nconfigs: {}\nfuture_key: 1\n"
        )
        with pytest.raises(
            HarborTaskError,
            match=r"declared/tests/docker-compose.yaml: top-level networks, secrets, configs, future_key are refused for "
            r"the separate verifier: Harbor's compose up applies them to the verifier project",
        ):
            load_task(folder / "declared")
        # An empty file is an empty mapping (as validate_task_compose reads it) and a main-only file still loads,
        # with the keys that shape no container: name (Harbor passes --project-name itself), the obsolete version
        # and an x- extension nothing references.
        (tests / "docker-compose.yaml").write_text("")
        assert load_task(folder / "declared").verifier_image == "org/verifier:1"
        (tests / "docker-compose.yaml").write_text("services:\n  main:\n    pull_policy: missing\n")
        assert load_task(folder / "declared").verifier_image == "org/verifier:1"
        (tests / "docker-compose.yaml").write_text(
            "name: other\nversion: '3.8'\nx-common:\n  CHECK_MODE: strict\nservices:\n  main:\n    pull_policy: missing\n"
        )
        assert load_task(folder / "declared").verifier_image == "org/verifier:1"
        # A verifier prepare builds from tests/Dockerfile: the loader refuses the include before it looks for the
        # record (validate_task_compose refuses the same file at build, so there is none).
        write_separate_task(folder / "built", dockerfile=PULL_DOCKERFILE)
        (folder / "built" / "tests" / "other.yaml").write_text(included)
        (folder / "built" / "tests" / "docker-compose.yaml").write_text("include:\n  - other.yaml\nservices: {}\n")
        with pytest.raises(HarborTaskError, match=r"built/tests/docker-compose.yaml: include is refused"):
            load_task(folder / "built")
        # A top-level networks block on a built verifier: validate_task_compose reads nothing of it, so build_many
        # refuses it before any command (no bud names the tests/ context, no record is written) and the loader
        # refuses the task the same way.
        (folder / "built" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: ${CONTEXT_DIR}\n"
            "networks:\n  default:\n    internal: true\n"
        )
        with pytest.raises(HarborTaskError, match=r"built/tests/docker-compose.yaml: top-level networks is refused"):
            load_task(folder / "built")
        fake = FakeTool("buildah")
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake),
        )
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["declared"] and sorted(prepared.skipped) == [
            "built",
            "inherited",
        ]
        assert prepared.skipped["built"].startswith("Could not build the Dockerfile image(s) for task(s) built: ")
        assert "task(s) built: " in prepared.skipped["built"]
        assert "built/tests/docker-compose.yaml: top-level networks is refused" in prepared.skipped["built"]
        built_tests = str((folder / "built" / "tests").resolve())
        assert not any(built_tests in call for call in fake.calls)
        assert not (folder / BUILT_IMAGES_FILE).exists()  # the only build context was refused: nothing recorded

    def test_a_heredoc_tests_dockerfile_is_refused_by_the_preflight_naming_the_verifiers_file(
        self, tmp_path, monkeypatch
    ):
        """tests/Dockerfile takes main's validate and refusal path: the heredoc preflight here."""
        monkeypatch.delenv("CONTAINERS_CONF_OVERRIDE", raising=False)
        monkeypatch.delenv("CONTAINERS_REGISTRIES_CONF", raising=False)
        folder = tmp_path / "ds"
        heredoc = "FROM python:3.13-slim-bookworm\nRUN <<EOF\npip install pytest==8.4.1\nEOF\nCOPY . /tests/\n"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE, tests_dockerfile=heredoc)
        write_base_record(folder)
        runner = HeredocTool("buildah", heredocs=False)
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=runner)
        with pytest.raises(BuildError) as info:
            build_images(folder, builder=builder, log=lambda _: None)
        message = str(info.value)
        assert message.startswith(str((folder / "a" / "tests" / "Dockerfile").resolve()))
        assert "this host's buildah cannot build one" in message
        assert [call for call in runner.calls if any("hb__" in arg for arg in call)] == []

    def test_dockerfile_inline_in_tests_compose_is_refused_on_the_direct_path_for_the_verifiers_main(self, tmp_path):
        """The verifier entry builds Harbor's main of the tests/ context, so main's Compose-only keys are refused."""
        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        (folder / "a" / "tests" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: ${CONTEXT_DIR}\n"
            "      dockerfile_inline: |\n        FROM python:3.13\n        RUN pip install pytest\n"
        )
        (verifier,) = plan_builds(folder)
        assert verifier.verifier and verifier.compose_service == "main"
        fake = FakeTool("buildah")
        builder = ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake)
        with pytest.raises(
            BuildError, match=r"services\.main\.build\.dockerfile_inline is not supported by the direct buildah build"
        ):
            build_images(folder, builder=builder, log=lambda _: None)
        assert not any("bud" in call for call in fake.calls)

    def test_a_context_of_tasks_already_out_is_not_built(self, tmp_path, monkeypatch, capsys):
        """a and b share tests/ (one verifier context) but own distinct environment/ folders; the shared verifier
        build fails after a's environment/ is built and before b's. b's environment/ is then skipped: b is already
        out, and a build for it alone would be wasted and recorded against a skipped task."""
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_separate_task(folder / "a", extra={"tools/setup.sh": "#!/bin/sh\necho a\n"})
        write_separate_task(folder / "b", extra={"tools/setup.sh": "#!/bin/sh\necho b\n"})
        write_task(folder / "c", extra={"tools/setup.sh": "#!/bin/sh\necho c\n"})  # shared mode: main alone
        shared_tests = (folder / "a" / "tests").resolve()
        assert harbor_hash(shared_tests) == harbor_hash(folder / "b" / "tests")
        # Plan order: a's environment/, the shared tests/ (a, b), b's environment/, c's environment/.
        assert [(e.service, e.tasks) for e in plan_builds(folder)] == [
            ("main", ("a",)),
            ("verifier", ("a", "b")),
            ("main", ("b",)),
            ("main", ("c",)),
        ]
        fake = FakeTool("buildah", fail=str(shared_tests))
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake),
        )
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["c"]
        assert sorted(prepared.skipped) == ["a", "b"]
        assert prepared.skipped["b"].startswith("Could not build the Dockerfile image(s) for task(s) a, b: Building")
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert harbor_hash(folder / "a" / "environment") in built  # built before the shared verifier failed
        assert harbor_hash(folder / "b" / "environment") not in built  # never built: b was already out
        assert harbor_hash(shared_tests) not in built
        assert len([call for call in fake.calls if "bud" in call]) == 3  # a main, the failed verifier, c main
        assert "Built 2 image(s) with buildah, pushed 2" in capsys.readouterr().out

    def test_docker_builds_the_verifier_as_harbors_main_of_the_tests_context(self, tmp_path, monkeypatch):
        folder = tmp_path / "ds"
        write_separate_task(folder / "a", dockerfile=PULL_DOCKERFILE)
        write_base_record(folder)
        tests = (folder / "a" / "tests").resolve()
        compose_files: list[tuple[dict, dict]] = []

        def docker(argv, env=None):
            if argv[:2] == ["docker", "compose"] and "build" in argv:
                files = [Path(argv[i + 1]) for i, arg in enumerate(argv) if arg == "-f"]
                compose_files.append(tuple(yaml.safe_load(path.read_text()) for path in files))
                compose_files.append(({"argv": argv}, {"env": env}))
            return FakeTool("docker")(argv, env)

        (verifier,) = build_images(folder, builder=ImageBuilder(tool="docker", run=docker), log=lambda _: None)
        assert verifier.tag == f"hb__{verifier.digest}-verifier"
        (template, images), (meta, call_env) = compose_files
        # Harbor's template for the tests/ context; Gym's override tags its main with the verifier's name.
        assert template == harbor_compose_document(tests, f"hb__{verifier.digest}")
        assert images == {"services": {"main": {"image": verifier.tag}}}
        argv = meta["argv"]
        assert argv[argv.index("--project-directory") + 1] == str(tests)
        assert argv[argv.index("--project-name") + 1] == f"hb__{verifier.digest}"
        assert argv[-1] == "build"  # the whole project: the entry is its main
        assert call_env["env"]["CONTEXT_DIR"] == str(tests)
        assert call_env["env"]["MAIN_IMAGE_NAME"] == f"hb__{verifier.digest}"
        task = load_task(folder / "a")
        assert task.image == "ubuntu:24.04" and task.verifier_image == verifier.tag

    def test_a_failed_verifier_build_leaves_its_task_out_and_the_others_prepare(self, tmp_path, monkeypatch, capsys):
        from nemo_gym.data.tasks.harbor.prepare import prepare_target

        folder = tmp_path / "ds"
        write_separate_task(folder / "ok")
        write_separate_task(folder / "bad", tests_dockerfile="FROM python:3.13\nRUN false\n")
        failing = str((folder / "bad" / "tests").resolve())
        fake = FakeTool("buildah", fail=failing)
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="buildah", repository=REPO, storage_root=tmp_path / "store", run=fake),
        )
        prepared = prepare_target(str(folder), output_root=tmp_path / "out")
        assert [t.task_id for t in prepared.tasks] == ["ok"]
        (ok,) = prepared.tasks
        assert ok.verifier_image == f"{REPO}/hb__{harbor_hash(folder / 'ok' / 'tests')}-verifier@sha256:pushed01"
        assert ok.image == f"{REPO}/hb__{harbor_hash(folder / 'ok' / 'environment')}@sha256:pushed01"
        assert list(prepared.skipped) == ["bad"]
        assert prepared.skipped["bad"].startswith("Could not build the Dockerfile image(s) for task(s) bad: Building")
        assert f"hb__{harbor_hash(folder / 'bad' / 'tests')}-verifier" in prepared.skipped["bad"]
        assert json.loads((prepared.output_dir / "skipped.json").read_text()) == prepared.skipped
        # bad's main was built and recorded before its verifier failed; the failed build was attempted once.
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert harbor_hash(folder / "bad" / "environment") in built
        assert harbor_hash(folder / "bad" / "tests") not in built
        assert len([call for call in fake.calls if "bud" in call and failing in call]) == 1
        # The shared environment/ image and ok's verifier: bad's failed verifier build is not counted.
        assert "Built 2 image(s) with buildah, pushed 2" in capsys.readouterr().out


class TestSandboxProviderAtPrepare:
    """M28: prepare learns the deployment's provider name from the `sandbox:` block's keys, never its values."""

    def test_provider_name_is_read_from_the_block_keys_without_resolving_values(self, tmp_path, monkeypatch):
        import nemo_gym.data.tasks.harbor.prepare as harbor_cli
        from nemo_gym.cli.eval import sandbox_provider_from_overrides

        dotenv = tmp_path / "env.yaml"
        dotenv.write_text("some_key: 1\n")
        monkeypatch.setattr(build_options, "_resolve_under_cwd_or_install", lambda rel: tmp_path / rel)
        assert sandbox_provider_from_overrides([], None) is None
        # The block's values are never resolved, so a credential interpolation does not have to succeed here.
        dotenv.write_text(
            "sandbox:\n  opensandbox:\n    connection:\n      api_key: ${oc.env:NEMO_GYM_M28_NO_SUCH_KEY}\n"
            "  default_metadata: {team: t}\n"
        )
        monkeypatch.delenv("NEMO_GYM_M28_NO_SUCH_KEY", raising=False)
        assert sandbox_provider_from_overrides([], None) == "opensandbox"
        # `--sandbox <provider>` adds that provider file below the rest.
        dotenv.write_text("some_key: 1\n")
        provider_file = tmp_path / "docker.yaml"
        provider_file.write_text("sandbox:\n  docker:\n    create: {use_init: true}\n")
        monkeypatch.setattr(harbor_cli, "resolve_sandbox_config", lambda name: provider_file)
        monkeypatch.setattr(cli_eval, "resolve_sandbox_config", lambda name: provider_file)
        assert sandbox_provider_from_overrides([], "docker") == "docker"
        # A ++ override on the command line is part of the picture too.
        assert sandbox_provider_from_overrides(["++sandbox.opensandbox.connection.domain=x"], None) == "opensandbox"

    def test_run_and_validate_hand_the_provider_to_prepare(self, tmp_path, monkeypatch):
        import argparse

        import nemo_gym.data.tasks.harbor.prepare as harbor_cli

        seen = {}

        def fake_prepare(target, **kwargs):
            seen[target] = kwargs
            raise harbor_cli.HubError("stop here")

        monkeypatch.setattr(harbor_cli, "prepare_target", fake_prepare)
        monkeypatch.setattr(cli_eval, "prepare_target", fake_prepare)
        monkeypatch.setattr(cli_eval, "build_options_from_overrides", lambda overrides: build_options.BuildOptions())
        monkeypatch.setattr(cli_eval, "sandbox_provider_from_overrides", lambda overrides, sandbox: f"p:{sandbox}")
        monkeypatch.setattr(cli_eval, "gpu_sandbox_provider_from_overrides", lambda overrides, sandbox: f"g:{sandbox}")
        monkeypatch.setattr(cli_eval, "compose_platform_from_overrides", lambda overrides: "linux/arm64")
        # The M22 readers of the same blocks are stubbed too: the sandbox file this test names does not exist.
        monkeypatch.setattr(cli_eval, "volume_sources_from_overrides", lambda overrides, sandbox: None)
        monkeypatch.setattr(cli_eval, "gpu_volume_sources_from_overrides", lambda overrides, sandbox: None)
        monkeypatch.setattr(cli_eval, "shared_storage_from_overrides", lambda overrides, sandbox: None)
        monkeypatch.setattr(cli_eval, "gpu_shared_storage_from_overrides", lambda overrides, sandbox: None)
        monkeypatch.setattr(cli_eval, "registry_credentials_from_overrides", lambda overrides, sandbox: None)
        # The rows file is scoped to the agent, so the stub has to name one.
        monkeypatch.setattr(
            harbor_cli, "resolve_agent", lambda name: harbor_cli.AgentSelection(Path("a.yaml"), name, name)
        )
        monkeypatch.setattr(
            cli_eval, "resolve_agent", lambda name: harbor_cli.AgentSelection(Path("a.yaml"), name, name)
        )
        monkeypatch.setattr(harbor_cli, "resolve_sandbox_config", lambda name: tmp_path / f"{name}.yaml")
        monkeypatch.setattr(cli_eval, "resolve_sandbox_config", lambda name: tmp_path / f"{name}.yaml")
        parser = argparse.ArgumentParser()
        parser.error = lambda message: (_ for _ in ()).throw(SystemExit(message))
        args = argparse.Namespace(target="run-target", agent="hermes_agent", sandbox="docker", _parser=parser)
        with pytest.raises(SystemExit, match="stop here"):
            cli_eval.run_target(args, [])
        assert seen["run-target"]["sandbox_provider"] == "p:docker"
        assert seen["run-target"]["gpu_sandbox_provider"] == "g:docker"
        assert seen["run-target"]["compose_platform"] == "linux/arm64"
        args = argparse.Namespace(target="validate-target", sandbox=None, _parser=parser)
        with pytest.raises(SystemExit, match="stop here"):
            cli_eval.validate_target(args, [])
        assert seen["validate-target"]["sandbox_provider"] == "p:None"
        assert seen["validate-target"]["gpu_sandbox_provider"] == "g:None"
        assert seen["validate-target"]["compose_platform"] == "linux/arm64"

    def test_the_gpu_block_provider_and_the_platform_are_read_from_the_same_configuration(self, tmp_path, monkeypatch):
        """The server routes GPU sidecars to `gpu_sandbox_provider` and checks `platform` against
        `sandbox_provider_options.platform`; prepare reads both from the configuration the command names."""
        from nemo_gym.cli.eval import (
            GPU_SANDBOX_PROVIDER_KEY,
            SANDBOX_PROVIDER_OPTIONS_KEY,
            compose_platform_from_overrides,
            gpu_sandbox_provider_from_overrides,
        )

        assert GPU_SANDBOX_PROVIDER_KEY == "harbor_environment_server.environment_servers.harbor.gpu_sandbox_provider"
        dotenv = tmp_path / "env.yaml"
        dotenv.write_text(
            "sandbox:\n  opensandbox: {}\nsandbox_gpu:\n  docker:\n    create: {use_init: true}\n"
            "  default_metadata: {pool: h100}\n"
        )
        monkeypatch.setattr(build_options, "_resolve_under_cwd_or_install", lambda rel: tmp_path / rel)
        gpu_block = f"++{GPU_SANDBOX_PROVIDER_KEY}=sandbox_gpu"
        # No GPU block named: every service is checked against the sandbox: block. Named: that block's keys.
        assert gpu_sandbox_provider_from_overrides([], None) is None
        assert gpu_sandbox_provider_from_overrides([gpu_block], None) == "docker"
        # A GPU block that names no provider defers to the adapter, as the sandbox: block does.
        assert gpu_sandbox_provider_from_overrides([f"++{GPU_SANDBOX_PROVIDER_KEY}=sandbox_other"], None) is None
        with pytest.raises(ValueError, match="must name a top-level sandbox block"):
            gpu_sandbox_provider_from_overrides([f"++{GPU_SANDBOX_PROVIDER_KEY}=[a]"], None)
        # The platform: sandbox_provider_options.platform ({os, arch}), else linux/amd64, as the server reads it.
        assert compose_platform_from_overrides([]) == "linux/amd64"
        options = [
            f"++{SANDBOX_PROVIDER_OPTIONS_KEY}.platform.os=linux",
            f"++{SANDBOX_PROVIDER_OPTIONS_KEY}.platform.arch=aarch64",
        ]
        assert compose_platform_from_overrides(options) == "linux/arm64"
        # A block that is not a mapping selects no platform; the server's own validation names it at dispatch.
        assert compose_platform_from_overrides([f"++{SANDBOX_PROVIDER_OPTIONS_KEY}=x"]) == "linux/amd64"

    def test_the_platform_is_read_without_resolving_the_other_provider_options(self, tmp_path, monkeypatch):
        """Review: an OpenSandbox volume with `host.path: ${results_dir}/task-data` is a supported option whose
        default the server's configuration parser supplies at dispatch; run and validate read the platform before
        prepare_target, so only `sandbox_provider_options.platform` is resolved and the sibling volume is left."""
        from nemo_gym.cli.eval import SANDBOX_PROVIDER_OPTIONS_KEY, compose_platform_from_overrides
        from nemo_gym.config.global_config import ConfigInterpolationError
        from nemo_gym.data.tasks.harbor.build_options import _block_from_overrides

        block: object = {
            "platform": {"os": "linux", "arch": "aarch64"},
            "volumes": [{"host": {"path": "${results_dir}/task-data"}, "container": {"path": "/data"}}],
        }
        for part in reversed(SANDBOX_PROVIDER_OPTIONS_KEY.split(".")):
            block = {part: block}
        (tmp_path / "env.yaml").write_text(yaml.safe_dump(block))
        monkeypatch.setattr(build_options, "_resolve_under_cwd_or_install", lambda rel: tmp_path / rel)
        # Nothing sets results_dir before the server's parser runs, so the block as a whole does not resolve here.
        with pytest.raises(ConfigInterpolationError):
            _block_from_overrides([], SANDBOX_PROVIDER_OPTIONS_KEY)
        assert compose_platform_from_overrides([]) == "linux/arm64"
        # A command-line platform beside the same unresolved volume.
        override = f"++{SANDBOX_PROVIDER_OPTIONS_KEY}.platform.arch=x86_64"
        assert compose_platform_from_overrides([override]) == "linux/amd64"
