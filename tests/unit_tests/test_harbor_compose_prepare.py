# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""D5: the sandbox Compose adapter's field check runs at prepare, per task, on the document the sandbox receives,
where a service's ``pull_policy`` is its runtime pull policy and only a ``build`` policy leaves with the build it
names. With M28 the check is ``check_task_compose_fields``, whose first step is the key allow-list; a refused key is
a skipped task."""

import json
from pathlib import Path

import pytest

import nemo_gym.tasks.harbor.prepare_builds as prepare_builds
from nemo_gym.tasks.harbor import image_configs
from nemo_gym.tasks.harbor.build_options import BuildOptions
from nemo_gym.tasks.harbor.build_plan import DEFAULT_PLATFORM, BuildError, plan_builds
from nemo_gym.tasks.harbor.build_records import build_inputs_digest, settle_entry
from nemo_gym.tasks.harbor.cli import prepare_target
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.image_configs import (
    check_task_compose_fields,
    load_compose_document,
    record_compose_images,
    without_pull_policy,
    without_restated_main_build,
)
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE
from nemo_gym.tasks.harbor.registry import ImageConfigError
from nemo_gym.tasks.harbor.task import discover_tasks, load_task


PULL_DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"
OVERLAY_DOCKERFILE = "FROM ubuntu:24.04\nRUN echo hi > /hi\nWORKDIR /app\n"
BUILD_DOCKERFILE = "FROM ubuntu:24.04\nCOPY tools/ /opt/tools/\nWORKDIR /app\n"
AGENT_IMAGE = "harborframework/harbor-index:cybergym-agent@sha256:" + "a" * 64
TASK_SERVER_IMAGE = "harborframework/harbor-index:cybergym-task-server@sha256:" + "c" * 64

# The three hub shapes D_TESTS reproduced: featurebench 265e447753be, rh-swe-bench 0884ee3e43b7, harbor-index 9d8c77f891ab.
# M28 gives ``deploy``, ``extra_hosts`` and ``platform`` their meaning, so only featurebench's ``count: all`` is
# still refused, in the check's own words; a key no provider starts (``privileged``) is the allow-list's refusal.
FEATUREBENCH = """services:
  main:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""
RH_SWE_BENCH = """services:
  main:
    extra_hosts:
      - "pypi.org:0.0.0.0"
      - "github.com:0.0.0.0"
"""
HARBOR_INDEX = f"""services:
  main:
    depends_on:
      task-server:
        condition: service_healthy

  task-server:
    image: {TASK_SERVER_IMAGE}
    platform: linux/amd64
    environment:
      - AUTH_TOKEN=96f3e34ef9fdbd539b72b2931eae61fb
    expose:
      - "9111"
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4096M
    healthcheck:
      test: ["CMD", "python3", "-c", "print()"]
      interval: 3s
      timeout: 5s
      retries: 20
      start_period: 5s
"""
# Harbor's own template puts pull_policy: build on main; a task file may set it on a sidecar too.
PULL_POLICIES = """services:
  main:
    image: ${MAIN_IMAGE_NAME}
    pull_policy: build
    build:
      context: ${CONTEXT_DIR}
      target: prod
  api:
    image: org/api:1
    pull_policy: always
"""
SIDECAR_CACHE = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  cache:\n    image: redis:7-alpine\n"
SIDECAR_BUILD = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  api:\n    image: org/api:1\n    build: .\n"
PRIVILEGED = "services:\n  main:\n    privileged: true\n"


class StubRegistry:
    """Answers every reference with a Linux/amd64 record, so no test reaches a registry."""

    def image_config(self, reference: str) -> dict:
        return {
            "architecture": "amd64",
            "config": {"Cmd": ["bash"]},
            "config_digest": "sha256:c",
            "image": reference,
            "os": "linux",
            "record_version": RECORD_VERSION,
        }


@pytest.fixture(autouse=True)
def stub_registry(monkeypatch):
    monkeypatch.setattr(image_configs, "RegistryClient", StubRegistry)


def write_task(
    root: Path, name: str, *, compose: str, dockerfile: str | None = PULL_DOCKERFILE, docker_image: str | None = None
) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    declared = f'docker_image = "{docker_image}"\n' if docker_image else ""
    (task / "task.toml").write_text(f'schema_version = "1.4"\n\n[environment]\n{declared}')
    (task / "instruction.md").write_text("Do the thing.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (task / "environment" / "docker-compose.yaml").write_text(compose)
    if dockerfile is not None:
        (task / "environment" / "Dockerfile").write_text(dockerfile)
    if dockerfile == BUILD_DOCKERFILE:
        (task / "environment" / "tools").mkdir()
        (task / "environment" / "tools" / "setup.sh").write_text("#!/bin/sh\necho ok\n")
    return task


def record_built(folder: Path, task: Path, *, architecture: str = "amd64") -> str:
    """The records prepare writes after building ``task``'s context (``built-images.json`` and the image's
    configuration in ``compose-images.json``): the task then loads in pull mode from it."""
    digest = context_digest(task / "environment")
    image = f"registry.example/ds/hb__{digest}@sha256:" + "d" * 64
    path = folder / BUILT_IMAGES_FILE
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    recorded[digest] = {"image": image, "tasks": [task.name], "tool": "docker", "pushed": True}
    path.write_text(json.dumps(recorded))
    configs_path = folder / IMAGE_CONFIGS_FILE
    configs = json.loads(configs_path.read_text()) if configs_path.is_file() else {}
    configs[image] = {
        "os": "linux",
        "architecture": architecture,
        "image": image,
        "config": {},
        "record_version": RECORD_VERSION,
    }
    entry = next(entry for entry in plan_builds(task) if entry.is_main)
    try:
        # The digest a build of the context records, so prepare reuses this record rather than rebuilding it.
        configs[image]["inputs_digest"] = build_inputs_digest(
            settle_entry(entry, repository=None, platform=DEFAULT_PLATFORM)
        )
    except BuildError:
        # A Compose file the build would refuse (a sidecar build): the record stands in for a build Gym never ran.
        pass
    configs_path.write_text(json.dumps(configs))
    return image


UNBUILT_NOTE = " cannot be applied in pull or overlay mode (Gym does not build this Dockerfile)"
GOOD = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"


class TestWithoutPullPolicy:
    def test_drops_the_key_from_every_service_and_nothing_else(self):
        document = {
            "services": {
                "main": {"image": "a", "pull_policy": "build", "build": {"context": "."}},
                "api": {"image": "b", "pull_policy": "always", "expose": ["80"]},
                "db": {"image": "c"},
                "nil": None,
            },
            "volumes": {"data": {}},
        }
        untouched = document["services"]["db"]
        result = without_pull_policy(document)
        assert result is document
        assert result["services"]["main"] == {"image": "a", "build": {"context": "."}}
        assert result["services"]["api"] == {"image": "b", "expose": ["80"]}
        assert result["services"]["db"] is untouched and result["services"]["nil"] is None
        assert result["volumes"] == {"data": {}}

    @pytest.mark.parametrize("document", [{}, {"services": None}, {"services": []}, "text"])
    def test_a_document_without_services_mappings_is_returned_as_is(self, document):
        assert without_pull_policy(document) == document


class TestCheckTaskComposeFields:
    def test_a_task_without_a_compose_file_is_not_a_compose_task(self, tmp_path):
        task = write_task(tmp_path, "plain", compose="")
        (task / "environment" / "docker-compose.yaml").unlink()
        image_configs.record_base_images(tmp_path)
        assert load_compose_document(load_task(task)) is None
        assert record_compose_images([load_task(task)], tmp_path) is None

    def test_a_sidecar_build_is_still_a_refused_field(self, tmp_path):
        task = write_task(tmp_path, "ctr", compose=SIDECAR_BUILD, dockerfile=BUILD_DOCKERFILE)
        record_built(tmp_path, task)
        image_configs.record_base_images(tmp_path)
        (loaded,) = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError) as info:
            check_task_compose_fields(loaded, load_compose_document(loaded))
        assert (
            str(info.value)
            == "ctr: environment/docker-compose.yaml: Service 'api': unsupported Compose fields ['build']"
        )

    def test_malformed_network_and_hook_values_are_skipped_tasks_not_an_aborted_prepare(self, tmp_path):
        """Review: ``networks: {front: 1}`` on a service escaped the check as a ``TypeError``, a top-level
        ``networks: {front: bridge}`` as a ``ValueError`` and a hook ``environment: A=1`` as the interpolation
        error, each aborting ``record_compose_images`` for the whole dataset. Each is the field's own error now, so
        the task is recorded under ``skipped`` with the field named and its valid sibling is prepared."""
        main = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
        write_task(tmp_path, "service-network", compose=main + "    networks: {front: 1}\nnetworks:\n  front: {}\n")
        write_task(tmp_path, "top-network", compose=main + "networks:\n  front: bridge\n")
        write_task(
            tmp_path, "hook-env", compose=main + '    post_start:\n      - command: "true"\n        environment: A=1\n'
        )
        write_task(tmp_path, "good", compose=main)
        image_configs.record_base_images(tmp_path)
        tasks = sorted(discover_tasks(tmp_path), key=lambda task: task.task_id)
        skipped: dict[str, str] = {}

        path = record_compose_images(tasks, tmp_path, skipped=skipped)

        assert path is not None and tmp_path.joinpath("compose-images.json").is_file()
        assert skipped == {
            "service-network": "service-network: environment/docker-compose.yaml: service 'main' networks.front "
            "must be a mapping, got 1",
            "top-network": "top-network: environment/docker-compose.yaml: network 'front' must be a mapping, got "
            "'bridge'",
            "hook-env": "hook-env: environment/docker-compose.yaml: service 'main' post_start[0].environment must "
            "be a list or a mapping",
        }
        with pytest.raises(ImageConfigError, match="3 Compose task"):
            record_compose_images(tasks, tmp_path)

    def test_malformed_deploy_memory_is_a_skipped_task_not_an_aborted_prepare(self, tmp_path):
        """Review: ``deploy: {resources: {limits: {memory: 1.5}}}`` escaped ``_bytes`` as an ``AttributeError`` (its
        string branch lower-cases), past the field check's own handler and the per-task boundary, so one such task
        aborted ``record_compose_images`` for the dataset. It is the field's error now: the task is recorded under
        ``skipped`` with the field named, a mapping value too, and the valid sibling is prepared."""
        main = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
        write_task(tmp_path, "float-memory", compose=main + "    deploy: {resources: {limits: {memory: 1.5}}}\n")
        write_task(
            tmp_path, "mapping-memory", compose=main + "    deploy: {resources: {limits: {memory: {size: 1g}}}}\n"
        )
        write_task(tmp_path, "good", compose=main + "    deploy: {resources: {limits: {memory: 512M}}}\n")
        image_configs.record_base_images(tmp_path)
        tasks = sorted(discover_tasks(tmp_path), key=lambda task: task.task_id)
        skipped: dict[str, str] = {}

        path = record_compose_images(tasks, tmp_path, skipped=skipped)

        assert path is not None and tmp_path.joinpath("compose-images.json").is_file()
        assert skipped == {
            "float-memory": "float-memory: environment/docker-compose.yaml: deploy.resources.limits.memory must be a "
            "memory size such as 512M, got 1.5",
            "mapping-memory": "mapping-memory: environment/docker-compose.yaml: deploy.resources.limits.memory must "
            "be a memory size such as 512M, got {'size': '1g'}",
        }
        with pytest.raises(ImageConfigError, match="2 Compose task"):
            record_compose_images(tasks, tmp_path)

    def test_an_out_of_range_port_target_is_a_skipped_task_not_a_failed_seed(self, tmp_path):
        """Review: ``ports: ["0"]`` and ``ports: [{target: 65536}]`` passed the field check, so the task prepared and
        failed at every seed, where ``SandboxSpec`` requires ports from 1 through 65535. The check applies the spec's
        bounds to the container target now, short and long form alike, and the task is recorded under ``skipped``
        naming the service and ``ports``; ``published: 0`` and the bounds themselves still prepare."""
        main = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
        write_task(tmp_path, "zero-short", compose=main + '    ports: ["0"]\n')
        write_task(tmp_path, "high-long", compose=main + "    ports: [{target: 65536}]\n")
        write_task(tmp_path, "good", compose=main + '    ports: ["1", "0:65535", {target: 8080, published: 0}]\n')
        image_configs.record_base_images(tmp_path)
        tasks = sorted(discover_tasks(tmp_path), key=lambda task: task.task_id)
        skipped: dict[str, str] = {}

        path = record_compose_images(tasks, tmp_path, skipped=skipped)

        assert path is not None and tmp_path.joinpath("compose-images.json").is_file()
        assert skipped == {
            "zero-short": "zero-short: environment/docker-compose.yaml: service 'main' ports: target must be between "
            "1 and 65535, got 0",
            "high-long": "high-long: environment/docker-compose.yaml: service 'main' ports: target must be between "
            "1 and 65535, got 65536",
        }
        with pytest.raises(ImageConfigError, match="2 Compose task"):
            record_compose_images(tasks, tmp_path)

    def test_an_interpolated_gpu_count_prepares_as_the_integer_it_resolves_to(self, tmp_path):
        """Review: ``count: "${GPUS:-1}"`` resolves to the string ``"1"`` (the loader substitutes strings and the type
        restoration covers healthchecks alone), which the check refused as a non-integer count, skipping a valid
        GPU-sidecar task. The resolved document now normalizes to ``gpus: 1`` and is a fixed point of the check."""
        compose = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  model:\n    image: org/model:1\n    deploy:\n      resources:\n        reservations:\n"
            '          devices:\n            - capabilities: [gpu]\n              count: "${GPUS:-1}"\n'
        )
        write_task(tmp_path, "gpu-sidecar", compose=compose)
        image_configs.record_base_images(tmp_path)
        (task,) = discover_tasks(tmp_path)
        document = load_compose_document(task)
        assert document["services"]["model"]["deploy"]["resources"]["reservations"]["devices"][0]["count"] == "1"

        checked = check_task_compose_fields(task, document, provider="docker", gpu_provider="opensandbox")

        assert checked["services"]["model"]["deploy"]["resources"]["reservations"] == {"gpus": 1}
        assert check_task_compose_fields(task, json.loads(json.dumps(checked)), gpu_provider="opensandbox") == checked


class TestRestatedMainBuild:
    """A pull or overlay task whose file only restates ``main``'s build (Harbor's own template supplies it, so the
    restatement is a no-op there: orca-bench, terminal_bench_aai, data-eng-bench) starts; a build key Gym cannot
    apply without a build is that task's reason, naming the key."""

    @pytest.mark.parametrize(
        ("dockerfile", "build"),
        [
            (PULL_DOCKERFILE, '{context: "${CONTEXT_DIR}", pull: true}'),
            (OVERLAY_DOCKERFILE, "{dockerfile: Dockerfile}"),
            (PULL_DOCKERFILE, "{context: ., dockerfile: ./Dockerfile}"),
            (OVERLAY_DOCKERFILE, "."),
        ],
    )
    def test_a_restated_build_is_dropped_and_the_task_prepares(self, tmp_path, dockerfile, build):
        folder = tmp_path / "ds"
        folder.mkdir()
        compose = f"services:\n  main:\n    image: ${{MAIN_IMAGE_NAME}}\n    build: {build}\n"
        write_task(folder, "restated", compose=compose, dockerfile=dockerfile)
        image_configs.record_base_images(folder)
        (task,) = discover_tasks(folder)
        assert task.built_image is None and bool(task.overlay) == (dockerfile is OVERLAY_DOCKERFILE)
        document, reason = without_restated_main_build(task, load_compose_document(task))
        assert reason is None and document["services"]["main"] == {"image": task.image}
        assert "build" not in check_task_compose_fields(task, load_compose_document(task))["services"]["main"]
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["restated"] and prepared.skipped == {}
        assert len(prepared.rows_path.read_text().splitlines()) == 1

    @pytest.mark.parametrize("build", ["    build: .\n", ""], ids=["with-restated-build", "without-build"])
    def test_mains_build_pull_policy_restates_the_template_and_any_other_policy_stays(self, tmp_path, build):
        """Harbor's build template gives ``main`` ``pull_policy: build`` with its ``build:``, so a pull or overlay task
        repeating it restates the template, with or without the ``build:`` beside it, and the key leaves with the
        section. A runtime policy on ``main`` or a sidecar is kept, in the adapter's spelling, for the adapter to
        apply (review: ``pull_policy`` was dropped from every service before the check, so the advertised policies
        reached no sandbox)."""
        restating = f"services:\n  main:\n    image: ${{MAIN_IMAGE_NAME}}\n{build}    pull_policy: build\n"
        runtime = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: if_not_present\n"
            "  api:\n    image: org/api:1\n    pull_policy: never\n"
        )
        write_task(tmp_path, "restating", compose=restating, dockerfile=PULL_DOCKERFILE)
        write_task(tmp_path, "runtime", compose=runtime, dockerfile=OVERLAY_DOCKERFILE)
        image_configs.record_base_images(tmp_path)
        tasks = {task.task_id: task for task in discover_tasks(tmp_path)}
        document, reason = without_restated_main_build(tasks["restating"], load_compose_document(tasks["restating"]))
        assert reason is None and document["services"]["main"] == {"image": tasks["restating"].image}
        checked = check_task_compose_fields(tasks["runtime"], load_compose_document(tasks["runtime"]))
        assert checked["services"]["main"]["pull_policy"] == "missing"
        assert checked["services"]["api"]["pull_policy"] == "never"
        prepared = prepare_target(str(tmp_path), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["restating", "runtime"] and prepared.skipped == {}

    @pytest.mark.parametrize(
        ("build", "keys"),
        [
            ('{context: "${CONTEXT_DIR}", args: {X: "1"}}', "main.build.args"),
            ("{target: prod, dockerfile_inline: 'FROM x'}", "main.build.dockerfile_inline, main.build.target"),
            ("{dockerfile: Dockerfile.gpu}", "main.build.dockerfile"),
            ("{context: ../other}", "main.build.context"),
        ],
    )
    def test_a_build_key_gym_cannot_apply_without_a_build_is_the_reason(self, tmp_path, build, keys):
        folder = tmp_path / "ds"
        folder.mkdir()
        compose = f"services:\n  main:\n    image: ${{MAIN_IMAGE_NAME}}\n    build: {build}\n"
        write_task(folder, "unbuilt", compose=compose, dockerfile=PULL_DOCKERFILE)
        write_task(folder, "good", compose=GOOD)
        image_configs.record_base_images(folder)
        task = next(t for t in discover_tasks(folder) if t.task_id == "unbuilt")
        document = load_compose_document(task)
        assert without_restated_main_build(task, document) == (document, f"{keys}{UNBUILT_NOTE}")
        assert "build" in document["services"]["main"]
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["good"]
        assert prepared.skipped == {"unbuilt": f"unbuilt: environment/docker-compose.yaml: {keys}{UNBUILT_NOTE}"}

    def test_a_built_task_and_a_prebuilt_image_are_left_to_the_other_rules(self, tmp_path):
        built = write_task(tmp_path, "built", compose=PULL_POLICIES, dockerfile=BUILD_DOCKERFILE)
        record_built(tmp_path, built)
        prebuilt_compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build: {target: prod}\n"
        write_task(tmp_path, "prebuilt", compose=prebuilt_compose, dockerfile=None, docker_image=AGENT_IMAGE)
        write_task(tmp_path, "pulled", compose=PULL_POLICIES)
        image_configs.record_base_images(tmp_path)
        tasks = {task.task_id: task for task in discover_tasks(tmp_path)}
        for task_id in ("built", "prebuilt"):
            document = load_compose_document(tasks[task_id])
            assert without_restated_main_build(tasks[task_id], document) == (document, None)
        # The loader already put the recorded image in for the built main, its build: and pull_policy gone
        # (with_built_images); the prebuilt task's main keeps its section for the field check to report.
        built_document = load_compose_document(tasks["built"])
        assert built_document["services"]["main"] == {"image": tasks["built"].built_image}
        assert "build" in load_compose_document(tasks["prebuilt"])["services"]["main"]
        # The document checked is the one the sandbox receives: the built task's main build section and its
        # pull_policy: build are gone from it, so they are not reported; the sidecar's runtime policy stays, in the
        # adapter's spelling, for the adapter to apply (review: the advertised policies reached no sandbox).
        checked = check_task_compose_fields(tasks["built"], built_document)
        assert "build" not in checked["services"]["main"] and "pull_policy" not in checked["services"]["main"]
        assert checked["services"]["api"]["pull_policy"] == "always"
        # The same file on a task that was not built keeps a main build with a target Gym cannot apply.
        with pytest.raises(ImageConfigError) as info:
            check_task_compose_fields(tasks["pulled"], load_compose_document(tasks["pulled"]))
        assert str(info.value) == "pulled: environment/docker-compose.yaml: main.build.target" + UNBUILT_NOTE
        # The loader's own document is untouched by the check.
        assert "pull_policy" in load_compose_document(tasks["pulled"])["services"]["api"]


class TestPrepareChecksTheFieldsBeforeBuilding:
    """The key allow-list, the field check's first step, also runs on the raw file of every task about to be built,
    so a task refused by key costs no build or push and a build that would fail for it does not stop the
    preparation. A value the check refuses is read from the resolved document once the tasks load, after their
    builds (see ``test_a_value_the_check_refuses_is_read_after_the_build_and_still_leaves_the_task_out``)."""

    @staticmethod
    def docker_tool(monkeypatch):
        """prepare_target's builder as a docker FakeTool, whatever ``builds.tool`` says."""
        from nemo_gym.tasks.harbor.build_tools import ImageBuilder
        from tests.unit_tests.harbor_build_helpers import FakeTool

        tool = FakeTool("docker")
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="docker", run=tool, **{k: v for k, v in kw.items() if k != "tool"}),
        )
        return tool

    @staticmethod
    def write_with_sidecar(folder: Path, name: str, main_build: str | None, **kwargs) -> Path:
        """A task whose Compose file has ``main`` with ``build: main_build`` (none when ``None``) and a sidecar ``api``
        the plan builds from ``environment/api/Dockerfile``."""
        build = f"    build: {main_build}\n" if main_build is not None else ""
        compose = f"services:\n  main:\n    image: ${{MAIN_IMAGE_NAME}}\n{build}  api:\n    build: ./api\n"
        task = write_task(folder, name, compose=compose, **kwargs)
        (task / "environment" / "api").mkdir()
        (task / "environment" / "api" / "Dockerfile").write_text("FROM postgres:16\n")
        return task

    @pytest.mark.parametrize("dockerfile", [PULL_DOCKERFILE, OVERLAY_DOCKERFILE], ids=["pull", "overlay"])
    @pytest.mark.parametrize(
        "build", ["{context: .}", '{context: "${CONTEXT_DIR}", pull: true}', ".", "{dockerfile: Dockerfile}"]
    )
    def test_a_restated_main_build_beside_a_built_sidecar_passes_the_preflight_and_the_task_prepares(
        self, tmp_path, monkeypatch, dockerfile, build
    ):
        """Review (M24): a pull or overlay ``main`` whose ``build:`` only restates Harbor's own template, beside a
        sidecar the plan builds. The plan holds the sidecar only, so the preflight left main's section in place and
        refused the whole task for ``main.build`` before any build, while the loader accepts the same file
        (``without_restated_main_build``; orca-bench, terminal_bench_aai and data-eng-bench ship the restatement). The
        same rule drops it here, ``${CONTEXT_DIR}`` read as the raw file spells it: the sidecar is built and recorded,
        the task loads, and the document the sandbox receives has no ``build`` on main."""
        folder = tmp_path / "ds"
        folder.mkdir()
        task = self.write_with_sidecar(folder, "t", build, dockerfile=dockerfile)
        image_configs.record_base_images(folder)  # as prepare does before planning: the overlay's base is known
        plan = plan_builds(folder)
        assert [(entry.service, entry.tasks) for entry in plan] == [("api", ("t",))]
        assert prepare_builds.refused_before_build(plan) == {}
        tool = self.docker_tool(monkeypatch)

        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(repository="registry.example/r")
        )

        assert prepared.skipped == {} and [t.task_id for t in prepared.tasks] == ["t"]
        (loaded,) = prepared.tasks
        assert loaded.built_image is None and bool(loaded.overlay) == (dockerfile is OVERLAY_DOCKERFILE)
        digest = context_digest(task / "environment")
        assert any("build" in call and str(task / "environment") in " ".join(call) for call in tool.calls)
        from nemo_gym.tasks.harbor.task import built_service_images

        assert built_service_images(loaded.path) == {"api": f"registry.example/r/hb__{digest}-api@sha256:pushed02"}
        # The loader's own rule agrees with the preflight: the restatement is dropped, no reason, and the sidecar runs
        # from its recorded image.
        document, reason = without_restated_main_build(loaded, load_compose_document(loaded))
        assert reason is None and document["services"]["main"] == {"image": loaded.image}
        assert document["services"]["api"] == {"image": f"registry.example/r/hb__{digest}-api@sha256:pushed02"}

    def test_a_main_build_key_gym_cannot_apply_is_the_reason_before_the_sidecar_builds(self, tmp_path, monkeypatch):
        """The loader's rule, with the loader's wording: ``args`` would change the image Harbor builds and has no
        equivalent without a build, so the task is left out before its sidecar costs a build or a push."""
        folder = tmp_path / "ds"
        folder.mkdir()
        task = self.write_with_sidecar(folder, "t", '{context: ., args: {X: "1"}}')
        write_task(folder, "good", compose=GOOD)
        expected = {"t": "t: environment/docker-compose.yaml: main.build.args" + UNBUILT_NOTE}
        refused = prepare_builds.refused_before_build(plan_builds(folder))
        assert {task_id: str(error) for task_id, error in refused.items()} == expected
        tool = self.docker_tool(monkeypatch)

        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(repository="registry.example/r")
        )

        assert prepared.skipped == expected and [t.task_id for t in prepared.tasks] == ["good"]
        assert not any("build" in call and str(task / "environment") in " ".join(call) for call in tool.calls)
        assert not any("push" in call for call in tool.calls)
        assert not (folder / BUILT_IMAGES_FILE).exists()

    def test_a_prebuilt_main_with_a_build_section_beside_a_sidecar_is_still_refused_by_the_allow_list(
        self, tmp_path, monkeypatch
    ):
        """Unchanged: a prebuilt ``main`` (``docker_image``, no Dockerfile) has no build of Harbor's to restate, so its
        section stays and the allow-list reports it, before the sidecar builds."""
        folder = tmp_path / "ds"
        folder.mkdir()
        task = self.write_with_sidecar(folder, "prebuilt", "{target: prod}", dockerfile=None, docker_image=AGENT_IMAGE)
        write_task(folder, "good", compose=GOOD)
        allow_list = "Service 'main': unsupported Compose fields ['build']"
        refused = prepare_builds.refused_before_build(plan_builds(folder))
        assert list(refused) == ["prebuilt"] and allow_list in str(refused["prebuilt"])
        tool = self.docker_tool(monkeypatch)
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(repository="registry.example/r")
        )
        assert list(prepared.skipped) == ["prebuilt"] and allow_list in prepared.skipped["prebuilt"]
        assert [t.task_id for t in prepared.tasks] == ["good"]
        assert not any(str(task / "environment") in " ".join(call) for call in tool.calls)

    def test_a_refused_task_is_neither_built_nor_pushed_and_its_failing_build_stops_nothing(
        self, tmp_path, monkeypatch
    ):
        from nemo_gym.tasks.harbor.build_tools import ImageBuilder
        from tests.unit_tests.harbor_build_helpers import FakeTool

        folder = tmp_path / "ds"
        folder.mkdir()
        bad = write_task(folder, "privileged", compose=PRIVILEGED, dockerfile=BUILD_DOCKERFILE)
        fine = write_task(folder, "fine", compose=PULL_POLICIES, dockerfile=BUILD_DOCKERFILE)
        write_task(folder, "good", compose=GOOD)
        tool = FakeTool("docker", fail=str(bad / "environment"))
        monkeypatch.setattr(
            prepare_builds,
            "ImageBuilder",
            lambda **kw: ImageBuilder(tool="docker", run=tool, **{k: v for k, v in kw.items() if k != "tool"}),
        )

        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(repository="registry.example/r")
        )

        assert [task.task_id for task in prepared.tasks] == ["fine", "good"]
        assert len(prepared.rows_path.read_text().splitlines()) == 2
        expected = {
            "privileged": "privileged: environment/docker-compose.yaml: "
            "Service 'main': unsupported Compose fields ['privileged']"
        }
        assert prepared.skipped == expected
        assert json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text()) == expected
        # The refused context never reached the tool; the other build ran and was pushed.
        assert not any(str(bad / "environment") in " ".join(call) for call in tool.calls)
        assert any("build" in call and str(fine / "environment") in " ".join(call) for call in tool.calls)
        assert any("push" in call for call in tool.calls)
        recorded = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert [entry["tasks"] for entry in recorded.values()] == [["fine"]]

    def test_a_value_the_check_refuses_is_read_after_the_build_and_still_leaves_the_task_out(
        self, tmp_path, monkeypatch
    ):
        """The screen before the build reads keys only: featurebench's ``count: all`` is a value, which interpolation
        could still change in the raw file, so its task is built and pushed like any other and then refused by the
        check on the resolved document, with the field named, and left out of the rows (review: the value-level case
        the key-level one replaced, and the guarantee the README now states)."""
        folder = tmp_path / "ds"
        folder.mkdir()
        bad = write_task(folder, "featurebench", compose=FEATUREBENCH, dockerfile=BUILD_DOCKERFILE)
        write_task(folder, "good", compose=GOOD)
        assert prepare_builds.refused_before_build(plan_builds(folder)) == {}
        tool = self.docker_tool(monkeypatch)

        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(repository="registry.example/r")
        )

        assert [task.task_id for task in prepared.tasks] == ["good"]
        assert len(prepared.rows_path.read_text().splitlines()) == 1
        assert list(prepared.skipped) == ["featurebench"]
        assert prepared.skipped["featurebench"].startswith(
            "featurebench: environment/docker-compose.yaml: deploy.resources.reservations.devices"
        )
        assert "count: all is not supported" in prepared.skipped["featurebench"]
        assert json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text()) == prepared.skipped
        # The key-level guarantee only: the refused value did not spare its context the build and the push.
        assert any("build" in call and str(bad / "environment") in " ".join(call) for call in tool.calls)
        assert any("push" in call for call in tool.calls)
        recorded = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert [entry["tasks"] for entry in recorded.values()] == [["featurebench"]]

    def test_with_builds_disabled_the_field_is_still_the_reason(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        write_task(folder, "privileged", compose=PRIVILEGED, dockerfile=BUILD_DOCKERFILE)
        write_task(folder, "good", compose=GOOD)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [task.task_id for task in prepared.tasks] == ["good"]
        assert prepared.skipped == {
            "privileged": "privileged: environment/docker-compose.yaml: "
            "Service 'main': unsupported Compose fields ['privileged']"
        }


class TestPreparedImages:
    def test_images_list_sidecars_and_the_separate_verifier_image(self, tmp_path):
        """The pre-pull manifest names every image a run starts, not only the task images."""
        folder = tmp_path / "ds"
        folder.mkdir()
        task = write_task(folder, "svc", compose=SIDECAR_CACHE)
        (task / "task.toml").write_text(
            'schema_version = "1.4"\n\n[verifier]\nenvironment_mode = "separate"\n\n'
            '[verifier.environment]\ndocker_image = "org/verifier:1"\n\n[environment]\n'
        )
        write_task(folder, "plain", compose=GOOD)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["plain", "svc"]
        assert prepared.images == ("org/verifier:1", "redis:7-alpine", "ubuntu:24.04")
        assert json.loads((tmp_path / "out" / "ds" / "images.json").read_text()) == list(prepared.images)

    def test_a_sidecar_image_is_listed_as_its_record_pins_it(self, tmp_path):
        from nemo_gym.tasks.harbor.cli import run_images

        folder = tmp_path / "ds"
        folder.mkdir()
        write_task(folder, "svc", compose=SIDECAR_CACHE)
        (task,) = discover_tasks(folder)
        pinned = "docker.io/library/redis@sha256:" + "e" * 64
        records = {
            "redis:7-alpine": {
                "os": "linux",
                "architecture": "amd64",
                "image": pinned,
                "config": {},
                "record_version": RECORD_VERSION,
            }
        }
        assert run_images([task], records) == (pinned, "ubuntu:24.04")
        assert run_images([task], {}) == ("redis:7-alpine", "ubuntu:24.04")


class TestPrepareReportsRefusedFieldsPerTask:
    def test_the_hub_shapes_prepare_or_are_skipped_naming_the_field(self, tmp_path, caplog):
        """M28 gives ``extra_hosts`` (rh-swe-bench) and ``platform`` with ``deploy`` limits (harbor-index) their
        meaning, so those tasks prepare; featurebench's ``count: all`` and a key no provider starts are each a
        skipped task with the field named, as an unloadable task is, not a seed failure of every episode.
        featurebench and privileged carry a stale built-image record: the record does not put a refused task in
        the rows."""
        folder = tmp_path / "ds"
        folder.mkdir()
        record_built(folder, write_task(folder, "featurebench", compose=FEATUREBENCH, dockerfile=BUILD_DOCKERFILE))
        swe = record_built(
            folder, write_task(folder, "rh-swe-bench", compose=RH_SWE_BENCH, dockerfile=BUILD_DOCKERFILE)
        )
        write_task(folder, "harbor-index", compose=HARBOR_INDEX, dockerfile=None, docker_image=AGENT_IMAGE)
        record_built(folder, write_task(folder, "privileged", compose=PRIVILEGED, dockerfile=BUILD_DOCKERFILE))
        good = record_built(folder, write_task(folder, "good", compose=PULL_POLICIES, dockerfile=BUILD_DOCKERFILE))

        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))

        assert [task.task_id for task in prepared.tasks] == ["good", "harbor-index", "rh-swe-bench"]
        # The sidecar images (org/api:1 for good, the task server for harbor-index) are pre-pulled with the tasks'.
        assert set(prepared.images) == {"org/api:1", good, swe, AGENT_IMAGE, TASK_SERVER_IMAGE}
        skipped = json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text())
        assert skipped == prepared.skipped
        assert set(skipped) == {"featurebench", "privileged"}
        assert skipped["featurebench"].startswith(
            "featurebench: environment/docker-compose.yaml: deploy.resources.reservations.devices"
        )
        assert "count: all is not supported" in skipped["featurebench"]
        assert skipped["privileged"] == (
            "privileged: environment/docker-compose.yaml: Service 'main': unsupported Compose fields ['privileged']"
        )
        rows = prepared.rows_path.read_text().splitlines()
        assert len(rows) == 3 and '"good"' in rows[0]
        for task_id in skipped:
            assert f"Skipping task {task_id}: {task_id}: environment/docker-compose.yaml: " in caplog.text
        # The images were still recorded, so a fixed task file needs no new registry round-trip.
        recorded = json.loads((folder / "compose-images.json").read_text())
        assert TASK_SERVER_IMAGE in recorded and AGENT_IMAGE in recorded

    def test_a_dataset_with_no_task_left_fails_after_writing_the_reasons(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        write_task(folder, "only", compose=PRIVILEGED)
        with pytest.raises(ValueError, match="No runnable task left"):
            prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        skipped = json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text())
        assert skipped == {
            "only": "only: environment/docker-compose.yaml: Service 'main': unsupported Compose fields ['privileged']"
        }

    def test_a_refused_field_does_not_hide_a_task_the_selection_left_out(self, tmp_path):
        """``--exclude-tasks`` still wins: a left-out task is neither checked nor reported as broken."""
        folder = tmp_path / "ds"
        folder.mkdir()
        write_task(folder, "bad", compose=FEATUREBENCH)
        write_task(folder, "good", compose="services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n")
        prepared = prepare_target(
            str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False), exclude=["bad"]
        )
        assert [task.task_id for task in prepared.tasks] == ["good"] and prepared.skipped == {}

    def test_a_malformed_volume_declaration_is_a_skipped_task_not_an_aborted_prepare(self, tmp_path):
        """Review (round 14): ``volumes: {data: 123}`` escaped the planner as a ``TypeError`` (``set(123)``) and
        aborted ``record_compose_images`` for the whole dataset; it is the planner's own error now, so the task is
        reported in ``skipped.json`` with the volume named and its valid sibling is prepared. The same for a
        long-form entry whose ``bind`` or ``volume`` options are not a mapping (M22 review: ``set(123)`` again)."""
        folder = tmp_path / "ds"
        folder.mkdir()
        malformed = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    volumes:\n      - data:/data\n"
            "volumes:\n  data: 123\n"
        )
        nested = (
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    volumes:\n"
            "      - type: bind\n        source: ./seed.json\n        target: /seed.json\n        bind: 123\n"
        )
        write_task(folder, "malformed", compose=malformed)
        write_task(folder, "nested", compose=nested)
        write_task(folder, "good", compose="services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n")
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [task.task_id for task in prepared.tasks] == ["good"]
        assert prepared.skipped == {
            "malformed": (
                "malformed: environment/docker-compose.yaml: volume 'data': declaration must be a mapping, got 123"
            ),
            "nested": (
                "nested: environment/docker-compose.yaml: service 'main' volumes: bind options must be a mapping, "
                "got 123"
            ),
        }
        assert json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text()) == prepared.skipped
