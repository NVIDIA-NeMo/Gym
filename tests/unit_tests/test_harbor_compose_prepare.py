# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""D5: the sandbox Compose adapter's field check runs at prepare, per task, on the document the sandbox receives,
and ``pull_policy`` is dropped from every service of that document."""

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
    load_compose_document,
    without_pull_policy,
    without_restated_main_build,
)
from nemo_gym.tasks.harbor.prepare_builds import unsupported_compose_fields_error
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE
from nemo_gym.tasks.harbor.task import discover_tasks, load_task


PULL_DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"
OVERLAY_DOCKERFILE = "FROM ubuntu:24.04\nRUN echo hi > /hi\nWORKDIR /app\n"
BUILD_DOCKERFILE = "FROM ubuntu:24.04\nCOPY tools/ /opt/tools/\nWORKDIR /app\n"
AGENT_IMAGE = "harborframework/harbor-index:cybergym-agent@sha256:" + "a" * 64
TASK_SERVER_IMAGE = "harborframework/harbor-index:cybergym-task-server@sha256:" + "c" * 64

# The three hub shapes D_TESTS reproduced: featurebench 265e447753be, rh-swe-bench 0884ee3e43b7, harbor-index 9d8c77f891ab.
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
    (entry,) = plan_builds(task)
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


REFUSED_NOTE = " (the sandbox Compose adapter does not start them)"
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


class TestUnsupportedComposeFieldsError:
    def test_a_task_without_a_compose_file_is_not_a_compose_task(self, tmp_path):
        task = write_task(tmp_path, "plain", compose="")
        (task / "environment" / "docker-compose.yaml").unlink()
        image_configs.record_base_images(tmp_path)
        assert unsupported_compose_fields_error(load_task(task)) is None

    def test_a_sidecar_build_is_still_a_refused_field(self, tmp_path):
        task = write_task(tmp_path, "ctr", compose=SIDECAR_BUILD, dockerfile=BUILD_DOCKERFILE)
        record_built(tmp_path, task)
        image_configs.record_base_images(tmp_path)
        (loaded,) = discover_tasks(tmp_path)
        error = unsupported_compose_fields_error(loaded)
        assert str(error) == "docker-compose.yaml: Service 'api': unsupported Compose fields ['build']" + REFUSED_NOTE


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
        assert unsupported_compose_fields_error(task) is None
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [t.task_id for t in prepared.tasks] == ["restated"] and prepared.skipped == {}
        assert len(prepared.rows_path.read_text().splitlines()) == 1

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
        assert prepared.skipped == {"unbuilt": f"docker-compose.yaml: {keys}{UNBUILT_NOTE}"}

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
            assert "build" in document["services"]["main"]
        assert unsupported_compose_fields_error(tasks["built"]) is None
        # The same file on a task that was not built keeps a main build with a target Gym cannot apply.
        error = unsupported_compose_fields_error(tasks["pulled"])
        assert str(error) == "docker-compose.yaml: main.build.target" + UNBUILT_NOTE


class TestPrepareChecksTheFieldsBeforeBuilding:
    """The field check also runs on the raw file of every task about to be built, so a refused task costs no build
    or push and a build that would fail for it does not stop the preparation."""

    def test_a_refused_task_is_neither_built_nor_pushed_and_its_failing_build_stops_nothing(
        self, tmp_path, monkeypatch
    ):
        from nemo_gym.tasks.harbor.build_tools import ImageBuilder
        from tests.unit_tests.harbor_build_helpers import FakeTool

        folder = tmp_path / "ds"
        folder.mkdir()
        bad = write_task(folder, "featurebench", compose=FEATUREBENCH, dockerfile=BUILD_DOCKERFILE)
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
            "featurebench": "docker-compose.yaml: Service 'main': unsupported Compose fields ['deploy']" + REFUSED_NOTE
        }
        assert prepared.skipped == expected
        assert json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text()) == expected
        # The refused context never reached the tool; the other build ran and was pushed.
        assert not any(str(bad / "environment") in " ".join(call) for call in tool.calls)
        assert any("build" in call and str(fine / "environment") in " ".join(call) for call in tool.calls)
        assert any("push" in call for call in tool.calls)
        recorded = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert [entry["tasks"] for entry in recorded.values()] == [["fine"]]

    def test_with_builds_disabled_the_field_is_still_the_reason(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        write_task(folder, "featurebench", compose=FEATUREBENCH, dockerfile=BUILD_DOCKERFILE)
        write_task(folder, "good", compose=GOOD)
        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert [task.task_id for task in prepared.tasks] == ["good"]
        assert prepared.skipped == {
            "featurebench": "docker-compose.yaml: Service 'main': unsupported Compose fields ['deploy']" + REFUSED_NOTE
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
    def test_the_three_hub_shapes_are_skipped_at_prepare_naming_the_field(self, tmp_path, caplog):
        """featurebench and rh-swe-bench carry a stale built-image record: the record does not put a refused task
        in the rows."""
        folder = tmp_path / "ds"
        folder.mkdir()
        record_built(folder, write_task(folder, "featurebench", compose=FEATUREBENCH, dockerfile=BUILD_DOCKERFILE))
        record_built(folder, write_task(folder, "rh-swe-bench", compose=RH_SWE_BENCH, dockerfile=BUILD_DOCKERFILE))
        write_task(folder, "harbor-index", compose=HARBOR_INDEX, dockerfile=None, docker_image=AGENT_IMAGE)
        good = record_built(folder, write_task(folder, "good", compose=PULL_POLICIES, dockerfile=BUILD_DOCKERFILE))

        prepared = prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))

        assert [task.task_id for task in prepared.tasks] == ["good"]
        assert prepared.images == ("org/api:1", good)  # the sidecar's image is pre-pulled with the task's
        skipped = json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text())
        assert skipped == prepared.skipped
        assert skipped == {
            "featurebench": "docker-compose.yaml: Service 'main': unsupported Compose fields ['deploy']"
            + REFUSED_NOTE,
            "rh-swe-bench": "docker-compose.yaml: Service 'main': unsupported Compose fields ['extra_hosts']"
            + REFUSED_NOTE,
            "harbor-index": "docker-compose.yaml: Service 'task-server': unsupported Compose fields "
            "['deploy', 'platform']" + REFUSED_NOTE,
        }
        rows = prepared.rows_path.read_text().splitlines()
        assert len(rows) == 1 and '"good"' in rows[0]
        for task_id in skipped:
            assert f"Skipping task {task_id}: docker-compose.yaml: Service" in caplog.text
        # The images were still recorded, so a fixed task file needs no new registry round-trip.
        recorded = json.loads((folder / "compose-images.json").read_text())
        assert TASK_SERVER_IMAGE in recorded and AGENT_IMAGE in recorded

    def test_a_dataset_with_no_task_left_fails_after_writing_the_reasons(self, tmp_path):
        folder = tmp_path / "ds"
        folder.mkdir()
        write_task(folder, "only", compose=RH_SWE_BENCH)
        with pytest.raises(ValueError, match="No runnable task left"):
            prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        skipped = json.loads((tmp_path / "out" / "ds" / "skipped.json").read_text())
        assert skipped == {
            "only": "docker-compose.yaml: Service 'main': unsupported Compose fields ['extra_hosts']" + REFUSED_NOTE
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
