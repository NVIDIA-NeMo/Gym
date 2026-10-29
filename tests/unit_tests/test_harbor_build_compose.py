# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the docker compose build backend.

Harbor's build template and build environment, the task Compose file as the build reads it, the
refusals on the build host, and the main image spellings the task file may use.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from nemo_gym.data import BUILT_IMAGES_FILE, load_compose_document, load_task
from nemo_gym.data.tasks.harbor.build_compose import (
    COMPOSE_BUILD_FILE,
    COMPOSE_IMAGES_FILE,
    compose_build_env,
    harbor_compose_document,
    resolve_task_env,
    validate_task_compose,
)
from nemo_gym.data.tasks.harbor.build_plan import BuildError, BuildPlanEntry, plan_builds
from nemo_gym.data.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.data.tasks.harbor.image_build import build_images
from nemo_gym.data.tasks.harbor.records import IMAGE_CONFIGS_FILE
from nemo_gym.data.tasks.harbor.task import built_image, built_service_images
from tests.unit_tests.harbor_build_helpers import (
    DOCKER_INSPECT,
    OVERLAY_DOCKERFILE,
    TOML,
    main_record,
    write_base_record,
    write_task,
)


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
    if len(files) > 1:
        record["images"] = open(files[-1]).read()
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
elif argv[0] == "compose" and "build" in argv:
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
            "-f",
            build["files"][1],
            "build",
        ]
        assert [Path(f).name for f in build["files"]] == [COMPOSE_BUILD_FILE, COMPOSE_IMAGES_FILE]
        # The file docker compose read is Harbor's, resolved for this context, tagged with Harbor's name; Gym's
        # images override, last, says the same for main and nothing else.
        assert yaml.safe_load(build["compose"]) == harbor_compose_document(context, f"hb__{result.digest}")
        assert yaml.safe_load(build["images"]) == {"services": {"main": {"image": f"hb__{result.digest}"}}}
        assert build["env"]["MAIN_IMAGE_NAME"] == f"hb__{result.digest}" == result.tag
        assert build["env"]["CONTEXT_DIR"] == str(context)
        assert build["env"]["DOCKER_DEFAULT_PLATFORM"] == "linux/amd64"
        assert build["env"]["HOST_VERIFIER_LOGS_PATH"] == "/logs/verifier"
        assert build["env"]["ENV_ARTIFACTS_PATH"] == "/logs/artifacts"
        # The recorded output is unchanged: the loader and the pull-mode path read the same files.
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {result.digest: main_record(result.tag, ["a"], "docker", False)}
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
        # Harbor's order: its own build file first, the task's docker-compose.yaml after it, then Gym's images
        # override, project directory the environment folder so relative paths resolve there; only main has a
        # build section to build, so the whole project is built, as Harbor builds it.
        assert build["files"] == [build["files"][0], str(context / "docker-compose.yaml"), build["files"][2]]
        assert Path(build["files"][2]).name == COMPOSE_IMAGES_FILE
        assert build["argv"][-1] == "build" and build["argv"][-3] == "-f"
        assert yaml.safe_load(build["images"]) == {"services": {"main": {"image": result.tag}}}
        assert build["argv"][build["argv"].index("--project-directory") + 1] == str(context)
        assert build["env"]["MAIN_IMAGE_NAME"] == result.tag and build["env"]["CONTEXT_DIR"] == str(context)
        assert lines == [f"Built {result.image} (local, docker) for task(s) a"]

    @pytest.mark.parametrize(
        "sidecar",
        ["  db:\n    build: ./db\n", "  db:\n    image: ${MAIN_IMAGE_NAME}_db\n    build: ./db\n"],
        ids=["no-image", "own-image"],
    )
    def test_a_sidecar_with_a_build_section_builds_in_the_same_compose_build(self, tmp_path, fake_docker, sidecar):
        # G2: Harbor's `docker compose build` (no service argument) builds every service with a build section, so
        # Gym does too, in one compose build of the project, and tags the sidecar with its own name through the
        # images override passed last, whatever the task file's image: for it says.
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n" + sidecar
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        context = folder / "a" / "environment"
        lines: list[str] = []
        main, db = build_images(folder, builder=ImageBuilder(tool="docker"), log=lines.append)
        assert (main.service, db.service) == ("main", "db") and main.digest == db.digest
        assert main.tag == f"hb__{main.digest}" and db.tag == f"hb__{main.digest}-db"
        calls = fake_docker()
        assert [c["argv"][:2] for c in calls] == [
            ["compose", "version"],
            ["compose", "--project-name"],
            ["image", "inspect"],
            ["image", "inspect"],
        ]
        build = calls[1]
        assert build["files"] == [build["files"][0], str(context / "docker-compose.yaml"), build["files"][2]]
        assert [Path(f).name for f in (build["files"][0], build["files"][2])] == [
            COMPOSE_BUILD_FILE,
            COMPOSE_IMAGES_FILE,
        ]
        # A full build names no service, as Harbor's does; the override names main and db with Gym's tags.
        assert build["argv"][-1] == "build"
        assert yaml.safe_load(build["images"]) == {"services": {"main": {"image": main.tag}, "db": {"image": db.tag}}}
        assert build["env"]["MAIN_IMAGE_NAME"] == main.tag
        assert [c["argv"][2] for c in calls[2:]] == [main.tag, db.tag]
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        assert built == {
            main.digest: {
                "image": main.tag,
                "tasks": ["a"],
                "tool": "docker",
                "pushed": False,
                "services": {
                    "main": {"image": main.tag, "tool": "docker", "pushed": False},
                    "db": {"image": db.tag, "tool": "docker", "pushed": False},
                },
            }
        }
        configs = json.loads((folder / IMAGE_CONFIGS_FILE).read_text())
        assert configs[db.tag]["config"]["WorkingDir"] == "/srv" and configs[db.tag]["image"] == db.tag
        assert lines == [
            f"Built {main.tag} (local, docker) for task(s) a",
            f"Built {db.tag} (local, docker) for task(s) a",
        ]
        # The loader substitutes both: the task runs from main's image and the sidecar from its recorded one.
        task = load_task(folder / "a")
        assert task.image == main.tag and built_service_images(folder / "a") == {"main": main.tag, "db": db.tag}

    def test_a_recorded_sidecar_leaves_a_partial_build_naming_the_missing_service(self, tmp_path, fake_docker):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ./db\n"
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        builder = ImageBuilder(tool="docker")
        main, db = build_images(folder, builder=builder, log=lambda _: None)
        # A prepare interrupted after db: drop main's record and keep db's.
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())
        del built[main.digest]["services"]["main"], built[main.digest]["image"]
        (folder / BUILT_IMAGES_FILE).write_text(json.dumps(built))
        assert built_image(folder / "a") is None and built_service_images(folder / "a") == {"db": db.tag}
        lines: list[str] = []
        (again,) = build_images(folder, builder=builder, log=lines.append)
        assert again.service == "main" and again.tag == main.tag
        build = fake_docker()[-2]
        # Only the missing service is built; the recorded one is reused and said so.
        assert build["argv"][-2:] == ["build", "main"]
        assert yaml.safe_load(build["images"]) == {"services": {"main": {"image": main.tag}}}
        assert lines == [
            f"Built image recorded for a (db): {db.tag}",
            f"Built {main.tag} (local, docker) for task(s) a",
        ]
        assert built_service_images(folder / "a") == {"main": main.tag, "db": db.tag}
        # Everything recorded: nothing is built, each service is reported.
        lines.clear()
        assert build_images(folder, builder=builder, log=lines.append) == []
        assert lines == [f"Built image recorded for a: {main.tag}", f"Built image recorded for a (db): {db.tag}"]

    def test_a_pull_mode_main_with_a_built_sidecar_builds_the_sidecar_alone(self, tmp_path, fake_docker):
        # itsm-bench's shape: main is a base image plus RUN lines (overlay mode), the sidecar has a build section.
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  runtime:\n    build: ./runtime\n"
        write_task(
            folder / "a",
            dockerfile=OVERLAY_DOCKERFILE,
            extra={"docker-compose.yaml": compose, "runtime/Dockerfile": "FROM python:3.12\nCOPY . /app\n"},
        )
        write_base_record(folder)
        (runtime,) = build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert runtime.service == "runtime" and runtime.tag == f"hb__{runtime.digest}-runtime"
        build = fake_docker()[1]
        # main is not built from Harbor's template: the build names the sidecar; MAIN_IMAGE_NAME is still exported.
        assert build["argv"][-2:] == ["build", "runtime"]
        assert yaml.safe_load(build["images"]) == {"services": {"runtime": {"image": runtime.tag}}}
        assert build["env"]["MAIN_IMAGE_NAME"] == f"hb__{runtime.digest}"
        built = json.loads((folder / BUILT_IMAGES_FILE).read_text())[runtime.digest]
        assert "image" not in built and built["services"] == {
            "runtime": {"image": runtime.tag, "tool": "docker", "pushed": False}
        }
        # No main: the top-level flags are the first recorded service's.
        assert built["tool"] == "docker" and built["pushed"] is False
        task = load_task(folder / "a")
        assert task.built_image is None and task.image == "ubuntu:24.04" and task.overlay

    def test_mains_image_spelling_is_not_checked_when_main_is_not_built(self, tmp_path, fake_docker):
        # A prebuilt main (docker_image declared) whose task file spells `${MAIN_IMAGE_NAME:-mytask-main}` (the
        # hil-bench form) with a built sidecar: Harbor starts main from the prebuilt image and `up` builds db. Gym
        # builds db alone and tags no main, so the spelling rule of test_main_image_other_than_harbors_is_refused
        # does not apply; the loader keeps the declared image for main and substitutes the built sidecar.
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME:-mytask-main}\n  db:\n    build: ./db\n"
        write_task(
            folder / "a",
            toml=TOML + 'docker_image = "org/prebuilt:1"\n',
            extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"},
        )
        (db,) = build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert db.service == "db" and db.tag == f"hb__{db.digest}-db"
        build = fake_docker()[1]
        assert build["argv"][-2:] == ["build", "db"]
        assert yaml.safe_load(build["images"]) == {"services": {"db": {"image": db.tag}}}
        task = load_task(folder / "a")
        assert task.image == "org/prebuilt:1" and task.built_image is None
        assert task.built_images == {"db": db.tag}
        document = load_compose_document(task)
        assert document["services"] == {"main": {"image": "org/prebuilt:1"}, "db": {"image": db.tag}}

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
        # Compose hands these to BuildKit as --allow: network.host lets RUN --network=host use the host's network.
        ("    build:\n      entitlements: [network.host]\n", r"build\.entitlements is refused"),
        ("    build:\n      privileged: true\n", r"build\.privileged is refused"),
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

    @pytest.mark.parametrize(("snippet", "message"), REFUSED, ids=[m[:28] for _, m in REFUSED])
    def test_host_reaching_build_keys_are_refused_on_a_sidecar_too(
        self, tmp_path, fake_docker, monkeypatch, snippet, message
    ):
        # The per-section checks run for every service with a build section; a sidecar reaches the host like main.
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n" + snippet
        write_task(folder / "a", extra={"docker-compose.yaml": compose, "db/Dockerfile": "FROM postgres:16\n"})
        context = folder / "a" / "environment"
        (context / "link").symlink_to(folder.parent / "elsewhere")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NEMO_GYM_NOT_SET", raising=False)
        with pytest.raises(BuildError, match=message.replace("main", "db")) as info:
            build_images(folder, builder=ImageBuilder(tool="docker"), log=lambda _: None)
        assert str(info.value).startswith(str(context / "docker-compose.yaml"))
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
        ("    build:\n      entitlements: ['${X:-network.host}']\n", {}, r"build\.entitlements is refused"),
        ("    build:\n      privileged: ${X:-true}\n", {}, r"build\.privileged is refused"),
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
        with caplog.at_level("WARNING", logger="nemo_gym.data.tasks.harbor.build_compose"):
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
        with caplog.at_level("WARNING", logger="nemo_gym.data.tasks.harbor.build_compose"):
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

    def test_task_env_is_left_out_without_a_compose_file_as_harbor_leaves_it(self, tmp_path, monkeypatch):
        folder = tmp_path / "ds"
        write_task(folder / "a", toml=TOML + '\n[environment.env]\nX = "${NOT_SET_ANYWHERE}"\n')
        (entry,) = plan_builds(folder)
        assert entry.env == {} and entry.cpus == 1 and entry.memory_mb == 2048
        # Without a Compose file nothing interpolates: the build's environment must not raise over the unresolvable
        # template and carries no X (the guard keeps a host that exports X from satisfying the check).
        monkeypatch.delenv("X", raising=False)
        env = compose_build_env(entry, "hb__x", None)
        assert "X" not in env

    def test_tasks_sharing_a_context_with_different_env_share_the_image_and_are_logged(self, tmp_path, caplog):
        folder = tmp_path / "ds"
        compose = "services:\n  main:\n    build:\n      args:\n        V: ${V}\n"
        write_task(folder / "a", toml=TOML + '\n[environment.env]\nV = "1"\n', extra={"docker-compose.yaml": compose})
        write_task(folder / "b", toml=TOML + '\n[environment.env]\nV = "2"\n', extra={"docker-compose.yaml": compose})
        write_task(folder / "c", toml=TOML + '\n[environment.env]\nV = "1"\n', extra={"docker-compose.yaml": compose})
        with caplog.at_level("WARNING", logger="nemo_gym.data.tasks.harbor.build_plan"):
            (entry,) = plan_builds(folder)
        assert entry.tasks == ("a", "b", "c") and entry.env == {"V": "1"}
        assert "Tasks a and b share build context" in caplog.text and "(V)" in caplog.text
        assert "2" not in caplog.text.split("(V)")[0].split("settings")[-1]  # values are not logged
        assert "Tasks a and c" not in caplog.text


class TestMainImageSpellings:
    """D4: main's image is accepted when it resolves to MAIN_IMAGE_NAME, whichever interpolation spells it."""

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
