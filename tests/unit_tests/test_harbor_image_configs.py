# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nemo_gym.sandbox.compose_config import _compose_literals
from nemo_gym.tasks.harbor import image_configs
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.image_configs import (
    ImageConfigError,
    RegistryClient,
    compose_image_references,
    compose_variables,
    dockerfile_base_images,
    load_compose_document,
    parse_image_ref,
    record_base_images,
    record_compose_images,
)
from nemo_gym.tasks.harbor.task import HarborTaskError, discover_tasks, write_json_record


MAIN_IMAGE = "harborframework/terminal-bench:ctr-environment-abc@sha256:" + "a" * 64
SIDECAR_IMAGE = "harborframework/terminal-bench:ctr-sidecar-api-def@sha256:" + "b" * 64


def write_task(
    root: Path,
    name: str,
    *,
    image: str | None = MAIN_IMAGE,
    compose: str | None = None,
    dockerfile: str | None = None,
) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    declared = f'docker_image = "{image}"\n' if image else ""
    (task / "task.toml").write_text(f'schema_version = "1.4"\n\n[environment]\n{declared}')
    (task / "instruction.md").write_text("Do the thing.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    if compose is not None:
        (task / "environment" / "docker-compose.yaml").write_text(compose)
    if dockerfile is not None:
        (task / "environment" / "Dockerfile").write_text(dockerfile)
    return task


COMPOSE = f"""services:
  main:
    network_mode: "service:api"
  api:
    image: {SIDECAR_IMAGE}
  cache:
    image: redis:7-alpine
"""


class TestParseImageRef:
    @pytest.mark.parametrize(
        ("reference", "registry", "repository", "tag", "digest"),
        [
            ("redis:7-alpine", "registry-1.docker.io", "library/redis", "7-alpine", None),
            ("redis", "registry-1.docker.io", "library/redis", None, None),
            ("apache/kafka-native:4.3.1", "registry-1.docker.io", "apache/kafka-native", "4.3.1", None),
            ("docker.io/apache/kafka-native:4.3.1", "registry-1.docker.io", "apache/kafka-native", "4.3.1", None),
            (
                MAIN_IMAGE,
                "registry-1.docker.io",
                "harborframework/terminal-bench",
                "ctr-environment-abc",
                "sha256:" + "a" * 64,
            ),
            ("ghcr.io/org/app:1.0", "ghcr.io", "org/app", "1.0", None),
            ("localhost:5000/app", "localhost:5000", "app", None, None),
            (
                "registry.example.com:5005/group/app@sha256:" + "c" * 64,
                "registry.example.com:5005",
                "group/app",
                None,
                "sha256:" + "c" * 64,
            ),
        ],
    )
    def test_splits_like_docker(self, reference, registry, repository, tag, digest):
        ref = parse_image_ref(reference)
        assert (ref.registry, ref.repository, ref.tag, ref.digest) == (registry, repository, tag, digest)

    def test_manifest_ref_prefers_the_digest(self):
        assert parse_image_ref(MAIN_IMAGE).manifest_ref == "sha256:" + "a" * 64
        assert parse_image_ref("redis:7-alpine").manifest_ref == "7-alpine"
        assert parse_image_ref("redis").manifest_ref == "latest"

    def test_pinned_repository_keeps_the_registry_name_and_adds_the_host_off_docker_hub(self):
        assert parse_image_ref("redis:7-alpine").pinned_repository == "library/redis"
        assert parse_image_ref("ghcr.io/org/app:1.0").pinned_repository == "ghcr.io/org/app"

    def test_rejects_an_empty_repository(self):
        with pytest.raises(ImageConfigError, match="Not an image reference"):
            parse_image_ref("ghcr.io/")


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        super().__init__(body)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeRegistry:
    """Serves the OCI distribution API for a few images and records every request."""

    INDEX = "application/vnd.oci.image.index.v1+json"
    MANIFEST = "application/vnd.oci.image.manifest.v1+json"

    def __init__(self, *, require_token: bool = True):
        self.require_token = require_token
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.blobs = {
            "sha256:cfg-amd64": {
                "os": "linux",
                "architecture": "amd64",
                "config": {
                    "Cmd": ["python", "server.py"],
                    "WorkingDir": "/app",
                    "ExposedPorts": {"5000/tcp": {}},
                    "Env": ["A=1"],
                },
            },
            "sha256:cfg-arm64": {"os": "linux", "architecture": "arm64", "config": {"Cmd": ["python", "server.py"]}},
        }
        self.manifests = {
            "sha256:m-amd64": {"mediaType": self.MANIFEST, "config": {"digest": "sha256:cfg-amd64"}},
            "sha256:m-arm64": {"mediaType": self.MANIFEST, "config": {"digest": "sha256:cfg-arm64"}},
            "sha256:m-att": {"mediaType": self.MANIFEST, "config": {"digest": "sha256:cfg-amd64"}},
        }
        self.tags = {
            "7-alpine": {
                "mediaType": self.INDEX,
                "manifests": [
                    {"digest": "sha256:m-arm64", "platform": {"os": "linux", "architecture": "arm64"}},
                    {
                        "digest": "sha256:m-att",
                        "platform": {"os": "unknown", "architecture": "unknown"},
                        "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
                    },
                    {"digest": "sha256:m-amd64", "platform": {"os": "linux", "architecture": "amd64"}},
                ],
            },
            "arm-only": {
                "mediaType": self.INDEX,
                "manifests": [{"digest": "sha256:m-arm64", "platform": {"os": "linux", "architecture": "arm64"}}],
            },
        }

    def __call__(self, request, timeout=None):
        url, headers = request.full_url, dict(request.header_items())
        self.requests.append((url, headers))
        if url.startswith("https://auth.example/token"):
            return FakeResponse(json.dumps({"token": "tok"}).encode())
        _, _, path = url.partition("/v2/")
        repository, kind, ref = path.rsplit("/", 2)
        if self.require_token and headers.get("Authorization") != "Bearer tok":
            challenge = (
                f'Bearer realm="https://auth.example/token",service="registry",scope="repository:{repository}:pull"'
            )
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {"WWW-Authenticate": challenge}, io.BytesIO(b""))
        if kind == "manifests":
            manifest = self.tags.get(ref) or self.manifests.get(ref)
            if manifest is None:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
            digest = ref if ref.startswith("sha256:") else "sha256:index-of-" + ref
            return FakeResponse(json.dumps(manifest).encode(), {"Docker-Content-Digest": digest})
        if kind == "blobs":
            return FakeResponse(json.dumps(self.blobs[ref]).encode())
        raise AssertionError(url)


@pytest.fixture
def registry(monkeypatch):
    fake = FakeRegistry()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


class TestRegistryClient:
    def test_records_the_linux_amd64_manifest_behind_a_floating_tag(self, registry):
        entry = RegistryClient().image_config("redis:7-alpine")
        assert entry == {
            "architecture": "amd64",
            "config": {
                "Cmd": ["python", "server.py"],
                "Env": ["A=1"],
                "ExposedPorts": {"5000/tcp": {}},
                "WorkingDir": "/app",
            },
            "config_digest": "sha256:cfg-amd64",
            "image": "library/redis@sha256:m-amd64",
            "os": "linux",
        }

    def test_a_digest_reference_is_pinned_to_itself(self, registry):
        entry = RegistryClient().image_config("ghcr.io/org/app@sha256:m-amd64")
        assert entry["image"] == "ghcr.io/org/app@sha256:m-amd64"
        assert registry.requests[-1][0] == "https://ghcr.io/v2/org/app/blobs/sha256:cfg-amd64"

    def test_the_token_is_fetched_once_per_repository(self, registry):
        client = RegistryClient()
        client.image_config("redis:7-alpine")
        client.image_config("redis@sha256:m-amd64")
        token_requests = [url for url, _ in registry.requests if url.startswith("https://auth.example/token")]
        assert len(token_requests) == 1
        assert "scope=repository%3Alibrary%2Fredis%3Apull" in token_requests[0]

    def test_credentials_are_sent_to_the_token_endpoint_as_basic_auth(self, registry):
        RegistryClient({"registry-1.docker.io": ("user", "secret")}).image_config("redis:7-alpine")
        token_request = next(headers for url, headers in registry.requests if url.startswith("https://auth.example"))
        assert token_request["Authorization"] == "Basic dXNlcjpzZWNyZXQ="

    def test_an_index_without_linux_amd64_is_rejected(self, registry):
        with pytest.raises(ImageConfigError, match="no linux/amd64 manifest"):
            RegistryClient().image_config("redis:arm-only")

    def test_a_single_platform_manifest_of_the_wrong_architecture_is_rejected(self, registry):
        with pytest.raises(ImageConfigError, match="linux/arm64, not linux/amd64"):
            RegistryClient().image_config("redis@sha256:m-arm64")

    def test_http_errors_carry_the_repository_and_status(self, registry):
        with pytest.raises(ImageConfigError, match="library/redis: registry returned HTTP 404"):
            RegistryClient().image_config("redis:missing")

    def test_connection_failures_are_retried_then_reported(self, monkeypatch):
        attempts = []

        def flaky(request, timeout=None):
            attempts.append(request.full_url)
            raise urllib.error.URLError("handshake timed out")

        monkeypatch.setattr(urllib.request, "urlopen", flaky)
        monkeypatch.setattr(image_configs.time, "sleep", lambda seconds: None)
        with pytest.raises(ImageConfigError, match="unreachable after 3 attempts"):
            RegistryClient(retries=2).image_config("redis:7-alpine")
        assert len(attempts) == 3


class TestComposeImageReferences:
    def test_lists_the_task_image_first_then_each_sidecar_once(self, tmp_path):
        write_task(tmp_path, "ctr", compose=COMPOSE)
        (task,) = discover_tasks(tmp_path)
        assert compose_image_references(task) == [MAIN_IMAGE, SIDECAR_IMAGE, "redis:7-alpine"]

    def test_a_main_service_image_wins_over_the_task_image(self, tmp_path):
        write_task(tmp_path, "ctr", compose="services:\n  main:\n    image: other:1\n  api:\n    image: redis:7\n")
        (task,) = discover_tasks(tmp_path)
        assert compose_image_references(task) == ["other:1", "redis:7"]

    def test_a_task_without_compose_has_none(self, tmp_path):
        write_task(tmp_path, "plain")
        (task,) = discover_tasks(tmp_path)
        assert compose_image_references(task) == []

    def test_a_sidecar_that_builds_is_rejected(self, tmp_path):
        write_task(tmp_path, "ctr", compose="services:\n  api:\n    build: ./api\n")
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError, match="service 'api' names no image"):
            compose_image_references(task)


class FakeClient:
    def __init__(self):
        self.resolved: list[str] = []

    def image_config(self, reference: str) -> dict:
        self.resolved.append(reference)
        return {
            "architecture": "amd64",
            "config": {"Cmd": ["run"]},
            "config_digest": "sha256:c",
            "image": reference,
            "os": "linux",
        }


class TestComposeVariablesAndEnvFile:
    """M19 and G3: Harbor's Compose interpolation variables and env_file, resolved at prepare in the order Harbor's
    ``docker compose`` resolves them: Gym's infrastructure set, the task's ``[environment].env``, the process
    environment, ``environment/.env``, then the expression's default or an empty string."""

    HIL_BENCH_STYLE = """services:
  main:
    image: ${MAIN_IMAGE_NAME}
    volumes:
      - ${CONTEXT_DIR}/seed.json:/app/seed.json:ro
      - ${HOST_VERIFIER_LOGS_PATH}:/logs/verifier
    environment:
      - CPU_BUDGET=${CPUS:-1}
      - MODE=${MODE}
  api:
    image: ${SIDECAR:-redis:7-alpine}
    env_file: api.env
    environment:
      PRICE: "$$5"
"""

    def test_variables_come_from_the_task_and_harbor_infra_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_DIR", "/from-the-shell")
        task_dir = write_task(tmp_path, "ctr", compose=self.HIL_BENCH_STYLE)
        (task_dir / "task.toml").write_text(
            'schema_version = "1.4"\n\n[environment]\n'
            f'docker_image = "{MAIN_IMAGE}"\ncpus = 4\nmemory_mb = 2048\n'
            'env = { MODE = "strict", MAIN_IMAGE_NAME = "task-tries-to-override", TEST_DIR = "/task-tries-too" }\n'
        )
        (task,) = discover_tasks(tmp_path)
        variables = compose_variables(task)
        assert variables["MAIN_IMAGE_NAME"] == MAIN_IMAGE  # infra wins over [environment].env
        assert variables["PREBUILT_IMAGE_NAME"] == MAIN_IMAGE
        assert variables["CONTEXT_DIR"] == str(task.path / "environment")
        # Gym defines the two Harbor leaves to Compose: where the tests land, and a stable stand-in for the project
        # name compose-go injects (Gym's adapter names the project compose-<uuid>; Harbor's is the trial session id).
        assert variables["TEST_DIR"] == "/tests"  # over the task's value and the shell's
        assert variables["COMPOSE_PROJECT_NAME"] == f"hb__{context_digest(task.path / 'environment')}"
        # MEMORY as Harbor spells it (`f"{memory_mb}M"` in docker.py), the same string the build's environment carries.
        assert (variables["CPUS"], variables["MEMORY"], variables["MODE"]) == ("4", "2048M", "strict")
        assert variables["HOST_VERIFIER_LOGS_PATH"] == variables["ENV_VERIFIER_LOGS_PATH"] == "/logs/verifier"
        assert (
            variables["HOST_AGENT_LOGS_PATH"] == "/logs/agent"
            and variables["HOST_ARTIFACTS_PATH"] == "/logs/artifacts"
        )

    def test_task_env_values_resolve_from_the_host_as_harbor_resolves_them(self, tmp_path, monkeypatch):
        # hil-bench-swe-ask-human on the hub: [environment.env] holds `${ASK_HUMAN_BACKEND:-litellm_proxy}` and the
        # Compose file references the same name again. Harbor's resolve_env_vars (harbor/utils/env.py) reads a
        # whole-value template from the host, the default standing in when the host lacks it; the task's value then
        # wins over the host's in the file. Anything but a whole-value template is literal.
        monkeypatch.setenv("ASK_HUMAN_BACKEND", "from-the-shell")
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        task_dir = write_task(
            tmp_path,
            "ctr",
            compose="services:\n  ask-human:\n    image: redis:7\n    environment:\n"
            "      - ASK_HUMAN_BACKEND=${ASK_HUMAN_BACKEND:-litellm_proxy}\n      - API_KEY=${LITELLM_API_KEY:-}\n",
        )
        (task_dir / "task.toml").write_text(
            f'schema_version = "1.4"\n\n[environment]\ndocker_image = "{MAIN_IMAGE}"\n\n[environment.env]\n'
            'ASK_HUMAN_BACKEND = "${ASK_HUMAN_BACKEND:-litellm_proxy}"\nLITELLM_API_KEY = "${LITELLM_API_KEY:-}"\n'
            'NOTE = "see ${CONTEXT_DIR}"\n'
        )
        (task,) = discover_tasks(tmp_path)
        variables = compose_variables(task)
        assert (variables["ASK_HUMAN_BACKEND"], variables["LITELLM_API_KEY"]) == ("from-the-shell", "")
        # Literal to the document, as Compose takes a value it splices in: the dollar is kept doubled, the convention
        # the document's literal check reads, so a file reading ${NOTE} starts with `see ${CONTEXT_DIR}` verbatim.
        assert variables["NOTE"] == "see $${CONTEXT_DIR}"
        environment = load_compose_document(task)["services"]["ask-human"]["environment"]
        assert environment == {"ASK_HUMAN_BACKEND": "from-the-shell", "API_KEY": ""}
        monkeypatch.delenv("ASK_HUMAN_BACKEND")
        assert compose_variables(task)["ASK_HUMAN_BACKEND"] == "litellm_proxy"

    def test_a_task_env_template_the_host_does_not_set_is_refused_naming_the_task_and_variable(
        self, tmp_path, monkeypatch
    ):
        # tau3 on the hub: `OPENAI_API_KEY = "${OPENAI_API_KEY}"`. Harbor fails the trial when the host lacks it.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        task_dir = write_task(tmp_path, "ctr", compose="services:\n  api:\n    image: redis:7\n")
        (task_dir / "task.toml").write_text(
            f'schema_version = "1.4"\n\n[environment]\ndocker_image = "{MAIN_IMAGE}"\n\n[environment.env]\n'
            'OPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
        )
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(
            ImageConfigError,
            match=r"ctr: \[environment\].env OPENAI_API_KEY reads \$\{OPENAI_API_KEY\}, which is not set",
        ):
            load_compose_document(task)
        monkeypatch.setenv("OPENAI_API_KEY", "from-the-shell")
        assert compose_variables(task)["OPENAI_API_KEY"] == "from-the-shell"

    ORDER_COMPOSE = """services:
  main:
    environment:
      - INFRA=${TEST_DIR}
      - TASK=${TASK_WINS}
      - SHELL=${SHELL_WINS}
      - DOTENV=${FROM_DOTENV}
      - EXPANDED=${EXPANDED}
      - LITERAL=${LITERAL}
      - DEFAULTED=${NOWHERE:-fallback}
      - DEFAULTED_IF_UNSET=${NOWHERE-fallback}
      - EMPTY=${NOWHERE}
      - EMPTY_AGAIN=${NOWHERE}$NOWHERE
"""

    def test_resolution_order_is_infra_task_env_host_dotenv_default_then_empty(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(image_configs, "_WARNED_UNSET", set())
        for name in ("TEST_DIR", "TASK_WINS", "SHELL_WINS"):
            monkeypatch.setenv(name, "shell")
        for name in ("FROM_DOTENV", "NOWHERE"):
            monkeypatch.delenv(name, raising=False)
        task_dir = write_task(tmp_path, "ctr", compose=self.ORDER_COMPOSE)
        (task_dir / "task.toml").write_text(
            f'schema_version = "1.4"\n\n[environment]\ndocker_image = "{MAIN_IMAGE}"\n\n[environment.env]\n'
            'TEST_DIR = "task"\nTASK_WINS = "task"\n'
        )
        # Compose's dotenv: an unquoted or double-quoted value is interpolated against the command's environment
        # and the lines above it; a single-quoted value is literal.
        (task_dir / "environment" / ".env").write_text(
            "TEST_DIR=dotenv\nTASK_WINS=dotenv\nSHELL_WINS=dotenv\nFROM_DOTENV=dotenv\n"
            'EXPANDED="${SHELL_WINS}-${FROM_DOTENV}-${NOWHERE:-x}"\n'
            "LITERAL='${SHELL_WINS}'\n"
        )
        (task,) = discover_tasks(tmp_path)
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.image_configs"):
            environment = load_compose_document(task)["services"]["main"]["environment"]
        assert environment == {
            "INFRA": "/tests",
            "TASK": "task",
            "SHELL": "shell",
            "DOTENV": "dotenv",
            "EXPANDED": "shell-dotenv-x",
            "LITERAL": "$${SHELL_WINS}",
            "DEFAULTED": "fallback",
            "DEFAULTED_IF_UNSET": "fallback",
            "EMPTY": "",
            "EMPTY_AGAIN": "",
        }
        # One warning per task and variable, however often the file reads it.
        assert [r.getMessage() for r in caplog.records] == [
            "ctr: Compose variable ${NOWHERE} has no value from the task, the process environment or "
            "environment/.env; it is an empty string, as docker compose makes it"
        ]
        # A later load of the same task says it no more: the server loads the document once per episode.
        caplog.clear()
        load_compose_document(task)
        assert caplog.records == []

    @pytest.mark.parametrize("form", ["${API_KEY:?set API_KEY for this task}", "${API_KEY?set API_KEY for this task}"])
    def test_a_required_variable_without_a_value_is_refused_with_its_message(self, tmp_path, monkeypatch, form):
        monkeypatch.delenv("API_KEY", raising=False)
        write_task(
            tmp_path,
            "ctr",
            compose=f"services:\n  api:\n    image: redis:7\n    environment:\n      - API_KEY={form}\n",
        )
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(
            ImageConfigError, match=r"ctr: docker-compose.yaml: \$\{API_KEY\} is required: set API_KEY for this task"
        ):
            load_compose_document(task)
        with pytest.raises(ImageConfigError, match=r"\$\{API_KEY\} is required"):
            record_compose_images([task], tmp_path, client=FakeClient())
        monkeypatch.setenv("API_KEY", "from-the-shell")
        assert load_compose_document(task)["services"]["api"]["environment"] == {"API_KEY": "from-the-shell"}

    def test_a_host_value_with_dollars_is_literal_to_the_document_and_reaches_the_sandbox_intact(
        self, tmp_path, monkeypatch
    ):
        # A bcrypt hash, an argon2 string or a generated secret carries `$`. Compose splices a variable's value in
        # verbatim and never re-scans it, so every source is literal to the document: no false "unresolved" skip,
        # and no `$$` collapsed to `$` when the sandbox starts.
        monkeypatch.setenv("DB_PASS", "$2b$12$abcHASH")
        monkeypatch.setenv("DOUBLED", "ab$$cd")
        monkeypatch.setenv("API_KEY", "sk-$argon2id$v=19")
        task_dir = write_task(
            tmp_path,
            "ctr",
            compose=(
                "services:\n  api:\n    image: redis:7\n    environment:\n"
                "      - DB_PASS=${DB_PASS}\n      - DOUBLED=${DOUBLED}\n      - URL=pw:${DB_PASS}@db\n"
                "      - FROM_DOTENV=${FROM_DOTENV}\n      - FROM_TASK=${API_KEY}\n      - NOTE=${NOTE}\n"
            ),
        )
        (task_dir / "task.toml").write_text(
            f'schema_version = "1.4"\n\n[environment]\ndocker_image = "{MAIN_IMAGE}"\n\n[environment.env]\n'
            'API_KEY = "${API_KEY}"\nNOTE = "costs $5 or ${MORE}"\n'
        )
        (task_dir / "environment" / ".env").write_text('FROM_DOTENV="${DB_PASS}-x"\n')
        (task,) = discover_tasks(tmp_path)
        environment = load_compose_document(task)["services"]["api"]["environment"]
        # Doubled in the document, as parse_env_file keeps a literal dollar; exact when the document is started.
        assert environment["DB_PASS"] == "$$2b$$12$$abcHASH"
        assert _compose_literals(environment) == {
            "DB_PASS": "$2b$12$abcHASH",
            "DOUBLED": "ab$$cd",
            "URL": "pw:$2b$12$abcHASH@db",
            "FROM_DOTENV": "$2b$12$abcHASH-x",
            "FROM_TASK": "sk-$argon2id$v=19",
            "NOTE": "costs $5 or ${MORE}",
        }
        assert record_compose_images([task], tmp_path, client=FakeClient()) is not None

    def test_a_residual_is_reported_by_its_key_path_and_no_host_value_reaches_the_message(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PASS", "$2b$12$abcHASH")
        write_task(
            tmp_path,
            "ctr",
            compose="services:\n  api:\n    image: redis:7\n    environment:\n      - DB_PASS=${DB_PASS}\n",
        )
        (task,) = discover_tasks(tmp_path)
        # Every source is literal to the document, so the resolver leaves nothing unresolved itself; the check is
        # the last line of defence, exercised through an env_file reader that lets a raw value and a $NAME through.
        monkeypatch.setattr(
            image_configs,
            "inline_env_files",
            lambda document, base_dir, variables=None: {
                "services": {"api": {"image": "redis:7", "environment": {"DB_PASS": "$2b$12$abcHASH", "L": "$LEFT"}}}
            },
        )
        with pytest.raises(ImageConfigError) as info:
            load_compose_document(task)
        assert (
            str(info.value)
            == "ctr: docker-compose.yaml: unresolved interpolation left at services.api.environment.DB_PASS"
        )
        skipped: dict[str, str] = {}
        record_compose_images([task], tmp_path, client=FakeClient(), skipped=skipped)
        assert "abcHASH" not in skipped["ctr"] and skipped["ctr"].endswith("services.api.environment.DB_PASS")

    def test_an_unset_image_variable_leaves_the_service_without_an_image(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(image_configs, "_WARNED_UNSET", set())
        monkeypatch.delenv("SIDECAR_IMAGE", raising=False)
        write_task(tmp_path, "ctr", compose="services:\n  api:\n    image: ${SIDECAR_IMAGE}\n")
        (task,) = discover_tasks(tmp_path)
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.image_configs"):
            with pytest.raises(ImageConfigError, match="ctr: Compose service 'api' names no image"):
                compose_image_references(task)
        (warning,) = [r.getMessage() for r in caplog.records]
        assert warning.startswith("ctr: Compose variable ${SIDECAR_IMAGE} has no value")

    def test_a_malformed_dotenv_line_fails_naming_the_line_only(self, tmp_path):
        task_dir = write_task(tmp_path, "ctr", compose="services:\n  api:\n    image: redis:7\n")
        (task_dir / "environment" / ".env").write_text("GOOD=1\nsecret value without an equals sign\n")
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(
            ImageConfigError, match=r"ctr: environment/\.env: env_file line 2 is not KEY=value"
        ) as info:
            load_compose_document(task)
        assert "secret value" not in str(info.value)

    def test_document_is_resolved_and_env_file_inlined(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODE", "from-the-shell")  # the task's own MODE wins over the shell's
        task_dir = write_task(tmp_path, "ctr", compose=self.HIL_BENCH_STYLE)
        (task_dir / "task.toml").write_text(
            f'schema_version = "1.4"\n\n[environment]\ndocker_image = "{MAIN_IMAGE}"\nenv = {{ MODE = "strict" }}\n'
        )
        (task_dir / "environment" / "api.env").write_text("API_KEY_NAME=demo\nPRICE=overridden-by-environment\n")
        (task,) = discover_tasks(tmp_path)
        document = load_compose_document(task)
        main, api = document["services"]["main"], document["services"]["api"]
        assert main["image"] == MAIN_IMAGE
        assert main["volumes"] == [
            f"{task.path / 'environment'}/seed.json:/app/seed.json:ro",
            "/logs/verifier:/logs/verifier",
        ]
        assert main["environment"] == {"CPU_BUDGET": "1", "MODE": "strict"}  # no cpus declared: the default applies
        assert api["image"] == "redis:7-alpine"
        assert "env_file" not in api
        assert api["environment"] == {"API_KEY_NAME": "demo", "PRICE": "$$5"}
        # Prepare's image list sees the substituted names.
        assert compose_image_references(task) == [MAIN_IMAGE, "redis:7-alpine"]

    def test_env_file_path_may_use_context_dir_and_values_stay_literal(self, tmp_path):
        task_dir = write_task(
            tmp_path,
            "ctr",
            compose="services:\n  api:\n    image: redis:7\n    env_file: ${CONTEXT_DIR}/api.env\n",
        )
        (task_dir / "environment" / "api.env").write_text("PW='ab$cd'\nREF=${MAIN_IMAGE_NAME}\n")
        (task,) = discover_tasks(tmp_path)
        environment = load_compose_document(task)["services"]["api"]["environment"]
        # env_file values are literal: a dollar is kept (doubled for the start-time literal check), never resolved.
        assert environment == {"PW": "ab$$cd", "REF": "$${MAIN_IMAGE_NAME}"}

    def test_a_missing_env_file_fails_prepare_naming_it(self, tmp_path):
        write_task(tmp_path, "ctr", compose="services:\n  api:\n    image: redis:7\n    env_file: secrets.env\n")
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError, match="env_file 'secrets.env' not found"):
            load_compose_document(task)

    def test_a_task_without_compose_has_no_document(self, tmp_path):
        write_task(tmp_path, "plain")
        (task,) = discover_tasks(tmp_path)
        assert load_compose_document(task) is None


class TestRecordComposeImages:
    def test_writes_one_entry_per_image_next_to_the_tasks(self, tmp_path):
        write_task(tmp_path, "ctr", compose=COMPOSE)
        write_task(tmp_path, "plain")
        client = FakeClient()
        path = record_compose_images(discover_tasks(tmp_path), tmp_path, client=client)
        assert path == tmp_path / "compose-images.json"
        assert sorted(json.loads(path.read_text())) == sorted([MAIN_IMAGE, SIDECAR_IMAGE, "redis:7-alpine"])
        assert client.resolved == [MAIN_IMAGE, SIDECAR_IMAGE, "redis:7-alpine"]

    def test_recorded_entries_are_kept_and_only_missing_ones_resolved(self, tmp_path):
        write_task(tmp_path, "ctr", compose=COMPOSE)
        pinned = {
            "redis:7-alpine": {
                "architecture": "amd64",
                "config": {},
                "config_digest": "sha256:old",
                "image": "library/redis@sha256:old",
                "os": "linux",
            }
        }
        (tmp_path / "compose-images.json").write_text(json.dumps(pinned))
        client = FakeClient()
        record_compose_images(discover_tasks(tmp_path), tmp_path, client=client)
        recorded = json.loads((tmp_path / "compose-images.json").read_text())
        assert recorded["redis:7-alpine"] == pinned["redis:7-alpine"]
        assert client.resolved == [MAIN_IMAGE, SIDECAR_IMAGE]

    def test_a_complete_file_means_no_registry_traffic(self, tmp_path):
        write_task(tmp_path, "ctr", compose=COMPOSE)
        tasks = discover_tasks(tmp_path)
        record_compose_images(tasks, tmp_path, client=FakeClient())
        client = FakeClient()
        record_compose_images(tasks, tmp_path, client=client)
        assert client.resolved == []

    def test_nothing_is_written_without_compose_tasks(self, tmp_path):
        write_task(tmp_path, "plain")
        assert record_compose_images(discover_tasks(tmp_path), tmp_path, client=FakeClient()) is None
        assert not (tmp_path / "compose-images.json").exists()

    def two_bad_tasks_among_three(self, tmp_path, monkeypatch):
        monkeypatch.delenv("X", raising=False)
        write_task(tmp_path, "good", compose=COMPOSE)
        write_task(
            tmp_path,
            "bad-required",
            compose="services:\n  api:\n    image: redis:7\n    environment:\n      - X=${X:?need X}\n",
        )
        write_task(
            tmp_path, "bad-env-file", compose="services:\n  api:\n    image: redis:7\n    env_file: secrets.env\n"
        )
        return discover_tasks(tmp_path)

    def test_tasks_whose_compose_file_does_not_resolve_are_skipped_with_reasons_when_asked(
        self, tmp_path, monkeypatch
    ):
        tasks = self.two_bad_tasks_among_three(tmp_path, monkeypatch)
        client = FakeClient()
        skipped: dict[str, str] = {}
        path = record_compose_images(tasks, tmp_path, client=client, skipped=skipped)
        # The good task's images are recorded; each bad task is reported by id with its own reason.
        assert path == tmp_path / "compose-images.json"
        assert client.resolved == [MAIN_IMAGE, SIDECAR_IMAGE, "redis:7-alpine"]
        assert sorted(skipped) == ["bad-env-file", "bad-required"]
        assert "env_file 'secrets.env' not found" in skipped["bad-env-file"]
        assert skipped["bad-required"] == "bad-required: docker-compose.yaml: ${X} is required: need X"

    def test_without_a_skipped_map_one_error_names_every_task_that_does_not_resolve(self, tmp_path, monkeypatch):
        tasks = self.two_bad_tasks_among_three(tmp_path, monkeypatch)
        client = FakeClient()
        with pytest.raises(ImageConfigError, match=r"2 Compose task\(s\) do not resolve: ") as info:
            record_compose_images(tasks, tmp_path, client=client)
        message = str(info.value)
        assert "bad-required: docker-compose.yaml: ${X} is required: need X" in message
        assert "bad-env-file: docker-compose.yaml: " in message and "secrets.env" in message
        assert client.resolved == [] and not (tmp_path / "compose-images.json").exists()


class TestDockerfileBaseImages:
    def test_lists_the_base_image_of_every_dockerfile_the_loader_resolves(self, tmp_path):
        write_task(tmp_path, "pull", image=None, dockerfile="FROM ubuntu:24.04\nWORKDIR /app\n")
        write_task(tmp_path, "overlay", image=None, dockerfile="FROM ubuntu:24.04\nRUN true\n")
        write_task(tmp_path, "settings-next-to-image", dockerfile="FROM python:3.12\nENV A=1\n")
        # A prebuilt image is the build's result: its overlay Dockerfile is not resolved, so no base image.
        write_task(tmp_path, "prebuilt", dockerfile="FROM python:3.12-slim\nRUN true\n")
        # Dockerfiles that need a build and tasks that do not parse are the loader's to report.
        write_task(tmp_path, "needs-build", image=None, dockerfile="FROM ubuntu\nCOPY . /app\n")
        write_task(tmp_path, "plain")
        broken = write_task(tmp_path, "broken", image=None, dockerfile="FROM alpine\n")
        (broken / "task.toml").write_text("not = [toml")
        assert dockerfile_base_images(tmp_path) == {
            "ubuntu:24.04": ["overlay", "pull"],
            "python:3.12": ["settings-next-to-image"],
        }

    def test_a_single_task_folder_is_scanned_as_one_task(self, tmp_path):
        task = write_task(tmp_path, "one", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n")
        assert dockerfile_base_images(task) == {"redis:7-alpine": ["one"]}

    def test_a_dataset_toml_docker_image_override_makes_the_dockerfile_prebuilt(self, tmp_path):
        (tmp_path / "dataset.toml").write_text(
            '[gym.tasks."t".environment]\ndocker_image = "ghcr.io/org/t:prebuilt"\n'
            '[gym.tasks."pull".environment]\ndocker_image = "ghcr.io/org/pull:img"\n'
        )
        # The override declares a prebuilt image, so the Dockerfile's RUN lines are already applied and the
        # loader never resolves its FROM: no base image to record, whether the scan starts at the dataset
        # or at the task folder (whose dataset.toml sits beside it).
        write_task(tmp_path, "t", image=None, dockerfile="FROM localbuild/base:dev\nRUN make\n")
        # A pull-mode Dockerfile next to an overridden image still contributes its ENV, so its base is recorded.
        write_task(tmp_path, "pull", image=None, dockerfile="FROM redis:7\nENV A=1\n")
        assert dockerfile_base_images(tmp_path) == {"redis:7": ["pull"]}
        assert dockerfile_base_images(tmp_path / "t") == {}

    def test_select_limits_the_scan_to_the_chosen_tasks(self, tmp_path):
        write_task(tmp_path, "a", image=None, dockerfile="FROM ubuntu:24.04\nRUN true\n")
        write_task(tmp_path, "z", image=None, dockerfile="FROM internal.registry/base:dev\nRUN true\n")
        assert dockerfile_base_images(tmp_path, select=lambda task_id: task_id == "a") == {"ubuntu:24.04": ["a"]}

    def test_a_broken_dataset_toml_records_nothing(self, tmp_path):
        # Every task fails to load with the dataset.toml error, which the loader reports; no image is fetched for it.
        write_task(tmp_path, "a", image=None, dockerfile="FROM ubuntu:24.04\nRUN true\n")
        (tmp_path / "dataset.toml").write_text("not = [toml")
        assert dockerfile_base_images(tmp_path) == {}


class TestRecordBaseImages:
    def test_records_next_to_the_task_folders_with_the_image_environment(self, tmp_path, registry):
        write_task(tmp_path, "overlay", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n")
        path = record_base_images(tmp_path)
        assert path == tmp_path / "compose-images.json"
        recorded = json.loads(path.read_text())["redis:7-alpine"]
        assert recorded["config"]["Env"] == ["A=1"] and recorded["config"]["WorkingDir"] == "/app"
        assert recorded["image"] == "library/redis@sha256:m-amd64"

    def test_a_single_task_folder_records_into_its_parent(self, tmp_path, registry):
        task = write_task(tmp_path, "one", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n")
        assert record_base_images(task) == tmp_path / "compose-images.json"

    def test_nothing_is_written_without_dockerfiles_to_resolve(self, tmp_path):
        write_task(tmp_path, "plain")
        assert record_base_images(tmp_path, client=FakeClient()) is None
        assert not (tmp_path / "compose-images.json").exists()

    def test_a_prebuilt_override_does_not_touch_the_registry(self, tmp_path):
        (tmp_path / "dataset.toml").write_text(
            '[gym.tasks."t".environment]\ndocker_image = "ghcr.io/org/t:prebuilt"\n'
        )
        write_task(tmp_path, "t", image=None, dockerfile="FROM localbuild/base:dev\nRUN make\n")

        class FailingClient(RegistryClient):
            def image_config(self, reference):
                raise ImageConfigError(f"{reference}: registry returned HTTP 404")

        assert record_base_images(tmp_path, client=FailingClient()) is None
        assert not (tmp_path / "compose-images.json").exists()
        # The loader agrees: the task runs the prebuilt image with no overlay.
        (task,) = discover_tasks(tmp_path)
        assert task.image == "ghcr.io/org/t:prebuilt" and task.overlay == ()

    def test_an_unresolvable_base_image_names_the_image_the_tasks_and_the_registry_answer(self, tmp_path, registry):
        write_task(tmp_path, "a", image=None, dockerfile="FROM redis:missing\nRUN true\n")
        write_task(tmp_path, "b", image=None, dockerfile="FROM redis:missing\nENV X=$Y\n")
        with pytest.raises(ImageConfigError) as excinfo:
            record_base_images(tmp_path)
        message = str(excinfo.value)
        assert "base image 'redis:missing'" in message
        assert "task(s) a, b" in message
        assert "library/redis: registry returned HTTP 404 for manifests/missing" in message
        assert not (tmp_path / "compose-images.json").exists()


class TestRecordImageConfigsUnderTheLock:
    """D2: compose-images.json is merged and replaced under its sidecar lock; the registry is asked outside it."""

    def test_an_entry_another_process_recorded_during_the_lookup_is_kept(self, tmp_path):
        write_task(tmp_path, "ctr", compose=COMPOSE)
        record_path = tmp_path / "compose-images.json"
        competitor = {"architecture": "arm64", "config": {}, "config_digest": "sha256:other", "os": "linux"}

        class RacingClient(FakeClient):
            def image_config(self, reference: str) -> dict:
                # Another process records the same reference, and one more, while this lookup is in flight.
                if reference == "redis:7-alpine":
                    write_json_record(record_path, {"redis:7-alpine": competitor, "busybox:1": competitor})
                return super().image_config(reference)

        client = RacingClient()
        assert record_compose_images(discover_tasks(tmp_path), tmp_path, client=client) == record_path
        recorded = json.loads(record_path.read_text())
        # Neither writer's entries are lost, and the first recording of a reference is the one kept.
        assert sorted(recorded) == sorted([MAIN_IMAGE, SIDECAR_IMAGE, "redis:7-alpine", "busybox:1"])
        assert recorded["redis:7-alpine"] == competitor and recorded["busybox:1"] == competitor
        assert recorded[MAIN_IMAGE]["image"] == MAIN_IMAGE
        assert client.resolved == [MAIN_IMAGE, SIDECAR_IMAGE, "redis:7-alpine"]
        assert (tmp_path / "compose-images.json.lock").is_file()
        assert not [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")]

    def test_a_record_file_that_stays_invalid_is_reported_by_name(self, tmp_path):
        write_task(tmp_path, "ctr", compose=COMPOSE)
        (tmp_path / "compose-images.json").write_text('{"redis:7-alpine": {"os": "linux"')
        with pytest.raises(HarborTaskError, match=r"compose-images\.json is not valid JSON \(read twice"):
            record_compose_images(discover_tasks(tmp_path), tmp_path, client=FakeClient())
