# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import nemo_gym.tasks.harbor.compose_variables as compose_variables_module
from nemo_gym.sandbox.compose_config import _compose_literals
from nemo_gym.sandbox.compose_fields import service_gpus
from nemo_gym.tasks.harbor import image_configs
from nemo_gym.tasks.harbor import registry as registry_module
from nemo_gym.tasks.harbor.compose_variables import compose_variables
from nemo_gym.tasks.harbor.digest import content_hash, context_digest, is_covered
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_configs import (
    check_task_compose_fields,
    compose_image_references,
    dockerfile_base_images,
    load_compose_document,
    plan_task_compose_volumes,
    record_base_images,
    record_compose_images,
    with_built_images,
)
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, write_json_record
from nemo_gym.tasks.harbor.registry import ImageConfigError, RegistryClient, parse_image_ref
from nemo_gym.tasks.harbor.task import discover_tasks


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
            # An amd64/v3 build: the same os/arch in the blob (a configuration blob may omit the variant), another Cmd.
            "sha256:cfg-amd64-v3": {"os": "linux", "architecture": "amd64", "config": {"Cmd": ["server-v3"]}},
        }
        self.manifests = {
            "sha256:m-amd64": {"mediaType": self.MANIFEST, "config": {"digest": "sha256:cfg-amd64"}},
            "sha256:m-arm64": {"mediaType": self.MANIFEST, "config": {"digest": "sha256:cfg-arm64"}},
            "sha256:m-att": {"mediaType": self.MANIFEST, "config": {"digest": "sha256:cfg-amd64"}},
            "sha256:m-amd64-v3": {"mediaType": self.MANIFEST, "config": {"digest": "sha256:cfg-amd64-v3"}},
        }
        self.tags = {
            # The Docker Official Images' shape (ubuntu:24.04, redis, python): the arm64 entry declares variant v8.
            "7-alpine": {
                "mediaType": self.INDEX,
                "manifests": [
                    {
                        "digest": "sha256:m-arm64",
                        "platform": {"os": "linux", "architecture": "arm64", "variant": "v8"},
                    },
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
                "manifests": [
                    {
                        "digest": "sha256:m-arm64",
                        "platform": {"os": "linux", "architecture": "arm64", "variant": "v8"},
                    }
                ],
            },
            # An amd64 entry at micro-architecture level v3 listed before the baseline one: another platform to
            # containerd's matcher (v2..v4 need CPU features a baseline host may lack), with its own configuration.
            "amd64-v3": {
                "mediaType": self.INDEX,
                "manifests": [
                    {
                        "digest": "sha256:m-amd64-v3",
                        "platform": {"os": "linux", "architecture": "amd64", "variant": "v3"},
                    },
                    {"digest": "sha256:m-amd64", "platform": {"os": "linux", "architecture": "amd64"}},
                ],
            },
            "amd64-v3-only": {
                "mediaType": self.INDEX,
                "manifests": [
                    {
                        "digest": "sha256:m-amd64-v3",
                        "platform": {"os": "linux", "architecture": "amd64", "variant": "v3"},
                    }
                ],
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
            # A registry serves an index by its own digest as it serves it by tag: that is the pinned reference.
            manifest = (
                self.tags.get(ref) or self.manifests.get(ref) or self.tags.get(ref.removeprefix("sha256:index-of-"))
            )
            if manifest is None:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
            digest = ref if ref.startswith("sha256:") else "sha256:index-of-" + ref
            return FakeResponse(json.dumps(manifest).encode(), {"Docker-Content-Digest": digest})
        if kind == "blobs":
            return FakeResponse(json.dumps(self.blobs[ref]).encode())
        raise AssertionError(url)


def patch_opener(monkeypatch, fake) -> None:
    """Route every request the client's opener makes (redirect handling included) to ``fake``."""
    monkeypatch.setattr(
        urllib.request.OpenerDirector, "open", lambda self, request, data=None, timeout=None: fake(request, timeout)
    )


@pytest.fixture
def registry(monkeypatch):
    fake = FakeRegistry()
    patch_opener(monkeypatch, fake)
    return fake


def serve(handler: type[BaseHTTPRequestHandler]) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestRegistryRedirects:
    def test_a_redirect_to_another_host_drops_the_registry_token_and_a_same_host_one_keeps_it(self):
        seen: dict[str, list[tuple[str, dict[str, str]]]] = {"registry": [], "storage": []}

        class Quiet(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body: bytes = b"{}") -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class Storage(Quiet):
            def do_GET(self):
                seen["storage"].append((self.path, dict(self.headers)))
                self.reply()

        storage = serve(Storage)

        class Registry(Quiet):
            def do_GET(self):
                seen["registry"].append((self.path, dict(self.headers)))
                if self.path == "/same-host":
                    return self.reply(b"same")
                # Blobs live on a storage host, as Docker Hub, GHCR and ECR serve them; manifests redirect within the registry.
                if "/blobs/" in self.path:
                    location = f"http://127.0.0.1:{storage.server_port}/blob"
                else:
                    location = f"http://127.0.0.1:{registry.server_port}/same-host"
                self.send_response(307)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

        registry = serve(Registry)
        try:
            client = RegistryClient(retries=0)
            base = f"http://127.0.0.1:{registry.server_port}/v2/library/x"
            headers = {"Authorization": "Bearer tok", "Accept": "*/*"}
            body, _ = client._open(f"{base}/blobs/sha256:a", headers)
            assert body == b"{}"
            (storage_request,) = seen["storage"]
            assert storage_request[0] == "/blob" and "Authorization" not in storage_request[1]
            body, _ = client._open(f"{base}/manifests/latest", headers)
            assert body == b"same"
            same_host = [request for path, request in seen["registry"] if path == "/same-host"]
            assert same_host and same_host[0]["Authorization"] == "Bearer tok"
            assert seen["registry"][0][1]["Authorization"] == "Bearer tok"
        finally:
            registry.shutdown()
            storage.shutdown()


class TestRegistryClient:
    def test_records_the_linux_amd64_configuration_pinned_to_the_index_behind_a_floating_tag(self, registry):
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
            # The pin is the index the tag named, not the amd64 manifest the configuration was read from.
            "image": "library/redis@sha256:index-of-7-alpine",
            "os": "linux",
            "record_version": 2,
        }

    def test_a_digest_reference_is_pinned_to_itself(self, registry):
        entry = RegistryClient().image_config("ghcr.io/org/app@sha256:m-amd64")
        assert entry["image"] == "ghcr.io/org/app@sha256:m-amd64"
        assert registry.requests[-1][0] == "https://ghcr.io/v2/org/app/blobs/sha256:cfg-amd64"

    def test_the_pinned_reference_names_the_whole_index_and_resolves_to_the_same_pin(self, registry):
        """What a task starts is ``repo@<index digest>``: the registry serves the index for it (every platform's
        manifest is still selectable from it) and reading it back lands on the same pin and configuration."""
        pinned = RegistryClient().image_config("redis:7-alpine")["image"]
        del registry.requests[:]
        again = RegistryClient().image_config(pinned)
        assert again["image"] == pinned and again["config_digest"] == "sha256:cfg-amd64"
        manifests = [url.rpartition("/manifests/")[2] for url, _ in registry.requests if "/manifests/" in url]
        # (the first request is repeated with the token the registry's challenge asked for)
        assert list(dict.fromkeys(manifests)) == ["sha256:index-of-7-alpine", "sha256:m-amd64"]
        assert RegistryClient().image_config("redis@sha256:m-amd64")["image"] == "library/redis@sha256:m-amd64", (
            "a single-platform reference has no index to pin to: its one manifest is the pin"
        )

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

    def test_resolves_for_the_platform_asked_for(self, registry):
        """``builds.platform: linux/arm64`` deployments record the arm64 configuration. The pin stays the index,
        so one pinned reference serves an arm64 pod and an amd64 one, each selecting its own manifest from it.
        Review (M24): the index's arm64 entry carries ``variant: v8``, as every Docker Official Image's does, and
        ``linux/arm64`` selects it as ``docker --platform linux/arm64`` does (containerd reads arm64 as arm64/v8);
        the selection used to refuse any entry with a variant and report that the index had no arm64 manifest."""
        client = RegistryClient(platform=("linux", "arm64"))
        entry = client.image_config("redis:7-alpine")
        assert entry["image"] == "library/redis@sha256:index-of-7-alpine" and entry["architecture"] == "arm64"
        assert entry["config_digest"] == "sha256:cfg-arm64"
        assert client.image_config(entry["image"])["config_digest"] == "sha256:cfg-arm64"
        assert RegistryClient().image_config(entry["image"])["config_digest"] == "sha256:cfg-amd64"
        assert client.image_config("redis:arm-only")["config_digest"] == "sha256:cfg-arm64"
        with pytest.raises(ImageConfigError, match="linux/amd64, not linux/arm64"):
            client.image_config("redis@sha256:m-amd64")
        # The explicit variant and the architecture alias name the same platform and select the same manifest.
        for platform in ("linux/arm64", "linux/arm64/v8", "linux/aarch64"):
            spelled = RegistryClient(platform=registry_module.platform_pair(platform))
            assert spelled.platform == ("linux", "arm64", "v8")
            assert spelled.image_config("redis:7-alpine")["config_digest"] == "sha256:cfg-arm64"
        # Review (M24): an amd64 entry at micro-architecture level v3 is not baseline amd64. containerd's matcher
        # (the rule `docker --platform linux/amd64` selects by) answers a baseline request with an entry of variant
        # v1 or none only, since v2..v4 need CPU features a baseline host may lack; the levels used to be folded into
        # one platform, so the default client recorded v3's configuration whenever v3 was listed first. Baseline is
        # selected past a v3 entry, a v3 request selects v3, and an index of v3 alone has no linux/amd64 manifest.
        assert RegistryClient().image_config("redis:amd64-v3")["config_digest"] == "sha256:cfg-amd64"
        v3 = RegistryClient(platform=("linux", "amd64", "v3"))
        assert v3.platform == ("linux", "amd64", "v3")
        assert v3.image_config("redis:amd64-v3")["config_digest"] == "sha256:cfg-amd64-v3"
        with pytest.raises(ImageConfigError, match="no linux/amd64 manifest"):
            RegistryClient().image_config("redis:amd64-v3-only")
        assert RegistryClient(platform=("linux", "amd64", "v1")).platform == ("linux", "amd64", "")
        # Another variant of a 32-bit arm is another platform, and the message names it.
        with pytest.raises(ImageConfigError, match="no linux/arm/v6 manifest"):
            RegistryClient(platform=("linux", "arm", "v6")).image_config("redis:7-alpine")

    def test_record_image_configs_resolves_for_builds_platform(self, tmp_path, monkeypatch):
        platforms = []

        class Recording:
            def __init__(self, *args, **kwargs):
                platforms.append(kwargs.get("platform"))

            def image_config(self, reference):
                return {"os": "linux", "architecture": "arm64", "image": reference, "config": {}}

        monkeypatch.setattr(image_configs, "RegistryClient", Recording)
        image_configs.record_image_configs({"redis:7": ["a"]}, tmp_path, platform="linux/arm64/v8")
        image_configs.record_image_configs({"redis:8": ["a"]}, tmp_path)
        assert platforms == [("linux", "arm64", "v8"), None], "the default client keeps its own default platform"
        assert registry_module.platform_pair(None) == registry_module.platform_pair("") == ("linux", "amd64", "")
        # Normalised as containerd normalises a platform: the variant arm64 defaults to, aliases, amd64's baseline
        # level v1. A higher amd64 level is another platform, and the records keep os/arch only (reusable_record,
        # recorded_image_mismatch), so builds.platform cannot ask for one: a v3 record would serve a baseline run.
        assert registry_module.platform_pair("linux/arm64/v8") == ("linux", "arm64", "v8")
        assert registry_module.platform_pair("linux/aarch64") == registry_module.platform_pair("linux/arm64")
        assert registry_module.platform_pair("linux/x86_64") == registry_module.platform_pair("linux/amd64/v1")
        assert registry_module.platform_pair("linux/amd64/v1") == ("linux", "amd64", "")
        with pytest.raises(ImageConfigError, match="Unsupported platform: 'linux/amd64/v3'"):
            registry_module.platform_pair("linux/amd64/v3")
        assert registry_module.platform_pair("linux/arm") == ("linux", "arm", "v7")
        assert registry_module.platform_name(registry_module.platform_pair("linux/arm64/v8")) == "linux/arm64"
        assert registry_module.platform_name(registry_module.platform_pair("linux/arm/v6")) == "linux/arm/v6"
        with pytest.raises(ImageConfigError, match="Not a platform: 'arm64'"):
            registry_module.platform_pair("arm64")
        with pytest.raises(ImageConfigError, match="Not a platform: 'linux/arm64/v8/extra'"):
            registry_module.platform_pair("linux/arm64/v8/extra")

    def test_http_errors_carry_the_repository_and_status(self, registry):
        with pytest.raises(ImageConfigError, match="library/redis: registry returned HTTP 404"):
            RegistryClient().image_config("redis:missing")

    def test_connection_failures_are_retried_then_reported(self, monkeypatch):
        attempts = []

        def flaky(request, timeout=None):
            attempts.append(request.full_url)
            raise urllib.error.URLError("handshake timed out")

        patch_opener(monkeypatch, flaky)
        monkeypatch.setattr(registry_module.time, "sleep", lambda seconds: None)
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

    @pytest.mark.parametrize(
        ("service", "message"),
        [
            ("    build: ./api\n", "service 'api' has a build section and prepare recorded no build for it"),
            (
                "    image: skillsbench-visual-stability-api\n    build: ./api\n",
                "service 'api' has a build section and prepare recorded no build for it",
            ),
            ("    expose: ['1']\n", "service 'api' names no image and prepare recorded no build for it"),
        ],
        ids=["build-only", "image-and-build", "neither"],
    )
    def test_a_sidecar_with_a_build_and_no_record_is_refused_naming_it(self, tmp_path, service, message):
        # Under Compose and Harbor a service with build: is built, never pulled, so its image: (when it has one) is
        # a local tag: it is not resolved from the registry, and without a record the task is refused by name. A
        # service with neither key is a different task defect, reported in its own words.
        write_task(tmp_path, "ctr", compose="services:\n  api:\n" + service)
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError, match=message):
            compose_image_references(task)

    def test_a_recorded_main_is_not_substituted_for_a_task_pinned_to_a_prebuilt_image(self, tmp_path):
        # The context's main was built (for another task sharing environment/, or before the override) and the
        # record names it; this task declares a docker_image, so Harbor runs the prebuilt image and so does Gym:
        # main keeps ${MAIN_IMAGE_NAME} = task.image, and the sidecar is substituted as it always is.
        compose = "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n  db:\n    build: ./db\n"
        task_dir = write_task(
            tmp_path, "t", image=None, compose=compose, dockerfile="FROM ubuntu:24.04\nCOPY . /app\n"
        )
        (tmp_path / "dataset.toml").write_text('[gym.tasks."t".environment]\ndocker_image = "org/pinned:2"\n')
        built = "reg.example/r/hb__" + "0" * 32 + "@sha256:" + "d" * 64
        db = "reg.example/r/hb__" + "0" * 32 + "-db@sha256:" + "e" * 64
        record = {"image": built, "tasks": ["t"], "services": {"main": {"image": built}, "db": {"image": db}}}
        (tmp_path / BUILT_IMAGES_FILE).write_text(json.dumps({context_digest(task_dir / "environment"): record}))
        config = {"os": "linux", "architecture": "amd64", "config": {"WorkingDir": "/app"}, "record_version": 2}
        (tmp_path / "compose-images.json").write_text(
            json.dumps({image: {**config, "image": image} for image in (built, db, "org/pinned:2")})
        )
        (task,) = discover_tasks(tmp_path)
        assert task.image == "org/pinned:2" and task.built_image is None
        assert task.built_images == {"main": built, "db": db}
        document = load_compose_document(task)
        assert document["services"]["main"] == {"image": "org/pinned:2"}
        assert document["services"]["db"] == {"image": db}
        assert compose_image_references(task) == ["org/pinned:2", db]
        # The complementary case: the override names another task, so this one runs from its built main.
        (tmp_path / "dataset.toml").write_text('[gym.tasks."other".environment]\ndocker_image = "org/pinned:2"\n')
        (task,) = discover_tasks(tmp_path)
        assert task.built_image == built == task.image
        assert load_compose_document(task)["services"]["main"] == {"image": built}
        assert compose_image_references(task) == [built, db]

    def test_a_sidecar_prepare_built_is_listed_by_its_recorded_image_and_not_resolved_again(self, tmp_path):
        # G2: prepare built `api` from ./api and recorded it under the context digest by service name; the loader
        # substitutes that image, so the reference is the recorded one and the registry is never asked for it.
        compose = (
            "services:\n  main: {}\n  api:\n    build: ./api\n    pull_policy: build\n    expose: ['3080']\n"
            "  cache:\n    image: redis:7-alpine\n"
        )
        task_dir = write_task(tmp_path, "ctr", compose=compose)
        built = "reg.example/r/hb__" + "0" * 32 + "-api@sha256:" + "e" * 64
        (tmp_path / BUILT_IMAGES_FILE).write_text(
            json.dumps(
                {context_digest(task_dir / "environment"): {"tasks": ["ctr"], "services": {"api": {"image": built}}}}
            )
        )
        (tmp_path / "compose-images.json").write_text(
            json.dumps(
                {
                    built: {
                        "os": "linux",
                        "architecture": "amd64",
                        "image": built,
                        "config": {"Cmd": ["api"]},
                        "record_version": 2,
                    }
                }
            )
        )
        (task,) = discover_tasks(tmp_path)
        assert compose_image_references(task) == [MAIN_IMAGE, built, "redis:7-alpine"]
        document = load_compose_document(task)
        assert document["services"]["api"] == {"image": built, "expose": ["3080"]}
        assert document["services"]["main"] == {}
        client = FakeClient()
        record_compose_images([task], tmp_path, client=client)
        assert client.resolved == [MAIN_IMAGE, "redis:7-alpine"]
        recorded = json.loads((tmp_path / "compose-images.json").read_text())
        assert recorded[built]["config"] == {"Cmd": ["api"]}

    def test_with_built_images_drops_the_build_and_its_policy_only_on_named_services(self):
        """A recorded service loses its ``build:`` and a ``pull_policy: build`` (Harbor's template spelling, in any
        case), the two that describe the build prepare ran. A runtime policy (``always``, ``missing``, ``never``) is
        not part of the build and stays with the recorded image (review: it was dropped with the build, so a built
        service's policy never reached the field check or the adapter)."""
        document = {
            "services": {
                "main": {"image": "x", "build": {"context": "."}, "pull_policy": "build", "cpus": 1},
                "api": {"build": "./api", "pull_policy": "Build ", "expose": ["1"]},
                "worker": {"build": "./worker", "pull_policy": "always"},
                "cache": {"image": "local/cache", "build": "./cache", "pull_policy": "never"},
                "db": {"image": "pg:16", "build": "./db"},
                "empty": None,
            }
        }
        images = {
            "main": "reg/hb__d@sha256:1",
            "api": "reg/hb__d-api@sha256:2",
            "worker": "reg/hb__d-worker@sha256:4",
            "cache": "reg/hb__d-cache@sha256:5",
            "empty": "reg/hb__d-empty@sha256:3",
            "gone": "reg/x",
        }
        assert with_built_images(document, images) == {
            "services": {
                "main": {"cpus": 1, "image": "reg/hb__d@sha256:1"},
                "api": {"expose": ["1"], "image": "reg/hb__d-api@sha256:2"},
                "worker": {"pull_policy": "always", "image": "reg/hb__d-worker@sha256:4"},
                "cache": {"pull_policy": "never", "image": "reg/hb__d-cache@sha256:5"},
                # Not recorded: left as written, so the adapter (or compose_image_references) reports it.
                "db": {"image": "pg:16", "build": "./db"},
                "empty": {"image": "reg/hb__d-empty@sha256:3"},
            }
        }
        assert with_built_images({"services": "nope"}, images) == {"services": "nope"}


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
            "record_version": 2,
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
        monkeypatch.setattr(compose_variables_module, "_WARNED_UNSET", set())
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
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.compose_variables"):
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
        monkeypatch.setattr(compose_variables_module, "_WARNED_UNSET", set())
        monkeypatch.delenv("SIDECAR_IMAGE", raising=False)
        write_task(tmp_path, "ctr", compose="services:\n  api:\n    image: ${SIDECAR_IMAGE}\n")
        (task,) = discover_tasks(tmp_path)
        with caplog.at_level(logging.WARNING, logger="nemo_gym.tasks.harbor.compose_variables"):
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

    @pytest.mark.asyncio
    async def test_interpolated_healthcheck_fields_keep_their_types_through_to_the_adapter(self, tmp_path):
        """``${DISABLE:-false}`` must not disable the check by being a non-empty string, and ``${RETRIES:-3}``
        must reach the adapter as an integer: the loader restores the types after interpolation."""
        from types import SimpleNamespace

        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
        from nemo_gym.sandbox.compose_config import resolve_compose

        write_task(
            tmp_path,
            "ctr",
            compose=(
                "services:\n  api:\n    image: redis:7\n    healthcheck:\n"
                '      test: ["CMD", "redis-cli", "ping"]\n'
                "      disable: ${DISABLE:-false}\n      retries: ${RETRIES:-3}\n"
            ),
        )
        (task,) = discover_tasks(tmp_path)
        document = load_compose_document(task)
        assert document["services"]["api"]["healthcheck"] == {
            "test": ["CMD", "redis-cli", "ping"],
            "disable": False,
            "retries": 3,
        }
        records = {
            image: {"os": "linux", "architecture": "amd64", "image": image, "config": {"Cmd": ["run"]}}
            for image in (MAIN_IMAGE, "redis:7")
        }
        group = AsyncSandboxCompose(SimpleNamespace(), "compose.yaml")
        group.document = resolve_compose(document, MAIN_IMAGE, records)
        await group._prepare()
        health = group._plans["api"]["health"]
        assert health["test"] == ["CMD", "redis-cli", "ping"] and health["retries"] == 3

    def test_a_missing_env_file_fails_prepare_naming_it(self, tmp_path):
        write_task(tmp_path, "ctr", compose="services:\n  api:\n    image: redis:7\n    env_file: secrets.env\n")
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError, match="env_file 'secrets.env' not found"):
            load_compose_document(task)

    def test_a_task_without_compose_has_no_document(self, tmp_path):
        write_task(tmp_path, "plain")
        (task,) = discover_tasks(tmp_path)
        assert load_compose_document(task) is None


FIXTURES = Path(__file__).parent / "fixtures" / "harbor_compose_volumes"


class TestComposeVolumesAtPrepare:
    """M22: Compose volumes are planned at prepare, from the task folder and the deployment's ``volume_sources``."""

    def write_task_aw(self, root: Path) -> Path:
        task_dir = write_task(root, "task-aw")
        for name in ("docker-compose.yaml", "seed.json"):
            (task_dir / "environment" / name).write_bytes((FIXTURES / "task-aw" / "environment" / name).read_bytes())
        return task_dir

    def test_task_aw_seed_file_becomes_an_upload_into_the_sidecar(self, tmp_path):
        self.write_task_aw(tmp_path)
        (task,) = discover_tasks(tmp_path)
        document = load_compose_document(task)
        uploads = plan_task_compose_volumes(task, document)
        assert uploads == {
            "api": [{"source": str(task.path / "environment" / "seed.json"), "target": "/app/seed.json"}]
        }
        assert document["services"]["api"]["x-sandbox"]["uploads"] == uploads["api"]
        assert "volumes" not in document["services"]["api"]
        assert document["services"]["main"]["image"] == MAIN_IMAGE

    def test_record_compose_images_fails_prepare_naming_an_unmapped_external_volume(self, tmp_path):
        task_dir = write_task(
            tmp_path,
            "pg",
            compose=f"""services:
  main:
    image: {MAIN_IMAGE}
  db:
    image: {SIDECAR_IMAGE}
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
    external: true
""",
        )
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError, match=r"pg: environment/docker-compose.yaml: external volume 'pgdata'"):
            record_compose_images([task], tmp_path, client=FakeClient())
        assert not (tmp_path / "compose-images.json").exists()
        # Mapped in the deployment's sandbox block, the same dataset prepares.
        path = record_compose_images([task], tmp_path, client=FakeClient(), volume_sources={"pgdata": "pg/data"})
        assert sorted(json.loads(path.read_text())) == sorted([MAIN_IMAGE, SIDECAR_IMAGE])
        assert task_dir.is_dir()

    @pytest.mark.parametrize("service", ["main", "api"])
    @pytest.mark.parametrize("source", ["/etc/passwd", "OUTSIDE"])
    def test_a_task_authored_x_sandbox_upload_fails_prepare_naming_the_field(self, tmp_path, service, source):
        # Review blocker: a hub task's Compose file is untrusted. An ``x-sandbox.uploads`` written there (the
        # planner's own output form) would have the server copy any readable host file into the sandbox.
        outside = tmp_path / "outside" / "secret"
        outside.parent.mkdir()
        outside.write_text("secret")
        source = str(outside) if source == "OUTSIDE" else source
        smuggled = f"    x-sandbox:\n      uploads:\n        - {{source: {source}, target: /tmp/stolen}}\n"
        compose = f"services:\n  main:\n    image: {MAIN_IMAGE}\n"
        compose += smuggled if service == "main" else ""
        compose += f"  api:\n    image: {SIDECAR_IMAGE}\n"
        compose += smuggled if service == "api" else ""
        write_task(tmp_path / "ds", "smuggler", compose=compose)
        (task,) = discover_tasks(tmp_path / "ds")
        document = load_compose_document(task)
        with pytest.raises(
            ImageConfigError,
            match=f"smuggler: environment/docker-compose.yaml: service {service!r} declares x-sandbox",
        ):
            plan_task_compose_volumes(task, document)
        with pytest.raises(ImageConfigError, match="x-sandbox"):
            record_compose_images([task], tmp_path / "ds", client=FakeClient())
        assert not (tmp_path / "ds" / "compose-images.json").exists()

    def test_a_task_that_needs_shared_storage_is_left_out_at_prepare_when_the_block_has_none(self, tmp_path):
        """M35 D6 (hil-bench's named volume) and D2 (kumo's sidecar bind of ``/logs/verifier``): both need the
        provider's shared storage; told the block has none, prepare leaves the task out naming the setting instead of
        every seed failing; told it has it, the kumo task plans the shared logs volume into both services."""
        hil = write_task(
            tmp_path / "hil",
            "hil",
            compose=f"""services:
  main:
    image: {MAIN_IMAGE}
    volumes:
      - ask-human-metrics:/harbor_shared
  ask-human:
    image: {SIDECAR_IMAGE}
    volumes:
      - ask-human-metrics:/harbor_shared
volumes:
  ask-human-metrics:
""",
        )
        kumo = write_task(
            tmp_path / "kumo",
            "kumo",
            compose=f"""services:
  verifier:
    image: {SIDECAR_IMAGE}
    volumes:
      - ${{HOST_VERIFIER_LOGS_PATH}}:${{ENV_VERIFIER_LOGS_PATH}}
  main:
    image: {MAIN_IMAGE}
    volumes:
      - ${{HOST_VERIFIER_LOGS_PATH}}:${{ENV_VERIFIER_LOGS_PATH}}
      - ${{HOST_AGENT_LOGS_PATH}}:${{ENV_AGENT_LOGS_PATH}}
""",
        )
        assert hil.is_dir() and kumo.is_dir()
        for root, message in (
            (tmp_path / "hil", "mounts 'ask-human-metrics'"),
            (tmp_path / "kumo", "binds the logs tree '/logs/verifier'"),
        ):
            (task,) = discover_tasks(root)
            skipped: dict[str, str] = {}
            record_compose_images([task], root, client=FakeClient(), shared_storage=False, skipped=skipped)
            assert list(skipped) == [task.task_id]
            assert message in skipped[task.task_id] and "shared_storage.host_path" in skipped[task.task_id]
            # The images are still recorded, as for any other refused field.
            assert (root / "compose-images.json").exists()
            for storage in (True, None):
                assert record_compose_images([task], root, client=FakeClient(), shared_storage=storage, skipped={})
        (task,) = discover_tasks(tmp_path / "kumo")
        document = load_compose_document(task)
        assert plan_task_compose_volumes(task, document, shared_storage=True) == {}
        assert document["volumes"] == {"nemo-gym-logs-verifier": {}}
        for name in ("verifier", "main"):
            (mount,) = document["services"][name]["volumes"]
            assert (mount["source"], mount["target"], mount["volume"]) == (
                "nemo-gym-logs-verifier",
                "/logs/verifier",
                {"nocopy": True},
            )
        # A GPU-declaring task is answered for the GPU block, where the server seeds it.
        gpu = write_task(tmp_path / "gpu", "gpu", compose=(hil / "environment" / "docker-compose.yaml").read_text())
        (gpu / "task.toml").write_text((gpu / "task.toml").read_text() + "gpus = 1\n")
        (task,) = discover_tasks(tmp_path / "gpu")
        skipped = {}
        record_compose_images(
            [task],
            tmp_path / "gpu",
            client=FakeClient(),
            shared_storage=True,
            gpu_shared_storage=False,
            skipped=skipped,
        )
        assert "shared_storage.host_path" in skipped[task.task_id]
        assert record_compose_images(
            [task], tmp_path / "gpu", client=FakeClient(), shared_storage=False, gpu_shared_storage=True, skipped={}
        )

    def test_gpu_tasks_are_checked_against_the_gpu_blocks_volume_sources(self, tmp_path):
        # Review: the server seeds a GPU-declaring task from the ``gpu_sandbox_provider`` block, so prepare checks
        # its volumes against that block's mapping (empty when the block has none), not the ``sandbox`` block's.
        compose = f"""services:
  main:
    image: {MAIN_IMAGE}
  db:
    image: {SIDECAR_IMAGE}
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
    external: true
"""
        for name, gpus in (("cpu", 0), ("gpu", 1)):
            task_dir = write_task(tmp_path, name, compose=compose)
            toml = task_dir / "task.toml"
            toml.write_text(toml.read_text() + f"gpus = {gpus}\n")
        cpu, gpu = sorted(discover_tasks(tmp_path), key=lambda task: task.task_id)
        mapped = {"pgdata": "pg/data"}
        # No GPU block configured: every task is checked against the sandbox block.
        assert record_compose_images([cpu, gpu], tmp_path, client=FakeClient(), volume_sources=mapped)
        # A GPU block without the mapping: the GPU task fails at prepare, the CPU task does not.
        with pytest.raises(ImageConfigError, match="gpu: .*external volume 'pgdata'"):
            record_compose_images(
                [cpu, gpu], tmp_path, client=FakeClient(), volume_sources=mapped, gpu_volume_sources={}
            )
        assert record_compose_images(
            [cpu], tmp_path, client=FakeClient(), volume_sources=mapped, gpu_volume_sources={}
        )
        # The mirror: mapped under the GPU block only.
        assert record_compose_images([gpu], tmp_path, client=FakeClient(), gpu_volume_sources=mapped)
        with pytest.raises(ImageConfigError, match="cpu: .*external volume 'pgdata'"):
            record_compose_images([cpu], tmp_path, client=FakeClient(), gpu_volume_sources=mapped)

    def test_a_gpu_reserving_sidecar_of_a_cpu_task_is_checked_against_the_gpu_blocks_volume_sources(self, tmp_path):
        # Review (round 13): the server routes a GPU-reserving sidecar to the GPU block even when the task declares
        # no GPUs, so its external volume is checked against that block's mapping; main and a plain sidecar stay
        # on the task's.
        compose = f"""services:
  main:
    image: {MAIN_IMAGE}
  model:
    image: {SIDECAR_IMAGE}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: 1
    volumes:
      - models:/models
volumes:
  models:
    external: true
"""
        write_task(tmp_path, "cpu", compose=compose)
        (task,) = discover_tasks(tmp_path)
        skipped: dict[str, str] = {}
        record_compose_images(
            [task],
            tmp_path,
            client=FakeClient(),
            volume_sources={},
            gpu_volume_sources={"models": "gpu/models"},
            skipped=skipped,
        )
        assert skipped == {}
        # The task's own block mapping it is not enough once a GPU block exists: the sidecar runs there.
        record_compose_images(
            [task],
            tmp_path,
            client=FakeClient(),
            volume_sources={"models": "cpu/models"},
            gpu_volume_sources={},
            skipped=skipped,
        )
        assert "external volume 'models'" in skipped.pop("cpu")
        # Without a GPU block every service follows the task's block, as before.
        record_compose_images(
            [task], tmp_path, client=FakeClient(), volume_sources={"models": "cpu/models"}, skipped=skipped
        )
        assert skipped == {}

    def test_a_bind_source_outside_the_digest_fails_prepare_while_the_hash_stays_the_publishers(self, tmp_path):
        # Review (round 13): <task>/data/seed.json is uploadable but not digested, so an edit changed nothing in the
        # manifest while the server uploaded the edited bytes. The planner refuses it at prepare; content_hash
        # stays Harbor's publisher hash (the parity proof is test_archive_hash_is_the_published_hash_without_exclusions).
        task_dir = write_task(
            tmp_path,
            "seeded",
            compose=f"services:\n  main:\n    image: {MAIN_IMAGE}\n  api:\n    image: {SIDECAR_IMAGE}\n"
            "    volumes:\n      - ../data/seed.json:/seed.json\n",
        )
        (task_dir / "data").mkdir()
        (task_dir / "data" / "seed.json").write_text("v1")
        (task,) = discover_tasks(tmp_path)
        before = content_hash(task.path)
        (task_dir / "data" / "seed.json").write_text("v2")
        assert content_hash(task.path) == before  # the edit is invisible to the task's identity
        assert not is_covered("data/seed.json") and is_covered("environment/seed.json") and is_covered("tests")
        assert is_covered("task.toml") and not is_covered("environment/__pycache__/x.pyc") and not is_covered(".")
        document = load_compose_document(task)
        with pytest.raises(
            ImageConfigError,
            match=r"seeded: environment/docker-compose.yaml: service 'api' binds '../data/seed.json', which the task digest does not cover",
        ):
            plan_task_compose_volumes(task, document)
        skipped: dict[str, str] = {}
        record_compose_images([task], tmp_path, client=FakeClient(), skipped=skipped)
        assert "../data/seed.json" in skipped["seeded"]

    def test_record_compose_images_fails_prepare_naming_an_unsupported_volume_field(self, tmp_path):
        write_task(
            tmp_path,
            "tmpfs",
            compose=f"services:\n  main:\n    image: {MAIN_IMAGE}\n    volumes:\n      - type: tmpfs\n        target: /scratch\n",
        )
        (task,) = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError, match="volume type 'tmpfs' is not supported"):
            record_compose_images([task], tmp_path, client=FakeClient())


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
                "record_version": 2,
            }
        }
        (tmp_path / "compose-images.json").write_text(json.dumps(pinned))
        client = FakeClient()
        record_compose_images(discover_tasks(tmp_path), tmp_path, client=client)
        recorded = json.loads((tmp_path / "compose-images.json").read_text())
        assert recorded["redis:7-alpine"] == pinned["redis:7-alpine"]
        assert client.resolved == [MAIN_IMAGE, SIDECAR_IMAGE]
        # Review (M24): a record is kept for the platform it was recorded for. Asked for the same platform, nothing is
        # resolved; asked for another, every entry is read again at its pin (the pinned digest, not the tag).
        client = FakeClient()
        record_compose_images(discover_tasks(tmp_path), tmp_path, client=client, platform="linux/amd64")
        assert client.resolved == []
        client = FakeClient()
        record_compose_images(discover_tasks(tmp_path), tmp_path, client=client, platform="linux/arm64")
        assert client.resolved == [MAIN_IMAGE, SIDECAR_IMAGE, "library/redis@sha256:old"]

    def test_a_complete_file_means_no_registry_traffic(self, tmp_path):
        write_task(tmp_path, "ctr", compose=COMPOSE)
        tasks = discover_tasks(tmp_path)
        record_compose_images(tasks, tmp_path, client=FakeClient())
        client = FakeClient()
        record_compose_images(tasks, tmp_path, client=client)
        assert client.resolved == []

    def test_a_sidecar_with_an_image_and_an_unrecorded_build_is_skipped_per_task_not_resolved(self, tmp_path):
        write_task(tmp_path, "good", compose=COMPOSE)
        write_task(
            tmp_path,
            "bad",
            compose="services:\n  main: {}\n  api:\n    image: skillsbench-visual-stability-api\n    build: ./api\n",
        )
        client = FakeClient()
        skipped: dict[str, str] = {}
        record_compose_images(discover_tasks(tmp_path), tmp_path, client=client, skipped=skipped)
        # The task lands in the skipped map like the no-image case; its local-only name never reaches the registry.
        assert skipped == {
            "bad": "bad: Compose service 'api' has a build section and prepare recorded no build for it"
        }
        assert client.resolved == [MAIN_IMAGE, SIDECAR_IMAGE, "redis:7-alpine"]

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
        assert recorded["image"] == "library/redis@sha256:index-of-7-alpine"

    def test_a_record_from_an_older_gym_is_read_again_at_its_pinned_digest(self, tmp_path, registry):
        write_task(tmp_path, "overlay", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n")
        # Written before Env (and record_version) were recorded; pinned to the manifest behind the tag then.
        legacy = {
            "redis:7-alpine": {
                "architecture": "amd64",
                "config": {"WorkingDir": "/app"},
                "config_digest": "sha256:cfg-amd64",
                "image": "library/redis@sha256:m-amd64",
                "os": "linux",
            }
        }
        (tmp_path / "compose-images.json").write_text(json.dumps(legacy))
        record_base_images(tmp_path)
        recorded = json.loads((tmp_path / "compose-images.json").read_text())["redis:7-alpine"]
        # The fields the resolver reads are now there, and the pin did not move: the digest was asked for, not the tag.
        assert recorded["record_version"] == 2 and recorded["config"]["Env"] == ["A=1"]
        assert recorded["image"] == "library/redis@sha256:m-amd64"
        manifests = [url for url, _ in registry.requests if "/manifests/" in url]
        assert manifests and all(url.endswith("/manifests/sha256:m-amd64") for url in manifests)
        requests = len(registry.requests)
        record_base_images(tmp_path)
        assert len(registry.requests) == requests

    AMD64_RECORD = {
        "architecture": "amd64",
        "config": {
            "Cmd": ["python", "server.py"],
            "WorkingDir": "/app",
            "ExposedPorts": {"5000/tcp": {}},
            "Env": ["A=1"],
        },
        "config_digest": "sha256:cfg-amd64",
        "image": "library/redis@sha256:index-of-7-alpine",
        "os": "linux",
        "record_version": 2,
    }

    def test_a_record_of_another_platform_is_read_again_at_its_pin_for_the_requested_platform(
        self, tmp_path, registry
    ):
        """Review (M24): ``compose-images.json`` holds a current-version amd64 record, and the dataset is prepared
        again with ``builds.platform: linux/arm64``. The record used to be kept by ``record_version`` alone, so the
        registry was never asked and ``resolve_compose`` refused the amd64 record at every seed with "prepare the
        dataset for that platform", which the user had just done. A record of another platform is now read again at
        its pin, the multi-arch index: the pin does not move and the configuration is the arm64 manifest's."""
        write_task(tmp_path, "overlay", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n")
        (tmp_path / "compose-images.json").write_text(json.dumps({"redis:7-alpine": self.AMD64_RECORD}))

        record_base_images(tmp_path, platform="linux/arm64")

        recorded = json.loads((tmp_path / "compose-images.json").read_text())["redis:7-alpine"]
        assert recorded["architecture"] == "arm64" and recorded["record_version"] == 2
        assert recorded["config_digest"] == "sha256:cfg-arm64" and recorded["config"] == {
            "Cmd": ["python", "server.py"]
        }
        assert recorded["image"] == "library/redis@sha256:index-of-7-alpine", "the pin never moves"
        manifests = [url for url, _ in registry.requests if "/manifests/" in url]
        assert manifests[0].endswith("/manifests/sha256:index-of-7-alpine"), "read at the pin, not the tag"
        assert all(
            url.endswith(("/manifests/sha256:index-of-7-alpine", "/manifests/sha256:m-arm64")) for url in manifests
        )
        # Recorded for arm64 now: the next arm64 prepare asks nothing; an amd64 prepare reads the pin back for amd64.
        requests = len(registry.requests)
        record_base_images(tmp_path, platform="linux/arm64")
        assert len(registry.requests) == requests
        record_base_images(tmp_path)
        recorded = json.loads((tmp_path / "compose-images.json").read_text())["redis:7-alpine"]
        assert recorded["architecture"] == "amd64" and recorded["config_digest"] == "sha256:cfg-amd64"
        assert recorded["image"] == "library/redis@sha256:index-of-7-alpine"

    def test_a_pin_to_a_single_manifest_of_another_platform_fails_the_prepare_with_the_remedy(
        self, tmp_path, registry
    ):
        """A pin that is one amd64 manifest (the reference was a digest, or an older record pinned behind the tag)
        has no arm64 manifest behind it: preparing for arm64 fails naming the image, the tasks and what to do, instead
        of keeping a record every arm64 seed refuses. The file is left as it was."""
        write_task(tmp_path, "overlay", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n")
        pinned = {"redis:7-alpine": {**self.AMD64_RECORD, "image": "library/redis@sha256:m-amd64"}}
        (tmp_path / "compose-images.json").write_text(json.dumps(pinned))
        with pytest.raises(ImageConfigError, match="resolved to linux/amd64, not linux/arm64") as info:
            record_base_images(tmp_path, platform="linux/arm64")
        message = str(info.value)
        assert "Could not record the configuration of base image 'redis:7-alpine' (used by task(s) overlay)" in message
        assert message.endswith(
            "library/redis@sha256:m-amd64: resolved to linux/amd64, not linux/arm64; pin a multi-arch tag or an image "
            "of linux/arm64, or prepare this dataset for linux/amd64"
        )
        assert json.loads((tmp_path / "compose-images.json").read_text()) == pinned

    def test_a_single_task_folder_records_into_its_parent(self, tmp_path, registry):
        task = write_task(tmp_path, "one", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n")
        assert record_base_images(task) == tmp_path / "compose-images.json"

    def test_a_task_on_a_recorded_base_starts_the_pin_and_a_compose_main_is_resolved_by_it(self, tmp_path, registry):
        """Prepare's order: base images are recorded, the tasks load pinned to them, then the Compose images
        (``main`` is the task image) are recorded. The record stays keyed by the reference the Dockerfile
        wrote, and ``main`` resolves by the pinned reference the task now carries."""
        from nemo_gym.sandbox.compose_config import resolve_compose

        compose = "services:\n  main:\n  cache:\n    image: redis:7-alpine\n"
        write_task(tmp_path, "t", image=None, dockerfile="FROM redis:7-alpine\nRUN true\n", compose=compose)
        record_base_images(tmp_path)
        (task,) = discover_tasks(tmp_path)
        assert task.image == "library/redis@sha256:index-of-7-alpine"
        assert compose_image_references(task) == [task.image, "redis:7-alpine"]
        record_compose_images([task], tmp_path)
        recorded = json.loads((tmp_path / "compose-images.json").read_text())
        assert recorded["redis:7-alpine"]["image"] == task.image, "looked up by the written reference"
        assert recorded[task.image]["image"] == task.image, "and by the pin, as the server asks for main"
        document = resolve_compose(
            {"services": {"main": None, "cache": {"image": "redis:7-alpine"}}}, task.image, recorded
        )
        assert document["services"]["main"]["image"] == document["services"]["cache"]["image"] == task.image

    def test_nothing_is_written_without_dockerfiles_to_resolve(self, tmp_path):
        write_task(tmp_path, "plain")
        assert record_base_images(tmp_path, client=FakeClient()) is None
        assert not (tmp_path / "compose-images.json").exists()

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
        # Of this run's platform (an entry of another platform would be read again at its pin, see
        # TestRecordBaseImages), told apart from the FakeClient's records by its config digest.
        competitor = {
            "architecture": "amd64",
            "config": {},
            "config_digest": "sha256:other",
            "os": "linux",
            "record_version": 2,
        }

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


FIELD_FIXTURES = Path(__file__).parent / "fixtures" / "harbor_compose_fields"


class TestComposeFields:
    """M28: service fields get their meaning at prepare, and one the group cannot run fails naming the field."""

    @pytest.mark.parametrize("name", ["ctf", "o11y"])
    def test_the_example_datasets_prepare_with_nothing_else(self, tmp_path, name):
        compose = (FIELD_FIXTURES / name / "environment" / "docker-compose.yaml").read_text()
        write_task(tmp_path, name, compose=compose)
        client = FakeClient()
        path = record_compose_images(discover_tasks(tmp_path), tmp_path, client=client, sandbox_provider="opensandbox")
        assert path == tmp_path / "compose-images.json"
        assert MAIN_IMAGE in client.resolved

    def test_the_document_the_server_reads_carries_the_normalized_fields(self, tmp_path):
        compose = (FIELD_FIXTURES / "ctf" / "environment" / "docker-compose.yaml").read_text()
        write_task(tmp_path, "ctf", compose=compose)
        (task,) = discover_tasks(tmp_path)
        document = check_task_compose_fields(task, load_compose_document(task), provider="opensandbox")
        assert document["services"]["main"]["restart"] == "always"
        assert document["services"]["main"]["networks"] == {"agent_network": {}}
        assert document["services"]["target"]["networks"] == {"agent_network": {"aliases": ["challenge"]}}

    def test_a_provider_option_the_deployment_cannot_apply_fails_prepare_naming_field_and_provider(self, tmp_path):
        compose = f"services:\n  main:\n    image: {MAIN_IMAGE}\n  db:\n    image: redis:7\n    sysctls:\n      net.core.somaxconn: 1024\n"
        write_task(tmp_path, "ctr", compose=compose)
        tasks = discover_tasks(tmp_path)
        with pytest.raises(
            ImageConfigError,
            match="ctr: environment/docker-compose.yaml: service 'db' sysctls is not supported by the opensandbox sandbox provider",
        ):
            record_compose_images(tasks, tmp_path, client=FakeClient(), sandbox_provider="opensandbox")
        assert not (tmp_path / "compose-images.json").exists()
        # Docker applies it as a run flag, so the same task prepares for a Docker deployment.
        assert record_compose_images(tasks, tmp_path, client=FakeClient(), sandbox_provider="docker") is not None
        # Without a provider name the check waits for the adapter, which has the live provider.
        assert record_compose_images(tasks, tmp_path, client=FakeClient()) is not None

    @pytest.mark.parametrize(
        "snippet, message",
        [
            (
                "    platform: linux/arm64\n",
                "service 'db' platform 'linux/arm64' does not match the deployment platform linux/amd64",
            ),
            ("    restart: maybe\n", "service 'db' restart must be one of"),
            ("    networks: [backend]\n", "service 'db' networks: 'backend' is not declared"),
            (
                "    deploy:\n      resources:\n        limits:\n          pids: 10\n",
                "service 'db' deploy.resources.limits.pids is not supported",
            ),
            (
                "    extra_hosts: ['host.docker.internal:host-gateway']\n",
                "service 'db' extra_hosts: host-gateway is not available",
            ),
            (
                "    post_start:\n      - command: id\n        privileged: true\n",
                "service 'db' post_start\\[0\\].privileged is not supported",
            ),
        ],
    )
    def test_a_field_the_group_cannot_run_fails_prepare_naming_it(self, tmp_path, snippet, message):
        compose = f"services:\n  main:\n    image: {MAIN_IMAGE}\n  db:\n    image: redis:7\n{snippet}"
        write_task(tmp_path, "ctr", compose=compose)
        with pytest.raises(ImageConfigError, match="ctr: environment/docker-compose.yaml: " + message):
            record_compose_images(discover_tasks(tmp_path), tmp_path, client=FakeClient(), sandbox_provider="docker")

    def test_a_gpu_sidecar_is_checked_against_the_gpu_block_provider(self, tmp_path):
        """The server runs a sidecar that reserves GPUs on the gpu_sandbox_provider block, so prepare checks its
        options against that block's provider; a task that declares GPUs itself runs everything there."""
        sidecar = (
            f"services:\n  main:\n    image: {MAIN_IMAGE}\n  model:\n    image: vllm/vllm-openai:v0.6\n"
            "    tmpfs: [/run]\n    deploy:\n      resources:\n        reservations:\n          devices:\n"
            "            - capabilities: [gpu]\n              count: 1\n"
        )
        write_task(tmp_path, "sidecar", compose=sidecar)
        tasks = discover_tasks(tmp_path)
        # Docker task block, OpenSandbox GPU block: the GPU sidecar lands on OpenSandbox, which applies no tmpfs.
        with pytest.raises(
            ImageConfigError,
            match="sidecar: environment/docker-compose.yaml: service 'model' tmpfs is not supported by the opensandbox sandbox provider",
        ):
            record_compose_images(
                tasks, tmp_path, client=FakeClient(), sandbox_provider="docker", gpu_sandbox_provider="opensandbox"
            )
        # The reverse pairing runs it: main on OpenSandbox declares nothing, the sidecar on Docker takes tmpfs.
        assert (
            record_compose_images(
                tasks, tmp_path, client=FakeClient(), sandbox_provider="opensandbox", gpu_sandbox_provider="docker"
            )
            is not None
        )
        # task.toml declares GPUs: its own services run on the GPU block too, main included.
        gpu_task = tmp_path / "gputask"
        write_task(tmp_path, "gputask", compose=f"services:\n  main:\n    image: {MAIN_IMAGE}\n    tmpfs: [/run]\n")
        with (gpu_task / "task.toml").open("a") as handle:
            handle.write("gpus = 1\n")
        (task,) = [task for task in discover_tasks(tmp_path) if task.task_id == "gputask"]
        with pytest.raises(
            ImageConfigError, match="service 'main' tmpfs is not supported by the opensandbox sandbox provider"
        ):
            check_task_compose_fields(task, load_compose_document(task), provider="docker", gpu_provider="opensandbox")
        assert check_task_compose_fields(task, load_compose_document(task), provider="docker")["services"]["main"][
            "tmpfs"
        ]

    def test_a_cpu_sidecar_with_legacy_memory_prepares_with_a_gpu_block_configured(self, tmp_path):
        """Review: with a GPU block configured prepare asks every sidecar for its GPU count; that read must not size
        the service, since the legacy `mem_limit: 512M` is only converted to bytes by resolve_compose later. The
        string used to reach int() and the ValueError, not a field error, escaped the per-task skip and aborted
        the whole preparation."""
        compose = (
            f"services:\n  main:\n    image: {MAIN_IMAGE}\n  db:\n    image: redis:7\n    mem_limit: 512M\n"
            "  model:\n    image: vllm/vllm-openai:v0.6\n    deploy:\n      resources:\n        reservations:\n"
            "          devices:\n            - capabilities: [gpu]\n              count: 1\n"
        )
        write_task(tmp_path, "mixed", compose=compose)
        (task,) = discover_tasks(tmp_path)
        skipped: dict[str, str] = {}
        path = record_compose_images(
            [task],
            tmp_path,
            client=FakeClient(),
            sandbox_provider="docker",
            gpu_sandbox_provider="opensandbox",
            skipped=skipped,
        )
        assert path is not None and skipped == {}
        assert {"redis:7", "vllm/vllm-openai:v0.6"} <= set(json.loads(path.read_text()))
        # The same check at seed: the CPU sidecar keeps its string, the GPU sidecar still counts.
        document = check_task_compose_fields(
            task, load_compose_document(task), provider="docker", gpu_provider="opensandbox"
        )
        assert document["services"]["db"]["mem_limit"] == "512M"
        assert (service_gpus(document["services"]["db"]), service_gpus(document["services"]["model"])) == (0, 1)

    def test_extra_hosts_on_a_non_root_sidecar_fail_prepare_once_the_image_user_is_known(self, tmp_path):
        """The group starts a non-root sidecar without host injection, so its extra_hosts would be lost; prepare
        learns the user from the recorded image configuration and fails there, naming task, file and service."""

        class NonRootClient(FakeClient):
            def image_config(self, reference):
                record = super().image_config(reference)
                if reference.startswith("grafana/"):
                    record["config"] = {**record["config"], "User": "472"}
                return record

        compose = (
            f"services:\n  main:\n    image: {MAIN_IMAGE}\n    extra_hosts: ['metrics.example:10.0.0.9']\n"
            "  grafana:\n    image: grafana/grafana:11\n    extra_hosts: ['metrics.example:10.0.0.9']\n"
        )
        write_task(tmp_path, "o11y", compose=compose)
        with pytest.raises(
            ImageConfigError,
            match="o11y: environment/docker-compose.yaml: service 'grafana' extra_hosts cannot be applied without "
            "host injection: the image runs as user '472'",
        ):
            record_compose_images(
                discover_tasks(tmp_path), tmp_path, client=NonRootClient(), sandbox_provider="docker"
            )
        # The same task with a root grafana image prepares; main takes host entries whatever its user.
        root = tmp_path / "root"
        root.mkdir()
        write_task(root, "o11y", compose=compose)
        assert record_compose_images(discover_tasks(root), root, client=FakeClient(), sandbox_provider="docker")

    def test_platform_is_the_deployments_at_prepare_as_at_seed(self, tmp_path):
        compose = (
            f"services:\n  main:\n    image: {MAIN_IMAGE}\n  db:\n    image: redis:7\n    platform: linux/arm64\n"
        )
        write_task(tmp_path, "arm", compose=compose)
        tasks = discover_tasks(tmp_path)
        with pytest.raises(ImageConfigError, match="does not match the deployment platform linux/amd64"):
            record_compose_images(tasks, tmp_path, client=FakeClient())
        # The deployment's platform (sandbox_provider_options.platform, as the server reads it) admits the task.
        assert (
            record_compose_images(tasks, tmp_path, client=FakeClient(), compose_platform="linux/aarch64") is not None
        )
        with pytest.raises(
            ImageConfigError,
            match="arm: environment/docker-compose.yaml: service 'db' platform 'linux/arm64' does not match the "
            "deployment platform linux/amd64",
        ):
            check_task_compose_fields(tasks[0], load_compose_document(tasks[0]), platform="linux/x86_64")
