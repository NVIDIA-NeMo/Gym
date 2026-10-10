# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nemo_gym.tasks.harbor import image_configs
from nemo_gym.tasks.harbor.image_configs import (
    ImageConfigError,
    RegistryClient,
    compose_image_references,
    parse_image_ref,
    record_compose_images,
)
from nemo_gym.tasks.harbor.task import discover_tasks


MAIN_IMAGE = "harborframework/terminal-bench:ctr-environment-abc@sha256:" + "a" * 64
SIDECAR_IMAGE = "harborframework/terminal-bench:ctr-sidecar-api-def@sha256:" + "b" * 64


def write_task(root: Path, name: str, *, image: str = MAIN_IMAGE, compose: str | None = None) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(f'schema_version = "1.4"\n\n[environment]\ndocker_image = "{image}"\n')
    (task / "instruction.md").write_text("Do the thing.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    if compose is not None:
        (task / "environment" / "docker-compose.yaml").write_text(compose)
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
            "config": {"Cmd": ["python", "server.py"], "ExposedPorts": {"5000/tcp": {}}, "WorkingDir": "/app"},
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
