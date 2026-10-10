# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the OCI configuration of every image a Compose task starts.

Docker Compose reads a service's entrypoint, command, user, working directory and
exposed ports from the image itself. A sandbox platform without Docker cannot, so
the harbor resources server reads them from ``compose-images.json`` next to the task
folders (see ``nemo_gym.sandbox.compose_config.resolve_compose``). This module writes
that file from the registry, one entry per image reference exactly as the task wrote
it, pinned to the linux/amd64 manifest digest it resolved to.
"""

import base64
import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemo_gym.tasks.harbor.task import HarborTask


LOGGER = logging.getLogger(__name__)

COMPOSE_IMAGES_FILE = "compose-images.json"
COMPOSE_FILE_NAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
# The startup fields Compose would read from the image; the rest of the config is not recorded.
RECORDED_CONFIG_KEYS = ("Cmd", "Entrypoint", "ExposedPorts", "Healthcheck", "User", "WorkingDir")
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
DOCKER_HUB_REGISTRY = "registry-1.docker.io"
PLATFORM = ("linux", "amd64")


class ImageConfigError(RuntimeError):
    """An image reference could not be resolved to a linux/amd64 configuration."""


@dataclass(frozen=True)
class ImageRef:
    """A parsed image reference: ``[registry/]repository[:tag][@digest]``."""

    registry: str
    repository: str
    tag: str | None
    digest: str | None

    @property
    def manifest_ref(self) -> str:
        return self.digest or self.tag or "latest"

    @property
    def pinned_repository(self) -> str:
        """The repository a digest pin is written against: as the registry names it, host included off Docker Hub."""
        if self.registry != DOCKER_HUB_REGISTRY:
            return f"{self.registry}/{self.repository}"
        return self.repository


def parse_image_ref(reference: str) -> ImageRef:
    """Split an image reference the way Docker does, defaulting to Docker Hub."""
    rest, _, digest = reference.partition("@")
    first, slash, remainder = rest.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        registry, path = first, remainder
    else:
        registry, path = DOCKER_HUB_REGISTRY, rest
    if registry == "docker.io":
        registry = DOCKER_HUB_REGISTRY
    repository, _, tag = path.partition(":")
    if registry == DOCKER_HUB_REGISTRY and "/" not in repository:
        repository = f"library/{repository}"
    if not repository:
        raise ImageConfigError(f"Not an image reference: {reference!r}")
    return ImageRef(registry=registry, repository=repository, tag=tag or None, digest=digest or None)


class RegistryClient:
    """Read manifests and configuration blobs over the OCI distribution API with the standard library.

    ``credentials`` maps a registry host to ``(username, password)`` for private repositories;
    public ones use the anonymous token the registry hands out.
    """

    def __init__(
        self, credentials: dict[str, tuple[str, str]] | None = None, *, timeout_s: float = 60, retries: int = 3
    ) -> None:
        self.credentials = credentials or {}
        self.timeout_s = timeout_s
        self.retries = retries
        self._tokens: dict[tuple[str, str], str] = {}

    def image_config(self, reference: str) -> dict[str, Any]:
        """One ``compose-images.json`` entry for ``reference``."""
        ref = parse_image_ref(reference)
        manifest, manifest_digest = self._manifest(ref, ref.manifest_ref)
        if manifest.get("mediaType") in INDEX_MEDIA_TYPES or "manifests" in manifest:
            manifest_digest = self._select_platform(ref, manifest)
            manifest, _ = self._manifest(ref, manifest_digest)
        config_digest = manifest["config"]["digest"]
        blob = json.loads(self._get(ref, f"blobs/{config_digest}", accept="*/*")[0])
        if (blob.get("os"), blob.get("architecture")) != PLATFORM:
            raise ImageConfigError(
                f"{reference}: resolved to {blob.get('os')}/{blob.get('architecture')}, not linux/amd64"
            )
        config = blob.get("config") or {}
        return {
            "architecture": blob["architecture"],
            "config": {key: config[key] for key in RECORDED_CONFIG_KEYS if config.get(key)},
            "config_digest": config_digest,
            "image": f"{ref.pinned_repository}@{manifest_digest}",
            "os": blob["os"],
        }

    def _select_platform(self, ref: ImageRef, index: dict[str, Any]) -> str:
        for entry in index.get("manifests", []):
            platform = entry.get("platform") or {}
            if (entry.get("annotations") or {}).get("vnd.docker.reference.type") == "attestation-manifest":
                continue
            if (platform.get("os"), platform.get("architecture")) == PLATFORM and not platform.get("variant"):
                return entry["digest"]
        raise ImageConfigError(f"{ref.pinned_repository}: the image index has no linux/amd64 manifest")

    def _manifest(self, ref: ImageRef, manifest_ref: str) -> tuple[dict[str, Any], str]:
        body, headers = self._get(ref, f"manifests/{manifest_ref}", accept=MANIFEST_ACCEPT)
        digest = headers.get("Docker-Content-Digest") or "sha256:" + hashlib.sha256(body).hexdigest()
        return json.loads(body), digest

    def _get(self, ref: ImageRef, path: str, *, accept: str) -> tuple[bytes, dict[str, str]]:
        url = f"https://{ref.registry}/v2/{ref.repository}/{path}"
        headers = {"Accept": accept}
        token = self._tokens.get((ref.registry, ref.repository))
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            return self._open(url, headers)
        except urllib.error.HTTPError as exc:
            challenge = exc.headers.get("WWW-Authenticate", "") if exc.code == 401 else ""
            if not challenge:
                raise ImageConfigError(
                    f"{ref.pinned_repository}: registry returned HTTP {exc.code} for {path}"
                ) from exc
        token = self._token(ref, challenge)
        self._tokens[(ref.registry, ref.repository)] = token
        headers["Authorization"] = f"Bearer {token}"
        try:
            return self._open(url, headers)
        except urllib.error.HTTPError as exc:
            raise ImageConfigError(f"{ref.pinned_repository}: registry returned HTTP {exc.code} for {path}") from exc

    def _token(self, ref: ImageRef, challenge: str) -> str:
        params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        realm = params.get("realm")
        if not challenge.lower().startswith("bearer") or not realm:
            raise ImageConfigError(f"{ref.pinned_repository}: unsupported registry challenge {challenge!r}")
        query = {key: params[key] for key in ("service", "scope") if key in params}
        query.setdefault("scope", f"repository:{ref.repository}:pull")
        headers = {}
        if ref.registry in self.credentials:
            username, password = self.credentials[ref.registry]
            headers["Authorization"] = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        try:
            body, _ = self._open(f"{realm}?{urllib.parse.urlencode(query)}", headers)
        except urllib.error.HTTPError as exc:
            raise ImageConfigError(f"{ref.pinned_repository}: token request failed with HTTP {exc.code}") from exc
        payload = json.loads(body)
        token = payload.get("token") or payload.get("access_token")
        if not token:
            raise ImageConfigError(f"{ref.pinned_repository}: token response carried no token")
        return token

    def _open(self, url: str, headers: dict[str, str]) -> tuple[bytes, dict[str, str]]:
        """One GET; connection-level failures (not HTTP errors) are retried with a short backoff."""
        request = urllib.request.Request(url, headers=headers)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return response.read(), dict(response.headers)
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.retries:
                    raise ImageConfigError(
                        f"registry unreachable after {attempt + 1} attempts: {url} ({exc})"
                    ) from exc
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def compose_file(task: HarborTask) -> Path | None:
    """The task's Compose overlay, if it has one."""
    for name in COMPOSE_FILE_NAMES:
        path = task.path / "environment" / name
        if path.is_file():
            return path
    return None


def compose_image_references(task: HarborTask) -> list[str]:
    """Every image a Compose task starts: the task image for ``main`` and each sidecar's image."""
    path = compose_file(task)
    if path is None:
        return []
    document = yaml.safe_load(path.read_text()) or {}
    services = document.get("services") or {}
    references: list[str] = []
    main_image = (services.get("main") or {}).get("image") or task.image
    if main_image:
        references.append(main_image)
    for name, service in services.items():
        if name == "main":
            continue
        image = (service or {}).get("image")
        if not image:
            raise ImageConfigError(
                f"{task.task_id}: Compose service {name!r} names no image; builds are not supported"
            )
        references.append(image)
    return list(dict.fromkeys(references))


def record_compose_images(
    tasks: list[HarborTask], folder: Path, *, client: RegistryClient | None = None
) -> Path | None:
    """Write ``compose-images.json`` in ``folder`` for the Compose tasks among ``tasks``.

    Entries already recorded are kept as they are, so a pinned recording never changes
    under a completed run; only references missing from the file are resolved. Returns
    the file's path, or ``None`` when no task uses Compose.
    """
    references = list(dict.fromkeys(ref for task in tasks for ref in compose_image_references(task)))
    if not references:
        return None
    path = Path(folder) / COMPOSE_IMAGES_FILE
    recorded: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    missing = [ref for ref in references if ref not in recorded]
    if not missing:
        return path
    client = client or RegistryClient()
    LOGGER.info(f"Recording the configuration of {len(missing)} Compose image(s) into {path}")
    for reference in missing:
        recorded[reference] = client.image_config(reference)
    path.write_text(json.dumps(dict(sorted(recorded.items())), indent=2, sort_keys=True) + "\n")
    return path
