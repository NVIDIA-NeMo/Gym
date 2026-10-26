# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an image's configuration from its OCI registry.

Parses Docker-style references with Docker Hub defaults, selects the platform's manifest from
an index, follows the bearer token challenge, drops authorization across redirects, retries
transient failures and returns the configuration keys the records keep, pinned to the digest
of the manifest the reference named. Nothing here knows about tasks or Compose. The module
keeps the record key list, the platform constants and ImageConfigError, which the rest of the
package imports from here.
"""

import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION


# The fields Compose would read from the image to start it, plus the build-time state (environment,
# user, working directory, the shell that runs RUN lines) a Dockerfile's instructions resolve from; the
# rest of the config is not recorded. Growing this tuple means bumping ``dockerfile.RECORD_VERSION``.
RECORDED_CONFIG_KEYS = ("Cmd", "Entrypoint", "Env", "ExposedPorts", "Healthcheck", "Shell", "User", "WorkingDir")
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

# containerd's platform normalisation (``platforms.Normalize``), the rule ``docker --platform`` matches an index
# entry by: Docker's architecture aliases fold to GOARCH names, an arm64 entry with no variant is ``v8`` (the
# variant Docker Official Images declare: ubuntu, redis and python all publish ``arm64/v8``), an arm entry with
# none is ``v7``, and amd64's baseline micro-architecture level ``v1`` is plain amd64. The higher levels ``v2``..``v4``
# are other platforms: each needs CPU features (SSE4/POPCNT, AVX2, AVX-512) a baseline host may lack, so containerd's
# matcher answers a ``linux/amd64`` request with a ``v1`` or variant-less entry only (``platforms.Only``: a request for
# ``amd64/v3`` also accepts v2, v1 and baseline, never the reverse).
_ARCHITECTURES = {"x86_64": "amd64", "x86-64": "amd64", "aarch64": "arm64", "armhf": "arm", "armel": "arm"}
_DEFAULT_VARIANTS = {"arm64": "v8", "arm": "v7"}
_AMD64_BASELINE = "v1"


def normalize_platform_parts(os: Any, architecture: Any, variant: Any = "") -> tuple[str, str, str]:
    """``(os, architecture, variant)`` as containerd spells them, so that two spellings of one platform compare
    equal: ``linux/arm64``, ``linux/aarch64`` and ``linux/arm64/v8`` are all ``("linux", "arm64", "v8")``, and
    ``linux/amd64/v1`` is ``("linux", "amd64", "")``; ``linux/amd64/v3`` keeps its ``v3``, another platform. A
    missing part is an empty string."""
    os, architecture, variant = (str(part or "").strip().lower() for part in (os, architecture, variant))
    architecture = _ARCHITECTURES.get(architecture, architecture)
    if architecture == "amd64" and variant == _AMD64_BASELINE:
        variant = ""
    return os, architecture, variant or _DEFAULT_VARIANTS.get(architecture, "")


def platform_name(platform: tuple[str, ...]) -> str:
    """``os/arch[/variant]`` of a (normalised) platform tuple, the variant shown only when it is not the
    architecture's default: ``linux/amd64``, ``linux/arm64``, ``linux/arm/v6``."""
    os, architecture, *rest = platform
    variant = rest[0] if rest else ""
    if variant == _DEFAULT_VARIANTS.get(architecture, ""):
        variant = ""
    return "/".join(part for part in (os, architecture, variant) if part)


def platform_pair(platform: str | None) -> tuple[str, str, str]:
    """The normalised ``(os, architecture, variant)`` of an ``os/arch[/variant]`` platform string such as
    ``builds.platform`` (:func:`normalize_platform_parts`: ``linux/arm64`` and ``linux/arm64/v8`` are one value);
    ``None`` (or an empty string) is linux/amd64, the default platform sandboxes run. An amd64 variant other than
    the baseline ``v1`` is refused: the records keep an image's os and architecture only (``reusable_record``,
    ``recorded_image_mismatch``), so a configuration read for ``linux/amd64/v3`` would serve a later baseline run."""
    parts = [part for part in (platform or "").lower().split("/") if part]
    if not parts:
        return normalize_platform_parts(*PLATFORM)
    if len(parts) < 2 or len(parts) > 3:
        raise ImageConfigError(f"Not a platform: {platform!r} (expected os/arch, for example linux/arm64)")
    normalized = normalize_platform_parts(*parts)
    if normalized[1] == "amd64" and normalized[2]:
        raise ImageConfigError(
            f"Unsupported platform: {platform!r} (an amd64 micro-architecture level above the baseline is another "
            "platform to docker, and images are recorded by os/arch only; use linux/amd64)"
        )
    return normalized


class ImageConfigError(RuntimeError):
    """An image reference could not be resolved to a configuration for the platform the sandboxes run."""


class PlatformMismatchError(ImageConfigError):
    """A single-platform reference resolved to an image of another platform than the one asked for; ``resolved``
    and ``requested`` name the two as ``os/arch[/variant]``."""

    def __init__(self, message: str, *, resolved: str, requested: str) -> None:
        super().__init__(message)
        self.resolved = resolved
        self.requested = requested


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


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow a registry redirect without carrying the registry's token to another host.

    Registries answer a blob request with a redirect to a storage host (Docker Hub, GHCR and ECR all
    do). The OCI distribution spec says a client must not send its ``Authorization`` there, and the
    default handler forwards every header, so the header is dropped whenever the host changes.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            redirected is not None
            and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


class RegistryClient:
    """Read manifests and configuration blobs over the OCI distribution API with the standard library.

    ``credentials`` maps a registry host to ``(username, password)`` for private repositories;
    public ones use the anonymous token the registry hands out. ``platform`` is the ``(os, architecture)`` or
    ``(os, architecture, variant)`` an image reference is resolved for (linux/amd64 by default), kept normalised
    (:func:`normalize_platform_parts`): the manifest an index names for it, matched as ``docker --platform``
    matches one, and the only one a single-platform reference may be.
    """

    def __init__(
        self,
        credentials: dict[str, tuple[str, str]] | None = None,
        *,
        timeout_s: float = 60,
        retries: int = 3,
        platform: tuple[str, ...] = PLATFORM,
    ) -> None:
        self.credentials = credentials or {}
        self.timeout_s = timeout_s
        self.retries = retries
        self.platform = normalize_platform_parts(*platform)
        self._tokens: dict[tuple[str, str], str] = {}
        self._opener = urllib.request.build_opener(_RedirectHandler())

    def image_config(self, reference: str) -> dict[str, Any]:
        """One ``compose-images.json`` entry for ``reference``."""
        ref = parse_image_ref(reference)
        manifest, manifest_digest = self._manifest(ref, ref.manifest_ref)
        # The pin is the digest of the manifest the reference named. Behind a tag that is the multi-arch
        # index itself, so a sandbox of any platform selects its own manifest from one immutable
        # reference; the configuration below is read from the manifest the index names for the recorded
        # platform. A single-platform reference pins to its one manifest.
        pinned_digest = manifest_digest
        if manifest.get("mediaType") in INDEX_MEDIA_TYPES or "manifests" in manifest:
            manifest_digest = self._select_platform(ref, manifest)
            manifest, _ = self._manifest(ref, manifest_digest)
        config_digest = manifest["config"]["digest"]
        blob = json.loads(self._get(ref, f"blobs/{config_digest}", accept="*/*")[0])
        # os and architecture only: a configuration blob may omit the variant its index entry names.
        if normalize_platform_parts(blob.get("os"), blob.get("architecture"))[:2] != self.platform[:2]:
            resolved = f"{blob.get('os')}/{blob.get('architecture')}"
            raise PlatformMismatchError(
                f"{reference}: resolved to {resolved}, not {platform_name(self.platform)}",
                resolved=resolved,
                requested=platform_name(self.platform),
            )
        config = blob.get("config") or {}
        return {
            "architecture": blob["architecture"],
            "config": {key: config[key] for key in RECORDED_CONFIG_KEYS if config.get(key)},
            "config_digest": config_digest,
            "image": f"{ref.pinned_repository}@{pinned_digest}",
            "os": blob["os"],
            "record_version": RECORD_VERSION,
        }

    def _select_platform(self, ref: ImageRef, index: dict[str, Any]) -> str:
        """The digest of the first manifest of ``index`` for :attr:`platform`, matched as containerd matches one:
        an arm64 entry with ``variant: v8`` (the Docker Official Images' shape) or without one both answer
        ``linux/arm64``, and an amd64 entry with ``variant: v1`` or none answers ``linux/amd64``, while one at a
        higher level (``v3``) does not: it is another platform, selected only when asked for by variant."""
        for entry in index.get("manifests", []):
            platform = entry.get("platform") or {}
            if (entry.get("annotations") or {}).get("vnd.docker.reference.type") == "attestation-manifest":
                continue
            candidate = (platform.get("os"), platform.get("architecture"), platform.get("variant"))
            if normalize_platform_parts(*candidate) == self.platform:
                return entry["digest"]
        raise ImageConfigError(
            f"{ref.pinned_repository}: the image index has no {platform_name(self.platform)} manifest"
        )

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
                with self._opener.open(request, timeout=self.timeout_s) as response:
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
