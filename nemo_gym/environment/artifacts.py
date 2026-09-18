# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable Python environment bundles, distributed through an OCI registry with ORAS."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import yaml

from nemo_gym import CACHE_DIR, __version__
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.environment.validation import validate_environment
from nemo_gym.registry import EnvironmentCatalogEntry


ARTIFACT_TYPE = "application/vnd.nvidia.nemo-gym.environment.v1"
ARTIFACT_ANNOTATION = "ai.nvidia.nemo-gym.artifact-type"
LAYER_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"
PACKAGE_METADATA = "gym-package.json"
MAX_PACKAGE_BYTES = 1024 * 1024 * 1024
MAX_PACKAGE_FILES = 10000
HUB_REGISTRY = "gitlab-master.nvidia.com:5005/cmunley/gym_environments_hub"
HUB_REPOSITORY = "ssh://git@gitlab-master.nvidia.com:12051/cmunley/gym_environments_hub.git"
HUB_URL = "https://gym-environments-hub-c2d724.gitlab-master-pages.nvidia.com/development.html"
ORAS_VERSION = "1.3.4"
# Official release checksums: https://github.com/oras-project/oras/releases/tag/v1.3.4
ORAS_CHECKSUMS = {
    "darwin_amd64": "5e964f3d5a36eb9499a9d3e252a86b09e7adf3e6f6447eec56fd249c6702af7e",
    "darwin_arm64": "217761a9500242ff473de8656b5aca21136ff39e17e9e61fd8936bbfd902704c",
    "linux_amd64": "f27adb935022d94df8dc77719c322dda592c78a0d57a6f7dcdd8d900b248c454",
    "linux_arm64": "15702c6e3a4a56a8bd8ac5c17efdbcab56d9bada661ccbcf017f5b10c1d89399",
}
_EXCLUDED = {"__pycache__", "cache", "results", "env.yaml", "node_modules"}


def _package_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ConfigError(f"Invalid package path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or not path.parts:
        raise ConfigError(f"Package paths must be normalized and relative: {value!r}")
    return path


def build_environment_package(entry: EnvironmentCatalogEntry, output: Path) -> Path:
    """Package only files explicitly selected in the workload's package.yaml."""
    if entry.manifest_path is None:
        raise ConfigError("An environment package requires manifest.yaml.")
    validate_environment(entry.manifest_path, entry.config_path)
    manifest = load_manifest(entry.manifest_path)
    workload_dir = entry.manifest_path.resolve().parent
    root = workload_dir.parent.parent
    if workload_dir.parent.name not in {"environments", "benchmarks"}:
        raise ConfigError("Package workloads must live under environments/NAME or benchmarks/NAME.")
    spec_path = workload_dir / "package.yaml"
    if not spec_path.is_file():
        raise ConfigError(f"Add {spec_path} with an explicit 'include' list of files or directories to publish.")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != {"include"} or not isinstance(spec["include"], list):
        raise ConfigError("package.yaml must contain only an 'include' list.")

    files: dict[str, bytes] = {}
    total_bytes = 0
    for include in [*spec["include"], str(spec_path.relative_to(root))]:
        relative = _package_path(include)
        source = root / relative
        if not source.exists() or source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ConfigError(f"Package include is missing or points outside its root: {include}")
        candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
        for candidate in candidates:
            name = candidate.relative_to(root).as_posix()
            if any(part.startswith(".") or part in _EXCLUDED for part in Path(name).parts):
                continue
            if candidate.is_symlink():
                raise ConfigError(f"Package includes cannot contain symbolic links: {name}")
            if candidate.is_dir():
                continue
            if not candidate.is_file() or candidate.suffix in {".pyc", ".pem", ".key"}:
                raise ConfigError(f"Unsupported package file: {name}")
            if name == PACKAGE_METADATA:
                raise ConfigError(f"{PACKAGE_METADATA} is reserved for generated release metadata.")
            if name in files:
                continue
            total_bytes += candidate.stat().st_size
            if total_bytes > MAX_PACKAGE_BYTES or len(files) >= MAX_PACKAGE_FILES:
                raise ConfigError("Package exceeds the 1 GiB / 10,000 file limit; reference bulk data separately.")
            files[name] = candidate.read_bytes()

    config_path = entry.config_path.resolve().relative_to(root).as_posix()
    manifest_path = entry.manifest_path.resolve().relative_to(root).as_posix()
    if config_path not in files or manifest_path not in files:
        raise ConfigError("package.yaml must include the environment's config and manifest.")
    metadata = {
        "schema_version": 1,
        "name": manifest.name,
        "version": manifest.version,
        "kind": manifest.kind.value,
        "gym_version": __version__,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    files[PACKAGE_METADATA] = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    output = Path(output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ConfigError(f"Package output already exists: {output}")
    # Fixed ordering, ownership, modes, and timestamps make equal contents produce the same digest.
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    return output


def _oras(*arguments: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    executable = shutil.which("oras")
    if executable is None:
        machine = platform.machine().lower()
        architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)
        target = f"{platform.system().lower()}_{architecture}"
        if target not in ORAS_CHECKSUMS:
            raise ConfigError(f"Automatic registry setup is unavailable for {target}; install ORAS 1.3+ on PATH.")
        cached = CACHE_DIR / "tools" / f"oras-{ORAS_VERSION}-{target}" / "oras"
        if not cached.is_file():
            url = f"https://github.com/oras-project/oras/releases/download/v{ORAS_VERSION}/oras_{ORAS_VERSION}_{target}.tar.gz"
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read(32 * 1024 * 1024)
                if hashlib.sha256(payload).hexdigest() != ORAS_CHECKSUMS[target]:
                    raise ConfigError("Registry helper download checksum mismatch; no executable was installed.")
                with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                    member = archive.getmember("oras")
                    if not member.isfile() or not 0 < member.size <= 32 * 1024 * 1024:
                        raise ConfigError("Registry helper archive does not contain a valid executable.")
                    binary = archive.extractfile(member).read()
                cached.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(prefix=".install-", dir=cached.parent) as temporary:
                    staged = Path(temporary) / "oras"
                    staged.write_bytes(binary)
                    staged.chmod(0o755)
                    staged.replace(cached)
            except (OSError, tarfile.TarError, KeyError) as error:
                raise ConfigError(f"Could not set up the registry helper automatically: {error}") from error
        executable = str(cached)
    try:
        result = subprocess.run(
            [executable, *arguments], cwd=cwd, capture_output=True, text=True, errors="replace", timeout=600
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ConfigError(f"Could not run ORAS: {error}") from error
    if check and result.returncode:
        if any(message in result.stderr.lower() for message in ("unauthorized", "authentication required", "denied")):
            raise ConfigError(
                "Registry access was denied. Check your registry credentials and package permissions, then retry. "
                "Gym uses the existing Docker/ORAS credential store."
            )
        raise ConfigError(f"ORAS failed: {result.stderr.strip()}")
    return result


def _oci_reference(reference: str) -> tuple[str, str]:
    reference = reference.removeprefix("oci://")
    if not re.fullmatch(r"[A-Za-z0-9._:/@+-]+", reference) or "/" not in reference:
        raise ConfigError(
            "Use a full OCI registry reference: registry/namespace/environment:VERSION or @sha256:DIGEST."
        )
    if "@" in reference:
        repository, digest = reference.rsplit("@", 1)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ConfigError("OCI digest references must use sha256 with 64 lowercase hex characters.")
    elif ":" in reference.rsplit("/", 1)[-1]:
        repository, tag = reference.rsplit(":", 1)
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
            raise ConfigError("OCI references require a nonempty, valid version tag.")
    else:
        raise ConfigError("An explicit version tag or digest is required; implicit 'latest' is unsupported.")
    host = repository.split("/", 1)[0]
    if "." not in host and ":" not in host and host != "localhost":
        if not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*/[a-z0-9]+(?:[._-][a-z0-9]+)*", repository):
            raise ConfigError("Use a namespaced environment reference: namespace/name:VERSION or @sha256:DIGEST.")
        registry = os.environ.get("GYM_ENV_REGISTRY", "").removeprefix("oci://").rstrip("/")
        if not registry:
            raise ConfigError("Set GYM_ENV_REGISTRY to your registry host/path, or use a full OCI registry reference.")
        if not re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]+)?(?:/[a-z0-9._-]+)*", registry):
            raise ConfigError("GYM_ENV_REGISTRY must be a registry host/path without a tag or digest.")
        reference, repository = f"{registry}/{reference}", f"{registry}/{repository}"
    return reference, repository


def _verify_package(root: Path) -> dict:
    try:
        metadata = json.loads((root / PACKAGE_METADATA).read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1 or not isinstance(metadata["files"], dict):
            raise ValueError("unsupported schema or missing file inventory")
        if metadata["gym_version"] != __version__:
            raise ValueError(f"requires nemo-gym=={metadata['gym_version']}; installed version is {__version__}")
        for name, expected in metadata["files"].items():
            path = root / _package_path(name)
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"file escapes package: {name}")
            if (
                not re.fullmatch(r"[0-9a-f]{64}", expected)
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"file checksum mismatch: {name}")
        for key in ("config_path", "manifest_path"):
            if metadata[key] not in metadata["files"]:
                raise ValueError(f"{key} is not a packaged file")
            _package_path(metadata[key])
        manifest = load_manifest(root / metadata["manifest_path"])
        if (manifest.name, manifest.version, manifest.kind.value) != (
            metadata["name"],
            metadata["version"],
            metadata["kind"],
        ):
            raise ValueError("release identity disagrees with environment manifest")
    except (KeyError, ValueError, TypeError, OSError) as error:
        raise ConfigError(f"Invalid environment package: {error}") from error
    return metadata


def push_environment_package(package: Path, reference: str) -> str:
    """Push a checked bundle, refusing existing tags, and return its immutable manifest reference."""
    reference, repository = _oci_reference(reference)
    if "@" in reference:
        raise ConfigError("Publish with an explicit version tag; use the returned digest for reproducible runs.")
    package = Path(package).resolve()
    with TemporaryDirectory(prefix="gym-publish-") as temporary:
        root = pull_environment_package(str(package), Path(temporary) / "check")
        metadata = _verify_package(root)
        if reference.rsplit(":", 1)[-1] != metadata["version"]:
            raise ConfigError(f"Registry tag must match manifest version {metadata['version']}.")
        existing = _oras("resolve", reference, check=False)
        if existing.returncode == 0:
            raise ConfigError(f"Release already exists: {reference}. Publish a new version instead.")
        if not re.search(r"not found|MANIFEST_UNKNOWN|NAME_UNKNOWN", existing.stderr, re.IGNORECASE):
            raise ConfigError(f"Could not check release availability: {existing.stderr.strip()}")
        upload_dir = Path(temporary) / "upload"
        upload_dir.mkdir()
        shutil.copyfile(package, upload_dir / "environment.tar.gz")
        (upload_dir / "config.json").write_text("{}", encoding="utf-8")
        # Older GitLab registries reject custom artifact/config media types. A standard
        # OCI 1.0 envelope retains the Gym type in an annotation and works on both versions.
        _oras(
            "push",
            "--image-spec",
            "v1.0",
            "--config",
            "config.json:application/vnd.oci.image.config.v1+json",
            "--annotation",
            f"{ARTIFACT_ANNOTATION}={ARTIFACT_TYPE}",
            "--annotation",
            f"org.opencontainers.image.title={metadata['name']}",
            "--annotation",
            f"org.opencontainers.image.version={metadata['version']}",
            "--export-manifest",
            "manifest.json",
            reference,
            f"environment.tar.gz:{LAYER_TYPE}",
            cwd=upload_dir,
        )
        digest = hashlib.sha256((upload_dir / "manifest.json").read_bytes()).hexdigest()
    return f"{repository}@sha256:{digest}"


def pull_environment_package(reference: str, output: Path | None = None) -> Path:
    """Fetch by digest and verify/extract a bundle without executing any downloaded code."""
    local = Path(reference).expanduser()
    is_local = local.is_file()
    with TemporaryDirectory(prefix="gym-pull-") as temporary:
        temporary_path = Path(temporary)
        if is_local:
            archive_path = local.resolve()
            if archive_path.stat().st_size > MAX_PACKAGE_BYTES:
                raise ConfigError("Environment archive exceeds the 1 GiB size limit.")
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        else:
            reference, repository = _oci_reference(reference)
            resolved_digest = _oras("resolve", reference).stdout.strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", resolved_digest):
                raise ConfigError("Registry did not return a valid sha256 digest.")
            immutable_reference = f"{repository}@{resolved_digest}"
            manifest_path = temporary_path / "manifest.json"
            _oras("manifest", "fetch", "--output", str(manifest_path), immutable_reference)
            manifest_bytes = manifest_path.read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != resolved_digest.split(":", 1)[1]:
                raise ConfigError("Registry manifest checksum mismatch.")
            try:
                manifest = json.loads(manifest_bytes)
            except ValueError as error:
                raise ConfigError("Registry returned an invalid JSON manifest.") from error
            if not isinstance(manifest, dict):
                raise ConfigError("Registry returned an invalid environment manifest.")
            layers = manifest.get("layers", [])
            artifact_type = manifest.get("artifactType") or manifest.get("annotations", {}).get(ARTIFACT_ANNOTATION)
            if artifact_type != ARTIFACT_TYPE or len(layers) != 1:
                raise ConfigError("Registry artifact is not a single-layer NeMo Gym environment package.")
            layer = layers[0]
            if (
                layer.get("mediaType") != LAYER_TYPE
                or not isinstance(layer.get("size"), int)
                or not 0 < layer["size"] <= MAX_PACKAGE_BYTES
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer.get("digest", ""))
            ):
                raise ConfigError("Invalid environment package layer.")
            digest = resolved_digest.split(":", 1)[1]

        destination = (
            (Path(output) if output is not None else CACHE_DIR / "environments" / digest).expanduser().absolute()
        )
        if destination.exists():
            if output is not None:
                raise ConfigError(f"Package destination already exists: {destination}")
            _verify_package(destination)
            return destination
        if not is_local:
            archive_path = temporary_path / "environment.tar.gz"
            _oras("blob", "fetch", "--output", str(archive_path), f"{repository}@{layer['digest']}")
            if hashlib.sha256(archive_path.read_bytes()).hexdigest() != layer["digest"].split(":", 1)[1]:
                raise ConfigError("Downloaded package checksum mismatch.")
        extracted = temporary_path / "extracted"
        extracted.mkdir()
        try:
            with tarfile.open(archive_path, mode="r:gz") as archive:
                names: set[str] = set()
                size = 0
                for member in archive:
                    _package_path(member.name)
                    size += member.size
                    if not member.isfile() or member.name in names:
                        raise ConfigError("Packages may contain only unique regular files; links are unsupported.")
                    names.add(member.name)
                    if size > MAX_PACKAGE_BYTES or len(names) > MAX_PACKAGE_FILES + 1:
                        raise ConfigError("Environment archive exceeds the 1 GiB / 10,000 file limit.")
                    archive.extract(member, extracted, filter="data")
            metadata = _verify_package(extracted)
            if names != {*metadata["files"], PACKAGE_METADATA}:
                raise ConfigError("Archive contents do not match the release file inventory.")
        except (tarfile.TarError, OSError, EOFError) as error:
            raise ConfigError(f"Could not extract environment package: {error}") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Stage on the destination filesystem so an interrupted extraction never creates a valid cache entry.
        with TemporaryDirectory(prefix=".gym-stage-", dir=destination.parent) as staging:
            staged = Path(staging) / "package"
            shutil.copytree(extracted, staged)
            staged.rename(destination)
    return destination


def submit_environment_package(reference: str) -> str:
    """Submit an existing registry package to the hub's shared review queue."""
    reference, repository = _oci_reference(reference)
    identity = repository.removeprefix(HUB_REGISTRY + "/")
    if not repository.startswith(HUB_REGISTRY + "/") or not re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*", identity
    ):
        raise ConfigError(f"Hub submissions require {HUB_REGISTRY}/publisher/name:version.")
    digest = _oras("resolve", reference).stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ConfigError("Registry did not return a valid sha256 digest.")
    immutable = f"{repository}@{digest}"
    package = pull_environment_package(immutable)
    metadata = _verify_package(package)
    manifest = load_manifest(package / metadata["manifest_path"])
    tag = f"{repository}:{manifest.version}"
    if _oras("resolve", tag).stdout.strip() != digest:
        raise ConfigError("The package version tag does not match the submitted digest.")
    url = HUB_URL + "#" + identity.replace("/", "%2F")
    run = {"model_type": "openai_model"}
    example = next((item.name for item in manifest.datasets if item.type.value == "example"), None)
    if example:
        run["sample_split"] = example
    defaults = {
        "id": identity,
        "title": manifest.name.replace("_", " ").title(),
        "description": manifest.description,
        "tags": [manifest.domain.value],
        "kind": manifest.kind.value,
        "run": run,
        "documentation_url": "",
        "comments": [],
        "runs": [],
    }
    try:
        with TemporaryDirectory(prefix="gym-hub-submit-") as temporary:
            root = Path(temporary)
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")

            def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["git", *args], cwd=root, env=env, capture_output=True, text=True, timeout=120, check=check
                )

            git("clone", "--depth", "1", "--single-branch", "--branch", "drafts", HUB_REPOSITORY, ".")
            path = root / "submissions" / (identity.replace("/", "--") + ".json")
            for attempt in range(3):
                if attempt:
                    git("fetch", "origin", "drafts")
                    git("reset", "--hard", "FETCH_HEAD")
                if path.parent.is_symlink() or path.is_symlink():
                    raise ConfigError("Hub submissions cannot use symbolic links.")
                record = json.loads(path.read_text()) if path.exists() else dict(defaults)
                if not isinstance(record, dict) or record.get("id") != identity:
                    raise ConfigError("Hub submission path belongs to a different environment.")
                if record.get("digest_reference") == immutable:
                    return url
                record = {**defaults, **record}
                record.update(
                    status="submitted",
                    version=manifest.version,
                    tag=tag,
                    digest_reference=immutable,
                )
                for key in ("archive_sha256", "reviewed_draft_commit", "published_by", "published_at"):
                    record.pop(key, None)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                git("add", "--", str(path.relative_to(root)))
                git("commit", "-s", "-m", f"Submit {identity} {manifest.version} for review [skip ci]")
                pushed = git("push", "origin", "HEAD:drafts", check=False)
                if pushed.returncode == 0:
                    return url
            raise ConfigError(f"Hub submission push failed: {pushed.stderr.strip()}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise ConfigError(f"Could not submit to the hub using GitLab SSH access: {detail}") from error
