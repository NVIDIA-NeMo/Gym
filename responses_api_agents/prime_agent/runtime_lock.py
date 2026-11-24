# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the shared Prime runtime's locked dependencies and stable cache identity."""

import base64
import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


RUNTIME_DIR = Path(__file__).with_name("runtime")
INSTALLER_PATH = Path(__file__).with_name("install_prime_runtime.sh")
HELPER_PATH = Path(__file__)
_ARCHIVES = ("prime-agent", "prime-agent-ai", "prime-agent-core", "prime-agent-tui")
_ARCHIVE_DIRECTORY = "__NEMO_GYM_PRIME_ARCHIVES__"
_INTEGRITIES = {name: "__NEMO_GYM_" + name.upper().replace("-", "_") + "_INTEGRITY__" for name in _ARCHIVES}
_TEMPLATES = {
    "package.json": "runtime-package.json",
    "package-lock.json": "runtime-package-lock.json",
    "python-constraints.txt": "runtime-python-constraints.txt",
    "python-build-constraints.txt": "runtime-python-build-constraints.txt",
}
_MARKER = re.compile(r"__[A-Za-z0-9_]+__")


def _check_integrity(value: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha512-"):
        raise ValueError("Prime archive and npm artifact integrities must use SHA512 SRI")
    try:
        decoded = base64.b64decode(value.removeprefix("sha512-"), validate=True)
    except ValueError as error:
        raise ValueError("Prime runtime integrity contains invalid base64") from error
    if len(decoded) != 64:
        raise ValueError("Prime runtime SHA512 integrity must contain exactly 64 digest bytes")


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for name, item in value.items():
            yield name
            yield from _strings(item)


def _validate_npm_templates(manifest: object, lock: object) -> None:
    file_references = {f"file:{_ARCHIVE_DIRECTORY}/{name}.tgz": name for name in _ARCHIVES}
    allowed_values = set(file_references) | set(_INTEGRITIES.values())
    for value in _strings([manifest, lock]):
        if "file:" in value and value not in file_references:
            raise ValueError("Prime runtime templates contain an unexpected file reference")
        if _MARKER.search(value) and value not in allowed_values:
            raise ValueError("Prime runtime templates contain an unknown or misplaced placeholder")
    expected = {"prime-agent": f"file:{_ARCHIVE_DIRECTORY}/prime-agent.tgz"}
    if (
        not isinstance(manifest, dict)
        or manifest.get("dependencies") != expected
        or manifest.get("private") is not True
    ):
        raise ValueError("Prime runtime manifest must contain only its verified Prime release dependency")
    if not isinstance(lock, dict) or lock.get("lockfileVersion") != 3 or not isinstance(lock.get("packages"), dict):
        raise ValueError("Prime runtime requires a complete npm lockfile version 3")
    if lock.get("name") != manifest.get("name") or lock.get("version") != manifest.get("version"):
        raise ValueError("Prime runtime manifest and lock identity disagree")
    root = lock["packages"].get("")
    if not isinstance(root, dict) or root.get("dependencies") != expected:
        raise ValueError("Prime runtime manifest and lock root dependencies disagree")
    local_archives = set()
    for location, package in lock["packages"].items():
        if not location:
            continue
        if not isinstance(package, dict) or package.get("link"):
            raise ValueError("Prime runtime lock contains an unsupported dependency link")
        resolved = package.get("resolved")
        if isinstance(resolved, str) and resolved in file_references:
            name = file_references[resolved]
            if package.get("integrity") != _INTEGRITIES[name]:
                raise ValueError("Prime runtime local archive has the wrong integrity placeholder")
            local_archives.add(name)
        elif resolved is None and package.get("inBundle"):
            # Bundled files are authenticated by their containing tarball's integrity.
            continue
        else:
            if not isinstance(resolved, str):
                raise ValueError("Prime runtime dependency has no resolved artifact")
            url = urlsplit(resolved)
            if (
                url.scheme != "https"
                or url.hostname != "registry.npmjs.org"
                or url.username
                or url.password
                or url.query
                or url.fragment
                or not url.path.endswith(".tgz")
            ):
                raise ValueError("Prime runtime lock contains an unexpected remote artifact")
            _check_integrity(package.get("integrity"))
    if local_archives != set(_ARCHIVES):
        raise ValueError("Prime runtime lock must include all four verified Prime release archives")


def stage_runtime_inputs(
    *, archive_directory: str, archive_integrities: Mapping[str, str], runtime_identity: str
) -> dict[str, bytes]:
    """Bind locked templates to verified local archives and fingerprint the installation recipe.

    The caller verifies the four original release archive SHA256 values before
    localizing them and supplying their actual SHA512 integrities. Registry
    integrity values stay exactly as committed. The profile fingerprints the
    static recipe and caller-supplied pinned runtime identity; session paths and
    gzip timestamps do not change that profile.
    """
    if (
        not isinstance(archive_directory, str)
        or not archive_directory
        or any(character in archive_directory for character in "\x00\r\n\\?#")
    ):
        raise ValueError("Prime archive directory must be an absolute filesystem path")
    directory = PurePosixPath(archive_directory)
    if not directory.is_absolute() or directory == PurePosixPath("/") or ".." in directory.parts:
        raise ValueError("Prime archive directory must be an absolute non-root path without parent traversal")
    if set(archive_integrities) != set(_ARCHIVES):
        raise ValueError("Prime runtime requires exactly one integrity for each of its four release archives")
    for integrity in archive_integrities.values():
        _check_integrity(integrity)
    if not isinstance(runtime_identity, str) or not runtime_identity.strip():
        raise ValueError("Prime runtime identity must include the pinned runtime versions and source hashes")
    templates = {name: (RUNTIME_DIR / name).read_bytes() for name in _TEMPLATES}
    decoded = {name: content.decode("utf-8") for name, content in templates.items()}
    _validate_npm_templates(json.loads(decoded["package.json"]), json.loads(decoded["package-lock.json"]))
    for name in ("python-constraints.txt", "python-build-constraints.txt"):
        if _MARKER.search(decoded[name]):
            raise ValueError("Prime Python constraints must not contain unresolved placeholders")
    # Escape only the directory's JSON string contents, preserving legitimate
    # filesystem spaces/quotes without allowing them to change the JSON structure.
    replacement = json.dumps(str(directory))[1:-1]
    staged = {}
    for name, destination in _TEMPLATES.items():
        content = decoded[name]
        if name.endswith(".json"):
            content = content.replace(_ARCHIVE_DIRECTORY, replacement)
            for archive, marker in _INTEGRITIES.items():
                content = content.replace(marker, archive_integrities[archive])
            json.loads(content)
        if _MARKER.search(content):
            raise ValueError("Prime runtime input contains an unresolved placeholder")
        staged[destination] = content.encode("utf-8")
    profile = hashlib.sha256(b"nemo-gym-prime-runtime-profile-v1\0")
    identity_files = {
        **templates,
        "install_prime_runtime.sh": INSTALLER_PATH.read_bytes(),
        "runtime_lock.py": HELPER_PATH.read_bytes(),
        "runtime_identity": runtime_identity.encode("utf-8"),
    }
    for name, content in sorted(identity_files.items()):
        profile.update(name.encode("utf-8") + b"\0" + len(content).to_bytes(8) + content)
    staged["runtime-profile.txt"] = (profile.hexdigest() + "\n").encode("ascii")
    return staged
