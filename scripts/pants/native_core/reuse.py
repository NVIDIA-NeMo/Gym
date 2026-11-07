# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed policy for an audited, distribution-free test group.

This module is deliberately not wired into execution. Source approval alone is
not permission to reuse results: the complete native process, environment and
runtime fingerprint must first pass cold/hit parity on the hosted platform.
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


RESULT_REUSE_ENABLED = False
AUDIT_PATH = Path(__file__).with_name("reuse_audit.json")
CANDIDATES = (
    "tests/unit_tests/models/test_anthropic_converter.py",
    "tests/unit_tests/models/test_anthropic_converter_egress.py",
    "tests/unit_tests/models/test_anthropic_reasoning.py",
    "tests/unit_tests/models/test_anthropic_utils.py",
    "tests/unit_tests/models/test_chat_completions_converter.py",
)
SOURCE_PREFIXES = ("nemo_gym/", "tests/", "scripts/ci/")
SOURCE_FILES = frozenset({"pyproject.toml", "uv.lock", ".python-version"})
EXPECTED_CONFTESTS = frozenset({"tests/conftest.py", "tests/unit_tests/conftest.py"})

# Caller must fingerprint these generated/locked inputs in addition to source.
PROCESS_INPUTS = (
    "canonical sampled editable metadata, finder and pth",
    "native runtime/plugin/bootstrap and generated BUILD/configuration",
    "core requirement and resolve artifacts, pytest-tool/plugin artifacts",
    "Pants/PEX artifacts, exact argv, markers and batch membership",
    "verified runtime fingerprint and complete effective process environment",
)
FORBIDDEN_DEPENDENCIES = (":assets", ":native-project", "git-state", "external fixture mounts")

# No inherited HOME, PYTHONPATH, plugin configuration, credentials or CI paths.
# Pants/PEX's own environment still needs a separate effective-process audit;
# this mapping alone must never be mistaken for the complete process key.
SEMANTIC_ENVIRONMENT = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"}
RUNTIME_DIGEST_FIELDS = (
    "interpreter_artifact_sha256",  # executable plus stdlib and bundled runtime
    "native_libraries_sha256",  # resolved shared-library contents, not ldd text
    "runner_image_sha256",  # immutable image/content identity, never a runner label
    "dependency_artifacts_sha256",  # exact wheel/PEX bytes including native wheels
    "pytest_plugins_sha256",  # installed inventory and entry-point metadata
    "effective_environment_sha256",  # observed Pants process, not only outer env
)
RUNTIME_VALUE_FIELDS = ("python_implementation", "python_version", "system", "machine", "kernel_release")


def audit_source(source: Path) -> dict:
    """Approve only the recorded code closure and unchanged inherited fixtures."""
    audit = json.loads(AUDIT_PATH.read_text())
    errors = []
    if tuple(audit["candidate_modules"]) != CANDIDATES:
        errors.append("Audit allowlist differs from the policy")
    if not {*CANDIDATES, *EXPECTED_CONFTESTS, *SOURCE_FILES} <= audit["observed_sha256"].keys():
        errors.append("Audit is missing required input fingerprints")
    for name, expected in audit["observed_sha256"].items():
        path = source / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"Missing or replaced audited input: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Audited input changed; review required: {name}")
    ancestors = {parent for name in CANDIDATES for parent in Path(name).parents}
    actual = {
        (parent / "conftest.py").as_posix() for parent in ancestors if (source / parent / "conftest.py").exists()
    }
    if actual != EXPECTED_CONFTESTS:
        errors.append("Inherited conftest membership changed; review required")
    for name in ("pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg"):
        if (source / name).exists():
            errors.append(f"Unaudited pytest configuration: {name}")
    return {
        "valid": not errors,
        "errors": errors,
        "candidates": list(CANDIDATES) if not errors else [],
        "audit_source": audit["source_commit"],
        "reuse_enabled": RESULT_REUSE_ENABLED,
    }


def reduced_source_inputs(source: Path) -> list[dict[str, str]]:
    """Inventory actual Git-tracked inputs, including names, modes and symlinks.

    Reading the controller's checkout Git index is intentional. Its bytes and
    commit identity must not enter the candidate sandbox or input digest.
    """
    output = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=source)
    tracked = {entry.split("\t", 1)[1] for entry in output.decode().split("\0") if entry}
    selected = {}
    for entry in output.decode().split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, _, stage = metadata.split()
        if not (name.startswith(SOURCE_PREFIXES) or name in SOURCE_FILES):
            continue
        if stage != "0" or mode == "160000":
            raise ValueError(f"Unsupported staged input: {name}")
        path = source / name
        if path.is_symlink():
            content = os.readlink(path).encode()
            resolved = path.resolve()
            if not resolved.is_relative_to(source.resolve()):
                raise ValueError(f"Input symlink escapes source: {name}")
            target = resolved.relative_to(source.resolve()).as_posix()
            if not (target.startswith(SOURCE_PREFIXES) or target in SOURCE_FILES):
                raise ValueError(f"Input symlink requires excluded content: {name}")
            if target not in tracked and not any(name.startswith(target + "/") for name in tracked):
                raise ValueError(f"Input symlink target is not tracked: {name}")
            actual_mode = "120000"
        else:
            content = path.read_bytes()
            actual_mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        if actual_mode != mode:
            raise ValueError(f"Input mode differs from tracked source: {name}")
        selected[name] = {"path": name, "mode": mode, "sha256": hashlib.sha256(content).hexdigest()}
    missing = SOURCE_FILES - selected.keys()
    if missing:
        raise ValueError(f"Missing required tracked configuration: {sorted(missing)}")
    return [selected[name] for name in sorted(selected)]


def candidate_environment() -> dict[str, str]:
    """Return fixed semantics; do not copy any variable from the calling shell."""
    return dict(SEMANTIC_ENVIRONMENT)


def runtime_fingerprint(evidence: dict) -> str:
    """Validate the required evidence shape and key it; do not collect or certify it.

    A future collector must verify the actual interpreter, loaded libraries and
    immutable runner image before supplying these digests. Local IPv6 binding is
    a real observed import-time input, so its current outcome is mandatory too.
    Missing evidence means forced execution. Passing this schema does not enable
    reuse or replace the native process/cold-hit parity checks.
    """
    required = {*RUNTIME_DIGEST_FIELDS, *RUNTIME_VALUE_FIELDS, "ipv6_bind_available"}
    if set(evidence) != required:
        raise ValueError("Incomplete or unknown runtime fingerprint evidence")
    for name in RUNTIME_DIGEST_FIELDS:
        if not isinstance(evidence[name], str) or not re.fullmatch(r"[0-9a-f]{64}", evidence[name]):
            raise ValueError(f"Expected verified content SHA256 for {name}")
    for name in RUNTIME_VALUE_FIELDS:
        if not isinstance(evidence[name], str) or not evidence[name].strip():
            raise ValueError(f"Missing runtime identity: {name}")
    if type(evidence["ipv6_bind_available"]) is not bool:
        raise ValueError("IPv6 bind capability must be measured, not omitted or assumed")
    if evidence["python_implementation"] != "CPython" or not re.fullmatch(r"3\.13\.\d+", evidence["python_version"]):
        raise ValueError("Interpreter is outside the audited Python family")
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
