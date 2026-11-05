# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit input boundary for the expensive repository migration tests."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


MIGRATION_TEST = "tests/unit_tests/tooling/test_add_legacy_agent_environment_servers.py"
SCANNED_ROOTS = (
    "benchmarks/",
    "environments/",
    "resources_servers/",
    "responses_api_agents/",
    "responses_api_models/",
)
AUDITED_FILES = (
    MIGRATION_TEST,
    "scripts/add_legacy_agent_environment_servers.py",
    "tests/conftest.py",
    "tests/unit_tests/conftest.py",
)
ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONPYCACHEPREFIX": ".incremental-bytecode",
    "HOME": "/nonexistent",
    "PATH": "/usr/bin:/bin",
}


def eligibility(source: Path, policy: Path) -> dict[str, object]:
    """A changed external-read boundary returns to forced execution for inspection."""
    expected = json.loads(policy.read_text())
    if not set(AUDITED_FILES) <= expected.get("audited_sha256", {}).keys():
        raise ValueError("Incremental audit omits required test/script/conftest inputs")
    failures = [
        name
        for name, digest in expected["audited_sha256"].items()
        if not (source / name).is_file() or hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
    ]
    conftests = {
        str(path.relative_to(source))
        for parent in (source / MIGRATION_TEST).parents
        if parent.is_relative_to(source)
        for path in [parent / "conftest.py"]
        if path.exists()
    }
    if conftests != {name for name in AUDITED_FILES if name.endswith("conftest.py")}:
        failures.append("inherited conftest membership")
    try:
        declared_files(source)
    except ValueError as error:
        failures.append(str(error))
    return {"eligible": not failures, "forced_reasons": failures, "candidate": MIGRATION_TEST}


def declared_files(source: Path) -> list[str]:
    """Inventory scanner trees, explicit dynamic imports and non-Python fixtures.

    Names as well as contents enter the generated target, so additions/deletions
    invalidate a scanner. Ordinary Python imports retain Pants inference.
    """
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    explicit = {*AUDITED_FILES, "pyproject.toml", "uv.lock", ".python-version"}
    selected = sorted(
        name
        for name in tracked
        if name
        and (
            name in explicit
            or name.startswith(SCANNED_ROOTS)
            or (name.startswith(("tests/", "nemo_gym/")) and not name.endswith(".py"))
            or name.startswith("tests/helpers/")
        )
    )
    files = set()
    for name in selected:
        path = source / name
        if path.is_symlink():
            resolved = path.resolve()
            if not resolved.is_relative_to(source.resolve()) or not resolved.is_file():
                raise ValueError(f"Scanner link needs explicit external/directory support: {name}")
            files.add(resolved.relative_to(source.resolve()).as_posix())
        elif not path.is_file():
            raise ValueError(f"Missing scanner input: {name}")
        else:
            files.add(name)
    return sorted(files)


def build_target(source: Path) -> str:
    """Keep Git snapshots and the genuine project wheel out of this target closure."""
    native = source / "scripts/pants/native_core"
    files = declared_files(source)
    links = {}
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    for name in filter(None, tracked):
        path = source / name
        if path.is_symlink() and name.startswith(SCANNED_ROOTS):
            links[name] = os.readlink(path)
    (native / "incremental-links.json").write_text(json.dumps(links, sort_keys=True) + "\n")
    policy = json.loads((native / "incremental-policy.json").read_text())
    files += list(policy["audited_sha256"])
    files += [str(path.relative_to(source)) for path in (native / "distribution").rglob("*") if path.is_file()]
    files += [
        f"scripts/pants/native_core/{name}"
        for name in (
            "incremental_runtime.py",
            "runtime_fingerprint.py",
            "incremental.py",
            "editable.py",
            "runtime-inputs.json",
            "incremental-policy.json",
            "incremental-links.json",
        )
    ]
    return f"files(name='incremental-assets', sources={sorted(set(files))!r})\n"
