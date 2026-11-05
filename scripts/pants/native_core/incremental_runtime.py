# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore sampled metadata and reject external-runtime drift before success caching."""

import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from scripts.pants.native_core.editable import relocate_finder
from scripts.pants.native_core.incremental import ENVIRONMENT
from scripts.pants.native_core.runtime_fingerprint import fingerprint


ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / "scripts/pants/native_core"


def verify() -> None:
    """A mutation produces a failed pytest process, which cannot enter SUCCESSFUL cache."""
    expected = json.loads((NATIVE / "runtime-inputs.json").read_text())
    if str(Path(sys.base_prefix).resolve()) not in expected["roots"]:
        raise RuntimeError("Actual native interpreter differs from the verified runtime")
    for name, value in ENVIRONMENT.items():
        accepted = {value}
        if name == "PATH":
            # PEX prepends its own declared virtualenv's bin directory. Reject
            # every other added search path; do not normalize ambient tools.
            accepted |= {
                str(Path(sys.executable).parent) + os.pathsep + value,
                str(Path(sys.prefix) / "bin") + os.pathsep + value,
            }
            actual = os.environ.get(name, "")
            prefix, separator, suffix = actual.partition(os.pathsep)
            if separator and suffix == value and Path(prefix).resolve() == (Path(sys.prefix) / "bin").resolve():
                accepted.add(actual)
        if os.environ.get(name) not in accepted:
            if name == "PATH":
                raise RuntimeError(
                    f"Incremental PATH differs: actual={os.environ.get(name)!r}, "
                    f"executable={sys.executable!r}, prefix={sys.prefix!r}"
                )
            raise RuntimeError(f"Incremental environment differs for {name}")
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        actual = fingerprint(expected["roots"], expected["paths"])
        if actual["sha256"] != expected["fingerprint"]["sha256"]:
            raise RuntimeError("External runtime changed before native result caching")


def pytest_load_initial_conftests(early_config, parser, args):
    verify()
    snapshot = NATIVE / "distribution"
    with (NATIVE / ".incremental.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = list(snapshot.glob("*.dist-info"))
        if len(metadata) != 1:
            raise RuntimeError("Missing canonical sampled Gym metadata")
        installed = ROOT / metadata[0].name
        if not installed.exists():
            for name, target in json.loads((NATIVE / "incremental-links.json").read_text()).items():
                link = ROOT / name
                link.parent.mkdir(parents=True, exist_ok=True)
                resolved = (link.parent / target).resolve()
                if not resolved.is_relative_to(ROOT) or not resolved.is_file():
                    raise RuntimeError(f"Missing declared scanner link target: {name}")
                link.symlink_to(target)
            shutil.copytree(metadata[0], installed)
            (installed / "direct_url.json").write_text(
                json.dumps({"dir_info": {"editable": True}, "url": ROOT.as_uri()})
            )
            manifest = json.loads((snapshot / "editable.json").read_text())
            finder = relocate_finder(
                (snapshot / manifest["finder"]).read_text(), original=manifest["source_root"], replacement=str(ROOT)
            )
            (ROOT / manifest["finder"]).write_text(finder)
            shutil.copy2(snapshot / manifest["hook"], ROOT / manifest["hook"])


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    try:
        verify()
    except (OSError, ValueError, RuntimeError):
        session.exitstatus = 1
        raise
