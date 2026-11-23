# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""_TARGET_INSTALL_SCRUB_COMMAND against a fake image root (NL2REPO_SCRUB_PREFIX), built from the leak shapes found in
the upstream images on 2026-11-02: a pip install from a local checkout under another name (paillier -> phe), a
setup.py-install egg (pytz), and checkouts left at the root (/app, /project, /jinja)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from resources_servers.nl2repobench.app import _TARGET_INSTALL_SCRUB_COMMAND


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash, python3"
)


def _write(path: Path, text: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _dist(site: Path, name: str, files: list[str], url: str | None) -> Path:
    di = site / f"{name}.dist-info"
    for rel in files:
        _write(site / rel)
    _write(di / "METADATA", f"Name: {name}\n")
    _write(di / "RECORD", "".join(f"{rel},,\n" for rel in files + [f"{name}.dist-info/METADATA"]))
    if url is not None:
        _write(di / "direct_url.json", json.dumps({"dir_info": {}, "url": url}))
    return di


@pytest.fixture()
def image(tmp_path: Path) -> Path:
    site = tmp_path / "usr/local/lib/python3.10/site-packages"
    # target installed from its checkout under a different distribution name, plus a console script and a stale pyc
    _dist(site, "phe-1.5.0", ["phe/__init__.py", "phe/paillier.py", "../../../bin/pheutil"], "file:///app")
    _write(site / "phe/__pycache__/paillier.cpython-310.pyc", "")
    # a real dependency from an index: no direct_url.json, and its import name equals the task-ish ident "sklearn"
    _dist(site, "scikit_learn-0.24.2", ["sklearn/__init__.py"], None)
    # a dependency installed from a URL that is not a local checkout
    _dist(site, "somedep-1.0", ["somedep.py"], "https://example.com/somedep-1.0.tar.gz")
    # a broken record pointing at the filesystem root must not delete it
    _dist(site, "weird-0.1", ["weird.py"], "file:///")
    # setup.py-install egg of the target and an unrelated egg, both on easy-install.pth
    _write(site / "pytz-2025.2-py3.10.egg/pytz/__init__.py")
    _write(site / "other-1.0-py3.10.egg/other/__init__.py")
    _write(site / "easy-install.pth", "./pytz-2025.2-py3.10.egg\n./other-1.0-py3.10.egg\n")
    # a target package dir already emptied by the workspace-install scrub (only bytecode left), and a target-named
    # dir that still holds a compiled extension (left alone)
    _write(site / "stamina/__pycache__/__init__.cpython-310.pyc", "")
    _write(site / "jinja/_speedups.so", "")
    # checkouts at the root
    _write(tmp_path / "app/phe/paillier.py")
    _write(tmp_path / "project/asteval/asteval.py")
    _write(tmp_path / "jinja/tests/test_api.py")
    _write(tmp_path / "jinja/pyproject.toml", "[project]\n")
    _write(tmp_path / "sklearn/notes.txt", "no project files here")  # named like an ident but not a project
    _write(tmp_path / "workspace/start.md", "# spec")
    _write(tmp_path / "opt/tool/run.py")
    return tmp_path


def _run(root: Path, idents: str) -> subprocess.CompletedProcess:
    cmd = _TARGET_INSTALL_SCRUB_COMMAND.replace("__IDENTS__", idents)
    return subprocess.run(
        ["bash", "-c", cmd],
        env={**os.environ, "NL2REPO_SCRUB_PREFIX": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_removes_reference_project_and_keeps_dependencies(image: Path) -> None:
    proc = _run(image, "paillier pytz jinja sklearn stamina")
    assert proc.returncode == 0, proc.stderr
    site = image / "usr/local/lib/python3.10/site-packages"

    # gone: the target's files, dist-info, console script, pyc-only package dir, egg, pth line, checkouts
    for gone in ("phe", "phe-1.5.0.dist-info", "pytz-2025.2-py3.10.egg", "stamina"):
        assert not (site / gone).exists(), gone
    assert not (image / "usr/local/bin/pheutil").exists()
    for gone in ("app", "project", "jinja"):
        assert not (image / gone).exists(), gone
    assert (site / "easy-install.pth").read_text() == "./other-1.0-py3.10.egg\n"

    # kept: index-installed deps (even one named like an ident), a non-file URL install, unrelated egg, the root
    # entry for a file:/// record, a same-named non-project dir, /workspace and /opt
    for kept in (
        "sklearn/__init__.py",
        "scikit_learn-0.24.2.dist-info",
        "somedep.py",
        "other-1.0-py3.10.egg",
        "jinja/_speedups.so",
    ):
        assert (site / kept).exists(), kept
    assert (image / "sklearn/notes.txt").exists()
    assert (image / "workspace/start.md").exists()
    assert (image / "opt/tool/run.py").exists()
    assert "target-install scrub removed" in proc.stdout


def test_is_idempotent(image: Path) -> None:
    assert _run(image, "paillier pytz jinja").returncode == 0
    second = _run(image, "paillier pytz jinja")
    assert second.returncode == 0, second.stderr
    assert "target-install scrub removed 0" in second.stdout
