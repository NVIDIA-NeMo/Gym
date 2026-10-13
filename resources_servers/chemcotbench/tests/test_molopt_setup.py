# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import subprocess
from unittest.mock import MagicMock

import pytest

from resources_servers.chemcotbench import setup_molopt
from resources_servers.chemcotbench.setup_molopt import ensure_molopt_runtime


@pytest.fixture(autouse=True)
def isolate_oracle_downloads(monkeypatch):
    # Runtime setup tests cover process/venv behavior; artifact tests below override this.
    monkeypatch.setattr(setup_molopt, "ORACLE_ARTIFACTS", {})


def test_external_runtime_is_checked_once_without_install(monkeypatch, tmp_path):
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    cache = tmp_path / "cache"
    (cache / "oracle").mkdir(parents=True)
    for name in ("drd2", "gsk3b", "jnk3"):
        (cache / "oracle" / f"{name}_current.pkl").write_bytes(b"fixture")
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    first = ensure_molopt_runtime(str(python), str(cache))
    assert first == (python, cache)
    assert run.call_count == 1
    assert run.call_args.args[0][0] == str(python)
    assert run.call_args.kwargs["cwd"] == cache
    assert ensure_molopt_runtime(str(python), str(cache)) == first
    assert run.call_count == 1


def test_failed_oracle_check_is_not_marked_ready(monkeypatch, tmp_path):
    run = MagicMock(side_effect=subprocess.CalledProcessError(1, "oracle check"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        ensure_molopt_runtime("/missing/python", str(tmp_path))
    assert not (tmp_path / "ready.json").exists()


def test_broken_managed_venv_is_recreated(monkeypatch, tmp_path):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(tmp_path / "missing-python")
    (tmp_path / "venv/pyvenv.cfg").write_text("home = /missing\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "venv":
            # uv rejects an existing environment unless recreation is explicit.
            if "--clear" not in command:
                raise subprocess.CalledProcessError(2, command)
            python.unlink()
            python.touch()

    monkeypatch.setattr(subprocess, "run", run)
    result = ensure_molopt_runtime(cache_dir=str(tmp_path))
    assert result == (python, tmp_path)
    assert python.exists() and not python.is_symlink()
    assert commands[1][1:3] == ["pip", "install"]
    assert commands[2][0] == str(python)
    assert (tmp_path / "ready.json").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_oracle_download_and_reuse(monkeypatch, tmp_path, existing):
    data = b"validated model artifact"
    monkeypatch.setattr(setup_molopt, "ORACLE_ARTIFACTS", {"drd2": (123, hashlib.sha256(data).hexdigest())})
    path = tmp_path / "oracle/drd2_current.pkl"
    if existing:
        path.parent.mkdir()
        path.write_bytes(data)
    download = MagicMock(side_effect=lambda *args, **kwargs: io.BytesIO(data))
    monkeypatch.setattr(setup_molopt, "urlopen", download)
    setup_molopt.ensure_oracle_artifacts(tmp_path)
    assert path.read_bytes() == data
    assert download.call_count == (0 if existing else 1)
    if not existing:
        request = download.call_args.args[0]
        assert request.full_url == "https://dataverse.harvard.edu/api/access/datafile/123"
        assert request.get_header("User-agent") == "NeMo-Gym/1.0 (ChemCoTBench oracle setup)"
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
def test_bad_oracle_never_reaches_pickle_loader(monkeypatch, tmp_path, existing):
    monkeypatch.setattr(setup_molopt, "ORACLE_ARTIFACTS", {"drd2": (123, hashlib.sha256(b"model").hexdigest())})
    path = tmp_path / "oracle/drd2_current.pkl"
    if existing:
        path.parent.mkdir()
        path.write_bytes(b"<html>504 Gateway Timeout</html>")
    monkeypatch.setattr(setup_molopt, "urlopen", lambda *a, **k: io.BytesIO(b"<html>504 Gateway Timeout</html>"))
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="Corrupt|checksum"):
        ensure_molopt_runtime("/fixture/python", str(tmp_path))
    run.assert_not_called()
    assert path.exists() is existing
    assert not (tmp_path / "ready.json").exists()
    assert list(path.parent.iterdir()) == ([path] if existing else [])


def test_missing_uv_reports_setup_requirement(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_molopt.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uv is required"):
        ensure_molopt_runtime(cache_dir=str(tmp_path))
