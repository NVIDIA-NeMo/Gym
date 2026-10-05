# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from resources_servers.chemcotbench import setup_upstream


def make_repo(path):
    for name in ("formal_cot", "evaluation", "baselines"):
        (path / name).mkdir(parents=True)


def test_local_repo_and_data(tmp_path):
    make_repo(tmp_path)
    assert setup_upstream.ensure_repository(str(tmp_path)) == tmp_path
    (tmp_path / "manifest.json").write_text("{}")
    assert setup_upstream.ensure_data(str(tmp_path)) == tmp_path


def test_bad_paths(tmp_path):
    with pytest.raises(ValueError, match="Not a ChemCoTBench"):
        setup_upstream.ensure_repository(str(tmp_path))
    with pytest.raises(ValueError, match="manifest.json"):
        setup_upstream.ensure_data(str(tmp_path))


def test_download_is_pinned_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_upstream, "__file__", str(tmp_path / "setup_upstream.py"))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "clone":
            make_repo(Path(command[-1]))
        assert kwargs["check"] is True

    monkeypatch.setattr(setup_upstream.subprocess, "run", run)
    repo = setup_upstream.ensure_repository(None)
    assert repo == tmp_path / ".upstream" / setup_upstream.REPO_REVISION
    assert calls[-1][-1] == setup_upstream.REPO_REVISION
    assert setup_upstream.ensure_repository(None) == repo
    assert len(calls) == 2


def test_data_download_revision(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")

    def download(repo, **kwargs):
        assert repo == "fresnellll/ChemCoTBench-V2"
        assert kwargs == {"repo_type": "dataset", "revision": setup_upstream.DATA_REVISION}
        return str(tmp_path)

    monkeypatch.setattr(setup_upstream, "snapshot_download", download)
    assert setup_upstream.ensure_data(None) == tmp_path
