# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the scripts delivered to sandboxes without a Gym installation."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.pdf_analyst_qa.documents import sandbox_script_files
from resources_servers.pdf_analyst_qa.sandbox_scripts import documents, network_probe


@pytest.mark.parametrize("valid_checksum", [True, False])
def test_document_copy_checks_contents_and_protects_selected_pdfs(tmp_path, monkeypatch, valid_checksum):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF selected document")
    digest = hashlib.sha256(source.read_bytes()).hexdigest() if valid_checksum else "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"source": str(source), "target": "selected/report.pdf", "sha256": digest}]))
    docs, scratch = tmp_path / "docs", tmp_path / "task"
    monkeypatch.setattr(documents.os, "chown", lambda *args: None)
    if valid_checksum:
        documents.materialize_documents(manifest, docs=docs, task=scratch)
        assert (docs / "selected/report.pdf").read_bytes() == source.read_bytes()
        assert not (docs / "selected/report.pdf").samefile(source)
        assert all(not path.stat().st_mode & 0o222 for path in [docs, *docs.rglob("*")])
        assert scratch.stat().st_mode & 0o777 == 0o700
        (docs / "selected").chmod(0o700)
        docs.chmod(0o700)
    else:
        with pytest.raises(RuntimeError, match="differs from the dataset PDF"):
            documents.materialize_documents(manifest, docs=docs, task=scratch)
    assert not manifest.exists()


@pytest.mark.parametrize("read_only", [True, False])
def test_document_store_hides_parent_of_bucket_mount(tmp_path, monkeypatch, read_only):
    private = tmp_path / "private"
    mount = private / "mount"
    source = mount / "data"
    source.mkdir(parents=True)
    flags = os.ST_RDONLY if read_only else 0
    monkeypatch.setattr(documents.os, "statvfs", lambda path: SimpleNamespace(f_flag=flags))
    monkeypatch.setattr(Path, "is_mount", lambda path: path in {mount, Path("/")})
    if read_only:
        documents.hide_document_store(source)
        assert private.stat().st_mode & 0o777 == 0o700
    else:
        with pytest.raises(RuntimeError, match="shared bucket mount must be read-only"):
            documents.hide_document_store(source)


@pytest.mark.parametrize("failure", [None, "model_unreachable", "public_network_allowed"])
def test_network_probe_requires_model_access_and_blocked_public_network(monkeypatch, failure):
    def connect(address, *, timeout):
        if address[0] == "model.example":
            if failure == "model_unreachable":
                raise OSError("model unreachable")
            return nullcontext()
        if failure == "public_network_allowed":
            return nullcontext()
        raise OSError("public network blocked")

    monkeypatch.setattr(network_probe.socket, "create_connection", connect)
    if failure == "model_unreachable":
        with pytest.raises(OSError, match="model unreachable"):
            network_probe.check("http://model.example:8000")
    elif failure == "public_network_allowed":
        with pytest.raises(RuntimeError, match="unrelated egress is allowed"):
            network_probe.check("http://model.example:8000")
    else:
        network_probe.check("http://model.example:8000")


@pytest.mark.parametrize("script", ["documents.py", "document_cache.py", "preflight.py", "network_probe.py"])
def test_delivered_script_runs_without_installed_packages(tmp_path, script):
    for destination, contents in sandbox_script_files().items():
        (tmp_path / Path(destination).name).write_text(contents)
    result = subprocess.run(
        [sys.executable, "-S", str(tmp_path / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
