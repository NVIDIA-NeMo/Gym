# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect duplicate-download locking and persistent node-cache correctness."""

import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.pdf_analyst_qa.sandbox_scripts import document_cache


@pytest.fixture
def cache_setup(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    workspace = "workspace_" + "a" * 24
    view = cache / "views" / workspace
    view.mkdir(parents=True)
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    source = bucket / "report.pdf"
    source.write_bytes(b"%PDF complete original")
    manifest = tmp_path / "manifest.json"
    records = [{"source": str(source), "target": "report.pdf", "bytes": source.stat().st_size}]
    manifest.write_text(json.dumps(records))
    boot_id = tmp_path / "boot-id"
    boot_id.write_text("test-worker-boot-id")
    options = {
        "manifest_path": manifest,
        "source_mount": bucket,
        "workspace": workspace,
        "cache": cache,
        "docs": view,
        "private": tmp_path / "private",
        "boot_id_path": boot_id,
    }
    yield options
    view.chmod(0o700)


@pytest.mark.parametrize("defect", ["wrong_size", "extra_pdf", "symlink", "writable_mount", None])
def test_node_view_rejects_wrong_documents_and_writable_mounts(tmp_path, monkeypatch, defect):
    docs, scratch = tmp_path / "docs", tmp_path / "scratch"
    docs.mkdir()
    pdf = docs / "report.pdf"
    pdf.write_bytes(b"%PDF report")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"target": "report.pdf", "bytes": pdf.stat().st_size}]))
    if defect == "wrong_size":
        pdf.write_bytes(b"truncated")
    elif defect == "extra_pdf":
        (docs / "unselected.pdf").write_bytes(b"%PDF outside the allowlist")
    elif defect == "symlink":
        backing = tmp_path / "backing.pdf"
        pdf.rename(backing)
        pdf.symlink_to(backing)
    flags = 0 if defect == "writable_mount" else os.ST_RDONLY
    monkeypatch.setattr(document_cache.os, "statvfs", lambda path: SimpleNamespace(f_flag=flags))
    monkeypatch.setattr(document_cache.os, "chown", lambda *args: None)
    if defect is None:
        document_cache.check_document_cache(manifest, docs=docs, task=scratch)
        assert scratch.is_dir()
        assert scratch.stat().st_mode & 0o777 == 0o700
    else:
        message = "read-only mount" if defect == "writable_mount" else "does not match"
        with pytest.raises(RuntimeError, match=message):
            document_cache.check_document_cache(manifest, docs=docs, task=scratch)
    assert not manifest.exists()


def test_concurrent_siblings_copy_each_pdf_once_and_share_one_view(cache_setup):
    options = cache_setup
    records = json.loads(options["manifest_path"].read_text())
    other = options["source_mount"] / "other.pdf"
    other.write_bytes(b"%PDF other" * 1000)
    records.append({"source": str(other), "target": "other.pdf", "bytes": other.stat().st_size})
    options["manifest_path"].write_text(json.dumps(records))
    commands = []
    for index in range(4):
        commands.append(
            [
                sys.executable,
                "-I",
                document_cache.__file__,
                str(options["manifest_path"]),
                str(options["source_mount"]),
                options["workspace"],
                "--cache",
                str(options["cache"]),
                "--docs",
                str(options["docs"]),
                "--private",
                str(options["private"].with_name(f"private-{index}")),
                "--boot-id-path",
                str(options["boot_id_path"]),
            ]
        )
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        for command in commands
    ]
    metrics = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        metrics.append(json.loads(stdout))
    assert len({item["cache_id"] for item in metrics}) == 1
    assert sum(item["copied_files"] for item in metrics) == 2
    assert sum(item["copied_bytes"] for item in metrics) == sum(record["bytes"] for record in records)
    for record in records:
        file = options["docs"] / record["target"]
        assert file.samefile(options["cache"] / "store" / Path(record["source"]).name)
        assert not file.stat().st_mode & 0o222
    assert not options["cache"].stat().st_mode & 0o077
    repeated = subprocess.run(commands[0], capture_output=True, text=True, errors="replace", timeout=30)
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout)["copied_bytes"] == 0


def test_truncated_cached_file_is_rejected_before_exposing_a_view(cache_setup):
    cache = cache_setup["cache"]
    (cache / "store").mkdir()
    cached = cache / "store/report.pdf"
    cached.write_bytes(b"truncated")
    cached.chmod(0o444)
    with pytest.raises(RuntimeError, match="file size or permissions"):
        document_cache.initialize_cache(**cache_setup)
    assert not (cache / "view-receipts" / (cache_setup["workspace"] + ".json")).exists()
    assert not list(cache_setup["docs"].iterdir())


@pytest.mark.parametrize("error_code,expected_attempts", [(errno.EIO, 4), (errno.ENOSPC, 1)])
def test_copy_retries_are_bounded_and_do_not_retry_disk_exhaustion(
    cache_setup, monkeypatch, error_code, expected_attempts
):
    attempts = 0

    def fail(*args):
        nonlocal attempts
        attempts += 1
        raise OSError(error_code, "injected transfer failure")

    monkeypatch.setattr(document_cache.shutil, "copyfile", fail)
    monkeypatch.setattr(document_cache.time, "sleep", lambda seconds: None)
    with pytest.raises(OSError) as error:
        document_cache.initialize_cache(**cache_setup)
    assert error.value.errno == error_code
    assert attempts == expected_attempts
    assert not (cache_setup["cache"] / "store/report.pdf").exists()


def test_transient_source_error_recovers_without_publishing_partial_bytes(cache_setup, monkeypatch):
    original = document_cache.shutil.copyfile
    attempts = 0

    def flaky(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            Path(target).write_bytes(b"partial")
            raise OSError(errno.EIO, "transient S3 read failure")
        return original(source, target)

    monkeypatch.setattr(document_cache.shutil, "copyfile", flaky)
    monkeypatch.setattr(document_cache.time, "sleep", lambda seconds: None)
    metrics = document_cache.initialize_cache(**cache_setup)
    assert metrics["copy_retries"] == 1 and metrics["copied_files"] == 1
    assert (cache_setup["docs"] / "report.pdf").read_bytes() == (
        cache_setup["source_mount"] / "report.pdf"
    ).read_bytes()
