# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small archives exercise real worker limits without exhausting host resources."""

import asyncio
import gzip
import io
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psutil
import pytest

from resources_servers.legal_agent_bench import archive_worker, artifacts
from resources_servers.legal_agent_bench.archive_worker import ArchiveLimits


def archive(path, entries):
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as output:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
    return path


@pytest.mark.parametrize(
    "entries, overrides, message",
    [
        ([("output/a", b"a")], {"max_archive_bytes": 8}, "max_archive_bytes"),
        ([("output/a", b"a" * 30_000)], {"max_expanded_bytes": 20_000}, "max_expanded_bytes"),
        ([("output/a", b"a" * 100)], {"max_file_bytes": 99}, "max_file_bytes"),
        ([("output/a", b""), ("output/b", b"")], {"max_entries": 1}, "max_entries"),
        ([("output/a/b/c", b"")], {"max_entries": 3}, "max_entries"),
        ([("output/long", b"")], {"max_path_bytes": 8}, "max_path_bytes"),
        ([("output/a/b", b"")], {"max_path_depth": 2}, "max_path_depth"),
        ([("output/a", b"one"), ("output/a", b"two")], {}, "Duplicate"),
        ([("output/A", b"one"), ("output/a", b"two")], {}, "Duplicate"),
        ([("output/a", b"one"), ("output/a/b", b"two")], {}, "Conflicting"),
    ],
)
async def test_limits_and_conflicts_preserve_previous_output(tmp_path, entries, overrides, message):
    source = archive(tmp_path / "archive.gz", entries)
    destination = tmp_path / "results"
    (destination / "output").mkdir(parents=True)
    (destination / "output/old.txt").write_text("keep")
    with pytest.raises(ValueError, match=message):
        await artifacts.extract_outputs(source, destination, prefix="output", limits=ArchiveLimits(**overrides))
    assert list((destination / "output").iterdir()) == [destination / "output/old.txt"]
    assert (destination / "output/old.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".lab-extract-*"))


async def test_oversized_pax_metadata_is_bounded_before_member_is_returned(tmp_path):
    source = tmp_path / "pax.gz"
    with tarfile.open(source, "w:gz", format=tarfile.PAX_FORMAT) as output:
        info = tarfile.TarInfo("output/answer.txt")
        info.pax_headers = {"comment": "x" * 100_000}
        output.addfile(info)
    assert source.stat().st_size < 1024
    with pytest.raises(ValueError, match="max_expanded_bytes"):
        await artifacts.extract_outputs(
            source, tmp_path / "results", prefix="output", limits=ArchiveLimits(max_expanded_bytes=32_768)
        )
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("trailing", [False, True])
async def test_truncated_payload_and_trailing_compression_bomb_publish_nothing(tmp_path, trailing):
    source = tmp_path / "archive.gz"
    if trailing:
        archive(source, [("output/answer.txt", b"valid")])
        with source.open("ab") as output:
            output.write(gzip.compress(b"\0" * 100_000))
    else:
        header = tarfile.TarInfo("output/answer.txt")
        header.size = 100_000
        source.write_bytes(gzip.compress(header.tobuf() + b"short"))
    with pytest.raises(ValueError):
        await artifacts.extract_outputs(
            source, tmp_path / "results", prefix="output", limits=ArchiveLimits(max_expanded_bytes=32_768)
        )
    assert not (tmp_path / "results").exists()


async def test_file_limit_rejects_huge_declared_size_without_huge_fixture(tmp_path):
    header = tarfile.TarInfo("output/answer.txt")
    header.size = 1024**4
    source = tmp_path / "archive.gz"
    source.write_bytes(gzip.compress(header.tobuf(format=tarfile.GNU_FORMAT)))
    with pytest.raises(ValueError, match="max_file_bytes"):
        await artifacts.extract_outputs(source, tmp_path / "results", prefix="output", limits=ArchiveLimits())
    assert not (tmp_path / "results").exists()


async def test_valid_nested_binary_output_replaces_previous_tree(tmp_path):
    data = bytes(range(256)) * 4096
    source = archive(tmp_path / "archive.gz", [("output/nested/report.docx", data)])
    destination = tmp_path / "results"
    (destination / "output").mkdir(parents=True)
    (destination / "output/stale").write_text("old")
    await artifacts.extract_outputs(source, destination, prefix="output", limits=ArchiveLimits())
    assert (destination / "output/nested/report.docx").read_bytes() == data
    assert not (destination / "output/stale").exists()
    assert not list(tmp_path.glob(".lab-extract-*"))


async def test_source_link_is_rejected(tmp_path):
    original = archive(tmp_path / "original.gz", [("output/a", b"ok")])
    source = tmp_path / "link.gz"
    source.symlink_to(original)
    with pytest.raises(ValueError):
        await artifacts.extract_outputs(source, tmp_path / "results", prefix="output", limits=ArchiveLimits())
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancellation"])
async def test_worker_is_reaped_and_staging_removed(tmp_path, monkeypatch, cancel):
    script = tmp_path / "sleep.py"
    pidfile = tmp_path / "worker.pid"
    script.write_text(f"import os,time\nopen({str(pidfile)!r},'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    monkeypatch.setattr(artifacts, "_WORKER", script)
    source = archive(tmp_path / "archive.gz", [("output/a", b"ok")])
    task = asyncio.create_task(
        artifacts.extract_outputs(
            source, tmp_path / "results", prefix="output", limits=ArchiveLimits(timeout_seconds=5 if cancel else 1)
        )
    )
    async with asyncio.timeout(4):
        while not pidfile.exists():
            await asyncio.sleep(0.01)
    pid = int(pidfile.read_text())
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert not psutil.pid_exists(pid)
    assert not (tmp_path / "results").exists()
    assert not list(tmp_path.glob(".lab-extract-*"))


async def test_parent_enforces_resident_memory_limit(tmp_path, monkeypatch):
    script = tmp_path / "memory.py"
    pidfile = tmp_path / "worker.pid"
    script.write_text(
        f"import os,time\nopen({str(pidfile)!r},'w').write(str(os.getpid()))\n"
        "data=bytearray(64*1024*1024)\ntime.sleep(60)\n"
    )
    monkeypatch.setattr(artifacts, "_WORKER", script)
    with pytest.raises(RuntimeError, match="worker_memory_bytes"):
        await artifacts._run_worker({}, limits=ArchiveLimits(worker_memory_bytes=32 * 1024 * 1024, timeout_seconds=5))
    assert not psutil.pid_exists(int(pidfile.read_text()))


@pytest.mark.parametrize("oversized", [False, True])
async def test_download_worker_file_limit_and_atomic_publication(tmp_path, monkeypatch, oversized):
    # Exercise the real worker's OS file-size cap with a tiny stand-in transfer.
    script = tmp_path / "transfer.py"
    root = str(Path(artifacts.__file__).parents[2])
    script.write_text(
        f"import sys\nsys.path.insert(0,{root!r})\n"
        "from pathlib import Path\nfrom resources_servers.legal_agent_bench import archive_worker as w\n"
        "async def transfer(request):\n"
        f"    Path(request['target']).write_bytes(b'x'*{1025 if oversized else 1024})\n"
        "w._download_archive=transfer\nw._main()\n"
    )
    monkeypatch.setattr(artifacts, "_WORKER", script)
    owner = SimpleNamespace(serialize=AsyncMock(return_value={"id": "owner"}))
    destination = tmp_path / "retained.gz"
    destination.write_bytes(b"previous")
    call = artifacts.download_archive(
        owner, "/remote.gz", destination, provider_config={}, limits=ArchiveLimits(max_archive_bytes=1024)
    )
    if oversized:
        with pytest.raises(ValueError, match="worker exited"):
            await call
        assert destination.read_bytes() == b"previous"
    else:
        await call
        assert destination.read_bytes() == b"x" * 1024
    assert not list(tmp_path.glob(".lab-download-*"))


async def test_download_uses_borrower_disconnect_even_on_transfer_failure(tmp_path, monkeypatch):
    import nemo_gym.sandbox as sandbox

    provider = SimpleNamespace(aclose=AsyncMock())
    borrower = SimpleNamespace(
        download=AsyncMock(side_effect=OSError("transfer failed")), disconnect=AsyncMock(), stop=AsyncMock()
    )
    connect = AsyncMock(return_value=borrower)
    monkeypatch.setattr(sandbox, "create_provider", lambda config: provider)
    monkeypatch.setattr(sandbox.AsyncSandbox, "connect", connect)
    with pytest.raises(OSError, match="transfer failed"):
        await archive_worker._download_archive(
            {
                "provider": {},
                "descriptor": {"id": "owner"},
                "remote_path": "/archive.gz",
                "target": str(tmp_path / "a.gz"),
            }
        )
    connect.assert_awaited_once_with({"id": "owner"}, provider=provider)
    borrower.disconnect.assert_awaited_once()
    borrower.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "field,value",
    [("max_entries", 0), ("max_file_bytes", -1), ("timeout_seconds", float("inf")), ("max_archive_bytes", True)],
)
def test_invalid_limits_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(ArchiveLimits(), **{field: value})


async def test_download_credentials_use_stdin_and_are_not_returned_in_errors(tmp_path, monkeypatch):
    script = tmp_path / "failed-transfer.py"
    marker = tmp_path / "stdin-verified"
    script.write_text(
        "import json,sys\nrequest=json.load(sys.stdin)\n"
        "secret=request['provider']['test']['api_key']\n"
        "assert secret not in ' '.join(sys.argv)\n"
        f"open({str(marker)!r}, 'w').write('checked')\n"
        "print(secret,file=sys.stderr)\nsys.exit(1)\n"
    )
    monkeypatch.setattr(artifacts, "_WORKER", script)
    owner = SimpleNamespace(serialize=AsyncMock(return_value={"id": "owner"}))
    with pytest.raises(ValueError, match="download worker exited") as error:
        await artifacts.download_archive(
            owner,
            "/remote.gz",
            tmp_path / "output.gz",
            provider_config={"test": {"api_key": "test-secret-do-not-log"}},
            limits=ArchiveLimits(),
        )
    assert marker.read_text() == "checked"
    assert "test-secret-do-not-log" not in str(error.value)
    assert not (tmp_path / "output.gz").exists()
