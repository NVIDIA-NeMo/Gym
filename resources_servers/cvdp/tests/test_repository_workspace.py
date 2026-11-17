# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from resources_servers.cvdp.repository_workspace import (
    apply_changes,
    collect_changes,
    extract_snapshot,
    resolve_archive,
    snapshot,
)
from resources_servers.cvdp.scripts.prepare_heavy import git, prepare


def test_project_replay_preserves_assets_deletions_links_and_modes(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    (before / "src").mkdir()
    (before / "src/core.v").write_text("buggy")
    (before / "old.v").write_text("remove")
    (before / "rom.bin").write_bytes(b"\x00\xff")
    (before / "link").symlink_to("src/core.v")
    shutil.copytree(before, after, symlinks=True)
    initial = snapshot(before)
    (after / "src/core.v").write_text("fixed")
    (after / "old.v").unlink()
    (after / "rom.bin").write_bytes(b"\xff\x00")
    (after / "run.sh").write_text("#!/bin/sh\ntrue\n")
    (after / "run.sh").chmod(0o755)
    (after / "new_root").mkdir()
    (after / "new_root/new.v").write_text("new module")
    (after / ".home").mkdir()
    (after / ".home/secret").write_text("agent runtime")
    (after / "rundir").mkdir()
    (after / "rundir/sim.log").write_text("not project output")
    delta = tmp_path / "changes.tar.gz"
    changes = collect_changes(after, initial, delta)
    changes["archive"] = base64.b64encode(delta.read_bytes()).decode()
    assert "old.v" in changes["deleted"]
    apply_changes(before, changes)
    assert snapshot(before) == snapshot(after)
    assert (before / "rom.bin").read_bytes() == b"\xff\x00"
    assert os.access(before / "run.sh", os.X_OK)
    assert (before / "link").is_symlink()
    assert not (before / ".home").exists()
    assert not (before / "rundir").exists()


@pytest.mark.parametrize("name", ["../escaped", "/tmp/escaped", ".cvdp_harness/src/test.py", "rundir/test.py"])
def test_rejects_delta_outside_project(tmp_path: Path, name: str) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.size = 4
        tar.addfile(member, io.BytesIO(b"evil"))
    with pytest.raises(ValueError):
        apply_changes(tmp_path, {"archive": base64.b64encode(raw.getvalue()).decode(), "deleted": []})


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        tar.addfile(member)
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(raw.getvalue())
    with pytest.raises(tarfile.FilterError):
        extract_snapshot(archive, tmp_path / "workspace")


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for bundle preparation")
def test_prepares_sanitized_project_without_reference_patch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    git("init", str(repo))
    git("-C", str(repo), "config", "user.email", "test@example.invalid")
    git("-C", str(repo), "config", "user.name", "Test")
    (repo / "external").mkdir()
    (repo / "external/core.v").write_text("starting buggy RTL")
    (repo / "external/rom.bin").write_bytes(b"\x00\xff")
    (repo / "external/run.sh").write_text("#!/bin/sh\ntrue\n")
    (repo / "external/run.sh").chmod(0o755)
    git("-C", str(repo), "add", ".")
    git("-C", str(repo), "commit", "-m", "context")
    commit = git("-C", str(repo), "rev-parse", "HEAD")
    git("-C", str(repo), "branch", "context-test")
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    git("-C", str(repo), "bundle", "create", str(bundles / "cvdp_heavy_test.bundle"), "context-test")
    (bundles / "manifest.txt").write_text(
        f"    id: task\n    original_commit: {'0' * 40}\n    sanitized_commit: {commit}\n"
    )
    row = {
        "id": "task",
        "context": {"repo": "cvdp_heavy_test", "commit": "0" * 40},
        "prompt": "Repair the module",
        "patch": {"core.v": "REFERENCE ANSWER"},
        "harness": {"docker-compose.yml": "services: {}"},
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = prepare(source, bundles, tmp_path / "snapshots", tmp_path / "gym.jsonl", tmp_path / "cache")
    converted = json.loads(output.read_text())
    digest = converted["verifier_metadata"]["repository_workspace_sha256"]
    archive = resolve_archive(str(tmp_path / "snapshots"), digest)
    extracted = tmp_path / "extracted"
    extract_snapshot(archive, extracted)
    assert (extracted / "core.v").read_text() == "starting buggy RTL"
    assert (extracted / "rom.bin").read_bytes() == b"\x00\xff"
    assert (extracted / "run.sh").stat().st_mode & 0o111
    assert "REFERENCE ANSWER" not in output.read_text()
    assert converted["responses_create_params"]["input"] == [{"role": "user", "content": row["prompt"]}]
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_archive(str(tmp_path / "snapshots"), digest)


@pytest.mark.asyncio
async def test_verifier_keeps_project_src_separate_from_hidden_harness(tmp_path: Path) -> None:
    import hashlib

    from resources_servers.cvdp.app import CVDPResourcesServerConfig
    from resources_servers.cvdp.testbench_runner import TestbenchRunner

    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src/core.v").write_text("starting RTL")
    (project / "old.v").write_text("old")
    archives = tmp_path / "snapshots"
    archives.mkdir()
    archive = archives / "starting.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(project / "src", arcname="src")
        stream.add(project / "old.v", arcname="old.v")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.rename(archives / f"{digest}.tar.gz")
    initial = snapshot(project)
    (project / "src/core.v").write_text("repaired RTL")
    (project / "old.v").unlink()
    delta = tmp_path / "changes.tar.gz"
    changes = collect_changes(project, initial, delta)
    changes["archive"] = base64.b64encode(delta.read_bytes()).decode()
    config = CVDPResourcesServerConfig(
        name="cvdp", entrypoint="app.py", host="127.0.0.1", port=0, prepared_repository_dir=str(archives)
    )
    runner = TestbenchRunner(config)

    async def service(workspace_archive, name, compose, harness_files, *, repository_workspace=False):
        assert repository_workspace
        with tarfile.open(workspace_archive, "r:gz") as stream:
            assert stream.extractfile("src/core.v").read() == b"repaired RTL"
            assert stream.extractfile(".cvdp_harness/src/test.py").read() == b"hidden test"
            assert "old.v" not in stream.getnames()
        return 0, "original harness passed"

    runner._run_service = service
    result = await runner.run(
        rtl_files={},
        harness_files={
            "docker-compose.yml": "services:\n  sanity:\n    image: sim\n    command: pytest /src/test.py\n",
            "src/test.py": "hidden test",
        },
        task_id="test",
        repository_workspace_sha256=digest,
        repository_changes=changes,
    )
    assert result[0] == 0


def test_snapshot_directory_is_relative_to_repository_not_server_cwd(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    from resources_servers.cvdp import repository_workspace

    prepared = tmp_path / "data/snapshots"
    prepared.mkdir(parents=True)
    digest = hashlib.sha256(b"snapshot").hexdigest()
    expected = prepared / f"{digest}.tar.gz"
    expected.write_bytes(b"snapshot")
    component = tmp_path / "component"
    component.mkdir()
    monkeypatch.setattr(repository_workspace, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(component)
    assert resolve_archive("data/snapshots", digest) == expected
