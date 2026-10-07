# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.swe_together import snapshot_worker
from resources_servers.swe_together.artifacts import RepositorySnapshots


def git_state(repo):
    return {
        name: (repo / ".git" / name).read_bytes() if (repo / ".git" / name).exists() else None
        for name in ["HEAD", "index", "config"]
    }


def test_unborn_parent_preserves_prepared_files_and_nested_repository(tmp_path, monkeypatch):
    repo = tmp_path / "prepared"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    for directory in [repo, nested]:
        subprocess.run(["git", "init", str(directory)], check=True, capture_output=True)
    (repo / "prepared.txt").write_text("untracked prepared baseline\n")
    (repo / "ignored.txt").write_text("prepared ignored file\n")
    (repo / ".git/info/exclude").write_text("ignored.txt\n")
    (nested / "source.txt").write_text("nested baseline\n")
    snapshot_worker.git(str(nested), "add", "-A")
    snapshot_worker.git(
        str(nested), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "nested"
    )
    replay = tmp_path / "replay"
    shutil.copytree(repo, replay)
    before = {str(path): git_state(path) for path in [repo, nested]}
    # A private index alone does not resolve read-only object/ref storage.
    for path in [repo / ".git", repo / ".git/objects", repo / ".git/refs"]:
        path.chmod(0o555)
    stores = tmp_path / "stores"
    stores.mkdir()
    monkeypatch.setattr(snapshot_worker.tempfile, "gettempdir", lambda: str(stores))
    monkeypatch.setattr(snapshot_worker, "ROOTS", [str(repo)])
    monkeypatch.delenv("HARBOR_REPO_PATHS", raising=False)
    request_path, output_path = tmp_path / "request.json", tmp_path / "output.json"
    request = {"namespace": "refs/nemo-gym/unborn-test"}
    monkeypatch.setattr(snapshot_worker.sys, "argv", ["snapshot", str(request_path), str(output_path)])

    def capture():
        request_path.write_text(json.dumps(request))
        snapshot_worker.main()
        return json.loads(output_path.read_text())

    baseline = capture()
    assert set(baseline["trees"]) == {str(repo), str(nested)}
    assert all(not row["cumulative"] for row in baseline["repositories"].values())
    assert {str(path): git_state(path) for path in [repo, nested]} == before
    assert (repo / ".git").stat().st_mode & 0o777 == 0o555
    assert not (repo / ".git/refs/nemo-gym").exists()
    request.update(baseline=baseline["trees"], previous=baseline["trees"])
    (repo / "prepared.txt").write_text("candidate update\n")
    (repo / "new.txt").write_text("candidate new file\n")
    (repo / "ignored.txt").write_text("still excluded\n")
    (nested / "source.txt").write_text("nested candidate update\n")
    final = capture()
    for original, destination in [(repo, replay), (nested, replay / "nested")]:
        patch = final["repositories"][str(original)]["binary"]
        applied = subprocess.run(
            ["git", "-C", str(destination), "apply", "--whitespace=nowarn", "-"],
            input=patch.encode(),
            capture_output=True,
        )
        assert applied.returncode == 0, applied.stderr
    assert (replay / "prepared.txt").read_text() == "candidate update\n"
    assert (replay / "new.txt").read_text() == "candidate new file\n"
    assert (replay / "nested/source.txt").read_text() == "nested candidate update\n"
    assert "ignored.txt" not in final["repositories"][str(repo)]["binary"]
    assert {str(path): git_state(path) for path in [repo, nested]} == before

    # A first candidate commit must not switch away from the original store.
    for path in [repo / ".git", repo / ".git/objects", repo / ".git/refs"]:
        path.chmod(0o755)
    snapshot_worker.git(str(repo), "add", "-A")
    snapshot_worker.git(
        str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "first"
    )
    committed = git_state(repo)
    assert capture() == final
    assert git_state(repo) == committed


@pytest.mark.asyncio
async def test_snapshot_failure_preserves_merged_provider_diagnostics(tmp_path):
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=1,
                error_type="command_failed",
                stdout="RuntimeError: git read-tree failed: Not a valid object name HEAD",
                stderr="exit status 1",
            )
        ),
    )
    snapshots = RepositorySnapshots(tmp_path)
    with pytest.raises(RuntimeError, match="Not a valid object name HEAD"):
        await snapshots.capture(sandbox)
    failure = json.loads((tmp_path / "snapshot-error.json").read_text())
    assert failure["return_code"] == 1 and failure["turn"] is None
    assert failure["error_type"] == "command_failed"
    assert "Not a valid object name HEAD" in failure["stdout"]
    assert "rm -f" in sandbox.exec.call_args.args[0]
