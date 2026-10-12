# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import stat
import tarfile
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.apex_agent import prebuilt_world_entrypoint as entrypoint
from responses_api_agents.apex_agent.tests.test_app import _agent, _body


def _roots(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ("filesystem", ".apps_data")}
    monkeypatch.setattr(entrypoint, "RESTORE_ROOTS", roots)
    monkeypatch.setattr(entrypoint, "OUTPUT", tmp_path / "output")
    return roots


def test_restore_preserves_work_and_uses_database_dump_instead_of_seed_csv(tmp_path, monkeypatch):
    roots = _roots(tmp_path, monkeypatch)
    roots["filesystem"].mkdir()
    (roots["filesystem"] / "baseline-only.txt").write_text("stale")
    snapshot = tmp_path / "world.zip"
    with zipfile.ZipFile(snapshot, "w") as archive:
        archive.writestr("filesystem/report.txt", "work from turn 12")
        archive.writestr(".apps_data/erpnext/database_dump.sql", "updated database")
        archive.writestr(".apps_data/erpnext/seed.csv", "original seed")
    entrypoint.restore_world_snapshot(snapshot)
    assert (roots["filesystem"] / "report.txt").read_text() == "work from turn 12"
    assert not (roots["filesystem"] / "baseline-only.txt").exists()
    assert (roots[".apps_data"] / "erpnext/database_dump.sql").read_text() == "updated database"
    entrypoint.hide_seed_csvs()
    assert not (roots[".apps_data"] / "erpnext/seed.csv").exists()
    entrypoint.restore_seed_csvs()
    assert (roots[".apps_data"] / "erpnext/seed.csv").read_text() == "original seed"


@pytest.mark.parametrize("name", ["../escape", "/escape", ".apps_data/../escape", "other/file"])
def test_invalid_checkpoint_archive_is_rejected_before_clearing_world(tmp_path, monkeypatch, name):
    roots = _roots(tmp_path, monkeypatch)
    roots["filesystem"].mkdir()
    marker = roots["filesystem"] / "keep.txt"
    marker.write_text("keep")
    source = tmp_path / "bad.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="unsafe resume"):
        entrypoint.restore_world_snapshot(source)
    assert marker.read_text() == "keep"


def test_world_restore_runs_after_reset_before_app_start_and_ownership(tmp_path, monkeypatch):
    script = tmp_path / "start.sh"
    script.write_text('reset world\n    echo "Task files setup complete"\nstart database\nPOPULATE_PIDS=""\n')
    monkeypatch.setattr(entrypoint, "WORLD_START_SCRIPT", script)
    data = tmp_path / "postgres"
    data.mkdir()
    patched = entrypoint.prepare_world_start_script(
        [{"path": str(data), "user": "svc", "group": "apps"}],
        tmp_path / "patched.sh",
        resume_world=tmp_path / "checkpoint.zip",
    ).read_text()
    assert patched.index("reset world") < patched.index("--restore-world")
    assert patched.index("--restore-world") < patched.index("fix_permissions") < patched.index("chown -R")
    assert patched.index("chown -R") < patched.index("start database")
    assert patched.index("start database") < patched.index("--hide-restore-seeds") < patched.index('POPULATE_PIDS=""')


def test_snapshot_requests_live_database_export_hooks(tmp_path, monkeypatch):
    hooks = tmp_path / "snapshot_hooks.json"
    payload = {"pre_snapshot_hooks": [{"name": "erpnext", "command": "export database"}]}
    hooks.write_text(json.dumps(payload))
    monkeypatch.setattr(entrypoint, "SNAPSHOT_HOOKS_PATH", hooks)
    requests = []

    def open_request(request, timeout):
        requests.append(request)
        assert timeout == 600
        return io.BytesIO(b"snapshot tar")

    monkeypatch.setattr(entrypoint.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(entrypoint, "snapshot_tar_to_zip", lambda source, destination: ["filesystem/report.txt"])
    entrypoint.capture_snapshot(tmp_path / "world.zip", run_hooks=True)
    assert json.loads(requests[0].data) == payload


def test_snapshot_roundtrip_preserves_executable_helpers_and_empty_directories(tmp_path, monkeypatch):
    roots = _roots(tmp_path, monkeypatch)
    source = tmp_path / "snapshot.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        script = tarfile.TarInfo("filesystem/helper.sh")
        script.mode = 0o755
        content = b"#!/bin/sh\necho continuing\n"
        script.size = len(content)
        archive.addfile(script, io.BytesIO(content))
        empty = tarfile.TarInfo("filesystem/empty/")
        empty.type = tarfile.DIRTYPE
        empty.mode = 0o755
        archive.addfile(empty)
    checkpoint = tmp_path / "world.zip"
    entrypoint.snapshot_tar_to_zip(source, checkpoint)
    entrypoint.restore_world_snapshot(checkpoint)
    assert (roots["filesystem"] / "helper.sh").stat().st_mode & stat.S_IXUSR
    assert (roots["filesystem"] / "empty").is_dir()


def test_prebuilt_run_uses_stable_checkpoint_identity_when_enabled(tmp_path):
    agent = _agent(resume_checkpoint_dir=str(tmp_path / "checkpoints"))
    agent.config.prebuilt_resume_enabled = True
    body = _body()
    body.runtime_mode = "prebuilt_world"
    body.__pydantic_extra__.update(_ng_task_index=8, _ng_rollout_index=2)
    expected = agent._resume_checkpoint_dir(body)
    agent._run_rollout = AsyncMock(return_value="result")
    assert asyncio.run(agent.run(SimpleNamespace(), body)) == "result"
    assert agent._run_rollout.call_args.args[-1] == expected


def test_prebuilt_sandbox_binds_checkpoint_and_passes_resume_permission(tmp_path):
    agent = _agent(resume_checkpoint_dir=str(tmp_path / "checkpoints"))
    agent.config.prebuilt_resume_enabled = True
    body = _body()
    agent._prebuilt_worlds = {body.world_id: {"startup_ownership": []}}
    directory = tmp_path / "checkpoint"
    spec = agent._sandbox_spec(
        body,
        "instruction",
        prebuilt_world=True,
        resume_dir=directory,
        resume_checkpoint=SimpleNamespace(),
    )
    assert f"{directory}:/checkpoint" in spec.provider_options["binds"]
    config = json.loads(next(value for name, value in spec.files.items() if name.endswith("runner_config.json")))
    assert config["resume_checkpoint_dir"] == "/checkpoint"
    assert config["resume_allowed"] is True
