# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from resources_servers.terminal_bench_4.lifecycle import Session, discard_workspace_snapshot


def make_session(tmp_path):
    directory = tmp_path / "tb4-test"
    (directory / "artifacts/app/sub").mkdir(parents=True)
    (directory / "artifacts/app/sub/big.bin").write_bytes(b"x" * 10)
    (directory / "artifacts/app/note.txt").write_text("n")
    (directory / "artifacts/logs/artifacts").mkdir(parents=True)
    (directory / "artifact-metadata").mkdir()
    (directory / "artifact-metadata/meta.json").write_text("{}")
    (directory / "verifier").mkdir()
    (directory / "verifier/reward.txt").write_text("1")
    (directory / "agent").mkdir()
    (directory / "agent/log.txt").write_text("l")
    (directory / "result.json").write_text("{}")
    return Session("identity", "owner", None, "tb4-test", directory)


def test_discard_removes_only_the_workspace_snapshot(tmp_path):
    session = make_session(tmp_path)
    asyncio.run(discard_workspace_snapshot(session))
    assert not (session.directory / "artifacts").exists()
    assert session.diagnostics == [{"operation": "workspace_snapshot_discarded", "files": 2}]
    for kept in ("artifact-metadata/meta.json", "verifier/reward.txt", "agent/log.txt", "result.json"):
        assert (session.directory / kept).exists()


def test_discard_without_snapshot_is_a_no_op(tmp_path):
    directory = tmp_path / "tb4-none"
    directory.mkdir()
    session = Session("identity", "owner", None, "tb4-none", directory)
    asyncio.run(discard_workspace_snapshot(session))
    assert session.diagnostics == []
