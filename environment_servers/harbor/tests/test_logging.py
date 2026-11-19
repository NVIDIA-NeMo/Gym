# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server's logging honours Gym's `--verbose` (`verbose: true`): the episode trace is DEBUG, quiet otherwise."""

import logging

from environment_servers.harbor.tests.conftest import make_server, run_episode


async def test_episode_trace_is_logged_at_debug_level_only(tmp_path, monkeypatch, caplog):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="environment_servers.harbor"):
        response = await run_episode(server, task)
    assert response.failure is None, response.failure
    records = [record for record in caplog.records if record.name.startswith("environment_servers.harbor")]
    messages = [record.getMessage() for record in records]
    seeded = next(message for message in messages if "seeded" in message)
    assert seeded.startswith("hello: episode ") and "workdir /app" in seeded and "shared verifier" in seeded
    verified = next(message for message in messages if "test.sh exited" in message)
    assert verified.startswith("hello: test.sh exited 0 after ")
    assert "rewards {'reward': 1.0}" in verified and response.result.verifier_logs_dir in verified
    assert any(message == f"hello: episode {response.episode_id.capture_key} scored 1.0" for message in messages)
    # The trace is for `--verbose` runs: none of it would reach a log at the default level.
    assert {record.levelno for record in records if "hello:" in record.getMessage()} == {logging.DEBUG}

    # Without `verbose`, the same episode logs nothing from this server.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="environment_servers.harbor"):
        await run_episode(server, task, rollout="r2")
    assert not [record for record in caplog.records if record.name.startswith("environment_servers.harbor")]
