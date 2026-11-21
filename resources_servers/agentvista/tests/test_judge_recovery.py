# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge-only recovery retains the same answer and reuses only measured verdicts."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.agentvista.app import AgentVistaResources, AgentVistaResourcesConfig
from responses_api_agents.agentvista_agent.runtime.results import AgentVistaVerifyRequest


@pytest.fixture
def server(tmp_path):
    return AgentVistaResources(
        config=AgentVistaResourcesConfig(
            host="127.0.0.1",
            port=1234,
            entrypoint="app.py",
            name="verifier",
            source_root=str(tmp_path),
            dataset="unused",
            image_folder=str(tmp_path),
            output_root=str(tmp_path),
            # Generous, so a slow runner never turns a judge run into an unintended timeout. The timeout case
            # bounds only the run it means to time out. The timeout is part of the receipt identity, so it
            # cannot change between attempts.
            verifier_timeout=60,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def verify_request():
    return AgentVistaVerifyRequest(
        responses_create_params={"input": "Act"},
        verifier_metadata={"task_id": "task", "category_id": "fixture", "source_sha256": "hash"},
        response={
            "id": "resp_saved_answer",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "agentvista_result": {
                "artifacts": "saved-agent-output",
                "verification_pending": True,
                "mask_sample": True,
                "judge_responses": ["unchanged saved answer"],
            },
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "timeout", "exit", "invalid_json", "cancel"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_same_saved_answer_recovers_then_survives_restart(server, monkeypatch, failure, reward):
    spawn = asyncio.create_subprocess_exec
    calls = []
    processes = []
    body = verify_request()
    status = "PASS" if reward else "FAIL"

    async def worker(*args, **kwargs):
        path = Path(args[-1])
        config = json.loads(path.read_text())
        calls.append(config)
        assert config["result"]["judge_responses"] == ["unchanged saved answer"]
        if len(calls) == 1:
            if failure in {"timeout", "cancel"}:
                script = "import time; time.sleep(60)"
            elif failure == "exit":
                script = "raise SystemExit(7)"
            elif failure == "invalid_json":
                script = "import pathlib,sys; pathlib.Path(sys.argv[1]).with_name('result.json').write_text('{')"
            else:
                script = (
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).with_name('result.json').write_text("
                    '\'{"mask_sample":true,"error":"judge unavailable"}\')'
                )
        else:
            script = (
                "import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); r=json.loads(p.read_text())['result']; "
                "r.pop('verification_pending'); r.update(mask_sample=False,verify="
                + repr({"status": status, "reward": reward})
                + "); p.with_name('result.json').write_text(json.dumps(r))"
            )
        process = await spawn(sys.executable, "-c", script, str(path), **kwargs)
        processes.append(process)
        return process

    wait_for = asyncio.wait_for

    async def first_attempt_times_out(awaitable, timeout):
        return await wait_for(awaitable, 1 if len(calls) == 1 else timeout)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", worker)
    if failure == "timeout":
        monkeypatch.setattr(asyncio, "wait_for", first_attempt_times_out)
    verify = judge_failsafe(server.verify)
    if failure == "cancel":
        task = asyncio.create_task(verify(body))
        async with asyncio.timeout(5):
            while not processes:
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        first = await verify(body)
        failed = json.loads(first.body)
        assert failed["mask_sample"] and failed["failure_kind"] == "judge_failed"
        assert failed["response"] == body.response.model_dump(mode="json")
        assert ".failed-" in failed["failure_reason"]
        expected_cause = {
            "provider": "judge unavailable",
            "timeout": "TimeoutError",
            "exit": "exited 7",
            "invalid_json": "JSONDecodeError",
        }[failure]
        assert expected_cause in failed["failure_reason"]
    assert processes[0].returncode is not None
    assert server._inflight == {}
    root = Path(server.config.output_root) / "verification"
    archives = list(root.glob("*.failed-*"))
    assert len(archives) == 1 and (archives[0] / "input.json").exists()

    # Regrade the identical body concurrently: no new rollout ID or agent episode.
    recovered, duplicate = await asyncio.gather(verify(body), verify(body))
    assert not recovered.mask_sample and recovered.reward == duplicate.reward == reward
    assert recovered.verify_status == status
    assert len(calls) == 2 and calls[0] == calls[1]
    restarted = type(server)(config=server.config, server_client=server.server_client)
    assert (await restarted.verify(body)).model_dump() == recovered.model_dump()
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt",
    [
        None,
        "{",
        "null",
        '{"mask_sample":true,"error":"old outage"}',
        '{"mask_sample":false,"verify":{"status":"PASS","reward":[]}}',
    ],
)
async def test_preexisting_failed_or_incomplete_receipt_is_archived_before_retry(server, monkeypatch, receipt):
    root = Path(server.config.output_root) / "verification"
    output = root / "old-attempt"
    output.mkdir(parents=True)
    (output / "input.json").write_text("{}")
    if receipt is not None:
        (output / "result.json").write_text(receipt)
    spawn = asyncio.create_subprocess_exec
    calls = []

    async def worker(*args, **kwargs):
        calls.append(args)
        return await spawn(sys.executable, "-c", "raise SystemExit(7)", **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", worker)
    with pytest.raises(JudgeError, match="exited 7"):
        await server._judge({"result": {"verification_pending": True}}, "old-attempt")
    assert len(calls) == 1
    assert not output.exists()
    assert len(list(root.glob("old-attempt.failed-*"))) == 2
