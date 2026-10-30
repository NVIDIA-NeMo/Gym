# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task backend's typed session preserves its protocol and its cleanup ownership."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.core import EpisodeId, TaskId
from resources_servers.agentic_if.episode import resources
from resources_servers.agentic_if.tests.support import Transport, response, seeded


class Remote:
    global_config_dict = {"sandbox": {"opensandbox": {}}}

    def __init__(self, seed, *, verify_error=False):
        self.seed = seed
        self.calls = []
        self.verify_error = verify_error

    async def post(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if kwargs["url_path"] == "/verify" and self.verify_error:
            raise OSError("resource unavailable")
        body = self.seed if kwargs["url_path"] == "/seed_session" else {"reward": 0.25, "evaluation_completed": True}

        async def read():
            return json.dumps(body).encode()

        return SimpleNamespace(read=read, cookies={"resource-session": "owned"}, status=200, ok=True)


@pytest.fixture
def transport(monkeypatch):
    sandbox = Transport()
    provider = SimpleNamespace(aclose=None)

    async def connect(descriptor, *, provider):
        sandbox.descriptor = descriptor
        return sandbox

    monkeypatch.setattr(resources, "create_provider", lambda config: provider)
    monkeypatch.setattr(resources.AsyncSandbox, "connect", connect)
    return sandbox


def row():
    return {"instance_id": "task-a", "agentic_if": {"checks": "private"}}


PARAMS = {"input": [{"role": "user", "content": "Fix it"}]}


def session(remote, *, task_row=None):
    return resources.SandboxTaskSession(
        client=remote,
        server_name="tasks",
        row=task_row or row(),
        client_session_id="attempt-a",
        timeout_s=60,
        episode_id=EpisodeId(rollout_id="test", attempt=0),
        task_id=TaskId(taskset="agentic_if", task_id="variant-a"),
    )


@pytest.mark.asyncio
async def test_resource_identity_and_cookies_survive_verification_then_owned_stop(transport):
    remote = Remote(seeded(workdir="/app/project"))
    task = session(remote, task_row=row() | {"agent_user": "1001"})
    await task.prepare()
    await task.sandbox.exec(command="pwd", timeout_s=30)
    command = transport.calls[-1]
    assert command["user"] == 1001
    assert command["cwd"] == "/app/project"
    assert command["env"] == {"HOME": "/home/agent", "SHELL": "/bin/bash"}
    result = await task.finish(response(), params=PARAMS)
    await task.close()
    assert result["reward"] == 0.25
    assert transport.stops == 0 and transport.disconnects == 1
    assert transport.descriptor == {"sandbox_id": "sandbox-a"}
    assert task.access.connection.provider_config_ref == "sandbox"
    verify = remote.calls[1]
    assert verify["cookies"] == {"resource-session": "owned"}
    # The backend's own flat verify body: its task fields, never the private IF configuration.
    assert verify["json"] == {
        "instance_id": "task-a",
        "agent_user": "1001",
        "responses_create_params": PARAMS,
        "response": verify["json"]["response"],
    }
    assert verify["json"]["response"]["id"] == "actual-response"
    seed = ResourcesSeedSessionRequest.model_validate(remote.calls[0]["json"])
    assert (seed.resources_session_id, seed.task_id.task_id) == ("attempt-a", "variant-a")
    assert seed.task_data == {"instance_id": "task-a", "agent_user": "1001"}
    assert remote.calls[-1]["url_path"] == "/close_session"


@pytest.mark.asyncio
async def test_row_workdir_wins_over_the_sandbox_default(transport):
    task = session(Remote(seeded(workdir="/image/default")), task_row=row() | {"workdir": "/required/repository"})
    await task.prepare()
    await task.sandbox.exec(command="pwd", timeout_s=30)
    assert transport.calls[-1]["cwd"] == "/required/repository"


@pytest.mark.asyncio
async def test_verification_failure_still_releases_the_correct_owned_resource(transport):
    task = session(Remote(seeded(), verify_error=True))
    await task.prepare()
    with pytest.raises(OSError, match="resource unavailable"):
        await task.finish(response(), params=PARAMS)
    with pytest.raises(RuntimeError, match="only run once"):
        await task.finish(response(), params=PARAMS)
    await task.close()
    assert (transport.stops, transport.disconnects) == (0, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["not an object", {"resources_session_id": "attempt-a"}, "unknown provider"])
async def test_a_seed_reply_without_a_usable_sandbox_is_an_infrastructure_failure(transport, monkeypatch, reply):
    """Not the task's fault: a RuntimeError, which seed reports as a retryable 503 rather than an invalid task."""
    if reply == "unknown provider":
        monkeypatch.setattr(Remote, "global_config_dict", {})
        reply = seeded()
    remote = Remote(reply)
    task = session(remote)
    with pytest.raises(RuntimeError, match="tasks"):
        await task.prepare()
    await task.close()
    # Nothing was bound, and the backend's session is still released.
    assert task.sandbox is None and remote.calls[-1]["url_path"] == "/close_session"


@pytest.mark.asyncio
async def test_nonroot_identity_failure_cannot_start_harness_but_still_cleans_up(transport):
    transport.uid = "0"
    task = session(Remote(seeded()), task_row=row() | {"agent_user": "agent"})
    with pytest.raises(RuntimeError, match="uid 0"):
        await task.prepare()
    await task.close()
    assert transport.stops == 0 and transport.disconnects == 1


@pytest.mark.asyncio
async def test_a_failed_release_raises_and_close_can_be_retried(transport):
    remote = Remote(seeded())
    task = session(remote)
    await task.prepare()
    post = remote.post
    failures = [OSError("backend unavailable")]

    async def flaky(**kwargs):
        if kwargs["url_path"] == "/close_session" and failures:
            raise failures.pop()
        return await post(**kwargs)

    remote.post = flaky
    with pytest.raises(OSError):
        await task.close()
    assert transport.disconnects == 0
    await task.close()
    await task.close()
    assert transport.disconnects == 1
    assert [call["url_path"] for call in remote.calls].count("/close_session") == 1
