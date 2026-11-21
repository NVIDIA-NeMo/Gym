# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pi sessions that resume their saved conversation across several activations."""

import asyncio
import base64
import json
import re
import sys
import time

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.agents import (
    AGENT_ACTIVATION_HEADER,
    AGENT_TIMEOUT_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentContinuationRequirements,
    AgentSeedSessionRequest,
)
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import MEDIA_DIRECTORY
from responses_api_agents.pi_agent.app import PiAgent
from responses_api_agents.pi_agent.tests.test_sandbox import (  # noqa: F401  (setup is a fixture)
    _http_server,
    _pinned_pi_package,
    _run_session_locally,
    close_body,
    image_part,
    png,
    seed,
    setup,
    staged,
    staged_attachments,
)


ROUTE = "/ng-rollout/pi-smoke-a2/v1/responses"


def continuation_seed(observations=("timing", "harness_tool_calls")) -> dict:
    request = seed().model_copy(
        update={"continuation": AgentContinuationRequirements(observations=list(observations))}
    )
    return request.model_dump(mode="json")


def activate(client, index: int, body: dict):
    return client.post(ROUTE, json=body, headers={AGENT_ACTIVATION_HEADER: str(index)})


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def root_invocations(observations: dict) -> list[tuple]:
    """Each root invocation's activation, status, error type and whether it reports a duration."""
    return [
        (r["activation_id"], r["status"], r["error_type"], r["duration_ms"] is not None)
        for r in observations["records"]
        if r["kind"] == "agent_invocation" and r["parent_invocation_id"] is None
    ]


def test_seed_declares_the_observations_pi_supports(setup):
    agent, _ = setup
    with TestClient(agent.setup_webserver()) as client:
        seeded = client.post("/v1/agent_sessions", json=continuation_seed())
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["capabilities"]["mode"] == "resume_harness_session"
        assert set(seeded.json()["capabilities"]["observations"]) == {
            "timing",
            "harness_tool_calls",
            "reasoning",
            "compaction",
        }
        # Pi builds no trajectory record, so a seed that requires one is refused before any setup.
        refused = client.post("/v1/agent_sessions", json=continuation_seed(observations=("trajectory",)))
        assert refused.status_code == 422


def test_later_activation_resumes_the_saved_conversation_with_its_own_images(setup):
    agent, sandbox = setup
    runs = []
    run = sandbox.run

    async def record_run(command, **kwargs):
        result = await run(command, **kwargs)
        runs.append(
            staged(sandbox) | {"models": json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])}
        )
        return result

    sandbox.launch.side_effect = record_run
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=continuation_seed()).json()["agent_session_id"]
        first = activate(
            client,
            1,
            {
                "input": [
                    {"role": "system", "content": "Answer in one word."},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Remember this."}, image_part(png(0))],
                    },
                ]
            },
        )
        assert first.status_code == 200, first.text
        second = activate(
            client,
            2,
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "Compare."}, image_part(png(255))]}
                ]
            },
        )
        assert second.status_code == 200, second.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text

    first_run, second_run = runs
    # Both runs save to and resume the same Pi session; neither discards it.
    for name in ("--session-dir", "--session-id"):
        assert option(first_run["command"], name) == option(second_run["command"], name)
    assert "--no-session" not in first_run["command"] + second_run["command"]
    # Pi does not save its system prompt, so the second run passes the first activation's again.
    assert option(second_run["command"], "--append-system-prompt") == "Answer in one word."
    assert [first_run["prompt"], second_run["prompt"]] == ["Remember this.", "Compare."]
    # Each activation stages and attaches only its own images.
    assert staged_attachments(sandbox, first_run) == [png(0)]
    assert staged_attachments(sandbox, second_run) == [png(255)]
    assert all(arg.startswith(f"@{MEDIA_DIRECTORY}/") for arg in second_run["command"] if arg.startswith("@"))
    # Each activation gets its own identity, in the response, the observations and the model calls' header.
    invocation_ids = [f"{session_id}:activation-1", f"{session_id}:activation-2"]
    assert [run["models"]["providers"]["nemo"]["headers"]["x-session-id"] for run in runs] == invocation_ids
    # Every activation's calls still use the episode's rollout route, which the model server's capture requires.
    assert {run["models"]["providers"]["nemo"]["baseUrl"] for run in runs} == {
        agent.resolve_model_base_url("policy", "pi-smoke-a2")
    }
    assert first.json()["metadata"]["activation_id"] == "1"
    assert second.json()["metadata"]["activation_id"] == "2"
    harness_session_id = option(first_run["command"], "--session-id")
    assert second.json()["metadata"]["harness_session_id"] == harness_session_id
    records = closed.json()["agent_observations"]["records"]
    invocations = [record for record in records if record["kind"] == "agent_invocation"]
    assert [(r["invocation_id"], r["activation_id"], r["harness_session_id"]) for r in invocations] == [
        (invocation_ids[0], 1, harness_session_id),
        (invocation_ids[1], 2, harness_session_id),
    ]
    assert {r["invocation_id"] for r in records if r["kind"] == "tool_call"} == set(invocation_ids)
    assert closed.json()["cleanup_confirmed"] is True
    # Each activation's response carries that activation's own observations.
    assert root_invocations(first.json()["_ng_agent_observations"]) == [(1, "completed", None, True)]
    assert root_invocations(second.json()["_ng_agent_observations"]) == [(2, "completed", None, True)]


def test_activation_decodes_its_images_once(setup, monkeypatch):
    agent, _ = setup
    validated = []
    validate = PiAgent._validate_request
    monkeypatch.setattr(
        PiAgent, "_validate_request", lambda self, body: validated.append(body) or validate(self, body)
    )
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=continuation_seed())
        image = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Look."}, image_part(png(0))]}]}
        assert activate(client, 1, image).status_code == 200
    assert len(validated) == 1


def test_image_input_stays_declared_after_a_turn_with_images(setup):
    agent, sandbox = setup
    declared = []
    run = sandbox.run

    async def record_run(command, **kwargs):
        result = await run(command, **kwargs)
        models = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])
        declared.append(models["providers"]["nemo"]["models"][0]["input"])
        return result

    sandbox.launch.side_effect = record_run
    text = {"role": "user", "content": "Describe it."}
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=continuation_seed())
        for index, content in enumerate(["Hello.", [image_part(png(0))], "What was in the image?"], start=1):
            response = activate(client, index, {"input": [text | {"content": content}]})
            assert response.status_code == 200, response.text

    # A text turn before any image stays text-only. A later text turn resends the saved conversation's image, which
    # Pi would replace with an "omitted" note unless image input is declared.
    assert declared == [["text"], ["text", "image"], ["text", "image"]]


def test_retries_replay_per_activation_and_changed_inputs_conflict(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=continuation_seed()).json()["agent_session_id"]
        first = activate(client, 1, {"input": "one"})
        assert first.status_code == 200, first.text
        assert activate(client, 1, {"input": "one"}).json() == first.json()
        assert activate(client, 1, {"input": "changed"}).status_code == 409
        assert activate(client, 3, {"input": "skipped"}).status_code == 409
        second = activate(client, 2, {"input": "two"})
        assert second.status_code == 200, second.text
        assert activate(client, 2, {"input": "two"}).json() == second.json()
        assert activate(client, 1, {"input": "one"}).json() == first.json()
        # The activation index is required, so a retry can never start another turn.
        assert client.post(ROUTE, json={"input": "two"}).status_code == 422
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    assert sandbox.launch.await_count == 2


@pytest.mark.parametrize(
    ("second", "status"),
    [
        ({"input": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "two"}]}, 200),
        ({"input": "two", "instructions": "Be brief."}, 200),
        ({"input": [{"role": "system", "content": "Be verbose."}, {"role": "user", "content": "two"}]}, 422),
        ({"input": [{"role": "user", "content": "two"}, {"role": "user", "content": "three"}]}, 422),
        ({"input": "two", "temperature": 0.5}, 422),
    ],
    ids=["same-system", "same-instructions", "changed-system", "two-user-messages", "unsupported-control"],
)
def test_activation_accepts_one_user_message_and_keeps_the_session_instructions(setup, second, status):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=continuation_seed()).json()["agent_session_id"]
        first = activate(
            client, 1, {"input": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "one"}]}
        )
        assert first.status_code == 200, first.text
        result = activate(client, 2, second)
        assert result.status_code == status, result.text
        if status == 422:
            # A rejected delta does not consume the index.
            assert activate(client, 2, {"input": "two"}).status_code == 200
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    assert sandbox.launch.await_count == 2


def test_each_activation_spends_only_its_own_timeout(setup):
    agent, sandbox = setup
    limits = []
    run = sandbox.run

    async def record_launch(command, **kwargs):
        limits.append(float(re.search(r"--timeout (\S+)", command)[1]))
        return await run(command, **kwargs)

    sandbox.launch.side_effect = record_launch
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=continuation_seed()).json()["agent_session_id"]
        for index in (1, 2):
            result = activate(client, index, {"input": "turn", "metadata": {AGENT_TIMEOUT_METADATA_KEY: "30"}})
            assert result.status_code == 200, result.text
            assert result.json()["metadata"][AGENT_TIMEOUT_METADATA_KEY] == "30.0"
            # Hold the first activation's clock, so a shared deadline would show in the second launch.
            time.sleep(1.5)
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    # The supervisor stops Pi at what remains of the activation's own timeout after its preparation.
    assert len(limits) == 2
    assert all(29 < limit <= 30 for limit in limits), limits


def http_request(session: dict, index: int | None = None) -> Request:
    headers = [(AGENT_ACTIVATION_HEADER.encode(), str(index).encode())] if index is not None else []
    return Request(
        {"type": "http", "headers": headers, "session": session, "path_params": {"rollout_id": "pi-smoke-a2"}}
    )


async def test_close_stops_the_running_activation(setup):
    agent, sandbox = setup
    session = {}
    await agent.seed_agent_session(http_request(session), AgentSeedSessionRequest.model_validate(continuation_seed()))
    body = NeMoGymResponseCreateParamsNonStreaming(input="one")
    await agent.responses(http_request(session, 1), body)
    # The second run blocks until the supervisor is told to stop it.
    sandbox.blocked = True
    sandbox.exited = asyncio.Event()
    second = asyncio.create_task(agent.responses(http_request(session, 2), body.model_copy(update={"input": "two"})))
    while sandbox.launch.await_count < 2:
        await asyncio.sleep(0.01)
    closed = await agent.close_agent_session(
        http_request(session), AgentCloseSessionRequest(**close_body("pi-session"))
    )
    with pytest.raises(asyncio.CancelledError):
        await second
    sandbox.signal.assert_awaited_once()
    sandbox.disconnect.assert_awaited_once()
    assert root_invocations(closed.agent_observations.model_dump(mode="json")) == [
        (1, "completed", None, True),
        (2, "incomplete", "cancelled", True),
    ]
    interrupted = [gap for gap in closed.agent_observations.gaps if gap.code == "agent_activation_interrupted"]
    assert [gap.detail for gap in interrupted] == ["CancelledError"]


async def unstarted_activation_ends_incomplete_and_session_continues(agent, sandbox, session: dict) -> None:
    """Activation 2, with a 0.2-second budget, ends incomplete without running Pi, and activation 3 still runs."""
    late = NeMoGymResponseCreateParamsNonStreaming(input="two", metadata={AGENT_TIMEOUT_METADATA_KEY: "0.2"})
    started = time.monotonic()
    response = await asyncio.wait_for(agent.responses(http_request(session, 2), late), 2)
    assert time.monotonic() - started < 2
    assert (response.status, response.output) == ("incomplete", [])
    assert response.metadata["activation_id"] == "2"
    assert root_invocations(response.model_dump()["_ng_agent_observations"]) == [(2, "incomplete", "timeout", True)]
    assert sandbox.launch.await_count == 1


async def test_activation_whose_deadline_passes_while_preparing_ends_incomplete(setup):
    agent, sandbox = setup
    execute = sandbox.execute

    async def slow_preparation(command, **kwargs):
        if command.startswith("mkdir -p -- ") and "/activation-2/" in command:
            await asyncio.sleep(5)
        return await execute(command, **kwargs)

    sandbox.exec.side_effect = slow_preparation
    session = {}
    await agent.seed_agent_session(http_request(session), AgentSeedSessionRequest.model_validate(continuation_seed()))
    await agent.responses(http_request(session, 1), NeMoGymResponseCreateParamsNonStreaming(input="one"))
    await unstarted_activation_ends_incomplete_and_session_continues(agent, sandbox, session)
    third = await agent.responses(http_request(session, 3), NeMoGymResponseCreateParamsNonStreaming(input="three"))
    assert third.status == "completed"
    closed = await agent.close_agent_session(
        http_request(session), AgentCloseSessionRequest(**close_body("pi-session"))
    )
    assert root_invocations(closed.agent_observations.model_dump(mode="json")) == [
        (1, "completed", None, True),
        (2, "incomplete", "timeout", True),
        (3, "completed", None, True),
    ]
    interrupted = [gap for gap in closed.agent_observations.gaps if gap.code == "agent_activation_interrupted"]
    assert [(gap.invocation_id, gap.detail) for gap in interrupted] == [("pi-session:activation-2", "TimeoutError")]


async def test_activation_queued_past_its_deadline_ends_without_running_pi(setup):
    agent, sandbox = setup
    session = {}
    await agent.seed_agent_session(http_request(session), AgentSeedSessionRequest.model_validate(continuation_seed()))
    await agent.responses(http_request(session, 1), NeMoGymResponseCreateParamsNonStreaming(input="one"))
    # Other sessions' runs hold every runner slot for longer than the activation's budget.
    for _ in range(agent.config.concurrency):
        await agent.sem.acquire()
    try:
        await unstarted_activation_ends_incomplete_and_session_continues(agent, sandbox, session)
    finally:
        for _ in range(agent.config.concurrency):
            agent.sem.release()
    third = await agent.responses(http_request(session, 3), NeMoGymResponseCreateParamsNonStreaming(input="three"))
    assert third.status == "completed"


def test_failed_activation_fences_the_next_one_until_close(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        session_id = client.post("/v1/agent_sessions", json=continuation_seed()).json()["agent_session_id"]
        sandbox.result["return_code"] = 1
        assert activate(client, 1, {"input": "one"}).status_code >= 500
        sandbox.result["return_code"] = 0
        assert activate(client, 2, {"input": "two"}).status_code == 409
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
    sandbox.launch.assert_awaited_once()
    # Pi's own events ended the run normally; its exit code makes the activation a failure.
    assert root_invocations(closed.json()["agent_observations"]) == [(1, "failed", "server_error", True)]


def test_one_activation_session_keeps_its_directory_and_single_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        assert client.post(ROUTE, json={"input": "one"}).status_code == 200
        assert client.post(ROUTE, json={"input": "two"}).status_code == 409
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    payload = staged(sandbox)
    assert "activation-" not in sandbox.directory
    # A session that never resumes saves no conversation.
    assert "--no-session" in payload["command"]
    assert "--session-dir" not in payload["command"] and "--session-id" not in payload["command"]
    sandbox.launch.assert_awaited_once()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
async def test_pinned_pi_resumes_the_saved_conversation_with_tool_results_and_images(setup, tmp_path, monkeypatch):
    """Two activations of the pinned Pi through the real runner and supervisor, against a recording model."""
    agent, sandbox = setup
    node, package = _pinned_pi_package()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    runtime = tmp_path / "runtime"
    (runtime / "node/bin").mkdir(parents=True)
    (runtime / "node/bin/node").symlink_to(node)
    (runtime / "pi/node_modules/@earendil-works").mkdir(parents=True)
    (runtime / "pi/node_modules/@earendil-works/pi-coding-agent").symlink_to(package)
    workdir = tmp_path / "work"
    workdir.mkdir()
    requests = []

    def model(body):
        requests.append(body)
        if len(requests) == 1:
            call = {"index": 0, "id": "call-1", "type": "function"}
            call["function"] = {"name": "bash", "arguments": json.dumps({"command": "echo tool-output-42"})}
            delta, finish = {"role": "assistant", "tool_calls": [call]}, "tool_calls"
        else:
            delta, finish = {"role": "assistant", "content": f"Reply {len(requests)}."}, "stop"
        chunk = {"id": f"chat-{len(requests)}", "object": "chat.completion.chunk", "created": 0, "model": "m"}
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        return 200, "text/event-stream", f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

    session = {}
    with _http_server(model) as model_url:
        agent.server_client._build_server_base_url.return_value = model_url
        body = AgentSeedSessionRequest.model_validate(continuation_seed())
        body.sandbox_access.workdir = str(workdir)
        await agent.seed_agent_session(http_request(session), body)
        state = agent._session_records[body.agent_session_id].state
        state.runtime = str(runtime)
        directory = tmp_path / "session"
        (directory / "home/.pi/agent").mkdir(parents=True)
        agent.config.session_close_timeout_seconds = 10
        _run_session_locally(sandbox, state, directory, limit=60)
        first = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Look."}, image_part(png(7))]}]}
        )
        try:
            one = await asyncio.wait_for(agent.responses(http_request(session, 1), first), 30)
            two = await asyncio.wait_for(
                agent.responses(http_request(session, 2), NeMoGymResponseCreateParamsNonStreaming(input="Again.")), 30
            )
        finally:
            await agent.close_agent_session(
                http_request(session), AgentCloseSessionRequest(**close_body("pi-session"))
            )

    assert one.status == two.status == "completed"
    assert [item.content[0].text for item in two.output if item.type == "message"] == ["Reply 3."]
    # The second activation's request carries the whole first turn: the image, the tool call and its result.
    history = requests[2]["messages"]
    assert [message["role"] for message in history] == ["system", "user", "assistant", "tool", "assistant", "user"]
    first_user = history[1]["content"]
    assert [part["type"] for part in first_user] == ["text", "image_url"]
    assert first_user[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(png(7)).decode()
    assert "tool-output-42" in json.dumps(history[3]["content"])
    assert history[5]["content"] in ("Again.", [{"type": "text", "text": "Again."}])
