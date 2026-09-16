# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, NOOASandboxRuntimeConfig
from responses_api_agents.nooa_agent.runner import NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.sandbox_protocol import SandboxRunArtifact, SandboxRunRequest
from responses_api_agents.nooa_agent.sandbox_runner import SandboxedNOOARunner
from responses_api_agents.nooa_agent.sandbox_worker import run_worker


def invocation() -> NOOAInvocationConfig:
    return NOOAInvocationConfig.model_validate(
        {
            "agent_class": "sandbox_only.agent:Agent",
            "entrypoint": "answer",
            "execution_mode": "sandbox",
            "arguments": {"task": {"source": "responses_create_params.input", "transform": "latest_user_text"}},
        }
    )


def result() -> NOOARunResult:
    return NOOARunResult(
        episode=AgentEpisode(
            response=NeMoGymResponse(
                id="sandbox-result",
                created_at=1,
                model="nooa",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
            observations=AgentObservationBundle(source="nooa", records=[], gaps=[]),
        ),
        return_value={"answer": 42},
        model_cookies={"model": "cookie"},
        resource_cookies={"resource": "cookie"},
    )


class FakeSandbox:
    instances: list["FakeSandbox"] = []
    artifact = SandboxRunArtifact.from_result(result(), complete=True)

    def __init__(self, provider: object, spec: object) -> None:
        self.provider = provider
        self.spec = spec
        self.started = False
        self.stopped = False
        self.uploads: list[tuple[bytes, str]] = []
        self.commands: list[str] = []
        self.__class__.instances.append(self)

    async def start(self) -> None:
        self.started = True

    @classmethod
    async def connect(cls, descriptor: object, *, provider: object) -> "FakeSandbox":
        sandbox = cls(provider, SimpleNamespace(metadata={"descriptor": descriptor}))
        sandbox.started = True
        return sandbox

    async def stop(self) -> None:
        self.stopped = True

    async def upload(self, local_path: Path, remote_path: str) -> None:
        self.uploads.append((Path(local_path).read_bytes(), remote_path))

    async def download(self, remote_path: str, local_path: Path) -> None:
        local_path.write_text(self.artifact.model_dump_json(), encoding="utf-8")

    async def exec(self, command: str, **_: object) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)


def runner() -> SandboxedNOOARunner:
    return SandboxedNOOARunner(
        invocation=invocation(),
        runtime=NOOASandboxRuntimeConfig(
            provider={"docker": {}},
            spec={"image": "task-image", "workdir": "/workspace"},
        ),
        global_config={},
        model_server_name="policy",
        resources_server_name="resources",
        model_base_url="http://model:8000",
        resources_base_url="http://resources:8001",
        max_steps=4,
        default_timeout_secs=600,
    )


def request() -> NOOARunRequest:
    row = SimpleNamespace(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "solve"}]),
        model_dump=lambda **_: {"responses_create_params": {"input": [{"role": "user", "content": "solve"}]}},
    )
    return NOOARunRequest(
        row=row,
        model_url_path="/ng-rollout/rollout-1/v1/responses",
        model_cookies={"session": "model"},
        resource_cookies={"session": "resource"},
        task_id="task-1",
        rollout_id="rollout-1",
    )


@pytest.mark.asyncio
async def test_sandbox_runner_uploads_request_executes_worker_and_defers_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeSandbox.instances.clear()
    monkeypatch.setattr("responses_api_agents.nooa_agent.sandbox_runner.AsyncSandbox", FakeSandbox)

    run_result = await runner().run(request())
    sandbox = FakeSandbox.instances[0]

    assert sandbox.started is True
    assert sandbox.stopped is False
    assert run_result.return_value == {"answer": 42}
    assert any("sandbox_worker" in command for command in sandbox.commands)
    request_upload = next(source for source, target in sandbox.uploads if target.endswith("request.json"))
    payload = json.loads(request_upload)
    assert payload["invocation"]["agent_class"] == "sandbox_only.agent:Agent"
    assert payload["endpoints"] == {
        "policy": "http://model:8000",
        "resources": "http://resources:8001",
    }

    await run_result.aclose()
    assert sandbox.stopped is True


@pytest.mark.asyncio
async def test_sandbox_runner_recovers_checkpoint_on_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSandbox(FakeSandbox):
        async def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
            result = await super().exec(command, **kwargs)
            if "sandbox_worker" in command:
                return SimpleNamespace(return_code=124, stdout="", stderr="timed out", error_type="timeout")
            return result

    checkpoint = SandboxRunArtifact.from_result(result(), complete=False)
    FailingSandbox.artifact = checkpoint
    FailingSandbox.instances.clear()
    monkeypatch.setattr("responses_api_agents.nooa_agent.sandbox_runner.AsyncSandbox", FailingSandbox)

    with pytest.raises(Exception, match="timed out"):
        await runner().run(request())

    assert FailingSandbox.instances[0].stopped is True


@pytest.mark.asyncio
async def test_sandbox_runner_connects_seeded_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.artifact = SandboxRunArtifact.from_result(result(), complete=True)
    FakeSandbox.instances.clear()
    monkeypatch.setattr("responses_api_agents.nooa_agent.sandbox_runner.AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(
        "responses_api_agents.nooa_agent.sandbox_runner.create_provider",
        lambda provider: {"resolved": provider},
    )
    run_request = request()
    run_request.sandbox_descriptor = {"sandbox_id": "shared-task-sandbox"}

    run_result = await runner().run(run_request)
    sandbox = FakeSandbox.instances[0]

    assert sandbox.started is True
    assert sandbox.spec.metadata["descriptor"] == {"sandbox_id": "shared-task-sandbox"}
    assert sandbox.provider == {"resolved": {"docker": {}}}
    await run_result.aclose()


@pytest.mark.asyncio
async def test_sandbox_runner_attaches_checkpoint_to_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    class CancelledSandbox(FakeSandbox):
        async def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
            if "sandbox_worker" in command:
                raise asyncio.CancelledError
            return await super().exec(command, **kwargs)

    CancelledSandbox.artifact = SandboxRunArtifact.from_result(result(), complete=False)
    CancelledSandbox.instances.clear()
    monkeypatch.setattr("responses_api_agents.nooa_agent.sandbox_runner.AsyncSandbox", CancelledSandbox)

    with pytest.raises(asyncio.CancelledError) as captured:
        await runner().run(request())

    assert captured.value.nooa_result.return_value == {"answer": 42}
    assert CancelledSandbox.instances[0].stopped is True


@pytest.mark.asyncio
async def test_worker_runs_agent_and_serializes_observations(tmp_path: Path) -> None:
    worker_request = SandboxRunRequest(
        row={"responses_create_params": {"input": [{"role": "user", "content": "hello"}]}},
        invocation={
            "agent_class": "responses_api_agents.nooa_agent.tests.test_config_mapping:ConstructorAgent",
            "entrypoint": "analyze",
            "execution_mode": "sandbox",
            "init_kwargs": {"label": "inside-sandbox"},
            "arguments": {"text": {"source": "responses_create_params.input", "transform": "latest_user_text"}},
        },
        endpoints={"policy": "http://unused", "resources": "http://unused"},
        model_server_name="policy",
        resources_server_name="resources",
        model_url_path="/v1/responses",
        model_cookies={},
        resource_cookies={},
        max_steps=2,
        task_id="task-worker",
        rollout_id="rollout-worker",
    )
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    request_path.write_text(worker_request.model_dump_json(), encoding="utf-8")

    return_code = await run_worker(request_path, result_path, checkpoint_path)
    artifact = SandboxRunArtifact.model_validate_json(result_path.read_text(encoding="utf-8"))
    run_result = artifact.to_result()

    assert return_code == 0
    assert artifact.complete is True
    assert run_result is not None
    assert run_result.return_value == "hello"
    assert run_result.episode.observations.source == "nooa"
