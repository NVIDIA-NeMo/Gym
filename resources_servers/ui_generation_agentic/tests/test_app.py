# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The server's seed, verify (guards G1-G6 and the judge), close and metrics, against a scripted fake sandbox.

No test runs the G1 kill command: on a local provider it would kill the host user's processes. Its behavior in the
real images is covered by the PR's Docker smoke runs.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from openai.types.responses.response import IncompleteDetails
from pytest import approx

import resources_servers.ui_generation_agentic.app as app_module
from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest, ModelServerRef
from nemo_gym.core import (
    EpisodeId,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    TaskId,
)
from nemo_gym.sandbox import ExecResult
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.ui_generation_agentic.app import (
    StackConfig,
    UiGenerationAgenticConfig,
    UiGenerationAgenticResourcesServer,
    agent_stop_reason,
)
from resources_servers.ui_generation_agentic.judge import SCHEMA_VERSION


ENTRY = b"export default function App() { return <main>Hello</main>; }\n"
RUNTIME = "proc=1:100\nproc=7:120\nproc=9:121\nmanifest_sha=pinned\n"


def _verdict(quality: int = 3, adherence: int = 2) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_adherence": {"score": adherence, "rationale": "r"},
        "quality": {"score": quality, "rationale": "r"},
        "critical_issues": [],
        "uncertainty": [],
        "execution_verified": False,
    }


def _completion(content: str) -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-reply",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }
    )


def _response(status: str = "completed", incomplete_reason: Optional[str] = None) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        status=status,
        incomplete_details=IncompleteDetails(reason=incomplete_reason) if incomplete_reason else None,
    )


class FakeSandbox:
    """Answers the server's commands from a script: each key names a step, each value its result(s)."""

    def __init__(self, **script: Any) -> None:
        self.script = {
            "baseline": ExecResult(stdout=RUNTIME, stderr="", return_code=0, error_type=None),
            "g1": ExecResult(stdout="killed=2 left=0\n", stderr="", return_code=0, error_type=None),
            "stat": ExecResult(stdout=f"regular file|{len(ENTRY)}\n", stderr="", return_code=0, error_type=None),
            "integrity": ExecResult(stdout="manifest=ok\nintegrity=done\n", stderr="", return_code=0, error_type=None),
            "mktemp": ExecResult(stdout="/tmp/ui-gen-verify.abc123\n", stderr="", return_code=0, error_type=None),
            "mkdir": ExecResult(stdout="", stderr="", return_code=0, error_type=None),
            "build": [ExecResult(stdout="built\n", stderr="", return_code=0, error_type=None)],
            **script,
        }
        self.entry = ENTRY
        self.commands: list[tuple[str, str, Optional[str]]] = []
        self.uploads: list[str] = []

    @staticmethod
    def _step(command: str) -> str:
        if "manifest_sha=" in command:
            return "baseline"
        if "killed=$killed" in command:
            return "g1"
        if command.startswith("stat -c"):
            return "stat"
        if "integrity=done" in command:
            return "integrity"
        if "mktemp -d" in command:
            return "mktemp"
        if command.startswith("mkdir "):
            return "mkdir"
        if "verify.sh" in command:
            return "build"
        raise AssertionError(f"unexpected command {command!r}")

    async def exec(self, command: str, *, cwd: str = "/", timeout_s: float = 0, user: Optional[str] = None):
        step = self._step(command)
        self.commands.append((step, command, user))
        result = self.script[step]
        if isinstance(result, list):
            result = result.pop(0) if len(result) > 1 else result[0]
        if isinstance(result, Exception):
            raise result
        return result

    async def download(self, source: str, destination: Path) -> None:
        if isinstance(self.script.get("download"), Exception):
            raise self.script["download"]
        Path(destination).write_bytes(self.entry)

    async def upload(self, source: Path, destination: str) -> None:
        self.uploads.append(destination)

    def steps(self) -> list[str]:
        return [step for step, _, _ in self.commands]


class FakeLender:
    """The LentSandboxes interface the server uses, lending FakeSandboxes."""

    def __init__(self, **script: Any) -> None:
        self.script = script
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.lent: list[str] = []
        self.stopped: list[FakeSandbox] = []
        self.fail_stops = 0

    async def lend(self, session_id: str, task_id: TaskId) -> None:
        if session_id not in self.sandboxes:
            self.lent.append(session_id)
            self.sandboxes[session_id] = FakeSandbox(**self.script)
        return None

    def get(self, session_id: str) -> Optional[FakeSandbox]:
        return self.sandboxes.get(session_id)

    async def release(self, session_id: str) -> None:
        if session_id not in self.sandboxes:
            return
        if self.fail_stops:
            self.fail_stops -= 1
            raise RuntimeError("stop failed")
        self.stopped.append(self.sandboxes.pop(session_id))

    async def release_all(self) -> None:
        for session_id in list(self.sandboxes):
            await self.release(session_id)


def _server(stack_fields: Optional[dict[str, Any]] = None, **fields: Any) -> UiGenerationAgenticResourcesServer:
    config = UiGenerationAgenticConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="ui_generation_agentic",
        sandbox_provider="sandbox",
        stacks={
            "ui-vite": StackConfig(
                **{
                    "image": "vite-image",
                    "workdir": "/app",
                    "entry_path": "src/App.tsx",
                    "language": "tsx",
                    "description": "vite",
                    "agent_user": "node",
                    **(stack_fields or {}),
                }
            ),
            "ui-static": StackConfig(
                image="static-image",
                workdir="/site",
                entry_path="index.html",
                language="html",
                description="static",
                agent_user="ui",
            ),
        },
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
        **{"agent_network_mode": "open", **fields},
    )
    return UiGenerationAgenticResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _with_lender(server: UiGenerationAgenticResourcesServer, **script: Any) -> FakeLender:
    lender = FakeLender(**script)
    server._lenders = {"ui-vite": lender, "ui-static": lender}
    return lender


def _seed(session_id: str = "s1", *, task: str = "t1", stack: str = "ui-vite", **task_data: Any):
    entry = {"ui-vite": "src/App.tsx", "ui-static": "index.html"}.get(stack, "x")
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="ui_generation_agentic:example", task_id=task),
        task_data={"stack": stack, "entry_path": entry, "user_prompt": "Build a counter.", **task_data},
    )


def _verify_body(stack: str = "ui-vite", **extra: Any) -> dict[str, Any]:
    entry = {"ui-vite": "src/App.tsx", "ui-static": "index.html"}[stack]
    return {
        "responses_create_params": {"input": []},
        "response": _response().model_dump(mode="json"),
        "stack": stack,
        "entry_path": entry,
        "user_prompt": "Build a counter.",
        **extra,
    }


@pytest.fixture
def judge(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=_completion(json.dumps(_verdict())))
    monkeypatch.setattr(app_module, "call_judge", mock)
    return mock


def _seed_and_verify(server, lender, *, body: Optional[dict[str, Any]] = None, seed=None) -> dict[str, Any]:
    client = TestClient(server.setup_webserver())
    seed = seed or _seed()
    response = client.post("/seed_session", json=seed.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    response = client.post("/verify", json=body or _verify_body(seed.task_data["stack"]))
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------------------------- verify


def test_a_built_page_scores_build_times_quality_and_releases_the_sandbox(judge) -> None:
    server = _server()
    lender = _with_lender(server)
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=_seed().model_dump(mode="json"))
    sandbox = lender.get("s1")

    result = client.post("/verify", json=_verify_body()).json()

    assert result["reward"] == approx(3 / 4)
    assert result["mask_sample"] is False
    assert result["build_passed"] is True and result["build_exit_code"] == 0
    assert result["reward_components"] == {"build": 1.0, "judge_quality": 0.75, "judge_prompt_adherence": 0.5}
    assert result["final_code"] == ENTRY.decode()
    assert result["judge_quality"] == 3 and result["judge_prompt_adherence"] == 2
    assert sandbox.steps() == ["baseline", "g1", "stat", "integrity", "mktemp", "mkdir", "build"]
    # G1 keeps exactly the processes recorded at seed, by identity.
    g1 = next(command for step, command, _ in sandbox.commands if step == "g1")
    assert 'keep=" 1:100 7:120 9:121 "' in g1 and "execd" not in g1
    assert sandbox.uploads == ["/tmp/ui-gen-verify.abc123/snapshot", "/tmp/ui-gen-verify.abc123/verify.sh"]
    assert lender.stopped == [sandbox] and not server.has_session_state("s1")
    judge.assert_awaited_once()

    # A repeated verify returns the same verdict without the sandbox or the judge.
    again = client.post("/verify", json=_verify_body()).json()
    assert again["reward"] == result["reward"] and judge.await_count == 1


@pytest.mark.parametrize(
    ("score", "expected"),
    [("quality", 3 / 4), ("prompt_adherence", 1 / 4), ("mean", 2 / 4), ("min", 1 / 4)],
)
def test_reward_score_picks_the_judge_score(judge, score: str, expected: float) -> None:
    judge.return_value = _completion(json.dumps(_verdict(quality=3, adherence=1)))
    server = _server(reward_score=score)

    result = _seed_and_verify(server, _with_lender(server))

    assert result["reward"] == approx(expected)


@pytest.mark.parametrize(
    ("script", "failure_reason"),
    [
        ({"stat": ExecResult(stdout="", stderr="no such file", return_code=1, error_type=None)}, "no_deliverable"),
        (
            {"stat": ExecResult(stdout="directory|4096", stderr="", return_code=0, error_type=None)},
            "invalid_deliverable",
        ),
        (
            {"stat": ExecResult(stdout="regular file|999999", stderr="", return_code=0, error_type=None)},
            "invalid_deliverable",
        ),
        ({"build": [ExecResult(stdout="error", stderr="", return_code=1, error_type=None)]}, "build_failed"),
        ({"build": [ExecResult(stdout="", stderr="", return_code=124, error_type=None)]}, "build_timeout"),
        (
            {
                "integrity": ExecResult(
                    stdout="manifest=bad\nbad=/x: FAILED\nintegrity=done", stderr="", return_code=0, error_type=None
                )
            },
            "tamper_detected",
        ),
        (
            {
                "integrity": ExecResult(
                    stdout="manifest=ok\nnewer=/opt/ui-gen/template/x\nintegrity=done",
                    stderr="",
                    return_code=0,
                    error_type=None,
                )
            },
            "tamper_detected",
        ),
    ],
)
def test_an_agent_failure_scores_zero_without_the_judge(judge, script: dict, failure_reason: str) -> None:
    server = _server()

    result = _seed_and_verify(server, _with_lender(server, **script))

    assert result["reward"] == 0.0 and result["mask_sample"] is False
    assert result["failure_reason"].startswith(failure_reason)
    assert result["tamper_detected"] is (failure_reason == "tamper_detected")
    judge.assert_not_awaited()


@pytest.mark.parametrize(
    ("script", "reason"),
    [
        (
            {
                "g1": ExecResult(
                    stdout="survivor=42 sleep 99\nkilled=9 left=1\n", stderr="", return_code=0, error_type=None
                )
            },
            "sleep 99",
        ),
        (
            {"g1": ExecResult(stdout="g1_error=unknown agent user\n", stderr="", return_code=0, error_type=None)},
            "could not stop",
        ),
        ({"g1": RuntimeError("provider down")}, "exec failed"),
        ({"download": RuntimeError("provider down")}, "download failed"),
        ({"integrity": ExecResult(stdout="", stderr="", return_code=0, error_type=None)}, "integrity check"),
        ({"mktemp": ExecResult(stdout="", stderr="disk full", return_code=1, error_type=None)}, "mktemp failed"),
        ({"mkdir": ExecResult(stdout="", stderr="denied", return_code=1, error_type=None)}, "build directory setup"),
        ({"build": [ExecResult(stdout="", stderr="", return_code=90, error_type=None)]}, "verifier setup failed"),
        ({"build": [ExecResult(stdout="", stderr="", return_code=127, error_type=None)]}, "verifier setup failed"),
        ({"build": [ExecResult(stdout="", stderr="", return_code=None, error_type="timeout")]}, "build could not run"),
        ({"build": [RuntimeError("provider down")]}, "exec failed"),
    ],
)
def test_an_infrastructure_failure_masks_the_sample(judge, script: dict, reason: str) -> None:
    server = _server()

    result = _seed_and_verify(server, _with_lender(server, **script))

    assert result["mask_sample"] is True and result["reward"] == 0.0
    assert reason in result["failure_reason"]
    judge.assert_not_awaited()


def test_an_inconclusive_build_is_retried(judge) -> None:
    server = _server()
    setup_failed = ExecResult(stdout="", stderr="", return_code=90, error_type=None)
    built = ExecResult(stdout="ok", stderr="", return_code=0, error_type=None)
    lender = _with_lender(server, build=[setup_failed, RuntimeError("provider blip"), built])
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=_seed().model_dump(mode="json"))
    sandbox = lender.get("s1")

    result = client.post("/verify", json=_verify_body()).json()

    assert result["build_passed"] is True and result["mask_sample"] is False
    assert sandbox.steps().count("build") == 3


def test_a_judge_failure_masks_and_a_retry_judges_the_retained_grade(judge) -> None:
    server = _server()
    lender = _with_lender(server)
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=_seed().model_dump(mode="json"))
    judge.return_value = _completion("not a verdict")

    failed = client.post("/verify", json=_verify_body()).json()
    judge.return_value = _completion(json.dumps(_verdict(quality=4)))
    retried = client.post("/verify", json=_verify_body()).json()

    assert failed["mask_sample"] is True
    assert retried["mask_sample"] is False and retried["reward"] == 1.0
    assert retried["final_code"] == ENTRY.decode()
    # The retry graded the retained snapshot: the sandbox was stopped after the first verify and built only once.
    assert len(lender.stopped) == 1 and lender.stopped[0].steps().count("build") == 1


def test_computed_fields_win_over_colliding_request_fields(judge) -> None:
    server = _server()
    colliding = {"reward": 7, "failure_reason": "x", "build_passed": False, "guard_mode": "x", "provenance": "kept"}

    result = _seed_and_verify(server, _with_lender(server), body=_verify_body(**colliding))
    lost = TestClient(_server().setup_webserver()).post("/verify", json=_verify_body(**colliding)).json()

    assert result["reward"] == approx(3 / 4) and result["failure_reason"] is None and result["build_passed"] is True
    assert result["guard_mode"] == "detection_only" and result["provenance"] == "kept"
    assert lost["mask_sample"] is True and lost["failure_kind"] == "session_lost" and lost["provenance"] == "kept"


def test_the_seeded_task_is_graded_not_the_verify_body(judge) -> None:
    server = _server()

    result = _seed_and_verify(server, _with_lender(server), body=_verify_body(stack="ui-static"))

    assert result["stack"] == "ui-vite" and result["entry_path"] == "src/App.tsx"


def test_full_guard_mode_targets_the_stack_agent_user_and_builds_as_the_verifier(judge) -> None:
    server = _server(guard_mode="full")
    work = ExecResult(stdout="/opt/ui-gen-verify/run.abc123\n", stderr="", return_code=0, error_type=None)
    lender = _with_lender(server, mktemp=work)
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=_seed().model_dump(mode="json"))
    sandbox = lender.get("s1")

    result = client.post("/verify", json=_verify_body()).json()

    assert result["build_passed"] is True
    users = {step: user for step, _, user in sandbox.commands}
    assert users["g1"] == users["stat"] == users["integrity"] == users["mkdir"] == "root"
    assert users["build"] == "ui-verify" and users["baseline"] is None
    g1 = next(command for step, command, _ in sandbox.commands if step == "g1")
    assert "uid=$(id -u -- node) ||" in g1
    assert "chown -R ui-verify /opt/ui-gen-verify/run.abc123" in next(
        c for s, c, _ in sandbox.commands if s == "mkdir"
    )


def test_an_oversized_entry_file_is_never_judged(judge) -> None:
    server = _server(max_deliverable_bytes=len(ENTRY) - 1)

    result = _seed_and_verify(server, _with_lender(server))

    assert result["failure_reason"].startswith("invalid_deliverable") and result["mask_sample"] is False
    judge.assert_not_awaited()


# ---------------------------------------------------------------------------------------------- sessions


def test_the_session_lifecycle_follows_the_resources_session_contract() -> None:
    server = _server()
    _with_lender(server)

    check_resources_session_contract(
        server.setup_webserver(),
        _seed("contract"),
        keeps_state=True,
        holds_state=server.has_session_state,
        verify_body=_verify_body(),
    )


@pytest.mark.parametrize(
    ("task_data", "fields"),
    [
        ({"stack": "ui-unknown"}, {}),
        ({"entry_path": "src/main.tsx"}, {}),
        ({}, {"stack_fields": {"image": None}}),
        ({"user_prompt": None}, {}),
    ],
)
def test_a_seed_the_server_cannot_serve_is_refused_with_422(task_data: dict, fields: dict) -> None:
    server = _server(**fields)
    lender = _with_lender(server)
    seed = _seed()
    seed.task_data.update(task_data)

    response = TestClient(server.setup_webserver()).post("/seed_session", json=seed.model_dump(mode="json"))

    assert response.status_code == 422, response.text
    assert lender.sandboxes == {}


def test_a_manifest_pin_mismatch_refuses_the_seed_and_stops_the_sandbox() -> None:
    server = _server(stack_fields={"manifest_sha256": "other"})
    lender = _with_lender(server)

    response = TestClient(server.setup_webserver()).post("/seed_session", json=_seed().model_dump(mode="json"))

    assert response.status_code == 422 and "pinned" in response.text
    assert lender.sandboxes == {} and len(lender.stopped) == 1


def test_a_seed_whose_runtime_cannot_be_recorded_stops_its_sandbox() -> None:
    server = _server()
    lender = _with_lender(server, baseline=ExecResult(stdout="", stderr="", return_code=None, error_type="timeout"))

    response = TestClient(server.setup_webserver(), raise_server_exceptions=False).post(
        "/seed_session", json=_seed().model_dump(mode="json")
    )

    assert response.status_code == 500
    assert lender.sandboxes == {} and len(lender.stopped) == 1


def test_a_repeated_seed_reuses_the_sandbox_and_another_task_is_refused() -> None:
    server = _server()
    lender = _with_lender(server)
    client = TestClient(server.setup_webserver())

    first = client.post("/seed_session", json=_seed().model_dump(mode="json"))
    again = client.post("/seed_session", json=_seed().model_dump(mode="json"))
    other_task = client.post("/seed_session", json=_seed(task="t2", stack="ui-static").model_dump(mode="json"))

    assert first.status_code == again.status_code == 200
    assert other_task.status_code == 409
    assert lender.lent == ["s1"]


def test_close_retries_a_failed_stop() -> None:
    server = _server()
    lender = _with_lender(server)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    seed = _seed()
    client.post("/seed_session", json=seed.model_dump(mode="json"))
    close = ResourcesCloseSessionRequest(resources_session_id="s1", episode_id=seed.episode_id).model_dump(mode="json")
    lender.fail_stops = 1

    failed = client.post("/close_session", json=close)
    retried = client.post("/close_session", json=close)

    assert failed.status_code == 500 and retried.status_code == 200
    assert len(lender.stopped) == 1 and not server.has_session_state("s1")


def test_retirement_stops_the_sandbox_without_fencing_the_session() -> None:
    server = _server()
    lender = _with_lender(server)
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json=_seed().model_dump(mode="json"))

    asyncio.run(server.retire_session_state("s1"))

    assert len(lender.stopped) == 1 and not server.has_session_state("s1")
    assert not server.session_is_closed("s1")


def test_shutdown_stops_every_stack_at_once() -> None:
    server = _server()
    started: list[str] = []
    gate = asyncio.Event()

    class SlowLender(FakeLender):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def release_all(self) -> None:
            started.append(self.name)
            if self.name == "ui-vite":
                await gate.wait()

    server._lenders = {"ui-vite": SlowLender("ui-vite"), "ui-static": SlowLender("ui-static")}

    async def run() -> None:
        task = asyncio.create_task(server.shutdown())
        for _ in range(3):
            await asyncio.sleep(0)
        assert sorted(started) == ["ui-static", "ui-vite"]
        gate.set()
        await task

    asyncio.run(run())


# ---------------------------------------------------------------------------------------------- config


def test_restricted_network_needs_the_opensandbox_provider(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})

    with pytest.raises(ValueError, match="opensandbox"):
        _server(agent_network_mode="restricted")


def test_a_user_set_network_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="network_policy"):
        _server(sandbox_config={"provider_options": {"network_policy": {"defaultAction": "allow"}}})


def test_several_workers_are_unsupported() -> None:
    assert _server().multiple_workers_unsupported_reason()


# ---------------------------------------------------------------------------------------------- metrics


@pytest.mark.parametrize(
    ("status", "incomplete_reason", "expected"),
    [
        ("completed", None, None),
        ("incomplete", None, "incomplete"),
        ("incomplete", "max_output_tokens", "incomplete:max_output_tokens"),
        ("failed", None, "failed"),
    ],
)
def test_agent_stop_reason(status: str, incomplete_reason: Optional[str], expected: Optional[str]) -> None:
    assert agent_stop_reason(_response(status, incomplete_reason)) == expected


def test_aggregate_metrics_counts_masked_rollouts_per_stack_and_scores_only_the_rest() -> None:
    rows = [
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "stack": "ui-vite", "reward": 1.0, "build_passed": True},
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "stack": "ui-vite", "reward": 0.0, "mask_sample": True},
        {"_ng_task_index": 2, "_ng_rollout_index": 0, "stack": "ui-static", "reward": 0.0, "mask_sample": True},
        {
            "_ng_task_index": 3,
            "_ng_rollout_index": 0,
            "stack": "ui-next",
            "reward": 0.0,
            "agent_stop_reason": "incomplete",
        },
    ]

    metrics = asyncio.run(_server().aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))).agent_metrics

    assert metrics["ui-vite/mean/masked"] == approx(0.5)
    assert metrics["ui-static/mean/masked"] == approx(1.0)
    assert metrics["ui-next/mean/masked"] == approx(0.0)
    assert metrics["ui-vite/mean/reward"] == approx(1.0)
    assert metrics["ui-next/mean/agent_incomplete"] == approx(1.0)
    assert "ui-static/mean/reward" not in metrics


def test_the_observed_judge_grading_returns_the_verify_response(judge, monkeypatch) -> None:
    from nemo_gym.trajectory import grading_result_binding

    observed: list[Any] = []
    real = app_module.observe_llm_judge_grading

    async def observe(calculate):
        result = await real(calculate)
        observed.append(result)
        return result

    monkeypatch.setattr(app_module, "observe_llm_judge_grading", observe)
    server = _server()

    result = _seed_and_verify(server, _with_lender(server))

    (graded,) = observed
    assert graded.reward == result["reward"] == approx(3 / 4)
    assert grading_result_binding(graded) is not None
