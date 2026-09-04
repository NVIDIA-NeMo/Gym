# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.job_bench.app import (
    AgentSandboxSession,
    JobBenchResourcesServer,
    JobBenchResourcesServerConfig,
    JobBenchSeedSessionRequest,
    JobBenchVerifyRequest,
    _build_archive,
    _extract_archive,
    _resolve_task_id,
)


SESSION = "session-1"


def _config(cache_dir: Path, tmp_path: Path, **overrides) -> JobBenchResourcesServerConfig:
    return JobBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="job_bench_resources_server",
        cache_dir=cache_dir,
        archives_dir=tmp_path / "archives",
        split="main",
        expected_task_count=1,
        sandbox_provider="test",
        sandbox_image="job-bench-sandbox:latest",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        **overrides,
    )


def _server(cache_dir: Path, tmp_path: Path, **overrides) -> JobBenchResourcesServer:
    return JobBenchResourcesServer(
        config=_config(cache_dir, tmp_path, **overrides),
        server_client=MagicMock(spec=ServerClient),
    )


def _request(session_id: str = SESSION) -> SimpleNamespace:
    # Deliberately not a MagicMock: judge_failsafe finds the verify body by duck
    # typing, and a MagicMock answers hasattr() for everything, so it would be
    # mistaken for the body.
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


def _body(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "verifier_metadata": {"task_id": task_id},
        "responses_create_params": {"input": [{"role": "user", "content": "do the task"}]},
        "response": {
            "output": [],
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def _sandbox(sandbox_id: str = "sandbox-1") -> MagicMock:
    sandbox = MagicMock()
    sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox_id})
    sandbox.upload = AsyncMock()
    sandbox.download = AsyncMock()
    sandbox.stop = AsyncMock()
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))
    return sandbox


def _judge_reply(content: str) -> MagicMock:
    reply = MagicMock()
    reply.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
    return reply


def _passing_verdict(num_criteria: int) -> str:
    return json.dumps(
        {
            "criteria_results": [
                {"index": i, "passed": True, "reasoning": "ok", "evidence": "e"} for i in range(num_criteria)
            ],
            "rubric_passed": True,
            "overall_reasoning": "all criteria met",
        }
    )


# --------------------------------------------------------------- task routing


def test_resolve_task_id_prefers_an_explicit_id() -> None:
    body = SimpleNamespace(task_id="main/a/task1", verifier_metadata={"task_id": "main/a/task1"})

    assert _resolve_task_id(body) == "main/a/task1"


def test_resolve_task_id_falls_back_to_verifier_metadata() -> None:
    body = SimpleNamespace(task_id=None, verifier_metadata={"task_id": "main/a/task1"})

    assert _resolve_task_id(body) == "main/a/task1"


def test_resolve_task_id_rejects_conflicting_ids() -> None:
    body = SimpleNamespace(task_id="main/a/task1", verifier_metadata={"task_id": "main/b/task2"})

    with pytest.raises(ValueError, match="Conflicting JobBench task IDs"):
        _resolve_task_id(body)


def test_resolve_task_id_requires_an_id() -> None:
    with pytest.raises(ValueError, match="must provide"):
        _resolve_task_id(SimpleNamespace(task_id=None, verifier_metadata={}))


# ------------------------------------------------------------ network posture


def test_open_network_is_the_default_and_adds_no_egress_policy(cache_dir: Path, tmp_path: Path) -> None:
    server = _server(cache_dir, tmp_path)

    # JobBench expects the open internet, so nothing should be injected.
    assert server._provider_options() == {}


def test_hermetic_mode_denies_all_egress_but_allows_the_model_server(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path
) -> None:
    server = _server(
        cache_dir,
        tmp_path,
        enforce_agent_no_network=True,
        sandbox_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    monkeypatch.setattr(
        "resources_servers.job_bench.app.get_global_config_dict",
        lambda: {"policy_model": {"responses_api_models": {"model": {"host": "model.internal", "port": 8000}}}},
    )

    options = server._provider_options()
    assert options["network_policy"]["defaultAction"] == "deny"
    assert options["network_policy"]["egress"] == [{"action": "allow", "target": "model.internal"}]


def test_hermetic_mode_rejects_a_loopback_model_host(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path
) -> None:
    server = _server(
        cache_dir,
        tmp_path,
        enforce_agent_no_network=True,
        sandbox_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    monkeypatch.setattr(
        "resources_servers.job_bench.app.get_global_config_dict",
        lambda: {"policy_model": {"responses_api_models": {"model": {"host": "127.0.0.1", "port": 8000}}}},
    )

    with pytest.raises(ValueError, match="cannot reach loopback model host"):
        server._provider_options()


def test_hermetic_mode_without_a_model_server_still_denies_all(cache_dir: Path, tmp_path: Path) -> None:
    server = _server(cache_dir, tmp_path, enforce_agent_no_network=True)

    assert server._provider_options()["network_policy"] == {"defaultAction": "deny", "egress": []}


# ----------------------------------------------------------------- archiving


def test_task_archive_contains_only_the_task_folder_by_default(cache_dir: Path, tmp_path: Path) -> None:
    server = _server(cache_dir, tmp_path)
    task = server._task_store.get("main/biostatisticians/task1")

    archive_path = _run(server._task_archive(task))
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()

    assert "task_folder/TASK_INSTRUCTIONS.txt" in names
    # The search files are the answer key's discovery half; they stay behind.
    assert not any(name.startswith("files_required_to_search") for name in names)


def test_task_archive_includes_search_files_when_enabled(cache_dir: Path, tmp_path: Path) -> None:
    server = _server(cache_dir, tmp_path, include_search_files=True)
    task = server._task_store.get("main/biostatisticians/task1")

    archive_path = _run(server._task_archive(task))
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()

    assert "files_required_to_search/guidance.txt" in names


def test_task_archive_is_built_once_and_reused(cache_dir: Path, tmp_path: Path) -> None:
    server = _server(cache_dir, tmp_path)
    task = server._task_store.get("main/biostatisticians/task1")

    first = _run(server._task_archive(task))
    mtime = first.stat().st_mtime_ns
    second = _run(server._task_archive(task))

    assert first == second
    assert second.stat().st_mtime_ns == mtime


def test_archive_roundtrip_preserves_content(tmp_path: Path) -> None:
    source = tmp_path / "output"
    source.mkdir()
    (source / "report.txt").write_text("done", encoding="utf-8")
    archive_path = tmp_path / "out.tgz"

    _build_archive({"output": source}, archive_path)
    destination = tmp_path / "restored"
    _extract_archive(archive_path, destination)

    assert (destination / "output" / "report.txt").read_text() == "done"


# --------------------------------------------------------------- seed_session


def test_seed_session_starts_a_sandbox_and_lays_out_the_workspace(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    sandbox = _sandbox()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))

    response = _run(server.seed_session(_request(), JobBenchSeedSessionRequest(**_body(task_id))))

    assert response.sandbox_handle == "sandbox-1"
    assert server._agent_sessions[SESSION].task_id == task_id
    sandbox.upload.assert_awaited_once()
    setup_command = sandbox.exec.await_args.args[0]
    assert "/workspace/output" in setup_command
    assert "tar -xzf" in setup_command


def test_seed_session_stops_the_sandbox_when_seeding_fails(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    sandbox = _sandbox()
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="", stderr="no such file"))
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))

    with pytest.raises(RuntimeError, match="Failed to seed JobBench workspace"):
        _run(server.seed_session(_request(), JobBenchSeedSessionRequest(**_body(task_id))))

    sandbox.stop.assert_awaited_once()
    assert SESSION not in server._agent_sessions


def test_seed_session_requires_a_sandbox_id(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    sandbox = _sandbox()
    sandbox.serialize = AsyncMock(return_value={})
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))

    with pytest.raises(RuntimeError, match="did not return a sandbox_id"):
        _run(server.seed_session(_request(), JobBenchSeedSessionRequest(**_body(task_id))))


def test_seed_session_replaces_a_stale_sandbox_for_the_same_session(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    stale = _sandbox("stale")
    server._agent_sessions[SESSION] = AgentSandboxSession(task_id=task_id, sandbox=stale, sandbox_handle="stale")
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=_sandbox()))

    _run(server.seed_session(_request(), JobBenchSeedSessionRequest(**_body(task_id))))

    stale.stop.assert_awaited_once()
    assert server._agent_sessions[SESSION].sandbox_handle == "sandbox-1"


# ---------------------------------------------------------------- collection


def test_collect_output_refuses_an_oversized_output_directory(cache_dir: Path, tmp_path: Path, task_id: str) -> None:
    server = _server(cache_dir, tmp_path, max_output_mib=1)
    task = server._task_store.get(task_id)
    sandbox = _sandbox()
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="9999\n", stderr=""))

    with pytest.raises(RuntimeError, match="above the 1 MiB cap"):
        _run(server._collect_output(sandbox, task, tmp_path / "dest"))


def test_collect_output_reports_a_missing_output_directory(cache_dir: Path, tmp_path: Path, task_id: str) -> None:
    server = _server(cache_dir, tmp_path)
    task = server._task_store.get(task_id)
    sandbox = _sandbox()
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="", stderr=""))

    with pytest.raises(RuntimeError, match="is missing in the agent sandbox"):
        _run(server._collect_output(sandbox, task, tmp_path / "dest"))


# -------------------------------------------------------------------- verify


def test_verify_without_a_seeded_session_scores_zero_and_explains_why(
    cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)

    response = _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert "No JobBench agent sandbox exists" in response.failure_reason
    # The rubric denominator is still reported so the row aggregates correctly.
    assert response.max_score == 15.0
    assert response.total_count == 2


def test_verify_scores_the_weighted_normalized_score(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    sandbox = _sandbox()
    server._agent_sessions[SESSION] = AgentSandboxSession(task_id=task_id, sandbox=sandbox, sandbox_handle="sandbox-1")

    async def fake_collect(_sandbox, _task, destination: Path) -> None:
        output = destination / "output"
        output.mkdir(parents=True)
        (output / "report.txt").write_text("Eligible population: 1,108", encoding="utf-8")

    monkeypatch.setattr(server, "_collect_output", fake_collect)
    # The first rubric passes both criteria; the second (the plot) fails.
    replies = iter([_judge_reply(_passing_verdict(2)), _judge_reply(json.dumps({"rubric_passed": False}))])
    monkeypatch.setattr(
        "resources_servers.job_bench.app.call_judge", AsyncMock(side_effect=lambda *a, **k: next(replies))
    )

    response = _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))

    assert response.evaluation_completed is True
    # 10 of 15 weight, not 1 of 2 rubrics.
    assert response.reward == pytest.approx(10 / 15, abs=1e-4)
    assert response.total_score == 10.0
    assert response.max_score == 15.0
    assert response.pass_rate == 0.5
    assert response.num_output_files == 1
    assert len(response.rubrics) == 2
    sandbox.stop.assert_awaited_once()


def test_verify_rejects_a_session_seeded_for_a_different_task(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    sandbox = _sandbox()
    server._agent_sessions[SESSION] = AgentSandboxSession(
        task_id="main/lawyers/task4", sandbox=sandbox, sandbox_handle="sandbox-1"
    )

    response = _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))

    assert response.reward == 0.0
    assert "does not match verify task" in response.failure_reason
    sandbox.stop.assert_awaited_once()


def test_verify_can_omit_rubric_details(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path, include_rubric_details_in_response=False)
    server._agent_sessions[SESSION] = AgentSandboxSession(
        task_id=task_id, sandbox=_sandbox(), sandbox_handle="sandbox-1"
    )

    async def fake_collect(_sandbox, _task, destination: Path) -> None:
        (destination / "output").mkdir(parents=True)

    monkeypatch.setattr(server, "_collect_output", fake_collect)
    monkeypatch.setattr(
        "resources_servers.job_bench.app.call_judge",
        AsyncMock(return_value=_judge_reply(json.dumps({"rubric_passed": False}))),
    )

    response = _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))

    assert response.rubrics is None
    assert response.reward == 0.0


def test_verify_propagates_a_judge_transport_failure(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    server._agent_sessions[SESSION] = AgentSandboxSession(
        task_id=task_id, sandbox=_sandbox(), sandbox_handle="sandbox-1"
    )

    async def fake_collect(_sandbox, _task, destination: Path) -> None:
        (destination / "output").mkdir(parents=True)

    monkeypatch.setattr(server, "_collect_output", fake_collect)
    monkeypatch.setattr(
        "resources_servers.job_bench.app.call_judge", AsyncMock(side_effect=JudgeError("judge unreachable"))
    )

    # A failed judge *call* is not a zero score: it must reach the /verify route's
    # judge_failsafe wrapper, which routes the row to the failures sidecar.
    with pytest.raises(JudgeError, match="judge unreachable"):
        _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))


@pytest.mark.parametrize("content", ["", "sorry, I cannot comply"])
def test_verify_scores_an_unusable_judge_reply_as_a_failed_rubric(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str, content: str
) -> None:
    server = _server(cache_dir, tmp_path)
    server._agent_sessions[SESSION] = AgentSandboxSession(
        task_id=task_id, sandbox=_sandbox(), sandbox_handle="sandbox-1"
    )

    async def fake_collect(_sandbox, _task, destination: Path) -> None:
        (destination / "output").mkdir(parents=True)

    monkeypatch.setattr(server, "_collect_output", fake_collect)
    monkeypatch.setattr("resources_servers.job_bench.app.call_judge", AsyncMock(return_value=_judge_reply(content)))

    response = _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))

    # A reply that arrived but says nothing usable is a zero, not a sidecar row.
    assert response.evaluation_completed is True
    assert response.reward == 0.0
    assert response.rubrics[0]["result"]["passed"] is False


def test_verify_attaches_images_only_to_the_visual_rubric(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    from resources_servers.job_bench.tests.test_judge import PNG_BYTES

    server = _server(cache_dir, tmp_path)
    server._agent_sessions[SESSION] = AgentSandboxSession(
        task_id=task_id, sandbox=_sandbox(), sandbox_handle="sandbox-1"
    )

    async def fake_collect(_sandbox, _task, destination: Path) -> None:
        output = destination / "output"
        output.mkdir(parents=True)
        (output / "ages.png").write_bytes(PNG_BYTES)

    monkeypatch.setattr(server, "_collect_output", fake_collect)
    call_judge = AsyncMock(return_value=_judge_reply(json.dumps({"rubric_passed": False})))
    monkeypatch.setattr("resources_servers.job_bench.app.call_judge", call_judge)

    response = _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))

    assert response.num_vision_images == 1
    contents = [call.kwargs["json"].messages[1]["content"] for call in call_judge.await_args_list]
    # Rubric 0 is textual; rubric 1 mentions a plot and gets the multimodal payload.
    text_only = [content for content in contents if isinstance(content, str)]
    multimodal = [content for content in contents if isinstance(content, list)]
    assert len(text_only) == 1
    assert len(multimodal) == 1
    assert any(part.get("type") == "image_url" for part in multimodal[0])


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)


# ------------------------------------------------------------ sandbox startup


def test_create_sandbox_pins_the_image_workdir_and_task_labels(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(
        cache_dir,
        tmp_path,
        sandbox_config={
            "ttl_s": 100,
            "ready_timeout_s": 60,
            "env": {"MPLBACKEND": "Agg"},
            "resources": {"cpu": 2.0, "memory_mib": 4096},
            "metadata": {"team": "gym"},
        },
    )
    task = server._task_store.get(task_id)

    monkeypatch.setattr("resources_servers.job_bench.app.get_global_config_dict", dict)
    monkeypatch.setattr("resources_servers.job_bench.app.resolve_provider_config", lambda *a: "provider")
    monkeypatch.setattr("resources_servers.job_bench.app.resolve_provider_metadata", lambda *a: {"origin": "test"})
    started = {}

    class FakeSandbox:
        def __init__(self, provider):
            started["provider"] = provider

        async def start(self, spec):
            started["spec"] = spec

    monkeypatch.setattr("resources_servers.job_bench.app.AsyncSandbox", FakeSandbox)

    _run(server._create_sandbox(task))

    spec = started["spec"]
    assert started["provider"] == "provider"
    assert spec.image == "job-bench-sandbox:latest"
    # The agent's prompt hard-codes /workspace, so the sandbox must start there.
    assert spec.workdir == "/workspace"
    assert spec.ttl_s == 100
    assert spec.env == {"MPLBACKEND": "Agg"}
    assert spec.metadata["origin"] == "test"
    assert spec.metadata["team"] == "gym"
    assert spec.metadata["benchmark"] == "job-bench"
    assert spec.metadata["job-bench-task"] == "main-biostatisticians-task1"


def test_stop_sandbox_swallows_provider_errors(cache_dir: Path, tmp_path: Path, task_id: str) -> None:
    server = _server(cache_dir, tmp_path)
    sandbox = _sandbox()
    sandbox.stop = AsyncMock(side_effect=RuntimeError("provider gone"))

    # A teardown failure must not mask the rollout's real result.
    _run(server._stop_sandbox(sandbox, task_id=task_id))


# ------------------------------------------------------- output collection


def test_collect_output_downloads_and_unpacks_the_archive(cache_dir: Path, tmp_path: Path, task_id: str) -> None:
    server = _server(cache_dir, tmp_path)
    task = server._task_store.get(task_id)

    source = tmp_path / "sandbox_output"
    source.mkdir()
    (source / "report.txt").write_text("1,108 subjects", encoding="utf-8")
    remote_archive = tmp_path / "remote.tgz"
    _build_archive({"output": source}, remote_archive)

    sandbox = _sandbox()
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="3\n", stderr=""))

    async def download(_remote: str, local: Path) -> None:
        Path(local).write_bytes(remote_archive.read_bytes())

    sandbox.download = AsyncMock(side_effect=download)

    destination = tmp_path / "collected"
    _run(server._collect_output(sandbox, task, destination))

    assert (destination / "output" / "report.txt").read_text() == "1,108 subjects"


def test_collect_output_reports_a_failed_archive_step(cache_dir: Path, tmp_path: Path, task_id: str) -> None:
    server = _server(cache_dir, tmp_path)
    task = server._task_store.get(task_id)
    sandbox = _sandbox()
    sandbox.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=0, stdout="1\n", stderr=""),
            SimpleNamespace(return_code=2, stdout="", stderr="tar: disk full"),
        ]
    )

    with pytest.raises(RuntimeError, match="Failed to archive JobBench output"):
        _run(server._collect_output(sandbox, task, tmp_path / "dest"))


def test_collect_output_tolerates_an_unparseable_size_probe(cache_dir: Path, tmp_path: Path, task_id: str) -> None:
    server = _server(cache_dir, tmp_path)
    task = server._task_store.get(task_id)

    source = tmp_path / "sandbox_output"
    source.mkdir()
    remote_archive = tmp_path / "remote.tgz"
    _build_archive({"output": source}, remote_archive)

    sandbox = _sandbox()
    # `du` printed something unexpected; treat the size as unknown rather than failing.
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))

    async def download(_remote: str, local: Path) -> None:
        Path(local).write_bytes(remote_archive.read_bytes())

    sandbox.download = AsyncMock(side_effect=download)

    _run(server._collect_output(sandbox, task, tmp_path / "collected"))


def test_verify_scores_zero_when_output_collection_fails(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    server = _server(cache_dir, tmp_path)
    sandbox = _sandbox()
    server._agent_sessions[SESSION] = AgentSandboxSession(task_id=task_id, sandbox=sandbox, sandbox_handle="sandbox-1")

    async def failing_collect(*_args):
        raise RuntimeError("sandbox vanished")

    monkeypatch.setattr(server, "_collect_output", failing_collect)

    response = _run(server.verify(_request(), JobBenchVerifyRequest(**_body(task_id))))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert "sandbox vanished" in response.collection_error
    sandbox.stop.assert_awaited_once()
