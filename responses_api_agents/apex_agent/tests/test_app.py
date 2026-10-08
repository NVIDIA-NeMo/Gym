# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.apex_agent import prebuilt_world_entrypoint
from responses_api_agents.apex_agent.app import (
    NG_FAILURE_CLASS_KEY,
    NG_FAILURE_TERMINAL_KEY,
    ApexAgent,
    ApexAgentConfig,
    ApexAgentRunRequest,
    PolicyEgressRelay,
    instruction_from_input,
    load_prebuilt_runner_source,
    load_runner_source,
)
from responses_api_agents.apex_agent.sandbox_entrypoint import _discover_gateway_url, _patch_code_mcp_cancellation_race
from responses_api_agents.apex_agent.stirrup_runtime import ResumeCheckpointer, load_resume_checkpoint


def _body() -> ApexAgentRunRequest:
    return ApexAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Do the work"}]},
            "task_id": "task-1",
            "world_id": "world-1",
            "verifier_metadata": {
                "rubric": [{"criteria": "secret rubric"}],
                "gold_response": "secret gold",
            },
        }
    )


def _agent(
    *,
    image: str = "registry.example/archipelago@sha256:1234",
    auto_build: bool = False,
    supports_vision: bool = True,
    truncation_recovery: bool = True,
    prebuilt_world_manifest: str | None = None,
    sandbox_provider: dict | None = None,
    policy_egress_relay: str = "auto",
    resume_checkpoint_dir: str | None = None,
) -> ApexAgent:
    config = ApexAgentConfig(
        host="0.0.0.0",
        port=8080,
        name="apex_agent",
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        concurrency=4,
        timeout=3600,
        image=image,
        image_build={
            "enabled": auto_build,
            "source_repo": "https://github.com/Mercor-Intelligence/archipelago.git",
            "source_revision": "0cb5c476c219a9df637e0bd37fb86b2361f4ab89",  # pragma: allowlist secret
            "source_root": None,
            "source_github_token": None,
            "dockerfile": "environment/Dockerfile",
            "docker_tag": "nemo-gym-archipelago:test",
            "timeout": 60,
        },
        sandbox_provider=sandbox_provider or {"apptainer": {}},
        sandbox_spec={},
        edgar_user_agent=None,
        max_turns=200,
        max_output_tokens=32_768,
        supports_vision=supports_vision,
        temperature=1.0,
        top_p=1.0,
        truncation_recovery=truncation_recovery,
        max_snapshot_bytes=None,
        max_world_bytes=None,
        artifact_output_dir=None,
        prebuilt_world_manifest=prebuilt_world_manifest,
        policy_egress_relay=policy_egress_relay,
        resume_checkpoint_dir=resume_checkpoint_dir,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy_model_name": "moonshotai/Kimi-K3"}
    agent = ApexAgent(config=config, server_client=client)
    agent._model_base_url = lambda _body: "http://model/v1"
    return agent


def test_instruction_from_input_only_uses_user_messages() -> None:
    body = _body()
    assert instruction_from_input(body.responses_create_params) == "Do the work"


def test_timeout_failure_is_routed_as_retryable_infra() -> None:
    result = _agent()._failure(
        _body(),
        "sandbox Stirrup rollout exited: direct apptainer command timed out after 12600s",
        return_code=125,
        failure_class="timeout_exceeded",
    )
    payload = result.model_dump()

    assert payload["reward"] == 0.0
    assert payload[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert "_ng_failure_terminal" not in payload
    assert "_ng_no_persist" not in payload


def test_non_timeout_failure_is_not_misclassified_as_timeout() -> None:
    payload = _agent()._failure(_body(), "sandbox Stirrup rollout exited: command failed", return_code=1).model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == "apex_error"
    assert payload[NG_FAILURE_CLASS_KEY] != "timeout_exceeded"


def test_failure_preserves_partial_trajectory_and_usage() -> None:
    trajectory = [{"role": "assistant", "content": "work in progress"}]
    payload = (
        _agent()
        ._failure(
            _body(),
            "sandbox failed",
            partial_result={
                "trajectory": trajectory,
                "completion_status": "error",
                "n_input_tokens": 12,
                "n_output_tokens": 7,
                "n_reasoning_tokens": 3,
            },
        )
        .model_dump()
    )

    assert payload["apex_trajectory"] == trajectory
    assert payload["apex_completion_status"] == "error"
    assert payload["response"]["apex_trajectory"] == trajectory
    assert payload["response"]["usage"]["input_tokens"] == 12
    assert payload["response"]["usage"]["output_tokens"] == 7


def test_terminal_failure_is_marked_for_sidecar_without_retry() -> None:
    payload = (
        _agent()
        ._failure(
            _body(),
            "maximum turns reached",
            failure_class="agent_incomplete",
            failure_terminal=True,
        )
        .model_dump()
    )

    assert payload[NG_FAILURE_CLASS_KEY] == "agent_incomplete"
    assert payload[NG_FAILURE_TERMINAL_KEY] is True


@pytest.mark.parametrize(
    ("error_type", "checkpoint_error", "expected_class"),
    [
        ("timeout", None, "timeout_exceeded"),
        (None, "ContextOverflowError: Context overflow reached the summarized context", "context_overflow"),
        (None, "RuntimeError: startup failed", "sandbox_error"),
    ],
)
async def test_run_classifies_sandbox_exit(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    error_type: str | None,
    checkpoint_error: str | None,
    expected_class: str,
) -> None:
    agent = _agent()
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._download_world = AsyncMock()
    agent._stirrup_archive = tmp_path / "stirrup-runtime.tar.gz"
    seed_response = MagicMock(cookies={})
    agent.server_client.post = AsyncMock(return_value=seed_response)
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())

    class FakeSandbox:
        def __init__(self) -> None:
            self._exec_results = [
                MagicMock(return_code=0),
                MagicMock(return_code=0),
                MagicMock(
                    return_code=125,
                    stderr="direct apptainer command timed out after 12600s",
                    stdout=None,
                    error_type=error_type,
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, *_args) -> None:
            return None

        async def download(self, _source: str, destination: Path) -> None:
            destination.write_text(
                json.dumps(
                    {
                        "trajectory": [{"role": "assistant", "content": "partial"}],
                        "completion_status": "running",
                        "n_input_tokens": 4,
                        "n_output_tokens": 2,
                        "checkpoint_error": checkpoint_error,
                    }
                ),
                encoding="utf-8",
            )

        async def exec(self, *_args, **_kwargs):
            return self._exec_results.pop(0)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", MagicMock(return_value=FakeSandbox()))

    result = await agent.run(MagicMock(cookies={}), _body())
    payload = result.model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == expected_class
    assert "_ng_failure_terminal" not in payload
    assert "_ng_no_persist" not in payload
    assert payload["apex_trajectory"] == [{"role": "assistant", "content": "partial"}]


def test_run_request_preserves_task_input_files() -> None:
    payload = _body().model_dump()
    payload["task_input_files"] = "snap_0123456789abcdef0123456789abcdef"

    body = ApexAgentRunRequest.model_validate(payload)

    assert body.task_input_files == "snap_0123456789abcdef0123456789abcdef"


def _prebuilt_body(world_id: str = "world_0123456789abcdef0123456789abcdef") -> ApexAgentRunRequest:
    payload = _body().model_dump()
    payload.update(
        {
            "runtime_mode": "prebuilt_world",
            "world_id": world_id,
            "task_slug": "accounting-example-b1-01234567",
        }
    )
    return ApexAgentRunRequest.model_validate(payload)


def _prebuilt_agent(tmp_path: Path, startup_ownership: list[dict[str, str]] | None = None) -> tuple[ApexAgent, Path]:
    cache_root = tmp_path / "images"
    cache_root.mkdir()
    world_id = "world_0123456789abcdef0123456789abcdef"
    image = cache_root / f"{world_id}.sif"
    image.write_bytes(b"SIF_MAGIC")
    manifest = tmp_path / "manifest.json"
    world_entry = {"runtime_image": str(image)}
    if startup_ownership is not None:
        world_entry["startup_ownership"] = startup_ownership
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "image_cache_root": str(cache_root),
                "worlds": {world_id: world_entry},
            }
        ),
        encoding="utf-8",
    )
    return _agent(prebuilt_world_manifest=str(manifest)), image


def test_prebuilt_world_selects_only_manifest_image(tmp_path: Path) -> None:
    agent, image = _prebuilt_agent(tmp_path)
    body = _prebuilt_body()

    assert agent._prebuilt_image(body) == str(image)
    spec = agent._sandbox_spec(body, "Do the work", image=str(image), prebuilt_world=True)
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert spec.image == str(image)
    assert runner["task_slug"] == "accounting-example-b1-01234567"
    assert runner["startup_timeout_seconds"] == 1800
    assert spec.files["/app/apex-gym/sandbox_entrypoint.py"] == load_prebuilt_runner_source()
    assert "secret rubric" not in json.dumps(runner)


def test_prebuilt_world_passes_validated_startup_ownership(tmp_path: Path) -> None:
    ownership = [
        {
            "path": "/app/tools/mcp_servers/docuseal/.state/postgres",
            "user": "svc_docuseal",
            "group": "appsdata_docuseal",
        }
    ]
    agent, image = _prebuilt_agent(tmp_path, ownership)
    spec = agent._sandbox_spec(_prebuilt_body(), "Do the work", image=str(image), prebuilt_world=True)
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    assert runner["startup_ownership"] == ownership


def test_prebuilt_world_rejects_unsafe_startup_ownership(tmp_path: Path) -> None:
    ownership = [{"path": "/etc", "user": "root", "group": "root"}]
    with pytest.raises(ValueError, match="invalid startup ownership"):
        _prebuilt_agent(tmp_path, ownership)


def test_prebuilt_world_rejects_missing_slug_and_unknown_world(tmp_path: Path) -> None:
    agent, _ = _prebuilt_agent(tmp_path)
    missing_slug = _prebuilt_body().model_copy(update={"task_slug": None})
    unknown_world = _prebuilt_body("world_ffffffffffffffffffffffffffffffff")

    try:
        agent._prebuilt_image(missing_slug)
    except ValueError as exc:
        assert "missing task_slug" in str(exc)
    else:
        raise AssertionError("missing task_slug was accepted")

    try:
        agent._prebuilt_image(unknown_world)
    except ValueError as exc:
        assert "trusted prebuilt-world manifest" in str(exc)
    else:
        raise AssertionError("unknown world was accepted")


async def test_prebuilt_world_config_errors_are_terminal(tmp_path: Path) -> None:
    agent, image = _prebuilt_agent(tmp_path)
    agent._ensure_runtime_setup = AsyncMock()
    request = MagicMock(cookies={})
    missing_slug = _prebuilt_body().model_copy(update={"task_slug": None})
    unknown_world = _prebuilt_body("world_ffffffffffffffffffffffffffffffff")

    for body, expected_error in (
        (missing_slug, "missing task_slug"),
        (unknown_world, "absent from the trusted prebuilt-world manifest"),
    ):
        payload = (await agent.run(request, body)).model_dump()
        assert payload[NG_FAILURE_CLASS_KEY] == "prebuilt_world_config_error"
        assert payload[NG_FAILURE_TERMINAL_KEY] is True
        assert expected_error in payload["apex_error"]

    image.unlink()
    payload = (await agent.run(request, _prebuilt_body())).model_dump()
    assert payload[NG_FAILURE_CLASS_KEY] == "prebuilt_world_config_error"
    assert payload[NG_FAILURE_TERMINAL_KEY] is True
    assert "prebuilt-world image is missing" in payload["apex_error"]
    agent._ensure_runtime_setup.assert_not_awaited()


async def test_prebuilt_world_adds_localhost_before_startup(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    agent, _ = _prebuilt_agent(tmp_path)
    agent._sandbox_provider = {"apptainer": {"create": {"extra_start_args": ["--fakeroot"]}}}
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._prepare_host_stirrup_runtime = AsyncMock(return_value=tmp_path / "runtime")
    commands: list[str] = []
    uploads: list[str] = []
    specs = []

    class FakeSandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, _source, destination: str) -> None:
            uploads.append(destination)
            return None

        async def exec(self, command: str, **_kwargs):
            commands.append(command)
            if len(commands) == 3:
                return MagicMock(return_code=1, stderr="hosts file is read-only")
            return MagicMock(return_code=0)

    def fake_sandbox(_provider, spec):
        specs.append(spec)
        return FakeSandbox()

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", fake_sandbox)
    payload = (await agent.run(MagicMock(cookies={}), _prebuilt_body())).model_dump()

    assert commands[2] == "printf '%s\\n' '127.0.0.1 localhost' >> /etc/hosts"
    assert "could not configure sandbox localhost" in payload["apex_error"]
    assert "sandbox_local_dns.py" not in str(specs[0].files)
    assert specs[0].provider_options["binds"] == [f"{tmp_path}/runtime:/app/stirrup-runtime:ro"]
    assert "/app/apex-gym/stirrup-runtime.tar.gz" not in uploads


async def test_failed_prebuilt_world_rows_point_at_the_saved_logs(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    agent, _ = _prebuilt_agent(tmp_path)
    agent.config.artifact_output_dir = str(tmp_path / "saved")
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._prepare_host_stirrup_runtime = AsyncMock(return_value=tmp_path / "runtime")
    exec_results = [
        MagicMock(return_code=0),
        MagicMock(return_code=0),
        MagicMock(return_code=0),
        MagicMock(
            return_code=1,
            stderr="TimeoutError: prebuilt world gateway did not become healthy",
            stdout=None,
            error_type=None,
        ),
    ]

    class FakeSandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, *_args) -> None:
            return None

        async def download(self, source: str, destination: Path) -> None:
            if not source.endswith("environment.log"):
                raise FileNotFoundError(source)
            destination.write_text("full startup log")

        async def exec(self, *_args, **_kwargs):
            return exec_results.pop(0)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", lambda *_args: FakeSandbox())

    payload = (await agent.run(MagicMock(cookies={}), _prebuilt_body())).model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == "sandbox_error"
    saved = Path(payload["artifact_output_dir"])
    assert saved.parent.parent == (tmp_path / "saved").resolve()
    assert (saved / "environment.log").read_text() == "full startup log"


async def test_host_stirrup_runtime_extracts_once(tmp_path: Path) -> None:
    agent = _agent()
    archive = tmp_path / "stirrup-runtime.tar.gz"
    payload = b"portable runtime"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("bin/python")
        member.mode = 0o755
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    runtime = await agent._prepare_host_stirrup_runtime(archive)
    assert (runtime / "bin/python").read_bytes() == payload
    archive.unlink()
    assert await agent._prepare_host_stirrup_runtime(archive) == runtime
    agent._host_stirrup_runtime.cleanup()


def test_prebuilt_manifest_rejects_image_outside_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "images"
    cache_root.mkdir()
    world_id = "world_0123456789abcdef0123456789abcdef"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "image_cache_root": str(cache_root),
                "worlds": {world_id: {"runtime_image": str(tmp_path / f"{world_id}.sif")}},
            }
        ),
        encoding="utf-8",
    )

    try:
        _agent(prebuilt_world_manifest=str(manifest))
    except ValueError as exc:
        assert "untrusted prebuilt-world image path" in str(exc)
    else:
        raise AssertionError("image outside the trusted cache was accepted")


def test_sandbox_config_never_contains_verifier_secrets() -> None:
    body = _body()
    spec = _agent()._sandbox_spec(body, "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    serialized = json.dumps(runner)

    assert runner["instruction"] == "Do the work"
    assert runner["policy_model"] == "moonshotai/Kimi-K3"
    assert runner["max_turns"] == 200
    assert runner["max_output_tokens"] == 32_768
    assert runner["supports_vision"] is True
    assert runner["truncation_recovery"] is True
    assert "tokenizer_path" not in runner
    assert "context_window_tokens" not in runner
    assert "max_tool_output_tokens" not in runner
    assert "secret rubric" not in serialized
    assert "secret gold" not in serialized
    assert "CODE_EXEC_RUN_AS_USER" not in spec.env
    assert "/app/apex-gym/stirrup_runtime.py" in spec.files
    assert "FOUNDRY_LOCAL_ROOT" not in spec.env


def test_sandbox_config_passes_the_policy_context_window_when_set() -> None:
    agent = _agent()
    agent.config.context_window_tokens = 262_144
    runner = json.loads(agent._sandbox_spec(_body(), "Do the work").files["/app/apex-gym/runner_config.json"])

    assert runner["context_window_tokens"] == 262_144
    assert runner["max_output_tokens"] == 32_768


def test_sandbox_config_propagates_text_only_model_capability() -> None:
    spec = _agent(supports_vision=False)._sandbox_spec(_body(), "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert runner["supports_vision"] is False


def test_sandbox_config_propagates_truncation_recovery_switch() -> None:
    spec = _agent(truncation_recovery=False)._sandbox_spec(_body(), "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert runner["truncation_recovery"] is False


def test_sandbox_runner_uses_archipelago_gateway_and_stirrup() -> None:
    source = load_runner_source()

    assert "_patch_code_mcp_cancellation_race()" in source
    assert "configure_gateway(" in source
    assert '"0"' in source
    assert "_discover_gateway_url(" in source
    assert "run_stirrup_rollout(" in source
    assert "checkpoint_path=PARTIAL_RESULT_PATH" in source
    assert 'write_snapshot(OUTPUT / "initial.zip")' in source
    assert "overlay_task_files(task_files_zip, scratch)" in source
    assert source.index("overlay_task_files(task_files_zip, scratch)") < source.index(
        'write_snapshot(OUTPUT / "initial.zip")'
    )
    assert "stdout=gateway_log" in source
    assert "stderr=asyncio.subprocess.STDOUT" in source
    assert "stdout=asyncio.subprocess.PIPE" not in source


def test_prebuilt_snapshot_converts_official_environment_export(tmp_path: Path) -> None:
    source = tmp_path / "snapshot.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name, content in (("filesystem/answer.docx", b"document"), (".apps_data/mail.json", b"{}")):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    snapshot = tmp_path / "snapshot.zip"
    manifest = prebuilt_world_entrypoint.snapshot_tar_to_zip(source, snapshot)

    assert manifest == ["filesystem/answer.docx", ".apps_data/mail.json"]
    with zipfile.ZipFile(snapshot) as archive:
        assert sorted(archive.namelist()) == [".apps_data/mail.json", "filesystem/answer.docx"]


async def test_gateway_url_uses_uvicorn_dynamic_port(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text("INFO: Uvicorn running on http://127.0.0.1:43127 (Press CTRL+C to quit)\n")
    process = MagicMock(returncode=None)

    assert await _discover_gateway_url(process, log_path) == "http://127.0.0.1:43127"


def test_code_mcp_cancellation_patch_is_idempotent(tmp_path: Path) -> None:
    session_path = tmp_path / "code/.venv/lib/python3.13/site-packages/mcp/shared/session.py"
    session_path.parent.mkdir(parents=True)
    session_path.write_text(
        "async def respond(self, response):\n"
        '        assert not self._completed, "Request already responded to"\n'
        "        await self._send(response)\n",
        encoding="utf-8",
    )

    _patch_code_mcp_cancellation_race(tmp_path)
    patched = session_path.read_text(encoding="utf-8")
    _patch_code_mcp_cancellation_race(tmp_path)

    assert "        if self._completed:\n            return\n" in patched
    assert session_path.read_text(encoding="utf-8") == patched


async def test_runtime_setup_resolves_image_before_stirrup(monkeypatch: MonkeyPatch) -> None:
    agent = _agent()
    events: list[str] = []
    monkeypatch.setattr(
        "responses_api_agents.apex_agent.app.resolve_image",
        MagicMock(side_effect=lambda **_kwargs: events.append("image") or "archipelago.sif"),
    )

    async def _build(_image: str) -> Path:
        events.append("runtime")
        return Path("/tmp/stirrup-runtime.tar.gz")

    agent._build_stirrup_archive = AsyncMock(side_effect=_build)

    async def _inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)

    await agent._ensure_runtime_setup()

    assert events == ["image", "runtime"]


def test_incomplete_rollout_snapshots_are_saved_without_grading(tmp_path: Path) -> None:
    agent = _agent()
    agent.config.artifact_output_dir = str(tmp_path / "saved")
    body = _body()

    output_dir = agent._persist_ungraded_snapshots(
        body,
        {"completion_status": "max_turns"},
        b"initial",
        b"final",
    )

    assert output_dir is not None
    assert (output_dir / "initial_snapshot.zip").read_bytes() == b"initial"
    assert (output_dir / "final_snapshot.zip").read_bytes() == b"final"
    assert json.loads((output_dir / "rollout.json").read_text())["completion_status"] == "max_turns"


# ---------------------------------------------------------------------------
# Mid-rollout checkpoint and resume: host-side dispatch rules
# ---------------------------------------------------------------------------


def _body_with(**extra) -> ApexAgentRunRequest:
    body = _body()
    return ApexAgentRunRequest.model_validate(body.model_dump() | extra)


def _gym_body(**extra) -> ApexAgentRunRequest:
    """A request as Gym dispatches it, with the row's task and rollout indices stamped."""
    return _body_with(_ng_task_index=0, _ng_rollout_index=0, **extra)


class _State:
    def __init__(self, turns: int) -> None:
        self.full_msg_history: list = []
        self.msgs = [SimpleNamespace(role="user", content="task")]
        for index in range(turns):
            self.msgs.append(SimpleNamespace(role="assistant", content=f"turn {index + 1}"))

    def to_dict(self) -> dict:
        return {
            "msgs": [{"role": message.role, "content": message.content} for message in self.msgs],
            "full_msg_history": [],
            "run_metadata_by_turn": {},
            "task_hash": "abc",
            "agent_name": "test",
        }


def _write_checkpoint(directory: Path, *, elapsed_seconds: float, turns: int = 3) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    initial = directory.parent / "initial-src.zip"
    with zipfile.ZipFile(initial, "w") as archive:
        archive.writestr("filesystem/report.txt", "pristine")

    def snapshot(destination: Path) -> list[str]:
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("filesystem/report.txt", "edited")
        return ["filesystem/report.txt"]

    ResumeCheckpointer(
        directory,
        snapshot_world=snapshot,
        apex_state=lambda: {"active_tools": [], "todos": []},
        initial_snapshot=initial,
        prior_elapsed_seconds=elapsed_seconds,
        clock=lambda: 0.0,
    ).write(_State(turns))


def test_resume_is_off_unless_a_checkpoint_dir_is_configured() -> None:
    assert _agent()._resume_checkpoint_dir(_gym_body()) is None
    spec = _agent()._sandbox_spec(_body(), "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    assert runner["resume_checkpoint_dir"] is None
    assert runner["resume_allowed"] is False
    assert "binds" not in spec.provider_options


def test_checkpoint_dir_is_keyed_by_task_rollout_and_attempt(tmp_path: Path) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path / "ckpt"))

    assert agent._resume_checkpoint_dir(_gym_body()) == tmp_path.resolve() / "ckpt" / "task-1" / "t0_r0_a0"
    keyed = agent._resume_checkpoint_dir(_body_with(_ng_task_index=7, _ng_rollout_index=2, _ng_attempt_index=1))
    assert keyed is not None and keyed.name == "t7_r2_a1"
    assert agent._resume_checkpoint_dir(_body()) is None


def test_sandbox_spec_mounts_the_checkpoint_dir_and_tells_the_runner(tmp_path: Path) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path))
    agent.config.sandbox_spec = {"provider_options": {"binds": "/data:/data:ro"}}
    resume_dir = tmp_path / "task-1" / "t0_r0_a0"

    fresh = agent._sandbox_spec(_body(), "Do the work", resume_dir=resume_dir)
    resumed = agent._sandbox_spec(_body(), "Do the work", resume_dir=resume_dir, resume_checkpoint=MagicMock())

    assert fresh.provider_options["binds"] == ["/data:/data:ro", f"{resume_dir}:/checkpoint"]
    fresh_runner = json.loads(fresh.files["/app/apex-gym/runner_config.json"])
    assert fresh_runner["resume_checkpoint_dir"] == "/checkpoint"
    assert fresh_runner["resume_allowed"] is False
    assert fresh_runner["resume_checkpoint_interval_seconds"] == 60.0
    assert json.loads(resumed.files["/app/apex-gym/runner_config.json"])["resume_allowed"] is True


async def test_prepare_resume_verifies_retries_once_and_never_deletes(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path))
    monkeypatch.setattr("responses_api_agents.apex_agent.app.asyncio.sleep", AsyncMock())
    directory = agent._resume_checkpoint_dir(_gym_body())
    assert directory is not None
    _write_checkpoint(directory, elapsed_seconds=100.0)

    checkpoint = await agent._prepare_resume(directory)
    assert checkpoint is not None and checkpoint.turn == 3 and checkpoint.elapsed_seconds == 100.0

    retry_dir = agent._resume_checkpoint_dir(_gym_body(_ng_attempt_index=1))
    assert retry_dir is not None and retry_dir != directory
    assert await agent._prepare_resume(retry_dir) is None
    assert retry_dir.is_dir() and (directory / "manifest.json").is_file()

    (directory / "manifest.json").write_text("{not json")
    loads: list[Path] = []
    real_load = load_resume_checkpoint

    def counting_load(path):
        loads.append(path)
        return real_load(path)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.load_resume_checkpoint", counting_load)
    assert await agent._prepare_resume(directory) is None
    assert loads == [directory, directory]
    assert (directory / "manifest.json").read_text() == "{not json"

    assert await agent._prepare_resume(None) is None

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    await agent._discard_resume_checkpoint(outside)
    assert outside.is_dir()


def test_response_carries_resume_metadata() -> None:
    response = ApexAgent._response_from_result(
        {"final_answer": "ok", "resume_segments": 2, "resumed_from_turn": 7, "n_resume_checkpoints": 3}, "m"
    )

    assert response.apex_resume_segments == 2
    assert response.apex_resumed_from_turn == 7
    assert response.apex_resume_checkpoints == 3


async def test_run_reports_a_timeout_when_the_budget_is_spent_before_resuming(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path))
    agent.config.timeout = 3600
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    directory = agent._resume_checkpoint_dir(_gym_body())
    assert directory is not None
    _write_checkpoint(directory, elapsed_seconds=3400.0)
    sandbox_factory = MagicMock()
    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", sandbox_factory)

    result = await agent.run(MagicMock(cookies={}), _gym_body())
    payload = result.model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert "budget exhausted" in payload["apex_error"]
    assert [m["content"] for m in payload["apex_trajectory"] if m["role"] == "assistant"] == [
        "turn 1",
        "turn 2",
        "turn 3",
    ]
    assert payload["response"]["apex_resume_segments"] == 1
    sandbox_factory.assert_not_called()
    assert not directory.exists(), "a written row ends the rollout's lineage"


async def test_run_resumes_with_the_remaining_budget_and_the_checkpointed_world(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path / "ckpt"))
    agent.config.timeout = 3600
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._download_world = AsyncMock()
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())
    directory = agent._resume_checkpoint_dir(_gym_body())
    assert directory is not None
    _write_checkpoint(directory, elapsed_seconds=1000.0)
    uploads: list[str] = []
    execs: list[dict] = []

    class FakeSandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, _source, remote: str) -> None:
            uploads.append(remote)

        async def download(self, _source: str, destination: Path) -> None:
            destination.write_text(json.dumps({"trajectory": [], "resume_segments": 2}), encoding="utf-8")

        async def exec(self, *_args, **kwargs):
            execs.append(kwargs)
            if len(execs) < 3:
                return MagicMock(return_code=0)
            return MagicMock(return_code=125, stderr="timed out", stdout=None, error_type="timeout")

    sandbox_factory = MagicMock(return_value=FakeSandbox())
    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", sandbox_factory)

    result = await agent.run(MagicMock(cookies={}), _gym_body())
    payload = result.model_dump()

    spec = sandbox_factory.call_args.args[1]
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    assert runner["resume_allowed"] is True
    assert spec.provider_options["binds"] == [f"{directory}:/checkpoint"]
    agent._download_world.assert_not_awaited()
    assert "/app/apex-gym/world.zip" not in uploads
    assert execs[-1]["timeout_s"] == 2600
    assert payload[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert payload["response"]["apex_resume_segments"] == 2
    assert not directory.exists()


async def test_a_kill_mid_flight_keeps_the_checkpoint_for_the_redispatch(tmp_path: Path) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path))
    directory = agent._resume_checkpoint_dir(_gym_body())
    assert directory is not None
    _write_checkpoint(directory, elapsed_seconds=10.0)
    agent._run_rollout = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await agent.run(MagicMock(cookies={}), _gym_body())

    assert (directory / "manifest.json").is_file()


def test_failed_prebuilt_world_logs_are_kept_in_full(tmp_path: Path) -> None:
    agent = _agent()
    agent.config.artifact_output_dir = str(tmp_path / "saved")
    body = _body()

    class FakeSandbox:
        async def download(self, source: str, destination: Path) -> None:
            if source.endswith("environment.log"):
                destination.write_text("full startup log")
            else:
                raise FileNotFoundError(source)

    output_dir = asyncio.run(agent._persist_world_logs(FakeSandbox(), body))

    assert output_dir is not None and "_failed_" in output_dir.name
    assert (output_dir / "environment.log").read_text() == "full startup log"
    assert not (output_dir / "world_bundle.txt").exists()


def test_failed_prebuilt_world_logs_need_an_artifact_dir() -> None:
    class FailingSandbox:
        async def download(self, source: str, destination: Path) -> None:
            raise AssertionError("nothing should be downloaded")

    assert asyncio.run(_agent()._persist_world_logs(FailingSandbox(), _body())) is None


async def _idle_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Accept a connection, hold it open until the peer goes away, then close it."""
    await reader.read()
    writer.close()


def test_policy_egress_relay_follows_private_network_namespace() -> None:
    private = {"apptainer": {"create": {"extra_start_args": ["--net", "--network=none", "--fakeroot"]}}}

    assert _agent(sandbox_provider=private)._policy_egress_relay_enabled() is True
    assert _agent()._policy_egress_relay_enabled() is False
    assert _agent(policy_egress_relay="always")._policy_egress_relay_enabled() is True
    assert _agent(sandbox_provider=private, policy_egress_relay="never")._policy_egress_relay_enabled() is False


def test_sandbox_config_binds_egress_socket_only_when_relaying(tmp_path: Path) -> None:
    relay = SimpleNamespace(socket_dir=tmp_path, socket_name="policy.sock")
    spec = _agent()._sandbox_spec(_body(), "Do the work", relay=relay)
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert spec.provider_options["binds"] == [f"{tmp_path}:/egress"]
    assert runner["model_egress_socket"] == "/egress/policy.sock"
    assert runner["model_base_url"] == "http://model/v1"

    plain = _agent()._sandbox_spec(_body(), "Do the work")

    assert "binds" not in plain.provider_options
    assert "model_egress_socket" not in json.loads(plain.files["/app/apex-gym/runner_config.json"])

    agent = _agent()
    agent.config.sandbox_spec = {"provider_options": {"binds": "/data:/data:ro"}}
    spec = agent._sandbox_spec(_body(), "Do the work", relay=relay)

    assert spec.provider_options["binds"] == ["/data:/data:ro", f"{tmp_path}:/egress"]


async def test_policy_egress_relay_forwards_to_model_server() -> None:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"echo:" + await reader.read(1024))
        await writer.drain()
        writer.close()

    upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = upstream.sockets[0].getsockname()[1]
    relay = await PolicyEgressRelay.start(f"http://127.0.0.1:{port}/v1")
    try:
        reader, writer = await asyncio.open_unix_connection(str(relay.socket_dir / relay.socket_name))
        writer.write(b"ping")
        await writer.drain()

        assert await reader.read(1024) == b"echo:ping"

        writer.close()
    finally:
        await relay.close()
        upstream.close()

    assert not relay.socket_dir.exists()


async def test_policy_egress_relay_rejects_non_http_model_urls() -> None:
    try:
        await PolicyEgressRelay.start("https://model/v1")
    except ValueError as exc:
        assert "http" in str(exc)
    else:
        raise AssertionError("https model URLs must be rejected")


async def test_policy_egress_relay_closes_with_idle_keepalive_connection() -> None:
    upstream = await asyncio.start_server(_idle_peer, "127.0.0.1", 0)
    port = upstream.sockets[0].getsockname()[1]
    relay = await PolicyEgressRelay.start(f"http://127.0.0.1:{port}/v1")
    reader, writer = await asyncio.open_unix_connection(str(relay.socket_dir / relay.socket_name))
    writer.write(b"GET /v1/models HTTP/1.1\r\n\r\n")
    await writer.drain()
    await asyncio.sleep(0.05)  # the bridge is now parked on both idle sockets
    try:
        await asyncio.wait_for(relay.close(), timeout=3)
    finally:
        writer.close()
        upstream.close()

    assert not relay.socket_dir.exists()


async def test_run_wires_and_tears_down_the_relay(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent(policy_egress_relay="always")
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._download_world = AsyncMock()
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())
    model = await asyncio.start_server(_idle_peer, "127.0.0.1", 0)
    agent._model_base_url = lambda _body: f"http://127.0.0.1:{model.sockets[0].getsockname()[1]}/v1"
    seen: dict[str, object] = {}

    class FakeSandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, *_args) -> None:
            return None

        async def download(self, _source: str, destination: Path) -> None:
            destination.write_text("{}", encoding="utf-8")

        def __init__(self) -> None:
            self._exec_results = [
                MagicMock(return_code=0),
                MagicMock(return_code=0),
                MagicMock(return_code=1, stderr="boom", stdout=None, error_type=None),
            ]

        async def exec(self, *_args, **_kwargs):
            return self._exec_results.pop(0)

    def fake_sandbox(_provider, spec):
        seen["spec"] = spec
        return FakeSandbox()

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", fake_sandbox)
    try:
        result = await agent.run(MagicMock(cookies={}), _body())
    finally:
        model.close()

    spec = seen["spec"]
    bind = spec.provider_options["binds"][0]
    socket_dir = Path(bind.split(":")[0])
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert bind.endswith(":/egress")
    assert runner["model_egress_socket"] == "/egress/policy.sock"
    assert result.model_dump()[NG_FAILURE_CLASS_KEY] == "sandbox_error"
    assert not socket_dir.exists()
