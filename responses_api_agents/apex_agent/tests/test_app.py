# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import os
import tarfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.artifacts import NG_NO_GENERATION_METADATA
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from responses_api_agents.apex_agent import prebuilt_world_entrypoint
from responses_api_agents.apex_agent.app import (
    AGENT_INCOMPLETE,
    FINAL_SNAPSHOT_TOO_LARGE,
    NG_FAILURE_CLASS_KEY,
    NG_FAILURE_TERMINAL_KEY,
    ApexAgent,
    ApexAgentConfig,
    ApexAgentRunRequest,
    PolicyEgressRelay,
    _read_sandbox_written_file,
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
    client.global_config_dict = {"policy_model_name": "example-org/example-model"}
    agent = ApexAgent(config=config, server_client=client)
    agent._model_base_url = lambda _body: "http://model/v1"
    return agent


def test_instruction_from_input_only_uses_user_messages() -> None:
    body = _body()
    assert instruction_from_input(body.responses_create_params) == "Do the work"


@pytest.mark.parametrize("failure_kind", ["agent_timeout", "model_context_limit", AGENT_INCOMPLETE])
def test_an_ending_the_policy_caused_is_a_scored_zero_not_a_failure_row(failure_kind: str) -> None:
    payload = (
        _agent()
        ._model_outcome(
            _body(), "the policy ran out of its budget", failure_kind, partial_result={"final_answer": "x"}
        )
        .model_dump()
    )

    # A result row: it stays in the mean, and Gym still retries an agent_timeout whose window held server faults.
    assert (payload["reward"], payload["mask_sample"]) == (0.0, False)
    assert payload["failure_kind"] == failure_kind
    assert payload["failure_reason"] == "the policy ran out of its budget"
    assert not any(key.startswith("_ng_failure") for key in payload)
    assert payload["response"].get("metadata") != NG_NO_GENERATION_METADATA


def test_a_fault_around_the_policy_is_a_retryable_failure_row_without_an_answer() -> None:
    payload = _agent()._failure(_body(), "sandbox Stirrup rollout exited: command failed", return_code=1).model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == "apex_error"
    assert payload["failure_kind"] == "apex_agents:apex_error"
    assert NG_FAILURE_TERMINAL_KEY not in payload
    # Collection must not keep the partial trajectory as the rollout's answer, or the row is never retried.
    assert payload["response"]["metadata"] == NG_NO_GENERATION_METADATA
    verification = _agent()._failure(_body(), "verify failed", failure_class="verification_error").model_dump()
    assert verification["failure_kind"] == "verifier_error"


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
            "task input contains no user instruction",
            failure_class="invalid_task_input",
            failure_terminal=True,
        )
        .model_dump()
    )

    assert payload[NG_FAILURE_CLASS_KEY] == "invalid_task_input"
    assert payload[NG_FAILURE_TERMINAL_KEY] is True


@pytest.mark.parametrize(
    ("error_type", "checkpoint_error", "expected_class", "expected_kind"),
    [
        ("timeout", None, None, "agent_timeout"),
        (None, "ContextOverflowError: Context overflow reached the summarized context", None, "model_context_limit"),
        (None, "RuntimeError: startup failed", "sandbox_error", "apex_agents:sandbox_error"),
    ],
)
async def test_run_classifies_sandbox_exit(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    error_type: str | None,
    checkpoint_error: str | None,
    expected_class: str | None,
    expected_kind: str,
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
                # The bounded copy of the partial result, made in the sandbox before it is downloaded.
                MagicMock(return_code=0),
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

    # The policy's own timeout or context overflow is a scored zero; a sandbox fault is a retryable failure row.
    assert payload.get(NG_FAILURE_CLASS_KEY) == expected_class
    assert payload["failure_kind"] == expected_kind
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
    assert spec.env["PYTHONDONTWRITEBYTECODE"] == "1"


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
    assert runner["policy_model"] == "example-org/example-model"
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
    assert "PYTHONDONTWRITEBYTECODE" not in spec.env
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
    """A request as Gym dispatches it, with the collector's run id and the row's task and rollout indices stamped."""
    return _body_with(**({"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_run_id": "run-a"} | extra))


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


def _write_checkpoint(directory: Path, *, elapsed_seconds: float, turns: int = 3, run_id: str = "run-a") -> None:
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
        run_id=run_id,
    ).write(_State(turns))


def test_resume_is_off_unless_a_checkpoint_dir_is_configured() -> None:
    assert _agent()._resume_checkpoint_dir(_gym_body()) is None
    spec = _agent()._sandbox_spec(_body(), "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    assert runner["resume_checkpoint_dir"] is None
    assert runner["resume_allowed"] is False
    assert "binds" not in spec.provider_options


def test_checkpoint_dir_is_keyed_by_run_task_rollout_and_attempt(tmp_path: Path) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path / "ckpt"))

    assert agent._resume_checkpoint_dir(_gym_body()) == tmp_path.resolve() / "ckpt" / "run-a" / "task-1" / "t0_r0_a0"
    keyed = agent._resume_checkpoint_dir(_gym_body(_ng_task_index=7, _ng_rollout_index=2, _ng_attempt_index=1))
    assert keyed is not None and keyed.name == "t7_r2_a1"
    assert agent._resume_checkpoint_dir(_body()) is None
    # Without the collector's run id nothing says whose checkpoint a directory holds, so there is none.
    assert agent._resume_checkpoint_dir(_body_with(_ng_task_index=0, _ng_rollout_index=0)) is None


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

    checkpoint = await agent._prepare_resume(directory, "run-a")
    assert checkpoint is not None and checkpoint.turn == 3 and checkpoint.elapsed_seconds == 100.0

    retry_dir = agent._resume_checkpoint_dir(_gym_body(_ng_attempt_index=1))
    assert retry_dir is not None and retry_dir != directory
    assert await agent._prepare_resume(retry_dir, "run-a") is None
    assert retry_dir.is_dir() and (directory / "manifest.json").is_file()

    (directory / "manifest.json").write_text("{not json")
    loads: list[Path] = []
    real_load = load_resume_checkpoint

    def counting_load(path):
        loads.append(path)
        return real_load(path)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.load_resume_checkpoint", counting_load)
    assert await agent._prepare_resume(directory, "run-a") is None
    assert loads == [directory, directory]
    assert (directory / "manifest.json").read_text() == "{not json"

    assert await agent._prepare_resume(None, "run-a") is None

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

    assert NG_FAILURE_CLASS_KEY not in payload
    assert payload["failure_kind"] == "agent_timeout"
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
            if len(execs) != 3:
                # Setup before the rollout, and the bounded copy of the partial result after it.
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
    assert execs[2]["timeout_s"] == 2600
    assert payload["failure_kind"] == "agent_timeout"
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


# ---------------------------------------------------------------------------
# The partial result sits on a bind the sandbox can write
# ---------------------------------------------------------------------------


def test_a_sandbox_written_file_is_read_only_when_it_is_a_bounded_regular_file(tmp_path: Path) -> None:
    regular = tmp_path / "regular.json"
    regular.write_bytes(b"0123456789")
    assert _read_sandbox_written_file(regular, 10) == b"0123456789"
    with pytest.raises(ValueError, match="limit is 9"):
        _read_sandbox_written_file(regular, 9)

    secret = tmp_path / "host-secret"
    secret.write_text("host only")
    link = tmp_path / "link.json"
    link.symlink_to(secret)
    with pytest.raises(OSError):
        _read_sandbox_written_file(link, 1024)

    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(ValueError, match="not a regular file"):
        _read_sandbox_written_file(fifo, 1024)
    assert time.monotonic() - started < 5.0, "opening a FIFO without a writer must not wait for one"

    with pytest.raises(ValueError, match="not a regular file"):
        _read_sandbox_written_file(tmp_path, 1024)


class _DownloadingSandbox:
    """A sandbox whose download leaves ``content`` at the destination, or raises ``error`` as a provider would."""

    def __init__(
        self, content: bytes | None = None, error: Exception | None = None, copy_result: object | None = None
    ) -> None:
        self.content = content
        self.error = error
        self.copy_result = copy_result if copy_result is not None else MagicMock(return_code=0)
        self.commands: list[tuple[str, dict]] = []
        self.sources: list[str] = []

    async def exec(self, command: str, **kwargs):
        self.commands.append((command, kwargs))
        if isinstance(self.copy_result, Exception):
            raise self.copy_result
        return self.copy_result

    async def download(self, source: str, destination: Path) -> None:
        self.sources.append(source)
        if self.error is not None:
            raise self.error
        destination.write_bytes(self.content or b"")


async def test_partial_result_recovery_downloads_then_reads_the_copy_off_the_event_loop(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    sandbox = _DownloadingSandbox(json.dumps({"trajectory": [1], "completion_status": "running"}).encode())
    threads: list[str] = []
    real_to_thread = asyncio.to_thread

    async def recording_to_thread(func, *args, **kwargs):
        threads.append(func.__name__)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.asyncio.to_thread", recording_to_thread)

    recovered = await _agent()._recover_partial_result(sandbox, tmp_path / "copy.json")

    assert recovered == {"trajectory": [1], "completion_status": "running"}
    # Only the copy cut at the limit inside the sandbox reaches the host, never the file the sandbox left.
    assert sandbox.sources == ["/sandbox/partial_result.json.bounded"]
    assert threads == ["_load_sandbox_written_json"]


async def test_partial_result_is_cut_at_the_limit_inside_the_sandbox_before_it_is_downloaded(tmp_path: Path) -> None:
    agent = _agent()
    agent.config.max_partial_result_bytes = 1000
    sandbox = _DownloadingSandbox(b"{}")

    assert await agent._recover_partial_result(sandbox, tmp_path / "copy.json") == {}

    [(command, kwargs)] = sandbox.commands
    assert command == (
        "rm -f -- /sandbox/partial_result.json.bounded && [ -f /sandbox/partial_result.json ] && "
        "[ ! -L /sandbox/partial_result.json ] && "
        "head -c 1001 -- /sandbox/partial_result.json > /sandbox/partial_result.json.bounded"
    )
    # A FIFO left at the path blocks the copy, so the command is bounded in time and runs as the default user.
    assert kwargs == {"timeout_s": 60}


async def test_nothing_is_downloaded_when_the_bounded_copy_ran_and_refused_the_path(tmp_path: Path) -> None:
    """The trusted command found no regular file there: a missing path, a link, a FIFO or a directory."""
    refused = MagicMock(return_code=1, stderr="", stdout="", error_type=None)
    sandbox = _DownloadingSandbox(b'{"trajectory": []}', copy_result=refused)

    assert await _agent()._recover_partial_result(sandbox, tmp_path / "copy.json") is None
    assert sandbox.sources == [], "a path the copy refused is never downloaded"


@pytest.mark.parametrize(
    "copy_result",
    [
        RuntimeError("Sandbox has not been started"),
        MagicMock(return_code=125, stderr="the sandbox has stopped", stdout=None, error_type="sandbox"),
        MagicMock(return_code=125, stderr="timed out", stdout=None, error_type="timeout"),
    ],
    ids=["exec_raises", "sandbox_stopped", "copy_timed_out"],
)
async def test_the_file_itself_is_downloaded_when_the_bounded_copy_could_not_run(tmp_path: Path, copy_result) -> None:
    """A direct-exec sandbox is gone after its rollout timed out; its partial trajectory is still on the bind."""
    content = json.dumps({"trajectory": [{"role": "assistant", "content": "partial"}]}).encode()
    sandbox = _DownloadingSandbox(content, copy_result=copy_result)

    recovered = await _agent()._recover_partial_result(sandbox, tmp_path / "copy.json")

    assert recovered == {"trajectory": [{"role": "assistant", "content": "partial"}]}
    assert sandbox.sources == ["/sandbox/partial_result.json"]


async def test_an_oversized_file_downloaded_without_the_bounded_copy_is_removed_from_the_host(tmp_path: Path) -> None:
    agent = _agent()
    content = json.dumps({"trajectory": ["x" * 64]}).encode()
    agent.config.max_partial_result_bytes = len(content) - 1
    sandbox = _DownloadingSandbox(content, copy_result=RuntimeError("Sandbox has not been started"))
    destination = tmp_path / "copy.json"

    assert await agent._recover_partial_result(sandbox, destination) is None
    assert sandbox.sources == ["/sandbox/partial_result.json"]
    assert not destination.exists(), "the oversized copy does not stay on the host"


@pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b"\xff\xfe"])
async def test_partial_result_recovery_returns_nothing_for_a_file_that_is_not_a_json_object(
    tmp_path: Path, content: bytes
) -> None:
    assert await _agent()._recover_partial_result(_DownloadingSandbox(content), tmp_path / "copy.json") is None


async def test_partial_result_recovery_refuses_a_file_over_the_configured_size(tmp_path: Path) -> None:
    agent = _agent()
    assert agent.config.max_partial_result_bytes == 256 * 1024 * 1024
    content = json.dumps({"trajectory": ["x" * 64]}).encode()

    # The host reader holds the limit on its own, whatever the sandbox-side copy let through.
    agent.config.max_partial_result_bytes = len(content) - 1
    assert await agent._recover_partial_result(_DownloadingSandbox(content), tmp_path / "copy.json") is None
    agent.config.max_partial_result_bytes = len(content)
    assert await agent._recover_partial_result(_DownloadingSandbox(content), tmp_path / "copy.json") is not None


async def test_partial_result_recovery_returns_nothing_when_the_download_is_refused(tmp_path: Path) -> None:
    """The provider refuses a symlink, a FIFO or a missing file by raising; the rollout then has no partial result."""
    sandbox = _DownloadingSandbox(error=ValueError("refusing to download a path that is not a regular file"))

    assert await _agent()._recover_partial_result(sandbox, tmp_path / "copy.json") is None
    assert not (tmp_path / "copy.json").exists()


# ---------------------------------------------------------------------------
# Snapshot size limit: the final snapshot is the policy's, the initial one the dataset's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("initial_bytes", "final_bytes", "expected_class", "expected_kind"),
    [
        (10, 2000, None, FINAL_SNAPSHOT_TOO_LARGE),
        (2000, 10, "snapshot_too_large", "apex_agents:snapshot_too_large"),
    ],
)
async def test_a_snapshot_over_its_limit_is_the_policys_zero_only_when_the_policy_wrote_it(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    initial_bytes: int,
    final_bytes: int,
    expected_class: str | None,
    expected_kind: str,
) -> None:
    agent = _agent()
    agent.config.max_snapshot_bytes = 1000
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._download_world = AsyncMock()
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())
    result = {
        "final_answer": "done",
        "completed": True,
        "completion_status": "completed",
        "trajectory": [{"role": "assistant", "content": "wrote a large file"}],
    }
    outputs = {
        "/app/apex-gym/output/result.json": json.dumps(result).encode(),
        "/app/apex-gym/output/initial.zip": b"i" * initial_bytes,
        "/app/apex-gym/output/final.zip": b"f" * final_bytes,
    }

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
            destination.write_bytes(outputs[source])

        async def exec(self, *_args, **_kwargs):
            return MagicMock(return_code=0)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", MagicMock(return_value=FakeSandbox()))

    payload = (await agent.run(MagicMock(cookies={}), _body())).model_dump()

    assert payload["reward"] == 0.0
    assert payload["failure_kind"] == expected_kind
    assert payload.get(NG_FAILURE_CLASS_KEY) == expected_class
    assert payload["apex_trajectory"] == result["trajectory"]
    # Neither outcome is graded: the only request to the resources server is the session seed.
    assert [call.kwargs["url_path"] for call in agent.server_client.post.await_args_list] == ["/seed_session"]
    if expected_class is None:
        # A result row that stays in the mean, with the policy's answer kept.
        assert payload["mask_sample"] is False
        assert not any(key.startswith("_ng_failure") for key in payload)
        assert payload["response"].get("metadata") != NG_NO_GENERATION_METADATA
        assert "final artifact snapshot is 2000 bytes; limit is 1000" in payload["failure_reason"]
    else:
        # The dataset's world is over the limit: no retry changes that, and it is not the policy's zero.
        assert payload[NG_FAILURE_TERMINAL_KEY] is True
        assert payload["response"]["metadata"] == NG_NO_GENERATION_METADATA


# ---------------------------------------------------------------------------
# Checkpoints belong to the collector run that wrote them
# ---------------------------------------------------------------------------


async def test_two_evaluations_sharing_a_checkpoint_root_never_restore_each_others_checkpoints(
    tmp_path: Path,
) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path))
    first, second = _gym_body(_ng_run_id="run-a"), _gym_body(_ng_run_id="run-b")
    first_dir, second_dir = agent._resume_checkpoint_dir(first), agent._resume_checkpoint_dir(second)
    assert first_dir is not None and second_dir is not None
    assert first_dir != second_dir and first_dir.name == second_dir.name == "t0_r0_a0"
    _write_checkpoint(first_dir, elapsed_seconds=100.0, run_id="run-a")

    # The second evaluation dispatches the same task and indices: it starts fresh.
    assert await agent._prepare_resume(second_dir, agent._run_id(second)) is None
    # A redispatch by the first evaluation restores its own checkpoint.
    restored = await agent._prepare_resume(agent._resume_checkpoint_dir(first), agent._run_id(first))
    assert restored is not None and restored.turn == 3 and restored.run_id == "run-a"

    # The second evaluation's row ends its own lineage and leaves the first's checkpoint alone.
    await agent._discard_resume_checkpoint(second_dir)
    assert not second_dir.exists()
    assert (first_dir / "manifest.json").is_file()
    await agent._discard_resume_checkpoint(first_dir)
    assert not first_dir.exists()


async def test_a_checkpoint_written_by_another_run_is_not_restored_even_from_this_runs_directory(
    tmp_path: Path,
) -> None:
    """Run ids are shortened and sanitized into directory names, so the manifest, not the path, names the owner."""
    agent = _agent(resume_checkpoint_dir=str(tmp_path))
    body = _gym_body(_ng_run_id="run-b")
    directory = agent._resume_checkpoint_dir(body)
    assert directory is not None
    _write_checkpoint(directory, elapsed_seconds=100.0, run_id="run-a")

    assert await agent._prepare_resume(directory, "run-b") is None
    assert (directory / "manifest.json").is_file(), "nothing is deleted on a refused restore"
    assert await agent._prepare_resume(directory, None) is None
    assert await agent._prepare_resume(directory, "run-a") is not None


async def test_discarding_a_checkpoint_only_removes_a_rollout_directory_of_the_configured_root(
    tmp_path: Path,
) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path / "root"))
    run_dir = tmp_path / "root" / "run-a"
    task_dir = run_dir / "task-1"
    rollout_dir = task_dir / "t0_r0_a0"
    other_run = tmp_path / "root" / "run-b" / "task-1" / "t0_r0_a0"
    outside = tmp_path / "elsewhere" / "run-a" / "task-1" / "t0_r0_a0"
    for directory in (rollout_dir, other_run, outside):
        directory.mkdir(parents=True)

    for not_a_rollout in (tmp_path / "root", run_dir, task_dir, outside):
        await agent._discard_resume_checkpoint(not_a_rollout)
        assert not_a_rollout.is_dir()
    assert rollout_dir.is_dir()

    await agent._discard_resume_checkpoint(rollout_dir)
    assert not rollout_dir.exists()
    assert task_dir.is_dir() and other_run.is_dir()


def test_the_runner_is_told_which_run_owns_its_checkpoints(tmp_path: Path) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path))

    runner = json.loads(agent._sandbox_spec(_gym_body(), "Do the work").files["/app/apex-gym/runner_config.json"])

    assert runner["run_id"] == "run-a"


async def test_a_rollout_killed_by_its_timeout_still_reports_its_unanswered_tool_calls(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    agent = _agent()
    agent._ensure_runtime_setup = AsyncMock(return_value=tmp_path / "stirrup-runtime.tar.gz")
    agent._download_world = AsyncMock()
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())
    partial = {"trajectory": [], "n_mcp_unanswered_calls": 4, "n_mcp_trailing_unanswered_calls": 4}
    execs: list[str] = []

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
            destination.write_text(json.dumps(partial), encoding="utf-8")

        async def exec(self, command: str, **_kwargs):
            execs.append(command)
            if command.endswith("sandbox_entrypoint.py"):
                return MagicMock(return_code=125, stderr="timed out", stdout=None, error_type="timeout")
            return MagicMock(return_code=0)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", MagicMock(return_value=FakeSandbox()))

    payload = (await agent.run(MagicMock(cookies={}), _body())).model_dump()

    # The count is recorded; the timeout is still the policy's scored zero.
    assert payload["response"]["apex_mcp_unanswered_calls"] == 4
    assert payload["failure_kind"] == "agent_timeout"
    assert NG_FAILURE_CLASS_KEY not in payload
    assert (payload["reward"], payload["mask_sample"]) == (0.0, False)


def test_requests_without_a_run_id_say_once_that_checkpoints_are_off(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = _agent(resume_checkpoint_dir=str(tmp_path))
    unstamped = _body_with(_ng_task_index=0, _ng_rollout_index=0)

    with caplog.at_level("WARNING", logger="responses_api_agents.apex_agent.app"):
        assert agent._resume_checkpoint_dir(_gym_body()) is not None
        assert agent._resume_checkpoint_dir(unstamped) is None
        assert agent._resume_checkpoint_dir(unstamped) is None
        # A request with no dispatch stamps at all was never resumable, and is not what the warning is about.
        assert agent._resume_checkpoint_dir(_body()) is None

    warnings = [record.getMessage() for record in caplog.records if "checkpoints are disabled" in record.getMessage()]
    assert len(warnings) == 1 and "_ng_run_id" in warnings[0]

    # With the feature off there is nothing to warn about.
    with caplog.at_level("WARNING", logger="responses_api_agents.apex_agent.app"):
        caplog.clear()
        assert _agent()._resume_checkpoint_dir(unstamped) is None
    assert not caplog.records
