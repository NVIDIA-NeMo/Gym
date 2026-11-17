# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise review reproductions through the actual configuration and verifier boundaries."""

import json
from itertools import count
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome, AgentOutcomePolicyConfig, agent_outcome_policy
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import AGENT_INVALID_OUTPUT, AGENT_LIMIT, AGENT_TIMEOUT, MODEL_TRANSPORT_ERROR, PROVIDER_OOM_KILLED
from nemo_gym.sandbox import SandboxExecResult, SandboxSpec
from resources_servers.mm_coding_training.app import CodingTrainingServer
from resources_servers.mm_coding_training.quality import Quality
from resources_servers.mm_coding_training.remote_renderer import RemoteRenderer
from resources_servers.mm_coding_training.tests.test_app import request
from resources_servers.mm_coding_training.tests.test_app import server as server
from resources_servers.mm_coding_training.tests.test_sessions import http_request, installed
from resources_servers.mm_coding_training.tests.test_sessions import session_server as session_server


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mode", ["multi", "mixed"])
@pytest.mark.parametrize("explicit_host", [False, True])
def test_only_policy_endpoint_is_task_reachable(monkeypatch, mode, explicit_host):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "offline-validation-only")
    for key in ("OPENCODE_PATH", "RG_PATH", "PLUGIN_BUNDLE_PATH", "RG_SOURCE", "PLUGIN_BUNDLE_SOURCE"):
        monkeypatch.setenv("MM_CODING_" + key, "/opt/test-runtime/fixture")
    for key in ("OPENCODE_SHA256", "RG_SHA256", "PLUGIN_BUNDLE_SHA256"):
        monkeypatch.setenv("MM_CODING_" + key, "a" * 64)
    monkeypatch.setenv("MM_CODING_SANDBOX_IMAGE", "example/mm-coding-sandbox@sha256:" + "a" * 64)
    monkeypatch.setenv("MM_CODING_JUDGE_MODEL", "quality-judge")
    monkeypatch.setattr("nemo_gym.config.global_config.gethostbyname", lambda name: "10.42.0.8")
    ports = count(12001)
    monkeypatch.setattr("nemo_gym.config.global_config._find_open_port_using_range", lambda **kwargs: next(ports))
    initial = {
        "config_paths": [
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            f"resources_servers/mm_coding_training/configs/mm_coding_{mode}.yaml",
            "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml",
        ],
        "policy_model_name": "offline-validation-only",
        "policy_base_url": "http://localhost:1/v1",
        "policy_api_key": "offline-validation-only",
    }
    if explicit_host:
        initial["default_host"] = "10.42.0.8"
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(initial),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=False,
        )
    )
    assert config.policy_model.responses_api_models.vllm_model.host == "10.42.0.8"
    assert config.head_server.host == "127.0.0.1"
    private = set()
    for name, entry in config.items():
        if not isinstance(entry, (dict, type(config))):
            continue
        for group in ("resources_servers", "responses_api_agents", "responses_api_models", "environment_servers"):
            if group in entry and name != "policy_model":
                for instance in entry[group].values():
                    assert instance.host == "127.0.0.1", name
                    private.add(name)
    assert {
        "mm_coding_quality_judge",
        "mm_coding_opencode_agent",
    } <= private


def test_default_reference_root_survives_component_launch_directory(server, monkeypatch):
    from resources_servers.mm_coding_training.task_data import CodingTask

    data_root = ROOT / "environments/mm_coding_single/data"
    row = json.loads((ROOT / "resources_servers/mm_coding_training/data/example.jsonl").read_text().splitlines()[0])
    task = CodingTask.model_validate(row["task"])
    monkeypatch.chdir(ROOT / "resources_servers/mm_coding_training")
    relative = server.config.model_copy(update={"reference_root": Path("environments/mm_coding_single/data")})
    deployed = CodingTrainingServer(config=relative, server_client=server.server_client)
    assert deployed._reference(task) == (data_root / task.reference_image).read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("session", [False, True])
async def test_response_fields_override_matching_transport_metadata(server, session_server, session):
    body = request("").model_copy(update={"task_id": "transport-alias", "reward": 0.99, "_rowidx": 7})
    if session:
        installed(session_server, body, code=44)
        result = await session_server.verify(http_request(), body)
    else:
        result = await server.verify(http_request(), body)
    assert result.task_id == body.task.task_id and result.reward == 0
    assert result.model_extra["_rowidx"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [AGENT_LIMIT, AGENT_INVALID_OUTPUT])
@pytest.mark.parametrize("missing", [False, True])
async def test_failed_model_reply_still_verifies_saved_artifact(session_server, kind, missing):
    body = request(status="failed")
    body.response.metadata = {AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=kind).model_dump_json()}
    assert agent_outcome_policy(body.response) == "verify"
    box = installed(session_server, body, code=44 if missing else 0)

    async def grade(body, source, result):
        assert source == "<svg/>" and result.response.status == "failed"
        result.reward = 0.8
        return result

    with patch.object(type(session_server), "grade_source", side_effect=grade) as grading:
        result = await session_server.verify(http_request(), body)
    assert not result.mask_sample
    assert result.reward == (0 if missing else 0.8)
    assert grading.await_count == (0 if missing else 1)
    box.stop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [PROVIDER_OOM_KILLED, MODEL_TRANSPORT_ERROR])
async def test_failed_infrastructure_reply_does_not_become_model_zero(session_server, kind):
    body = request(status="failed")
    body.response.metadata = {AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=kind).model_dump_json()}
    box = installed(session_server, body)
    result = await session_server.verify(http_request(), body)
    assert result.mask_sample and result.reward == 0
    box.download.assert_not_awaited()
    await session_server.release_session_state("session-1")


@pytest.mark.asyncio
async def test_selected_infrastructure_cause_overrides_model_limit_metadata(session_server):
    body = request(status="failed")
    body.response.metadata = {AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=AGENT_LIMIT).model_dump_json()}
    body.response = body.response.model_copy(
        update={
            "failure_report": {
                "failure_id": "infra",
                "events": {"infra": {"boundary": "model", "failure_kind": MODEL_TRANSPORT_ERROR}},
            }
        }
    )
    box = installed(session_server, body)
    result = await session_server.verify(http_request(), body)
    assert result.mask_sample
    box.download.assert_not_awaited()
    await session_server.release_session_state("session-1")


@pytest.mark.asyncio
async def test_verifier_retains_custom_model_failure_verification_decision(session_server):
    body = request(status="failed")
    body.response.metadata = {AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=AGENT_TIMEOUT).model_dump_json()}
    config = AgentOutcomePolicyConfig(by_failure_kind={AGENT_TIMEOUT: "verify"})
    assert agent_outcome_policy(body.response, config=config) == "verify"
    assert agent_outcome_policy(body.response) != "verify"
    box = installed(session_server, body, code=44)
    result = await session_server.verify(http_request(), body)
    assert result.reward == 0 and not result.mask_sample
    box.stop.assert_awaited_once()


@pytest.mark.parametrize("task,visual,quality", [(0.70, 0.75, 0.73), (0.85, 0.90, 0.88), (0.70, 0.75, 0.72)])
def test_average_accepts_rounded_boundary(task, visual, quality):
    Quality(quality=quality, task_fidelity=task, visual_fidelity=visual, reason="Rounded at the boundary")


@pytest.mark.parametrize("quality", [0.7300001, 0.74, 0.7199999])
def test_average_still_refuses_genuine_mismatch(quality):
    with pytest.raises(ValidationError):
        Quality(quality=quality, task_fidelity=0.70, visual_fidelity=0.75, reason="Outside the rounding boundary")


@pytest.mark.asyncio
async def test_masked_failure_logs_stage_and_cause_types_without_secret_text(server, caplog):
    try:
        raise OSError("provider-key=DO_NOT_LOG_THIS")
    except OSError as cause:
        error = RuntimeError("SDK payload DO_NOT_LOG_THIS")
        error.__cause__ = cause
    renderer = AsyncMock()
    renderer.render.side_effect = error
    with patch.object(type(server), "_renderer", return_value=renderer):
        result = await server.verify(http_request(), request())
    assert result.mask_sample
    assert "render" in caplog.text and "OSError" in caplog.text and "RuntimeError" in caplog.text
    assert "DO_NOT_LOG_THIS" not in caplog.text


@pytest.mark.asyncio
async def test_actual_renderer_setup_failure_has_distinct_safe_diagnostic(server, caplog):
    box = AsyncMock()
    box.exec.return_value = SandboxExecResult("", "SDK_DIAGNOSTIC_DO_NOT_LOG", 1)
    renderer = RemoteRenderer(
        {}, SandboxSpec(image=server.config.sandbox_image), execution_timeout=60, ready_timeout=60
    )
    with (
        patch.object(type(server), "_renderer", return_value=renderer),
        patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box),
    ):
        result = await server.verify(http_request(), request())
    assert result.mask_sample and result.reward == 0
    assert "stage=render" in caplog.text and "RendererSetupError" in caplog.text
    assert "SDK_DIAGNOSTIC_DO_NOT_LOG" not in caplog.text
    assert box.exec.await_count == server.config.inconclusive_verification_retries + 1
    assert box.stop.await_count == server.config.inconclusive_verification_retries + 1
