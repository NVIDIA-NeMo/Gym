# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.tests.test_app import _session_db
from responses_api_agents.opencode_multimodal_agent.app import (
    OpenCodeMultimodalAgent,
    OpenCodeMultimodalRunRequest,
)
from responses_api_agents.opencode_multimodal_agent.eval import (
    OpenCodeMultimodalEvalAgent,
    OpenCodeMultimodalEvalConfig,
)
from responses_api_agents.opencode_multimodal_agent.eval_failures import terminal_model_result
from responses_api_agents.opencode_multimodal_agent.media import stage_inputs
from responses_api_agents.opencode_multimodal_agent.tests.test_app import agent, body, config


def evaluation(monkeypatch, tmp_path, prompt="Exact combined prompt\n"):
    native = agent(monkeypatch, max_turns=20, context_window=131072, tool_rewards=None)
    cfg = OpenCodeMultimodalEvalConfig.model_validate(
        native.config.model_dump()
        | {
            "system_prompt": None,
            "final_answer_reserve": None,
            "strip_reasoning_for_grading": False,
            "build_system_prompt": prompt,
            "entrypoint": "eval.py",
            "preserve_image_format": True,
            "sandbox": {"bwrap_command": ["bwrap"], "python_env": str(tmp_path), "plugin_dependencies": str(tmp_path)},
        }
    )
    return OpenCodeMultimodalEvalAgent(config=cfg, server_client=MagicMock(spec=ServerClient))


def test_eval_keeps_budget_and_matches_original_prompt_image_tools(monkeypatch, tmp_path):
    prompt = "  Exact combined prompt\nwith trailing newline.\n"
    evaluation_agent = evaluation(monkeypatch, tmp_path, prompt)
    cfg = evaluation_agent._build_opencode_config("test")
    assert cfg["agent"]["build"]["prompt"] == prompt
    assert cfg["agent"]["build"]["steps"] == 18
    assert "permission" not in cfg["agent"]["build"]
    assert cfg["attachment"]["image"] == {
        "auto_resize": True,
        "max_width": 8000,
        "max_height": 8000,
        "max_base64_bytes": 1024**3,
    }
    assert cfg["compaction"] == {"auto": False, "prune": False}
    assert cfg["tools"]["task"] is True
    assert cfg["permission"]["*"] == "allow"
    evaluation_agent._write_opencode_config(tmp_path)
    plugin = (tmp_path / ".eval-system-prompt.mjs").read_text()
    assert '"experimental.chat.system.transform"' in plugin
    assert "output.system.splice(0, output.system.length, prompt)" in plugin
    assert evaluation_agent.config.system_prompt is None


@pytest.mark.parametrize("prefix", ["input-", "image_"])
@pytest.mark.parametrize("absolute_command", [False, True])
def test_sandbox_command_stages_image_in_cwd(monkeypatch, tmp_path, prefix, absolute_command):
    evaluation_agent = evaluation(monkeypatch, tmp_path)
    evaluation_agent.config.input_image_prefix = prefix
    source = tmp_path / "staging/input-0.jpg"
    source.parent.mkdir()
    source.write_bytes(b"exact JPEG bytes")
    second_source = source.with_name("input-1.png")
    second_source.write_bytes(b"exact PNG bytes")
    workspace = tmp_path / "episode"
    workspace.mkdir()
    captured = {}

    def sandbox(cfg, wd, binary, env, cmd):
        captured.update(workspace=wd, binary=binary, command=cmd)
        return cmd

    monkeypatch.setattr("responses_api_agents.opencode_multimodal_agent.app.sandbox_command", sandbox)
    binary = tmp_path / "bin/opencode"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    cmd = [
        str(binary) if absolute_command else "opencode",
        "run",
        "--dir",
        str(workspace),
        "--file",
        str(source),
        str(second_source),
        "--",
        "question",
    ]
    result = evaluation_agent._prepare_command(cmd, workspace, {"PATH": str(binary.parent)})
    assert captured["binary"] == binary.resolve()
    assert (workspace / f"{prefix}0.jpg").read_bytes() == source.read_bytes()
    assert (workspace / f"{prefix}1.png").read_bytes() == second_source.read_bytes()
    assert sorted(path.name for path in workspace.iterdir()) == [f"{prefix}0.jpg", f"{prefix}1.png"]
    assert result == [
        "/agent-bin/opencode",
        "run",
        "--dir",
        "/testbed",
        "--file",
        f"/testbed/{prefix}0.jpg",
        f"/testbed/{prefix}1.png",
        "--",
        "question",
    ]
    assert str(source) not in result


def test_image_naming_is_opt_in_and_eval_only(monkeypatch, tmp_path):
    cfg = evaluation(monkeypatch, tmp_path).config
    assert cfg.input_image_prefix == "input-"
    assert OpenCodeMultimodalEvalConfig.model_validate(cfg.model_dump() | {"input_image_prefix": "image_"})
    with pytest.raises(ValidationError):
        OpenCodeMultimodalEvalConfig.model_validate(cfg.model_dump() | {"input_image_prefix": "../"})
    assert "input_image_prefix" not in config(input_image_prefix="image_").model_dump()


def test_eval_compaction_and_native_image_resize_are_independent(monkeypatch, tmp_path):
    runner = evaluation(monkeypatch, tmp_path)
    runner.config.compaction_auto = True
    compact = runner._build_opencode_config("test")
    assert compact["compaction"] == {"auto": True, "prune": False}
    assert compact["attachment"]["image"]["max_width"] == 8000
    runner.config.use_default_image_resize = True
    resized = runner._build_opencode_config("test")
    assert "attachment" not in resized
    assert resized["compaction"] == compact["compaction"]
    native = config(compaction_auto=True, use_default_image_resize=True).model_dump()
    assert "compaction_auto" not in native
    assert "use_default_image_resize" not in native


def test_unlimited_eval_omits_runtime_turn_cap_and_keeps_training_bounded(monkeypatch, tmp_path):
    bounded = evaluation(monkeypatch, tmp_path)
    cfg = OpenCodeMultimodalEvalConfig.model_validate(bounded.config.model_dump() | {"max_turns": None})
    unlimited = OpenCodeMultimodalEvalAgent(config=cfg, server_client=MagicMock(spec=ServerClient))
    expected = bounded._build_opencode_config("test")
    del expected["agent"]["build"]["steps"]
    assert unlimited._build_opencode_config("test") == expected
    with pytest.raises(ValidationError):
        config(max_turns=None)
    assert config().max_turns == 8


@pytest.mark.parametrize("budget,threshold", [(55, 53), (20, 18), (8, 6), (3, 1), (2, 1), (1, 1)])
def test_eval_reserves_final_turns_without_lowering_parser_budget(monkeypatch, tmp_path, budget, threshold):
    runner = evaluation(monkeypatch, tmp_path)
    runner.config.max_turns = budget
    runtime = runner._build_opencode_config("budget-check")
    assert runtime["agent"]["build"]["steps"] == threshold
    assert runner.config.max_turns == budget
    # The training agent's policy remains unchanged.
    native = agent(monkeypatch, max_turns=budget)
    assert native._build_opencode_config("training-check")["agent"]["build"]["steps"] == budget


def test_enforced_grace_budget_uses_gateway_instead_of_stock_steps(monkeypatch, tmp_path):
    runner = evaluation(monkeypatch, tmp_path)
    values = runner.config.model_dump() | {"turn_budget": {"regular_turns": 55, "grace_turns": 10}}
    with pytest.raises(ValidationError, match="max_turns must equal"):
        OpenCodeMultimodalEvalConfig.model_validate(values)
    runner.config = OpenCodeMultimodalEvalConfig.model_validate(values | {"max_turns": 65})
    assert "steps" not in runner._build_opencode_config("enforced")["agent"]["build"]
    assert runner.config.max_turns == 65


def test_preserve_source_jpeg_bytes_and_dimensions(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (4755, 1500), (20, 60, 100)).save(stream, format="JPEG")
    raw = stream.getvalue()
    request = body(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Question"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64," + base64.b64encode(raw).decode(),
                        "detail": "auto",
                    },
                ],
            }
        ]
    )
    _, _, images = stage_inputs(request, tmp_path, max_source_pixels=150_000_000, max_images=33, preserve_format=True)
    assert images[0].name == "input-0.jpg"
    assert images[0].read_bytes() == raw
    assert Image.open(images[0]).size == (4755, 1500)


def test_training_still_rejects_prompt_override():
    with pytest.raises(ValidationError, match="Keep the runtime default prompt"):
        config(opencode_config={"agent": {"build": {"prompt": "override"}}})
    assert not config().preserve_image_format


def test_eval_requires_prompt_and_rejects_duplicate_prefix():
    native = config().model_dump()
    with pytest.raises(ValidationError):
        OpenCodeMultimodalEvalConfig.model_validate(native | {"system_prompt": None})
    with pytest.raises(ValidationError):
        OpenCodeMultimodalEvalConfig.model_validate(native | {"build_system_prompt": "override"})


def test_native_training_prompt_keeps_eval_runtime_repairs(monkeypatch, tmp_path):
    combined = evaluation(monkeypatch, tmp_path)
    cfg = OpenCodeMultimodalEvalConfig.model_validate(
        combined.config.model_dump() | {"build_system_prompt": None, "system_prompt": config().system_prompt}
    )
    native = OpenCodeMultimodalEvalAgent(config=cfg, server_client=MagicMock(spec=ServerClient))
    combined_runtime = combined._build_opencode_config("test")
    native_runtime = native._build_opencode_config("test")
    del combined_runtime["agent"]["build"]["prompt"]
    del combined_runtime["plugin"]
    assert native_runtime == combined_runtime
    assert native.config.system_prompt == config().system_prompt
    assert native.config.preserve_image_format
    native._write_opencode_config(tmp_path)
    assert not (tmp_path / ".eval-system-prompt.mjs").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,kind",
    [
        ("OpenCode failed (timeout=False, exit=133): Bun crashed", "agent_run_error"),
        ("OpenCode failed (timeout=True, exit=-9):", "agent_timeout"),
        ("Missing assistant turns or exceeded max_turns", "agent_run_error"),
        ("OpenCode recorded a model/runtime error; do not score this rollout", "agent_run_error"),
    ],
)
async def test_episode_crash_is_persistable_zero(monkeypatch, tmp_path, reason, kind):
    runner = evaluation(monkeypatch, tmp_path)
    error = ClientResponseError(SimpleNamespace(real_url="http://agent/v1/responses?_ng_rollout_id=1"), (), status=500)
    error.response_content = reason.encode()
    native_run = AsyncMock(side_effect=error)
    monkeypatch.setattr(OpenCodeMultimodalAgent, "run", native_run)
    request = OpenCodeMultimodalRunRequest(
        responses_create_params=body(), verifier_metadata={"index": "12", "benchmark": "VStarBench"}
    )
    result = await runner.run(MagicMock(), request)
    assert result.reward == 0
    assert result.mask_sample is False
    assert result.failure_kind == kind
    assert result.failure_reason == reason
    assert result.response.status == "failed"
    assert result.response.error.message == reason
    assert result.response.output == []
    assert result.verifier_metadata == request.verifier_metadata
    assert not result.resolved
    assert "_ng_failure_class" not in result.model_dump()
    assert native_run.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_error,finish,expected",
    [
        ({"name": "APIError", "data": {"statusCode": 400}}, None, ("model_request_error", 400)),
        ({"name": "APIError", "data": {"statusCode": 429}}, None, ("model_request_error", 429)),
        ({"name": "APIError", "data": {"statusCode": 500}}, None, ("model_request_error", 500)),
        ({"name": "APIError", "data": {"isRetryable": True}}, None, ("model_transport_error", None)),
        ({"name": "ContextOverflowError", "data": {}}, "error", ("model_context_limit", None)),
    ],
)
async def test_provider_evidence_survives_self_call_without_changing_score(
    monkeypatch, tmp_path, provider_error, finish, expected
):
    runner = evaluation(monkeypatch, tmp_path)
    runner.config.capture_failure_artifacts = True
    database = _session_db(
        tmp_path,
        [({"role": "assistant", "time": {"completed": 2000}, "finish": finish, "error": provider_error}, [])],
    )
    reason = "OpenCode failed (timeout=False, exit=1): controlled model failure"
    monkeypatch.setattr(OpenCodeMultimodalAgent, "_run_opencode", AsyncMock(side_effect=RuntimeError(reason)))
    with pytest.raises(HTTPException) as captured:
        await runner._run_opencode("question", None, artifact_path=database)
    detail = captured.value.detail
    outcome = AgentOutcome.model_validate_json(detail[AGENT_OUTCOME_METADATA_KEY])
    assert (outcome.failure_kind, outcome.status_code) == expected
    error = ClientResponseError(SimpleNamespace(real_url="http://agent/v1/responses"), (), status=500)
    error.response_content = json.dumps({"detail": detail}).encode()
    monkeypatch.setattr(OpenCodeMultimodalAgent, "run", AsyncMock(side_effect=error))
    result = await runner.run(MagicMock(), OpenCodeMultimodalRunRequest(responses_create_params=body()))
    assert AgentOutcome.model_validate_json(result.response.metadata[AGENT_OUTCOME_METADATA_KEY]) == outcome
    assert result.response.error.message == reason
    assert result.reward == 0 and result.mask_sample is False and result.response.output == []
    assert result.scorer == "agent-failure"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout=True, exit=-9", "timeout=False, exit=-11"])
async def test_crash_does_not_inherit_a_prior_provider_error(monkeypatch, tmp_path, failure):
    runner = evaluation(monkeypatch, tmp_path)
    runner.config.capture_failure_artifacts = True
    database = _session_db(
        tmp_path,
        [
            (
                {
                    "role": "assistant",
                    "time": {"completed": 2000},
                    "error": {"name": "APIError", "data": {"statusCode": 400}},
                },
                [],
            )
        ],
    )
    monkeypatch.setattr(
        OpenCodeMultimodalAgent, "_run_opencode", AsyncMock(side_effect=RuntimeError(f"OpenCode failed ({failure}):"))
    )
    with pytest.raises(HTTPException) as captured:
        await runner._run_opencode("question", None, artifact_path=database)
    assert AGENT_OUTCOME_METADATA_KEY not in captured.value.detail


def test_terminal_model_result_does_not_use_recovered_or_child_errors(tmp_path):
    error = {
        "role": "assistant",
        "time": {"completed": 2000},
        "error": {"name": "APIError", "data": {"statusCode": 400}},
    }
    database = _session_db(
        tmp_path,
        [
            (error, []),
            ({"role": "assistant", "time": {"completed": 3000}, "finish": "stop"}, []),
            ("child", error, []),
        ],
        sessions=[("root", None), ("child", "root")],
    )
    assert terminal_model_result(database) == {}


@pytest.mark.parametrize("contents", [b"", b"not a database"])
def test_terminal_model_result_handles_unavailable_evidence(tmp_path, contents):
    database = tmp_path / "session.db"
    assert terminal_model_result(database) == {}
    database.write_bytes(contents)
    assert terminal_model_result(database) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,status,reason",
    [
        ("/verify", 500, "JudgeError: unavailable"),
        ("/verify", 500, "OpenCode failed (timeout=False, exit=133): quoted in judge input"),
        ("/seed_session", 500, "seed failed"),
        ("/v1/responses", 503, "service unavailable"),
        ("/v1/responses", 500, "FileNotFoundError: input image"),
    ],
)
async def test_infrastructure_and_judge_failures_remain_unjudged(monkeypatch, tmp_path, path, status, reason):
    runner = evaluation(monkeypatch, tmp_path)
    error = ClientResponseError(SimpleNamespace(real_url="http://agent" + path), (), status=status)
    error.response_content = reason.encode()
    monkeypatch.setattr(OpenCodeMultimodalAgent, "run", AsyncMock(side_effect=error))
    with pytest.raises(ClientResponseError) as caught:
        await runner.run(MagicMock(), OpenCodeMultimodalRunRequest(responses_create_params=body()))
    assert caught.value is error


def test_eval_honors_image_policy_and_disables_subagents(monkeypatch, tmp_path):
    runner = evaluation(monkeypatch, tmp_path)
    runner.config.image_policy.auto_resize = False
    runner.config.image_policy.max_width = 4096
    runner.config.subagents_enabled = False
    runtime = runner._build_opencode_config("custom-policy")
    assert runtime["attachment"]["image"]["auto_resize"] is False
    assert runtime["attachment"]["image"]["max_width"] == 4096
    assert runtime["tools"]["task"] is False
    assert runtime["permission"]["task"] == "deny"
    assert runtime["permission"]["bash"]["*killall*"] == "deny"


def test_eval_missing_executable_fails_before_launch(monkeypatch, tmp_path):
    runner = evaluation(monkeypatch, tmp_path)
    command = ["missing-opencode", "run", "--dir", str(tmp_path), "--file", "--", "question"]
    with pytest.raises(FileNotFoundError, match="OpenCode executable not found"):
        runner._prepare_command(command, tmp_path, {"PATH": str(tmp_path)})
