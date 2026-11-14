# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked setup tests only; no real sandbox, model or benchmark samples."""

import base64
import hashlib
import json
import shlex
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent import app as parent_app
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent
from responses_api_agents.opencode_video_agent.app import (
    EXECUTION_FAILURE_METADATA_KEY,
    LAUNCHER,
    OpenCodeVideoAgent,
    OpenCodeVideoAgentConfig,
    OpenCodeVideoRunRequest,
    VideoBinding,
    tool_output,
    video_config,
)


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "invented.mp4"
    source.write_bytes(b"invented unit-test bytes, not a video")
    bundle = tmp_path / "runtime.tar.gz"
    bundle.write_bytes(b"invented unit-test bundle, never executed")
    config = OpenCodeVideoAgentConfig(
        name="video_agent",
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="verifier"),
        media_runtime_archive=bundle,
        media_runtime_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        allowed_video_roots=[tmp_path],
        opencode_version="1.17.11",
        preinstalled_opencode=True,
        media_profile={"max_context_tokens": 262144, "reserved_output_tokens": 32768},
        opencode_max_context_window=262144,
        max_output_tokens=32768,
        sandbox_provider="opensandbox",
        sandbox_config={"workdir": "/testbed/work"},
        sandbox_timeout=120,
        artifacts_dir=str(tmp_path / "evidence"),
        execution_failure_reward_zero=True,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    agent = OpenCodeVideoAgent(config=config, server_client=client)
    binding = VideoBinding(
        source=source, sandbox_path="video/invented.mp4", sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    return agent, binding


async def test_aggregate_metrics_uses_verifier_without_any_model_or_sandbox(setup):
    agent, _ = setup
    expected = AggregateMetrics(
        key_metrics={"mean/Video_MME_v2_agentic": 0.5, "Video_MME_v2_agentic/coverage": 1.0},
        agent_metrics={"Video_MME_v2_agentic/expected": 2},
    )
    reply = SimpleNamespace(ok=True, read=AsyncMock(return_value=expected.model_dump_json().encode()))
    agent.server_client.post = AsyncMock(return_value=reply)
    body = AggregateMetricsRequest(verify_responses=[])
    assert await agent.aggregate_metrics(body) == expected
    agent.server_client.post.assert_awaited_once_with(server_name="verifier", url_path="/aggregate_metrics", json=body)
    assert not agent._sandbox_id_to_sandbox


async def test_skip_verification_keeps_native_aggregation_without_proxy(setup, monkeypatch):
    agent, _ = setup
    agent.config.skip_verification = True
    expected = AggregateMetrics()
    native = AsyncMock(return_value=expected)
    monkeypatch.setattr(OpenCodeSandboxedAgent, "aggregate_metrics", native)
    agent.server_client.post = AsyncMock()
    body = AggregateMetricsRequest(verify_responses=[])
    assert await agent.aggregate_metrics(body) == expected
    native.assert_awaited_once_with(body)
    agent.server_client.post.assert_not_awaited()


async def test_aggregation_failure_is_not_replaced_by_empty_metrics(setup):
    agent, _ = setup
    reply = SimpleNamespace(ok=True, read=AsyncMock(return_value=b"not-json"))
    agent.server_client.post = AsyncMock(return_value=reply)
    with pytest.raises(ValueError):
        await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))


@pytest.mark.parametrize("name", ["/etc/passwd", "../outside.mp4", "video/../../outside.mp4", ""])
def test_binding_cannot_escape_workspace(setup, name):
    _, binding = setup
    with pytest.raises(ValidationError):
        VideoBinding(**{**binding.model_dump(), "sandbox_path": name})


def test_trusted_binding_is_json_serializable_in_native_seed_transport(setup):
    _, binding = setup
    body = OpenCodeVideoRunRequest(
        setting="agentic",
        video_binding=binding,
        responses_create_params={"input": [{"role": "user", "content": "Watch the invented test video."}]},
    )
    serialized = json.loads(json.dumps(body.model_dump()))
    assert serialized["video_binding"]["source"] == str(binding.source)
    assert body.video_binding.source.is_file()
    assert body.video_binding.sha256 == binding.sha256


def test_profile_and_native_limits_must_match(setup):
    agent, _ = setup
    values = agent.config.model_dump()
    values["max_output_tokens"] = 8192
    with pytest.raises(ValidationError, match="output"):
        OpenCodeVideoAgentConfig(**values)
    assert agent.config.sandbox_config["env"]["NEMOTRON_OPENCODE_MEDIA_LAUNCHER"] == str(LAUNCHER)


def test_dependencies_are_resolved_inside_packaged_node_modules(setup):
    agent, _ = setup
    assert agent.config.sandbox_config["env"]["NEMOTRON_OPENCODE_DEPENDENCY_ROOT"] == str(
        LAUNCHER.parent / "node_modules"
    )


def test_only_media_hooks_are_added_to_native_config():
    original = {
        "provider": {
            "nemo_gym": {
                "options": {"baseURL": "http://example.invalid/v1"},
                "models": {"dummy_model": {"limit": {"context": 262144, "output": 262144}}},
            }
        },
        "compaction": {"auto": True},
        "agent": {"build": {"temperature": 1}},
    }
    saved = deepcopy(original)
    result = video_config(original, output_tokens=32768)
    assert original == saved
    assert result["compaction"] == original["compaction"] and "instructions" not in result
    assert result["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"]["output"] == 32768
    assert result["plugin"][0].endswith("nemotron_media_plugin.mjs")


def test_named_model_keeps_its_settings_and_limits():
    original = {
        "model": "nemo_gym/dummy_model",
        "provider": {
            "nemo_gym": {
                "options": {"baseURL": "http://example.invalid/v1"},
                "models": {
                    "dummy_model": {
                        "temperature": True,
                        "limit": {"context": 262144, "input": 262144, "output": 262144},
                        "interleaved": {"field": "reasoning"},
                    }
                },
            }
        },
    }
    result = video_config(original, output_tokens=32768, model="nemo/nemotron-3.5-super-ga")
    assert result["model"] == "nemo/nemotron-3.5-super-ga"
    assert list(result["provider"]) == ["nemo"]
    provider = result["provider"]["nemo"]
    assert provider["options"] == {"baseURL": "http://example.invalid/v1"}
    assert provider["npm"].endswith("nemotron_media_provider.mjs")
    model = provider["models"]["nemotron-3.5-super-ga"]
    assert model["name"] == "nemotron-3.5-super-ga" and model["interleaved"] == {"field": "reasoning"}
    assert model["limit"] == {"context": 262144, "output": 32768}
    assert model["attachment"] is True and model["modalities"]["input"] == ["text", "image", "video"]


def test_media_root_must_contain_the_workdir(setup):
    agent, _ = setup
    fields = agent.config.model_dump()
    for root in ("testbed", "/", "/testbed/../x", "/elsewhere"):
        with pytest.raises(ValidationError, match="media_root"):
            OpenCodeVideoAgentConfig.model_validate({**fields, "media_root": root})
    assert OpenCodeVideoAgentConfig.model_validate({**fields, "media_root": "/testbed"}).media_root == "/testbed"


async def test_videos_outside_the_workdir_share_the_media_workspace(setup, monkeypatch):
    agent, binding = setup
    agent.config.media_root = "/testbed"
    agent.config.media_tool_timeout_seconds = 1260
    valid = SimpleNamespace(return_code=0, error_type=None, stdout="")
    source_ok = SimpleNamespace(return_code=0, error_type=None, stdout=binding.sha256 + "  video\n")
    sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=[valid, valid, valid, source_ok, valid, source_ok]), download=AsyncMock()
    )
    uploads = {}

    async def upload(local, remote):
        uploads[remote] = json.loads(Path(local).read_text()) if str(remote).endswith(".json") else str(local)

    sandbox.upload = upload
    agent._sandbox_id_to_sandbox["sid"] = sandbox
    agent._sandbox_id_to_run_result["sid"] = {"opencode_failed": False}
    monkeypatch.setattr(OpenCodeSandboxedAgent, "responses", AsyncMock(return_value="native-response"))
    request = SimpleNamespace(
        state=SimpleNamespace(_ng_video_binding=binding),
        cookies={"sandbox_id": "sid"},
        session={SESSION_ID_KEY: "sid"},
    )
    assert await agent.responses(request, MagicMock()) == "native-response"
    assert uploads["/testbed/video/invented.mp4"] == str(binding.source)
    media = next(value for key, value in uploads.items() if key.endswith("media.json"))
    launcher = next(value for key, value in uploads.items() if key.endswith("launcher.json"))
    assert media["workspace_root"] == "/testbed" and launcher["timeout_seconds"] == 1260
    # The video's directory and OpenCode's working directory are both made before anything runs there.
    made = sandbox.exec.await_args_list[0].kwargs["command"]
    assert "/testbed/video" in made and "/testbed/work" in made


def test_pruned_media_is_not_reattached():
    state = {
        "status": "completed",
        "time": {"compacted": 123},
        "attachments": [{"mime": "video/mp4", "url": "data:video/mp4;base64,ZmFrZQ=="}],
    }
    assert tool_output(state) == "[Old tool result content cleared]"


def test_export_preserves_tool_id_and_media(setup):
    agent, _ = setup
    export = {
        "messages": [
            {
                "info": {"role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "watch_video",
                        "callID": "original-call",
                        "state": {
                            "input": {},
                            "output": "",
                            "status": "completed",
                            "attachments": [{"mime": "video/mp4", "url": "data:video/mp4;base64,ZmFrZQ=="}],
                        },
                    }
                ],
            }
        ]
    }
    items = agent._opencode_export_to_output_items(export)
    assert [item.call_id for item in items] == ["original-call", "original-call"]
    assert items[1].output == [{"type": "input_video", "video_url": "data:video/mp4;base64,ZmFrZQ=="}]


async def test_parent_loop_is_used_and_private_binding_is_scoped(setup, monkeypatch):
    agent, binding = setup
    native = AsyncMock(return_value="native-result")
    monkeypatch.setattr(OpenCodeSandboxedAgent, "_run", native)
    body = OpenCodeVideoRunRequest(
        setting="agentic",
        video_binding=binding,
        responses_create_params={"input": [{"role": "user", "content": "Watch the video."}]},
    )
    request = SimpleNamespace(state=SimpleNamespace())
    assert await agent._run(request, body) == "native-result"
    native.assert_awaited_once_with(request, body)
    assert not hasattr(request.state, "_ng_video_binding")
    assert "video_binding" not in body.responses_create_params.model_dump()


async def test_source_hash_mismatch_stops_before_native_loop(setup, monkeypatch):
    agent, binding = setup
    native = AsyncMock()
    monkeypatch.setattr(OpenCodeSandboxedAgent, "_run", native)
    body = OpenCodeVideoRunRequest(
        setting="agentic",
        video_binding=binding.model_copy(update={"sha256": "0" * 64}),
        responses_create_params={"input": [{"role": "user", "content": "Watch the video."}]},
    )
    with pytest.raises(ValueError, match="source video"):
        await agent._run(SimpleNamespace(state=SimpleNamespace()), body)
    native.assert_not_awaited()


def native_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_invented",
        created_at=0,
        model="policy_model",
        object="response",
        output=[],
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=True,
    )


@pytest.mark.parametrize("capture,changed", [(False, False), (False, True), (True, False)])
async def test_setup_calls_native_responses_and_preserves_private_evidence(setup, monkeypatch, capture, changed):
    agent, binding = setup
    agent.config.capture_model_inputs = capture
    valid = SimpleNamespace(return_code=0, error_type=None, stdout="")
    correct_source = SimpleNamespace(return_code=0, error_type=None, stdout=binding.sha256 + "  video\n")
    final_source = SimpleNamespace(
        return_code=0, error_type=None, stdout=("0" * 64 if changed else binding.sha256) + "  video\n"
    )
    results = [valid, valid, valid, correct_source, valid] + ([valid] if capture else []) + [final_source]
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=results), upload=AsyncMock(), download=AsyncMock())
    agent._sandbox_id_to_sandbox["sid"] = sandbox
    agent._sandbox_id_to_run_result["sid"] = {"opencode_failed": False}
    native = AsyncMock(return_value=native_response())
    monkeypatch.setattr(OpenCodeSandboxedAgent, "responses", native)
    request = SimpleNamespace(
        state=SimpleNamespace(_ng_video_binding=binding),
        cookies={"sandbox_id": "sid"},
        session={SESSION_ID_KEY: "sid"},
    )
    response = await agent.responses(request, MagicMock())
    native.assert_awaited_once()
    assert agent._sandbox_id_to_run_result["sid"]["opencode_failed"] is changed
    # A modified source is an execution failure the verifier must also see when the row is replayed.
    assert (response.metadata or {}).get(EXECUTION_FAILURE_METADATA_KEY) == ("true" if changed else None)
    assert sandbox.upload.await_count == 4  # Runtime, source, media config, private launcher.
    assert sandbox.download.await_count == int(capture)
    # The directories are made from the root: the image need not contain the workdir.
    assert sandbox.exec.await_args_list[0].kwargs["cwd"] == "/"


@pytest.mark.parametrize("stage", [0, 1, 2, 3, 4, 5])
async def test_setup_and_evidence_failures_propagate_without_silent_fallback(setup, monkeypatch, stage):
    agent, binding = setup
    agent.config.capture_model_inputs = True
    valid = SimpleNamespace(return_code=0, error_type=None, stdout="")
    correct_source = SimpleNamespace(return_code=0, error_type=None, stdout=binding.sha256 + "  video\n")
    failed = SimpleNamespace(return_code=1, error_type=None, stdout="")
    results = [valid, valid, valid, correct_source, valid, valid]
    results[stage] = failed
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=results), upload=AsyncMock(), download=AsyncMock())
    agent._sandbox_id_to_sandbox["sid"] = sandbox
    agent._sandbox_id_to_run_result["sid"] = {"opencode_failed": False}
    native = AsyncMock(return_value="native-response")
    monkeypatch.setattr(OpenCodeSandboxedAgent, "responses", native)
    request = SimpleNamespace(
        state=SimpleNamespace(_ng_video_binding=binding),
        cookies={"sandbox_id": "sid"},
        session={SESSION_ID_KEY: "sid"},
    )
    with pytest.raises(RuntimeError):
        await agent.responses(request, MagicMock())
    assert native.await_count == int(stage == 5)


async def test_private_binding_is_cleared_when_parent_fails(setup, monkeypatch):
    agent, binding = setup
    monkeypatch.setattr(OpenCodeSandboxedAgent, "_run", AsyncMock(side_effect=RuntimeError("native failure")))
    body = OpenCodeVideoRunRequest(
        setting="agentic",
        video_binding=binding,
        responses_create_params={"input": [{"role": "user", "content": "Watch the video."}]},
    )
    request = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="native failure"):
        await agent._run(request, body)
    assert not hasattr(request.state, "_ng_video_binding")


def test_nonvisual_tools_keep_their_original_text():
    state = {
        "status": "completed",
        "output": "native explanation",
        "attachments": [{"mime": "image/png", "url": "data:image/png;base64,ZmFrZQ=="}],
    }
    assert tool_output(state, "read")[0] == {"type": "input_text", "text": "native explanation"}
    assert tool_output({"output": None, "error": "failed"}) == "failed"


@pytest.mark.parametrize(
    "attachment",
    [
        {"mime": "video/mp4", "url": "https://example.invalid/video"},
        {"mime": "application/pdf", "url": "data:application/pdf;base64,ZmFrZQ=="},
    ],
)
def test_visual_tool_invalid_attachments_fail_loudly(attachment):
    with pytest.raises(ValueError):
        tool_output({"status": "completed", "attachments": [attachment]})


@pytest.mark.parametrize(
    "updates",
    [
        {"preinstalled_opencode": False},
        {"opencode_version": "1.0.0"},
        {"opencode_max_context_window": 131072},
        {"artifacts_dir": None},
        {"sandbox_config": {"workdir": "/"}},
        {
            "max_output_tokens": 262144,
            "media_profile": {"max_context_tokens": 262144, "reserved_output_tokens": 262144},
        },
    ],
)
def test_invalid_runtime_configs_fail_before_execution(setup, updates):
    agent, _ = setup
    with pytest.raises(ValidationError):
        OpenCodeVideoAgentConfig(**{**agent.config.model_dump(), **updates})


async def test_unseeded_responses_are_rejected(setup):
    agent, _ = setup
    with pytest.raises(ValueError, match="seeded"):
        await agent.responses(SimpleNamespace(state=SimpleNamespace()), MagicMock())


async def test_config_and_run_delegate_to_parent_hooks(setup, monkeypatch):
    agent, _ = setup
    original = {"provider": {"nemo_gym": {"models": {"dummy_model": {"limit": {"context": 262144}}}}}}
    create = AsyncMock(return_value=original)
    run = AsyncMock(return_value="native-result")
    monkeypatch.setattr(OpenCodeSandboxedAgent, "_create_opencode_config", create)
    monkeypatch.setattr(OpenCodeSandboxedAgent, "run", run)
    request, body = MagicMock(), MagicMock()
    assert (await agent._create_opencode_config(request))["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"][
        "output"
    ] == 32768
    await agent._create_opencode_config(request, image_input=True)
    assert await agent.run(request, body) == "native-result"
    assert create.await_args_list == [call(request, image_input=False), call(request, image_input=True)]
    run.assert_awaited_once_with(request, body)


async def test_changed_runtime_is_rejected_before_parent_loop(setup, monkeypatch):
    agent, binding = setup
    agent.config.media_runtime_sha256 = "0" * 64
    native = AsyncMock()
    monkeypatch.setattr(OpenCodeSandboxedAgent, "_run", native)
    body = OpenCodeVideoRunRequest(
        setting="agentic",
        video_binding=binding,
        responses_create_params={"input": [{"role": "user", "content": "Watch the video."}]},
    )
    with pytest.raises(ValueError, match="runtime"):
        await agent._run(SimpleNamespace(state=SimpleNamespace()), body)
    native.assert_not_awaited()


def test_duplicate_exported_call_ids_are_rejected(setup):
    agent, _ = setup
    part = {"type": "tool", "tool": "watch_video", "callID": "duplicate", "state": {"input": {}, "output": ""}}
    with pytest.raises(ValueError, match="Duplicate"):
        agent._opencode_export_to_output_items({"messages": [{"info": {"role": "assistant"}, "parts": [part, part]}]})


def test_unrelated_native_attachments_keep_text_export():
    assert (
        tool_output(
            {
                "status": "completed",
                "output": "native text",
                "attachments": [{"mime": "application/pdf", "url": "data:application/pdf;base64,ZmFrZQ=="}],
            },
            "read",
        )
        == "native text"
    )


@pytest.mark.parametrize("tool", ["watch_video", "view_video"])
def test_video_receipts_keep_their_text_before_the_video(setup, tool):
    agent, _ = setup
    video = {"mime": "video/mp4", "url": "data:video/mp4;base64,ZmFrZQ=="}
    state = {"input": {}, "status": "completed", "output": "Frames: 120 total", "attachments": [video]}
    part = {"type": "tool", "tool": tool, "callID": "original-call", "state": state}
    items = agent._opencode_export_to_output_items({"messages": [{"info": {"role": "assistant"}, "parts": [part]}]})
    assert [item.call_id for item in items] == ["original-call", "original-call"]
    # The model saw the receipt beside the video (media_bridge.mjs), so the export keeps both, text first.
    assert items[1].output == [
        {"type": "input_text", "text": "Frames: 120 total"},
        {"type": "input_video", "video_url": video["url"]},
    ]


def test_view_image_exports_only_its_image():
    image = {"mime": "image/png", "url": "data:image/png;base64,ZmFrZQ=="}
    state = {"status": "completed", "output": "Shown at full resolution.", "attachments": [image]}
    assert tool_output(state, "view_image") == [{"type": "input_image", "image_url": image["url"], "detail": "auto"}]


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


async def run_real_parent_loop(agent, binding, monkeypatch, *, image=False, run_code=0, finish=None, final_sha=None):
    """The stock OpenCode loop with only sandbox I/O and the model URL mocked."""
    monkeypatch.setattr(parent_app, "sandbox_server_url", lambda _name, **kwargs: "http://model.invalid")
    files = {}
    hashes = [binding.sha256, final_sha or binding.sha256]
    assistant = {"role": "assistant", **({"finish": finish} if finish else {})}
    export = {"messages": [{"info": {"role": "user"}, "parts": []}, {"info": assistant, "parts": []}]}

    async def upload(local, remote):
        files[str(remote)] = Path(local).read_bytes()

    async def download(remote, local):
        Path(local).write_bytes(files.get(str(remote)) or json.dumps(export).encode())

    async def run(*, command, **kwargs):
        stdout = ""
        if "opencode run" in command:
            stdout = "Shell: /bin/bash\n" + ("OpenCode run finished" if run_code == 0 else "OpenCode crashed")
            return SimpleNamespace(return_code=run_code, error_type=None, stdout=stdout, stderr="")
        if "session list" in command:
            stdout = '[{"id": "session-id"}]'
        elif command.startswith("sha256sum "):
            stdout = hashes.pop(0) + "  video\n"
        return SimpleNamespace(return_code=0, error_type=None, stdout=stdout, stderr="")

    sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=run), upload=AsyncMock(side_effect=upload), download=AsyncMock(side_effect=download)
    )
    agent._sandbox_id_to_sandbox["sid"] = sandbox
    request = SimpleNamespace(
        state=SimpleNamespace(_ng_video_binding=binding),
        cookies={"sandbox_id": "sid"},
        session={SESSION_ID_KEY: "sid"},
        json=AsyncMock(return_value={"responses_create_params": {"input": "Watch the video."}}),
    )
    content = [{"type": "input_text", "text": "Watch video/invented.mp4 and answer."}]
    if image:
        content.append(
            {
                "type": "input_image",
                "image_url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
                "detail": "auto",
            }
        )
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": content}])
    response = await agent.responses(request, body)
    launch = next(c.kwargs["command"] for c in sandbox.exec.await_args_list if "opencode run" in c.kwargs["command"])
    config = next(arg.split("=", 1)[1] for arg in shlex.split(launch) if arg.startswith("OPENCODE_CONFIG_CONTENT="))
    return response, json.loads(config), launch


@pytest.mark.parametrize("image", [False, True], ids=["text_only", "inline_image"])
async def test_the_real_parent_loop_launches_opencode_with_the_video_config(setup, monkeypatch, image):
    agent, binding = setup
    response, config, launch = await run_real_parent_loop(agent, binding, monkeypatch, image=image)
    model = config["provider"]["nemo_gym"]["models"]["dummy_model"]
    assert model["attachment"] is True and model["modalities"]["input"] == ["text", "image", "video"]
    assert model["limit"] == {"context": 262144, "output": 32768}
    assert config["provider"]["nemo_gym"]["npm"].endswith("nemotron_media_provider.mjs")
    assert any(plugin.endswith("nemotron_media_plugin.mjs") for plugin in config["plugin"])
    # The parent's attachment handling stays: lifted image limits and the -f upload only with an image.
    assert config.get("attachment") == ({"image": parent_app.OPENCODE_IMAGE_LIMITS_UNRESTRICTED} if image else None)
    assert ("-f /tmp/nemo-gym-attachments/image-1.png" in launch) is image
    assert agent._sandbox_id_to_run_result["sid"]["opencode_failed"] is False
    assert EXECUTION_FAILURE_METADATA_KEY not in (response.metadata or {})


@pytest.mark.parametrize(
    "failure",
    [{"run_code": 1}, {"finish": "length"}, {"final_sha": "0" * 64}],
    ids=["run_failure", "length_limit", "source_modified"],
)
@pytest.mark.parametrize("reward_zero", [True, False])
async def test_failed_runs_are_marked_for_the_verifier(setup, monkeypatch, failure, reward_zero):
    agent, binding = setup
    agent.config.execution_failure_reward_zero = reward_zero
    response, _, _ = await run_real_parent_loop(agent, binding, monkeypatch, **failure)
    assert agent._sandbox_id_to_run_result["sid"]["opencode_failed"] is True
    # Only a run graded as a forced zero carries the marker, so its replayed answer stays zero too.
    assert (response.metadata or {}).get(EXECUTION_FAILURE_METADATA_KEY) == ("true" if reward_zero else None)
