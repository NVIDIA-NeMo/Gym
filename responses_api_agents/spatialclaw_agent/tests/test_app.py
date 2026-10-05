# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.spatialclaw_agent.app import (
    SpatialClawAgent,
    SpatialClawAgentConfig,
    SpatialClawAgentRunRequest,
    _AiohttpChatCompletions,
    _config_path,
    _extract_instruction_and_images,
    _frame_cache_dir,
    _install_gym_llm_transport,
    _metadata_video_references,
    _resolve_video_path,
    _session_id,
)


def test_base_config_declares_datasets_for_benchmark_inheritance() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "spatialclaw_agent.yaml"
    config = OmegaConf.load(config_path)
    assert config.spatialclaw_agent.responses_api_agents.spatialclaw_agent.datasets == []


def test_extracts_text_instruction_and_video_metadata() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "instructions": "Follow the answer format.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Question?\nA. one\nB. two"}]}],
            "metadata": {"video_path": "clip.mp4"},
        }
    )
    instruction, images = _extract_instruction_and_images(body)
    assert instruction == "Follow the answer format.\n\nQuestion?\nA. one\nB. two"
    assert images == []
    assert _metadata_video_references(body) == ["clip.mp4"]


def test_extracts_image_content() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "file:///data/reference.png",
                            "detail": "auto",
                        },
                        {"type": "input_text", "text": "Track it."},
                    ],
                }
            ]
        }
    )
    assert _extract_instruction_and_images(body) == ("Track it.", ["file:///data/reference.png"])


def test_video_metadata_keys_are_mutually_exclusive() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="question",
        metadata={"video_path": "a.mp4", "video_data": "data:video/mp4;base64,AA=="},
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _metadata_video_references(body)


def test_video_paths_accepts_json_list() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="question",
        metadata={"video_paths": json.dumps(["a.mp4", "b.mp4"])},
    )
    assert _metadata_video_references(body) == ["a.mp4", "b.mp4"]


async def test_materializes_grouped_and_reference_images(tmp_path) -> None:
    for name in ("first.png", "second.png", "reference.png"):
        (tmp_path / name).write_bytes(b"image")
    config = SpatialClawAgentConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="spatialclaw_test",
        resources_server=ResourcesServerRef(type="resources_servers", name="scorer"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        dataset_config="mmsivideo_spatialclaw_256f",
        video_root=str(tmp_path),
    )
    agent = SpatialClawAgent(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Compare the views.",
        metadata={
            "image_groups": json.dumps([["first.png"], ["second.png"]]),
            "ref_image_paths": json.dumps(["reference.png"]),
            "frame_indices_groups": json.dumps([[0], [0]]),
        },
    )
    instruction, images, metadata = await agent._materialize_inputs(
        body,
        tmp_path / "session",
        SimpleNamespace(),
    )
    assert instruction == "Compare the views."
    assert images == [str((tmp_path / "first.png").resolve()), str((tmp_path / "second.png").resolve())]
    assert metadata["image_groups"] == [[images[0]], [images[1]]]
    assert metadata["frame_indices_groups"] == [[0], [0]]
    assert metadata["ref_images"] == [str((tmp_path / "reference.png").resolve())]


def test_relative_video_path_is_confined_to_root(tmp_path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    clip = video_root / "clip.mp4"
    clip.write_bytes(b"video")
    assert _resolve_video_path("clip.mp4", str(video_root)) == str(clip.resolve())
    with pytest.raises(ValueError, match="escapes"):
        _resolve_video_path("../outside.mp4", str(video_root))


def test_session_id_rejects_path_traversal() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="question",
        metadata={"spatialclaw_session_id": "../escape"},
    )
    with pytest.raises(ValueError, match="filename-safe"):
        _session_id(body)


def test_frame_cache_key_includes_sampling_protocol(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    first = _frame_cache_dir(
        str(video), str(tmp_path / "cache"), SimpleNamespace(video_max_fps=1, video_frame_resize_short_edge=None)
    )
    second = _frame_cache_dir(
        str(video), str(tmp_path / "cache"), SimpleNamespace(video_max_fps=2, video_frame_resize_short_edge=None)
    )
    assert first != second


def test_config_path_supports_short_and_repo_relative_names(tmp_path) -> None:
    config = tmp_path / "spatial_agent" / "config" / "dataset" / "sample.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    assert _config_path(tmp_path, "sample", "dataset") == str(config.resolve())
    assert _config_path(tmp_path, "spatial_agent/config/dataset/sample.json", "dataset") == str(config.resolve())


def test_install_transport_forces_vllm_kwargs_without_discovery() -> None:
    client = SimpleNamespace(_client_pool={"old": object()}, _is_vllm=False)
    _install_gym_llm_transport(client)
    assert client._client_pool == {}
    assert client._is_vllm is True
    first = client._get_client("http://model/v1")
    assert first is client._get_client("http://model/v1")


async def test_aiohttp_adapter_encodes_openai_extra_body(monkeypatch) -> None:
    captured = {}
    response = MagicMock()

    async def fake_request(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("responses_api_agents.spatialclaw_agent.app.http_request", fake_request)
    monkeypatch.setattr("responses_api_agents.spatialclaw_agent.app.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "responses_api_agents.spatialclaw_agent.app.get_response_json",
        AsyncMock(
            return_value={
                "id": "chat-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
                "created": 1,
                "model": "model",
                "object": "chat.completion",
            }
        ),
    )
    adapter = _AiohttpChatCompletions("http://model/v1")
    result = await adapter.create(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    )
    assert result.choices[0].message.content == "done"
    assert captured["url"] == "http://model/v1/chat/completions"
    assert json.loads(captured["json"]["metadata"]["extra_body"]) == {"top_k": 20}
    assert json.loads(captured["json"]["metadata"]["chat_template_kwargs"]) == {"enable_thinking": True}
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["cookies"] is None
    assert adapter.cookies is response.cookies


async def test_responses_runs_spatialclaw_and_returns_gym_response(monkeypatch, tmp_path) -> None:
    close = AsyncMock()
    shutdown_all = AsyncMock()

    class FakeWorkflow:
        def __init__(self, _config):
            self.llm_client = SimpleNamespace(
                _client_pool={},
                _is_vllm=False,
                close=close,
            )
            self._kernel_pool = SimpleNamespace(shutdown_all=shutdown_all)

        async def arun(self, **kwargs):
            assert kwargs["instruction"] == "Which option is correct?"
            assert kwargs["video_source"] == "/videos/clip.mp4"
            return {
                "final_answer": {"text": "B"},
                "termination_reason": "answer_returned",
                "step_count": 3,
                "total_tool_calls": 2,
                "usage": {
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 4,
                    "total_reasoning_tokens": 2,
                },
            }

    config_module = ModuleType("spatial_agent.config")
    config_module.set_config = MagicMock()
    workflow_module = ModuleType("spatial_agent.workflow")
    workflow_module.SpatialAgentWorkflow = FakeWorkflow
    monkeypatch.setitem(sys.modules, "spatial_agent.config", config_module)
    monkeypatch.setitem(sys.modules, "spatial_agent.workflow", workflow_module)
    monkeypatch.setattr(
        SpatialClawAgent,
        "_resolve_spatialclaw_root",
        AsyncMock(return_value=tmp_path),
    )
    monkeypatch.setattr(
        SpatialClawAgent,
        "_build_spatialclaw_config",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(
        SpatialClawAgent,
        "_materialize_inputs",
        AsyncMock(
            return_value=(
                "Which option is correct?",
                ["/frames/frame-1.jpg"],
                {
                    "video_source": "/videos/clip.mp4",
                    "video_sources_per_video": ["/videos/clip.mp4"],
                    "frame_indices": [0],
                },
            )
        ),
    )

    config = SpatialClawAgentConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="spatialclaw_test",
        resources_server=ResourcesServerRef(type="resources_servers", name="vlm_test"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        dataset_config="videomme_spatialclaw_contextsafe",
        video_root="/videos",
        model_name="test-model",
        workspace_root=str(tmp_path / "workspaces"),
        keep_workspaces=False,
    )
    assert config.concurrency == 1
    agent = SpatialClawAgent(config=config, server_client=MagicMock(spec=ServerClient))
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Which option is correct?",
        metadata={"video_path": "clip.mp4", "spatialclaw_session_id": "sample-1"},
    )
    response = await agent.responses(MagicMock(), body)
    assert response.output_text == "B"
    assert response.metadata["spatialclaw_turns"] == "3"
    assert response.usage.total_tokens == 14
    close.assert_awaited_once()
    shutdown_all.assert_awaited_once()
    assert not (tmp_path / "workspaces" / "sample-1").exists()


async def test_run_keeps_resource_session_cookie_for_verification(monkeypatch) -> None:
    config = SpatialClawAgentConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="spatialclaw_test",
        resources_server=ResourcesServerRef(type="resources_servers", name="vlm_test"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        dataset_config="videomme_spatialclaw_contextsafe",
        video_root="/videos",
    )
    server_client = MagicMock(spec=ServerClient)
    seed_response = MagicMock(cookies={"resource_session": "seeded"})
    agent_response = MagicMock(cookies={})
    verify_response = MagicMock(cookies={})
    server_client.post = AsyncMock(side_effect=[seed_response, agent_response, verify_response])

    response_params = NeMoGymResponseCreateParamsNonStreaming(input="Which option is correct?")
    agent_json = {
        "id": "spatialclaw-sample",
        "created_at": 1,
        "model": "test-model",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "B", "annotations": []}],
            }
        ],
        "metadata": {
            "spatialclaw_turns": "3",
            "spatialclaw_termination_reason": "answer_returned",
        },
    }
    verified_json = {
        "responses_create_params": response_params.model_dump(),
        "response": agent_json,
        "reward": 1.0,
    }
    monkeypatch.setattr("responses_api_agents.spatialclaw_agent.app.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "responses_api_agents.spatialclaw_agent.app.get_response_json",
        AsyncMock(side_effect=[agent_json, verified_json]),
    )

    request = MagicMock()
    request.cookies = {"incoming": "cookie"}
    result = await SpatialClawAgent(config=config, server_client=server_client).run(
        request,
        SpatialClawAgentRunRequest(responses_create_params=response_params),
    )

    assert result.reward == 1.0
    assert result.turns_used == 3
    assert result.termination_reason == "answer_returned"
    assert server_client.post.call_args_list[0].kwargs["cookies"] == {"incoming": "cookie"}
    assert server_client.post.call_args_list[1].kwargs["cookies"] == {"resource_session": "seeded"}
    assert server_client.post.call_args_list[2].kwargs["cookies"] == {"resource_session": "seeded"}
