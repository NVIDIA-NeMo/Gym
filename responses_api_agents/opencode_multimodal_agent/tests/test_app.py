# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent
from responses_api_agents.opencode_agent.tests.test_app import _session_db
from responses_api_agents.opencode_multimodal_agent import app
from responses_api_agents.opencode_multimodal_agent.app import (
    OpenCodeMultimodalAgent,
    OpenCodeMultimodalAgentConfig,
    OpenCodeMultimodalRunRequest,
)
from responses_api_agents.opencode_multimodal_agent.media import final_answer, parse_visual_session, stage_inputs


def image_url() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), (20, 30, 40)).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def body(**updates):
    data = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Count the objects; $(do-not-execute)"},
                    {"type": "input_image", "image_url": image_url(), "detail": "auto"},
                ],
            }
        ],
        "temperature": 1.0,
        "top_p": 1.0,
    }
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(data | updates)


def config(**updates):
    data = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "name": "visual",
        "model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
        "resources_server": ResourcesServerRef(type="resources_servers", name="string_match"),
        "model": "policy",
    }
    return OpenCodeMultimodalAgentConfig(**(data | updates))


def agent(monkeypatch, **updates):
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.ensure_opencode", lambda version: None)
    monkeypatch.setattr(
        OpenCodeMultimodalAgent,
        "_resolve_model_base_url",
        lambda self, rollout_id=None: f"http://policy/ng-rollout/{rollout_id}/training-token-capture/v1",
    )
    return OpenCodeMultimodalAgent(config=config(**updates), server_client=MagicMock(spec=ServerClient))


def database(tmp_path, *, tool=True, child=False):
    user = ("user", [{"type": "file", "mime": "image/png", "url": image_url()}])
    assistant = {"role": "assistant", "time": {"created": 1000, "completed": 2000}}
    messages = [user]
    if tool:
        messages.append(
            (
                assistant,
                [
                    {
                        "type": "tool",
                        "callID": "read1",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/testbed/input-0.png"},
                            "output": "image",
                            "attachments": [{"type": "file", "mime": "image/png", "url": image_url()}],
                        },
                    }
                ],
            )
        )
    messages.append((assistant, [{"type": "text", "text": "\\boxed{42}"}]))
    sessions = [("root", None)]
    if child:
        sessions.append(("child", "root"))
        messages.append(("child", assistant, [{"type": "text", "text": "CHILD NOT FINAL"}]))
    return _session_db(tmp_path, messages, sessions=sessions)


def test_defaults_and_inheritance():
    cfg = config()
    assert issubclass(OpenCodeMultimodalAgent, OpenCodeAgent)
    assert (cfg.context_window, cfg.max_turns, cfg.opencode_version) == (49152, 8, "1.17.11")
    assert OpenCodeMultimodalAgent._run_opencode is OpenCodeAgent._run_opencode
    assert (cfg.concurrency, cfg.max_source_pixels, cfg.max_images) == (256, 150_000_000, 33)
    assert cfg.subagents_enabled and cfg.tool_rewards is not None


@pytest.mark.asyncio
async def test_inherited_startup_calls_parent_installer(monkeypatch):
    versions = []
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.ensure_opencode", versions.append)
    obj = OpenCodeMultimodalAgent(config=config(), server_client=MagicMock(spec=ServerClient))
    assert versions == []
    await obj._ensure_local_runtime()
    await obj._ensure_local_runtime()
    assert versions == ["1.17.11"]


def test_default_prompt_customization():
    prompt = config().system_prompt
    custom = (
        "If the image is present with the request, look at the image first. "
        "If the answer is clear, answer directly without code. Use code\n"
        "when the answer needs precision your eyes cannot give: exact counts, coordinates, pixel\n"
        "comparisons, or tracing a path. Before each tool call, say in one sentence what you are\n"
        "checking. You have python3 with pillow (PIL), numpy, scipy and opencv (cv2) via the bash\n"
        "tool and you have access to tools including bash, read, glob, grep, write, edit, todowrite."
    )
    assert prompt == custom
    assert "meticulous visual-reasoning agent" not in prompt
    assert "\\boxed{}" not in prompt


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_turns": 0},
        {"final_answer_reserve": 49152},
        {"extra_args": ["--continue"]},
        {"repo_dir": "/shared"},
        {"sandbox_provider": "another-provider"},
        {"opencode_config": {"agent": {"build": {"prompt": "replacement"}}}},
    ],
)
def test_rejects_unsafe_or_silent_overrides(overrides):
    with pytest.raises(ValidationError):
        config(**overrides)


def test_stock_provider_and_default_prompt_are_inherited(monkeypatch):
    obj = agent(monkeypatch)
    cfg = obj._build_opencode_config("r1")
    provider = cfg["provider"]["nemo"]
    assert provider["options"]["baseURL"] == "http://policy/ng-rollout/r1/training-token-capture/v1"
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    model = provider["models"]["policy"]
    assert model["interleaved"] == {"field": "reasoning_content"}
    assert model["modalities"]["input"] == ["text", "image"]
    assert cfg["agent"]["build"]["steps"] == 8
    assert "prompt" not in cfg["agent"]["build"]
    assert "instructions" not in cfg
    assert cfg["compaction"] == {"auto": False, "prune": False}
    assert cfg["agent"]["build"]["permission"]["webfetch"] == "deny"


def test_images_staged_losslessly_and_remote_urls_rejected(tmp_path):
    text, system, images = stage_inputs(body(), tmp_path, max_source_pixels=6, max_images=33)
    assert "$(do-not-execute)" in text and system is None
    with Image.open(images[0]) as image:
        assert image.size == (3, 2) and image.getpixel((0, 0)) == (20, 30, 40)
    with pytest.raises(ValueError, match="max_source_pixels"):
        stage_inputs(body(), tmp_path, max_source_pixels=5, max_images=33)
    remote = body(
        input=[
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://example.com/a.png", "detail": "auto"}],
            }
        ]
    )
    with pytest.raises(ValueError, match="remote"):
        stage_inputs(remote, tmp_path, max_source_pixels=10, max_images=33)


def test_replay_and_too_many_images_rejected(tmp_path):
    replay = body(
        input=[
            *body().model_dump()["input"],
            {"type": "function_call", "name": "read", "arguments": "{}", "call_id": "x"},
        ]
    )
    with pytest.raises(ValueError, match="replayed"):
        stage_inputs(replay, tmp_path, max_source_pixels=10, max_images=33)
    many = body(input=[*body().model_dump()["input"], *body().model_dump()["input"]])
    with pytest.raises(ValueError, match="max_images"):
        stage_inputs(many, tmp_path, max_source_pixels=10, max_images=1)


def test_artifact_keeps_tool_images_and_excludes_children(tmp_path):
    output, _, turns = parse_visual_session(database(tmp_path, child=True), expected_images=1, max_turns=8)
    assert turns == 2
    assert "CHILD NOT FINAL" not in str(output)
    result = next(item for item in output if item.type == "function_call_output")
    assert result.output[1]["type"] == "input_image"
    assert result.output[1]["image_url"] == image_url()
    assert final_answer([item.model_dump() for item in output]) == "\\boxed{42}"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("UPDATE part SET data='{}' WHERE json_extract(data, '$.type')='file'", "dropped"),
        (
            "UPDATE message SET data=json_set(data, '$.error', 'transport error') WHERE json_extract(data, '$.role')='assistant'",
            "error",
        ),
        (
            "UPDATE message SET data=json_remove(data, '$.time.completed') WHERE json_extract(data, '$.role')='assistant'",
            "Unfinished",
        ),
    ],
)
@pytest.mark.parametrize("max_turns", [8, None])
def test_invalid_runtime_evidence_fails_closed(tmp_path, mutation, match, max_turns):
    db = database(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute(mutation)
    with pytest.raises(ValueError, match=match):
        parse_visual_session(db, expected_images=1, max_turns=max_turns)


def test_turn_limit_is_counted_from_assistant_records(tmp_path):
    with pytest.raises(ValueError, match="max_turns"):
        parse_visual_session(database(tmp_path), expected_images=1, max_turns=1)


def test_unlimited_turns_preserve_long_trajectory_and_final_answer(tmp_path):
    user = ("user", [{"type": "file", "mime": "image/png", "url": image_url()}])
    assistant = {"role": "assistant", "time": {"created": 1000, "completed": 2000}}
    messages = [user] + [(assistant, [{"type": "text", "text": "Inspecting"}])] * 24
    messages.append((assistant, [{"type": "text", "text": "\\boxed{42}"}]))
    db = _session_db(tmp_path, messages)
    with pytest.raises(ValueError, match="max_turns"):
        parse_visual_session(db, expected_images=1, max_turns=20)
    output, _, turns = parse_visual_session(db, expected_images=1, max_turns=None)
    assert turns == 25
    assert final_answer([item.model_dump() for item in output]) == "\\boxed{42}"


def fake_runner(monkeypatch, database, captures, *, returncode=0):
    async def create_process(*args, **kwargs):
        data_home = Path(kwargs["env"]["XDG_DATA_HOME"])
        native_db = data_home / "opencode" / "opencode.db"
        native_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(database, native_db)
        project = Path(kwargs["cwd"])
        cfg = json.loads((project / "opencode.json").read_text())
        file_index, separator = args.index("--file"), args.index("--")
        captures.append(
            {
                "args": args,
                "env": kwargs["env"],
                "project": project,
                "config": cfg,
                "images": [Path(path).read_bytes() for path in args[file_index + 1 : separator]],
            }
        )
        await asyncio.sleep(0)
        process = MagicMock(returncode=returncode)
        process.communicate = AsyncMock(return_value=(b"", b"runtime diagnostic"))
        return process

    monkeypatch.setattr("responses_api_agents.opencode_agent.app.asyncio.create_subprocess_exec", create_process)


async def test_concurrent_episodes_use_parent_runner_and_keep_capture_scope(tmp_path, monkeypatch):
    captures = []
    fake_runner(monkeypatch, database(tmp_path), captures)
    obj = agent(monkeypatch, workspace_root=str(tmp_path / "workspaces"))
    episodes = await asyncio.gather(
        obj._create_episode(body(), rollout_id="r1"),
        obj._create_episode(body(temperature=0.5), rollout_id="r2"),
    )
    # Threaded image staging may finish in either order; match by rollout identity.
    captures.sort(key=lambda c: c["config"]["provider"]["nemo"]["options"]["baseURL"])
    assert [c["config"]["agent"]["build"]["temperature"] for c in captures] == [1.0, 0.5]
    for index, (capture, episode) in enumerate(zip(captures, episodes), 1):
        assert f"/r{index}/" in capture["config"]["provider"]["nemo"]["options"]["baseURL"]
        assert capture["args"][0] == "opencode"
        assert "$(do-not-execute)" in capture["args"][-1]
        assert len(capture["images"]) == 1
        assert capture["args"][-1] == obj.config.system_prompt + "\n\nCount the objects; $(do-not-execute)"
        assert "prompt" not in capture["config"]["agent"]["build"]
        assert "instructions" not in capture["config"]
        assert "OPENCODE_VISUAL_MAX_TURNS" not in capture["env"]
        assert "runtime" not in capture["config"]
        assert not capture["project"].exists()
        assert episode.response.opencode_evidence.tool_reward.tool_call_count == 1
        assert episode.response.opencode_evidence.turns_used == 2


@pytest.mark.parametrize("base_reward,expected", [(1.0, 1.1), (0.0, 0.02)])
@pytest.mark.parametrize("answer", ["42", "<think>\\boxed{wrong}</think>42"])
async def test_run_routes_original_verifier_and_shapes_reward(monkeypatch, base_reward, expected, answer):
    obj = agent(monkeypatch)
    seed, reply, verify = [MagicMock() for _ in range(3)]
    seed.cookies = {"session": "seed"}
    reply.cookies = {"session": "reply"}
    obj.server_client.post = AsyncMock(side_effect=[seed, reply, verify])
    raw = {
        "id": "resp",
        "created_at": 1,
        "object": "response",
        "model": "policy",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": "early",
                "status": "completed",
                "content": [{"type": "output_text", "text": "not the answer", "annotations": []}],
            },
            {
                "type": "message",
                "role": "assistant",
                "id": "final",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            },
        ],
        "opencode_evidence": {
            "turns_used": 2,
            "finished_naturally": True,
            "tool_reward": {
                "base_reward": 0,
                "auxiliary_reward": 0.02,
                "exploration_reward": 0.02,
                "penalty_reward": 0,
                "bash_success_count": 1,
                "tool_call_count": 1,
                "tool_error_count": 0,
                "duplicate_tool_call_count": 0,
            },
        },
    }
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=[raw, {"reward": base_reward}]))
    request = MagicMock(cookies={})
    run_body = OpenCodeMultimodalRunRequest(
        responses_create_params=body(),
        image_tools_base_agent_ref={"name": "mcqa_simple_agent", "type": "responses_api_agents"},
    )
    result = await obj.run(request, run_body)
    calls = obj.server_client.post.call_args_list
    assert calls[0].kwargs["server_name"] == calls[2].kwargs["server_name"] == "mcqa"
    assert len(calls[2].kwargs["json"]["response"]["output"]) == 1
    assert calls[2].kwargs["json"]["response"]["output"][0]["content"][0]["text"] == "42"
    assert calls[2].kwargs["cookies"] == reply.cookies
    assert result.reward == pytest.approx(expected)
    assert result.resolved == (base_reward == 1.0)
    assert len(result.response.output) == 2
    assert result.response.output[-1].content[0].text == answer


def test_capture_header_is_discovered_and_joined(tmp_path):
    from responses_api_agents.opencode_agent.app import _parse_opencode_session
    from responses_api_agents.opencode_agent.tests.test_model_call_correlation import (
        test_harness_property_reaches_model_capture_and_links_message as check_capture,
    )

    check_capture(tmp_path, _parse_opencode_session, "opencode_multimodal_agent")


@pytest.mark.parametrize("last_parts", [[{"type": "reasoning", "text": "still thinking"}], []])
def test_reasoning_only_or_empty_final_never_verifies_earlier_text(tmp_path, last_parts):
    assistant = {"role": "assistant", "time": {"created": 1000, "completed": 2000}}
    db = _session_db(
        tmp_path,
        [
            ("user", [{"type": "file", "mime": "image/png", "url": image_url()}]),
            (assistant, [{"type": "text", "text": "earlier explanation"}]),
            (assistant, last_parts),
        ],
    )
    output, _, _ = parse_visual_session(db, expected_images=1, max_turns=8)
    assert final_answer([item.model_dump() for item in output]) == ""


async def test_parent_workspace_cleaned_on_execution_failure(tmp_path, monkeypatch):
    captures = []
    fake_runner(monkeypatch, database(tmp_path), captures, returncode=125)
    with pytest.raises(RuntimeError, match="runtime diagnostic"):
        await agent(monkeypatch, workspace_root=str(tmp_path / "workspaces"))._create_episode(body())
    assert not captures[0]["project"].exists()


async def test_request_sampling_cap_and_instructions_do_not_mutate_shared_config(tmp_path, monkeypatch):
    captures = []
    fake_runner(monkeypatch, database(tmp_path), captures)
    obj = agent(monkeypatch, workspace_root=str(tmp_path / "workspaces"), system_prompt="Configured instructions")
    request = body(
        top_p=0.8,
        max_output_tokens=128,
        input=[{"role": "system", "content": "Request instructions"}, *body().model_dump()["input"]],
    )
    await obj._create_episode(request)
    assert captures[0]["config"]["agent"]["build"]["top_p"] == 0.8
    assert captures[0]["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "128"
    assert captures[0]["config"]["provider"]["nemo"]["models"]["policy"]["limit"]["output"] == 128
    assert captures[0]["args"][-1] == (
        "Configured instructions\n\nRequest instructions\n\nCount the objects; $(do-not-execute)"
    )
    assert "instructions" not in captures[0]["config"]
    assert obj.config.max_output_tokens == 49152
    assert obj.config.extra_args == [] and obj.config.opencode_config == {}


def test_responses_http_keeps_typed_evidence(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from omegaconf import OmegaConf

    captures = []
    fake_runner(monkeypatch, database(tmp_path), captures)
    obj = agent(monkeypatch, workspace_root=str(tmp_path / "workspaces"))
    obj.server_client = ServerClient(
        head_server_config={"host": "localhost", "port": 0}, global_config_dict=OmegaConf.create({})
    )
    with TestClient(obj.setup_webserver()) as client:
        result = client.post("/v1/responses", json=body().model_dump())
    assert result.status_code == 200, result.text
    assert result.json()["opencode_evidence"]["turns_used"] == 2
    assert result.json()["output"][1]["output"][1]["type"] == "input_image"


def test_training_opt_in_preserves_rollout_capture_self_call(monkeypatch):
    from omegaconf import OmegaConf

    obj = agent(monkeypatch, token_id_capture=True)
    obj.server_client = ServerClient(
        head_server_config={"host": "localhost", "port": 0},
        global_config_dict=OmegaConf.create({"token_id_capture": {"enabled": True}}),
    )
    run = OpenCodeMultimodalRunRequest(responses_create_params=body(), _ng_task_index=0, _ng_rollout_index=0)
    assert obj.url_path_for_run("/v1/responses", run) == "/ng-rollout/0-0/training-token-capture/v1/responses"


async def test_usage_includes_child_reasoning_without_double_counting(tmp_path, monkeypatch):
    db = _session_db(
        tmp_path,
        [
            ("user", [{"type": "file", "mime": "image/png", "url": image_url()}]),
            (
                "assistant",
                [
                    {"type": "text", "text": "Root answer"},
                    {
                        "type": "step-finish",
                        "tokens": {
                            "input": 100,
                            "output": 10,
                            "reasoning": 3,
                            "cache": {"read": 4, "write": 2},
                        },
                    },
                ],
            ),
            (
                "child",
                "assistant",
                [
                    {"type": "text", "text": "Child answer"},
                    {"type": "step-finish", "tokens": {"input": 20, "output": 5, "reasoning": 2}},
                ],
            ),
        ],
        sessions=[("root", None), ("child", "root")],
    )
    fake_runner(monkeypatch, db, [])
    obj = agent(monkeypatch, workspace_root=str(tmp_path / "workspaces"))
    episode = await obj._create_episode(body(), rollout_id="usage")
    usage = episode.response.usage
    assert usage.input_tokens == 126
    assert usage.output_tokens == 20
    assert usage.output_tokens_details.reasoning_tokens == 5
    assert usage.total_tokens == 146
    assert "Child answer" not in str(episode.response.output)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>\\boxed{oven}</think>stove", "stove"),
        ("<think>Answer: oven</think>Final answer: stove", "Final answer: stove"),
        ("<thinking>wrong\nanswer</thinking>42", "42"),
        ("<think>\\boxed{42}", ""),
        ("wrong</think>42", "42"),
        ("42", "42"),
    ],
)
def test_final_answer_excludes_inline_reasoning(text, expected):
    output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
    assert final_answer(output, strip_reasoning=True) == expected
    assert output[0]["content"][0]["text"] == text


def test_training_image_policy_and_override(monkeypatch):
    obj = agent(monkeypatch)
    assert obj._build_opencode_config()["attachment"]["image"] == {
        "auto_resize": True,
        "max_width": 8000,
        "max_height": 8000,
        "max_base64_bytes": 1024**3,
    }
    obj = agent(monkeypatch, image_policy={"auto_resize": False})
    assert obj._build_opencode_config()["attachment"]["image"]["auto_resize"] is False


@pytest.mark.asyncio
async def test_multimodal_rejects_parent_text_only_session_path(monkeypatch):
    obj = agent(monkeypatch)
    with pytest.raises(NotImplementedError, match="does not implement episode sessions"):
        await obj._seed_agent_session_state(MagicMock())
    with pytest.raises(NotImplementedError, match="does not implement episode sessions"):
        await obj._close_agent_session_state(MagicMock())
