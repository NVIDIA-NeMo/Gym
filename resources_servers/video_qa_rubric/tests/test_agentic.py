# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agentic setting: rows that bind their video, the config, and the sandbox the server lends each episode.

The lending tests run on the local sandbox provider in a temporary directory, with video_mcqa's fixtures
(tests/test_sandbox.py), because the server lends the same sandbox as the video MCQA servers.
"""

import base64
import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.video_mcqa.tests.helpers import response
from resources_servers.video_mcqa.tests.test_sandbox import CLIP, CLIP_SHA256, host, stopped  # noqa: F401
from resources_servers.video_qa_rubric.app import (
    VideoQARubricResourcesServer,
    VideoQARubricResourcesServerConfig,
    VideoQARubricVerifyRequest,
)
from resources_servers.video_qa_rubric.task import AGENTIC_PROMPT, prepare_agentic
from resources_servers.video_qa_rubric.tests.test_app import RUBRIC
from resources_servers.video_qa_rubric.tests.test_task import EXAMPLES, SERVER, report
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig


SERVERS = SERVER.parent
AGENTIC_EXAMPLES = [
    json.loads(line) for line in (SERVER / "data" / "example_agentic.jsonl").read_text(encoding="utf-8").splitlines()
]
QUESTION = "In which direction does the square move?"


def agentic_row(video: Path, sandbox_path: str = "videos/shapes.mp4", **binding: object) -> dict:
    row = dict(
        prepare_agentic("video_qa_rubric:one", QUESTION, copy.deepcopy(RUBRIC), video_path=f"/testbed/{sandbox_path}")
    )
    row["video_binding"] = {"source": str(video), "sandbox_path": sandbox_path, "sha256": CLIP_SHA256, **binding}
    return row


def test_an_agentic_row_names_the_video_in_the_sandbox_never_the_rubric() -> None:
    row = agentic_row(CLIP)
    assert (row["task_id"], row["setting"]) == ("video_qa_rubric:one", "agentic")
    assert row["responses_create_params"] == {
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": AGENTIC_PROMPT.format(video_path="/testbed/videos/shapes.mp4", question=QUESTION),
                    }
                ],
            }
        ]
    }
    assert row["verifier_metadata"] == {"question": QUESTION, "rubric": RUBRIC}
    assert "weight_points" not in json.dumps(row["responses_create_params"])
    assert report([row]).clean
    for video_path in ("", "videos/a.mp4\nrm -rf /"):
        with pytest.raises(ValueError, match="video_path"):
            prepare_agentic("video_qa_rubric:one", QUESTION, RUBRIC, video_path=video_path)


@pytest.mark.parametrize(
    "binding",
    [{"sandbox_path": "/etc/video.mp4"}, {"sandbox_path": "../outside.mp4"}, {"sha256": "short"}, {"url": "x"}],
    ids=["absolute_path", "escaping_path", "malformed_digest", "unknown_key"],
)
def test_a_malformed_video_binding_fails_validation(binding) -> None:
    row = agentic_row(CLIP)
    row["video_binding"].update(binding)
    assert report([row]).error_rows == 1


def test_the_agentic_examples_ask_the_single_turn_questions_about_the_committed_clips() -> None:
    assert report(AGENTIC_EXAMPLES).clean and len(AGENTIC_EXAMPLES) == 5
    for agentic, single_turn in zip(AGENTIC_EXAMPLES, EXAMPLES, strict=True):
        assert (agentic["task_id"], agentic["setting"]) == (single_turn["task_id"], "agentic")
        assert agentic["verifier_metadata"] == single_turn["verifier_metadata"]
        binding = agentic["video_binding"]
        clip = SERVERS.parent / binding["source"]
        assert clip.parent == SERVERS / "video_mcqa" / "data" / "example_videos"
        assert hashlib.sha256(clip.read_bytes()).hexdigest() == binding["sha256"]
        # The single-turn row inlines the same clip.
        (video,) = [
            part["video_url"]
            for part in single_turn["responses_create_params"]["input"][0]["content"]
            if part["type"] == "input_video"
        ]
        assert base64.b64decode(video.split(",", 1)[1]) == clip.read_bytes()
        text = agentic["responses_create_params"]["input"][0]["content"][0]["text"]
        assert text == AGENTIC_PROMPT.format(
            video_path=f"/testbed/{binding['sandbox_path']}", question=agentic["verifier_metadata"]["question"]
        )


def test_the_agentic_config_runs_the_video_mcqa_agent_and_sandbox_with_a_loopback_judge(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VIDEO_MCQA_SANDBOX_IMAGE", "registry.example/video-mcqa-sandbox:test")
    values = {
        "video_qa_rubric_agentic_video_roots": [str(tmp_path)],
        "videomme_v2_agentic_video_roots": [str(tmp_path)],
        "videomme_v2_agentic_cohort_manifest": str(tmp_path / "cohort.json"),
        "policy_model_name": "policy",
        "judge_base_url": "https://judge.example.invalid/v1",
        "judge_api_key": "placeholder",
        "judge_model_name": "judge-model",
    }
    loaded = OmegaConf.load(SERVER / "configs" / "video_qa_rubric_agentic.yaml")
    assert list(loaded.config_paths) == ["nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"]
    config = OmegaConf.to_container(OmegaConf.merge(loaded, OmegaConf.create(values)), resolve=True)
    name = "video_qa_rubric_agentic"
    settings = config[name]["resources_servers"]["video_qa_rubric"]
    resource = VideoQARubricResourcesServerConfig(name=name, port=8001, **settings)
    assert resource.sandbox_provider == "sandbox" and resource.sandbox_network_access == "model_only"
    assert resource.sandbox_config.image == "registry.example/video-mcqa-sandbox:test"
    assert resource.video_roots == [tmp_path] and resource.judge_model_server.name == "video_qa_rubric_judge"
    judge = config["video_qa_rubric_judge"]["responses_api_models"]["openai_model"]
    # The lent sandbox may reach every port of the policy model's host; the judge stays off that address.
    assert judge["host"] == resource.host == "127.0.0.1" and judge["openai_model"] == "judge-model"
    agent_config = config[f"{name}_opencode"]["responses_api_agents"]["opencode_agent"]
    agent = OpenCodeAgentConfig(name=f"{name}_opencode", port=8002, **agent_config)
    assert agent.resources_server.name == name and agent.host == "127.0.0.1"
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=f"{name}_environment",
        port=8003,
        **config[f"{name}_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert environment.agent_server.name == f"{name}_opencode" and environment.resources_server.name == name
    # The same agent, sandbox and episode settings as the agentic video MCQA configs: only the names differ.
    v2 = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(SERVERS / "videomme_v2" / "configs" / "videomme_v2_agentic.yaml"), OmegaConf.create(values)
        ),
        resolve=True,
    )
    v2_agent = v2["videomme_v2_agentic_opencode"]["responses_api_agents"]["opencode_agent"]
    assert {k: v for k, v in agent_config.items() if k not in ("description", "resources_server")} == {
        k: v for k, v in v2_agent.items() if k not in ("description", "resources_server")
    }
    v2_resources = v2["videomme_v2_agentic"]["resources_servers"]["videomme_v2"]
    sandbox_keys = [key for key in v2_resources if key.startswith("sandbox_")]
    assert sandbox_keys and {key: settings[key] for key in sandbox_keys} == {
        key: v2_resources[key] for key in sandbox_keys
    }
    v2_environment = v2["videomme_v2_agentic_environment"]["environment_servers"]["single_agent_turn_legacy"]
    rubric_environment = config[f"{name}_environment"]["environment_servers"]["single_agent_turn_legacy"]
    assert {k: v for k, v in rubric_environment.items() if k not in ("resources_server", "agent_server")} == {
        k: v for k, v in v2_environment.items() if k not in ("resources_server", "agent_server")
    }


def lending_server(host: Path, **fields) -> VideoQARubricResourcesServer:
    config = VideoQARubricResourcesServerConfig(
        name="video_qa_rubric",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        **{
            "sandbox_provider": "sandbox",
            "sandbox_config": {"image": "unused-by-the-local-provider"},
            "sandbox_media_root": str(host / "testbed"),
            "sandbox_workdir": str(host / "testbed" / "work"),
            "video_roots": [str(host / "videos")],
            "video_tools": {"command": [sys.executable, "-c", "import sys; sys.exit(0)"]},
            **fields,
        },
    )
    return VideoQARubricResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def seed(host: Path, session_id: str = "session", **binding: object) -> ResourcesSeedSessionRequest:
    row = agentic_row(host / "videos" / "shapes.mp4", **binding)
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="video_qa_rubric:agentic", task_id=row["task_id"]),
        task_data={key: value for key, value in row.items() if key != "responses_create_params"},
    )


def test_the_server_lends_each_agentic_episode_a_sandbox_with_its_video(host, stopped) -> None:  # noqa: F811
    server = lending_server(host)
    client = TestClient(server.setup_webserver())
    seeded = client.post("/seed_session", json=seed(host).model_dump(mode="json"))
    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_access"]["workdir"] == str(host / "testbed" / "work")
    assert (host / "testbed" / "videos" / "shapes.mp4").read_bytes() == CLIP.read_bytes()
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=seed(host).episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1 and server._lent_sandboxes.get("session") is None
    # A row whose video does not have its digest is refused, and its sandbox stopped.
    refused = TestClient(server.setup_webserver(), raise_server_exceptions=False).post(
        "/seed_session", json=seed(host, "other", sha256="0" * 64).model_dump(mode="json")
    )
    assert refused.status_code == 422 and "video_binding.sha256" in refused.text and len(stopped) == 2


def test_the_lent_sandbox_follows_the_session_contract(host, stopped) -> None:  # noqa: F811
    server = lending_server(host)
    check_resources_session_contract(
        server.setup_webserver(),
        seed(host, "contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: server._lent_sandboxes.get(resources_session_id) is not None,
    )


def test_without_a_sandbox_provider_the_server_grades_without_sessions(host) -> None:  # noqa: F811
    server = lending_server(host, sandbox_provider=None)
    assert server._lent_sandboxes is None and server.multiple_workers_unsupported_reason() is None
    seeded = TestClient(server.setup_webserver()).post("/seed_session", json=seed(host).model_dump(mode="json"))
    assert seeded.status_code == 200 and seeded.json()["sandbox_access"] is None
    # An agentic row with its binding is a valid verify request either way.
    request = VideoQARubricVerifyRequest.model_validate({**agentic_row(CLIP), "response": response("Right.")})
    assert request.setting == "agentic" and request.model_dump()["video_binding"]["sha256"] == CLIP_SHA256
