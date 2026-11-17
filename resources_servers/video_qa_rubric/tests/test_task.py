# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rows: the single-turn builder, the task-data schema the verifier's request mirrors, and the example data."""

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import DatasetConfig
from nemo_gym.data import TaskDataValidator, load_task_data_schema
from resources_servers.video_mcqa.tests.helpers import response
from resources_servers.video_qa_rubric.app import (
    VideoQARubricMetadata,
    VideoQARubricResourcesServerConfig,
    VideoQARubricVerifyRequest,
)
from resources_servers.video_qa_rubric.task import PROMPT, prepare_single_turn
from resources_servers.video_qa_rubric.task_data import TaskData
from resources_servers.video_qa_rubric.tests.test_app import RUBRIC
from responses_api_agents.simple_agent.app import SimpleAgentConfig


SERVER = Path(__file__).resolve().parents[1]
EXAMPLES = [json.loads(line) for line in (SERVER / "data" / "example.jsonl").read_text(encoding="utf-8").splitlines()]


def report(rows: list[dict]):
    validator = TaskDataValidator("video_qa_rubric", load_task_data_schema(SERVER), "rows.jsonl")
    for index, row in enumerate(rows):
        validator.validate_row(index, row)
    return validator.report


def test_a_single_turn_row_shows_the_model_the_video_and_question_only() -> None:
    rubric = copy.deepcopy(RUBRIC)
    row = prepare_single_turn("video_qa_rubric:one", "Which way does it move?", rubric, video_url="file:///v.mp4")
    assert (row["task_id"], row["setting"]) == ("video_qa_rubric:one", "single_turn")
    assert row["responses_create_params"] == {
        "tools": [],
        "tool_choice": "none",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_video", "video_url": "file:///v.mp4"},
                    {"type": "input_text", "text": PROMPT.format(question="Which way does it move?")},
                ],
            }
        ],
    }
    assert row["verifier_metadata"] == {"question": "Which way does it move?", "rubric": rubric}
    assert row["verifier_metadata"]["rubric"] is rubric
    assert "Names the object" not in json.dumps(row["responses_create_params"])


@pytest.mark.parametrize(
    ("task_id", "question", "video_url"),
    [("", "Q?", "file:///v.mp4"), ("one", " ", "file:///v.mp4"), ("one", "Q?", "")],
)
def test_a_row_needs_an_id_a_question_and_a_video(task_id, question, video_url) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        prepare_single_turn(task_id, question, RUBRIC, video_url=video_url)
    with pytest.raises(ValidationError, match="add up to 100"):
        prepare_single_turn("one", "Q?", {"criteria": RUBRIC["criteria"][:1]}, video_url="file:///v.mp4")


def test_the_schema_declares_the_fields_the_verifier_reads() -> None:
    nested = {
        name for name, field in TaskData.model_fields.items() if (field.json_schema_extra or {}).get("legacy_location")
    }
    top_level = set(TaskData.model_fields) - nested
    assert nested == set(VideoQARubricMetadata.model_fields) == {"question", "rubric"}
    assert top_level == {"task_id", "setting"} <= set(VideoQARubricVerifyRequest.model_fields)
    for model in (TaskData, VideoQARubricMetadata):
        assert all(field.is_required() for field in model.model_fields.values())


def test_rows_the_verifier_refuses_are_invalid_task_data() -> None:
    good = prepare_single_turn("video_qa_rubric:one", "Q?", RUBRIC, video_url="file:///v.mp4")
    bad = copy.deepcopy(good)
    bad["verifier_metadata"]["rubric"]["criteria"][0]["weight_points"] = 10
    assert report([good]).clean and report([bad]).error_rows == 1
    VideoQARubricVerifyRequest.model_validate(good | {"response": response("Right.")})
    with pytest.raises(ValidationError):
        VideoQARubricVerifyRequest.model_validate(bad | {"response": response("Right.")})


def test_the_examples_are_valid_rows_on_the_synthetic_clips() -> None:
    result = report(EXAMPLES)
    assert len(EXAMPLES) == result.rows == 5 and result.clean, result.summary()
    clips = SERVER.parent / "video_mcqa" / "data" / "example_videos"
    for number, row in enumerate(EXAMPLES, start=1):
        assert row["task_id"] == f"video_qa_rubric:example-{number}"
        VideoQARubricVerifyRequest.model_validate(row | {"response": response("An answer.")})
        video, text = row["responses_create_params"]["input"][0]["content"]
        assert text["text"] == PROMPT.format(question=row["verifier_metadata"]["question"])
        prefix, encoded = video["video_url"].split(",", 1)
        assert prefix == "data:video/mp4;base64"
        clip = (clips / f"synthetic-{number}.mp4").read_bytes()
        assert hashlib.sha256(base64.b64decode(encoded)).digest() == hashlib.sha256(clip).digest()


def test_the_config_binds_one_model_call_and_the_judge_to_this_server() -> None:
    values = {"judge_base_url": "https://judge.invalid/v1", "judge_api_key": "invented", "judge_model_name": "judge"}
    loaded = OmegaConf.merge(OmegaConf.load(SERVER / "configs" / "video_qa_rubric.yaml"), OmegaConf.create(values))
    config = OmegaConf.to_container(loaded, resolve=True)
    resource = VideoQARubricResourcesServerConfig(
        name="video_qa_rubric",
        host="127.0.0.1",
        port=8001,
        **config["video_qa_rubric"]["resources_servers"]["video_qa_rubric"],
    )
    assert resource.judge_model_server.name == "video_qa_rubric_judge" and resource.empty_answer_reward is None
    assert config["video_qa_rubric_judge"]["responses_api_models"]["openai_model"]["openai_model"] == "judge"
    agent_config = config["video_qa_rubric_single_turn"]["responses_api_agents"]["simple_agent"]
    agent = SimpleAgentConfig(name="video_qa_rubric_single_turn", host="127.0.0.1", port=8002, **agent_config)
    assert (agent.resources_server.name, agent.model_server.name, agent.max_steps) == (
        "video_qa_rubric",
        "policy_model",
        1,
    )
    (example,) = [DatasetConfig.model_validate(item) for item in agent_config["datasets"]]
    assert (example.type, example.license) == ("example", "Apache 2.0")
    assert example.jsonl_fpath == "resources_servers/video_qa_rubric/data/example.jsonl"
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name="video_qa_rubric_single_turn_environment",
        host="127.0.0.1",
        port=8003,
        **config["video_qa_rubric_single_turn_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert (environment.agent_server.name, environment.resources_server.name) == (
        "video_qa_rubric_single_turn",
        "video_qa_rubric",
    )
