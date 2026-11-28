# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core-suite tests for resources_servers/video_mcqa/task_data.py and the VideoMME-v1 and LVBench schemas on it.

tests/unit_tests/data/test_task_data.py imports every server's task_data.py, so the core job measures these schemas
against its coverage gate ([tool.coverage.report] fail_under). The servers' own tests, including
resources_servers/video_mcqa/tests/test_task_data.py, are not collected there (testpaths = ["tests"]), so this file
covers the schemas' validators in the core suite, on synthetic values.
"""

import pytest
from pydantic import ValidationError

from resources_servers.lvbench import task_data as lvbench
from resources_servers.video_mcqa.task_data import TaskData, VideoBinding, VideoMCQATaskData
from resources_servers.videomme_v1 import task_data as videomme_v1


ROW = {
    "task_id": "videomme_v1:invented",
    "setting": "single_turn",
    "expected_answer": "B",
    "option_count": 4,
    "dataset_source": "https://example.invalid/source",
    "dataset_revision": "a" * 40,
    "video_id": "invented",
    "video_source": "https://example.invalid/video.mp4",
    "video_sha256": "b" * 64,
    "question": "Which invented shape moves?",
    "options": ["Circle", "Square", "Triangle", "Star"],
}
FIVE_CHOICES = {"option_count": 5, "options": ["Circle", "Square", "Triangle", "Star", "Moon"]}
BINDING = {"source": "/data/invented.mp4", "sandbox_path": "video/invented.mp4", "sha256": "b" * 64}


@pytest.mark.parametrize("schema", [TaskData, VideoMCQATaskData])
def test_single_turn_rows_need_no_binding_and_agentic_rows_keep_theirs(schema):
    assert schema.model_validate(ROW).video_binding is None
    agentic = schema.model_validate({**ROW, "setting": "agentic", "video_binding": BINDING})
    assert agentic.video_binding == VideoBinding(**BINDING)


@pytest.mark.parametrize(
    "updates",
    [
        {"task_id": " "},
        {"video_id": "invented\x00"},
        {"expected_answer": "E"},
        {"question": " "},
        {"options": ["Circle", "Square", "Triangle"]},
        {"options": ["Circle", "Square", "Triangle", ""]},
        {"setting": "agentic"},
        {"setting": "agentic", "video_binding": {**BINDING, "sandbox_path": "/data/invented.mp4"}},
        {"setting": "agentic", "video_binding": {**BINDING, "sandbox_path": "video/../../invented.mp4"}},
        {"setting": "agentic", "video_binding": {**BINDING, "sandbox_path": "video/\ninvented.mp4"}},
        {"setting": "agentic", "video_binding": {**BINDING, "url": "https://example.invalid/video.mp4"}},
    ],
    ids=[
        "blank_text",
        "nul_byte",
        "unavailable_answer",
        "blank_question",
        "options_unlike_option_count",
        "blank_option",
        "agentic_without_binding",
        "absolute_binding_path",
        "escaping_binding_path",
        "multiline_binding_path",
        "unknown_binding_key",
    ],
)
def test_rows_the_runtime_refuses_are_rejected(updates):
    with pytest.raises(ValidationError):
        VideoMCQATaskData.model_validate({**ROW, **updates})


@pytest.mark.parametrize("missing", ["question", "options"])
def test_rows_without_their_question_and_options_are_rejected(missing):
    with pytest.raises(ValidationError, match=missing):
        VideoMCQATaskData.model_validate({key: value for key, value in ROW.items() if key != missing})


@pytest.mark.parametrize("updates", [{"level": 2}, {"group_id": "group-1"}])
def test_only_videomme_v2_rows_take_levels_and_groups(updates):
    assert VideoMCQATaskData.model_validate({**ROW, **updates})
    for schema in (TaskData, videomme_v1.TaskData, lvbench.TaskData):
        with pytest.raises(ValidationError, match="no v2 group or level"):
            schema.model_validate({**ROW, **updates})


def test_any_number_of_choices_except_in_videomme_v1_and_lvbench():
    assert TaskData.model_validate({**ROW, **FIVE_CHOICES}).option_count == 5
    for schema in (videomme_v1.TaskData, lvbench.TaskData):
        assert schema.model_validate(ROW).option_count == 4
        with pytest.raises(ValidationError, match="four choices"):
            schema.model_validate({**ROW, **FIVE_CHOICES})
