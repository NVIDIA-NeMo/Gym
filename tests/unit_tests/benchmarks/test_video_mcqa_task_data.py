# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core-suite tests for resources_servers/video_mcqa/task_data.py.

tests/unit_tests/data/test_task_data.py imports every server's task_data.py, so the core job measures this schema
against its coverage gate ([tool.coverage.report] fail_under). The servers' own tests, including
resources_servers/video_mcqa/tests/test_task_data.py, are not collected there (testpaths = ["tests"]), so this file
covers the schema's validators in the core suite, on synthetic values.
"""

import pytest
from pydantic import ValidationError

from resources_servers.video_mcqa.task_data import TaskData, VideoBinding, VideoMCQATaskData


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
}
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


@pytest.mark.parametrize("updates", [{"option_count": 5}, {"level": 2}, {"group_id": "group-1"}])
def test_only_videomme_v2_rows_take_other_option_counts_levels_and_groups(updates):
    assert VideoMCQATaskData.model_validate({**ROW, **updates})
    with pytest.raises(ValidationError, match="four choices"):
        TaskData.model_validate({**ROW, **updates})
