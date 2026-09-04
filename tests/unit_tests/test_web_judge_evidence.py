# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.web.judge_evidence import (
    compact_webvoyager_judge_evidence,
    expand_webvoyager_judge_screenshots,
    response_image_urls,
)


def _response(*image_urls: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "rollout-response",
            "created_at": 0.0,
            "model": "policy",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": image_url, "detail": "high"}],
                }
                for image_url in image_urls
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def test_compact_evidence_references_trajectory_images_and_inlines_only_missing_edges():
    response = _response("data:image/png;base64,middle-1", "data:image/png;base64,middle-2")
    screenshots = [
        "data:image/png;base64,initial",
        "data:image/png;base64,middle-1",
        "data:image/png;base64,middle-2",
        "data:image/png;base64,terminal",
    ]

    evidence = compact_webvoyager_judge_evidence(
        response=response,
        final_answer="42",
        screenshots=screenshots,
        page_urls=["https://example.test/initial", "https://example.test/result"],
    )

    assert response_image_urls(response) == screenshots[1:3]
    assert evidence == {
        "schema_version": 1,
        "final_answer": "42",
        "page_urls": ["https://example.test/initial", "https://example.test/result"],
        "screenshot_sequence": [
            {"inline_data_url": screenshots[0]},
            {"response_image_index": 0},
            {"response_image_index": 1},
            {"inline_data_url": screenshots[3]},
        ],
    }
    assert expand_webvoyager_judge_screenshots(evidence, response) == screenshots


def test_expand_evidence_accepts_rows_from_the_pre_compaction_schema():
    evidence = {
        "final_answer": "legacy",
        "screenshots": ["data:image/png;base64,legacy"],
        "page_urls": [],
    }

    assert expand_webvoyager_judge_screenshots(evidence, _response()) == evidence["screenshots"]
