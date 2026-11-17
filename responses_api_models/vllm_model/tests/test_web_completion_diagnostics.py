# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from copy import deepcopy

from responses_api_models.vllm_model.app import _log_web_completion


def test_web_completion_exposes_stop_reason_without_logging_or_rewriting_text(caplog) -> None:
    raw = {
        "id": "response-1",
        "choices": [
            {
                "finish_reason": "stop",
                "stop_reason": 2,
                "message": {
                    "reasoning": "private reasoning with a <function=computer> marker",
                    "content": None,
                    "tool_calls": [],
                },
            }
        ],
        "usage": {"prompt_tokens": 10200, "completion_tokens": 27, "total_tokens": 10227},
    }
    before = deepcopy(raw)
    with caplog.at_level(logging.INFO, logger="nemo_gym.vllm_model"):
        _log_web_completion(raw, {"adapter": "web_agent", "task_id": "Huggingface--38", "step": 8, "parse_attempt": 0})
    assert "stop_reason=2" in caplog.text
    assert "tool_calls=0" in caplog.text
    assert "reasoning_tool_markup=True" in caplog.text
    assert "private reasoning" not in caplog.text
    assert raw == before


def test_completion_diagnostics_do_not_add_logging_to_other_agents(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nemo_gym.vllm_model"):
        _log_web_completion({"choices": []}, {})
    assert "web_upstream_completion" not in caplog.text
