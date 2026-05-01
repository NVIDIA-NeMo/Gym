# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest


DEFAULT_ROLLOUT_JSONL = "output/hle-token-test/hle-token-test-rollouts.jsonl"
GENERATED_ITEM_TYPES = {"message", "function_call"}
TOKEN_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
TOOL_ERROR_PATTERNS = (
    r"\b401\b",
    r"\b403\b",
    r"\bunauthorized\b",
    r"\bforbidden\b",
    r"\bpermission denied\b",
    r"\bserper\b.*\berror\b",
    r"\btraceback\b",
    r"\bexception\b",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _rollout_jsonl_path() -> Path:
    raw_path = os.environ.get("VERIFIERS_ROLLOUT_JSONL", DEFAULT_ROLLOUT_JSONL)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


@pytest.fixture(scope="module")
def rollout_rows() -> list[dict[str, Any]]:
    path = _rollout_jsonl_path()
    if not path.exists():
        pytest.skip(
            f"Rollout artifact not found at {path}. Run ng_collect_rollouts first "
            "or set VERIFIERS_ROLLOUT_JSONL=/path/to/rollouts.jsonl."
        )

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_no} is not valid JSON") from exc

    assert rows, f"{path} exists but contains no JSONL rows"
    return rows


def _response(row: dict[str, Any], row_idx: int) -> dict[str, Any]:
    response = row.get("response")
    assert isinstance(response, dict), f"row {row_idx}: missing object field `response`"
    return response


def _metrics(response: dict[str, Any], row_idx: int) -> dict[str, Any]:
    metrics = response.get("metrics")
    assert isinstance(metrics, dict), f"row {row_idx}: missing object field `response.metrics`"
    return metrics


def _output(response: dict[str, Any], row_idx: int) -> list[dict[str, Any]]:
    output = response.get("output")
    assert isinstance(output, list), f"row {row_idx}: missing list field `response.output`"
    assert output, f"row {row_idx}: `response.output` is empty"
    return output


def _generated_items(output: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (idx, item)
        for idx, item in enumerate(output)
        if isinstance(item, dict) and item.get("type") in GENERATED_ITEM_TYPES
    ]


def _has_all_token_fields(item: dict[str, Any]) -> bool:
    return all(key in item for key in TOKEN_KEYS)


def _assert_valid_training_tokens(item: dict[str, Any], context: str) -> None:
    missing = [key for key in TOKEN_KEYS if key not in item]
    assert not missing, f"{context}: missing token fields {missing}"

    prompt_token_ids = item["prompt_token_ids"]
    generation_token_ids = item["generation_token_ids"]
    generation_log_probs = item["generation_log_probs"]

    assert isinstance(prompt_token_ids, list), f"{context}: prompt_token_ids is not a list"
    assert isinstance(generation_token_ids, list), f"{context}: generation_token_ids is not a list"
    assert isinstance(generation_log_probs, list), f"{context}: generation_log_probs is not a list"
    assert prompt_token_ids, f"{context}: prompt_token_ids is empty"
    assert generation_token_ids, f"{context}: generation_token_ids is empty"
    assert generation_log_probs, f"{context}: generation_log_probs is empty"
    assert len(generation_token_ids) == len(generation_log_probs), (
        f"{context}: generation token/logprob length mismatch "
        f"({len(generation_token_ids)} != {len(generation_log_probs)})"
    )


def test_rollout_has_no_verifier_errors_and_has_judge_metrics(rollout_rows: list[dict[str, Any]]) -> None:
    for row_idx, row in enumerate(rollout_rows):
        response = _response(row, row_idx)
        assert response.get("error") in (None, "", []), f"row {row_idx}: response.error is set"

        reward = response.get("reward", row.get("reward"))
        assert isinstance(reward, int | float), f"row {row_idx}: missing numeric reward"

        metrics = _metrics(response, row_idx)
        assert isinstance(metrics.get("judge_score"), int | float), (
            f"row {row_idx}: missing numeric metrics.judge_score; judge may not have run"
        )
        assert isinstance(metrics.get("num_turns"), int | float), (
            f"row {row_idx}: missing numeric metrics.num_turns"
        )


def test_generated_items_have_training_token_ids(rollout_rows: list[dict[str, Any]]) -> None:
    require_every_generated_item = os.environ.get("REQUIRE_EVERY_GENERATED_ITEM_TOKENS") == "1"

    for row_idx, row in enumerate(rollout_rows):
        output = _output(_response(row, row_idx), row_idx)
        generated = _generated_items(output)
        assert generated, f"row {row_idx}: no generated message/function_call items found"

        tokenized_items = []
        missing_token_items = []
        for item_idx, item in generated:
            context = f"row {row_idx} output[{item_idx}] type={item.get('type')}"
            if _has_all_token_fields(item):
                _assert_valid_training_tokens(item, context)
                tokenized_items.append(item_idx)
            else:
                missing_token_items.append(item_idx)

        assert tokenized_items, (
            f"row {row_idx}: no generated items contain training token IDs. "
            "This usually means the run did not use vllm_model_for_training.yaml, "
            "or the policy endpoint did not return token/logprob metadata."
        )

        final_idx, final_item = generated[-1]
        _assert_valid_training_tokens(
            final_item,
            f"row {row_idx} final generated output[{final_idx}] type={final_item.get('type')}",
        )

        if require_every_generated_item:
            assert not missing_token_items, (
                f"row {row_idx}: generated items missing token fields at output indices "
                f"{missing_token_items}"
            )


def test_hle_tool_and_judge_sanity(rollout_rows: list[dict[str, Any]]) -> None:
    tool_call_mode = os.environ.get("EXPECT_TOOL_CALLS", "auto").strip().lower()
    valid_tool_call_modes = {"auto", "1", "true", "yes", "required", "0", "false", "no"}
    assert tool_call_mode in valid_tool_call_modes, (
        "EXPECT_TOOL_CALLS must be one of "
        f"{sorted(valid_tool_call_modes)}, got {tool_call_mode!r}"
    )
    require_tool_calls = tool_call_mode in {"1", "true", "yes", "required"}
    allow_final_tool_call = os.environ.get("ALLOW_FINAL_TOOL_CALL", "0").lower() in {"1", "true", "yes"}

    for row_idx, row in enumerate(rollout_rows):
        response = _response(row, row_idx)
        metrics = _metrics(response, row_idx)
        output = _output(response, row_idx)

        assert isinstance(metrics.get("judge_score"), int | float), (
            f"row {row_idx}: judge_score is missing; judge model may not have been invoked"
        )

        tool_calls = [item for item in output if isinstance(item, dict) and item.get("type") == "function_call"]
        tool_outputs = [
            item for item in output if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        total_tool_calls = metrics.get("total_tool_calls", 0)

        if require_tool_calls:
            assert tool_calls, f"row {row_idx}: expected at least one tool call"
            assert tool_outputs, f"row {row_idx}: expected at least one tool output"
            assert total_tool_calls > 0, f"row {row_idx}: expected metrics.total_tool_calls > 0"
        elif not (tool_calls or tool_outputs or total_tool_calls):
            pytest.skip(
                "No tool calls found in rollout artifact; skipping tool-call/output consistency checks. "
                "Set EXPECT_TOOL_CALLS=1 to require tool use."
            )
        elif tool_calls or tool_outputs or total_tool_calls:
            assert tool_calls, (
                f"row {row_idx}: metrics/tool outputs indicate tool use, but no function_call items exist"
            )
            assert tool_outputs, (
                f"row {row_idx}: function_call items exist, but no function_call_output items exist"
            )
            assert total_tool_calls > 0, (
                f"row {row_idx}: function_call items exist, but metrics.total_tool_calls is not > 0"
            )

        tool_output_text = "\n".join(str(item.get("output", "")) for item in tool_outputs)
        matching_error_patterns = [
            pattern
            for pattern in TOOL_ERROR_PATTERNS
            if re.search(pattern, tool_output_text, flags=re.IGNORECASE)
        ]
        assert not matching_error_patterns, (
            f"row {row_idx}: tool output looks like an API/runtime error: {matching_error_patterns}"
        )

        generated = _generated_items(output)
        assert generated, f"row {row_idx}: no generated message/function_call items found"
        final_idx, final_item = generated[-1]
        if not allow_final_tool_call:
            assert final_item.get("type") == "message", (
                f"row {row_idx}: final generated item output[{final_idx}] is "
                f"{final_item.get('type')!r}, not a final answer message. "
                "This often means max_turns was hit before the model finalized."
            )
