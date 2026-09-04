# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import json
import pickle
import warnings
from asyncio import Future
from collections import Counter
from copy import deepcopy
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import yaml
from aiohttp import ClientConnectorError, ClientResponseError, ServerDisconnectedError
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

import nemo_gym.rollout_collection
import nemo_gym.token_id_capture.delivery
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import (
    _DEFAULT_MAX_ROLLOUT_ATTEMPTS,
    AGENT_REQUEST_FAILED_FAILURE_CLASS,
    AGENT_RUN_ERROR_FAILURE_CLASS,
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_PERF_KEY,
    NG_TERMINAL_KEY,
    NG_TRAJECTORY_KEY,
    E2ERolloutCollectionConfig,
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _attach_ng_perf,
    _attach_trajectory_record,
    _build_ng_perf,
    _build_trajectory_record,
    _CompletedRollout,
    _expand_input_glob,
    _failure_rows_counted_as_zero,
    _failures_path_for,
    _get_max_rollout_attempts,
    _rollout_for_export,
    _rollout_request_debug_summary,
    loads_jsonl_line,
)
from nemo_gym.token_id_capture import (
    LineageResolution,
    ParentResolutionStatus,
    TokenCaptureSnapshot,
    TokenCaptureStore,
    TokenEntry,
    clear_token_captures_for_rollouts,
    stamp_lineage,
)
from nemo_gym.token_id_capture.delivery import (
    MASK_SAMPLE_KEY,
    TOKEN_CAPTURE_KEY,
    capture_build_can_retire,
    finalize_rollout_token_capture,
    retire_rollout_token_capture,
    rollout_carries_token_ids,
)


class _StubLineageStore:
    """Satisfy the normal custom-sink contract in collector-only tests."""

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        return LineageResolution(ParentResolutionStatus.ROOT)

    def is_process_shared(self) -> bool:
        return True

    async def close(self) -> None:
        pass


@pytest.fixture
def empty_global_config(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    get_global_config_dict = MagicMock(return_value={})
    monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", get_global_config_dict)
    return get_global_config_dict


class FakeResponse:
    """The parts of aiohttp's ClientResponse that the rollout dispatcher touches."""

    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status = status
        self.ok = 200 <= status < 300
        self.payload = payload
        self.released = False

    def release(self) -> None:
        self.released = True


def http_error(status: int, message: str = "boom", body: bytes | None = None) -> ClientResponseError:
    request_info = SimpleNamespace(method="POST", url="http://agent/run", real_url="http://agent/run")
    error = ClientResponseError(request_info=request_info, history=(), status=status, message=message)
    if body is not None:
        error.response_content = body
    return error


def install_fake_server_client(monkeypatch: pytest.MonkeyPatch, post: AsyncMock) -> MagicMock:
    """Route every dispatcher HTTP call through `post` and unwrap FakeResponse."""
    server_client = MagicMock()
    server_client.post = post
    server_client.global_config_dict = OmegaConf.create({"my_agent": {"responses_api_agents": {"impl": {}}}})
    monkeypatch.setattr(
        nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: server_client
    )

    async def raise_for_status(response: FakeResponse) -> None:
        if not response.ok:
            raise http_error(response.status)

    async def get_response_json(response: FakeResponse) -> dict | None:
        return response.payload

    monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", raise_for_status)
    monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", get_response_json)
    return server_client


def failing_row(task_index: int = 7) -> dict:
    return {
        AGENT_REF_KEY_NAME: {"name": "my_agent"},
        TASK_INDEX_KEY_NAME: task_index,
        ROLLOUT_INDEX_KEY_NAME: 0,
        "responses_create_params": {"input": []},
    }


class TestLoadsJsonlLine:
    def test_parses_valid_line(self) -> None:
        assert loads_jsonl_line('{"a": 1}', "f.jsonl", 1) == {"a": 1}

    def test_malformed_line_raises_config_error_with_location(self) -> None:
        with pytest.raises(ConfigError, match=r"Malformed JSON in 'f.jsonl' at line 3"):
            loads_jsonl_line("{not json", "f.jsonl", 3)


class TestUploadRolloutsDeprecation:
    BASE = {"input_jsonl_fpath": "in.jsonl", "output_jsonl_fpath": "out.jsonl"}

    def test_defaults_to_true(self) -> None:
        assert RolloutCollectionConfig.model_validate(self.BASE).upload_rollouts

    def test_deprecated_key_maps_and_warns(self) -> None:
        with pytest.warns(DeprecationWarning, match="upload_rollouts_to_wandb"):
            config = RolloutCollectionConfig.model_validate({**self.BASE, "upload_rollouts_to_wandb": False})

        assert not config.upload_rollouts

    def test_new_key_wins_over_the_deprecated_one(self) -> None:
        with pytest.warns(DeprecationWarning):
            config = RolloutCollectionConfig.model_validate(
                {**self.BASE, "upload_rollouts_to_wandb": False, "upload_rollouts": True}
            )

        assert config.upload_rollouts

    def test_new_key_alone_does_not_warn(self, recwarn) -> None:
        config = RolloutCollectionConfig.model_validate({**self.BASE, "upload_rollouts": False})

        assert not config.upload_rollouts
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


class TestGetMaxRolloutAttempts:
    def test_default_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", raising=False)
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS

    def test_default_when_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "")
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS

    def test_valid_value(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "5")
        assert _get_max_rollout_attempts() == 5

    def test_non_integer_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "not-an-int")
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS

    def test_non_positive_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "0")
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS


class TestRolloutCollection:
    def test_rollout_request_debug_summary_compact(self) -> None:
        row = {
            AGENT_REF_KEY_NAME: {"name": "my_agent"},
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
            "env_specific_metadata": "do not include",
            "responses_create_params": {"input": "large prompt", "tools": ["large schema"]},
        }

        assert _rollout_request_debug_summary(row) == {
            "agent_name": "my_agent",
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
        }

    def test_build_trajectory_record_merges_all_evidence_sources(self) -> None:
        row = {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 3}
        result = {
            "ng_trajectory": {
                "task_id": "2",
                "rollout_id": "2-3",
                "invocations": [
                    {
                        "invocation_id": "root",
                        "status": "completed",
                        "conversation": [
                            {"type": "function_call_output", "call_id": "tool-1", "output": "result"},
                            {"type": "function_call_output", "call_id": "observed-only", "output": "new"},
                        ],
                    }
                ],
                "tool_calls": [
                    {"invocation_id": "root", "tool_call_id": "producer-only", "output": "kept"},
                    {
                        "invocation_id": "root",
                        "tool_call_id": "tool-1",
                        "output": "stale",
                        "status": "failed",
                        "started_at": 10.2,
                        "completed_at": 10.4,
                        "duration_ms": 200.0,
                    },
                ],
            },
            "ng_agent_observations": {
                "source": "test",
                "records": [
                    {
                        "kind": "agent_invocation",
                        "invocation_id": "root",
                    },
                    {
                        "kind": "agent_invocation",
                        "invocation_id": "observed",
                    },
                    {
                        "kind": "tool_call",
                        "invocation_id": "root",
                        "tool_call_id": "tool-1",
                    },
                    {
                        "kind": "tool_call",
                        "invocation_id": "root",
                        "tool_call_id": "observed-only",
                        "status": "completed",
                        "started_at": 10.2,
                        "completed_at": 10.4,
                        "duration_ms": 200.0,
                    },
                ],
            },
            "ng_model_call_capture": {"calls": [{"model_call_id": "capture-only"}]},
        }

        trajectory = _build_trajectory_record(row, result)
        producer_only, merged, observed_only = trajectory.tool_calls

        assert [invocation.invocation_id for invocation in trajectory.invocations] == ["root", "observed"]
        assert trajectory.invocations[0].status == "completed"
        assert len(trajectory.invocations[0].conversation) == 2
        assert [call.model_call_id for call in trajectory.model_calls] == ["capture-only"]
        assert producer_only.tool_call_id == "producer-only" and producer_only.output == "kept"
        assert (merged.output, merged.status, merged.started_at, merged.completed_at, merged.duration_ms) == (
            "result",
            "failed",
            10.2,
            10.4,
            200.0,
        )
        assert observed_only.tool_call_id == "observed-only" and observed_only.output == "new"

    def test_build_trajectory_record_normalizes_identity_and_merges_model_calls(self) -> None:
        row = {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 3, "task_id": "collector-task"}
        result = {
            "ng_trajectory": {
                "task_id": "producer-task",
                "rollout_id": "producer-rollout",
                "turns": [
                    {
                        "invocation_id": "root",
                        "task_id": "producer-task",
                        "rollout_id": "producer-rollout",
                        "turn_no": 1,
                        "timestamp": 1.0,
                        "step_count": 0,
                    }
                ],
                "model_calls": [
                    {"model_call_id": "producer-only", "request": "kept"},
                    {
                        "model_call_id": "shared",
                        "request": "stale",
                        "response_metadata": {"model": "producer-model"},
                    },
                ],
            },
            "ng_model_call_capture": {
                "calls": [
                    {
                        "model_call_id": "shared",
                        "request": "captured",
                        "response": {"status": "incomplete"},
                        "response_status": "completed",
                    },
                    {"model_call_id": "capture-only", "request": "new"},
                ]
            },
        }

        trajectory = _build_trajectory_record(row, result)

        assert (trajectory.task_id, trajectory.rollout_id) == ("collector-task", "2-3")
        assert (trajectory.turns[0].task_id, trajectory.turns[0].rollout_id) == ("collector-task", "2-3")
        assert {gap.code for gap in trajectory.gaps} >= {"producer_trajectory_identity_mismatch"}
        assert [call.model_call_id for call in trajectory.model_calls] == ["producer-only", "shared", "capture-only"]
        assert [call.request for call in trajectory.model_calls] == ["kept", "captured", "new"]
        assert trajectory.model_calls[1].response_metadata.model_dump(exclude_none=True) == {
            "model": "producer-model",
            "response_status": "completed",
        }

    def test_trajectory_projection_failure_preserves_rollout(self) -> None:
        row = {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 3}
        result = {
            "ng_model_call_capture": {
                "calls": [
                    {
                        "model_call_id": "model-1",
                        "latency_total_ms": -1,
                        "request": {"input": "question"},
                        "response": {"output": "answer"},
                    }
                ]
            }
        }

        _attach_trajectory_record(row, result)

        assert "ng_trajectory" not in result
        assert result["ng_model_call_capture"]["gaps"][-1]["code"] == "trajectory_projection_failed"
        assert result["ng_model_call_capture"]["calls"][0]["request"] == {"input": "question"}
        assert result["ng_model_call_capture"]["calls"][0]["response"] == {"output": "answer"}

    def test_trajectory_projection_failure_without_attachment_keeps_gap(self, monkeypatch) -> None:
        row = {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 3}
        result = {"ng_trajectory": {}}
        monkeypatch.setattr(nemo_gym.rollout_collection, "_build_trajectory_record", MagicMock(side_effect=ValueError))

        _attach_trajectory_record(row, result)

        assert result["ng_trajectory"]["gaps"] == [
            {"code": "trajectory_projection_failed", "invocation_id": None, "detail": "ValueError"}
        ]

    def test_rollout_for_export_omits_new_trajectory_and_raw_capture_payloads(self) -> None:
        result = {
            "response": {"output": "existing rollout content"},
            "ng_trajectory": {"invocations": [{"conversation": ["trajectory secret"]}]},
            "ng_model_call_capture": {
                "calls": [
                    {
                        "model_call_id": "model-1",
                        "request": {"input": "request secret"},
                        "response": {"output": "response secret"},
                        "request_raw": "raw request secret",
                        "response_raw": "raw response secret",
                    }
                ],
            },
        }

        sanitized = _rollout_for_export(result)

        assert "ng_trajectory" not in sanitized
        assert sanitized["ng_model_call_capture"]["calls"] == [{"model_call_id": "model-1"}]
        assert sanitized["response"] == result["response"]
        assert result["ng_model_call_capture"]["calls"][0]["request"] == {"input": "request secret"}
        assert "ng_trajectory" in result
        malformed = (
            {"ng_model_call_capture": "secret"},
            {"ng_model_call_capture": {"calls": {"request": "secret"}}},
            {"ng_model_call_capture": {"calls": ["secret", {"request": "secret"}]}},
        )
        for malformed_result in malformed:
            sanitized = _rollout_for_export(malformed_result)
            assert b"secret" not in orjson.dumps(sanitized)

    def test_build_ng_perf_absent_without_trajectory(self) -> None:
        assert _build_ng_perf({}, rollout_latency_ms=12.0) is None
        assert _build_ng_perf({NG_TRAJECTORY_KEY: "not-a-dict"}, rollout_latency_ms=12.0) is None

    def test_build_ng_perf_absent_when_trajectory_invalid(self) -> None:
        result = {NG_TRAJECTORY_KEY: {"invocations": [{"invocation_id": "root"}, {"invocation_id": "root"}]}}
        assert _build_ng_perf(result, rollout_latency_ms=12.0) is None

    def test_build_ng_perf_absent_without_invocations(self) -> None:
        result = {NG_TRAJECTORY_KEY: {"task_id": "t", "rollout_id": "t-0", "invocations": []}}
        assert _build_ng_perf(result, rollout_latency_ms=12.0) is None

    def test_build_ng_perf_sums_owned_calls_and_matched_tool_calls(self) -> None:
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "t",
                "rollout_id": "t-0",
                "invocations": [
                    {
                        "invocation_id": "root",
                        "model_calls": [{"model_call_id": "call-1"}],
                    },
                    {
                        "invocation_id": "sub",
                        "model_calls": [{"model_call_id": "call-2"}, {"model_call_id": "unresolved-response"}],
                    },
                ],
                "tool_calls": [
                    {"invocation_id": "root", "tool_call_id": "t1"},
                    {"invocation_id": "sub", "tool_call_id": "t2"},
                    {"invocation_id": "sub", "tool_call_id": "t3"},
                    # Orphaned: no invocation in this trajectory owns "ghost".
                    {"invocation_id": "ghost", "tool_call_id": "t4"},
                ],
                "model_calls": [
                    {
                        "model_call_id": "call-1",
                        "token_stats": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "cached_tokens": 10,
                        },
                    },
                    {
                        "model_call_id": "call-2",
                        "token_stats": {
                            "prompt_tokens": 50,
                            "completion_tokens": 5,
                            "reasoning_tokens": 3,
                        },
                    },
                    # Present in raw capture but never referenced by any invocation's
                    # model_calls (e.g. compaction-owned, or unjoined) -- must not be summed.
                    {
                        "model_call_id": "unowned",
                        "token_stats": {"prompt_tokens": 999, "completion_tokens": 999},
                    },
                ],
            }
        }

        ng_perf = _build_ng_perf(result, rollout_latency_ms=1234.5)

        assert ng_perf == {
            "num_turns": 2,
            "num_tool_calls": 3,
            "token_observability_coverage": 1.0,
            "prompt_tokens": 150,
            "cached_prompt_tokens": 10,
            "completion_tokens": 25,
            "reasoning_tokens": 3,
            "total_latency_ms": 1234.5,
        }

    def test_build_ng_perf_omits_absent_token_fields_and_latency(self) -> None:
        # No model_calls/tool_calls at all on the trajectory, and no independent latency
        # measurement, so every optional ng_perf field must be *absent*, not present-as-None.
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "t",
                "rollout_id": "t-0",
                "invocations": [{"invocation_id": "root"}],
            }
        }

        ng_perf = _build_ng_perf(result, rollout_latency_ms=None)

        for absent_key in (
            "prompt_tokens",
            "cached_prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_latency_ms",
        ):
            assert absent_key not in ng_perf
        assert ng_perf == {"num_turns": 1, "num_tool_calls": 0, "token_observability_coverage": 0.0}

    def test_build_ng_perf_dedupes_model_call_claimed_by_two_invocations(self) -> None:
        # Simulates a join_model_call_observations conflict: the losing invocation keeps an
        # unresolved ref pointing at a model_call_id another invocation already claimed. Tokens
        # for that call must be counted once, not twice.
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "t",
                "rollout_id": "t-0",
                "invocations": [
                    {"invocation_id": "root", "model_calls": [{"model_call_id": "shared"}]},
                    {"invocation_id": "sub", "model_calls": [{"model_call_id": "shared"}]},
                ],
                "model_calls": [
                    {"model_call_id": "shared", "token_stats": {"prompt_tokens": 100, "completion_tokens": 10}},
                ],
            }
        }

        ng_perf = _build_ng_perf(result, rollout_latency_ms=None)

        # Token dedup must not erase the losing invocation from the turn count: the shared
        # call contributes one owned-call turn to "root", and "sub" -- a real conversation
        # left with no owned calls -- still counts as at least one turn.
        assert ng_perf["num_turns"] == 2
        assert ng_perf["prompt_tokens"] == 100
        assert ng_perf["completion_tokens"] == 10
        assert ng_perf["token_observability_coverage"] == 0.5

    def test_ng_perf_matches_model_calls_by_response_id_pair_like_simple_agent(self) -> None:
        # Integration test: simple_agent sets result["ng_trajectory"] directly (see
        # responses_api_agents/simple_agent/app.py), bypassing join_model_call_observations
        # entirely -- so its ModelCallRef is never canonicalized with a model_call_id and only
        # ever carries (model_ref, response_id). Token fields must still populate.
        row = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "0",
                "rollout_id": "0-0",
                "invocations": [
                    {
                        "invocation_id": "root",
                        "model_calls": [
                            {
                                "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                                "response_id": "resp_123",
                            }
                        ],
                    }
                ],
            },
            "ng_model_call_capture": {
                "calls": [
                    {
                        "model_call_id": "captured-1",
                        "response_id": "resp_123",
                        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                        "tokens_in": 100,
                        "tokens_out": 20,
                    }
                ]
            },
        }

        _attach_trajectory_record(row, result)
        ng_perf = _build_ng_perf(result, rollout_latency_ms=None)

        assert ng_perf["prompt_tokens"] == 100
        assert ng_perf["completion_tokens"] == 20

    def test_ng_perf_does_not_guess_an_ambiguous_response_id_match(self) -> None:
        # Two captured calls share the same (model_ref, response_id) pair -- the ref must
        # resolve to no match rather than guessing either one, so its tokens stay uncounted.
        row = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "0",
                "rollout_id": "0-0",
                "invocations": [
                    {
                        "invocation_id": "root",
                        "model_calls": [
                            {
                                "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                                "response_id": "resp_dup",
                            }
                        ],
                    }
                ],
            },
            "ng_model_call_capture": {
                "calls": [
                    {
                        "model_call_id": "captured-1",
                        "response_id": "resp_dup",
                        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                        "tokens_in": 100,
                        "tokens_out": 20,
                    },
                    {
                        "model_call_id": "captured-2",
                        "response_id": "resp_dup",
                        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                        "tokens_in": 999,
                        "tokens_out": 999,
                    },
                ]
            },
        }

        _attach_trajectory_record(row, result)
        ng_perf = _build_ng_perf(result, rollout_latency_ms=None)

        assert "prompt_tokens" not in ng_perf
        assert "completion_tokens" not in ng_perf

    def test_build_ng_perf_counts_explicit_turns_over_invocations(self) -> None:
        # simple_agent-style trajectory: the whole multi-turn loop runs under a single "root"
        # invocation with one TrajectoryTurn per step. num_turns must count the turns (3), not
        # the invocations (1) -- otherwise tokens-per-turn degenerates into total tokens.
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "t",
                "rollout_id": "t-0",
                "invocations": [{"invocation_id": "root", "model_calls": [{"model_call_id": "call-1"}]}],
                "turns": [
                    {
                        "invocation_id": "root",
                        "task_id": "t",
                        "rollout_id": "t-0",
                        "turn_no": turn_no,
                        "timestamp": 1.0,
                        "step_count": 0,
                    }
                    for turn_no in (1, 2, 3)
                ],
                "model_calls": [
                    {"model_call_id": "call-1", "token_stats": {"prompt_tokens": 100, "completion_tokens": 30}},
                ],
            }
        }

        ng_perf = _build_ng_perf(result, rollout_latency_ms=None)

        assert ng_perf["num_turns"] == 3
        assert ng_perf["completion_tokens"] == 30
        assert ng_perf["token_observability_coverage"] == pytest.approx(1 / 3)

    def test_build_ng_perf_sums_turns_across_invocations_with_per_invocation_fallback(self) -> None:
        # Hybrid trajectory: "root" emits explicit TrajectoryTurn records (2 turns), "sub-a"
        # emits none but owns 4 resolved model calls, "sub-b" emits nothing at all. Each
        # invocation contributes its own best turn count: 2 + 4 + 1 = 7.
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "t",
                "rollout_id": "t-0",
                "invocations": [
                    {"invocation_id": "root"},
                    {"invocation_id": "sub-a", "model_calls": [{"model_call_id": f"call-{i}"} for i in range(4)]},
                    {"invocation_id": "sub-b"},
                ],
                "turns": [
                    {
                        "invocation_id": "root",
                        "task_id": "t",
                        "rollout_id": "t-0",
                        "turn_no": turn_no,
                        "timestamp": 1.0,
                        "step_count": 0,
                    }
                    for turn_no in (1, 2)
                ],
                "model_calls": [
                    {"model_call_id": f"call-{i}", "token_stats": {"completion_tokens": 10}} for i in range(4)
                ],
            }
        }

        ng_perf = _build_ng_perf(result, rollout_latency_ms=None)

        assert ng_perf["num_turns"] == 7
        assert ng_perf["completion_tokens"] == 40
        assert ng_perf["token_observability_coverage"] == pytest.approx(4 / 7)

    def test_attach_ng_perf_absent_when_observability_disabled(self) -> None:
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "t",
                "rollout_id": "t-0",
                "invocations": [{"invocation_id": "root"}],
            },
        }

        _attach_ng_perf(result, observability_enabled=False, rollout_latency_ms=42.0)

        assert NG_PERF_KEY not in result

    def test_attach_ng_perf_sets_ng_perf_when_enabled(self) -> None:
        result = {
            NG_TRAJECTORY_KEY: {
                "task_id": "t",
                "rollout_id": "t-0",
                "invocations": [{"invocation_id": "root"}],
            },
        }

        _attach_ng_perf(result, observability_enabled=True, rollout_latency_ms=42.0)

        assert result[NG_PERF_KEY] == {
            "num_turns": 1,
            "num_tool_calls": 0,
            "token_observability_coverage": 0.0,
            "total_latency_ms": 42.0,
        }

    def test_attach_ng_perf_swallows_build_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An unexpected assembly failure must never take down rollout collection over an observability side-channel.
        result = {NG_TRAJECTORY_KEY: {}}
        monkeypatch.setattr(nemo_gym.rollout_collection, "_build_ng_perf", MagicMock(side_effect=ValueError))

        _attach_ng_perf(result, observability_enabled=True, rollout_latency_ms=42.0)

        assert NG_PERF_KEY not in result

    async def test_run_examples_logs_failed_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        row = {
            AGENT_REF_KEY_NAME: {"name": "my_agent"},
            TASK_INDEX_KEY_NAME: 7,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "env_specific_metadata": "do not log this either",
            "responses_create_params": {"input": "do not log this"},
        }
        response = MagicMock()
        response.status = 500

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=response)
        mock_server_client.global_config_dict = OmegaConf.create({"my_agent": {"responses_api_agents": {"impl": {}}}})

        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )

        async def fail_raise_for_status(_response):
            raise RuntimeError("boom")

        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", fail_raise_for_status)

        with pytest.raises(RuntimeError, match="boom"):
            await next(RolloutCollectionHelper().run_examples([row]))

        captured = capsys.readouterr()
        assert "[rollout_collection] /run failed status=500" in captured.out
        assert '"_ng_task_index": 7' in captured.out
        assert '"_ng_rollout_index": 0' in captured.out
        assert '"agent_name": "my_agent"' in captured.out
        assert "env_specific_metadata" not in captured.out
        assert "do not log this either" not in captured.out
        assert "responses_create_params" not in captured.out
        assert "do not log this" not in captured.out
        assert "[rollout_collection] /run failed" in captured.out

    async def test_run_examples_records_agent_http_failure_as_a_failure_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 5xx from /run becomes a row-associated failure, not a reward-zero rollout."""
        row = failing_row()
        post = AsyncMock(return_value=FakeResponse(500))
        install_fake_server_client(monkeypatch, post)

        async def raise_for_status(_response):
            raise http_error(500, body=b'{"detail": "unhandled tool-call json"}')

        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", raise_for_status)

        returned_row, result = await next(
            RolloutCollectionHelper().run_examples([row], route_failures_to_sidecar=True)
        )

        assert returned_row is row
        assert result[NG_FAILURE_CLASS_KEY] == AGENT_RUN_ERROR_FAILURE_CLASS
        assert result["_ng_failure_type"] == "ClientResponseError"
        assert result["_ng_failure_http_status"] == 500
        assert result["_ng_failure_response_body"] == '{"detail": "unhandled tool-call json"}'
        # No invented verifier output: no reward, no placeholder response, no token payload.
        assert "reward" not in result
        assert "response" not in result
        assert NG_TERMINAL_KEY not in result

    @pytest.mark.parametrize("status", [401, 429, 503, 504])
    async def test_run_examples_records_any_status_without_resending(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        """No status is retried in place; the agent may already have run."""
        post = AsyncMock(return_value=FakeResponse(status))
        install_fake_server_client(monkeypatch, post)

        _, result = await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))

        assert result["_ng_failure_http_status"] == status
        assert post.await_count == 1

    @pytest.mark.parametrize(
        ("error", "status", "expected_class"),
        [
            (
                ClientConnectorError(MagicMock(), OSError("connection refused")),
                None,
                AGENT_REQUEST_FAILED_FAILURE_CLASS,
            ),
            (ServerDisconnectedError(), None, AGENT_REQUEST_FAILED_FAILURE_CLASS),
            (http_error(503), 503, AGENT_REQUEST_FAILED_FAILURE_CLASS),
            (http_error(500), 500, AGENT_RUN_ERROR_FAILURE_CLASS),
            (http_error(400), 400, AGENT_RUN_ERROR_FAILURE_CLASS),
            (orjson.JSONDecodeError("unexpected end of data", "", 0), 200, AGENT_RUN_ERROR_FAILURE_CLASS),
        ],
        ids=["connection refused", "dropped mid-flight", "gateway 503", "agent 500", "agent 400", "unreadable body"],
    )
    async def test_run_examples_classifies_by_who_answered(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException, status: int | None, expected_class: str
    ) -> None:
        """A NeMo Gym agent answers 500 when its handler raises, so its own statuses mean it ran.

        A gateway status or no reply at all does not, and neither is ever resent from here.
        """
        post = AsyncMock(side_effect=error) if status is None else AsyncMock(return_value=FakeResponse(status))
        install_fake_server_client(monkeypatch, post)
        if status == 200:

            async def get_response_json(_response):
                raise error

            monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", get_response_json)

        _, result = await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))

        assert result[NG_FAILURE_CLASS_KEY] == expected_class
        assert result["_ng_failure_type"] == type(error).__name__
        assert result["_ng_failure_http_status"] == status
        assert "reward" not in result
        assert post.await_count == 1

    async def test_run_examples_raises_for_direct_callers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Library callers (NeMo-RL) keep the exception contract they have today."""
        post = AsyncMock(return_value=FakeResponse(500))
        install_fake_server_client(monkeypatch, post)

        with pytest.raises(ClientResponseError):
            await next(RolloutCollectionHelper().run_examples([failing_row()]))

    @pytest.mark.parametrize("error", [RuntimeError("dispatcher bug"), asyncio.CancelledError()])
    async def test_run_examples_propagates_non_request_failures(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        """Routing covers request failures only; a bug or a cancellation still ends the run."""
        post = AsyncMock(side_effect=error)
        install_fake_server_client(monkeypatch, post)

        with pytest.raises(type(error)):
            await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))

    async def test_failure_row_survives_serialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The record has to cross the jsonl, pickle and Ray boundaries as plain data."""
        post = AsyncMock(return_value=FakeResponse(500))
        install_fake_server_client(monkeypatch, post)

        _, result = await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))

        assert pickle.loads(pickle.dumps(result)) == result
        assert orjson.loads(orjson.dumps(result)) == result

    async def test_run_from_config_routes_agent_failure_to_sidecar_and_out_of_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        empty_global_config: MagicMock,
    ) -> None:
        """End to end: one 500 and one success, through the real dispatch and aggregation path."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            "\n".join(
                json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}, "x": i})
                for i in range(2)
            )
            + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        aggregated: dict[str, list[dict]] = {}

        async def post(server_name: str, url_path: str, json: dict, **kwargs):
            if url_path == "/run":
                if json["x"] == 0:
                    raise http_error(500, "unhandled tool-call json")
                return FakeResponse(200, {"reward": 1.0, "response": {"usage": {"total_tokens": 3}}})
            assert url_path == "/aggregate_metrics"
            aggregated["verify_responses"] = [dict(r) for r in json.verify_responses]
            return FakeResponse(200, compute_aggregate_metrics(aggregated["verify_responses"]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            route_failures_to_sidecar=True,
            disable_health_check=True,
        )
        results = await RolloutCollectionHelper().run_from_config(config)

        assert len(results) == 2

        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert [r[TASK_INDEX_KEY_NAME] for r in persisted] == [1]
        assert [r["reward"] for r in persisted] == [1.0]

        failures = [orjson.loads(line) for line in _failures_path_for(output_jsonl_fpath).read_bytes().splitlines()]
        assert len(failures) == 1
        assert failures[0][NG_FAILURE_CLASS_KEY] == AGENT_RUN_ERROR_FAILURE_CLASS
        assert failures[0][TASK_INDEX_KEY_NAME] == 0
        assert failures[0][ROLLOUT_INDEX_KEY_NAME] == 0
        assert failures[0][AGENT_REF_KEY_NAME] == {"name": "my_agent"}
        assert "reward" not in failures[0]

        # The failed rollout reaches neither the aggregator's input nor its denominator.
        assert [r[TASK_INDEX_KEY_NAME] for r in aggregated["verify_responses"]] == [1]
        metrics_fpath = output_jsonl_fpath.with_stem(output_jsonl_fpath.stem + "_aggregate_metrics").with_suffix(
            ".json"
        )
        agent_metrics = orjson.loads(metrics_fpath.read_bytes())[0]["key_metrics"]
        assert agent_metrics["mean/reward"] == 1.0

        # The run says the setting is on, names each dropped rollout once, and closes with the count.
        printed = capsys.readouterr().out
        assert "route_failures_to_sidecar is on" in printed
        assert printed.count("rollout dropped from the score") == 1
        assert "Rollouts missing from the score: 1 of 2 materialized" in printed
        assert "Metrics cover: 1 of 2 rollouts" in printed
        assert str(_failures_path_for(output_jsonl_fpath)) in printed

    async def test_run_from_config_resume_retries_an_agent_failure_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """A failure row is one attempt, so resume re-dispatches it with a fresh attempt index."""
        output_jsonl_fpath = tmp_path / "output.jsonl"
        output_jsonl_fpath.write_bytes(b"")
        materialized_fpath = tmp_path / "output_materialized_inputs.jsonl"
        materialized_fpath.write_bytes(
            orjson.dumps(
                {
                    "responses_create_params": {"input": []},
                    AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                }
            )
            + b"\n"
        )
        _failures_path_for(output_jsonl_fpath).write_bytes(
            orjson.dumps(
                {
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    NG_FAILURE_CLASS_KEY: AGENT_REQUEST_FAILED_FAILURE_CLASS,
                }
            )
            + b"\n"
        )

        dispatched: list[dict] = []

        async def post(server_name: str, url_path: str, json: dict, **kwargs):
            if url_path == "/run":
                dispatched.append(json)
                return FakeResponse(200, {"reward": 1.0})
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            disable_health_check=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        assert len(dispatched) == 1
        assert dispatched[0][ATTEMPT_INDEX_KEY_NAME] == 1
        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert [r["reward"] for r in persisted] == [1.0]

    def test_failure_rows_counted_as_zero_selects_the_last_attempt_of_each_rollout(self, tmp_path: Path) -> None:
        """The last attempt stands, so it is chosen before the wanted classes are picked out."""
        failures_fpath = tmp_path / "output_failures.jsonl"

        def attempt(task_index: int, failure_class: str, reward: float | None = 0.0) -> dict:
            row = {
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
                NG_FAILURE_CLASS_KEY: failure_class,
                "_ng_failure_http_status": 500,
            }
            return row if reward is None else {**row, "reward": reward}

        failures_fpath.write_bytes(
            b"\n".join(
                orjson.dumps(row)
                for row in [
                    attempt(0, AGENT_RUN_ERROR_FAILURE_CLASS),
                    attempt(1, AGENT_RUN_ERROR_FAILURE_CLASS),
                    attempt(1, AGENT_RUN_ERROR_FAILURE_CLASS),
                    attempt(2, AGENT_RUN_ERROR_FAILURE_CLASS, reward=None),
                    attempt(3, AGENT_RUN_ERROR_FAILURE_CLASS),
                    attempt(3, AGENT_REQUEST_FAILED_FAILURE_CLASS),
                    attempt(4, AGENT_REQUEST_FAILED_FAILURE_CLASS),
                    attempt(4, AGENT_RUN_ERROR_FAILURE_CLASS),
                ]
            )
            + b"\n"
        )

        # Task 0 succeeded on a later attempt. Task 1 failed twice and counts once. Task 3 ended in
        # a class the caller did not ask for, so its earlier attempt must not stand in for it.
        rows = _failure_rows_counted_as_zero([failures_fpath], [AGENT_RUN_ERROR_FAILURE_CLASS], {(0, 0)})
        assert sorted(row[TASK_INDEX_KEY_NAME] for row in rows) == [1, 2, 4]

        assert _failure_rows_counted_as_zero([failures_fpath], [], set()) == []

        # A row with no reward is scored zero here and only here, and diagnostics never reach the
        # aggregator, which averages every number it is handed.
        scoreless = next(row for row in rows if row[TASK_INDEX_KEY_NAME] == 2)
        assert scoreless["reward"] == 0.0
        assert not any(key.startswith("_ng_failure_") for key in scoreless)
        sidecar = [orjson.loads(line) for line in failures_fpath.read_bytes().splitlines()]
        assert "reward" not in sidecar[3]
        assert sidecar[3]["_ng_failure_http_status"] == 500

    @pytest.mark.parametrize(
        ("counted_classes", "expected_scored", "expected_mean"),
        [([], 1, 1.0), ([AGENT_RUN_ERROR_FAILURE_CLASS], 2, 0.5)],
        ids=["off by default", "opted in"],
    )
    async def test_run_from_config_counts_an_opted_in_failure_class_as_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        counted_classes: list[str],
        expected_scored: int,
        expected_mean: float,
    ) -> None:
        """One 500 and one success, end to end: the failed rollout counts only when asked for.

        `key_metrics` is asserted whole, because the row's diagnostic fields are numbers and the
        aggregator averages every number it is handed.
        """
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            "\n".join(
                json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}, "x": i})
                for i in range(2)
            )
            + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        aggregated: dict[str, list[dict]] = {}

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                if json["x"] == 0:
                    raise http_error(500, "unhandled tool-call json")
                return FakeResponse(200, {"reward": 1.0})
            aggregated["verify_responses"] = [dict(r) for r in json.verify_responses]
            return FakeResponse(200, compute_aggregate_metrics(aggregated["verify_responses"]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            route_failures_to_sidecar=True,
            count_failure_classes_as_zero=counted_classes,
            disable_health_check=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert [row["reward"] for row in persisted] == [1.0]

        failures = [orjson.loads(line) for line in _failures_path_for(output_jsonl_fpath).read_bytes().splitlines()]
        assert failures[0][NG_FAILURE_CLASS_KEY] == AGENT_RUN_ERROR_FAILURE_CLASS
        assert "reward" not in failures[0]

        assert len(aggregated["verify_responses"]) == expected_scored
        metrics_fpath = output_jsonl_fpath.with_stem(output_jsonl_fpath.stem + "_aggregate_metrics").with_suffix(
            ".json"
        )
        assert orjson.loads(metrics_fpath.read_bytes())[0]["key_metrics"] == {"mean/reward": expected_mean}

    async def test_run_from_config_fails_when_no_rollout_produced_a_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """Routing failures out of the score must not turn a dead run into a quiet success."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(500)))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            route_failures_to_sidecar=True,
            disable_health_check=True,
        )

        with pytest.raises(RuntimeError, match="produced a result"):
            await RolloutCollectionHelper().run_from_config(config)

        # The attempt is still on disk, so resume can pick it up.
        failures = [orjson.loads(line) for line in _failures_path_for(output_jsonl_fpath).read_bytes().splitlines()]
        assert [row[NG_FAILURE_CLASS_KEY] for row in failures] == [AGENT_RUN_ERROR_FAILURE_CLASS]

    async def test_aggregate_counts_an_opted_in_failure_class_from_each_shard_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """`gym eval aggregate` applies the same option offline, over the shards' own sidecars."""
        shard_fpath = tmp_path / "rollouts-chunk0.jsonl"
        shard_fpath.write_bytes(
            orjson.dumps(
                {
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    "reward": 1.0,
                }
            )
            + b"\n"
        )
        _failures_path_for(shard_fpath).write_bytes(
            orjson.dumps(
                {
                    TASK_INDEX_KEY_NAME: 1,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    "reward": 0.0,
                    NG_FAILURE_CLASS_KEY: "verify_failed",
                }
            )
            + b"\n"
        )
        merged_fpath = tmp_path / "rollouts.jsonl"
        aggregated: dict[str, list[dict]] = {}

        async def post(server_name: str, url_path: str, json, **kwargs):
            aggregated["verify_responses"] = [dict(r) for r in json.verify_responses]
            return FakeResponse(200, compute_aggregate_metrics(aggregated["verify_responses"]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutAggregationConfig(
            input_glob=str(shard_fpath),
            output_jsonl_fpath=str(merged_fpath),
            count_failure_classes_as_zero=["verify_failed"],
            disable_health_check=True,
        )
        await RolloutAggregationHelper().run_from_config(config)

        assert sorted(row["reward"] for row in aggregated["verify_responses"]) == [0.0, 1.0]
        # The merged rollouts file keeps only the rollouts that produced a result.
        merged = [orjson.loads(line) for line in merged_fpath.read_bytes().splitlines()]
        assert [row["reward"] for row in merged] == [1.0]

    async def test_run_examples_never_leaks_rollout_latency_into_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Direct callers (e.g. NeMo-RL) get exactly the raw /run result, with no Gym-private fields."""
        row = {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        response = MagicMock()
        response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=response)
        mock_server_client.global_config_dict = OmegaConf.create({"my_agent": {"responses_api_agents": {"impl": {}}}})
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"response": {}}))

        returned_row, result = await next(RolloutCollectionHelper().run_examples([row]))

        assert returned_row is row
        assert result == {"response": {}}
        assert "_ng_rollout_latency_ms" not in result

    async def test_run_examples_with_metadata_carries_rollout_latency_alongside_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Internal callers get the timing via _CompletedRollout, never through the result dict."""
        row = {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        response = MagicMock()
        response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=response)
        mock_server_client.global_config_dict = OmegaConf.create({"my_agent": {"responses_api_agents": {"impl": {}}}})
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"response": {}}))

        completed = await next(RolloutCollectionHelper()._run_examples_with_metadata([row]))

        assert completed.row is row
        assert completed.result == {"response": {}}
        assert isinstance(completed.rollout_latency_ms, float)
        assert completed.rollout_latency_ms >= 0

    async def test_run_from_config_does_not_route_failures_unless_asked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """Dropping failed rollouts shrinks the denominator, so it never happens unasked."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(500)))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            disable_health_check=True,
        )

        with pytest.raises(ClientResponseError):
            await RolloutCollectionHelper().run_from_config(config)

        assert (
            not _failures_path_for(output_jsonl_fpath).exists()
            or not _failures_path_for(output_jsonl_fpath).read_bytes()
        )

    async def test_run_from_config_reports_rollouts_dropped_by_the_agent_with_routing_off(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        empty_global_config: MagicMock,
    ) -> None:
        """Agents route their own failures whatever this flag says, so those are announced too."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            "\n".join(
                json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
                for i in range(2)
            )
            + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                assert kwargs["route_failures_to_sidecar"] is False
                futures = []
                for example in examples:
                    future = Future()
                    scored = {"reward": 1.0}
                    judge_failed = {"reward": 0.0, NG_FAILURE_CLASS_KEY: "judge_failed", "error": "judge 503"}
                    future.set_result(
                        _CompletedRollout(
                            row=example,
                            result=scored if example["x"] == 0 else judge_failed,
                            rollout_latency_ms=None,
                        )
                    )
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            disable_health_check=True,
        )
        await Helper().run_from_config(config)

        printed = capsys.readouterr().out
        assert "route_failures_to_sidecar is on" not in printed
        assert printed.count("rollout dropped from the score") == 1
        assert "class=judge_failed" in printed
        assert "judge 503" in printed
        assert "Rollouts missing from the score: 1 of 2 materialized" in printed

    async def test_run_from_config_reports_coverage_against_the_materialized_input_on_resume(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        empty_global_config: MagicMock,
    ) -> None:
        """A resumed hop dispatches little and can still be missing rollouts from earlier hops."""
        output_jsonl_fpath = tmp_path / "output.jsonl"
        materialized_fpath = tmp_path / "output_materialized_inputs.jsonl"
        rows = [
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"name": "my agent name"},
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for task_index in range(3)
        ]
        materialized_fpath.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
        # Two rollouts are already scored, and the third is out of attempts, so this hop runs nothing.
        output_jsonl_fpath.write_bytes(b"\n".join(orjson.dumps({**row, "reward": 1.0}) for row in rows[:2]) + b"\n")
        _failures_path_for(output_jsonl_fpath).write_bytes(
            orjson.dumps({**rows[2], NG_FAILURE_CLASS_KEY: AGENT_RUN_ERROR_FAILURE_CLASS, NG_TERMINAL_KEY: True})
            + b"\n"
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                assert examples == []
                return []

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            disable_health_check=True,
        )
        await Helper().run_from_config(config)

        printed = capsys.readouterr().out
        assert "Rollouts missing from the score: 1 of 3 materialized" in printed
        assert "Metrics cover: 2 of 3 rollouts" in printed

    def test_preprocess_rows_with_prompt_config(self, tmp_path: Path) -> None:
        """prompt_config builds responses_create_params.input from template."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"system": "You are a math tutor.", "user": "Solve: {question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [
            {"question": "What is 2+2?", "expected_answer": "4"},
            {"question": "What is 3*5?", "expected_answer": "15"},
        ]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(result) == 2
        assert result[0]["responses_create_params"]["input"] == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve: What is 2+2?"},
        ]
        assert result[0]["expected_answer"] == "4"
        assert result[1]["responses_create_params"]["input"][1]["content"] == "Solve: What is 3*5?"

    def test_preprocess_rows_prompt_config_rejects_prebaked(self, tmp_path: Path) -> None:
        """prompt_config raises when rows already have responses_create_params.input."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [{"question": "test", "responses_create_params": {"input": [{"role": "user", "content": "baked"}]}}]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)

    def test_preprocess_rows_missing_input_raises_config_error(self, tmp_path: Path) -> None:
        """A non-existent input file fails with a clean ConfigPathNotFoundError, not a raw FileNotFoundError."""
        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(tmp_path / "does_not_exist.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        )

        with pytest.raises(ConfigPathNotFoundError, match="does_not_exist.jsonl.*--input"):
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)

    def test_preprocess_rows_prompt_config_preserves_rcp_fields(self, tmp_path: Path) -> None:
        """prompt_config preserves other responses_create_params fields like tools."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [{"question": "test", "responses_create_params": {"tools": [{"type": "function", "name": "calc"}]}}]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert result[0]["responses_create_params"]["tools"] == [{"type": "function", "name": "calc"}]
        assert result[0]["responses_create_params"]["input"] == [{"role": "user", "content": "test"}]

    def test_preprocess_rows_from_config(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(10)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath="abcd",
            limit=3,
            num_repeats=2,
            num_repeats_add_seed=True,
            num_samples_in_parallel=None,
            responses_create_params=dict(temperature=0.1),
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows == [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
        ]

    def test_preprocess_rows_stamps_skills_ref(self, tmp_path: Path) -> None:
        """skills.path is a run-level knob: each row is stamped with skills_ref (path + hash +
        metadata) without the source dataset carrying any skills field."""
        skills_dir = tmp_path / "variant_a"
        skill = skills_dir / "cot_enhanced"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: cot_enhanced\ndescription: Think step by step.\n---\n# Body\n")

        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(2)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            skills={"path": str(skills_dir)},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(rows) == 2
        for row in rows:
            skills_ref = row["skills_ref"]
            assert skills_ref["path"] == str(skills_dir)
            assert len(skills_ref["hash"]) == 12
            assert [s["name"] for s in skills_ref["skills"]] == ["cot_enhanced"]
            assert skills_ref["skills"][0]["description"] == "Think step by step."

    def test_preprocess_rows_no_skills_leaves_rows_clean(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")
        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert "skills_ref" not in rows[0]

    def test_skills_ref_survives_resume_from_cache(self, tmp_path: Path) -> None:
        """skills_ref is stamped once at preprocess, persisted to materialized inputs, and
        re-read onto already-done rows on resume -- even after the source skill dir is gone.
        Identity is byte-for-byte from the materialized cache, not recomputed at resume."""
        import shutil

        skills_dir = tmp_path / "variant_a"
        skill = skills_dir / "cot_enhanced"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: cot_enhanced\ndescription: Think step by step.\n---\n# Body\n")

        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(2)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            skills={"path": str(skills_dir)},
            resume_from_cache=True,
        )

        # Preprocess stamps skills_ref, then we persist exactly what a prior run would have written.
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        stamped_skills_ref = rows[0]["skills_ref"]
        config.materialized_jsonl_fpath.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")

        # Only the first task's rollout is "done" in the main output jsonl.
        done = {k: rows[0][k] for k in (TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME)} | {"reward": 1.0}
        Path(config.output_jsonl_fpath).write_bytes(orjson.dumps(done) + b"\n")

        # The source skill dir disappears before resume (e.g. an optimizer overwrote /tmp).
        shutil.rmtree(skills_dir)

        input_rows, resumed_rows, _results, _result_strs = RolloutCollectionHelper()._load_from_cache(config)

        # The already-done row carries the original skills_ref read back from the cache.
        assert resumed_rows[0]["skills_ref"] == stamped_skills_ref
        # And the still-to-run rows do too, so the second pass stamps results identically.
        assert all(r["skills_ref"] == stamped_skills_ref for r in input_rows)

    def test_preprocess_rows_num_repeats_add_seed_passes_pydantic_validation(self, tmp_path: Path) -> None:
        """Rows emitted with num_repeats_add_seed=True must round-trip through the strict
        NeMoGymResponseCreateParamsNonStreaming schema (extra='forbid'). Seed is passed via
        metadata.extra_body so it doesn't violate the OpenAI Responses schema."""
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(2)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats=3,
            num_repeats_add_seed=True,
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(rows) == 6
        seeds_seen = []
        for row in rows:
            rcp = row["responses_create_params"]
            # seed lives in metadata.extra_body, not at the top level
            assert "seed" not in rcp
            extra_body = json.loads(rcp["metadata"]["extra_body"])
            seeds_seen.append(extra_body["seed"])
            # Must still pass the strict schema validation
            NeMoGymResponseCreateParamsNonStreaming.model_validate(rcp)
        # Seeds should track rollout index within each task (0, 1, 2 per task).
        assert seeds_seen == [0, 1, 2, 0, 1, 2]

    def test_preprocess_rows_num_repeats_dict_form(self, tmp_path: Path) -> None:
        """Dict-form num_repeats applies the per-agent value to each row."""
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 1}),
        ]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 2, "beta": 4},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        per_agent_counts = Counter(row[AGENT_REF_KEY_NAME]["name"] for row in rows)
        assert per_agent_counts == Counter({"alpha": 2, "beta": 4})
        assert [r[ROLLOUT_INDEX_KEY_NAME] for r in rows if r[AGENT_REF_KEY_NAME]["name"] == "alpha"] == [0, 1]
        assert [r[ROLLOUT_INDEX_KEY_NAME] for r in rows if r[AGENT_REF_KEY_NAME]["name"] == "beta"] == [0, 1, 2, 3]

    def test_preprocess_rows_num_repeats_dict_with_default(self, tmp_path: Path) -> None:
        """`_default` key acts as the fallback for agents not explicitly listed."""
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 1}),
        ]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 3, "_default": 1},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        per_agent_counts = Counter(row[AGENT_REF_KEY_NAME]["name"] for row in rows)
        assert per_agent_counts == Counter({"alpha": 3, "beta": 1})

    def test_preprocess_rows_num_repeats_dict_raises_on_missing_agent_no_default(self, tmp_path: Path) -> None:
        """Dict form without `_default` raises if a row's agent is unlisted, and reports ALL
        missing agents in one error so the user can fix them in one pass."""
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 1}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "gamma"}, "x": 2}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 3}),
        ]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 2},
        )

        with pytest.raises(ValueError) as exc_info:
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        msg = str(exc_info.value)
        # All missing agents reported in one shot, deduped:
        assert "'beta'" in msg
        assert "'gamma'" in msg

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_preprocess_rows_num_repeats_rejects_zero_or_negative(self, tmp_path: Path, bad_value: int) -> None:
        # int form
        with pytest.raises(ValueError, match="num_repeats"):
            RolloutCollectionConfig(
                agent_name="my_agent",
                input_jsonl_fpath=str(tmp_path / "in.jsonl"),
                output_jsonl_fpath=str(tmp_path / "out.jsonl"),
                num_repeats=bad_value,
            )
        # dict form
        with pytest.raises(ValueError, match="num_repeats dict"):
            RolloutCollectionConfig(
                agent_name="my_agent",
                input_jsonl_fpath=str(tmp_path / "in.jsonl"),
                output_jsonl_fpath=str(tmp_path / "out.jsonl"),
                num_repeats={"alpha": bad_value},
            )

    def test_num_repeats_null_coerces_to_one(self, tmp_path: Path) -> None:
        # `--num-repeats null` (None) restores the pre-#1356 default of 1.
        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(tmp_path / "in.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats=None,
        )
        assert config.num_repeats == 1

    def test_preprocess_rows_num_repeats_dict_unknown_agent_warns(self, tmp_path: Path) -> None:
        """An agent listed in the dict that never appears in input rows warns (likely typo)."""
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0})]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 2, "alpah_typo": 3},
        )

        with pytest.warns(UserWarning, match="alpah_typo"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 2

    async def test_run_from_config_sanity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        clear_captures = MagicMock()
        merge_capture = MagicMock()
        monkeypatch.setattr(nemo_gym.rollout_collection, "clear_model_call_captures_for_rollouts", clear_captures)
        monkeypatch.setattr(nemo_gym.rollout_collection, "merge_model_call_capture_into_record", merge_capture)
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(10)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def _run_examples_with_metadata(
                self,
                examples: list[dict],
                *args,
                **kwargs,
            ):
                futures = []
                for example in examples:
                    future = Future()
                    # (row, result)
                    future.set_result(
                        _CompletedRollout(
                            row=example, result={"response": {"usage": {"abc usage": 1}}}, rollout_latency_ms=None
                        )
                    )
                    futures.append(future)

                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                """Compute aggregate metrics locally (no server needed)."""
                stripped = [{k: v for k, v in r.items() if k not in ("responses_create_params",)} for r in results]
                agg = compute_aggregate_metrics(stripped)
                metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
                metrics_fpath.write_bytes(
                    orjson.dumps(
                        [{"agent_ref": {"name": "my agent name"}, **agg.model_dump()}], option=orjson.OPT_INDENT_2
                    )
                )
                return metrics_fpath

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)
        empty_global_config.assert_called_once_with()
        clear_captures.assert_not_called()
        merge_capture.assert_not_called()

        expected_results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
        ]

        assert expected_results == actual_returned_results

        expected_materialized_inputs_len = 6
        with (tmp_path / "output_materialized_inputs.jsonl").open() as f:
            actual_materialized_inputs_len = len(list(f))
        assert expected_materialized_inputs_len == actual_materialized_inputs_len

        with output_jsonl_fpath.open() as f:
            actual_written_results = [json.loads(line) for line in f]
        assert expected_results == actual_written_results

        aggregate_metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        actual_aggregate_metrics = json.loads(aggregate_metrics_fpath.read_text())
        assert len(actual_aggregate_metrics) == 1
        assert actual_aggregate_metrics[0]["agent_ref"] == {"name": "my agent name"}

        # Base per-rollout stats are unaffected by the repeat-level aggregation merged in below.
        agent_metrics = actual_aggregate_metrics[0]["agent_metrics"]
        assert agent_metrics["mean/abc usage"] == pytest.approx(1.0)
        assert agent_metrics["max/abc usage"] == 1
        assert agent_metrics["min/abc usage"] == 1
        assert agent_metrics["median/abc usage"] == pytest.approx(1.0)
        assert agent_metrics["std/abc usage"] == pytest.approx(0.0)
        assert actual_aggregate_metrics[0]["key_metrics"]["mean/abc usage"] == pytest.approx(1.0)

        # num_repeats=2 -> repeat_level_metrics has one entry per rollout_index (0 and 1),
        # each aggregating the "abc usage" metric across all 3 tasks at that repeat.
        repeat_level_metrics = actual_aggregate_metrics[0]["repeat_level_metrics"]
        assert len(repeat_level_metrics) == 2
        rollout_indices = {entry[ROLLOUT_INDEX_KEY_NAME] for entry in repeat_level_metrics}
        assert rollout_indices == {0, 1}
        for entry in repeat_level_metrics:
            assert entry["sample_count"] == 3
            assert entry["missing_count"] == 0
            assert entry["mean/abc usage"] == pytest.approx(1.0)
            assert entry["std/abc usage"] == pytest.approx(0.0)

        # Cross-repeat aggregates (mean/median/se of the per-repeat "mean/abc usage" estimate)
        # are merged into agent_metrics -- both repeats agree exactly (constant "abc usage"=1),
        # so the cross-repeat mean/median equal 1.0 and the SE across repeats is 0.
        assert agent_metrics["mean_across_repeats/mean/abc usage"] == pytest.approx(1.0)
        assert agent_metrics["median_across_repeats/mean/abc usage"] == pytest.approx(1.0)
        assert agent_metrics["se_across_repeats/mean/abc usage"] == pytest.approx(0.0)

    async def test_run_from_config_repeat_level_metrics_e2e(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """End-to-end: full run_from_config pipeline -> aggregate metrics JSON on disk carries
        variability statistics (mean/std/sem/CI) per rollout_index when num_repeats >= 2, computed
        from a per-task reward that varies by both task and rollout so the stats aren't degenerate.
        """
        clear_captures = MagicMock()
        merge_capture = MagicMock()
        monkeypatch.setattr(nemo_gym.rollout_collection, "clear_model_call_captures_for_rollouts", clear_captures)
        monkeypatch.setattr(nemo_gym.rollout_collection, "merge_model_call_capture_into_record", merge_capture)

        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(4)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            num_repeats=3,
        )

        # Deterministic per-(task, rollout) reward so we can hand-verify mean/std below:
        # rollout 0 rewards across the 4 tasks: 0, 1, 2, 3 (mean=1.5)
        # rollout 1 rewards across the 4 tasks: 1, 2, 3, 4 (mean=2.5)
        # rollout 2 rewards across the 4 tasks: 2, 3, 4, 5 (mean=3.5)
        def reward_for(task_idx: int, rollout_idx: int) -> float:
            return float(task_idx + rollout_idx)

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    task_idx = example[TASK_INDEX_KEY_NAME]
                    rollout_idx = example[ROLLOUT_INDEX_KEY_NAME]
                    future.set_result(
                        _CompletedRollout(
                            row=example,
                            result={"response": {}, "reward": reward_for(task_idx, rollout_idx)},
                            rollout_latency_ms=None,
                        )
                    )
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                stripped = [{k: v for k, v in r.items() if k not in ("responses_create_params",)} for r in results]
                agg = compute_aggregate_metrics(stripped)
                metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
                metrics_fpath.write_bytes(
                    orjson.dumps(
                        [{"agent_ref": {"name": "my agent name"}, **agg.model_dump()}], option=orjson.OPT_INDENT_2
                    )
                )
                return metrics_fpath

        await TestRolloutCollectionHelper().run_from_config(config)

        aggregate_metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        actual_aggregate_metrics = json.loads(aggregate_metrics_fpath.read_text())
        assert len(actual_aggregate_metrics) == 1

        repeat_level_metrics = actual_aggregate_metrics[0]["repeat_level_metrics"]
        assert len(repeat_level_metrics) == 3
        by_rollout_idx = {entry[ROLLOUT_INDEX_KEY_NAME]: entry for entry in repeat_level_metrics}
        assert set(by_rollout_idx) == {0, 1, 2}

        for rollout_idx, expected_mean in ((0, 1.5), (1, 2.5), (2, 3.5)):
            entry = by_rollout_idx[rollout_idx]
            assert entry["sample_count"] == 4
            assert entry["missing_count"] == 0
            assert entry["mean/reward"] == pytest.approx(expected_mean)
            # rewards at each repeat are 4 consecutive integers -> population-style sample std
            # (ddof=1) of [n, n+1, n+2, n+3] is sqrt(20/12*... ) == std of [0,1,2,3] == ~1.29099
            assert entry["std/reward"] == pytest.approx(1.2909944, rel=1e-4)
            assert entry["min/reward"] == pytest.approx(expected_mean - 1.5)
            assert entry["max/reward"] == pytest.approx(expected_mean + 1.5)
            # 4 samples -> sem and 95% CI are emitted
            assert entry["sem/reward"] == pytest.approx(entry["std/reward"] / (4**0.5))
            assert entry["ci_low_95/reward"] < entry["mean/reward"] < entry["ci_high_95/reward"]

        # Repeats differ (task+rollout reward), so the cross-repeat means themselves vary --
        # a real regression in the grouping (e.g. averaging over rollout_index instead of by it)
        # would collapse these to a single repeated value.
        means = [by_rollout_idx[i]["mean/reward"] for i in range(3)]
        assert means == sorted(means)
        assert len(set(means)) == 3

    async def test_run_from_config_repeat_level_metrics_absent_for_single_repeat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """With num_repeats=1 there is nothing to compare across repeats, so the aggregate metrics
        JSON on disk should carry an empty repeat_level_metrics list rather than a single-entry one.
        """
        clear_captures = MagicMock()
        merge_capture = MagicMock()
        monkeypatch.setattr(nemo_gym.rollout_collection, "clear_model_call_captures_for_rollouts", clear_captures)
        monkeypatch.setattr(nemo_gym.rollout_collection, "merge_model_call_capture_into_record", merge_capture)

        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(4)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            num_repeats=1,
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    future.set_result(
                        _CompletedRollout(row=example, result={"response": {}, "reward": 1.0}, rollout_latency_ms=None)
                    )
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                stripped = [{k: v for k, v in r.items() if k not in ("responses_create_params",)} for r in results]
                agg = compute_aggregate_metrics(stripped)
                metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
                metrics_fpath.write_bytes(
                    orjson.dumps(
                        [{"agent_ref": {"name": "my agent name"}, **agg.model_dump()}], option=orjson.OPT_INDENT_2
                    )
                )
                return metrics_fpath

        await TestRolloutCollectionHelper().run_from_config(config)

        aggregate_metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        actual_aggregate_metrics = json.loads(aggregate_metrics_fpath.read_text())
        assert actual_aggregate_metrics[0]["repeat_level_metrics"] == []
        expected_aggregate_metrics = [
            {
                "agent_ref": {"name": "my agent name"},
                "agent_metrics": {
                    "mean/reward": 1.0,
                    "max/reward": 1.0,
                    "min/reward": 1.0,
                    "median/reward": 1.0,
                    "std/reward": 0.0,
                },
                "key_metrics": {"mean/reward": 1.0},
                "group_level_metrics": actual_aggregate_metrics[0]["group_level_metrics"],
                "perf_summary": None,
                "repeat_level_metrics": [],
            }
        ]
        assert expected_aggregate_metrics == actual_aggregate_metrics

    async def test_run_from_config_clears_failure_sidecar_on_fresh_run(
        self, tmp_path: Path, empty_global_config: MagicMock
    ) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        failures_fpath = _failures_path_for(output_jsonl_fpath)
        failures_fpath.write_text(json.dumps({"_ng_failure_class": "stale_failure"}) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=False,
            disable_aggregation=True,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                future = Future()
                future.set_result(_CompletedRollout(row=examples[0], result={"reward": 1.0}, rollout_latency_ms=None))
                return [future]

        await Helper().run_from_config(config)

        assert failures_fpath.read_bytes() == b""

    @pytest.mark.parametrize("resume_from_cache", [False, True])
    async def test_run_from_config_creates_missing_output_dir(
        self, tmp_path: Path, empty_global_config: MagicMock, resume_from_cache: bool
    ) -> None:
        """--output under a directory that doesn't exist yet must not raise.

        The first artifact written is the materialized inputs, so a mkdir placed after it (or only
        alongside the rollouts write) leaves this failing. resume_from_cache=True takes the same
        path here because neither cached file exists, and must not be tripped up by the new dir.
        """
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}}) + "\n"
        )
        # Two levels deep so `parents=True` is exercised, not just a single missing dir.
        output_jsonl_fpath = tmp_path / "results" / "nested" / "rollouts.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=resume_from_cache,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    future.set_result(
                        _CompletedRollout(
                            row=example, result={"response": {"usage": {"abc usage": 1}}}, rollout_latency_ms=None
                        )
                    )
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
                metrics_fpath.write_bytes(orjson.dumps([]))
                return metrics_fpath

        await Helper().run_from_config(config)

        # All four artifacts share output_fpath's parent, so one mkdir has to cover all of them.
        assert config.materialized_jsonl_fpath.exists()
        assert output_jsonl_fpath.exists()
        assert _failures_path_for(output_jsonl_fpath).exists()
        assert output_jsonl_fpath.with_name("rollouts_aggregate_metrics.json").exists()

    @pytest.mark.parametrize("resume_from_cache", [False, True])
    @pytest.mark.parametrize("redact_payloads", [False, True])
    async def test_run_from_config_replaces_stale_capture_before_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_from_cache: bool, redact_payloads: bool
    ) -> None:
        from nemo_gym.base_responses_api_model import CaptureStore

        capture_dir = tmp_path / "captures"
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_global_config_dict",
            lambda: {"observability_enabled": True, "model_call_capture_dir": str(capture_dir)},
        )

        source_row = {"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "agent"}}
        row = {**source_row, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        input_fpath = tmp_path / "input.jsonl"
        output_fpath = tmp_path / "output.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            resume_from_cache=resume_from_cache,
            disable_aggregation=True,
        )
        if resume_from_cache:
            output_fpath.touch()
            config.materialized_jsonl_fpath.write_bytes(orjson.dumps(row) + b"\n")
        else:
            input_fpath.write_bytes(orjson.dumps(source_row) + b"\n")

        store = CaptureStore(capture_dir)
        store.record("0-0", {"model_call_id": "stale", "dialect": "responses", "request": {}, "response": {}})

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                [example] = examples
                assert example[TASK_INDEX_KEY_NAME] == 0 and example[ROLLOUT_INDEX_KEY_NAME] == 0
                assert store.read("0-0") == []
                request = {"input": [{"type": "input_image", "image_url": "data:image/png;base64,secret"}]}
                store.record(
                    "0-0",
                    {"model_call_id": "fresh", "dialect": "responses", "request": request, "response": {}},
                )
                future = Future()
                result = {"response": {"usage": {}}}
                if redact_payloads:
                    result["ng_trajectory"] = {
                        "schema_version": "1.0",
                        "task_id": "0",
                        "rollout_id": "0-0",
                        "gaps": [{"code": "multimodal_history_redacted"}],
                    }
                future.set_result(_CompletedRollout(row=example, result=result, rollout_latency_ms=None))
                return [future]

        results = await Helper().run_from_config(config)

        assert [exchange["model_call_id"] for exchange in store.read("0-0")] == ["fresh"]
        assert [call["model_call_id"] for call in results[0]["ng_model_call_capture"]["calls"]] == ["fresh"]
        trajectory_request = results[0]["ng_trajectory"]["model_calls"][0]["request"]
        trajectory_response = results[0]["ng_trajectory"]["model_calls"][0]["response"]
        if redact_payloads:
            assert trajectory_request is None and trajectory_response is None
        else:
            assert trajectory_request["input"][0]["type"] == "input_image" and trajectory_response == {}
        assert "request" not in results[0]["ng_model_call_capture"]["calls"][0]
        assert store.read("0-0")[0]["request"]["input"][0]["type"] == "input_image"
        if redact_payloads:
            assert "data:image/png;base64,secret" not in orjson.dumps(results[0]).decode()

    async def test_run_from_config_keys_capture_by_an_explicit_rollout_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_gym.base_responses_api_model import CaptureStore
        from nemo_gym.global_config import ROLLOUT_ID_KEY_NAME

        capture_dir = tmp_path / "captures"
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_global_config_dict",
            lambda: {"observability_enabled": True, "model_call_capture_dir": str(capture_dir)},
        )

        # These indices would derive ``0-0``.
        # The explicit id must win for both writer and consumer.
        # Otherwise readback finds no matching capture.
        source_row = {
            "responses_create_params": {"input": []},
            AGENT_REF_KEY_NAME: {"name": "agent"},
            ROLLOUT_ID_KEY_NAME: "step7.0-0",
        }
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_bytes(orjson.dumps(source_row) + b"\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            resume_from_cache=False,
            disable_aggregation=True,
        )

        store = CaptureStore(capture_dir)

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                [example] = examples
                store.record(
                    "step7.0-0",
                    {"model_call_id": "call", "dialect": "responses", "request": {}, "response": {}},
                )
                future = Future()
                future.set_result(
                    _CompletedRollout(row=example, result={"response": {"usage": {}}}, rollout_latency_ms=None)
                )
                return [future]

        results = await Helper().run_from_config(config)

        assert results[0][ROLLOUT_ID_KEY_NAME] == "step7.0-0"
        assert [call["model_call_id"] for call in results[0]["ng_model_call_capture"]["calls"]] == ["call"]
        # No capture uses the derived id.
        # The explicit id replaces it.
        assert store.read("0-0") == []

    async def test_run_from_config_does_not_finalize_a_nonparticipating_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture_dir = tmp_path / "tokens"
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_global_config_dict",
            lambda: {
                "token_id_capture": {"enabled": True, "dir": str(capture_dir)},
                "agent": {"responses_api_agents": {"implementation": {"token_id_capture": False}}},
            },
        )
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_bytes(
            orjson.dumps(
                {
                    "responses_create_params": {"input": []},
                    AGENT_REF_KEY_NAME: {"name": "agent"},
                }
            )
            + b"\n"
        )
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            resume_from_cache=False,
            disable_aggregation=True,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                [example] = examples
                future = Future()
                future.set_result(
                    _CompletedRollout(
                        row=example, result={"response": {"output": [], "usage": {}}}, rollout_latency_ms=None
                    )
                )
                return [future]

        [result] = await Helper().run_from_config(config)

        assert MASK_SAMPLE_KEY not in result
        assert TOKEN_CAPTURE_KEY not in result

    async def test_run_from_config_requires_source_before_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_global_config_dict",
            lambda: {
                "token_id_capture": {
                    "enabled": True,
                    "all_agents": True,
                    "sink": "framework.capture:Sink",
                    "rebuild_response": True,
                    "lineage_store": f"{__name__}:_StubLineageStore",
                }
            },
        )
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_bytes(
            orjson.dumps(
                {
                    "responses_create_params": {"input": []},
                    AGENT_REF_KEY_NAME: {"name": "agent"},
                }
            )
            + b"\n"
        )
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            resume_from_cache=False,
            disable_aggregation=True,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                raise AssertionError("Dispatch must not start without a TokenSource.")

        with pytest.raises(ValueError, match="rollout-collector process"):
            await Helper().run_from_config(config)

    async def test_run_from_config_does_not_close_an_installed_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Source:
            closed = False

            async def freeze(self, rollout_id):
                return TokenCaptureSnapshot(
                    rollout_id=rollout_id,
                    entries=(),
                    incomplete=False,
                    snapshot_id="snapshot",
                    version=1,
                )

            async def drop(self, rollout_id, *, snapshot_id, version):
                return True

            async def close(self):
                self.closed = True

        source = Source()
        monkeypatch.setattr(nemo_gym.rollout_collection, "installed_token_source", lambda: source)
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_global_config_dict",
            lambda: {
                "token_id_capture": {
                    "enabled": True,
                    "all_agents": True,
                    "sink": "framework.capture:Sink",
                    "rebuild_response": True,
                    "lineage_store": f"{__name__}:_StubLineageStore",
                }
            },
        )
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_bytes(
            orjson.dumps(
                {
                    "responses_create_params": {"input": []},
                    AGENT_REF_KEY_NAME: {"name": "agent"},
                }
            )
            + b"\n"
        )
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            resume_from_cache=False,
            disable_aggregation=True,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                [example] = examples
                future = Future()
                future.set_result(
                    _CompletedRollout(
                        row=example, result={"response": {"output": [], "usage": {}}}, rollout_latency_ms=None
                    )
                )
                return [future]

        with pytest.warns(UserWarning, match="capture contains no token records"):
            await Helper().run_from_config(config)

        assert source.closed is False

    async def test_run_from_config_sorted(self, tmp_path: Path, empty_global_config: MagicMock) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(10)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def _run_examples_with_metadata(
                self,
                examples: list[dict],
                *args,
                **kwargs,
            ):
                futures = []
                for example in examples:
                    future = Future()
                    # (row, result)
                    future.set_result(
                        _CompletedRollout(
                            row=example, result={"response": {"usage": {"abc usage": 1}}}, rollout_latency_ms=None
                        )
                    )
                    futures.append(future)

                # Reverse!
                futures = reversed(futures)

                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)

        expected_results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
        ]

        assert expected_results == actual_returned_results

    async def test_run_from_config_aggregate_metrics_excludes_non_persisted_rows(
        self, tmp_path: Path, empty_global_config: MagicMock
    ) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(3)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=1,
        )

        captured: dict[str, list[dict]] = {}

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def _run_examples_with_metadata(
                self,
                examples: list[dict],
                *args,
                **kwargs,
            ):
                futures = []
                for example in examples:
                    future = Future()
                    result = {
                        "response": {"usage": {"abc usage": example["x"] + 1}},
                        "case": f"case-{example['x']}",
                    }
                    if example["x"] == 1:
                        result[NG_FAILURE_CLASS_KEY] = "verify_failed"
                    elif example["x"] == 2:
                        result[NG_NO_PERSIST_KEY] = True
                    future.set_result(_CompletedRollout(row=example, result=result, rollout_latency_ms=None))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                captured["results"] = results
                captured["rows"] = rows
                metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
                metrics_fpath.write_text("[]")
                return metrics_fpath

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)

        assert [result["case"] for result in actual_returned_results] == ["case-0", "case-1", "case-2"]
        assert [result["case"] for result in captured["results"]] == ["case-0"]
        assert [row["x"] for row in captured["rows"]] == [0]

        with output_jsonl_fpath.open() as f:
            actual_written_results = [json.loads(line) for line in f]
        assert [result["case"] for result in actual_written_results] == ["case-0"]

        failures_fpath = _failures_path_for(output_jsonl_fpath)
        with failures_fpath.open() as f:
            actual_failure_results = [json.loads(line) for line in f]
        assert [result["case"] for result in actual_failure_results] == ["case-1"]
        assert actual_failure_results[0][NG_FAILURE_CLASS_KEY] == "verify_failed"

    async def test_run_from_config_aggregate_metrics_includes_cached_persisted_rows(
        self, tmp_path: Path, empty_global_config: MagicMock
    ) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        output_jsonl_fpath = tmp_path / "output.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
        )

        materialized_rows = [
            {
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
                AGENT_REF_KEY_NAME: {"name": "my agent name"},
                "x": task_index,
            }
            for task_index in (0, 1)
        ]
        config.materialized_jsonl_fpath.write_bytes(b"\n".join(orjson.dumps(row) for row in materialized_rows) + b"\n")
        cached_result = {
            TASK_INDEX_KEY_NAME: 1,
            ROLLOUT_INDEX_KEY_NAME: 0,
            AGENT_REF_KEY_NAME: {"name": "my agent name"},
            "case": "cached",
        }
        output_jsonl_fpath.write_bytes(orjson.dumps(cached_result) + b"\n")

        captured: dict[str, list[dict]] = {}

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                [example] = examples
                future = Future()
                future.set_result(_CompletedRollout(row=example, result={"case": "new"}, rollout_latency_ms=None))
                return [future]

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                captured["results"] = results
                captured["rows"] = rows
                return None

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)

        assert [result["case"] for result in actual_returned_results] == ["new", "cached"]
        assert [result["case"] for result in captured["results"]] == ["new", "cached"]
        assert [row["x"] for row in captured["rows"]] == [0, 1]

    def test_load_from_cache(self, tmp_path: Path) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        materialized_inputs_jsonl_fpath = tmp_path / "output_materialized_inputs.jsonl"

        materialized_inputs = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "input": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 1, "input": True},
        ]
        materialized_inputs_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, materialized_inputs)) + b"\n")

        outputs = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True},
        ]
        output_jsonl_fpath = tmp_path / "output.jsonl"
        output_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, outputs)) + b"\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        actual_returned_results = RolloutCollectionHelper()._load_from_cache(config)

        expected_results = (
            [
                {"_ng_task_index": 1, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 2, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 2, "_ng_rollout_index": 1, "input": True},
            ],
            [
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 0, "_ng_rollout_index": 1, "input": True},
                {"_ng_task_index": 1, "_ng_rollout_index": 1, "input": True},
            ],
            [
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True},
                {"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True},
                {"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True},
            ],
            [
                [orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True})],
                [orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True})],
                [orjson.dumps({"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True})],
            ],
        )

        assert expected_results == actual_returned_results

    async def test_call_aggregate_metrics(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test _call_aggregate_metrics with a mocked server client."""

        agg = AggregateMetrics(
            agent_metrics={"mean/reward": 0.5},
            key_metrics={"mean/reward": 0.5},
            group_level_metrics=[{"mean/reward": 1.0}, {"mean/reward": 0.0}],
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        mock_response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=mock_response)

        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        helper = RolloutCollectionHelper()

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1},
        ]
        results = [
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "response": {"usage": {"tokens": 10}},
                "ng_agent_observations": {"invocations": [{"conversation": ["large"]}]},
                "ng_model_call_capture": {"calls": [{"request": "large"}]},
                "ng_trajectory": {"model_calls": [{"request": "large"}]},
            },
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 12}}},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 8}}},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 15}}},
        ]

        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = await helper._call_aggregate_metrics(results, rows, output_fpath)

        # Verify file was written
        assert metrics_fpath is not None
        assert metrics_fpath.exists()
        written = json.loads(metrics_fpath.read_text())
        assert len(written) == 1
        assert written[0][AGENT_REF_KEY_NAME] == {"name": "my_agent"}
        assert written[0]["agent_metrics"]["mean/reward"] == 0.5
        assert written[0]["key_metrics"]["mean/reward"] == 0.5
        assert len(written[0]["group_level_metrics"]) == 2

        # Verify server_client.post was called with stripped data (usage preserved)
        call_kwargs = mock_server_client.post.call_args
        sent_request = call_kwargs.kwargs["json"]
        sent_data = (
            sent_request.verify_responses
            if isinstance(sent_request, AggregateMetricsRequest)
            else sent_request["verify_responses"]
        )
        for item in sent_data:
            assert "responses_create_params" not in item
            assert "ng_agent_observations" not in item
            assert "ng_model_call_capture" not in item
            assert "ng_trajectory" not in item
            assert "usage" in item["response"]

    async def test_call_aggregate_metrics_includes_perf_summary_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """perf_summary must survive _call_aggregate_metrics' hand-picked agent_entry dict --
        it's easy to add a field to AggregateMetrics and forget this call site only forwards an
        explicit allowlist rather than the whole model."""
        agg = AggregateMetrics(
            agent_metrics={"mean/reward": 0.5},
            key_metrics={"mean/reward": 0.5},
            group_level_metrics=[{"mean/reward": 1.0}],
            perf_summary={"mean_num_turns": 3.0, "total_latency_mean_ms": 1000.0},
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        mock_response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        helper = RolloutCollectionHelper()

        rows = [{AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}]
        results = [{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0}]

        metrics_fpath = await helper._call_aggregate_metrics(results, rows, tmp_path / "output.jsonl")

        written = json.loads(metrics_fpath.read_text())
        assert written[0]["perf_summary"] == {"mean_num_turns": 3.0, "total_latency_mean_ms": 1000.0}

    async def test_call_aggregate_metrics_omits_perf_summary_when_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        agg = AggregateMetrics(agent_metrics={"mean/reward": 0.5}, key_metrics={"mean/reward": 0.5})

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        mock_response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        helper = RolloutCollectionHelper()

        rows = [{AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}]
        results = [{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0}]

        metrics_fpath = await helper._call_aggregate_metrics(results, rows, tmp_path / "output.jsonl")

        written = json.loads(metrics_fpath.read_text())
        assert "perf_summary" not in written[0]

    async def test_call_aggregate_metrics_multiple_agents(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test _call_aggregate_metrics with multiple agents runs concurrently via as_completed."""

        agg_a = AggregateMetrics(
            agent_metrics={"mean/reward": 1.0},
            key_metrics={"mean/reward": 1.0},
            group_level_metrics=[{"mean/reward": 1.0}],
        )
        agg_b = AggregateMetrics(
            agent_metrics={"mean/reward": 0.0},
            key_metrics={"mean/reward": 0.0},
            group_level_metrics=[{"mean/reward": 0.0}],
        )

        # Return different responses per agent based on server_name
        async def mock_post(server_name, **kwargs):
            agg = agg_a if server_name == "agent_a" else agg_b
            resp = AsyncMock()
            resp.raise_for_status = MagicMock()
            resp.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
            resp.status = 200
            return resp

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(side_effect=mock_post)

        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        helper = RolloutCollectionHelper()

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
        ]
        results = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 10}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 1.0, "response": {"usage": {"tokens": 12}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, "response": {"usage": {"tokens": 8}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 15}}},
        ]

        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = await helper._call_aggregate_metrics(results, rows, output_fpath)

        written = json.loads(metrics_fpath.read_text())
        assert len(written) == 2

        # Both agents should be present (order may vary due to as_completed)
        agent_names = {entry[AGENT_REF_KEY_NAME]["name"] for entry in written}
        assert agent_names == {"agent_a", "agent_b"}

        for entry in written:
            if entry[AGENT_REF_KEY_NAME]["name"] == "agent_a":
                assert entry["agent_metrics"]["mean/reward"] == 1.0
            else:
                assert entry["agent_metrics"]["mean/reward"] == 0.0

        # Verify both agents were called
        assert mock_server_client.post.call_count == 2

    async def test_call_aggregate_metrics_empty(self, tmp_path: Path) -> None:
        """_call_aggregate_metrics returns None for empty results."""
        helper = RolloutCollectionHelper()
        output_fpath = tmp_path / "output.jsonl"
        result = await helper._call_aggregate_metrics([], [], output_fpath)
        assert result is None


class TestExpandInputGlob:
    """`_expand_input_glob` accepts a single glob, a comma-separated list of globs, or a mix.

    Mirrors the multi-pattern conventions used elsewhere in NeMo Skills
    (e.g. comma-separated `config_paths` on `ns nemo_gym_rollouts`).
    """

    def test_single_path(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        a.write_text("{}\n")
        assert _expand_input_glob(str(a)) == [str(a)]

    def test_single_glob(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"rollouts-chunk{i}.jsonl").write_text("{}\n")
        result = _expand_input_glob(str(tmp_path / "rollouts-chunk*.jsonl"))
        assert result == sorted(str(tmp_path / f"rollouts-chunk{i}.jsonl") for i in range(3))

    def test_comma_separated_paths(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text("{}\n")
        b.write_text("{}\n")
        result = _expand_input_glob(f"{a},{b}")
        assert set(result) == {str(a), str(b)}

    def test_comma_separated_globs(self, tmp_path: Path) -> None:
        for sub in ("run1", "run2"):
            (tmp_path / sub).mkdir()
            (tmp_path / sub / "rollouts.jsonl").write_text("{}\n")
            (tmp_path / sub / "extra.txt").write_text("ignore me")
        result = _expand_input_glob(f"{tmp_path / 'run1' / 'rollouts*.jsonl'},{tmp_path / 'run2' / 'rollouts*.jsonl'}")
        assert set(result) == {
            str(tmp_path / "run1" / "rollouts.jsonl"),
            str(tmp_path / "run2" / "rollouts.jsonl"),
        }

    def test_whitespace_around_commas_is_stripped(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text("{}\n")
        b.write_text("{}\n")
        result = _expand_input_glob(f"  {a}  ,  {b}  ")
        assert set(result) == {str(a), str(b)}

    def test_overlapping_patterns_dedup(self, tmp_path: Path) -> None:
        """A file matched by two patterns appears once in the output."""
        a = tmp_path / "a.jsonl"
        a.write_text("{}\n")
        result = _expand_input_glob(f"{tmp_path / '*.jsonl'},{a}")
        assert result == [str(a)]

    def test_no_matches_returns_empty(self, tmp_path: Path) -> None:
        assert _expand_input_glob(str(tmp_path / "nonexistent-*.jsonl")) == []

    def test_empty_strings_in_csv_are_dropped(self, tmp_path: Path) -> None:
        """Trailing/leading commas don't produce an empty-pattern glob that matches everything."""
        a = tmp_path / "a.jsonl"
        a.write_text("{}\n")
        result = _expand_input_glob(f",{a},,")
        assert result == [str(a)]


class TestDisableAggregationAndCallerTaskIndex:
    """Branches added for sharded rollouts: `disable_aggregation` flag and
    caller-provided `_ng_task_index`. Both must be backward-compatible with
    the existing default-on aggregation + auto-numbering behaviour.
    """

    async def test_run_from_config_disable_aggregation_skips_call(
        self, tmp_path: Path, empty_global_config: MagicMock
    ) -> None:
        """When disable_aggregation=True, _call_aggregate_metrics MUST NOT run.

        Shows up in chunked-rollouts flows where the aggregation pass is deferred
        to a single ng_aggregate_rollouts run over the union of shards.
        """
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, "x": 0}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            disable_aggregation=True,
            num_repeats=1,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for ex in examples:
                    fut = Future()
                    fut.set_result(
                        _CompletedRollout(row=ex, result={"response": {"usage": {}}}, rollout_latency_ms=None)
                    )
                    futures.append(fut)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                raise AssertionError("aggregator must not run when disable_aggregation=True")

        await Helper().run_from_config(config)

        # Rollouts file written (proves the rollout phase ran); aggregator file absent.
        assert output_jsonl_fpath.exists()
        assert not (tmp_path / "output_aggregate_metrics.json").exists()
        assert not (tmp_path / "quality_summary.json").exists()
        assert not (tmp_path / "rollout_verdicts.jsonl").exists()

    def test_preprocess_honors_caller_task_index(self, tmp_path: Path) -> None:
        """A row arriving with `_ng_task_index` pre-set is used verbatim — the
        original `row_to_task_idx` auto-numbering is bypassed. This is the seam
        an upstream slicer relies on to keep task identifiers globally-stable
        across shards.
        """
        fpath = tmp_path / "input.jsonl"
        rows = [
            # Same prompt twice with *different* caller-stamped indices — must
            # NOT be collapsed to one task by the row_str dedup path.
            {"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, TASK_INDEX_KEY_NAME: 42},
            {"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, TASK_INDEX_KEY_NAME: 99},
            # And a third row with no caller index — auto-numbering still applies.
            {"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, "diff": "row"},
        ]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="a",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        indices = [r[TASK_INDEX_KEY_NAME] for r in result]

        # Caller-provided indices preserved; the no-index row gets an auto-generated
        # one starting at 0 (the row_to_task_idx counter is independent of caller stamps).
        assert indices[:2] == [42, 99]
        assert indices[2] == 0  # auto-assigned; not 100 or 43


class TestRolloutAggregationHelper:
    """End-to-end shape of `ng_aggregate_rollouts`: glob → load → sort → aggregate."""

    async def test_run_from_config_full_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two shards. Records have globally-stamped task indices (out of order)
        # — the helper should sort by (task_index, rollout_index) before calling
        # _call_aggregate_metrics so downstream groupby is deterministic.
        shard0 = tmp_path / "rollouts-chunk0.jsonl"
        shard1 = tmp_path / "rollouts-chunk1.jsonl"
        records_shard0 = [
            {
                AGENT_REF_KEY_NAME: {"name": "a"},
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "response": {"usage": {"x": 2}},
                "reward": 1.0,
            },
            {
                AGENT_REF_KEY_NAME: {"name": "a"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "response": {"usage": {"x": 1}},
                "reward": 0.0,
            },
        ]
        records_shard1 = [
            {
                AGENT_REF_KEY_NAME: {"name": "a"},
                TASK_INDEX_KEY_NAME: 2,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "response": {"usage": {"x": 3}},
                "reward": 1.0,
            },
        ]
        shard0.write_text("\n".join(json.dumps(r) for r in records_shard0) + "\n")
        shard1.write_text("\n".join(json.dumps(r) for r in records_shard1) + "\n")

        output_fpath = tmp_path / "rollouts.jsonl"

        captured: dict[str, list] = {}

        async def fake_call(self, results, rows, output_fpath):
            captured["results"] = results
            captured["rows"] = rows
            captured["output_fpath"] = output_fpath
            # Touch a sentinel file so the helper's return value is meaningful.
            metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
            metrics_fpath.write_text("[]")
            return metrics_fpath

        monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", fake_call)

        cfg = RolloutAggregationConfig(
            input_glob=f"{shard0},{shard1}",
            output_jsonl_fpath=str(output_fpath),
            merge_shards=True,
        )
        metrics_fpath = await RolloutAggregationHelper().run_from_config(cfg)

        # 3 records total, sorted by (task_index, rollout_index): tasks 0, 1, 2.
        assert [r[TASK_INDEX_KEY_NAME] for r in captured["results"]] == [0, 1, 2]
        # rows passed twice == results (helper uses results both ways since each
        # row already carries AGENT_REF_KEY_NAME).
        assert captured["rows"] is captured["results"]
        # Merged shard concatenation honoured (merge_shards=True).
        assert output_fpath.exists()
        assert sum(1 for _ in output_fpath.open()) == 3
        # Metrics file path returned and points next to the merged JSONL.
        assert metrics_fpath == tmp_path / "rollouts_aggregate_metrics.json"
        assert metrics_fpath.exists()

    async def test_run_from_config_no_matches_raises(self, tmp_path: Path) -> None:
        cfg = RolloutAggregationConfig(
            input_glob=str(tmp_path / "nothing-matches-*.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        )
        with pytest.raises(FileNotFoundError, match="No shards matched"):
            await RolloutAggregationHelper().run_from_config(cfg)

    async def test_run_from_config_merge_shards_false_skips_concat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shard = tmp_path / "shard.jsonl"
        record = {
            AGENT_REF_KEY_NAME: {"name": "a"},
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "response": {"usage": {}},
            "reward": 0.5,
        }
        shard.write_text(json.dumps(record) + "\n")
        output_fpath = tmp_path / "rollouts.jsonl"

        async def _noop(self, results, rows, output_fpath):
            m = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
            m.write_text("[]")
            return m

        monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", _noop)
        cfg = RolloutAggregationConfig(
            input_glob=str(shard),
            output_jsonl_fpath=str(output_fpath),
            merge_shards=False,
        )
        await RolloutAggregationHelper().run_from_config(cfg)

        # merge_shards=False ⇒ no concatenated rollouts file is written, even
        # though output_jsonl_fpath is used to derive the metrics path.
        assert not output_fpath.exists()
        assert (tmp_path / "rollouts_aggregate_metrics.json").exists()

    async def test_health_failure_does_not_fail_aggregation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        shard = tmp_path / "shard.jsonl"
        shard.write_text(
            json.dumps(
                {
                    AGENT_REF_KEY_NAME: {"name": "a"},
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    "response": {"usage": {}},
                    "reward": 0.5,
                }
            )
            + "\n"
        )
        output_fpath = tmp_path / "rollouts.jsonl"

        async def fake_call(self, results, rows, output_fpath):
            metrics_path = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
            metrics_path.write_text("[]")
            return metrics_path

        caller_thread = get_ident()
        health_thread = None

        def broken_health_check(*args, **kwargs):
            nonlocal health_thread
            health_thread = get_ident()
            raise RuntimeError("health failed")

        monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", fake_call)
        monkeypatch.setattr("nemo_gym.rollout_health.run_health_checks", broken_health_check)
        config = RolloutAggregationConfig(
            input_glob=str(shard),
            output_jsonl_fpath=str(output_fpath),
            merge_shards=True,
        )

        metrics_path = await RolloutAggregationHelper().run_from_config(config)

        assert metrics_path.exists()
        assert output_fpath.exists()
        assert health_thread is not None
        assert health_thread != caller_thread
        assert "Rollout health checks failed after aggregation" in caplog.text


class TestTokenCaptureRetention:
    """Test retirement after handoff and stale-record clearing.

    ``TokenCaptureStore.append`` uses append mode.
    Rollout ids are deterministic.
    Clearing prevents a rerun from merging different attempts.
    Retirement prevents unbounded growth after durable handoff.
    """

    @staticmethod
    def _entry(rollout_id: str, mcid: str) -> TokenEntry:
        return TokenEntry(
            rollout_id=rollout_id,
            model_call_id=mcid,
            prompt_token_ids=[1, 2, 3],
            generation_token_ids=[4, 5],
            generation_log_probs=[-0.1, -0.2],
        )

    async def test_clear_removes_stale_records_before_dispatch(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        store.append(self._entry("0-0", "old"))
        await store.mark_incomplete("0-0", "old")
        rows = [{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}]

        clear_token_captures_for_rollouts(rows, [tmp_path])

        assert store.read_entries("0-0") == []
        assert not store.is_incomplete("0-0")

    def test_clear_is_a_noop_without_capture_dirs(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        store.append(self._entry("0-0", "keep"))
        clear_token_captures_for_rollouts([{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}], [])
        assert len(store.read_entries("0-0")) == 1

    def test_clear_skips_rows_without_a_derivable_rollout_id(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        store.append(self._entry("0-0", "keep"))
        clear_token_captures_for_rollouts([{"unrelated": True}], [tmp_path])
        assert len(store.read_entries("0-0")) == 1


class TestFinalizeRolloutTokenCapture:
    """Test the per-record token-capture finalizer.

    The finalizer accepts a record and a ``TokenSource``.
    A framework can provide a source without using Gym configuration.
    """

    @staticmethod
    def _record(output: list | None = None) -> dict:
        return {
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "reward": 1.0,
            "response": {"model": "m", "output": output if output is not None else []},
        }

    @staticmethod
    def _capture(store: TokenCaptureStore) -> None:
        entry = TokenEntry(
            rollout_id="0-0",
            model_call_id="c1",
            prompt_token_ids=[1, 2, 3],
            generation_token_ids=[4, 5],
            generation_log_probs=[-0.1, -0.2],
            output_items=[{"type": "message", "role": "assistant", "content": []}],
            token_item_index=0,
        )
        stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
        store.append(entry)

    async def test_rebuilds_a_rollout_that_has_no_token_ids(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        self._capture(store)
        result = self._record()

        built = await finalize_rollout_token_capture(result, store)

        [item] = result["response"]["output"]
        assert item["generation_token_ids"] == [4, 5]
        assert result["reward"] == 1.0  # Preserve harness and verifier output.
        assert result[TOKEN_CAPTURE_KEY]["delivered_fraction"] == 1.0
        assert built is not None and built["rebuilt_response"] is not None
        assert len(store.read_entries("0-0")) == 1  # Retain evidence until durable handoff.
        assert await retire_rollout_token_capture("0-0", store, built) is True
        assert store.read_entries("0-0") == []

    async def test_retirement_cannot_delete_a_newer_rollout_attempt(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        self._capture(store)
        built = await finalize_rollout_token_capture(self._record(), store)

        store.delete("0-0")
        replacement = TokenEntry(
            rollout_id="0-0",
            model_call_id="new",
            prompt_token_ids=[1],
            generation_token_ids=[2],
            generation_log_probs=[-0.1],
        )
        store.append(replacement)

        assert await retire_rollout_token_capture("0-0", store, built) is False
        assert [entry.model_call_id for entry in store.read_entries("0-0")] == ["new"]

    async def test_a_rollout_that_already_has_token_ids_is_left_alone(self, tmp_path: Path) -> None:
        """Keep the token ids sampled by a native agent.

        A reconstruction may differ from the sampled ids.
        Overwriting them would silently train on that difference.
        """
        store = TokenCaptureStore(tmp_path)
        self._capture(store)
        native = [{"type": "message", "role": "assistant", "generation_token_ids": [9, 9], "content": []}]
        result = self._record(output=native)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Existing ids are not an error.
            built = await finalize_rollout_token_capture(result, store)

        assert result["response"]["output"] == native
        assert TOKEN_CAPTURE_KEY not in result
        assert capture_build_can_retire(built)
        assert len(store.read_entries("0-0")) == 1
        assert await retire_rollout_token_capture("0-0", store, built) is True
        assert store.read_entries("0-0") == []

    async def test_native_and_external_rollouts_are_handled_in_one_batch(self, tmp_path: Path) -> None:
        """Finalize native and external rollouts through the same call."""
        store = TokenCaptureStore(tmp_path)
        self._capture(store)
        native = self._record(
            output=[{"type": "message", "role": "assistant", "generation_token_ids": [7], "content": []}]
        )
        external = self._record()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            native_build = await finalize_rollout_token_capture(native, store)
        built = await finalize_rollout_token_capture(external, store)

        assert native["response"]["output"][0]["generation_token_ids"] == [7]
        assert external["response"]["output"][0]["generation_token_ids"] == [4, 5]
        assert capture_build_can_retire(native_build)
        assert built is not None

    async def test_a_second_call_is_a_no_op(self, tmp_path: Path) -> None:
        """Leave a finalized rollout unchanged on a second call."""
        store = TokenCaptureStore(tmp_path)
        self._capture(store)
        result = self._record()

        await finalize_rollout_token_capture(result, store)
        rebuilt = deepcopy(result["response"]["output"])
        second = await finalize_rollout_token_capture(result, store)
        assert second is not None
        assert second.get("rebuilt_response") is None
        assert second.get("_capture_snapshot", {}).get("snapshot_id")
        assert result["response"]["output"] == rebuilt

    async def test_no_source_means_this_caller_is_not_capturing(self, tmp_path: Path) -> None:
        result = self._record()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert await finalize_rollout_token_capture(result, None) is None
        assert result["response"]["output"] == []

    async def test_a_masked_rollout_is_flagged_at_the_top_of_the_record(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        self._capture(store)
        # A call that failed to capture leaves a chain that looks contiguous but is missing a turn.
        await store.mark_incomplete("0-0", "c2")
        result = self._record()

        with pytest.warns(UserWarning, match="marked for masking"):
            await finalize_rollout_token_capture(result, store)

        # Keep the masking decision in one top-level field.
        assert result[MASK_SAMPLE_KEY] is True
        assert MASK_SAMPLE_KEY not in result[TOKEN_CAPTURE_KEY]
        assert result[TOKEN_CAPTURE_KEY]["capture_incomplete"] is True

    async def test_a_healthy_rollout_is_not_flagged(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        self._capture(store)
        result = self._record()

        await finalize_rollout_token_capture(result, store)

        # Omit the field so presence-based consumers keep healthy samples.
        assert MASK_SAMPLE_KEY not in result

    async def test_a_failed_build_keeps_its_records_and_reports_why(self, tmp_path: Path) -> None:
        store = TokenCaptureStore(tmp_path)
        malformed = TokenEntry(
            rollout_id="0-0",
            model_call_id="c1",
            prompt_token_ids=[1, 2],
            generation_token_ids=[4, 5],
            generation_log_probs=[-0.1, -0.2],
            output_items=[{"type": "message", "role": "assistant", "content": []}],
            token_item_index=0,
        )
        malformed.generation_log_probs = [-0.1]
        store.append(malformed)
        result = self._record()

        with pytest.warns(UserWarning, match="marked for masking"):
            await finalize_rollout_token_capture(result, store)

        assert result[MASK_SAMPLE_KEY] is True
        assert "ValidationError" in result[TOKEN_CAPTURE_KEY]["error"]
        # Retain failed-build records as diagnostic evidence.
        assert store.path_for("0-0").stat().st_size > 0

    async def test_a_rollout_with_no_capture_key_is_masked(self, tmp_path: Path) -> None:
        result = self._record()
        del result[TASK_INDEX_KEY_NAME]
        del result[ROLLOUT_INDEX_KEY_NAME]

        with pytest.warns(UserWarning, match="carries no id"):
            built = await finalize_rollout_token_capture(result, TokenCaptureStore(tmp_path))

        # Mask the rollout before it reaches the trainer without ids.
        assert result[MASK_SAMPLE_KEY] is True
        assert result[TOKEN_CAPTURE_KEY]["error"] == "no capture key"
        assert built is not None and built["rebuilt_response"] is None

    async def test_nothing_recorded_for_a_rollout_that_needs_ids_is_masked(self, tmp_path: Path) -> None:
        result = self._record()

        with pytest.warns(UserWarning, match="marked for masking"):
            built = await finalize_rollout_token_capture(result, TokenCaptureStore(tmp_path))

        assert result[MASK_SAMPLE_KEY] is True
        assert result[TOKEN_CAPTURE_KEY]["error"] == "capture contains no token records"
        # Report the rollout as both masked and unbuilt.
        assert built is not None and built[MASK_SAMPLE_KEY] is True and built["rebuilt_response"] is None

    async def test_a_source_that_raises_loses_one_rollout_not_the_batch(self, tmp_path: Path) -> None:
        """Keep transport failures scoped to their rollout."""

        class _Failing:
            async def freeze(self, rollout_id: str):
                raise ConnectionError("data plane unreachable")

            async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
                return False

            async def close(self) -> None: ...

        result = self._record()

        with pytest.warns(UserWarning, match="marked for masking"):
            built = await finalize_rollout_token_capture(result, _Failing())

        assert result[MASK_SAMPLE_KEY] is True
        assert "ConnectionError" in result[TOKEN_CAPTURE_KEY]["error"]
        assert built is not None and built["rebuilt_response"] is None


class TestRolloutCarriesTokenIds:
    def test_true_when_any_item_carries_generated_ids(self) -> None:
        result = {"response": {"output": [{"type": "message"}, {"generation_token_ids": [1]}]}}
        assert rollout_carries_token_ids(result) is True

    @pytest.mark.parametrize(
        "response",
        [
            {"output": []},
            {"output": [{"type": "message", "content": []}]},
            {"output": [{"generation_token_ids": []}]},  # An empty list contains no sampled ids.
            {},
            None,
        ],
    )
    def test_false_without_them(self, response) -> None:
        assert rollout_carries_token_ids({"response": response}) is False


class TestE2EInputJsonlFpathRejected:
    def test_e2e_config_rejects_input_jsonl_fpath(self) -> None:
        with pytest.raises(ConfigError, match=r"not supported when serving end-to-end"):
            E2ERolloutCollectionConfig.model_validate(
                {
                    "output_jsonl_fpath": "out.jsonl",
                    "split": "train",
                    "input_jsonl_fpath": "my_data.jsonl",
                }
            )

    def test_e2e_config_rejects_input_jsonl_fpath_from_dictconfig(self) -> None:
        # The CLI passes an OmegaConf DictConfig (a Mapping, not a dict). An isinstance(dict)
        # check silently let input_jsonl_fpath through on the real path — pin the Mapping match.
        with pytest.raises(ConfigError, match=r"not supported when serving end-to-end"):
            E2ERolloutCollectionConfig.model_validate(
                DictConfig(
                    {
                        "output_jsonl_fpath": "out.jsonl",
                        "split": "train",
                        "input_jsonl_fpath": "my_data.jsonl",
                    }
                )
            )

    def test_e2e_config_accepts_without_input_jsonl_fpath(self) -> None:
        config = E2ERolloutCollectionConfig.model_validate({"output_jsonl_fpath": "out.jsonl", "split": "train"})
        assert config.split == "train"

    def test_no_serve_config_still_accepts_input_jsonl_fpath(self) -> None:
        config = RolloutCollectionConfig.model_validate(
            {"output_jsonl_fpath": "out.jsonl", "input_jsonl_fpath": "my_data.jsonl"}
        )
        assert config.input_jsonl_fpath == "my_data.jsonl"


class TestE2EExampleSplitRejected:
    @pytest.mark.parametrize("wrap", [dict, DictConfig])
    def test_example_split_gets_actionable_error_not_literal_error(self, wrap) -> None:
        with pytest.raises(ConfigError, match=r"--no-serve --agent <agent> --input"):
            E2ERolloutCollectionConfig.model_validate(wrap({"output_jsonl_fpath": "out.jsonl", "split": "example"}))

    def test_other_invalid_splits_still_fail_literal_validation(self) -> None:
        with pytest.raises(ValidationError, match=r"split"):
            E2ERolloutCollectionConfig.model_validate({"output_jsonl_fpath": "out.jsonl", "split": "test"})


class TestAgentMapRouting:
    """Pins the agent_map / agent_name routing contract (see dataset-decoupling RFC).

    These exist to fail loudly if the precedence semantics are ever changed silently
    (the #761 failure mode, where override quietly became backfill).
    """

    def _write_rows(self, tmp_path, rows):
        fpath = tmp_path / "input.jsonl"
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return fpath

    def _config(self, tmp_path, fpath, **kwargs):
        return RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath), output_jsonl_fpath=str(tmp_path / "out.jsonl"), **kwargs
        )

    def _rcp_row(self, agent=None, content="q"):
        row = {"responses_create_params": {"input": [{"role": "user", "content": content}]}}
        if agent is not None:
            row["agent_ref"] = {"name": agent}
        return row

    def test_agent_name_overrides_existing_agent_ref(self, tmp_path) -> None:
        """agent_name re-routes ALL rows (restored #568 semantics), warning about the override."""
        fpath = self._write_rows(tmp_path, [self._rcp_row("old_agent"), self._rcp_row()])
        config = self._config(tmp_path, fpath, agent_name="new_agent")
        with pytest.warns(UserWarning, match="overrode agent_ref"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["new_agent", "new_agent"]

    def test_agent_name_is_sugar_for_agent_map_default(self, tmp_path) -> None:
        config = self._config(tmp_path, self._write_rows(tmp_path, [self._rcp_row()]), agent_name="a")
        assert config.agent_map == {"_default": "a"}

    def test_agent_name_conflicting_with_agent_map_default_raises(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row()])
        with pytest.raises(ValueError, match="conflicts with agent_map._default"):
            self._config(tmp_path, fpath, agent_name="a", agent_map={"_default": "b"})

    def test_agent_map_specific_beats_default(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row("x"), self._rcp_row("y"), self._rcp_row()])
        config = self._config(tmp_path, fpath, agent_map={"x": "mapped_x", "_default": "fallback"})
        with pytest.warns(UserWarning, match="overrode agent_ref"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["mapped_x", "fallback", "fallback"]

    def test_agent_map_without_default_leaves_unmapped_rows_alone(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row("x"), self._rcp_row("y")])
        config = self._config(tmp_path, fpath, agent_map={"x": "mapped_x"})
        with pytest.warns(UserWarning, match="overrode agent_ref"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["mapped_x", "y"]

    def test_row_agent_ref_wins_when_no_map(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row("x")])
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, self._config(tmp_path, fpath))
        assert rows[0]["agent_ref"]["name"] == "x"

    def test_missing_agent_still_hard_errors(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row()])
        config = self._config(tmp_path, fpath, agent_map={"x": "y"})
        with pytest.raises(ValueError, match="No agent specified"):
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)

    def test_identity_mapping_does_not_warn(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row("a")])
        config = self._config(tmp_path, fpath, agent_name="a")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows[0]["agent_ref"]["name"] == "a"


class TestValidateAgentNames:
    def _rows(self, *names):
        return [{"agent_ref": {"name": n}} for n in names]

    def _cfg(self, **entries):
        return OmegaConf.create(
            {name: {"responses_api_agents": {"impl": {}}} for name in entries.get("agents", [])}
            | entries.get("extra", {})
        )

    def test_all_known_passes(self) -> None:
        cfg = self._cfg(agents=["agent_a", "agent_b"])
        RolloutCollectionHelper._validate_agent_names(self._rows("agent_a", "agent_b"), cfg)

    def test_unknown_agent_raises_with_suggestion(self) -> None:
        cfg = self._cfg(agents=["math_with_judge_simple_agent"])
        with pytest.raises(ValueError, match="did you mean 'math_with_judge_simple_agent'"):
            RolloutCollectionHelper._validate_agent_names(self._rows("math_with_judge_simple_agnet"), cfg)

    def test_unknown_agent_without_close_match_raises(self) -> None:
        cfg = self._cfg(agents=["a"])
        with pytest.raises(ValueError, match="not present in the running config"):
            RolloutCollectionHelper._validate_agent_names(self._rows("zzz_completely_unrelated"), cfg)

    def test_non_agent_instance_raises(self) -> None:
        """Routing to an existing but non-agent instance (e.g. agent_map to an RS name) must fail
        pre-dispatch: /run only exists on agent servers."""
        cfg = self._cfg(agents=["math_agent"], extra={"math_rs": {"resources_servers": {"impl": {}}}})
        with pytest.raises(ValueError, match="exists but is not an agent instance"):
            RolloutCollectionHelper._validate_agent_names(self._rows("math_rs"), cfg)


# A merged config shaped like real ones: one RS instance, agents pointing at RSes via the
# resources_server.name edge, plus a self-contained agent that declares no RS.
_RESOLVER_CONFIG = {
    "math_rs": {"resources_servers": {"math_rs_impl": {"entrypoint": "app.py"}}},
    "math_agent": {
        "responses_api_agents": {
            "simple_agent": {"resources_server": {"type": "resources_servers", "name": "math_rs"}}
        }
    },
    "tau2_agent": {"responses_api_agents": {"tau2": {"entrypoint": "app.py"}}},
    "shared_rs": {"resources_servers": {"impl": {}}},
    "shared_agent_a": {"responses_api_agents": {"a": {"resources_server": {"name": "shared_rs"}}}},
    "shared_agent_b": {"responses_api_agents": {"b": {"resources_server": {"name": "shared_rs"}}}},
    "orphan_rs": {"resources_servers": {"impl": {}}},
    # Dataset-level `agent:` pins (the escape hatch for ambiguous configs).
    "pinned_rs": {"resources_servers": {"impl": {"datasets": [{"agent": "pinned_agent_b"}]}}},
    "pinned_agent_a": {"responses_api_agents": {"a": {"resources_server": {"name": "pinned_rs"}}}},
    "pinned_agent_b": {"responses_api_agents": {"b": {"resources_server": {"name": "pinned_rs"}}}},
    "mispinned_rs": {"resources_servers": {"impl": {"datasets": [{"agent": "math_agent"}]}}},
    "conflict_rs": {
        "resources_servers": {"impl": {"datasets": [{"agent": "shared_agent_a"}, {"agent": "shared_agent_b"}]}}
    },
    "mispinned_agent": {"responses_api_agents": {"a": {"datasets": [{"agent": "math_agent"}]}}},
}


class TestResolveTaskSources:
    """Pins the task_source -> agent resolution contract (dataset-decoupling RFC)."""

    def _resolve(self, rows):
        RolloutCollectionHelper.resolve_task_sources(rows, OmegaConf.create(_RESOLVER_CONFIG))
        return rows

    def test_rs_task_source_inverts_to_unique_agent(self) -> None:
        rows = [{"task_source": "math_rs"}]
        assert self._resolve(rows)[0]["agent_ref"] == {"name": "math_agent"}

    def test_agent_task_source_routes_directly(self) -> None:
        """Self-contained environments: the declaring instance IS the agent."""
        rows = [{"task_source": "tau2_agent"}]
        assert self._resolve(rows)[0]["agent_ref"] == {"name": "tau2_agent"}

    def test_existing_agent_ref_wins_over_task_source(self) -> None:
        rows = [{"task_source": "math_rs", "agent_ref": {"name": "tau2_agent"}}]
        assert self._resolve(rows)[0]["agent_ref"] == {"name": "tau2_agent"}

    def test_no_task_source_rows_is_noop(self) -> None:
        rows = [{"agent_ref": {"name": "math_agent"}}]
        assert self._resolve(rows) == [{"agent_ref": {"name": "math_agent"}}]

    def test_unknown_task_source_raises_with_suggestion(self) -> None:
        with pytest.raises(ValueError, match="did you mean 'math_rs'"):
            self._resolve([{"task_source": "math_rss"}])

    def test_ambiguous_rs_raises_naming_agent_map(self) -> None:
        with pytest.raises(ValueError, match=r"2 agents reference this resources server.*agent_map"):
            self._resolve([{"task_source": "shared_rs"}])

    def test_rs_with_no_agent_raises(self) -> None:
        with pytest.raises(ValueError, match="no agent in the running config references"):
            self._resolve([{"task_source": "orphan_rs"}])

    def test_agent_pin_disambiguates_shared_rs(self) -> None:
        """The dataset-level `agent:` pin reaches dispatch: an RS referenced by two agents routes
        to the pinned one instead of erroring as ambiguous."""
        rows = [{"task_source": "pinned_rs"}]
        assert self._resolve(rows)[0]["agent_ref"] == {"name": "pinned_agent_b"}

    def test_agent_pin_not_referencing_the_rs_raises(self) -> None:
        """A pin naming an agent wired to a different RS must not silently re-route."""
        with pytest.raises(ValueError, match="no agent of that name references resources server 'mispinned_rs'"):
            self._resolve([{"task_source": "mispinned_rs"}])

    def test_conflicting_agent_pins_raise_naming_agent_map(self) -> None:
        with pytest.raises(ValueError, match=r"conflicting agents.*agent_map"):
            self._resolve([{"task_source": "conflict_rs"}])

    def test_agent_pin_on_agent_declared_dataset_must_name_the_declarer(self) -> None:
        with pytest.raises(ValueError, match="the pin would silently not apply"):
            self._resolve([{"task_source": "mispinned_agent"}])

    def test_task_source_survives_resolution(self) -> None:
        """The stamp stays on the row (provenance); only agent_ref is added."""
        rows = self._resolve([{"task_source": "math_rs"}])
        assert rows[0]["task_source"] == "math_rs"

    def test_legacy_agent_ref_rows_warn_deprecation(self) -> None:
        """Rows routed purely by their baked-in agent_ref (no task_source) are the legacy
        path, slated for removal after the deprecation cycle; each run warns once with a count."""
        rows = [{"agent_ref": {"name": "math_agent"}}, {"agent_ref": {"name": "math_agent"}}]
        with pytest.warns(DeprecationWarning, match="2 rows routed via their baked-in agent_ref"):
            self._resolve(rows)
        assert all(r["agent_ref"] == {"name": "math_agent"} for r in rows)

    def test_task_source_rows_do_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self._resolve([{"task_source": "math_rs"}])


class TestFanOut:
    def _write_rows(self, tmp_path, rows):
        fpath = tmp_path / "input.jsonl"
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return fpath

    def _rcp_row(self, **extra):
        return {"responses_create_params": {"input": [{"role": "user", "content": "q"}]}, **extra}

    def test_fan_out_by_task_source_emits_one_copy_per_agent(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="shared_rs")])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            fan_out={"shared_rs": ["shared_agent_a", "shared_agent_b"]},
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["shared_agent_a", "shared_agent_b"]
        assert [r[ROLLOUT_INDEX_KEY_NAME] for r in rows] == [0, 1]
        assert len({r[TASK_INDEX_KEY_NAME] for r in rows}) == 1

    def test_fan_out_by_agent_ref_name(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(agent_ref={"name": "agent_a"})])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            fan_out={"agent_a": ["agent_x", "agent_y"]},
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["agent_x", "agent_y"]

    def test_fan_out_composes_with_per_agent_num_repeats(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="shared_rs")])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            fan_out={"shared_rs": ["shared_agent_a", "shared_agent_b"]},
            num_repeats={"shared_agent_a": 2, "shared_agent_b": 1},
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["shared_agent_a", "shared_agent_a", "shared_agent_b"]
        assert [r[ROLLOUT_INDEX_KEY_NAME] for r in rows] == [0, 1, 2]

    async def test_fanned_copies_dispatch_to_distinct_agents(self, tmp_path, monkeypatch) -> None:
        """End-to-end through run_examples: each fan-out copy is POSTed to its own agent server."""
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="shared_rs")])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            fan_out={"shared_rs": ["shared_agent_a", "shared_agent_b"]},
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        posted = []

        async def fake_post(server_name, url_path, **kwargs):
            posted.append(server_name)
            response = MagicMock()
            response.ok = True
            response.read = AsyncMock(return_value=b"{}")
            return response

        mock_client = MagicMock()
        mock_client.post = fake_post
        mock_client.global_config_dict = OmegaConf.create(
            {name: {"responses_api_agents": {"impl": {}}} for name in ("shared_agent_a", "shared_agent_b")}
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "setup_server_client_utils", lambda *a, **k: mock_client)
        for fut in RolloutCollectionHelper().run_examples(rows):
            await fut
        assert sorted(posted) == ["shared_agent_a", "shared_agent_b"]

    def test_fan_out_keys_match_data_side_name_and_win_over_agent_map(self, tmp_path) -> None:
        """fan_out keys match the name the DATA carries (pre-override); its targets are final,
        so a competing agent_map rewrite does not leak into fanned copies."""
        fpath = self._write_rows(tmp_path, [self._rcp_row(agent_ref={"name": "agent_a"})])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            agent_map={"agent_a": "agent_z"},
            fan_out={"agent_a": ["agent_x", "agent_y"]},
        )
        with pytest.warns(UserWarning, match="overrode agent_ref"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["agent_x", "agent_y"]

    def test_unmatched_rows_pass_through_fan_out(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(agent_ref={"name": "other"})])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            fan_out={"shared_rs": ["shared_agent_a"]},
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert [r["agent_ref"]["name"] for r in rows] == ["other"]


class TestTaskSourcePreprocess:
    def _write_rows(self, tmp_path, rows):
        fpath = tmp_path / "input.jsonl"
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return fpath

    def _rcp_row(self, **extra):
        return {"responses_create_params": {"input": [{"role": "user", "content": "q"}]}, **extra}

    def test_task_source_row_defers_resolution(self, tmp_path) -> None:
        """Preprocess leaves task_source rows without agent_ref; resolution happens at dispatch prep."""
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="math_rs")])
        config = RolloutCollectionConfig(input_jsonl_fpath=str(fpath), output_jsonl_fpath=str(tmp_path / "o.jsonl"))
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert "agent_ref" not in rows[0]
        assert rows[0]["task_source"] == "math_rs"

    def test_agent_map_keyed_by_task_source(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="math_rs")])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "o.jsonl"),
            agent_map={"math_rs": "some_agent"},
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows[0]["agent_ref"] == {"name": "some_agent"}

    def test_agent_default_covers_task_source_rows(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="math_rs")])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath), output_jsonl_fpath=str(tmp_path / "o.jsonl"), agent_name="z"
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows[0]["agent_ref"] == {"name": "z"}

    def test_agent_map_task_source_key_matches_dual_stamped_row(self, tmp_path) -> None:
        """Derived artifacts carry BOTH task_source and a resolved agent_ref; a map entry
        keyed by either must re-route them (agent-name entry wins over task_source entry)."""
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="math_rs", agent_ref={"name": "math_agent"})])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "o.jsonl"),
            agent_map={"math_rs": "swe_agent"},
        )
        with pytest.warns(UserWarning, match="overrode agent_ref"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows[0]["agent_ref"] == {"name": "swe_agent"}

    def test_agent_map_agent_key_beats_task_source_key(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="math_rs", agent_ref={"name": "math_agent"})])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "o.jsonl"),
            agent_map={"math_agent": "by_agent", "math_rs": "by_source"},
        )
        with pytest.warns(UserWarning, match="overrode agent_ref"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows[0]["agent_ref"] == {"name": "by_agent"}

    def test_num_repeats_keyed_by_task_source(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._rcp_row(task_source="math_rs")])
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "o.jsonl"),
            num_repeats={"math_rs": 3},
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 3

    async def test_run_from_config_resolves_task_source_before_materialized_write(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """task_source-only rows must be resolved to an agent BEFORE the materialized-inputs
        file is written: custom drivers (e.g. gdpval's orchestrator) read agent_ref from it."""
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", lambda: {})

        source_row = {"responses_create_params": {"input": []}, "task_source": "math_rs"}
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_bytes(orjson.dumps(source_row) + b"\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            disable_aggregation=True,
        )

        mock_client = MagicMock()
        mock_client.global_config_dict = OmegaConf.create(
            {
                "math_rs": {"resources_servers": {"impl": {}}},
                "math_agent": {"responses_api_agents": {"impl": {"resources_server": {"name": "math_rs"}}}},
            }
        )

        class Helper(RolloutCollectionHelper):
            def setup_server_client(self, head_server_config=None):
                return mock_client

            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                future = Future()
                future.set_result(_CompletedRollout(row=examples[0], result={"response": {}}, rollout_latency_ms=None))
                return [future]

        await Helper().run_from_config(config)

        [materialized] = [orjson.loads(line) for line in config.materialized_jsonl_fpath.read_bytes().splitlines()]
        assert materialized["agent_ref"] == {"name": "math_agent"}
        assert materialized["task_source"] == "math_rs"


class TestFanOutValidation:
    """fan_out misconfiguration must fail loudly at config time, not drop rows silently."""

    def _config(self, tmp_path, **kwargs):
        return RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "in.jsonl"), output_jsonl_fpath=str(tmp_path / "out.jsonl"), **kwargs
        )

    def test_empty_target_list_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="empty list"):
            self._config(tmp_path, fan_out={"math": []})

    def test_duplicate_targets_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="more than once.*agent_a"):
            self._config(tmp_path, fan_out={"math": ["agent_a", "agent_a", "agent_b"]})


class TestNumRepeatsKeyPrecedence:
    """num_repeats keys match the dispatched agent OR the row's original routing key; the
    dispatched agent wins on conflict. Pins the documented agent_map+num_repeats combination."""

    def _write_rows(self, tmp_path, rows):
        fpath = tmp_path / "input.jsonl"
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return fpath

    def _config(self, tmp_path, fpath, **kwargs):
        return RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath), output_jsonl_fpath=str(tmp_path / "out.jsonl"), **kwargs
        )

    def _ts_row(self, task_source="math"):
        return {"responses_create_params": {"input": [{"role": "user", "content": "q"}]}, "task_source": task_source}

    def test_agent_map_composes_with_source_keyed_num_repeats(self, tmp_path) -> None:
        """agent_map={math: math_agent} + num_repeats={math: 3}: rows route to math_agent AND
        repeat 3 times via their original routing key."""
        fpath = self._write_rows(tmp_path, [self._ts_row("math")])
        config = self._config(tmp_path, fpath, agent_map={"math": "math_agent"}, num_repeats={"math": 3})
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 3
        assert all(r["agent_ref"]["name"] == "math_agent" for r in rows)

    def test_target_keyed_num_repeats_still_matches(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._ts_row("math")])
        config = self._config(tmp_path, fpath, agent_map={"math": "math_agent"}, num_repeats={"math_agent": 2})
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 2

    def test_dispatched_agent_wins_over_routing_key(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._ts_row("math")])
        config = self._config(
            tmp_path, fpath, agent_map={"math": "math_agent"}, num_repeats={"math": 5, "math_agent": 2}
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 2

    def test_fan_out_composes_with_source_keyed_num_repeats(self, tmp_path) -> None:
        """fan_out={math: [a, b]} + num_repeats={math: 2}: each fanned copy repeats 2 times."""
        fpath = self._write_rows(tmp_path, [self._ts_row("math")])
        config = self._config(tmp_path, fpath, fan_out={"math": ["agent_a", "agent_b"]}, num_repeats={"math": 2})
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 4
        by_agent = Counter(r["agent_ref"]["name"] for r in rows)
        assert by_agent == {"agent_a": 2, "agent_b": 2}

    def test_source_keyed_entry_does_not_trigger_typo_warning(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._ts_row("math")])
        config = self._config(tmp_path, fpath, agent_map={"math": "math_agent"}, num_repeats={"math": 3})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)

    def test_missing_key_error_names_both_candidates(self, tmp_path) -> None:
        fpath = self._write_rows(tmp_path, [self._ts_row("math")])
        config = self._config(tmp_path, fpath, agent_map={"math": "math_agent"}, num_repeats={"other": 1})
        with pytest.raises(ValueError, match="math_agent / math"):
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)


class TestPreprocessExamples:
    """Public preprocessing entry point for direct run_examples callers (e.g. NeMo RL): applies
    agent_map/fan_out/num_repeats to caller-held rows without touching the filesystem."""

    def _ts_row(self, task_source="math"):
        return {"responses_create_params": {"input": [{"role": "user", "content": "q"}]}, "task_source": task_source}

    def test_applies_all_knobs(self) -> None:
        examples = [self._ts_row("math"), self._ts_row("other")]
        rows = RolloutCollectionHelper().preprocess_examples(
            examples,
            agent_map={"other": "other_agent"},
            fan_out={"math": ["agent_a", "agent_b"]},
            num_repeats={"math": 2, "_default": 1},
        )
        by_agent = Counter(r["agent_ref"]["name"] for r in rows)
        assert by_agent == {"agent_a": 2, "agent_b": 2, "other_agent": 1}
        # Rollout indexes enumerate copies within each task.
        assert sorted(r["_ng_rollout_index"] for r in rows if r["_ng_task_index"] == 0) == [0, 1, 2, 3]

    def test_does_not_mutate_inputs(self) -> None:
        examples = [self._ts_row("math")]
        snapshot = json.dumps(examples, sort_keys=True)
        RolloutCollectionHelper().preprocess_examples(examples, num_repeats=3)
        assert json.dumps(examples, sort_keys=True) == snapshot

    def test_resolves_task_sources_when_config_given(self) -> None:
        cfg = OmegaConf.create(_RESOLVER_CONFIG)
        rows = RolloutCollectionHelper().preprocess_examples(
            [self._ts_row("math_rs")], global_config_dict=cfg, num_repeats=2
        )
        assert len(rows) == 2
        assert all(r["agent_ref"]["name"] == "math_agent" for r in rows)

    def test_validates_knobs_like_the_cli(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            RolloutCollectionHelper().preprocess_examples([self._ts_row()], fan_out={"math": []})
