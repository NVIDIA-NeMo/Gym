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
import gc
import json
import pickle
import warnings
import weakref
from asyncio import Future
from collections import Counter, defaultdict
from contextlib import nullcontext
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

import nemo_gym.batch_status
import nemo_gym.rollout_collection
import nemo_gym.token_id_capture.delivery
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.batch_status import observe_materialized_rows
from nemo_gym.config_types import AmbiguousEnvironmentServerError, ConfigError, ConfigPathNotFoundError
from nemo_gym.failure_kinds import CANCELLED
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.path_utils import materialized_path_for
from nemo_gym.reward_profile import compute_aggregate_metrics, compute_subset_metrics
from nemo_gym.rollout_collection import (
    _DEFAULT_MAX_ROLLOUT_ATTEMPTS,
    AGENT_REQUEST_FAILED_FAILURE_CLASS,
    AGENT_RUN_ERROR_FAILURE_CLASS,
    AGGREGATION_ERROR_KEY,
    ENVIRONMENT_SERVER_FAILURE_CLASS,
    NG_DISPATCH_DRAINED_KEY,
    NG_ENVIRONMENT_SERVER_KEY,
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_PERF_KEY,
    NG_TERMINAL_KEY,
    NG_TRAJECTORY_KEY,
    DispatchLatencyTracker,
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
    _drop_truncated_tail,
    _expand_input_glob,
    _failure_rows_counted_as_zero,
    _failures_path_for,
    _get_max_rollout_attempts,
    _masking_step_metrics,
    _missing_rollout_rows_counted_as_zero,
    _rollout_for_export,
    _rollout_order_key,
    _rollout_request_debug_summary,
    _round_robin_by_agent,
    get_max_rollout_attempts,
    is_terminal_failure,
    loads_jsonl_line,
    migrate_invalid_judge_main_rows,
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


def _environment_server_config() -> DictConfig:
    return OmegaConf.create(
        {
            "swe": {
                "resources_servers": {
                    "swebench_pro": {
                        "allowed_agents": ["hermes_agent"],
                    }
                }
            },
            "hermes": {
                "responses_api_agents": {
                    "hermes_agent": {
                        "resources_server": {"type": "resources_servers", "name": "swe"},
                    }
                }
            },
            "environment": {
                "environment_servers": {
                    "single_agent": {
                        "agent_server": {"type": "responses_api_agents", "name": "hermes"},
                        "resources_server": {"type": "resources_servers", "name": "swe"},
                    }
                }
            },
        }
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


def install_fake_server_client(
    monkeypatch: pytest.MonkeyPatch, post: AsyncMock, *, agent_names: tuple[str, ...] = ()
) -> MagicMock:
    """Route every dispatcher HTTP call through `post` and unwrap FakeResponse."""
    server_client = MagicMock()
    server_client.post = post
    server_client.global_config_dict = OmegaConf.create(
        {
            "my_agent": {"responses_api_agents": {"impl": {}}},
            "my_environment_server": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}},
        }
    )
    if agent_names:
        server_client.global_config_dict = OmegaConf.create(
            {
                key: value
                for agent in agent_names
                for key, value in (
                    (agent, {"responses_api_agents": {"impl": {}}}),
                    (
                        f"{agent}_environment_server",
                        {"environment_servers": {"legacy_agent": {"agent_server": {"name": agent}}}},
                    ),
                )
            }
        )
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

    def test_private_name_aliases_the_public_one(self) -> None:
        assert _get_max_rollout_attempts is get_max_rollout_attempts


class TestRolloutConcurrencyConfig:
    BASE = {"input_jsonl_fpath": "in.jsonl", "output_jsonl_fpath": "out.jsonl"}

    @pytest.mark.parametrize(
        "config_type, config",
        [
            (RolloutCollectionConfig, BASE),
            (E2ERolloutCollectionConfig, {"output_jsonl_fpath": "out.jsonl", "split": "train"}),
        ],
    )
    @pytest.mark.parametrize("limit", [None, 1, 512])
    def test_global_limit_is_available_in_both_collection_modes(
        self,
        config_type: type[RolloutCollectionConfig | E2ERolloutCollectionConfig],
        config: dict[str, str],
        limit: int | None,
    ) -> None:
        """Both collection entrypoints accept the same optional run-wide limit."""
        validated = config_type.model_validate({**config, "num_samples_in_parallel": limit})

        assert validated.num_samples_in_parallel == limit
        assert validated.model_dump()["num_samples_in_parallel"] == limit

    @pytest.mark.parametrize("limit", [0, -1])
    def test_global_limit_must_be_positive(self, limit: int) -> None:
        """Reject unusable global limits before dispatching any work."""
        with pytest.raises(ValidationError):
            RolloutCollectionConfig.model_validate({**self.BASE, "num_samples_in_parallel": limit})


class TestRolloutCollection:
    @pytest.mark.parametrize("environment_routed", [False, True])
    def test_round_robin_by_agent_preserves_agent_order_and_rollout_identity(self, environment_routed: bool) -> None:
        def row(agent: str, task_index: int, rollout_index: int, seed: int) -> dict:
            return {
                **(
                    {NG_ENVIRONMENT_SERVER_KEY: agent} if environment_routed else {AGENT_REF_KEY_NAME: {"name": agent}}
                ),
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: rollout_index,
                "responses_create_params": {"seed": seed},
            }

        examples = [
            row("alpha", 0, 0, 10),
            row("alpha", 0, 1, 11),
            row("alpha", 1, 0, 12),
            row("beta", 2, 0, 20),
            row("beta", 3, 0, 21),
            row("gamma", 4, 0, 30),
        ]
        original = deepcopy(examples)

        scheduled = _round_robin_by_agent(examples)

        expected = [examples[index] for index in (0, 3, 5, 1, 4, 2)]
        assert [id(row) for row in scheduled] == [id(row) for row in expected]
        assert examples == original

    @pytest.mark.parametrize("max_resident_tasks", [None, 1, 2])
    async def test_run_from_config_round_robins_agents_without_rewriting_repeat_identity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        max_resident_tasks: int | None,
    ) -> None:
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        AGENT_REF_KEY_NAME: {"name": agent},
                        TASK_SOURCE_KEY_NAME: agent,
                        "responses_create_params": {"input": []},
                        "source_index": source_index,
                    }
                )
                for source_index, agent in enumerate(("alpha", "alpha", "beta", "gamma"))
            )
            + "\n"
        )
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            num_repeats=2,
            num_repeats_add_seed=True,
            num_samples_in_parallel=1,
            max_resident_rollout_tasks=max_resident_tasks,
            disable_aggregation=True,
            disable_health_check=True,
        )
        dispatched = []

        async def post(server_name: str, url_path: str, json: dict) -> FakeResponse:
            assert url_path == "/run"
            assert server_name == f"{json[AGENT_REF_KEY_NAME]['name']}_environment_server"
            dispatched.append(json)
            return FakeResponse(200, {"response": {}})

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("alpha", "beta", "gamma"))

        await RolloutCollectionHelper().run_from_config(config)

        def identity(row: dict) -> tuple[str, int, int, int]:
            extra_body = json.loads(row["responses_create_params"]["metadata"]["extra_body"])
            return (
                row[AGENT_REF_KEY_NAME]["name"],
                row[TASK_INDEX_KEY_NAME],
                row[ROLLOUT_INDEX_KEY_NAME],
                extra_body["seed"],
            )

        assert [identity(row) for row in dispatched] == [
            ("alpha", 0, 0, 0),
            ("beta", 2, 0, 0),
            ("gamma", 3, 0, 0),
            ("alpha", 0, 1, 1),
            ("beta", 2, 1, 1),
            ("gamma", 3, 1, 1),
            ("alpha", 1, 0, 0),
            ("alpha", 1, 1, 1),
        ]
        materialized = [orjson.loads(line) for line in config.materialized_jsonl_fpath.read_bytes().splitlines()]
        assert [row[AGENT_REF_KEY_NAME]["name"] for row in materialized] == [
            "alpha",
            "alpha",
            "alpha",
            "alpha",
            "beta",
            "beta",
            "gamma",
            "gamma",
        ]
        empty_global_config.assert_called_once_with()

    @pytest.mark.parametrize("resume_from_cache", [False, True])
    @pytest.mark.parametrize("route_failures_to_sidecar", [False, True])
    @pytest.mark.parametrize("retain_results_in_memory", [False, True])
    async def test_batch_status_tracks_collection_and_standalone_aggregation_repair(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        resume_from_cache: bool,
        route_failures_to_sidecar: bool,
        retain_results_in_memory: bool,
    ) -> None:
        """Validate the full batch on fresh and resumed runs, then refresh repaired scores."""
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        AGENT_REF_KEY_NAME: {"name": agent},
                        TASK_SOURCE_KEY_NAME: f"{agent}_source",
                        "responses_create_params": {"input": []},
                    }
                )
                for agent in ("alpha", "beta")
            )
            + "\n"
        )
        output_fpath = tmp_path / "rollouts.jsonl"
        manifest_fpath = tmp_path / "batch_manifest.json"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            batch_manifest_fpath=str(manifest_fpath),
            disable_health_check=True,
            resume_from_cache=resume_from_cache,
            route_failures_to_sidecar=route_failures_to_sidecar,
            retain_results_in_memory=retain_results_in_memory,
            max_resident_rollout_tasks=1,
        )
        materialized_rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        if resume_from_cache:
            config.materialized_jsonl_fpath.write_bytes(
                b"".join(orjson.dumps(row) + b"\n" for row in materialized_rows)
            )
            output_fpath.write_bytes(orjson.dumps({**materialized_rows[0], "reward": 1.0}) + b"\n")
        observations = observe_materialized_rows(materialized_rows)
        manifest_fpath.write_bytes(
            orjson.dumps(
                {
                    "schema_version": "1",
                    "members": {
                        f"{agent}-benchmark": {
                            "agent_name": agent,
                            "task_sources": observed.task_sources,
                            "dataset_sha256": observed.dataset_sha256,
                            "expected_task_count": observed.task_count,
                            "expected_rollout_count": observed.rollout_count,
                            "repeat_policy": observed.repeat_policy.model_dump(),
                            "resolved_recipe_sha256": "a" * 64,
                            "metric_keys": ["mean/reward"],
                        }
                        for agent, observed in observations.items()
                    },
                }
            )
        )

        fail_beta_aggregation = True

        async def post(server_name: str, url_path: str, json, **kwargs) -> FakeResponse:
            if url_path == "/run":
                assert not (resume_from_cache and server_name == "alpha_environment_server")
                return FakeResponse(200, {"reward": 1.0 if server_name == "alpha_environment_server" else 0.0})
            assert url_path == "/aggregate_metrics"
            if server_name == "beta_environment_server" and fail_beta_aggregation:
                return FakeResponse(500)
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("alpha", "beta"))
        monkeypatch.setattr(nemo_gym.batch_status, "BATCH_STATUS_WRITE_INTERVAL_SECONDS", 0)
        status_snapshots = []
        atomic_write = nemo_gym.batch_status._atomic_write_json

        def capture_status_write(path: Path, payload: dict) -> None:
            status_snapshots.append(deepcopy(payload))
            atomic_write(path, payload)

        monkeypatch.setattr(nemo_gym.batch_status, "_atomic_write_json", capture_status_write)

        with pytest.raises(RuntimeError, match="Aggregation failed for agents: beta"):
            await RolloutCollectionHelper().run_from_config(config)

        assert [orjson.loads(line) for line in config.materialized_jsonl_fpath.read_bytes().splitlines()] == (
            materialized_rows
        )
        assert len(output_fpath.read_bytes().splitlines()) == 2
        metrics = orjson.loads((tmp_path / "rollouts_aggregate_metrics.json").read_bytes())
        assert metrics[0]["key_metrics"]["mean/reward"] == 1.0
        assert AGGREGATION_ERROR_KEY not in metrics[0]
        status_fpath = tmp_path / "batch_status.json"
        status = orjson.loads(status_fpath.read_bytes())
        assert status["members"]["alpha"]["completed_rollout_count"] == 1
        assert status["members"]["alpha"]["aggregation_status"] == "complete"
        assert status["members"]["beta"]["completed_rollout_count"] == 1
        assert status["members"]["beta"]["aggregation_status"] == "error"
        assert status["members"]["beta"][AGGREGATION_ERROR_KEY] == {
            "type": "ClientResponseError",
            "http_status": 500,
        }
        observed_progress = {
            sum(member["completed_rollout_count"] for member in snapshot["members"].values())
            for snapshot in status_snapshots
        }
        assert observed_progress == ({1, 2} if resume_from_cache else {0, 1, 2})

        aggregate_config = RolloutAggregationConfig(
            input_glob=str(output_fpath),
            output_jsonl_fpath=str(output_fpath),
            batch_manifest_fpath=str(manifest_fpath),
            disable_health_check=True,
        )
        with pytest.raises(RuntimeError, match="Aggregation failed for agents: beta"):
            await RolloutAggregationHelper().run_from_config(aggregate_config)
        failed_status = orjson.loads(status_fpath.read_bytes())
        assert failed_status["members"]["alpha"]["aggregation_status"] == "complete"
        assert failed_status["members"]["beta"]["aggregation_status"] == "error"

        fail_beta_aggregation = False
        await RolloutAggregationHelper().run_from_config(aggregate_config)

        repaired_status = orjson.loads(status_fpath.read_bytes())
        assert repaired_status["members"]["alpha"]["aggregation_status"] == "complete"
        assert repaired_status["members"]["beta"]["aggregation_status"] == "complete"
        assert repaired_status["members"]["alpha"]["observed_metric_keys"] == ["mean/reward"]
        assert repaired_status["members"]["beta"]["observed_metric_keys"] == ["mean/reward"]
        assert not any(AGGREGATION_ERROR_KEY in member for member in repaired_status["members"].values())

    @pytest.mark.parametrize(
        "resume_from_cache, existing_rollouts",
        [(False, True), (True, True), (True, False)],
        ids=["fresh-run", "resume", "resume-missing-output"],
    )
    async def test_batch_manifest_failure_preserves_existing_artifacts(
        self, tmp_path: Path, resume_from_cache: bool, existing_rollouts: bool
    ) -> None:
        """Reject an invalid manifest without replacing inputs or clearing saved progress."""
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            json.dumps(
                {
                    AGENT_REF_KEY_NAME: {"name": "alpha"},
                    TASK_SOURCE_KEY_NAME: "alpha_source",
                    "responses_create_params": {"input": []},
                }
            )
            + "\n"
        )
        output_fpath = tmp_path / "rollouts.jsonl"
        failures_fpath = _failures_path_for(output_fpath)
        manifest_fpath = tmp_path / "batch_manifest.json"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            batch_manifest_fpath=str(manifest_fpath),
            disable_health_check=True,
            resume_from_cache=resume_from_cache,
        )
        materialized_rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        cached_row = deepcopy(materialized_rows[0])
        cached_row["responses_create_params"]["input"] = [{"role": "user", "content": "original prompt"}]
        saved_artifacts = {
            config.materialized_jsonl_fpath: orjson.dumps(cached_row) + b"\n",
            failures_fpath: orjson.dumps({**cached_row, "_ng_failure_class": "agent_error"}) + b"\n",
            tmp_path / "batch_status.json": b'{"members": {}}\n',
        }
        if existing_rollouts:
            saved_artifacts[output_fpath] = orjson.dumps({**cached_row, "reward": 1.0}) + b"\n"
        for path, content in saved_artifacts.items():
            path.write_bytes(content)

        observations = observe_materialized_rows(materialized_rows)
        observed = observations["alpha"]
        manifest_fpath.write_bytes(
            orjson.dumps(
                {
                    "schema_version": "1",
                    "members": {
                        "alpha-benchmark": {
                            "agent_name": "alpha",
                            "task_sources": observed.task_sources,
                            "dataset_sha256": "0" * 64,
                            "expected_task_count": observed.task_count,
                            "expected_rollout_count": observed.rollout_count,
                            "repeat_policy": observed.repeat_policy.model_dump(),
                            "resolved_recipe_sha256": "a" * 64,
                            "metric_keys": ["mean/reward"],
                        }
                    },
                }
            )
        )

        with pytest.raises(ConfigError, match="dataset_sha256 mismatch"):
            await RolloutCollectionHelper().run_from_config(config)

        for path, content in saved_artifacts.items():
            assert path.read_bytes() == content, f"Manifest validation changed {path.name}"
        assert output_fpath.exists() == existing_rollouts

    async def test_run_examples_queues_preordered_rows_deterministically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        examples = [
            {
                AGENT_REF_KEY_NAME: {"name": agent},
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for task_index, agent in enumerate(("alpha", "beta", "alpha", "beta"))
        ]
        started = []
        first_wave_started = asyncio.Event()
        release = asyncio.Event()

        async def post(server_name: str, url_path: str, json: dict) -> FakeResponse:
            assert url_path == "/run"
            started.append((json[AGENT_REF_KEY_NAME]["name"], json[TASK_INDEX_KEY_NAME]))
            if len(started) == 2:
                first_wave_started.set()
            await release.wait()
            return FakeResponse(200, {"response": {}})

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("alpha", "beta"))

        with pytest.warns(DeprecationWarning, match="legacy path"):
            completions = RolloutCollectionHelper()._run_examples_with_metadata(
                examples, semaphore=asyncio.Semaphore(2)
            )
        await asyncio.sleep(0)
        assert started == []
        completion_waiters = list(completions)
        collection = asyncio.gather(*completion_waiters)
        try:
            await asyncio.wait_for(first_wave_started.wait(), timeout=1)
            assert started == [("alpha", 0), ("beta", 1)]
        finally:
            release.set()
            await collection

        assert started == [("alpha", 0), ("beta", 1), ("alpha", 2), ("beta", 3)]

    @pytest.mark.parametrize("limit", [1, 2, 3])
    async def test_global_limit_is_shared_across_agents(self, monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
        """Mixed-agent work fills but never exceeds the single shared limit."""
        examples = [
            {
                AGENT_REF_KEY_NAME: {"name": agent},
                TASK_SOURCE_KEY_NAME: agent,
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for task_index, agent in enumerate(("alpha", "beta", "alpha", "beta", "alpha", "beta"))
        ]
        active_by_agent = Counter()
        active_total = 0
        max_active_total = 0
        first_wave_started = asyncio.Event()
        release = asyncio.Event()

        async def post(server_name: str, url_path: str, json: dict) -> FakeResponse:
            nonlocal active_total, max_active_total
            assert url_path == "/run"
            active_total += 1
            active_by_agent[json[AGENT_REF_KEY_NAME]["name"]] += 1
            max_active_total = max(max_active_total, active_total)
            if active_total == limit:
                first_wave_started.set()
            try:
                await release.wait()
                return FakeResponse(200, {"response": {}})
            finally:
                active_total -= 1
                active_by_agent[json[AGENT_REF_KEY_NAME]["name"]] -= 1

        server_client = install_fake_server_client(
            monkeypatch, AsyncMock(side_effect=post), agent_names=("alpha", "beta")
        )
        completions = RolloutCollectionHelper()._run_examples_with_metadata(
            examples,
            semaphore=asyncio.Semaphore(limit),
        )
        collection = asyncio.gather(*list(completions))

        try:
            await asyncio.wait_for(first_wave_started.wait(), timeout=1)
            assert active_total == limit
            assert active_by_agent == Counter(row[AGENT_REF_KEY_NAME]["name"] for row in examples[:limit])
        finally:
            release.set()
            await collection

        assert max_active_total == limit
        assert active_total == 0
        assert server_client.post.await_count == len(examples)

    async def test_single_agent_can_use_all_global_slots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The global limit does not reserve slots for other configured agents."""
        examples = [
            {
                AGENT_REF_KEY_NAME: {"name": "gamma"},
                TASK_SOURCE_KEY_NAME: "gamma",
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for task_index in range(3)
        ]
        active = 0
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def post(server_name: str, url_path: str, json: dict) -> FakeResponse:
            nonlocal active
            assert server_name == "gamma_environment_server"
            assert url_path == "/run"
            active += 1
            if active == 3:
                all_started.set()
            try:
                await release.wait()
                return FakeResponse(200, {"response": {}})
            finally:
                active -= 1

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("alpha", "gamma"))
        completions = RolloutCollectionHelper().run_examples(
            examples,
            semaphore=asyncio.Semaphore(3),
        )
        collection = asyncio.gather(*list(completions))

        try:
            await asyncio.wait_for(all_started.wait(), timeout=1)
            assert active == 3
        finally:
            release.set()
            await collection

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
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
            }
        )

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

    async def test_run_examples_rejects_non_positive_resident_task_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
        ]

        install_fake_server_client(monkeypatch, AsyncMock())

        with pytest.raises(ValueError, match="max_resident_tasks must be >= 1"):
            RolloutCollectionHelper().run_examples(rows, max_resident_tasks=0)

    async def test_run_examples_bounds_resident_rollout_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        num_rows = 32
        max_resident_tasks = 4
        started = 0
        peak_started = 0
        release = asyncio.Event()
        resident_window_started = asyncio.Event()

        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for i in range(num_rows)
        ]

        async def post(*args, **kwargs):
            nonlocal started, peak_started
            started += 1
            peak_started = max(peak_started, started)
            if started == max_resident_tasks:
                resident_window_started.set()
            await release.wait()
            started -= 1
            return FakeResponse(200)

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_response_json",
            AsyncMock(return_value={"reward": 1}),
        )

        futures = RolloutCollectionHelper().run_examples(
            rows,
            max_resident_tasks=max_resident_tasks,
        )

        first = asyncio.create_task(next(futures))

        await asyncio.wait_for(resident_window_started.wait(), timeout=1)
        assert peak_started == max_resident_tasks
        assert started == max_resident_tasks

        release.set()
        await first

        for future in futures:
            await future

        assert started == 0

    async def test_bounded_admission_is_independent_of_request_concurrency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for i in range(16)
        ]
        release = asyncio.Event()
        request_started = asyncio.Event()
        active_requests = 0
        peak_active_requests = 0

        async def post(*args, **kwargs):
            nonlocal active_requests, peak_active_requests
            active_requests += 1
            peak_active_requests = max(peak_active_requests, active_requests)
            request_started.set()
            await release.wait()
            active_requests -= 1
            return FakeResponse(200)

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_response_json",
            AsyncMock(return_value={"reward": 1}),
        )

        completions = RolloutCollectionHelper()._run_examples_with_metadata(
            rows,
            semaphore=asyncio.Semaphore(1),
            max_resident_tasks=4,
        )

        first = asyncio.create_task(next(completions))

        await asyncio.wait_for(request_started.wait(), timeout=1)
        assert completions._resident_task_count == 4
        assert active_requests == 1
        assert peak_active_requests == 1

        release.set()
        await first

        for future in completions:
            await future

        assert peak_active_requests == 1

    async def test_bounded_admission_preserves_simultaneous_completions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        num_rows = 24
        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for i in range(num_rows)
        ]
        release = asyncio.Event()
        resident_window_started = asyncio.Event()
        started = 0
        seen: list[int] = []

        async def post(*args, **kwargs):
            nonlocal started
            row = kwargs["json"]
            started += 1
            if started == 8:
                resident_window_started.set()
            await release.wait()
            seen.append(row[TASK_INDEX_KEY_NAME])
            return FakeResponse(200)

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_response_json",
            AsyncMock(return_value={"reward": 1}),
        )

        completions = RolloutCollectionHelper()._run_examples_with_metadata(
            rows,
            max_resident_tasks=8,
        )

        first = asyncio.create_task(next(completions))

        await asyncio.wait_for(resident_window_started.wait(), timeout=1)
        assert completions._resident_task_count == 8

        release.set()

        completed = [(await first).row[TASK_INDEX_KEY_NAME]]
        for future in completions:
            completed.append((await future).row[TASK_INDEX_KEY_NAME])

        assert len(completed) == num_rows
        assert len(set(completed)) == num_rows
        assert sorted(completed) == list(range(num_rows))
        assert sorted(seen) == list(range(num_rows))
        assert completions._resident_task_count == 0

    async def test_default_dispatch_starts_rollouts_in_input_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Repeats in the round-by-round order that interleave_repeats prepares.
        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: rollout_index,
            }
            for rollout_index in range(3)
            for task_index in range(3)
        ]
        started: list[tuple[int, int]] = []

        async def post(*args, **kwargs):
            row = kwargs["json"]
            started.append((row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]))
            return FakeResponse(200, {"reward": 1})

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        for future in RolloutCollectionHelper()._run_examples_with_metadata(rows, semaphore=asyncio.Semaphore(1)):
            await future

        assert started == [(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in rows]

    async def test_bounded_failure_propagates_with_concurrent_consumers(self) -> None:
        fail = asyncio.Event()
        block = asyncio.Event()

        async def failing_rollout():
            await fail.wait()
            raise RuntimeError("rollout failed")

        async def blocked_rollout():
            await block.wait()

        completions = nemo_gym.rollout_collection._BoundedCompletionIterator(
            iter([failing_rollout(), blocked_rollout()]),
            max_resident_tasks=2,
            total=2,
        )
        first = asyncio.create_task(next(completions))
        await asyncio.sleep(0)
        second = asyncio.create_task(next(completions))
        await asyncio.sleep(0)

        fail.set()
        with pytest.raises(RuntimeError, match="rollout failed"):
            await asyncio.wait_for(first, timeout=1)

        assert not second.done()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        await asyncio.wait_for(completions.aclose(), timeout=1)

    async def test_bounded_aclose_with_concurrent_consumer_blocked_does_not_deadlock(self) -> None:
        fail = asyncio.Event()

        async def failing_rollout():
            await fail.wait()
            raise RuntimeError("rollout failed")

        async def blocked_rollout():
            await asyncio.Event().wait()

        completions = nemo_gym.rollout_collection._BoundedCompletionIterator(
            iter([failing_rollout(), blocked_rollout()]),
            max_resident_tasks=2,
            total=2,
        )
        first = asyncio.create_task(next(completions))
        await asyncio.sleep(0)
        second = asyncio.create_task(next(completions))
        await asyncio.sleep(0)

        fail.set()
        with pytest.raises(RuntimeError, match="rollout failed"):
            await asyncio.wait_for(first, timeout=1)

        await asyncio.wait_for(completions.aclose(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(second, timeout=1)

    async def test_bounded_admission_aclose_cancels_only_resident_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        num_rows = 32
        max_resident_tasks = 4
        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for i in range(num_rows)
        ]
        started: set[int] = set()
        cancelled: set[int] = set()
        blocker = asyncio.Event()
        resident_window_started = asyncio.Event()

        async def post(*args, **kwargs):
            row = kwargs["json"]
            task_index = row[TASK_INDEX_KEY_NAME]
            started.add(task_index)
            if len(started) == max_resident_tasks:
                resident_window_started.set()
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                cancelled.add(task_index)
                raise
            return FakeResponse(200)

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        completions = RolloutCollectionHelper()._run_examples_with_metadata(
            rows,
            max_resident_tasks=max_resident_tasks,
        )

        first = asyncio.create_task(next(completions))

        await asyncio.wait_for(resident_window_started.wait(), timeout=1)
        assert len(started) == max_resident_tasks
        assert completions._resident_task_count == max_resident_tasks

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(completions.aclose(), timeout=1)

        assert len(started) == max_resident_tasks
        assert cancelled == started
        assert completions._resident_task_count == 0

    async def test_run_examples_bounded_admission_processes_each_row_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for i in range(17)
        ]
        seen: list[int] = []

        async def post(*args, **kwargs):
            row = kwargs["json"]
            seen.append(row[TASK_INDEX_KEY_NAME])
            await asyncio.sleep(0)
            return FakeResponse(200)

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_response_json",
            AsyncMock(return_value={"reward": 1}),
        )

        completed = []
        for future in RolloutCollectionHelper().run_examples(
            rows,
            max_resident_tasks=3,
        ):
            row, _ = await future
            completed.append(row[TASK_INDEX_KEY_NAME])

        assert sorted(seen) == list(range(17))
        assert sorted(completed) == list(range(17))

    async def test_failure_row_survives_serialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The record has to cross the jsonl, pickle and Ray boundaries as plain data."""
        post = AsyncMock(return_value=FakeResponse(500))
        install_fake_server_client(monkeypatch, post)

        _, result = await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))

        assert pickle.loads(pickle.dumps(result)) == result
        assert orjson.loads(orjson.dumps(result)) == result

    @pytest.mark.parametrize("require_complete", [False, True])
    async def test_run_from_config_routes_agent_failure_to_sidecar_and_out_of_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        empty_global_config: MagicMock,
        require_complete: bool,
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
            require_complete=require_complete,
        )
        with (
            pytest.raises(RuntimeError, match="EVAL FAILED: 1/2 samples completed")
            if require_complete
            else nullcontext()
        ):
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
                return FakeResponse(200, {"reward": 0.0})
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            disable_health_check=True,
            require_complete=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        assert len(dispatched) == 1
        assert dispatched[0][ATTEMPT_INDEX_KEY_NAME] == 1
        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert [r["reward"] for r in persisted] == [0.0]

    async def test_run_from_config_resume_drops_a_truncated_final_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """A hard kill mid-write leaves half a line; resume drops it and re-runs that rollout."""
        output_jsonl_fpath = tmp_path / "output.jsonl"
        materialized_fpath = tmp_path / "output_materialized_inputs.jsonl"
        materialized_fpath.write_bytes(
            b"".join(
                orjson.dumps(
                    {
                        "responses_create_params": {"input": []},
                        AGENT_REF_KEY_NAME: {"name": "my_agent"},
                        TASK_INDEX_KEY_NAME: task_index,
                        ROLLOUT_INDEX_KEY_NAME: 0,
                    }
                )
                + b"\n"
                for task_index in (0, 1)
            )
        )
        done = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, AGENT_REF_KEY_NAME: {"name": "my_agent"}}
        output_jsonl_fpath.write_bytes(orjson.dumps({**done, "reward": 1.0}) + b"\n" + b'{"_ng_task_index": 1, "rew')

        dispatched: list[dict] = []

        async def post(server_name: str, url_path: str, json: dict, **kwargs):
            if url_path == "/run":
                dispatched.append(json)
                return FakeResponse(200, {"reward": 0.0})
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            disable_health_check=True,
            require_complete=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        assert [row[TASK_INDEX_KEY_NAME] for row in dispatched] == [1]
        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert sorted((r[TASK_INDEX_KEY_NAME], r["reward"]) for r in persisted) == [(0, 1.0), (1, 0.0)]

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (b'{"a": 1}\n{"b": 2', b'{"a": 1}\n'),
            (b'{"a": 1}\n{"b": 2}', b'{"a": 1}\n{"b": 2}\n'),
            (b'{"a": 1}\n', b'{"a": 1}\n'),
            (b'{"a"', b""),
            (b"", b""),
        ],
    )
    def test_drop_truncated_tail(self, tmp_path: Path, content: bytes, expected: bytes) -> None:
        fpath = tmp_path / "rollouts.jsonl"
        fpath.write_bytes(content)
        _drop_truncated_tail(fpath)
        assert fpath.read_bytes() == expected

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

    def test_missing_rollout_rows_counted_as_zero_covers_only_what_produced_nothing(self, tmp_path: Path) -> None:
        """A materialized rollout with no row anywhere is the one case no failure class reaches."""
        first = tmp_path / "shard0_materialized_inputs.jsonl"
        second = tmp_path / "shard1_materialized_inputs.jsonl"

        def materialized(task_index: int) -> dict:
            return {
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "agent_ref": {"name": "my_agent"},
                "task_name": f"task-{task_index}",
                "responses_create_params": {"input": []},
            }

        first.write_bytes(b"\n".join(orjson.dumps(materialized(i)) for i in range(3)) + b"\n")
        second.write_bytes(orjson.dumps(materialized(2)) + b"\n")

        # Task 0 was scored and task 2 appears in both shards, so neither may be counted twice.
        rows = _missing_rollout_rows_counted_as_zero([first, second], [], {(0, 0)})
        assert [row[TASK_INDEX_KEY_NAME] for row in rows] == [1, 2]
        assert all(row["reward"] == 0.0 for row in rows)

        # Only identity travels; dataset fields are added later, and only those real rows carry.
        assert rows[0] == {
            TASK_INDEX_KEY_NAME: 1,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "agent_ref": {"name": "my_agent"},
            "reward": 0.0,
        }

        assert _missing_rollout_rows_counted_as_zero([tmp_path / "absent.jsonl"], [], set()) == []

    def test_a_failure_the_caller_did_not_select_is_not_counted_as_missing(self, tmp_path: Path) -> None:
        """Otherwise the selection becomes a no-op: what it excludes comes back as a zero."""
        materialized = tmp_path / "shard0_materialized_inputs.jsonl"
        failures = tmp_path / "shard0_failures.jsonl"

        def row(task_index: int) -> dict:
            return {
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "agent_ref": {"name": "my_agent"},
                "task_name": f"task-{task_index}",
            }

        materialized.write_bytes(b"\n".join(orjson.dumps(row(i)) for i in range(3)) + b"\n")
        failures.write_bytes(orjson.dumps(row(1) | {NG_FAILURE_CLASS_KEY: "agent_request_failed"}) + b"\n")

        # Task 0 scored, task 1 failed in a class the caller left out, task 2 produced nothing.
        counted = _missing_rollout_rows_counted_as_zero([materialized], [failures], {(0, 0)})

        assert [r[TASK_INDEX_KEY_NAME] for r in counted] == [2]

    async def test_a_missing_native_rollout_is_scored_by_its_environment_server(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A native taskset row names no agent, so its zero reaches metrics only through the server stamp."""
        planned = [
            {TASK_INDEX_KEY_NAME: task_index, ROLLOUT_INDEX_KEY_NAME: 0, NG_ENVIRONMENT_SERVER_KEY: "environment"}
            for task_index in range(2)
        ]
        materialized = tmp_path / "output_materialized_inputs.jsonl"
        materialized.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in planned))

        # Task 0 scored 1.0, task 1 produced nothing.
        missing = _missing_rollout_rows_counted_as_zero([materialized], [], {(0, 0)})
        assert [row[NG_ENVIRONMENT_SERVER_KEY] for row in missing] == ["environment"]

        async def post(server_name: str, url_path: str, json, **kwargs):
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = _environment_server_config()
        scored = [dict(planned[0], reward=1.0)] + missing

        metrics_fpath = await RolloutCollectionHelper()._call_aggregate_metrics(
            scored, scored, tmp_path / "output.jsonl"
        )

        [entry] = json.loads(metrics_fpath.read_text())
        assert entry[NG_ENVIRONMENT_SERVER_KEY] == "environment"
        assert entry["key_metrics"] == {"mean/reward": 0.5}

    def test_a_row_without_indices_does_not_break_the_ordering(self) -> None:
        """A sort key of None against an int raises, and every row is ordered before aggregation."""
        assert _rollout_order_key({TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 3}) == (2, 3)
        assert _rollout_order_key({}) == (0, 0)
        assert _rollout_order_key({TASK_INDEX_KEY_NAME: None, ROLLOUT_INDEX_KEY_NAME: None}) == (0, 0)

    @pytest.mark.parametrize(
        ("count_missing", "expected_scored", "expected_mean"),
        [(False, 1, 1.0), (True, 2, 0.5)],
        ids=["off by default", "opted in"],
    )
    @pytest.mark.parametrize("retain_results", [True, False], ids=["retaining", "non-retaining"])
    async def test_run_from_config_counts_a_rollout_that_produced_no_row_as_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        count_missing: bool,
        retain_results: bool,
        expected_scored: int,
        expected_mean: float,
    ) -> None:
        """One success and one rollout written nowhere: the second counts only when asked for.

        A kill-shaped result reaches neither the rollouts jsonl nor the sidecar, so the failure
        classes cannot see it and the materialized inputs are the only record that it was owed.
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
                    return FakeResponse(200, {NG_NO_PERSIST_KEY: True})
                return FakeResponse(200, {"reward": 1.0})
            aggregated["verify_responses"] = [dict(r) for r in json.verify_responses]
            return FakeResponse(200, compute_aggregate_metrics(aggregated["verify_responses"]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            count_missing_rollouts_as_zero=count_missing,
            retain_results_in_memory=retain_results,
            disable_health_check=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert [row["reward"] for row in persisted] == [1.0]

        # Nothing in the sidecar either, so no failure class could have reached it.
        failures_fpath = _failures_path_for(output_jsonl_fpath)
        assert not failures_fpath.exists() or not failures_fpath.read_bytes().strip()

        assert len(aggregated["verify_responses"]) == expected_scored
        assert [row[TASK_INDEX_KEY_NAME] for row in aggregated["verify_responses"]] == (
            [0, 1] if count_missing else [1]
        )
        metrics_fpath = output_jsonl_fpath.with_stem(output_jsonl_fpath.stem + "_aggregate_metrics").with_suffix(
            ".json"
        )
        assert orjson.loads(metrics_fpath.read_bytes())[0]["key_metrics"] == {"mean/reward": expected_mean}

    @pytest.mark.parametrize("count_missing", [False, True], ids=["off by default", "opted in"])
    async def test_a_run_whose_every_rollout_went_missing_is_scored_only_when_opted_in(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        count_missing: bool,
    ) -> None:
        """The no-result guard counts the opted-in zeros, as it already counts opted-in failure rows."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                return FakeResponse(200, {NG_NO_PERSIST_KEY: True})
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            count_missing_rollouts_as_zero=count_missing,
            disable_health_check=True,
        )
        if count_missing:
            await RolloutCollectionHelper().run_from_config(config)
            metrics_fpath = output_jsonl_fpath.with_stem(output_jsonl_fpath.stem + "_aggregate_metrics").with_suffix(
                ".json"
            )
            assert orjson.loads(metrics_fpath.read_bytes())[0]["key_metrics"] == {"mean/reward": 0.0}
        else:
            with pytest.raises(RuntimeError, match="produced a result"):
                await RolloutCollectionHelper().run_from_config(config)

    @pytest.mark.parametrize("label_from", ["dataset", "verifier"])
    @pytest.mark.parametrize("counted_as", ["failure", "missing"])
    async def test_a_counted_first_repeat_keeps_its_task_in_the_task_level_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        counted_as: str,
        label_from: str,
    ) -> None:
        """The zero for repeat 0 comes first in its task, where hooks read task-level fields.

        It carries the dataset fields real rows carry, and no other: the aggregator averages
        every number it is handed, so a field no real row reports would become a metric. A
        label the verifier computes is not in the dataset, so the hook reads it from a real row.
        """
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        "responses_create_params": {"input": []},
                        "agent_ref": {"name": "my_agent"},
                        "verifier_metadata": {"level": difficulty},
                        "source_id": 7,
                    }
                    | ({"difficulty": difficulty} if label_from == "dataset" else {})
                )
                for difficulty in ("easy", "hard")
            )
            + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        aggregated: dict[str, list[dict]] = {}

        def subset_metrics(tasks: list[list[dict]]) -> dict:
            return compute_subset_metrics(tasks, "difficulty")

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                level = json["verifier_metadata"]["level"]
                if level == "easy" and json[ROLLOUT_INDEX_KEY_NAME] == 0:
                    if counted_as == "failure":
                        raise ServerDisconnectedError()
                    return FakeResponse(200, {NG_NO_PERSIST_KEY: True})
                # The verifier reports the label and drops source_id, which it does not declare.
                return FakeResponse(200, {"reward": 1.0, "difficulty": level})
            aggregated["verify_responses"] = [dict(r) for r in json.verify_responses]
            metrics = compute_aggregate_metrics(aggregated["verify_responses"], compute_metrics_fn=subset_metrics)
            return FakeResponse(200, metrics.model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            num_repeats=2,
            route_failures_to_sidecar=True,
            count_failure_classes_as_zero=[AGENT_REQUEST_FAILED_FAILURE_CLASS] if counted_as == "failure" else [],
            count_missing_rollouts_as_zero=counted_as == "missing",
            disable_health_check=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        first = aggregated["verify_responses"][0]
        assert (first[TASK_INDEX_KEY_NAME], first[ROLLOUT_INDEX_KEY_NAME], first["reward"]) == (0, 0, 0.0)
        assert first.get("difficulty") == ("easy" if label_from == "dataset" else None)
        assert "source_id" not in first
        metrics_fpath = output_jsonl_fpath.with_stem(output_jsonl_fpath.stem + "_aggregate_metrics").with_suffix(
            ".json"
        )
        agent_metrics = orjson.loads(metrics_fpath.read_bytes())[0]["agent_metrics"]
        assert agent_metrics["easy/pass@1[avg-of-2]/accuracy"] == 50.0

    async def test_rows_and_results_stay_aligned_across_two_servers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """Rows and results are zipped positionally, so a zero must not take its neighbour's server."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            "\n".join(
                json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": agent}, "x": i})
                for i, agent in enumerate(("agent_a", "agent_b"))
            )
            + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        by_server: dict[str, list[float]] = {}

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                if json["x"] == 0:
                    return FakeResponse(200, {NG_NO_PERSIST_KEY: True})
                return FakeResponse(200, {"reward": 1.0})
            by_server[server_name] = [r["reward"] for r in json.verify_responses]
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(
            {
                "agent_a": {"responses_api_agents": {"impl": {}}},
                "agent_b": {"responses_api_agents": {"impl": {}}},
                "env_a": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent_a"}}}},
                "env_b": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent_b"}}}},
            }
        )

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            count_missing_rollouts_as_zero=True,
            disable_health_check=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        assert by_server == {"env_a": [0.0], "env_b": [1.0]}

    async def test_imputed_rows_are_reported_in_the_coverage(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An imputed zero is in the score, so the report names it rather than going silent."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            "\n".join(
                json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}, "x": i})
                for i in range(2)
            )
            + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        exported: dict[str, float] = {}

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                if json["x"] == 0:
                    return FakeResponse(200, {NG_NO_PERSIST_KEY: True})
                return FakeResponse(200, {"reward": 1.0})
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        monkeypatch.setattr("nemo_gym.rollout_collection.get_exporters", lambda: ["any"])
        monkeypatch.setattr(
            "nemo_gym.rollout_collection.export_metrics", lambda metrics, **kw: exported.update(metrics)
        )

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            count_missing_rollouts_as_zero=True,
            disable_health_check=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        coverage = {k: v for k, v in exported.items() if k.startswith("coverage/")}
        assert coverage == {"coverage/expected": 2, "coverage/scored": 2, "coverage/missing": 0, "coverage/imputed": 1}
        assert "Scored as zero because they left no row: 1 of 2 rollouts" in capsys.readouterr().out

    async def test_coverage_is_exported_when_aggregation_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """disable_aggregation skips the branch that counts imputed rows; the export still runs."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        exported: dict[str, float] = {}

        async def post(server_name: str, url_path: str, json, **kwargs):
            return FakeResponse(200, {"reward": 1.0})

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        monkeypatch.setattr("nemo_gym.rollout_collection.get_exporters", lambda: ["any"])
        monkeypatch.setattr(
            "nemo_gym.rollout_collection.export_metrics", lambda metrics, **kw: exported.update(metrics)
        )

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            count_missing_rollouts_as_zero=True,
            disable_aggregation=True,
            disable_health_check=True,
        )
        await RolloutCollectionHelper().run_from_config(config)

        assert exported["coverage/imputed"] == 0

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

    @pytest.mark.parametrize("count_judge_failure", [False, True])
    async def test_masked_judge_failure_counts_as_zero_only_when_opted_in(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        count_judge_failure: bool,
    ) -> None:
        """Online and offline aggregation honor the opt-in without rewriting failure evidence."""
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(
            "".join(
                json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}, "x": i})
                + "\n"
                for i in range(4)
            )
        )
        output_path = tmp_path / "output.jsonl"
        failure = {
            "reward": 0.0,
            "mask_sample": True,
            "failure_kind": "judge_failed",
            "failure_reason": "judge unavailable",
            "instance_config": {"mask_sample": True},
            NG_FAILURE_CLASS_KEY: "judge_failed",
            "_ng_failure_judge_error": "judge unavailable",
        }
        metrics = []
        metric_inputs = []

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                return FakeResponse(200, failure if json["x"] == 0 else {"reward": 1.0})
            metric_inputs.append(json.verify_responses)
            result = compute_aggregate_metrics(json.verify_responses)
            metrics.append(result)
            return FakeResponse(200, result.model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        counted = ["judge_failed"] if count_judge_failure else []
        await RolloutCollectionHelper().run_from_config(
            RolloutCollectionConfig(
                input_jsonl_fpath=str(input_path),
                output_jsonl_fpath=str(output_path),
                route_failures_to_sidecar=True,
                count_failure_classes_as_zero=counted,
                disable_health_check=True,
            )
        )
        sidecar_path = _failures_path_for(output_path)
        original_sidecar = sidecar_path.read_bytes()
        await RolloutAggregationHelper().run_from_config(
            RolloutAggregationConfig(
                input_glob=str(output_path),
                output_jsonl_fpath=str(tmp_path / "merged.jsonl"),
                count_failure_classes_as_zero=counted,
                disable_health_check=True,
            )
        )
        assert len(metrics) == 2
        for result, inputs in zip(metrics, metric_inputs):
            assert result.key_metrics == {"mean/reward": 0.75 if count_judge_failure else 1.0}
            assert result.agent_metrics.get("coverage/masked_rollouts", 0) == 0
            assert len(inputs) == (4 if count_judge_failure else 3)
            assert all("failure_kind" not in row and "failure_reason" not in row for row in inputs)
        assert sidecar_path.read_bytes() == original_sidecar
        saved_failure = orjson.loads(original_sidecar)
        assert all(saved_failure[key] == value for key, value in failure.items())
        assert len(output_path.read_text().splitlines()) == 3

    @pytest.mark.parametrize("num_failures", [1, 4])
    @pytest.mark.parametrize("counted_classes", [[], [AGENT_RUN_ERROR_FAILURE_CLASS], ["judge_failed"]])
    @pytest.mark.parametrize("disable_aggregation", [False, True])
    async def test_all_judge_failures_only_produce_metrics_when_explicitly_counted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        num_failures: int,
        counted_classes: list[str],
        disable_aggregation: bool,
    ) -> None:
        """An opted-in all-failure run has a score, but never synthetic successful rollouts."""
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(
            "".join(
                json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n"
                for _ in range(num_failures)
            )
        )
        output_path = tmp_path / "output.jsonl"
        failure = {
            "reward": 0.0,
            "mask_sample": True,
            "failure_kind": "judge_failed",
            "failure_reason": "judge unavailable",
            "instance_config": {"mask_sample": True},
            NG_FAILURE_CLASS_KEY: "judge_failed",
            "_ng_failure_judge_error": "judge unavailable",
        }
        metric_inputs = []

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                return FakeResponse(200, deepcopy(failure))
            metric_inputs.append(json.verify_responses)
            return FakeResponse(200, compute_aggregate_metrics(json.verify_responses).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_path),
            output_jsonl_fpath=str(output_path),
            route_failures_to_sidecar=True,
            count_failure_classes_as_zero=counted_classes,
            disable_aggregation=disable_aggregation,
            disable_health_check=True,
        )
        counted = "judge_failed" in counted_classes
        if counted:
            results = await RolloutCollectionHelper().run_from_config(config)
            assert len(results) == num_failures and all(row["mask_sample"] for row in results)
        else:
            # A nonempty opt-in for another class must not bypass the no-results guard.
            with pytest.raises(RuntimeError, match="produced a result"):
                await RolloutCollectionHelper().run_from_config(config)

        sidecar_path = _failures_path_for(output_path)
        original_sidecar = sidecar_path.read_bytes()
        saved = [orjson.loads(line) for line in original_sidecar.splitlines()]
        assert len(saved) == num_failures
        assert all(all(row[key] == value for key, value in failure.items()) for row in saved)
        assert output_path.read_bytes() == b""
        online_path = output_path.with_stem("output_aggregate_metrics").with_suffix(".json")
        if counted and not disable_aggregation:
            assert orjson.loads(online_path.read_bytes())[0]["key_metrics"] == {"mean/reward": 0.0}
        else:
            assert not online_path.exists() and not metric_inputs

        offline_path = await RolloutAggregationHelper().run_from_config(
            RolloutAggregationConfig(
                input_glob=str(output_path),
                output_jsonl_fpath=str(tmp_path / "merged.jsonl"),
                count_failure_classes_as_zero=counted_classes,
                disable_health_check=True,
            )
        )
        if counted:
            assert orjson.loads(offline_path.read_bytes())[0]["key_metrics"] == {"mean/reward": 0.0}
            assert len(metric_inputs) == (1 if disable_aggregation else 2)
            for inputs in metric_inputs:
                assert len(inputs) == num_failures
                assert all(row["reward"] == 0.0 and row["mask_sample"] is False for row in inputs)
        else:
            assert offline_path is None and not metric_inputs
        assert sidecar_path.read_bytes() == original_sidecar
        assert output_path.read_bytes() == (tmp_path / "merged.jsonl").read_bytes() == b""

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

    async def test_aggregate_counts_a_shard_rollout_that_produced_no_row_as_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """`gym eval aggregate` reads each shard's own materialized inputs for what is owed."""
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
        materialized_path_for(shard_fpath).write_bytes(
            b"\n".join(
                orjson.dumps(
                    {
                        TASK_INDEX_KEY_NAME: task_index,
                        ROLLOUT_INDEX_KEY_NAME: 0,
                        AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    }
                )
                for task_index in range(2)
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
            count_missing_rollouts_as_zero=True,
            disable_health_check=True,
        )
        await RolloutAggregationHelper().run_from_config(config)

        assert sorted(row["reward"] for row in aggregated["verify_responses"]) == [0.0, 1.0]

    async def test_aggregate_hands_a_missing_early_repeat_to_metrics_in_its_own_place(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """pass@k reads a task's repeats positionally, so a zero appended at the end shifts them."""
        shard_fpath = tmp_path / "rollouts-chunk0.jsonl"
        shard_fpath.write_bytes(
            orjson.dumps(
                {
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 1,
                    AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    "reward": 1.0,
                }
            )
            + b"\n"
        )
        materialized_path_for(shard_fpath).write_bytes(
            b"\n".join(
                orjson.dumps(
                    {
                        TASK_INDEX_KEY_NAME: 0,
                        ROLLOUT_INDEX_KEY_NAME: rollout_index,
                        AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    }
                )
                for rollout_index in range(2)
            )
            + b"\n"
        )
        aggregated: dict[str, list[dict]] = {}

        async def post(server_name: str, url_path: str, json, **kwargs):
            aggregated["verify_responses"] = [dict(r) for r in json.verify_responses]
            return FakeResponse(200, compute_aggregate_metrics(aggregated["verify_responses"]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))

        config = RolloutAggregationConfig(
            input_glob=str(shard_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            count_missing_rollouts_as_zero=True,
            disable_health_check=True,
        )
        await RolloutAggregationHelper().run_from_config(config)

        # The repeat that went missing is repeat 0, so it must arrive before the one that landed.
        assert [row[ROLLOUT_INDEX_KEY_NAME] for row in aggregated["verify_responses"]] == [0, 1]
        assert [row["reward"] for row in aggregated["verify_responses"]] == [0.0, 1.0]

    async def test_run_from_config_all_failure_non_retaining_upload_removes_spool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(500)))
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_exporters", lambda: [object()])
        export_rollouts = MagicMock()
        monkeypatch.setattr(nemo_gym.rollout_collection, "export_rollouts", export_rollouts)

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            route_failures_to_sidecar=True,
            retain_results_in_memory=False,
            upload_rollouts=True,
            disable_health_check=True,
        )

        with pytest.raises(RuntimeError, match="produced a result"):
            await RolloutCollectionHelper().run_from_config(config)

        export_rollouts.assert_not_called()
        assert not output_jsonl_fpath.with_suffix(".jsonl.upload.tmp").exists()

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
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
            }
        )
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"response": {}}))

        returned_row, result = await next(RolloutCollectionHelper().run_examples([row]))

        assert returned_row is row
        assert result == {"response": {}}
        assert "_ng_rollout_latency_ms" not in result

    async def test_run_examples_rejects_agent_fronted_by_several_environment_servers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row routed by its agent fails before any dispatch when two environment servers name that agent."""
        row = {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock()
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_legacy_server": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}},
                "my_native_server": {
                    "environment_servers": {"single_agent_turn": {"agent_server": {"name": "my_agent"}}}
                },
            }
        )
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )

        with pytest.raises(AmbiguousEnvironmentServerError, match="my_legacy_server.*my_native_server"):
            next(RolloutCollectionHelper().run_examples([row]))
        mock_server_client.post.assert_not_awaited()

    async def test_run_examples_allows_several_servers_for_an_agent_no_row_routes_by(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Twin servers for one agent are valid; only rows that route by that agent need a single server."""
        row = {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        response = MagicMock()
        response.status = 200
        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=response)
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "other_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
                "other_legacy_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "other_agent"}}}
                },
                "other_native_server": {
                    "environment_servers": {"single_agent_turn": {"agent_server": {"name": "other_agent"}}}
                },
            }
        )
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"response": {}}))

        await next(RolloutCollectionHelper().run_examples([row]))

        assert mock_server_client.post.await_args.kwargs["server_name"] == "my_environment_server"

    async def test_run_examples_with_metadata_carries_rollout_latency_alongside_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Internal callers get the timing via _CompletedRollout, never through the result dict."""
        row = {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        response = MagicMock()
        response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=response)
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
            }
        )
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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

    @pytest.mark.parametrize("count_failures_as_zero", [False, True])
    async def test_run_from_config_reports_coverage_against_the_materialized_input_on_resume(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        empty_global_config: MagicMock,
        count_failures_as_zero: bool,
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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
                return None

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            disable_health_check=True,
            require_complete=True,
            count_failure_classes_as_zero=[AGENT_RUN_ERROR_FAILURE_CLASS] if count_failures_as_zero else [],
        )
        with pytest.raises(RuntimeError, match="EVAL FAILED: 2/3 samples completed"):
            await Helper().run_from_config(config)

        printed = capsys.readouterr().out
        if count_failures_as_zero:
            assert "Counting 1 failure row(s) as scored zeros" in printed
        else:
            assert "Rollouts missing from the score: 1 of 3 materialized" in printed
            assert "Metrics cover: 2 of 3 rollouts" in printed

    async def test_run_from_config_resume_progress_counts_only_dispatched_rollouts(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        empty_global_config: MagicMock,
    ) -> None:
        output_fpath = tmp_path / "output.jsonl"
        materialized_fpath = tmp_path / "output_materialized_inputs.jsonl"
        rows = [
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"name": "my agent name"},
                TASK_INDEX_KEY_NAME: task_index,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for task_index in range(2)
        ]
        materialized_fpath.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
        output_fpath.write_bytes(orjson.dumps({**rows[0], "reward": 1.0}) + b"\n")

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                assert examples == [rows[1]]
                future = Future()
                future.set_result(
                    _CompletedRollout(
                        row=rows[1],
                        result={"reward": 1.0},
                        rollout_latency_ms=None,
                    )
                )
                return [future]

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
                return None

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_fpath),
            resume_from_cache=True,
            disable_health_check=True,
        )

        await Helper().run_from_config(config)

        printed = capsys.readouterr().out
        assert "Finished 1 / 1 rollouts (100%)" in printed
        assert "200%" not in printed

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

    def test_preprocess_rows_num_repeats_add_seed_dict_seeds_only_selected_agents(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "apex"}}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "other"}}),
        ]
        fpath.write_text("\n".join(samples) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"apex": 3, "_default": 1},
            num_repeats_add_seed={"apex": True, "_default": False},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        apex_rows = [row for row in rows if row[AGENT_REF_KEY_NAME]["name"] == "apex"]
        other_rows = [row for row in rows if row[AGENT_REF_KEY_NAME]["name"] == "other"]
        assert [json.loads(row["responses_create_params"]["metadata"]["extra_body"])["seed"] for row in apex_rows] == [
            0,
            1,
            2,
        ]
        assert len(other_rows) == 1
        assert "metadata" not in other_rows[0]["responses_create_params"]

    def test_preprocess_rows_num_repeats_add_seed_prefers_dispatched_agent(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        fpath.write_text(json.dumps({"responses_create_params": {"input": []}, "task_source": "source_agent"}) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            agent_map={"source_agent": "target_agent"},
            num_repeats=2,
            num_repeats_add_seed={"target_agent": True, "source_agent": False},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert [row[AGENT_REF_KEY_NAME]["name"] for row in rows] == ["target_agent", "target_agent"]
        assert [json.loads(row["responses_create_params"]["metadata"]["extra_body"])["seed"] for row in rows] == [0, 1]

    def test_preprocess_rows_num_repeats_add_seed_does_not_leak_across_fan_out(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        fpath.write_text(
            json.dumps(
                {
                    "responses_create_params": {"input": [], "metadata": {"trace": "original"}},
                    "task_source": "source_agent",
                }
            )
            + "\n"
        )
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            fan_out={"source_agent": ["seeded_agent", "unseeded_agent"]},
            num_repeats_add_seed={"seeded_agent": True, "unseeded_agent": False},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        seeded_row, unseeded_row = rows
        assert json.loads(seeded_row["responses_create_params"]["metadata"]["extra_body"])["seed"] == 0
        assert unseeded_row["responses_create_params"]["metadata"] == {"trace": "original"}

    def test_preprocess_rows_num_repeats_add_seed_dict_requires_complete_policy(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}}),
        ]
        fpath.write_text("\n".join(samples) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats_add_seed={"alpha": True},
        )

        with pytest.raises(ValueError, match="num_repeats_add_seed dict") as exc_info:
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert "beta" in str(exc_info.value)

    def test_preprocess_rows_num_repeats_add_seed_dict_unknown_agent_warns(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        fpath.write_text(json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}}) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats_add_seed={"alpha": True, "alpah_typo": False},
        )

        with pytest.warns(UserWarning, match="alpah_typo"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 1

    def test_preprocess_examples_seeds_preserve_existing_metadata(self) -> None:
        examples = [
            {
                "agent_ref": {"name": "my_agent"},
                "responses_create_params": {
                    "input": [{"role": "user", "content": "Say hello."}],
                    "metadata": {"source": "example", "extra_body": '{"top_k": 10, "seed": 99}'},
                },
            }
        ]
        original = deepcopy(examples)

        rows = RolloutCollectionHelper().preprocess_examples(examples, num_repeats=3, num_repeats_add_seed=True)

        params = [
            NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"]) for row in rows
        ]
        assert [json.loads(param.metadata["extra_body"]) for param in params] == [
            {"top_k": 10, "seed": seed} for seed in range(3)
        ]
        assert [param.metadata["source"] for param in params] == ["example"] * 3
        assert examples == original

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

    def test_preprocess_rows_interleave_repeats(self, tmp_path: Path) -> None:
        """interleave_repeats dispatches round by round: every task's first repeat, then every second one."""
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": i})
            for i in range(3)
        ]
        fpath.write_text("\n".join(samples) + "\n")

        def order(interleave: bool) -> list:
            config = RolloutCollectionConfig(
                input_jsonl_fpath=str(fpath),
                output_jsonl_fpath=str(tmp_path / "out.jsonl"),
                num_repeats=2,
                interleave_repeats=interleave,
            )
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
            return [(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) for r in rows]

        assert order(False) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        assert order(True) == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

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

    async def test_run_from_config_dispatch_setup_failure_closes_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
    ) -> None:
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "agent"}}) + "\n"
        )
        output_fpath = tmp_path / "output.jsonl"
        failures_fpath = _failures_path_for(output_fpath)
        upload_spool_fpath = output_fpath.with_suffix(".jsonl.upload.tmp")
        tracked_paths = {output_fpath, failures_fpath, upload_spool_fpath}
        opened_files = []
        original_open = Path.open

        def tracked_open(path: Path, *args, **kwargs):
            file = original_open(path, *args, **kwargs)
            if path in tracked_paths:
                opened_files.append(file)
            return file

        monkeypatch.setattr(Path, "open", tracked_open)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_exporters", lambda: [object()])

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, *args, **kwargs):
                raise ValueError("dispatch validation failed")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            retain_results_in_memory=False,
            upload_rollouts=True,
            disable_aggregation=True,
            disable_health_check=True,
        )

        with pytest.raises(ValueError, match="dispatch validation failed"):
            await Helper().run_from_config(config)

        assert len(opened_files) == 3
        assert all(file.closed for file in opened_files)
        assert not upload_spool_fpath.exists()

    async def test_run_from_config_processing_failure_cancels_bounded_resident_tasks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
    ) -> None:
        window = 3
        offered = 12
        started: set[int] = set()
        cancelled: set[int] = set()
        release_first = asyncio.Event()

        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        "responses_create_params": {"input": []},
                        AGENT_REF_KEY_NAME: {"name": "agent"},
                        "case": i,
                    }
                )
                for i in range(offered)
            )
            + "\n"
        )

        class Helper(RolloutCollectionHelper):
            async def _post(self, row, semaphore=None, *, route_failures_to_sidecar=False):
                del semaphore, route_failures_to_sidecar
                case = row["case"]
                started.add(case)
                try:
                    if case == 0:
                        await release_first.wait()
                        return _CompletedRollout(
                            row=row,
                            result={"reward": 1.0, "not_serializable": object()},
                            rollout_latency_ms=None,
                        )

                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")
                except asyncio.CancelledError:
                    cancelled.add(case)
                    raise

            def _run_examples_with_metadata(
                self,
                examples,
                semaphore=None,
                *,
                route_failures_to_sidecar=False,
                max_resident_tasks=None,
                interleave_by_agent=False,
                dispatch_budget_s=None,
                drain_margin_s=None,
                latency_tracker=None,
            ):
                awaitables = map(
                    lambda row: self._post(
                        row,
                        semaphore,
                        route_failures_to_sidecar=route_failures_to_sidecar,
                    ),
                    examples,
                )
                return nemo_gym.rollout_collection._BoundedCompletionIterator(
                    awaitables,
                    max_resident_tasks=max_resident_tasks,
                    total=len(examples),
                )

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            max_resident_rollout_tasks=window,
            disable_aggregation=True,
            disable_health_check=True,
            upload_rollouts=True,
        )

        monkeypatch.setattr(nemo_gym.rollout_collection, "get_exporters", lambda: [object()])
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "export_rollouts",
            lambda rows: pytest.fail("collection failure must prevent upload"),
        )

        async def release_after_window_is_resident():
            while len(started) < window:
                await asyncio.sleep(0)
            release_first.set()

        releaser = asyncio.create_task(release_after_window_is_resident())
        try:
            with pytest.raises(TypeError):
                await asyncio.wait_for(Helper().run_from_config(config), timeout=5)
        finally:
            releaser.cancel()

        assert 0 in started
        assert len(started) <= window + 1
        assert cancelled == started - {0}
        assert len(started) < offered
        assert not (tmp_path / "output.jsonl.upload.tmp").exists()

    async def test_run_from_config_non_retaining_resume_counts_cached_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        output_path = tmp_path / "output.jsonl"
        rows = [
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "x": i,
            }
            for i in range(3)
        ]
        materialized_path = tmp_path / "output_materialized_inputs.jsonl"
        materialized_path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
        output_path.write_bytes(orjson.dumps({**rows[0], "reward": 1.0}) + b"\n")
        failure_path = _failures_path_for(output_path)
        failure_path.write_bytes(
            orjson.dumps(
                {
                    **rows[1],
                    NG_FAILURE_CLASS_KEY: AGENT_RUN_ERROR_FAILURE_CLASS,
                    NG_TERMINAL_KEY: True,
                }
            )
            + b"\n"
        )

        dispatched = []
        aggregated = []

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                dispatched.append(json["x"])
                return FakeResponse(200, {"reward": 1.0})
            aggregated.extend(dict(row) for row in json.verify_responses)
            return FakeResponse(200, compute_aggregate_metrics(aggregated).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_path),
            resume_from_cache=True,
            retain_results_in_memory=False,
            route_failures_to_sidecar=True,
            count_failure_classes_as_zero=[AGENT_RUN_ERROR_FAILURE_CLASS],
            disable_health_check=True,
        )

        assert await RolloutCollectionHelper().run_from_config(config) == []
        assert dispatched == [2]
        assert sorted(row[TASK_INDEX_KEY_NAME] for row in aggregated) == [0, 1, 2]
        assert sorted(row["reward"] for row in aggregated) == [0.0, 1.0, 1.0]
        assert [orjson.loads(line)[TASK_INDEX_KEY_NAME] for line in output_path.read_bytes().splitlines()] == [0, 2]
        assert failure_path.read_bytes().splitlines() == [
            orjson.dumps(
                {
                    **rows[1],
                    NG_FAILURE_CLASS_KEY: AGENT_RUN_ERROR_FAILURE_CLASS,
                    NG_TERMINAL_KEY: True,
                }
            )
        ]

    async def test_run_from_config_never_exceeds_resident_rollout_task_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        offered, window = 64, 4
        baseline = len(asyncio.all_tasks())
        peak_resident = 0

        class Response:
            status = 200
            ok = True

            def release(self) -> None:
                pass

        async def post(*, json, **kwargs):
            nonlocal peak_resident
            peak_resident = max(peak_resident, len(asyncio.all_tasks()) - baseline)
            for _ in range(json["i"] % 3):
                await asyncio.sleep(0)
            return Response()

        client = MagicMock()
        client.post = post
        client.global_config_dict = OmegaConf.create(
            {
                "agent": {"responses_api_agents": {"impl": {}}},
                "environment": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent"}}}},
            }
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "setup_server_client_utils", lambda *a, **k: client)
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_response_json",
            AsyncMock(side_effect=lambda response: {"reward": 1.0}),
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", lambda: {})

        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        "responses_create_params": {"input": []},
                        AGENT_REF_KEY_NAME: {"name": "agent"},
                        "i": i,
                    }
                )
                for i in range(offered)
            )
            + "\n"
        )
        output_fpath = tmp_path / "output.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            max_resident_rollout_tasks=window,
            disable_aggregation=True,
            disable_health_check=True,
        )

        results = await asyncio.wait_for(RolloutCollectionHelper().run_from_config(config), timeout=10)
        assert len(results) == offered
        assert len(output_fpath.read_bytes().splitlines()) == offered
        assert peak_resident == window

    @pytest.mark.parametrize("retain", [True, False])
    @pytest.mark.parametrize("with_batch_manifest", [True, False])
    async def test_run_from_config_releases_completed_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        retain: bool,
        with_batch_manifest: bool,
    ) -> None:
        offered = 16
        produced = []
        peak_live_completed = 0

        class TrackedResult(dict):
            pass

        class Response:
            status = 200
            ok = True

            def release(self) -> None:
                pass

        async def post(*, json, **kwargs):
            nonlocal peak_live_completed
            gc.collect()
            peak_live_completed = max(
                peak_live_completed,
                sum(ref() is not None for ref in produced),
            )
            return Response()

        async def get_json(response):
            result = TrackedResult(reward=1.0)
            produced.append(weakref.ref(result))
            return result

        client = MagicMock()
        client.post = post
        client.global_config_dict = OmegaConf.create(
            {
                "agent": {"responses_api_agents": {"impl": {}}},
                "environment": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent"}}}},
            }
        )
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "setup_server_client_utils",
            lambda *a, **k: client,
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", get_json)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", lambda: {})

        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        "responses_create_params": {"input": []},
                        AGENT_REF_KEY_NAME: {"name": "agent"},
                        TASK_SOURCE_KEY_NAME: "benchmark",
                        "i": i,
                    }
                )
                for i in range(offered)
            )
            + "\n"
        )
        output_fpath = tmp_path / "output.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            max_resident_rollout_tasks=1,
            retain_results_in_memory=retain,
            disable_aggregation=True,
            disable_health_check=True,
        )

        if with_batch_manifest:
            materialized_rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
            observed = observe_materialized_rows(materialized_rows)["agent"]
            manifest_fpath = tmp_path / "batch_manifest.json"
            manifest_fpath.write_bytes(
                orjson.dumps(
                    {
                        "schema_version": "1",
                        "members": {
                            "benchmark": {
                                "agent_name": "agent",
                                "task_sources": observed.task_sources,
                                "dataset_sha256": observed.dataset_sha256,
                                "expected_task_count": observed.task_count,
                                "expected_rollout_count": observed.rollout_count,
                                "repeat_policy": observed.repeat_policy.model_dump(),
                                "resolved_recipe_sha256": "a" * 64,
                                "metric_keys": ["mean/reward"],
                            }
                        },
                    }
                )
            )
            config.batch_manifest_fpath = str(manifest_fpath)

        results = await RolloutCollectionHelper().run_from_config(config)
        assert len(output_fpath.read_bytes().splitlines()) == offered
        if with_batch_manifest:
            status = orjson.loads((tmp_path / "batch_status.json").read_bytes())["members"]["agent"]
            assert status["completed_rollout_count"] == offered
            assert status["remaining_rollout_count"] == 0
            assert status["aggregation_status"] == "deferred"
        if retain:
            assert len(results) == offered
            assert peak_live_completed == offered - 1
        else:
            assert results == []
            assert peak_live_completed <= 1

    async def test_run_from_config_non_retaining_preserves_upload_semantics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
    ) -> None:
        input_fpath = tmp_path / "input-upload.jsonl"
        input_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        "responses_create_params": {"input": []},
                        AGENT_REF_KEY_NAME: {"name": "agent"},
                        "case": case,
                    }
                )
                for case in ("success", "failure", "no-persist")
            )
            + "\n"
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                results_by_case = {
                    "success": {"reward": 1.0, "case": "success"},
                    "failure": {
                        "reward": 0.0,
                        "case": "failure",
                        NG_FAILURE_CLASS_KEY: "verify_failed",
                    },
                    "no-persist": {
                        "reward": 0.0,
                        "case": "no-persist",
                        NG_NO_PERSIST_KEY: True,
                    },
                }
                futures = []
                for row in examples:
                    future = Future()
                    future.set_result(
                        _CompletedRollout(
                            row=row,
                            result=results_by_case[row["case"]].copy(),
                            rollout_latency_ms=None,
                        )
                    )
                    futures.append(future)
                return futures

        exported: list[list[dict]] = []

        monkeypatch.setattr(nemo_gym.rollout_collection, "get_exporters", lambda: [object()])
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "export_rollouts",
            lambda rows: exported.append(rows),
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "export_metrics", lambda *args, **kwargs: None)

        async def run(*, retain: bool, output_fpath: Path, resume: bool = False) -> list[dict]:
            config = RolloutCollectionConfig(
                input_jsonl_fpath=str(input_fpath),
                output_jsonl_fpath=str(output_fpath),
                retain_results_in_memory=retain,
                route_failures_to_sidecar=True,
                disable_aggregation=True,
                disable_health_check=True,
                upload_rollouts=True,
                resume_from_cache=resume,
            )
            if resume:
                materialized_rows = [
                    {
                        "responses_create_params": {"input": []},
                        AGENT_REF_KEY_NAME: {"name": "agent"},
                        "case": case,
                        TASK_INDEX_KEY_NAME: task_index,
                        ROLLOUT_INDEX_KEY_NAME: 0,
                    }
                    for task_index, case in enumerate(("success", "failure", "no-persist"))
                ]
                config.materialized_jsonl_fpath.write_bytes(
                    b"\n".join(orjson.dumps(row) for row in materialized_rows) + b"\n"
                )
                output_fpath.write_bytes(
                    orjson.dumps(
                        {
                            **materialized_rows[0],
                            "reward": 1.0,
                        }
                    )
                    + b"\n"
                )
            await Helper().run_from_config(config)
            assert len(exported) == 1
            return exported.pop()

        retaining = await run(retain=True, output_fpath=tmp_path / "retaining-upload.jsonl")
        non_retaining = await run(retain=False, output_fpath=tmp_path / "non-retaining-upload.jsonl")

        assert non_retaining == retaining
        assert [result["case"] for result in non_retaining] == [
            "success",
            "failure",
            "no-persist",
        ]
        assert not (tmp_path / "non-retaining-upload.jsonl.upload.tmp").exists()
        assert [
            orjson.loads(line)["case"] for line in (tmp_path / "non-retaining-upload.jsonl").read_bytes().splitlines()
        ] == ["success"]

        retaining_resume = await run(
            retain=True,
            output_fpath=tmp_path / "retaining-resume-upload.jsonl",
            resume=True,
        )
        non_retaining_resume = await run(
            retain=False,
            output_fpath=tmp_path / "non-retaining-resume-upload.jsonl",
            resume=True,
        )

        assert non_retaining_resume == retaining_resume

    async def test_run_from_config_non_retaining_preserves_artifacts_without_reloading_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
    ) -> None:
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            "\n".join(
                json.dumps(
                    {
                        "responses_create_params": {"input": []},
                        AGENT_REF_KEY_NAME: {"name": "my agent name"},
                        "case": case,
                    }
                )
                for case in ("success", "failure", "no-persist")
            )
            + "\n"
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples: list[dict], *args, **kwargs):
                results_by_case = {
                    "success": {"reward": 1.0, "case": "success"},
                    "failure": {
                        "reward": 0.0,
                        "case": "failure",
                        NG_FAILURE_CLASS_KEY: "verify_failed",
                    },
                    "no-persist": {
                        "reward": 0.0,
                        "case": "no-persist",
                        NG_NO_PERSIST_KEY: True,
                    },
                }
                futures = []
                for row in examples:
                    future = Future()
                    future.set_result(
                        _CompletedRollout(
                            row=row,
                            result=results_by_case[row["case"]].copy(),
                            rollout_latency_ms=None,
                        )
                    )
                    futures.append(future)
                return futures

        async def run(*, retain: bool, output_fpath: Path):
            config = RolloutCollectionConfig(
                input_jsonl_fpath=str(input_fpath),
                output_jsonl_fpath=str(output_fpath),
                retain_results_in_memory=retain,
                route_failures_to_sidecar=True,
                disable_aggregation=True,
                disable_health_check=True,
                upload_rollouts=False,
            )
            return await Helper().run_from_config(config)

        retaining_output = tmp_path / "retaining.jsonl"
        retaining_results = await run(retain=True, output_fpath=retaining_output)
        retaining_main = retaining_output.read_bytes()
        retaining_failures = _failures_path_for(retaining_output).read_bytes()

        def fail_if_reloaded(path: Path):
            raise AssertionError(f"artifact-only non-retaining run reloaded {path}")

        monkeypatch.setattr(nemo_gym.rollout_collection, "_read_jsonl", fail_if_reloaded)

        non_retaining_output = tmp_path / "non-retaining.jsonl"
        non_retaining_results = await run(retain=False, output_fpath=non_retaining_output)

        assert [result["case"] for result in retaining_results] == [
            "success",
            "failure",
            "no-persist",
        ]
        assert non_retaining_results == []
        assert non_retaining_output.read_bytes() == retaining_main
        assert _failures_path_for(non_retaining_output).read_bytes() == retaining_failures

        main_rows = [orjson.loads(line) for line in retaining_main.splitlines()]
        failure_rows = [orjson.loads(line) for line in retaining_failures.splitlines()]
        assert [row["case"] for row in main_rows] == ["success"]
        assert [row["case"] for row in failure_rows] == ["failure"]
        assert b"no-persist" not in retaining_main
        assert b"no-persist" not in retaining_failures

    async def test_run_from_config_non_retaining_runs_aggregation_and_health(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
    ) -> None:
        import nemo_gym.rollout_health

        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "agent"}}) + "\n"
        )
        output_fpath = tmp_path / "output.jsonl"
        aggregated = {}

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                future = Future()
                future.set_result(
                    _CompletedRollout(
                        row=examples[0],
                        result={"reward": 1.0, "case": "success"},
                        rollout_latency_ms=None,
                    )
                )
                return [future]

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
                aggregated["results"] = results
                aggregated["rows"] = rows
                return None

        health_result = object()
        run_health_checks = MagicMock(return_value=health_result)
        format_health_report = MagicMock(return_value="health checks passed")
        monkeypatch.setattr(nemo_gym.rollout_health, "run_health_checks", run_health_checks)
        monkeypatch.setattr(nemo_gym.rollout_health, "format_health_report", format_health_report)

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            retain_results_in_memory=False,
        )

        returned = await Helper().run_from_config(config)

        assert returned == []
        assert [result["case"] for result in aggregated["results"]] == ["success"]
        assert aggregated["rows"] == aggregated["results"]
        run_health_checks.assert_called_once_with(output_fpath, workers=None, ignored_checks=[])
        format_health_report.assert_called_once_with(health_result)

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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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
                    "num_repeats": 1,
                },
                "key_metrics": {"mean/reward": 1.0},
                "group_level_metrics": actual_aggregate_metrics[0]["group_level_metrics"],
                "perf_summary": None,
                "repeat_level_metrics": [],
            }
        ]
        assert expected_aggregate_metrics == actual_aggregate_metrics

    async def test_fresh_run_clears_stale_failure_and_metrics_artifacts(
        self, tmp_path: Path, empty_global_config: MagicMock
    ) -> None:
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
        output_fpath = tmp_path / "output.jsonl"
        output_fpath.write_text('{"stale":true}\n')
        failures_fpath = _failures_path_for(output_fpath)
        failures_fpath.write_text(
            orjson.dumps(
                {
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    NG_FAILURE_CLASS_KEY: "skipped",
                    NG_TERMINAL_KEY: True,
                }
            ).decode()
            + "\n"
        )
        metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        metrics_fpath.write_text('{"stale":true}\n')

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(output_fpath),
            resume_from_cache=False,
            disable_aggregation=True,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                future = Future()
                future.set_result(_CompletedRollout(row=examples[0], result={"reward": 1.0}, rollout_latency_ms=None))
                return [future]

        await Helper().run_from_config(config)

        assert [orjson.loads(line) for line in output_fpath.read_bytes().splitlines()] == [
            {
                "reward": 1.0,
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                AGENT_REF_KEY_NAME: {"name": "agent"},
            }
        ]
        assert failures_fpath.read_bytes() == b""
        assert not metrics_fpath.exists()

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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

    async def test_run_from_config_closes_owned_token_source_on_processing_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed = False
        original_close = TokenCaptureStore.close

        async def tracked_close(store):
            nonlocal closed
            closed = True
            await original_close(store)

        monkeypatch.setattr(TokenCaptureStore, "close", tracked_close)
        monkeypatch.setattr(nemo_gym.rollout_collection, "installed_token_source", lambda: None)
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_global_config_dict",
            lambda: {
                "token_id_capture": {
                    "enabled": True,
                    "all_agents": True,
                    "dir": str(tmp_path / "captures"),
                    "rebuild_response": True,
                    "lineage_store": f"{__name__}:_StubLineageStore",
                }
            },
        )

        input_fpath = tmp_path / "input-owned-source.jsonl"
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
            output_jsonl_fpath=str(tmp_path / "output-owned-source.jsonl"),
            resume_from_cache=False,
            disable_aggregation=True,
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                [example] = examples
                future = Future()
                future.set_result(
                    _CompletedRollout(
                        row=example,
                        result={"not_serializable": object()},
                        rollout_latency_ms=None,
                    )
                )
                return [future]

        with pytest.raises(TypeError):
            await Helper().run_from_config(config)

        assert closed is True

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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                name: block
                for agent in ("agent_a", "agent_b", "my_agent")
                for name, block in (
                    (agent, {"responses_api_agents": {"impl": {}}}),
                    (
                        f"{agent}_environment_server",
                        {"environment_servers": {"legacy_agent": {"agent_server": {"name": agent}}}},
                    ),
                )
            }
        )

        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        helper = RolloutCollectionHelper()

        rows = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1},
        ]
        results = [
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                "reward": 1.0,
                "response": {
                    "usage": {"tokens": 10},
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
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
        assert sent_data[0]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}

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
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                name: block
                for agent in ("agent_a", "agent_b", "my_agent")
                for name, block in (
                    (agent, {"responses_api_agents": {"impl": {}}}),
                    (
                        f"{agent}_environment_server",
                        {"environment_servers": {"legacy_agent": {"agent_server": {"name": agent}}}},
                    ),
                )
            }
        )
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
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                name: block
                for agent in ("agent_a", "agent_b", "my_agent")
                for name, block in (
                    (agent, {"responses_api_agents": {"impl": {}}}),
                    (
                        f"{agent}_environment_server",
                        {"environment_servers": {"legacy_agent": {"agent_server": {"name": agent}}}},
                    ),
                )
            }
        )
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
            agg = agg_a if server_name == "agent_a_environment_server" else agg_b
            resp = AsyncMock()
            resp.raise_for_status = MagicMock()
            resp.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
            resp.status = 200
            return resp

        mock_server_client = MagicMock()
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                name: block
                for agent in ("agent_a", "agent_b", "my_agent")
                for name, block in (
                    (agent, {"responses_api_agents": {"impl": {}}}),
                    (
                        f"{agent}_environment_server",
                        {"environment_servers": {"legacy_agent": {"agent_server": {"name": agent}}}},
                    ),
                )
            }
        )
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

    @pytest.mark.parametrize("raise_on_error", [None, True, False], ids=["default", "raise", "defer"])
    @pytest.mark.parametrize("include_successful_agent", [False, True], ids=["single-agent", "mixed-agents"])
    async def test_call_aggregate_metrics_isolates_one_agent_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        raise_on_error: bool | None,
        include_successful_agent: bool,
    ) -> None:
        """Direct callers receive the original exception after all metrics are saved; deferral returns normally."""
        original_error = http_error(500)
        failed_agent_called = asyncio.Event()

        async def post(server_name: str, url_path: str, json: AggregateMetricsRequest) -> FakeResponse:
            assert url_path == "/aggregate_metrics"
            if server_name == "agent_b_environment_server":
                failed_agent_called.set()
                raise original_error
            await failed_agent_called.wait()
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("agent_a", "agent_b"))
        rows = [
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
        ]
        results = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0},
        ]

        if not include_successful_agent:
            rows, results = rows[1:], results[1:]

        metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        kwargs = {} if raise_on_error is None else {"raise_on_error": raise_on_error}
        expected_error = pytest.raises(ClientResponseError) if raise_on_error is not False else nullcontext()
        with expected_error as exc_info:
            returned = await RolloutCollectionHelper()._call_aggregate_metrics(
                results, rows, tmp_path / "output.jsonl", **kwargs
            )
            assert returned == metrics_fpath
        if raise_on_error is not False:
            assert exc_info.value is original_error
            assert exc_info.traceback[-1].name == "post"

        written = orjson.loads(metrics_fpath.read_bytes())
        expected_agents = ["agent_a", "agent_b"] if include_successful_agent else ["agent_b"]
        assert [entry[AGENT_REF_KEY_NAME]["name"] for entry in written] == expected_agents
        if include_successful_agent:
            assert written[0]["key_metrics"]["mean/reward"] == 1.0
            assert AGGREGATION_ERROR_KEY not in written[0]
        assert written[-1] == {
            AGENT_REF_KEY_NAME: {"name": "agent_b"},
            NG_ENVIRONMENT_SERVER_KEY: "agent_b_environment_server",
            "agent_metrics": {},
            "key_metrics": {},
            "group_level_metrics": [],
            "repeat_level_metrics": [],
            AGGREGATION_ERROR_KEY: {
                "type": "ClientResponseError",
                "message": "500, message='boom', url='http://agent/run'",
                "http_status": 500,
            },
        }
        assert "Aggregate-metrics request failed for agent 'agent_b'" in caplog.text

    @pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
    async def test_call_aggregate_metrics_records_unexpected_agent_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
    ) -> None:
        """Non-HTTP failures retain their exception identity and saved diagnostics too."""
        original_error = error_type("aggregator bug")
        install_fake_server_client(monkeypatch, AsyncMock(side_effect=original_error), agent_names=("agent_a",))
        rows = [{AGENT_REF_KEY_NAME: {"name": "agent_a"}}]

        with pytest.raises(error_type, match="aggregator bug") as exc_info:
            await RolloutCollectionHelper()._call_aggregate_metrics([{"reward": 1.0}], rows, tmp_path / "output.jsonl")
        assert exc_info.value is original_error

        metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        written = orjson.loads(metrics_fpath.read_bytes())
        assert written[0][AGGREGATION_ERROR_KEY]["type"] == error_type.__name__
        assert written[0][AGGREGATION_ERROR_KEY]["message"] == "aggregator bug"

    async def test_call_aggregate_metrics_raises_first_error_after_recording_all_failures(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Later failures do not replace the original exception or disappear from saved metrics."""
        first_error = http_error(500)
        second_error = ValueError("invalid metrics")
        first_agent_called = asyncio.Event()

        async def post(server_name: str, **kwargs) -> FakeResponse:
            if server_name == "agent_a_environment_server":
                first_agent_called.set()
                raise first_error
            await first_agent_called.wait()
            raise second_error

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("agent_a", "agent_b"))
        rows = [{AGENT_REF_KEY_NAME: {"name": agent}} for agent in ("agent_a", "agent_b")]

        with pytest.raises(ClientResponseError) as exc_info:
            await RolloutCollectionHelper()._call_aggregate_metrics(
                [{"reward": 1.0}, {"reward": 0.0}], rows, tmp_path / "output.jsonl"
            )
        assert exc_info.value is first_error

        written = orjson.loads((tmp_path / "output_aggregate_metrics.json").read_bytes())
        assert [entry[AGENT_REF_KEY_NAME]["name"] for entry in written] == ["agent_a", "agent_b"]
        assert [entry[AGGREGATION_ERROR_KEY]["type"] for entry in written] == ["ClientResponseError", "ValueError"]
        assert written[0][AGGREGATION_ERROR_KEY]["http_status"] == 500
        assert written[1][AGGREGATION_ERROR_KEY]["message"] == "invalid metrics"

    async def test_call_aggregate_metrics_builds_the_agent_server_map_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unstamped flat rows resolve their server from one map instead of scanning the config per row."""
        agg = AggregateMetrics(agent_metrics={"mean/reward": 1.0}, key_metrics={}, group_level_metrics=[])
        mock_response = AsyncMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        mock_response.status = 200
        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=mock_response)
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
            }
        )
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: mock_server_client
        )
        build_calls: list[int] = []
        build_map = nemo_gym.rollout_collection._environment_servers_by_agent
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "_environment_servers_by_agent",
            lambda config: build_calls.append(1) or build_map(config),
        )
        rows = [
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0}
            for i in range(100)
        ]
        results = [{TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0} for i in range(100)]

        await RolloutCollectionHelper()._call_aggregate_metrics(results, rows, tmp_path / "output.jsonl")

        assert build_calls == [1]
        assert mock_server_client.post.await_args.kwargs["server_name"] == "my_environment_server"

    async def test_call_aggregate_metrics_empty(self, tmp_path: Path) -> None:
        """_call_aggregate_metrics returns None for empty results."""
        helper = RolloutCollectionHelper()
        output_fpath = tmp_path / "output.jsonl"
        result = await helper._call_aggregate_metrics([], [], output_fpath)
        assert result is None


class TestDispatchBudget:
    """A task that cannot finish before the deadline must not be started."""

    @staticmethod
    def _row(task_index: int) -> dict:
        return {
            AGENT_REF_KEY_NAME: {"name": "my_agent"},
            TASK_INDEX_KEY_NAME: task_index,
            ROLLOUT_INDEX_KEY_NAME: 0,
        }

    def _helper(self, posted: list) -> RolloutCollectionHelper:
        async def post(server_name, url_path, json):  # noqa: A002
            posted.append(json[TASK_INDEX_KEY_NAME])
            return MagicMock(status=200)

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(side_effect=post)
        # Pre-dispatch validation reads the running config; the fake agent must
        # exist there as an agent instance or run_examples refuses the rows.
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
            }
        )

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                return mock_server_client

        return MockHelper()

    async def test_no_budget_dispatches_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        posted: list = []
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))

        rows = [self._row(i) for i in range(3)]
        results = [await future for future in self._helper(posted).run_examples(rows)]

        assert sorted(posted) == [0, 1, 2]
        assert all(NG_NO_PERSIST_KEY not in result for _row, result in results)

    async def test_exhausted_budget_drains_without_dispatching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        posted: list = []
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))

        tracker = DispatchLatencyTracker()
        rows = [self._row(i) for i in range(3)]
        results = [
            await future
            for future in self._helper(posted).run_examples(rows, dispatch_budget_s=-1.0, latency_tracker=tracker)
        ]

        assert posted == [], "no /run request may be sent once the budget is spent"
        assert tracker.drained == 3
        for _row, result in results:
            # no_persist means no row is written anywhere, so the task is
            # re-dispatched intact on the next resume.
            assert result[NG_NO_PERSIST_KEY] is True
            assert result[NG_DISPATCH_DRAINED_KEY] is True
            assert result[NG_FAILURE_CLASS_KEY] == CANCELLED
            assert NG_TERMINAL_KEY not in result
            assert "not dispatched" in result["error_message"]

    async def test_bounded_admission_still_honors_the_budget_and_input_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`max_resident_tasks` admits rows in order, and a spent budget still drains every one."""
        posted: list = []
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))

        rows = [self._row(i) for i in (4, 2, 7, 1)]
        semaphore = asyncio.Semaphore(1)
        results = [
            await future
            for future in self._helper(posted).run_examples(rows, semaphore=semaphore, max_resident_tasks=2)
        ]
        assert posted == [4, 2, 7, 1], "a single slot must start rows in the order given"
        assert len(results) == 4

        posted.clear()
        tracker = DispatchLatencyTracker()
        drained = [
            await future
            for future in self._helper(posted).run_examples(
                rows, semaphore=semaphore, max_resident_tasks=2, dispatch_budget_s=-1.0, latency_tracker=tracker
            )
        ]
        assert posted == []
        assert tracker.drained == 4
        assert all(result[NG_DISPATCH_DRAINED_KEY] for _row, result in drained)

    async def test_drain_margin_blocks_a_task_that_cannot_finish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        posted: list = []
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))

        tracker = DispatchLatencyTracker()
        results = [
            await future
            for future in self._helper(posted).run_examples(
                [self._row(0)],
                dispatch_budget_s=60.0,
                drain_margin_s=3600.0,
                latency_tracker=tracker,
            )
        ]

        assert posted == []
        assert tracker.drained == 1
        assert "below the 3600s drain margin" in results[0][1]["error_message"]

    @pytest.mark.parametrize(
        "drain_margin_s, expected",
        [
            (None, "observed p75 task duration"),
            (600.0, "under 10 min left"),
            (0.0, "under 0 min left"),
        ],
    )
    async def test_run_from_config_announces_the_budget(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        empty_global_config: MagicMock,
        drain_margin_s: float | None,
        expected: str,
    ) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}}) + "\n"
        )
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            disable_aggregation=True,
            num_samples_in_parallel=1,
            dispatch_budget_s=1800.0,
            drain_margin_s=drain_margin_s,
        )

        class MockHelper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    future.set_result(_CompletedRollout(row=example, result={"reward": 1.0}, rollout_latency_ms=None))
                    futures.append(future)
                return futures

        await MockHelper().run_from_config(config)

        out = capsys.readouterr().out
        assert "Dispatch budget: 30 min" in out
        assert expected in out
        # run_examples is stubbed here, so nothing was timed; the summary still
        # prints rather than being skipped.
        assert "No completed rollouts to report latency for." in out

    def test_dispatch_budget_requires_finite_concurrency(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="num_samples_in_parallel or max_resident_rollout_tasks"):
            RolloutCollectionConfig(
                input_jsonl_fpath=str(tmp_path / "input.jsonl"),
                output_jsonl_fpath=str(tmp_path / "output.jsonl"),
                dispatch_budget_s=1800.0,
            )

    def test_dispatch_budget_accepts_max_resident_rollout_tasks_without_num_samples_in_parallel(
        self, tmp_path: Path
    ) -> None:
        RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            dispatch_budget_s=1800.0,
            max_resident_rollout_tasks=4,
        )

    @pytest.mark.parametrize("concurrency", [0, -1])
    def test_concurrency_must_always_be_positive(self, tmp_path: Path, concurrency: int) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            RolloutCollectionConfig(
                input_jsonl_fpath=str(tmp_path / "input.jsonl"),
                output_jsonl_fpath=str(tmp_path / "output.jsonl"),
                num_samples_in_parallel=concurrency,
            )

    async def test_zero_budget_drains_rather_than_disabling_the_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """0.0 is a spent budget, not an absent one -- truthiness gets this wrong."""
        posted: list = []
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))

        tracker = DispatchLatencyTracker()
        results = [
            await future
            for future in self._helper(posted).run_examples(
                [self._row(0)], dispatch_budget_s=0.0, latency_tracker=tracker
            )
        ]

        assert posted == []
        assert tracker.drained == 1
        assert results[0][1][NG_NO_PERSIST_KEY] is True

    async def test_generous_budget_still_dispatches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        posted: list = []
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))

        tracker = DispatchLatencyTracker()
        results = [
            await future
            for future in self._helper(posted).run_examples(
                [self._row(0)], dispatch_budget_s=3600.0, drain_margin_s=1.0, latency_tracker=tracker
            )
        ]

        assert posted == [0]
        assert tracker.drained == 0
        # run_examples returns the raw /run result; latency lives on _CompletedRollout.
        assert results[0][1] == {"reward": 1.0}
        # The completed rollout is timed, so the adaptive margin has a basis.
        assert tracker.quantile(0.5) is not None

    @pytest.mark.parametrize(("preloaded_s", "recorded"), [(3600.0, False), (1e-9, True)])
    async def test_a_failed_run_feeds_the_margin_only_when_it_ran_long(
        self, monkeypatch: pytest.MonkeyPatch, preloaded_s: float, recorded: bool
    ) -> None:
        """A failure counts toward the adaptive margin only when it is at least as long as the median."""
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock(side_effect=http_error(502)))
        tracker = DispatchLatencyTracker()
        for _ in range(5):
            tracker.record(preloaded_s)

        results = [
            await future
            for future in self._helper([])._run_examples_with_metadata(
                [self._row(0)],
                dispatch_budget_s=3600.0,
                drain_margin_s=1.0,
                latency_tracker=tracker,
                route_failures_to_sidecar=True,
            )
        ]

        assert results[0].result[NG_FAILURE_CLASS_KEY] is not None
        assert (len(tracker._durations) == 6) is recorded

    @pytest.mark.parametrize("field", ["dispatch_budget_s", "drain_margin_s"])
    def test_budget_and_margin_cannot_be_negative(self, tmp_path: Path, field: str) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            RolloutCollectionConfig(
                input_jsonl_fpath=str(tmp_path / "input.jsonl"),
                output_jsonl_fpath=str(tmp_path / "output.jsonl"),
                num_samples_in_parallel=1,
                **{field: -1.0},
            )

    @staticmethod
    def _write_inputs(tmp_path: Path, count: int) -> Path:
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_bytes(
            b"".join(
                orjson.dumps(
                    {"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "my_agent"}, "x": i}
                )
                + b"\n"
                for i in range(count)
            )
        )
        return input_fpath

    async def test_drained_rows_skip_token_capture_and_progress_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A row the budget never started is not a rollout.

        When the budget runs out the whole queue drains at once. Treated as rollouts, those rows
        reached token finalization as masked "nothing recorded" captures and could trip
        max_mask_fraction, aborting the in-flight rollouts the budget exists to protect.
        """
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
                    "max_mask_fraction": 0.5,
                    "mask_fraction_min_samples": 1,
                }
            },
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "installed_token_source", lambda: object())
        finalized: list[dict] = []

        async def finalize(result: dict, source: object, **kwargs: object) -> dict:
            finalized.append(result)
            # The real finalizer masks a rollout that recorded nothing, as a drained row would.
            return {MASK_SAMPLE_KEY: NG_DISPATCH_DRAINED_KEY in result, "metrics": {}}

        monkeypatch.setattr(nemo_gym.rollout_collection, "finalize_rollout_token_capture", finalize)

        # Each /run takes 25 s of a 60 s budget: three rows start and the other four drain.
        clock = SimpleNamespace(now=0.0)
        real_time = nemo_gym.rollout_collection.time
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "time", SimpleNamespace(monotonic=lambda: clock.now, time=real_time.time)
        )

        async def post(server_name: str, url_path: str, json: dict, **kwargs):
            clock.now += 25.0
            return FakeResponse(200, {"reward": 1.0})

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        output_fpath = tmp_path / "output.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(self._write_inputs(tmp_path, 7)),
            output_jsonl_fpath=str(output_fpath),
            disable_aggregation=True,
            num_samples_in_parallel=1,
            dispatch_budget_s=60.0,
        )

        results = await RolloutCollectionHelper().run_from_config(config)

        assert len(finalized) == 3
        assert not any(NG_DISPATCH_DRAINED_KEY in result for result in finalized)
        assert sum(bool(result.get(NG_DISPATCH_DRAINED_KEY)) for result in results) == 4
        persisted = [orjson.loads(line) for line in output_fpath.read_bytes().splitlines()]
        assert [row[TASK_INDEX_KEY_NAME] for row in persisted] == [0, 1, 2]
        assert _failures_path_for(output_fpath).read_bytes() == b""
        out = capsys.readouterr().out
        assert "Found 3 / 7 " in out and "Found 4 / 7 " not in out
        assert "Drained (not dispatched, no time left in the budget): 4." in out

    @pytest.mark.parametrize("count_missing", [False, True], ids=["default", "missing counted as zero"])
    async def test_a_run_that_drained_everything_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock, count_missing: bool
    ) -> None:
        """With aggregation disabled this run scores nothing, so counting missing rows cannot rescue it."""
        posted: list = []

        async def post(server_name: str, url_path: str, json: dict, **kwargs):
            posted.append(json)
            return FakeResponse(200, {"reward": 1.0})

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(self._write_inputs(tmp_path, 2)),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            disable_aggregation=True,
            num_samples_in_parallel=1,
            dispatch_budget_s=0.0,
            count_missing_rollouts_as_zero=count_missing,
        )

        with pytest.raises(RuntimeError, match="2 of them were never started because dispatch_budget_s ran out"):
            await RolloutCollectionHelper().run_from_config(config)
        assert posted == []

    async def test_rows_the_budget_never_started_count_as_zero_until_resume_runs_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """A drained row writes nothing, so count_missing_rollouts_as_zero scores it like a lost one."""
        posted: list = []

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                posted.append(json)
                return FakeResponse(200, {"reward": 1.0})
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")

        def config(**overrides) -> RolloutCollectionConfig:
            return RolloutCollectionConfig(
                input_jsonl_fpath=str(self._write_inputs(tmp_path, 2)),
                output_jsonl_fpath=str(output_fpath),
                num_samples_in_parallel=1,
                count_missing_rollouts_as_zero=True,
                disable_health_check=True,
                **overrides,
            )

        await RolloutCollectionHelper().run_from_config(config(dispatch_budget_s=0.0))
        assert posted == []
        assert orjson.loads(metrics_fpath.read_bytes())[0]["key_metrics"] == {"mean/reward": 0.0}

        # The zeros lived in the metrics only, so the resumed run dispatches both rows and scores them.
        await RolloutCollectionHelper().run_from_config(config(resume_from_cache=True))
        assert len(posted) == 2
        assert orjson.loads(metrics_fpath.read_bytes())[0]["key_metrics"] == {"mean/reward": 1.0}


class TestDispatchOrderReachesTheServer:
    """List order is not dispatch order unless the tasks are scheduled in order.

    `asyncio.as_completed` builds its task set from `set(fs)`, so handing it bare
    coroutines scrambles them before any of them reaches the semaphore -- which
    made `dispatch_longest_first` a no-op while its unit test still passed.
    """

    async def test_rows_hit_the_server_in_list_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))

        posted: list = []

        async def post(server_name, url_path, json):  # noqa: A002
            posted.append(json[TASK_INDEX_KEY_NAME])
            await asyncio.sleep(0)
            return MagicMock(status=200)

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(side_effect=post)
        mock_server_client.global_config_dict = OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
            }
        )

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                return mock_server_client

        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for i in range(12)
        ]
        # Concurrency 1 so acquisition order is unambiguous.
        semaphore = asyncio.Semaphore(1)
        for future in MockHelper().run_examples(rows, semaphore=semaphore):
            await future

        assert posted == list(range(12))

    def test_rows_are_scheduled_when_iteration_starts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling run_examples schedules nothing, so it may be built outside the loop that consumes it."""
        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", AsyncMock())
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", AsyncMock(return_value={"reward": 1.0}))
        posted: list = []
        rows = [TestDispatchBudget._row(i) for i in range(3)]

        futures = TestDispatchBudget()._helper(posted).run_examples(rows)

        async def consume() -> list:
            return [await future for future in futures]

        results = asyncio.run(consume())

        assert posted == [0, 1, 2]
        assert [result for _row, result in results] == [{"reward": 1.0}] * 3


class TestLongestFirstDispatch:
    """On resume, known-long tasks go first so they don't straddle the boundary."""

    @staticmethod
    def _write_resume_state(tmp_path: Path, failures: list, **config_overrides: bool) -> RolloutCollectionConfig:
        materialized = [{TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "input": True} for i in range(4)]
        (tmp_path / "output_materialized_inputs.jsonl").write_bytes(
            b"\n".join(map(orjson.dumps, materialized)) + b"\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        output_jsonl_fpath.write_bytes(b"")

        failures_fpath = _failures_path_for(output_jsonl_fpath)
        if failures:
            failures_fpath.write_bytes(b"\n".join(map(orjson.dumps, failures)) + b"\n")

        return RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_jsonl_fpath),
            # Off by default upstream: opt in explicitly for these tests.
            dispatch_longest_first=True,
            **config_overrides,
        )

    def test_known_timings_lead_unseen_rows_keep_order(self, tmp_path: Path) -> None:
        # Task 2 ran longest, task 0 next; tasks 1 and 3 have never been timed.
        failures = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "elapsed_seconds": 600},
            {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, "elapsed_seconds": 9000},
        ]
        config = self._write_resume_state(tmp_path, failures)

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert [r[TASK_INDEX_KEY_NAME] for r in input_rows] == [2, 0, 1, 3]

    def test_longest_attempt_wins_when_a_task_failed_twice(self, tmp_path: Path) -> None:
        failures = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "elapsed_seconds": 30},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "elapsed_seconds": 9000},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "elapsed_seconds": 60},
        ]
        config = self._write_resume_state(tmp_path, failures)

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert [r[TASK_INDEX_KEY_NAME] for r in input_rows][:2] == [0, 1]

    def test_disabled_preserves_input_order(self, tmp_path: Path) -> None:
        failures = [{TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, "elapsed_seconds": 9000}]
        config = self._write_resume_state(tmp_path, failures)
        config.dispatch_longest_first = False

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert [r[TASK_INDEX_KEY_NAME] for r in input_rows] == [0, 1, 2, 3]

    def test_fresh_run_without_timings_is_untouched(self, tmp_path: Path) -> None:
        config = self._write_resume_state(tmp_path, [])

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert [r[TASK_INDEX_KEY_NAME] for r in input_rows] == [0, 1, 2, 3]

    def test_timeout_failures_are_retried_not_gated(self, tmp_path: Path) -> None:
        """A legacy terminal timeout is a statement about load, not the task."""
        failures = [
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "elapsed_seconds": 12600,
                NG_FAILURE_CLASS_KEY: "timeout_exceeded",
                NG_TERMINAL_KEY: True,
            }
        ]
        config = self._write_resume_state(tmp_path, failures, retry_terminal_timeouts=True)

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert 1 in [r[TASK_INDEX_KEY_NAME] for r in input_rows]

    def test_terminal_timeout_stays_gated_by_default(self, tmp_path: Path) -> None:
        """Without the opt-in, an agent that stamps its timeouts terminal keeps them gated."""
        failures = [
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "elapsed_seconds": 12600,
                NG_FAILURE_CLASS_KEY: "timeout_exceeded",
                NG_TERMINAL_KEY: True,
            },
            # Not stamped terminal: retried in both modes.
            {TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "timeout_exceeded"},
        ]
        config = self._write_resume_state(tmp_path, failures)

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert [r[TASK_INDEX_KEY_NAME] for r in input_rows] == [0, 2, 3]

    def test_retry_propagates_cached_deliverable_signal(self, tmp_path: Path) -> None:
        failures = [
            {
                TASK_INDEX_KEY_NAME: 2,
                ROLLOUT_INDEX_KEY_NAME: 0,
                NG_FAILURE_CLASS_KEY: "judge_invalid",
                "reuse_cached_deliverable": True,
            }
        ]
        config = self._write_resume_state(tmp_path, failures)

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)
        by_task = {row[TASK_INDEX_KEY_NAME]: row for row in input_rows}

        assert by_task[2]["reuse_cached_deliverable"] is True
        assert "reuse_cached_deliverable" not in by_task[1]

    def test_legacy_invalid_judge_main_row_migrates_to_retryable_sidecar(self, tmp_path: Path) -> None:
        config = self._write_resume_state(tmp_path, [], retry_invalid_judge_responses=True)
        output = Path(config.output_jsonl_fpath)
        deliverables_dir = tmp_path / "deliverables" / "task-2"
        deliverables_dir.mkdir(parents=True)
        (deliverables_dir / "answer.txt").write_text("answer")
        legacy_row = {
            TASK_INDEX_KEY_NAME: 2,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "invalid_judge_response": True,
            "reward": 0.0,
            "deliverables_dir": str(deliverables_dir),
        }
        output.write_bytes(orjson.dumps(legacy_row) + b"\n")

        input_rows, cached_rows, cached_results, _ = RolloutCollectionHelper()._load_from_cache(config)

        by_task = {row[TASK_INDEX_KEY_NAME]: row for row in input_rows}
        assert by_task[2]["reuse_cached_deliverable"] is True
        assert by_task[2][ATTEMPT_INDEX_KEY_NAME] == 1
        assert cached_rows == []
        assert cached_results == []
        assert output.read_bytes() == b""
        migrated = [orjson.loads(line) for line in _failures_path_for(output).read_bytes().splitlines()]
        assert len(migrated) == 1
        assert migrated[0][NG_FAILURE_CLASS_KEY] == "judge_invalid"
        assert migrated[0]["reuse_cached_deliverable"] is True

        # Simulate a crash after the sidecar append but before the main-file
        # replacement: replaying the same attempt must not append it twice.
        replayed_row = dict(legacy_row)
        replayed_row[ATTEMPT_INDEX_KEY_NAME] = 0
        output.write_bytes(orjson.dumps(replayed_row) + b"\n")
        RolloutCollectionHelper()._load_from_cache(config)
        assert len(_failures_path_for(output).read_bytes().splitlines()) == 1

    def test_migration_keeps_valid_rows_byte_for_byte_and_is_idempotent(self, tmp_path: Path) -> None:
        output = tmp_path / "output.jsonl"
        assert migrate_invalid_judge_main_rows(output) == 0

        valid = orjson.dumps({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "z": 1, "reward": 1.0})
        invalid = orjson.dumps(
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "invalid_judge_response": True, "reward": 0.0}
        )
        output.write_bytes(valid + b"\n\n" + invalid + b"\n")

        assert migrate_invalid_judge_main_rows(output) == 1
        assert output.read_bytes() == valid + b"\n"
        sidecar = _failures_path_for(output).read_bytes()
        assert [orjson.loads(line)[TASK_INDEX_KEY_NAME] for line in sidecar.splitlines()] == [1]

        assert migrate_invalid_judge_main_rows(output) == 0
        assert output.read_bytes() == valid + b"\n"
        assert _failures_path_for(output).read_bytes() == sidecar

    def test_interrupted_migration_keeps_the_main_file_and_resumes_idempotently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "output.jsonl"
        # An unusable deliverables_dir only means the policy output is not reused.
        legacy = orjson.dumps(
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "invalid_judge_response": True, "deliverables_dir": 1}
        )
        output.write_bytes(legacy + b"\n")

        def interrupted_replace(src, dst) -> None:
            raise OSError("interrupted before the main file was replaced")

        with monkeypatch.context() as patch:
            patch.setattr(nemo_gym.rollout_collection.os, "replace", interrupted_replace)
            with pytest.raises(OSError, match="interrupted"):
                migrate_invalid_judge_main_rows(output)

        # The retry record is durable first; the main file and directory are untouched.
        assert output.read_bytes() == legacy + b"\n"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["output.jsonl", "output_failures.jsonl"]
        [migrated] = [orjson.loads(line) for line in _failures_path_for(output).read_bytes().splitlines()]
        assert "reuse_cached_deliverable" not in migrated

        failures_fpath = _failures_path_for(output)
        failures_fpath.write_bytes(failures_fpath.read_bytes() + b"\n")
        assert migrate_invalid_judge_main_rows(output) == 1
        assert output.read_bytes() == b""
        assert [line for line in failures_fpath.read_bytes().splitlines() if line] == [orjson.dumps(migrated)]

    def test_legacy_invalid_judge_distinct_attempts_accumulate_and_gate(self, tmp_path: Path) -> None:
        config = self._write_resume_state(tmp_path, [], retry_invalid_judge_responses=True)
        output = Path(config.output_jsonl_fpath)
        legacy_rows = [
            {
                TASK_INDEX_KEY_NAME: 2,
                ROLLOUT_INDEX_KEY_NAME: 0,
                ATTEMPT_INDEX_KEY_NAME: attempt,
                "invalid_judge_response": True,
            }
            for attempt in range(_DEFAULT_MAX_ROLLOUT_ATTEMPTS)
        ]
        output.write_bytes(b"\n".join(map(orjson.dumps, legacy_rows)) + b"\n")

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        assert 2 not in [row[TASK_INDEX_KEY_NAME] for row in input_rows]
        migrated = [orjson.loads(line) for line in _failures_path_for(output).read_bytes().splitlines()]
        assert [row[ATTEMPT_INDEX_KEY_NAME] for row in migrated] == list(range(_DEFAULT_MAX_ROLLOUT_ATTEMPTS))

    def test_legacy_invalid_judge_same_attempt_in_different_stages_stays_distinct(self, tmp_path: Path) -> None:
        config = self._write_resume_state(tmp_path, [], retry_invalid_judge_responses=True)
        output = Path(config.output_jsonl_fpath)
        legacy_rows = [
            {
                TASK_INDEX_KEY_NAME: 2,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "stage_index": stage,
                "invalid_judge_response": True,
            }
            for stage in (0, 1)
        ]
        output.write_bytes(b"\n".join(map(orjson.dumps, legacy_rows)) + b"\n")

        RolloutCollectionHelper()._load_from_cache(config)

        migrated = [orjson.loads(line) for line in _failures_path_for(output).read_bytes().splitlines()]
        assert [row["stage_index"] for row in migrated] == [0, 1]

    def test_legacy_invalid_judge_without_artifact_does_not_request_reuse(self, tmp_path: Path) -> None:
        config = self._write_resume_state(tmp_path, [], retry_invalid_judge_responses=True)
        output = Path(config.output_jsonl_fpath)
        output.write_bytes(
            orjson.dumps(
                {
                    TASK_INDEX_KEY_NAME: 2,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    "invalid_judge_response": True,
                    "reuse_cached_deliverable": True,
                    "deliverables_dir": str(tmp_path / "missing-deliverables"),
                }
            )
            + b"\n"
        )

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        by_task = {row[TASK_INDEX_KEY_NAME]: row for row in input_rows}
        assert "reuse_cached_deliverable" not in by_task[2]
        migrated = [orjson.loads(line) for line in _failures_path_for(output).read_bytes().splitlines()]
        assert "reuse_cached_deliverable" not in migrated[0]

    def test_legacy_invalid_judge_with_only_run_state_does_not_request_reuse(self, tmp_path: Path) -> None:
        config = self._write_resume_state(tmp_path, [], retry_invalid_judge_responses=True)
        output = Path(config.output_jsonl_fpath)
        deliverables_dir = tmp_path / "deliverables" / "task-2"
        deliverables_dir.mkdir(parents=True)
        (deliverables_dir / "finish_params.json").write_text("{}")
        (deliverables_dir / "history.json").write_text("[]")
        (deliverables_dir / "reference_files").mkdir()
        output.write_bytes(
            orjson.dumps(
                {
                    TASK_INDEX_KEY_NAME: 2,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    "invalid_judge_response": True,
                    "reuse_cached_deliverable": True,
                    "deliverables_dir": str(deliverables_dir),
                }
            )
            + b"\n"
        )

        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

        by_task = {row[TASK_INDEX_KEY_NAME]: row for row in input_rows}
        assert "reuse_cached_deliverable" not in by_task[2]
        migrated = [orjson.loads(line) for line in _failures_path_for(output).read_bytes().splitlines()]
        assert "reuse_cached_deliverable" not in migrated[0]

    def test_invalid_judge_main_rows_are_left_alone_by_default(self, tmp_path: Path) -> None:
        """Without the opt-in, a verifier's scored invalid-judge row stays a scored row."""
        config = self._write_resume_state(tmp_path, [])
        output = Path(config.output_jsonl_fpath)
        scored_row = {
            TASK_INDEX_KEY_NAME: 2,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "invalid_judge_response": True,
            "reward": 0.0,
        }
        output.write_bytes(orjson.dumps(scored_row) + b"\n")

        input_rows, cached_rows, cached_results, _ = RolloutCollectionHelper()._load_from_cache(config)

        assert 2 not in [row[TASK_INDEX_KEY_NAME] for row in input_rows]
        assert cached_results == [scored_row]
        assert [row[TASK_INDEX_KEY_NAME] for row in cached_rows] == [2]
        assert output.read_bytes() == orjson.dumps(scored_row) + b"\n"
        assert not _failures_path_for(output).exists()

    def test_skipped_without_terminal_flag_is_gated_only_when_opted_in(self, tmp_path: Path) -> None:
        failures = [{TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "skipped"}]

        config = self._write_resume_state(tmp_path, failures)
        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)
        assert 1 in [r[TASK_INDEX_KEY_NAME] for r in input_rows]

        config = self._write_resume_state(tmp_path, failures, retry_terminal_timeouts=True)
        input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)
        assert 1 not in [r[TASK_INDEX_KEY_NAME] for r in input_rows]

    def test_skipped_failures_stay_terminal(self, tmp_path: Path) -> None:
        failures = [
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                NG_FAILURE_CLASS_KEY: "skipped",
                NG_TERMINAL_KEY: True,
            }
        ]
        for retry_terminal_timeouts in (False, True):
            config = self._write_resume_state(tmp_path, failures, retry_terminal_timeouts=retry_terminal_timeouts)

            input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

            assert 1 not in [r[TASK_INDEX_KEY_NAME] for r in input_rows]

    def test_explicitly_terminal_rows_stay_gated(self, tmp_path: Path) -> None:
        failures = [{TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, NG_TERMINAL_KEY: True}]
        for retry_terminal_timeouts in (False, True):
            config = self._write_resume_state(tmp_path, failures, retry_terminal_timeouts=retry_terminal_timeouts)

            input_rows, *_ = RolloutCollectionHelper()._load_from_cache(config)

            assert 1 not in [r[TASK_INDEX_KEY_NAME] for r in input_rows]


class TestIsTerminalFailure:
    """The failure contract is opt-in; off, a row is terminal iff it is stamped terminal."""

    @pytest.mark.parametrize(
        ("failure_class", "stamped", "off", "on"),
        [
            ("timeout_exceeded", True, True, False),
            ("timeout_exceeded", False, False, False),
            ("reference_missing", True, True, False),
            ("eval_missing", True, True, False),
            ("transport_ineligible", True, True, False),
            ("skipped", False, False, True),
            ("skipped", True, True, True),
            ("agent_run_error", True, True, True),
            ("agent_run_error", False, False, False),
            (None, True, True, True),
            (None, False, False, False),
        ],
    )
    def test_off_matches_the_terminal_flag_and_on_applies_the_contract(
        self, failure_class: str | None, stamped: bool, off: bool, on: bool
    ) -> None:
        row: dict = {}
        if failure_class is not None:
            row[NG_FAILURE_CLASS_KEY] = failure_class
        if stamped:
            row[NG_TERMINAL_KEY] = True

        assert is_terminal_failure(row) is off
        assert is_terminal_failure(row, retry_terminal_timeouts=True) is on

    def test_config_defaults_are_off(self, tmp_path: Path) -> None:
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"), output_jsonl_fpath=str(tmp_path / "output.jsonl")
        )

        assert config.retry_terminal_timeouts is False
        assert config.retry_invalid_judge_responses is False


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

    def test_rollout_glob_excludes_failure_sidecars_and_materialized_inputs(self, tmp_path: Path) -> None:
        rollout = tmp_path / "rollouts-rs0-chunk0.jsonl"
        failure = tmp_path / "rollouts-rs0-chunk0_failures.jsonl"
        rollout.write_text("{}\n")
        failure.write_text("{}\n")
        materialized_path_for(rollout).write_text("{}\n")

        assert _expand_input_glob(str(tmp_path / "rollouts-rs*-chunk*.jsonl")) == [str(rollout)]

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

            async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

        async def fake_call(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

        async def _noop(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

    @pytest.mark.parametrize("include_successful_agent", [False, True])
    async def test_standalone_aggregation_repairs_a_failed_agent_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_successful_agent: bool
    ) -> None:
        """Failed single- and multi-agent aggregation keeps repairable metrics but raises."""
        shard = tmp_path / "shard.jsonl"
        records = [
            {
                AGENT_REF_KEY_NAME: {"name": "agent_a"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
            },
            {
                AGENT_REF_KEY_NAME: {"name": "agent_b"},
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 0.0,
            },
        ]
        if not include_successful_agent:
            records = records[1:]
        shard.write_bytes(b"\n".join(orjson.dumps(record) for record in records) + b"\n")
        fail_agent_b = True

        async def post(server_name: str, url_path: str, json: AggregateMetricsRequest) -> FakeResponse:
            assert url_path == "/aggregate_metrics"
            if server_name == "agent_b_environment_server" and fail_agent_b:
                return FakeResponse(500)
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("agent_a", "agent_b"))
        config = RolloutAggregationConfig(
            input_glob=str(shard),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            disable_health_check=True,
        )

        with pytest.raises(RuntimeError, match="Aggregation failed for agents: agent_b"):
            await RolloutAggregationHelper().run_from_config(config)
        metrics_fpath = tmp_path / "rollouts_aggregate_metrics.json"
        first_attempt = orjson.loads(metrics_fpath.read_bytes())
        assert first_attempt[-1][AGGREGATION_ERROR_KEY]["http_status"] == 500
        if include_successful_agent:
            assert AGGREGATION_ERROR_KEY not in first_attempt[0]
            assert first_attempt[0]["key_metrics"]["mean/reward"] == 1.0

        fail_agent_b = False
        repaired_fpath = await RolloutAggregationHelper().run_from_config(config)

        assert repaired_fpath == metrics_fpath
        repaired = orjson.loads(repaired_fpath.read_bytes())
        assert not any(AGGREGATION_ERROR_KEY in entry for entry in repaired)
        assert repaired[-1]["key_metrics"]["mean/reward"] == 0.0
        if include_successful_agent:
            assert repaired[0]["key_metrics"]["mean/reward"] == 1.0

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

        async def fake_call(self, results, rows, output_fpath, *, raise_on_error: bool = True):
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

    @staticmethod
    def _serve(monkeypatch: pytest.MonkeyPatch, by_server: dict, servers: dict[str, str]) -> None:
        """Front each agent with its environment servers and record what each server is asked to score."""

        async def post(server_name: str, url_path: str, json, **kwargs):
            rows = [dict(r) for r in json.verify_responses]
            by_server[server_name] = sorted(rows, key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
            return FakeResponse(200, compute_aggregate_metrics(rows).model_dump())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(
            {
                **{agent: {"responses_api_agents": {"impl": {}}} for agent in set(servers.values())},
                **{
                    server: {"environment_servers": {"legacy_agent": {"agent_server": {"name": agent}}}}
                    for server, agent in servers.items()
                },
            }
        )

    @staticmethod
    def _scores(rows: list[dict]) -> list[tuple]:
        return [(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME], r["reward"]) for r in rows]

    @staticmethod
    def _write_rows(fpath: Path, *rows: dict) -> None:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))

    @staticmethod
    def _rollout(task_index: int, agent: str, **fields) -> dict:
        return {
            TASK_INDEX_KEY_NAME: task_index,
            ROLLOUT_INDEX_KEY_NAME: 0,
            AGENT_REF_KEY_NAME: {"name": agent},
            **fields,
        }

    async def _aggregate(self, tmp_path: Path, *shards: Path) -> None:
        config = RolloutAggregationConfig(
            input_glob=",".join(str(shard) for shard in shards),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            count_missing_rollouts_as_zero=True,
            disable_health_check=True,
        )
        await RolloutAggregationHelper().run_from_config(config)

    @pytest.mark.parametrize("run1_has", ["a row", "a failure"])
    async def test_runs_that_reuse_task_indices_do_not_hide_each_others_lost_rollouts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock, run1_has: str
    ) -> None:
        """Every run numbers its tasks from 0, so run1's (0, 0) is not run2's."""
        run1 = tmp_path / "run1" / "rollouts.jsonl"
        run2 = tmp_path / "run2" / "rollouts.jsonl"
        self._write_rows(materialized_path_for(run1), self._rollout(0, "agent_a"), self._rollout(1, "agent_a"))
        if run1_has == "a row":
            self._write_rows(run1, self._rollout(0, "agent_a", reward=1.0), self._rollout(1, "agent_a", reward=1.0))
        else:
            self._write_rows(run1, self._rollout(1, "agent_a", reward=1.0))
            self._write_rows(
                _failures_path_for(run1), self._rollout(0, "agent_a", **{NG_FAILURE_CLASS_KEY: "agent_run_error"})
            )
        # run2 lost (0, 0): no row and no sidecar entry.
        self._write_rows(materialized_path_for(run2), self._rollout(0, "agent_b"), self._rollout(1, "agent_b"))
        self._write_rows(run2, self._rollout(1, "agent_b", reward=1.0))
        by_server: dict[str, list] = {}
        self._serve(monkeypatch, by_server, {"env_a": "agent_a", "env_b": "agent_b"})

        await self._aggregate(tmp_path, run1, run2)

        assert self._scores(by_server["env_b"]) == [(0, 0, 0.0), (1, 0, 1.0)]

    @pytest.mark.parametrize("run1_failed_too", [False, True], ids=["run1 landed it", "both failed"])
    async def test_a_counted_failure_is_not_hidden_by_another_runs_rollout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock, run1_failed_too: bool
    ) -> None:
        """Failure rows are read per shard too, so run1's (0, 0) neither hides nor replaces run2's."""
        run1 = tmp_path / "run1" / "rollouts.jsonl"
        run2 = tmp_path / "run2" / "rollouts.jsonl"
        failure = {NG_FAILURE_CLASS_KEY: AGENT_RUN_ERROR_FAILURE_CLASS}
        for run, agent in ((run1, "agent_a"), (run2, "agent_b")):
            self._write_rows(materialized_path_for(run), self._rollout(0, agent), self._rollout(1, agent))
        if run1_failed_too:
            self._write_rows(run1, self._rollout(1, "agent_a", reward=1.0))
            self._write_rows(_failures_path_for(run1), self._rollout(0, "agent_a", **failure))
        else:
            self._write_rows(run1, self._rollout(0, "agent_a", reward=1.0), self._rollout(1, "agent_a", reward=1.0))
        self._write_rows(run2, self._rollout(1, "agent_b", reward=1.0))
        self._write_rows(_failures_path_for(run2), self._rollout(0, "agent_b", **failure))
        by_server: dict[str, list] = {}
        self._serve(monkeypatch, by_server, {"env_a": "agent_a", "env_b": "agent_b"})

        config = RolloutAggregationConfig(
            input_glob=f"{run1},{run2}",
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            count_failure_classes_as_zero=[AGENT_RUN_ERROR_FAILURE_CLASS],
            disable_health_check=True,
        )
        await RolloutAggregationHelper().run_from_config(config)

        assert self._scores(by_server["env_b"]) == [(0, 0, 0.0), (1, 0, 1.0)]
        assert self._scores(by_server["env_a"]) == [(0, 0, 0.0 if run1_failed_too else 1.0), (1, 0, 1.0)]

    @pytest.mark.parametrize("bench1_lost_it_too", [False, True], ids=["one lost", "both lost"])
    async def test_one_agent_behind_two_servers_scores_each_servers_lost_rollouts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        bench1_lost_it_too: bool,
    ) -> None:
        """Rows that name their server can share an agent; rollouts are told apart by the server."""

        def rollout(task_index: int, server: str, **fields) -> dict:
            return self._rollout(task_index, "agent_x", **{NG_ENVIRONMENT_SERVER_KEY: server}, **fields)

        bench1 = tmp_path / "bench1" / "rollouts.jsonl"
        bench2 = tmp_path / "bench2" / "rollouts.jsonl"
        self._write_rows(materialized_path_for(bench1), rollout(0, "env1"), rollout(1, "env1"))
        self._write_rows(bench1, *[rollout(t, "env1", reward=1.0) for t in ((1,) if bench1_lost_it_too else (0, 1))])
        self._write_rows(materialized_path_for(bench2), rollout(0, "env2"), rollout(1, "env2"))
        self._write_rows(bench2, rollout(1, "env2", reward=1.0))
        by_server: dict[str, list] = {}
        self._serve(monkeypatch, by_server, {"env1": "agent_x", "env2": "agent_x"})

        await self._aggregate(tmp_path, bench1, bench2)

        assert self._scores(by_server["env2"]) == [(0, 0, 0.0), (1, 0, 1.0)]
        assert self._scores(by_server["env1"]) == [(0, 0, 0.0 if bench1_lost_it_too else 1.0), (1, 0, 1.0)]

    async def test_a_lost_rollout_takes_its_task_fields_from_its_own_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """The zero carries the fields its own run's real rows carry, and no other run's values."""
        run1 = tmp_path / "run1" / "rollouts.jsonl"
        run2 = tmp_path / "run2" / "rollouts.jsonl"
        self._write_rows(materialized_path_for(run1), self._rollout(0, "agent_a", difficulty="hard", source_id=1))
        self._write_rows(run1, self._rollout(0, "agent_a", difficulty="hard", reward=1.0))
        self._write_rows(
            materialized_path_for(run2),
            self._rollout(0, "agent_b", difficulty="easy", source_id=2),
            self._rollout(1, "agent_b", difficulty="easy", source_id=3),
        )
        self._write_rows(run2, self._rollout(1, "agent_b", difficulty="easy", reward=1.0))
        by_server: dict[str, list] = {}
        self._serve(monkeypatch, by_server, {"env_a": "agent_a", "env_b": "agent_b"})

        await self._aggregate(tmp_path, run1, run2)

        zero = by_server["env_b"][0]
        assert (zero[TASK_INDEX_KEY_NAME], zero["reward"], zero["difficulty"]) == (0, 0.0, "easy")
        # No real row reports source_id, so it would only add a metric of its own.
        assert "source_id" not in zero

    async def test_a_failure_another_run_landed_the_key_of_still_counts_as_dropped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_global_config: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The coverage report reads each shard's sidecar against its own rows too."""
        run1 = tmp_path / "run1" / "rollouts.jsonl"
        run2 = tmp_path / "run2" / "rollouts.jsonl"
        self._write_rows(run1, self._rollout(0, "agent_a", reward=1.0), self._rollout(1, "agent_a", reward=1.0))
        self._write_rows(run2, self._rollout(1, "agent_b", reward=1.0))
        self._write_rows(
            _failures_path_for(run2),
            self._rollout(0, "agent_b", **{NG_FAILURE_CLASS_KEY: AGENT_RUN_ERROR_FAILURE_CLASS}),
        )
        self._serve(monkeypatch, {}, {"env_a": "agent_a", "env_b": "agent_b"})

        config = RolloutAggregationConfig(
            input_glob=f"{run1},{run2}", output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"), disable_health_check=True
        )
        await RolloutAggregationHelper().run_from_config(config)

        out = capsys.readouterr().out
        assert "Rollouts missing from the score: 1 of 4 materialized (1 agent_run_error routed this run" in out
        assert "Metrics cover: 3 of 4 rollouts" in out

    async def test_an_unstamped_inventory_is_keyed_by_the_server_its_results_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """An agent-routed run stamps its results with their server but not its inventory."""
        original = tmp_path / "original" / "rollouts.jsonl"
        retry = tmp_path / "retry" / "rollouts.jsonl"
        stamped = {NG_ENVIRONMENT_SERVER_KEY: "env_a"}
        self._write_rows(materialized_path_for(original), self._rollout(0, "agent_a"), self._rollout(1, "agent_a"))
        self._write_rows(original, self._rollout(1, "agent_a", reward=1.0, **stamped))
        self._write_rows(materialized_path_for(retry), self._rollout(0, "agent_a"))
        self._write_rows(retry, self._rollout(0, "agent_a", reward=1.0, **stamped))
        by_server: dict[str, list] = {}
        self._serve(monkeypatch, by_server, {"env_a": "agent_a", "env_b": "agent_b"})

        await self._aggregate(tmp_path, original, retry)

        assert {server: self._scores(rows) for server, rows in by_server.items()} == {
            "env_a": [(0, 0, 1.0), (1, 0, 1.0)]
        }

    async def test_a_run_with_nothing_to_score_does_not_need_the_head_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Only scoring reaches the head server, so a shard whose every rollout failed still reports coverage."""
        shard = tmp_path / "rollouts.jsonl"
        shard.write_bytes(b"")
        # A sidecar written before rows were stamped with their server.
        self._write_rows(
            _failures_path_for(shard),
            self._rollout(0, "agent_a", **{NG_FAILURE_CLASS_KEY: AGENT_RUN_ERROR_FAILURE_CLASS}),
        )

        def no_head_server(*args, **kwargs):
            raise RuntimeError("Could not connect to the head server")

        monkeypatch.setattr(nemo_gym.rollout_collection, "setup_server_client_utils", no_head_server)
        config = RolloutAggregationConfig(
            input_glob=str(shard), output_jsonl_fpath=str(tmp_path / "merged.jsonl"), disable_health_check=True
        )

        assert await RolloutAggregationHelper().run_from_config(config) is None
        assert "1 agent_run_error routed this run" in capsys.readouterr().out

    async def test_a_rollout_another_shard_landed_for_the_same_server_is_not_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config: MagicMock
    ) -> None:
        """A retry shard that landed the lost rollout scores it once, with no zero beside it."""
        original = tmp_path / "original" / "rollouts.jsonl"
        retry = tmp_path / "retry" / "rollouts.jsonl"
        self._write_rows(materialized_path_for(original), self._rollout(0, "agent_a"), self._rollout(1, "agent_a"))
        self._write_rows(original, self._rollout(1, "agent_a", reward=1.0))
        self._write_rows(materialized_path_for(retry), self._rollout(0, "agent_a"))
        self._write_rows(retry, self._rollout(0, "agent_a", reward=1.0))
        by_server: dict[str, list] = {}
        self._serve(monkeypatch, by_server, {"env_a": "agent_a", "env_b": "agent_b"})

        await self._aggregate(tmp_path, original, retry)

        assert {server: self._scores(rows) for server, rows in by_server.items()} == {
            "env_a": [(0, 0, 1.0), (1, 0, 1.0)]
        }


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

        store.delete_now(["0-0"])
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

    def test_e2e_config_preserves_batch_manifest_path(self) -> None:
        config = E2ERolloutCollectionConfig.model_validate(
            {"output_jsonl_fpath": "out.jsonl", "split": "benchmark", "batch_manifest_fpath": "batch_manifest.json"}
        )
        assert config.batch_manifest_fpath == "batch_manifest.json"

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
            {
                name: block
                for agent in ("shared_agent_a", "shared_agent_b")
                for name, block in (
                    (agent, {"responses_api_agents": {"impl": {}}}),
                    (
                        f"{agent}_environment_server",
                        {"environment_servers": {"legacy_agent": {"agent_server": {"name": agent}}}},
                    ),
                )
            }
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "setup_server_client_utils", lambda *a, **k: mock_client)
        for fut in RolloutCollectionHelper().run_examples(rows):
            await fut
        assert sorted(posted) == ["shared_agent_a_environment_server", "shared_agent_b_environment_server"]

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
                "math_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "math_agent"}}}
                },
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


class TestTurnsFromModelCalls:
    """Turns come from captured model calls when the agent reports no turns of its own."""

    @staticmethod
    def _call(call_id: str, response: dict, *, started_at: float, response_id: str | None = None):
        from nemo_gym.rollout_observability import TrajectoryModelCall

        return TrajectoryModelCall.model_validate(
            {
                "model_call_id": call_id,
                "started_at": started_at,
                "request": {"input": [{"role": "user", "content": call_id}]},
                "response": response,
                "response_metadata": {
                    "response_id": response_id,
                    "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                },
            }
        )

    @staticmethod
    def _invocation(invocation_id: str, call_ids: list[str]):
        from nemo_gym.rollout_observability import AgentInvocation

        return AgentInvocation.model_validate(
            {"invocation_id": invocation_id, "model_calls": [{"model_call_id": call_id} for call_id in call_ids]}
        )

    def test_responses_calls_become_turns_of_the_referencing_invocation(self) -> None:
        tool_call = {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"}
        reasoning = {"type": "reasoning", "id": "r1", "summary": []}
        message = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "cold"}]}
        calls = [
            self._call("first", {"output": [reasoning, tool_call]}, started_at=1.0),
            self._call("second", {"output": [message]}, started_at=2.0),
        ]

        turns = nemo_gym.rollout_collection._turns_from_model_calls(
            "task", "rollout", [self._invocation("root", ["first", "second"])], calls, resolved=True
        )

        assert [(turn.invocation_id, turn.turn_no) for turn in turns] == [("root", 1), ("root", 2)]
        assert turns[0].answer == [tool_call]
        assert turns[0].reasoning_content == [reasoning]
        assert turns[0].question == [{"type": "message", "role": "user", "content": "first"}]
        # step_count counts tool calls made before the turn, as agents that build their own turns do.
        assert [turn.step_count for turn in turns] == [0, 1]
        assert turns[0].model_calls[0].model_call_id == "first"
        assert [turn.resolved for turn in turns] == [None, True]

    def test_chat_completion_calls_split_the_message_and_its_reasoning(self) -> None:
        message = {"role": "assistant", "content": "", "reasoning_content": "think", "tool_calls": [{"id": "t1"}]}
        calls = [self._call("only", {"choices": [{"message": message}]}, started_at=1.0)]

        [turn] = nemo_gym.rollout_collection._turns_from_model_calls(
            "task", "rollout", [self._invocation("root", ["only"])], calls, resolved=None
        )

        assert turn.invocation_id == "root"
        assert turn.reasoning_content == "think"
        assert turn.answer == {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]}

    def test_a_call_that_returned_nothing_is_not_a_turn(self) -> None:
        from nemo_gym.rollout_observability import TrajectoryModelCall

        failed = TrajectoryModelCall.model_validate({"model_call_id": "failed", "started_at": 1.0})
        answered = self._call("answered", {"output": []}, started_at=2.0)

        turns = nemo_gym.rollout_collection._turns_from_model_calls(
            "task", "rollout", [self._invocation("root", ["failed", "answered"])], [failed, answered], None
        )

        assert [(turn.model_calls[0].model_call_id, turn.turn_no) for turn in turns] == [("answered", 1)]

    @pytest.mark.parametrize("status", [400, 429, 500])
    def test_http_error_payload_is_retained_without_becoming_a_turn(self, status: int) -> None:
        failed = self._call("failed", {"error": {"message": "try again"}}, started_at=1.0)
        failed.response_metadata.status_code = status
        answered = self._call("answered", {"output": []}, started_at=2.0)
        invocation = self._invocation("root", ["failed", "answered"])
        turns = nemo_gym.rollout_collection._turns_from_model_calls(
            "task", "rollout", [invocation], [failed, answered], None
        )
        assert [(turn.model_calls[0].model_call_id, turn.turn_no) for turn in turns] == [("answered", 1)]
        assert [ref.model_call_id for ref in invocation.model_calls] == ["failed", "answered"]
        assert failed.response == {"error": {"message": "try again"}}

    @pytest.mark.parametrize("invocation_count", [1, 2])
    def test_a_call_no_invocation_references_is_skipped(self, invocation_count: int) -> None:
        calls = [
            self._call("owned", {"output": []}, started_at=1.0),
            self._call("orphan", {"output": []}, started_at=2.0),
        ]
        invocations = [self._invocation("assistant", ["owned"]), self._invocation("user", [])][:invocation_count]

        turns = nemo_gym.rollout_collection._turns_from_model_calls("task", "rollout", invocations, calls, None)

        assert [turn.model_calls[0].model_call_id for turn in turns] == ["owned"]

    @pytest.mark.parametrize("has_invocation", [False, True])
    def test_unowned_capture_is_preserved_without_inventing_turns(self, has_invocation: bool) -> None:
        result = {
            "ng_model_call_capture": {
                "calls": [{"model_call_id": "unowned", "response": {"output": []}, "status_code": 200}]
            },
        }
        if has_invocation:
            result["ng_agent_observations"] = {
                "source": "external_harness",
                "records": [{"kind": "agent_invocation", "invocation_id": "root"}],
            }

        _attach_trajectory_record({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}, result)

        trajectory = result[NG_TRAJECTORY_KEY]
        assert trajectory["turns"] == []
        assert [call["model_call_id"] for call in trajectory["model_calls"]] == ["unowned"]
        assert "turns_unavailable" in {gap["code"] for gap in trajectory["gaps"]}

    @pytest.mark.parametrize("reference_key", ["model_call_id", "response_id"])
    def test_judge_capture_does_not_become_a_policy_turn(self, reference_key: str) -> None:
        from nemo_gym.health.checks import (
            _bind_policy_call_views,
            _normalized_trajectory_calls,
            _rollout_token_count_mismatch,
        )

        row = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        result = {
            "response": {"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
            "ng_agent_observations": {
                "source": "hermes",
                "records": [
                    {
                        "kind": "agent_invocation",
                        "invocation_id": "root",
                        "model_calls": [
                            {
                                reference_key: "policy" if reference_key == "model_call_id" else "resp-policy",
                                "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                            }
                        ],
                    }
                ],
            },
            "ng_model_call_capture": {
                "calls": [
                    {
                        "model_call_id": call_id,
                        "response_id": f"resp-{call_id}",
                        "model_ref": {"type": "responses_api_models", "name": model_name},
                        "request": {"input": [{"role": "user", "content": call_id}]},
                        "response": {
                            "id": f"resp-{call_id}",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "answer"}],
                                }
                            ],
                        },
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "started_at": started_at,
                        "status_code": 200,
                    }
                    for call_id, model_name, tokens_in, tokens_out, started_at in [
                        ("policy", "policy_model", 10, 2, 1.0),
                        ("judge", "judge_model", 20, 3, 2.0),
                    ]
                ]
            },
        }

        _attach_trajectory_record(row, result)

        trajectory = result[NG_TRAJECTORY_KEY]
        assert [call["model_call_id"] for call in trajectory["model_calls"]] == ["policy", "judge"]
        assert [turn["model_calls"][0]["model_call_id"] for turn in trajectory["turns"]] == ["policy"]
        turns, _ = _bind_policy_call_views(trajectory, _normalized_trajectory_calls(trajectory))
        assert _rollout_token_count_mismatch(result, turns, {"task_index": 0}) == []
        perf = _build_ng_perf(result, rollout_latency_ms=None)
        assert perf["num_turns"] == 1
        assert perf["prompt_tokens"] == 10
        assert perf["completion_tokens"] == 2
        assert perf["token_observability_coverage"] == 1.0


class TestMaskingStepMetrics:
    """Progress accounting covers persisted rollouts; dropped attempts are counted apart."""

    def test_a_healthy_run_adds_no_keys(self) -> None:
        assert _masking_step_metrics("my_agent", Counter({"reward": 2.0, "count": 4}), Counter()) == {}

    def test_masked_rollouts_report_their_share_and_the_score_without_them(self) -> None:
        # 10 persisted, 2 masked; the 8 unmasked ones scored 4.0 in total.
        metrics = _masking_step_metrics("my_agent", Counter({"reward": 4.0, "count": 8, "masked": 2}), Counter())

        assert metrics == {
            "progress/my_agent/masked_pct": 20.0,
            "progress/my_agent/reward_unmasked": 50.0,
        }

    def test_every_persisted_rollout_masked_publishes_no_score(self) -> None:
        """No unmasked rollout means no honest average to publish."""
        assert _masking_step_metrics("my_agent", Counter({"masked": 6}), Counter()) == {
            "progress/my_agent/masked_pct": 100.0
        }

    def test_failed_and_omitted_attempts_do_not_enter_the_quality_average(self) -> None:
        """A sidecar row and a kill-shaped one are counted, never averaged as a zero."""
        metrics = _masking_step_metrics(
            "my_agent",
            Counter({"reward": 4.0, "count": 4}),
            Counter({"failed": 3, "omitted": 2}),
        )

        assert metrics == {
            "progress/my_agent/reward_unmasked": 100.0,
            "progress/my_agent/failed": 3,
            "progress/my_agent/omitted": 2,
        }


class TestAnAgentThatOnlyEverFails:
    """The wiring case: a total failure must not fall out of the export.

    `_masking_step_metrics` is correct on its own Counters; what this covers is the loop
    that feeds it. An agent whose every request returns no result never lands in
    `agent_name_to_counts`, so iterating that dict would drop exactly the agent whose
    failure the series exists to surface.
    """

    def _exported_agents(self, scored: dict, dropped: dict) -> set:
        """Reproduce the export loop's selection over the two counter dicts."""
        agent_name_to_scored = defaultdict(Counter, {k: Counter(v) for k, v in scored.items()})
        agent_name_to_dropped = defaultdict(Counter, {k: Counter(v) for k, v in dropped.items()})

        step_metrics: dict = {}
        for agent_name in sorted(agent_name_to_scored.keys() | agent_name_to_dropped.keys()):
            step_metrics.update(
                _masking_step_metrics(
                    agent_name,
                    agent_name_to_scored.get(agent_name, Counter()),
                    agent_name_to_dropped.get(agent_name, Counter()),
                )
            )
        return {key.split("/")[1] for key in step_metrics}

    def test_an_agent_with_no_successful_result_still_reports_its_failures(self) -> None:
        exported = self._exported_agents(
            scored={"healthy_agent": {"reward": 3.0, "count": 4}},
            dropped={"broken_agent": {"failed": 4}},
        )

        assert "broken_agent" in exported

    def test_the_healthy_agent_is_not_lost_in_the_process(self) -> None:
        exported = self._exported_agents(
            scored={"healthy_agent": {"reward": 3.0, "count": 4, "masked": 1}},
            dropped={"broken_agent": {"failed": 4}},
        )

        assert exported == {"healthy_agent", "broken_agent"}

    def test_a_run_with_nothing_wrong_still_exports_nothing(self) -> None:
        """The series stays empty on a healthy run, as before."""
        assert self._exported_agents(scored={"healthy_agent": {"reward": 3.0, "count": 4}}, dropped={}) == set()

    def test_the_counters_are_not_grown_by_being_read(self) -> None:
        agent_name_to_scored: dict = defaultdict(Counter, {"healthy_agent": Counter({"count": 1})})
        agent_name_to_dropped: dict = defaultdict(Counter, {"broken_agent": Counter({"failed": 1})})

        for agent_name in sorted(agent_name_to_scored.keys() | agent_name_to_dropped.keys()):
            _masking_step_metrics(
                agent_name,
                agent_name_to_scored.get(agent_name, Counter()),
                agent_name_to_dropped.get(agent_name, Counter()),
            )

        assert set(agent_name_to_scored) == {"healthy_agent"}
        assert set(agent_name_to_dropped) == {"broken_agent"}


class TestEnvironmentServerRouting:
    def _row(self) -> dict:
        return {
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            ATTEMPT_INDEX_KEY_NAME: 1,
            "task_source": "swe",
            "instance_id": "instance",
            "base_commit": "abc",
            "responses_create_params": {"input": "fix it"},
        }

    def test_default_environment_preprocesses_rows_without_legacy_routing_fields(self) -> None:
        row = {"responses_create_params": {"input": "fix it"}}
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_routing_mode="legacy",
            environment_server_name="environment",
            num_repeats=1,
        )

        rows = RolloutCollectionHelper._preprocess_raw_rows(
            [(0, orjson.dumps(row).decode(), row)],
            config,
        )

        assert len(rows) == 1
        assert AGENT_REF_KEY_NAME not in rows[0]
        assert rows[0][TASK_INDEX_KEY_NAME] == 0
        assert rows[0][ROLLOUT_INDEX_KEY_NAME] == 0
        assert rows[0][NG_ENVIRONMENT_SERVER_KEY] == "environment"

    async def test_taskset_route_builds_native_episode_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"reward": 1.0, "agent_ref": {"name": "hermes"}}
        post = AsyncMock(return_value=FakeResponse(200, payload))
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = _environment_server_config()
        materialized = {
            "task_id": {
                "taskset": "swe_pro",
                "task_id": "instance",
            },
            "task_input": {
                "responses_create_params": {"input": "fix it"},
                "task_data": {"instance_id": "instance"},
            },
        }
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_routing_mode="taskset",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=1,
        )
        rows = RolloutCollectionHelper._preprocess_raw_rows(
            [(0, orjson.dumps(materialized).decode(), materialized)],
            config,
        )

        _, result = await next(
            RolloutCollectionHelper().run_examples(
                rows,
            )
        )

        assert result == payload
        assert post.await_args.kwargs["server_name"] == "environment"
        assert AGENT_REF_KEY_NAME not in rows[0]
        assert post.await_args.kwargs["json"] == {
            "episode_id": {"rollout_id": "0-0", "attempt": 0},
            "task": {
                "task_id": {
                    "taskset": "swe_pro",
                    "task_id": "instance",
                },
                "task_input": {
                    "responses_create_params": {"input": "fix it"},
                    "task_data": {"instance_id": "instance"},
                },
            },
        }

    def test_taskset_route_requires_a_configured_environment_server(self) -> None:
        row = {
            "task_id": {"taskset": "swe_pro", "task_id": "instance"},
            "task_input": {"responses_create_params": {"input": "fix it"}, "task_data": {}},
        }
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_routing_mode="taskset",
            environment_server_routes={"other": "environment"},
            num_repeats=1,
        )

        with pytest.raises(ValueError, match="No environment server route.*swe_pro"):
            RolloutCollectionHelper._preprocess_raw_rows(
                [(0, orjson.dumps(row).decode(), row)],
                config,
            )

    def test_malformed_materialized_input_still_routes_by_taskset(self) -> None:
        row = {
            "task_id": {"taskset": "swe_pro", "task_id": "instance"},
            "task_input": None,
        }
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_routing_mode="taskset",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=1,
        )

        assert nemo_gym.rollout_collection._environment_server_for_config_row(row, config) == "environment"

    async def test_routes_legacy_row_to_selected_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "reward": 1.0,
            "model_patch": "patch",
            "ng_agent_observations": {"source": "hermes", "records": [], "gaps": []},
        }
        post = AsyncMock(return_value=FakeResponse(200, payload))
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = _environment_server_config()
        row = self._row()

        returned_row, result = await next(
            RolloutCollectionHelper().run_examples(
                [row],
                environment_server_name="environment",
            )
        )

        assert returned_row is row
        assert result["reward"] == 1.0
        assert result["model_patch"] == "patch"
        assert result["ng_agent_observations"]["source"] == "hermes"
        assert post.await_args.kwargs["server_name"] == "environment"
        assert post.await_args.kwargs["url_path"] == "/run"
        assert post.await_args.kwargs["json"] is row
        assert row[AGENT_REF_KEY_NAME] == {"name": "hermes"}

    async def test_environment_route_rejects_mismatched_resources_server(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        post = AsyncMock()
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = _environment_server_config()
        row = self._row() | {"task_source": "other"}

        with pytest.raises(ValueError, match="does not match.*resources server"):
            next(
                RolloutCollectionHelper().run_examples(
                    [row],
                    environment_server_name="environment",
                )
            )
        post.assert_not_awaited()

    async def test_environment_route_rejects_mismatched_agent_ref(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = AsyncMock()
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = _environment_server_config()
        row = self._row() | {AGENT_REF_KEY_NAME: {"name": "other"}}

        with pytest.raises(ValueError, match="is not an agent of environment server"):
            next(
                RolloutCollectionHelper().run_examples(
                    [row],
                    environment_server_name="environment",
                )
            )
        post.assert_not_awaited()

    async def test_accepts_projected_http_200_failure_for_the_sidecar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            NG_FAILURE_CLASS_KEY: ENVIRONMENT_SERVER_FAILURE_CLASS,
            NG_TERMINAL_KEY: False,
            "_ng_failure_message": "agent unavailable",
            "_ng_failure_stage": "agent",
            "_ng_failure_partial_response": {"id": "partial"},
        }
        post = AsyncMock(return_value=FakeResponse(200, payload))
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = _environment_server_config()

        _, result = await next(
            RolloutCollectionHelper().run_examples(
                [self._row()],
                environment_server_name="environment",
            )
        )

        assert result[NG_FAILURE_CLASS_KEY] == ENVIRONMENT_SERVER_FAILURE_CLASS
        assert result[NG_TERMINAL_KEY] is False
        assert result["_ng_failure_terminal"] is False
        assert result["_ng_failure_stage"] == "agent"
        assert result["_ng_failure_partial_response"]["id"] == "partial"

    @staticmethod
    def _mixed_batch_config() -> DictConfig:
        """The native SWE Pro pairing plus a second, compatibility-routed pairing on the same server."""
        config = _environment_server_config()
        config["hermes_legacy"] = {
            "responses_api_agents": {
                "hermes_agent": {
                    "resources_server": {"type": "resources_servers", "name": "swe"},
                }
            }
        }
        config["legacy_environment"] = {
            "environment_servers": {
                "legacy_agent": {
                    "agent_server": {"type": "responses_api_agents", "name": "hermes_legacy"},
                }
            }
        }
        return config

    @staticmethod
    def _materialized_row() -> dict:
        return {
            "task_id": {"taskset": "swe_pro", "task_id": "instance"},
            "task_input": {
                "responses_create_params": {"input": "fix it"},
                "task_data": {"instance_id": "instance"},
            },
        }

    async def test_one_batch_mixes_native_and_compatibility_routed_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A materialized taskset and a legacy flat row travel in one batch, each to its own environment server."""
        post = AsyncMock(return_value=FakeResponse(200, {"reward": 1.0}))
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = self._mixed_batch_config()
        materialized = self._materialized_row()
        flat = self._row() | {AGENT_REF_KEY_NAME: {"name": "hermes_legacy"}}
        del flat[ATTEMPT_INDEX_KEY_NAME]
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=1,
        )
        assert config.environment_routing_mode == "agent"

        rows = RolloutCollectionHelper._preprocess_raw_rows(
            [(0, orjson.dumps(materialized).decode(), materialized), (1, orjson.dumps(flat).decode(), flat)],
            config,
        )

        # Identity is decided once, at preprocessing, and stamped only on the native row.
        native_row = next(row for row in rows if "task_input" in row)
        flat_row = next(row for row in rows if "task_input" not in row)
        assert native_row[NG_ENVIRONMENT_SERVER_KEY] == "environment"
        assert NG_ENVIRONMENT_SERVER_KEY not in flat_row

        futures = list(RolloutCollectionHelper().run_examples(rows))
        dispatched = [await future for future in futures]

        assert len(dispatched) == 2
        calls = {call.kwargs["server_name"]: call.kwargs["json"] for call in post.await_args_list}
        assert set(calls) == {"environment", "legacy_environment"}
        # The native row goes to its taskset's route as an episode request.
        assert calls["environment"]["task"]["task_id"]["taskset"] == "swe_pro"
        assert calls["environment"]["episode_id"] == {"rollout_id": "0-0", "attempt": 0}
        # The flat row goes to the environment server fronting its agent, as today's flat body.
        assert calls["legacy_environment"] is flat_row
        assert calls["legacy_environment"][AGENT_REF_KEY_NAME] == {"name": "hermes_legacy"}

    def test_materialized_row_requires_a_route_in_agent_mode(self) -> None:
        materialized = self._materialized_row()
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            num_repeats=1,
        )

        with pytest.raises(ValueError, match="No environment server route is configured for taskset 'swe_pro'"):
            RolloutCollectionHelper._preprocess_raw_rows(
                [(0, orjson.dumps(materialized).decode(), materialized)],
                config,
            )

    async def test_legacy_mode_routes_flat_rows_to_one_server_and_materialized_rows_by_taskset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy mode: every flat row to the named server, materialized rows still by taskset, through dispatch."""
        post = AsyncMock(return_value=FakeResponse(200, {"reward": 1.0}))
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = self._mixed_batch_config()
        materialized = self._materialized_row()
        flat = self._row()
        del flat[ATTEMPT_INDEX_KEY_NAME]
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_routing_mode="legacy",
            environment_server_name="legacy_environment",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=1,
        )

        rows = RolloutCollectionHelper._preprocess_raw_rows(
            [(0, orjson.dumps(materialized).decode(), materialized), (1, orjson.dumps(flat).decode(), flat)],
            config,
        )

        native_row = next(row for row in rows if "task_input" in row)
        flat_row = next(row for row in rows if "task_input" not in row)
        assert native_row[NG_ENVIRONMENT_SERVER_KEY] == "environment"
        assert flat_row[NG_ENVIRONMENT_SERVER_KEY] == "legacy_environment"

        # Dispatch validates the flat row against a legacy_agent server that binds no resources server,
        # stamps the agent that server fronts, and sends today's flat body; the native row is unaffected.
        for future in list(RolloutCollectionHelper().run_examples(rows)):
            await future

        calls = {call.kwargs["server_name"]: call.kwargs["json"] for call in post.await_args_list}
        assert set(calls) == {"environment", "legacy_environment"}
        assert calls["environment"]["task"]["task_id"]["taskset"] == "swe_pro"
        assert calls["legacy_environment"] is flat_row
        assert flat_row[AGENT_REF_KEY_NAME] == {"name": "hermes_legacy"}

    @pytest.mark.parametrize(
        "seed_policy",
        [False, {"_default": False}, {"swe_pro": False}, {"swe_pro": False, "_default": True}],
    )
    def test_materialized_row_accepts_disabled_seed_policy(self, seed_policy: bool | dict[str, bool]) -> None:
        """Native repeats preserve task input when their effective seed setting is disabled."""
        materialized = self._materialized_row()
        original = deepcopy(materialized)
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=2,
            num_repeats_add_seed=seed_policy,
        )

        rows = RolloutCollectionHelper._preprocess_raw_rows(
            [(0, orjson.dumps(materialized).decode(), materialized)], config
        )

        assert len(rows) == 2
        assert [row[ROLLOUT_INDEX_KEY_NAME] for row in rows] == [0, 1]
        for row in rows:
            assert row["task_id"] == original["task_id"]
            assert row["task_input"] == original["task_input"]
            assert row[NG_ENVIRONMENT_SERVER_KEY] == "environment"
            assert "responses_create_params" not in row
            assert AGENT_REF_KEY_NAME not in row

    @pytest.mark.parametrize(
        "seed_policy",
        [{"hermes_legacy": True, "_default": False}, {"hermes_legacy": True, "swe_pro": False}],
    )
    def test_mixed_rows_seed_only_the_legacy_agent(self, seed_policy: dict[str, bool]) -> None:
        """One policy can seed legacy repeats without rejecting unseeded native repeats."""
        materialized = self._materialized_row()
        original_task_input = deepcopy(materialized["task_input"])
        flat = {
            AGENT_REF_KEY_NAME: {"name": "hermes_legacy"},
            "responses_create_params": {"input": "fix it"},
        }
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=2,
            num_repeats_add_seed=seed_policy,
        )

        rows = RolloutCollectionHelper._preprocess_raw_rows(
            [(0, orjson.dumps(materialized).decode(), materialized), (1, orjson.dumps(flat).decode(), flat)],
            config,
        )

        native_rows = [row for row in rows if "task_input" in row]
        legacy_rows = [row for row in rows if "task_input" not in row]
        assert len(native_rows) == len(legacy_rows) == 2
        for row in native_rows:
            assert row["task_input"] == original_task_input
            assert "responses_create_params" not in row
            assert row[NG_ENVIRONMENT_SERVER_KEY] == "environment"
        assert [row[ROLLOUT_INDEX_KEY_NAME] for row in native_rows] == [0, 1]
        assert [row[ROLLOUT_INDEX_KEY_NAME] for row in legacy_rows] == [0, 1]
        assert [
            json.loads(row["responses_create_params"]["metadata"]["extra_body"])["seed"] for row in legacy_rows
        ] == [0, 1]
        assert all(row[AGENT_REF_KEY_NAME] == {"name": "hermes_legacy"} for row in legacy_rows)
        assert "metadata" not in flat["responses_create_params"]

    @pytest.mark.parametrize("seed_policy", [True, {"_default": True}, {"swe_pro": True, "_default": False}])
    def test_materialized_row_rejects_num_repeats_add_seed(self, seed_policy: bool | dict[str, bool]) -> None:
        """The seed lives in the top-level prompt, which a materialized row keeps under task_input."""
        materialized = self._materialized_row()
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=2,
            num_repeats_add_seed=seed_policy,
        )

        with pytest.raises(ValueError, match="num_repeats_add_seed is not supported for materialized task rows"):
            RolloutCollectionHelper._preprocess_raw_rows(
                [(0, orjson.dumps(materialized).decode(), materialized)],
                config,
            )

    def test_taskset_mode_stays_native_only(self) -> None:
        flat = self._row()
        config = RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            environment_routing_mode="taskset",
            environment_server_routes={"swe_pro": "environment"},
            num_repeats=1,
        )

        with pytest.raises(ValueError, match="environment_routing_mode=agent to mix"):
            RolloutCollectionHelper._preprocess_raw_rows([(0, orjson.dumps(flat).decode(), flat)], config)

    # -- Identity through native failures, retries and aggregation --------------------------------

    @staticmethod
    def _native_identity(task: str) -> dict:
        return {
            "episode_id": {"rollout_id": f"0-{task}", "attempt": 0},
            "task_id": {"taskset": "swe_pro", "task_id": task},
        }

    def test_episode_failure_reply_becomes_a_sidecar_row(self) -> None:
        reply = self._native_identity("a") | {
            "result": None,
            "failure": {
                "failure_reason": "agent unavailable",
                "terminal": False,
                "stage": "agent",
                "partial_response": {"id": "partial"},
            },
        }
        assert nemo_gym.rollout_collection._is_episode_response(reply)

        record = nemo_gym.rollout_collection._episode_record(reply)

        assert record[NG_FAILURE_CLASS_KEY] == ENVIRONMENT_SERVER_FAILURE_CLASS
        assert record[NG_TERMINAL_KEY] is False
        assert record["_ng_failure_message"] == "agent unavailable"
        assert record["_ng_failure_stage"] == "agent"
        assert record["_ng_failure_partial_response"] == {"id": "partial"}
        assert record[nemo_gym.rollout_collection.NG_TASK_ID_KEY]["task_id"] == "a"
        assert "reward" not in record

        terminal = nemo_gym.rollout_collection._episode_record(
            self._native_identity("a") | {"failure": {"failure_reason": "bad task", "terminal": True}}
        )
        assert terminal[NG_TERMINAL_KEY] is True

    def test_episode_result_is_stored_as_returned_with_only_collector_keys_added(self) -> None:
        """The collector stores any Environment Server result without knowing its fields."""
        observations = {"source": "hermes", "records": [], "gaps": []}
        result = {
            "reward": 1.0,
            "response": {"usage": {"tokens": 3}},
            "mask_sample": False,
            # A verify response may echo dataset columns, including a string task_id.
            "task_id": "dataset-task-7",
            "ng_agent_observations": observations,
        }
        reply = self._native_identity("a") | {"failure": None, "result": result}

        record = nemo_gym.rollout_collection._episode_record(reply)

        assert record == result | {nemo_gym.rollout_collection.NG_TASK_ID_KEY: {"taskset": "swe_pro", "task_id": "a"}}
        # The envelope is not stored twice: no nested result, no episode_id, no second reward.
        assert "result" not in record and "episode_id" not in record and "failure" not in record

    def test_episode_result_is_scored_only_through_its_top_level_reward(self) -> None:
        """Any Environment Server type is scored through top-level reward and reward_components."""
        scored = nemo_gym.rollout_collection._episode_record(
            self._native_identity("a") | {"result": {"reward": 0.5, "reward_components": {"quality": 0.5}}}
        )
        nested = nemo_gym.rollout_collection._episode_record(
            self._native_identity("b") | {"result": {"verification": {"reward": 1.0}}}
        )

        assert (scored["reward"], scored["reward_components"]) == (0.5, {"quality": 0.5})
        # A reward inside another field is stored as data and leaves the record unscored.
        assert "reward" not in nested
        assert nested["verification"] == {"reward": 1.0}

    def test_episode_result_may_not_use_collector_keys(self) -> None:
        reply = self._native_identity("a") | {"result": {"reward": 1.0, "ng_trajectory": {}}}

        with pytest.raises(ValueError, match=r"reserved for rollout collection: \['ng_trajectory'\]"):
            nemo_gym.rollout_collection._episode_record(reply)

    def test_episode_detection_needs_object_identities_and_an_object_failure(self) -> None:
        """An agent reply echoing identity fields as strings is not an episode reply; a bad failure is an error."""
        is_episode = nemo_gym.rollout_collection._is_episode_response
        assert is_episode(self._native_identity("a") | {"failure": {"failure_reason": "x", "terminal": True}})
        assert not is_episode({"episode_id": "0-a", "task_id": "a", "reward": 1.0, "failure": {"reason": "echoed"}})
        assert not is_episode(self._native_identity("a") | {"reward": 1.0})

        with pytest.raises(ValueError, match="non-object failure: 'timeout'"):
            nemo_gym.rollout_collection._episode_record(
                self._native_identity("a") | {"result": None, "failure": "timeout"}
            )

    async def test_legacy_reply_echoing_identity_fields_is_not_projected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Projection is decided by the request the collector sent, not by sniffing the reply."""
        echoed = {"episode_id": {"rollout_id": "0-0", "attempt": 0}, "task_id": {"taskset": "t", "task_id": "x"}}
        post = AsyncMock(return_value=FakeResponse(200, echoed | {"reward": 1.0, "failure": {"reason": "echoed"}}))
        client = install_fake_server_client(monkeypatch, post)
        client.global_config_dict = self._mixed_batch_config()
        flat = self._row() | {AGENT_REF_KEY_NAME: {"name": "hermes_legacy"}}
        del flat[ATTEMPT_INDEX_KEY_NAME]
        flat[TASK_INDEX_KEY_NAME] = 0
        flat[ROLLOUT_INDEX_KEY_NAME] = 0

        _, result = await next(RolloutCollectionHelper().run_examples([flat]))

        assert result["reward"] == 1.0
        assert NG_FAILURE_CLASS_KEY not in result
        assert nemo_gym.rollout_collection._materialized_taskset(flat) is None

    async def test_call_aggregate_metrics_groups_by_environment_server(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Native rows aggregate on the server stamped at dispatch; legacy rows keep their agent's server."""
        agg = AggregateMetrics(agent_metrics={"mean/reward": 1.0}, key_metrics={"mean/reward": 1.0})
        posts: dict[str, list[dict]] = {}

        async def post(server_name: str, url_path: str, json, **kwargs):
            assert url_path == "/aggregate_metrics"
            posts[server_name] = [dict(r) for r in json.verify_responses]
            return FakeResponse(200, agg.model_dump())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = self._mixed_batch_config()

        episode_result = {
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            NG_ENVIRONMENT_SERVER_KEY: "environment",
            nemo_gym.rollout_collection.NG_TASK_ID_KEY: {"taskset": "swe_pro", "task_id": "a"},
            "reward": 1.0,
            "response": {"usage": {"tokens": 3}},
        }
        legacy_result = {
            TASK_INDEX_KEY_NAME: 1,
            ROLLOUT_INDEX_KEY_NAME: 0,
            AGENT_REF_KEY_NAME: {"name": "hermes_legacy"},
            "reward": 0.0,
        }
        rows = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_ENVIRONMENT_SERVER_KEY: "environment"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, AGENT_REF_KEY_NAME: {"name": "hermes_legacy"}},
        ]

        metrics_fpath = await RolloutCollectionHelper()._call_aggregate_metrics(
            [episode_result, legacy_result], rows, tmp_path / "output.jsonl"
        )

        assert set(posts) == {"environment", "legacy_environment"}
        # An episode record is sent as stored; it already has the verify-response shape.
        assert posts["environment"] == [episode_result]
        assert posts["legacy_environment"][0][TASK_INDEX_KEY_NAME] == 1
        written = {entry[NG_ENVIRONMENT_SERVER_KEY]: entry for entry in json.loads(metrics_fpath.read_text())}
        assert set(written) == {"environment", "legacy_environment"}
        # The native group is labelled by the agent its environment server binds.
        assert written["environment"][AGENT_REF_KEY_NAME] == {"name": "hermes"}
        assert written["legacy_environment"][AGENT_REF_KEY_NAME] == {"name": "hermes_legacy"}

    async def test_call_aggregate_metrics_labels_two_servers_of_one_agent_distinctly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A native server and its legacy_agent twin bind the same agent; `gym eval compare` needs distinct labels."""
        agg = AggregateMetrics(agent_metrics={"mean/reward": 1.0}, key_metrics={"mean/reward": 1.0})
        client = install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, agg.model_dump())))
        config = _environment_server_config()
        config["hermes_environment_server"] = {
            "environment_servers": {
                "legacy_agent": {"agent_server": {"type": "responses_api_agents", "name": "hermes"}}
            }
        }
        client.global_config_dict = config
        rows = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_ENVIRONMENT_SERVER_KEY: "environment"},
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                AGENT_REF_KEY_NAME: {"name": "hermes"},
                NG_ENVIRONMENT_SERVER_KEY: "hermes_environment_server",
            },
        ]
        results = [dict(row, reward=1.0) for row in rows]

        metrics_fpath = await RolloutCollectionHelper()._call_aggregate_metrics(
            results, rows, tmp_path / "output.jsonl"
        )

        written = {entry[NG_ENVIRONMENT_SERVER_KEY]: entry for entry in json.loads(metrics_fpath.read_text())}
        # Both servers front hermes, so each is labelled by its own name, whatever order rows arrive in.
        assert written["environment"][AGENT_REF_KEY_NAME] == {"name": "environment"}
        assert written["hermes_environment_server"][AGENT_REF_KEY_NAME] == {"name": "hermes_environment_server"}

    async def test_call_aggregate_metrics_names_a_stamped_server_missing_from_the_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = install_fake_server_client(monkeypatch, AsyncMock())
        client.global_config_dict = _environment_server_config()
        rows = [{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_ENVIRONMENT_SERVER_KEY: "renamed_environment"}]

        with pytest.raises(
            ValueError, match="'renamed_environment', which is not in the running config .*'environment'"
        ):
            await RolloutCollectionHelper()._call_aggregate_metrics(
                [dict(rows[0], reward=1.0)], rows, tmp_path / "output.jsonl"
            )

    async def test_run_from_config_stamps_server_and_result_type_on_agent_routed_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flat row routed by its agent is stamped with the relay that ran it, so readers need not use agent_ref."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_bytes(
            orjson.dumps({"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "hermes_legacy"}})
            + b"\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                assert server_name == "legacy_environment"
                return FakeResponse(200, {"reward": 1.0, "response": {"usage": {"total_tokens": 3}}})
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = self._mixed_batch_config()
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", lambda: client.global_config_dict)

        await RolloutCollectionHelper().run_from_config(
            RolloutCollectionConfig(
                input_jsonl_fpath=str(input_jsonl_fpath),
                output_jsonl_fpath=str(output_jsonl_fpath),
                disable_health_check=True,
            )
        )

        [record] = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert record[NG_ENVIRONMENT_SERVER_KEY] == "legacy_environment"
        assert record[nemo_gym.rollout_collection.NG_RESULT_TYPE_KEY] == "legacy_agent"
        assert record[AGENT_REF_KEY_NAME] == {"name": "hermes_legacy"}

    async def test_progress_reward_averages_only_results_that_report_a_reward(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unscored result, one without a reward, does not dilute the reward progress metrics."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_bytes(
            b"".join(
                orjson.dumps(
                    {
                        "responses_create_params": {"input": [{"role": "user", "content": str(i)}]},
                        AGENT_REF_KEY_NAME: {"name": "hermes_legacy"},
                    }
                )
                + b"\n"
                for i in range(2)
            )
        )
        replies = iter([{"reward": 1.0}, {"response_note": "unscored"}])

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                return FakeResponse(200, next(replies))
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = self._mixed_batch_config()
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_config_dict", lambda: client.global_config_dict)
        exported: list[dict] = []
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_exporters", lambda: [object()])
        monkeypatch.setattr(
            nemo_gym.rollout_collection, "export_metrics", lambda metrics, **_: exported.append(metrics)
        )

        await RolloutCollectionHelper().run_from_config(
            RolloutCollectionConfig(
                input_jsonl_fpath=str(input_jsonl_fpath),
                output_jsonl_fpath=str(tmp_path / "output.jsonl"),
                disable_health_check=True,
            )
        )

        final = [m for m in exported if "progress/hermes_legacy/reward" in m][-1]
        assert final["progress/hermes_legacy/reward"] == 100.0
        assert final["progress/hermes_legacy/reward_lower_bound"] == 100.0

    async def test_run_from_config_sidecars_a_native_failure_and_retries_it_on_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Identity persists: the stamped server is on the result, the sidecar row, and the retry.

        No global-config fallback is installed: a run with routes must read the merged config from the
        environment-server client it sets up, so this test fails if that client is not created.
        """
        global_config = self._mixed_batch_config()
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_bytes(
            b"".join(
                orjson.dumps(self._materialized_row() | {"task_id": {"taskset": "swe_pro", "task_id": task}}) + b"\n"
                for task in ("a", "b")
            )
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        dispatched: list[tuple[str, dict]] = []
        failed_once: set[str] = set()

        async def post(server_name: str, url_path: str, json, **kwargs):
            if url_path == "/run":
                dispatched.append((server_name, json))
                task = json["task"]["task_id"]["task_id"]
                identity = {"episode_id": json["episode_id"], "task_id": json["task"]["task_id"]}
                if task == "a" and task not in failed_once:
                    failed_once.add(task)
                    return FakeResponse(
                        200,
                        identity
                        | {"result": None, "failure": {"failure_reason": "agent unavailable", "terminal": False}},
                    )
                return FakeResponse(
                    200,
                    identity
                    | {
                        "failure": None,
                        "result": {"reward": 1.0, "response": {"usage": {"total_tokens": 3}}},
                    },
                )
            assert url_path == "/aggregate_metrics"
            return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = global_config

        def config(resume: bool) -> RolloutCollectionConfig:
            return RolloutCollectionConfig(
                input_jsonl_fpath=str(input_jsonl_fpath),
                output_jsonl_fpath=str(output_jsonl_fpath),
                environment_server_routes={"swe_pro": "environment"},
                resume_from_cache=resume,
                disable_health_check=True,
            )

        await RolloutCollectionHelper().run_from_config(config(resume=False))

        assert [server for server, _ in dispatched] == ["environment", "environment"]
        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert [r[nemo_gym.rollout_collection.NG_TASK_ID_KEY]["task_id"] for r in persisted] == ["b"]
        assert persisted[0]["reward"] == 1.0
        assert persisted[0][NG_ENVIRONMENT_SERVER_KEY] == "environment"
        assert persisted[0][nemo_gym.rollout_collection.NG_RESULT_TYPE_KEY] == "single_agent"
        assert "result" not in persisted[0] and "episode_id" not in persisted[0]
        failures = [orjson.loads(line) for line in _failures_path_for(output_jsonl_fpath).read_bytes().splitlines()]
        assert len(failures) == 1
        assert failures[0][NG_FAILURE_CLASS_KEY] == ENVIRONMENT_SERVER_FAILURE_CLASS
        assert failures[0][NG_TERMINAL_KEY] is False
        assert failures[0][NG_ENVIRONMENT_SERVER_KEY] == "environment"
        assert failures[0][nemo_gym.rollout_collection.NG_TASK_ID_KEY]["task_id"] == "a"
        metrics_fpath = output_jsonl_fpath.with_stem(output_jsonl_fpath.stem + "_aggregate_metrics").with_suffix(
            ".json"
        )
        entries = orjson.loads(metrics_fpath.read_bytes())
        assert [entry[NG_ENVIRONMENT_SERVER_KEY] for entry in entries] == ["environment"]
        assert entries[0]["key_metrics"]["mean/reward"] == 1.0

        # Resume: only the failed episode is re-dispatched, to the same server, as attempt 1.
        dispatched.clear()
        await RolloutCollectionHelper().run_from_config(config(resume=True))

        assert len(dispatched) == 1
        server_name, body = dispatched[0]
        assert server_name == "environment"
        assert body["task"]["task_id"]["task_id"] == "a"
        assert body["episode_id"]["attempt"] == 1
        persisted = [orjson.loads(line) for line in output_jsonl_fpath.read_bytes().splitlines()]
        assert sorted(r[nemo_gym.rollout_collection.NG_TASK_ID_KEY]["task_id"] for r in persisted) == ["a", "b"]
        assert all(r[NG_ENVIRONMENT_SERVER_KEY] == "environment" for r in persisted)


class TestMultiAgentEnvironmentServers:
    """An environment server can front several agents through fields other than `agent_server`."""

    @staticmethod
    def _config() -> DictConfig:
        def agent() -> dict:
            return {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}}

        def reference(name: str) -> dict:
            return {"type": "responses_api_agents", "name": name}

        def server(**references: dict) -> dict:
            return {"environment_servers": {"conversation": {"entrypoint": "app.py", **references}}}

        return OmegaConf.create(
            {
                "user": agent(),
                "assistant": agent(),
                "outsider": agent(),
                "conversation": server(user_agent=reference("user"), assistant_agent=reference("assistant")),
                "assistant_only": server(assistant_agent=reference("assistant")),
                "relay": server(agent_server=reference("assistant")),
            }
        )

    @pytest.mark.parametrize(
        ("server", "agent"),
        [("relay", "assistant"), ("assistant_only", "assistant"), ("conversation", None)],
        ids=["agent-server", "only-agent", "several-agents"],
    )
    def test_results_are_attributed_to_one_agent_only_when_the_server_has_one(
        self, server: str, agent: str | None
    ) -> None:
        row = {nemo_gym.rollout_collection.NG_ENVIRONMENT_SERVER_KEY: server}

        assert RolloutCollectionHelper._agent_name_for_row(row, self._config()) == agent

    def test_a_row_may_name_any_agent_of_its_environment_server(self) -> None:
        row = {nemo_gym.rollout_collection.NG_ENVIRONMENT_SERVER_KEY: "conversation", "agent_ref": {"name": "user"}}

        RolloutCollectionHelper._validate_environment_servers([row], self._config())

    def test_a_row_naming_another_agent_is_rejected(self) -> None:
        row = {
            nemo_gym.rollout_collection.NG_ENVIRONMENT_SERVER_KEY: "conversation",
            "agent_ref": {"name": "outsider"},
        }

        with pytest.raises(
            ValueError, match="'outsider' is not an agent of environment server 'conversation': 'user', 'assistant'"
        ):
            RolloutCollectionHelper._validate_environment_servers([row], self._config())
