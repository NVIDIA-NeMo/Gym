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
from copy import deepcopy
from pathlib import Path

import orjson
import pytest

from nemo_gym.batch_status import (
    AGGREGATION_ERROR_KEY,
    BatchManifest,
    BatchStatusTracker,
    observe_materialized_rows,
    validate_batch_manifest,
)
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
)


def materialized_row(
    agent: str,
    task_source: str,
    task_index: int,
    rollout_index: int,
    *,
    content: str = "question",
    seed: int | None = None,
) -> dict:
    response_params = {"input": [{"role": "user", "content": content}]}
    if seed is not None:
        response_params["metadata"] = {"extra_body": json.dumps({"seed": seed})}
    return {
        AGENT_REF_KEY_NAME: {"name": agent},
        TASK_SOURCE_KEY_NAME: task_source,
        TASK_INDEX_KEY_NAME: task_index,
        ROLLOUT_INDEX_KEY_NAME: rollout_index,
        "responses_create_params": response_params,
    }


def manifest_payload(rows: list[dict], metric_keys: dict[str, list[str]] | None = None) -> dict:
    observations = observe_materialized_rows(rows)
    return {
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
                "metric_keys": (metric_keys or {}).get(agent, ["mean/reward"]),
            }
            for agent, observed in observations.items()
        },
    }


def write_manifest(path: Path, rows: list[dict], metric_keys: dict[str, list[str]] | None = None) -> Path:
    path.write_bytes(orjson.dumps(manifest_payload(rows, metric_keys), option=orjson.OPT_INDENT_2))
    return path


class TestObserveMaterializedRows:
    def test_dataset_hash_excludes_repeat_and_runtime_fields(self) -> None:
        first = materialized_row("alpha", "alpha_source", 10, 0, seed=100)
        second = materialized_row("alpha", "alpha_source", 10, 1, seed=101)
        second[ATTEMPT_INDEX_KEY_NAME] = 2
        second[SKILLS_REF_KEY_NAME] = {"hash": "runtime-skill-hash"}

        once = observe_materialized_rows([first])["alpha"]
        repeated = observe_materialized_rows([first, second])["alpha"]

        assert once.dataset_sha256 == repeated.dataset_sha256
        assert once.task_count == repeated.task_count == 1
        assert repeated.rollout_count == 2
        assert repeated.repeat_policy.model_dump() == {"num_repeats": 2, "seeded": True}

        changed = deepcopy(first)
        changed["responses_create_params"]["input"][0]["content"] = "different question"
        assert observe_materialized_rows([changed])["alpha"].dataset_sha256 != once.dataset_sha256

    @pytest.mark.parametrize(
        "rows, message",
        [
            (
                [
                    materialized_row("alpha", "source", 0, 0),
                    materialized_row("alpha", "source", 0, 0),
                ],
                "duplicate materialized rollout identity",
            ),
            (
                [
                    materialized_row("alpha", "source", 0, 0),
                    materialized_row("alpha", "source", 0, 1, seed=1),
                ],
                "seeds on only some",
            ),
            (
                [
                    materialized_row("alpha", "source", 0, 0, seed=1),
                    materialized_row("alpha", "source", 0, 1, seed=1),
                ],
                "duplicate seeds",
            ),
        ],
    )
    def test_rejects_ambiguous_identity_or_repeat_policy(self, rows: list[dict], message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            observe_materialized_rows(rows)

    @pytest.mark.parametrize(
        "rows, message",
        [
            ([{}], "has no agent_ref.name"),
            ([materialized_row("alpha", "source", True, 0)], "invalid task index"),
            ([materialized_row("alpha", "source", 0, -1)], "no stable task/rollout identity"),
            ([materialized_row("alpha", "", 0, 0)], "has no task_source"),
            (
                [
                    materialized_row("alpha", "source", 0, 0, content="first"),
                    materialized_row("alpha", "source", 0, 1, content="second"),
                ],
                "different task content",
            ),
            (
                [
                    materialized_row("alpha", "source", 0, 0),
                    materialized_row("alpha", "source", 0, 1),
                    materialized_row("alpha", "source", 1, 0),
                ],
                "non-uniform repeat policy",
            ),
            ([], "no batch members"),
        ],
    )
    def test_rejects_incomplete_materialized_contract(self, rows: list[dict], message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            observe_materialized_rows(rows)

    def test_dataset_hash_handles_non_seed_extra_body_values(self) -> None:
        malformed = materialized_row("alpha", "source", 0, 0)
        malformed["responses_create_params"]["metadata"] = {"extra_body": "{"}
        assert observe_materialized_rows([malformed])["alpha"].repeat_policy.seeded is False

        seeded = materialized_row("alpha", "source", 0, 0)
        seeded["responses_create_params"]["metadata"] = {"extra_body": json.dumps({"seed": 7, "temperature": 0.5})}
        without_seed = materialized_row("alpha", "source", 0, 0)
        without_seed["responses_create_params"]["metadata"] = {"extra_body": '{"temperature":0.5}'}

        assert (
            observe_materialized_rows([seeded])["alpha"].dataset_sha256
            == observe_materialized_rows([without_seed])["alpha"].dataset_sha256
        )


class TestBatchManifest:
    def test_rejects_secret_or_unknown_fields(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "alpha_source", 0, 0)]
        payload = manifest_payload(rows)
        payload["api_key"] = "must-not-be-accepted"
        manifest_fpath = tmp_path / "batch_manifest.json"
        manifest_fpath.write_bytes(orjson.dumps(payload))

        with pytest.raises(ConfigError, match="Extra inputs are not permitted") as exc_info:
            BatchStatusTracker(manifest_fpath, rows)
        assert "must-not-be-accepted" not in str(exc_info.value)

    def test_rejects_duplicate_agent_names(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "alpha_source", 0, 0)]
        payload = manifest_payload(rows)
        payload["members"]["second-benchmark"] = deepcopy(payload["members"]["alpha-benchmark"])
        manifest_fpath = tmp_path / "batch_manifest.json"
        manifest_fpath.write_bytes(orjson.dumps(payload))

        with pytest.raises(ConfigError, match="unique agent names"):
            BatchStatusTracker(manifest_fpath, rows)

    def test_rejects_materialized_dataset_mismatch(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "alpha_source", 0, 0)]
        payload = manifest_payload(rows)
        payload["members"]["alpha-benchmark"]["dataset_sha256"] = "0" * 64
        manifest_fpath = tmp_path / "batch_manifest.json"
        manifest_fpath.write_bytes(orjson.dumps(payload))

        with pytest.raises(ConfigError, match="dataset_sha256 mismatch"):
            BatchStatusTracker(manifest_fpath, rows)

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("agent_name", " ", "agent_name must not be blank"),
            ("task_sources", [""], "values must not be blank"),
            ("task_sources", ["source", "source"], "values must be unique"),
            ("dataset_sha256", "not-a-digest", "64-character SHA-256"),
            ("expected_rollout_count", 2, "expected_rollout_count must equal"),
        ],
    )
    def test_rejects_invalid_member_fields(self, tmp_path: Path, field: str, value: object, message: str) -> None:
        rows = [materialized_row("alpha", "source", 0, 0)]
        payload = manifest_payload(rows)
        payload["members"]["alpha-benchmark"][field] = value
        manifest_fpath = tmp_path / f"{field}.json"
        manifest_fpath.write_bytes(orjson.dumps(payload))

        with pytest.raises(ConfigError, match=message):
            BatchStatusTracker(manifest_fpath, rows)

    def test_rejects_blank_benchmark_slug(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "source", 0, 0)]
        payload = manifest_payload(rows)
        payload["members"][" "] = payload["members"].pop("alpha-benchmark")
        manifest_fpath = tmp_path / "batch_manifest.json"
        manifest_fpath.write_bytes(orjson.dumps(payload))

        with pytest.raises(ConfigError, match="benchmark slugs must not be blank"):
            BatchStatusTracker(manifest_fpath, rows)

    def test_wraps_missing_and_malformed_manifest_errors(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "source", 0, 0)]

        with pytest.raises(ConfigError, match="Cannot read batch manifest"):
            BatchStatusTracker(tmp_path / "missing.json", rows)

        malformed_fpath = tmp_path / "malformed.json"
        malformed_fpath.write_text("{")
        with pytest.raises(ConfigError, match="is not valid JSON"):
            BatchStatusTracker(malformed_fpath, rows)

    def test_reports_missing_and_unexpected_materialized_agents(self) -> None:
        alpha_rows = [materialized_row("alpha", "alpha_source", 0, 0)]
        beta_rows = [materialized_row("beta", "beta_source", 0, 0)]
        manifest = BatchManifest.model_validate(manifest_payload(alpha_rows))

        with pytest.raises(ConfigError) as exc_info:
            validate_batch_manifest(manifest, observe_materialized_rows(beta_rows))

        assert "manifest agents missing" in str(exc_info.value)
        assert "agents absent from the manifest" in str(exc_info.value)


class TestBatchStatusTracker:
    def test_writes_compact_per_agent_status_without_error_messages(self, tmp_path: Path) -> None:
        rows = [
            materialized_row("alpha", "alpha_source", 0, 0),
            materialized_row("beta", "beta_source", 1, 0),
        ]
        manifest_fpath = write_manifest(tmp_path / "batch_manifest.json", rows)
        tracker = BatchStatusTracker(manifest_fpath, rows, write_interval_seconds=0)
        completed = [{**rows[0], "reward": 1.0}]
        failures = [{**rows[1], "_ng_failure_class": "agent_request_failed"}]
        metrics_fpath = tmp_path / "rollouts_aggregate_metrics.json"
        metrics_fpath.write_bytes(
            orjson.dumps(
                [
                    {
                        AGENT_REF_KEY_NAME: {"name": "alpha"},
                        "agent_metrics": {"mean/reward": 1.0},
                        "key_metrics": {"mean/reward": 1.0},
                        "group_level_metrics": [],
                        "repeat_level_metrics": [],
                    },
                    {
                        AGENT_REF_KEY_NAME: {"name": "beta"},
                        "agent_metrics": {},
                        "key_metrics": {},
                        "group_level_metrics": [],
                        "repeat_level_metrics": [],
                        AGGREGATION_ERROR_KEY: {
                            "type": "ClientResponseError",
                            "message": "secret-token-must-not-reach-status",
                            "http_status": 500,
                        },
                    },
                ]
            )
        )

        assert tracker.write_status(completed, failures, aggregate_metrics_fpath=metrics_fpath, force=True)

        raw_status = tracker.status_fpath.read_text()
        status = json.loads(raw_status)
        assert "secret-token-must-not-reach-status" not in raw_status
        assert set(status["members"]) == {"alpha", "beta"}
        assert status["members"]["alpha"]["completed_rollout_count"] == 1
        assert status["members"]["alpha"]["aggregation_status"] == "complete"
        assert status["members"]["beta"]["completed_rollout_count"] == 0
        assert status["members"]["beta"]["failures_by_class"] == {"agent_request_failed": 1}
        assert status["members"]["beta"]["aggregation_status"] == "error"
        assert status["members"]["beta"][AGGREGATION_ERROR_KEY] == {
            "type": "ClientResponseError",
            "http_status": 500,
        }
        assert not list(tmp_path.glob(".batch_status.json.*.tmp"))

    def test_reports_missing_expected_metric_keys(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "alpha_source", 0, 0)]
        manifest_fpath = write_manifest(
            tmp_path / "batch_manifest.json", rows, metric_keys={"alpha": ["mean/reward", "pass@1"]}
        )
        tracker = BatchStatusTracker(manifest_fpath, rows)
        metrics_fpath = tmp_path / "metrics.json"
        metrics_fpath.write_bytes(
            orjson.dumps(
                [
                    {
                        AGENT_REF_KEY_NAME: {"name": "alpha"},
                        "agent_metrics": {"mean/reward": 1.0},
                        "key_metrics": {"mean/reward": 1.0},
                        "group_level_metrics": [],
                        "repeat_level_metrics": [],
                    }
                ]
            )
        )

        status = tracker.build_status(rows, [], aggregate_metrics_fpath=metrics_fpath)

        assert status["members"]["alpha"]["aggregation_status"] == "error"
        assert status["members"]["alpha"][AGGREGATION_ERROR_KEY] == {
            "type": "MissingExpectedMetricKeys",
            "missing_metric_keys": ["pass@1"],
        }

    def test_reports_pending_deferred_and_resolved_failure_status(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "source", 0, 0)]
        tracker = BatchStatusTracker(write_manifest(tmp_path / "batch_manifest.json", rows), rows)
        failure = {**rows[0], "_ng_failure_class": "agent_request_failed"}

        pending = tracker.build_status(rows, [failure])
        deferred = tracker.build_status([], [failure], aggregation_deferred=True)

        assert pending["members"]["alpha"]["aggregation_status"] == "pending"
        assert pending["members"]["alpha"]["failures_by_class"] == {}
        assert deferred["members"]["alpha"]["aggregation_status"] == "deferred"
        assert deferred["members"]["alpha"]["failures_by_class"] == {"agent_request_failed": 1}

    @pytest.mark.parametrize(
        "payload",
        [
            b"{",
            orjson.dumps({}),
            orjson.dumps(["not-an-entry"]),
            orjson.dumps([{}]),
            orjson.dumps(
                [
                    {AGENT_REF_KEY_NAME: {"name": "alpha"}},
                    {AGENT_REF_KEY_NAME: {"name": "alpha"}},
                ]
            ),
        ],
    )
    def test_reports_invalid_aggregate_metrics_artifacts(self, tmp_path: Path, payload: bytes) -> None:
        rows = [materialized_row("alpha", "source", 0, 0)]
        tracker = BatchStatusTracker(write_manifest(tmp_path / "batch_manifest.json", rows), rows)
        metrics_fpath = tmp_path / "metrics.json"
        metrics_fpath.write_bytes(payload)

        status = tracker.build_status(rows, [], aggregate_metrics_fpath=metrics_fpath)

        assert status["members"]["alpha"]["aggregation_status"] == "error"
        assert status["members"]["alpha"][AGGREGATION_ERROR_KEY] == {"type": "InvalidAggregateMetricsArtifact"}

    def test_reports_missing_agent_entry_and_sanitizes_non_mapping_error(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "source", 0, 0)]
        tracker = BatchStatusTracker(write_manifest(tmp_path / "batch_manifest.json", rows), rows)
        metrics_fpath = tmp_path / "metrics.json"
        metrics_fpath.write_bytes(orjson.dumps([{AGENT_REF_KEY_NAME: {"name": "beta"}}]))

        missing = tracker.build_status(rows, [], aggregate_metrics_fpath=metrics_fpath)
        assert missing["members"]["alpha"][AGGREGATION_ERROR_KEY] == {"type": "MissingAggregateMetricsEntry"}

        metrics_fpath.write_bytes(
            orjson.dumps(
                [
                    {
                        AGENT_REF_KEY_NAME: {"name": "alpha"},
                        "key_metrics": {},
                        AGGREGATION_ERROR_KEY: "do-not-copy-this-value",
                    }
                ]
            )
        )
        sanitized = tracker.build_status(rows, [], aggregate_metrics_fpath=metrics_fpath)
        assert sanitized["members"]["alpha"][AGGREGATION_ERROR_KEY] == {"type": "AggregationError"}

    def test_throttles_redundant_status_writes_but_not_first_progress(self, tmp_path: Path) -> None:
        rows = [materialized_row("alpha", "source", 0, 0)]
        tracker = BatchStatusTracker(
            write_manifest(tmp_path / "batch_manifest.json", rows), rows, write_interval_seconds=3600
        )

        assert tracker.write_status([], [], force=True)
        assert tracker.write_status(rows, [])
        assert not tracker.write_status(rows, [])
