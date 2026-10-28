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
import hashlib
import json
import logging
import re
import tempfile
from itertools import batched
from os import environ
from pathlib import Path
from time import gmtime, strftime, time
from typing import Any, ClassVar, Iterator, Optional

import orjson
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.utils.validation import (
    MAX_ENTITY_KEY_LENGTH,
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
)
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.config_types import MLFlowConfig
from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.exporters.base import BaseExporter


logger = logging.getLogger(__name__)


def _flatten_config(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (dotted_key, leaf) pairs. MLflow params are flat, unlike W&B's nested config."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _flatten_config(inner, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            yield from _flatten_config(inner, f"{prefix}[{index}]")
    elif prefix:
        yield prefix, value


_INVALID_KEY_CHARS = re.compile(r"[^/\w.\- :]")
_LEADING_DOTS = re.compile(r"^\.+")
# Stands in for the elided middle of an over-long name; must itself be an accepted character.
_ELISION_MARKER = "___"


def _normalize_key_path(key: str) -> str:
    """Neutralize only the path segments MLflow rejects, leaving the rest of the name intact.

    MLflow requires an already-normalized relative path, so it rejects empty segments (from
    leading, doubled or trailing slashes) and the traversal forms `.` and `..`.
    """
    segments = [segment for segment in key.split("/") if segment]
    # An all-dot segment is a traversal segment; same length keeps neighbouring names distinct.
    segments = ["_" * len(segment) if set(segment) == {"."} else segment for segment in segments]
    # A surviving name may still open with dots ("...x"), which MLflow also reads as traversal.
    return _LEADING_DOTS.sub(lambda match: "_" * len(match.group()), "/".join(segments))


def _elide_key(key: str, limit: int = MAX_ENTITY_KEY_LENGTH) -> str:
    """Cut an over-long name in the middle, keeping both ends.

    Gym names carry their identity at the edges — benchmark and agent in the prefix, the statistic
    in the suffix (`gpqa.../simple_agent/mean/reward`) — so the middle is what can be spared.
    """
    if len(key) <= limit:
        return key
    kept = limit - len(_ELISION_MARKER)
    # Bias the head, which carries the benchmark and agent names.
    head, tail = kept - kept // 2, kept // 2
    return key[:head] + _ELISION_MARKER + (key[-tail:] if tail else "")


def _sanitize_key(key: str) -> Optional[str]:
    """Map a metric/param/tag name onto what MLflow accepts: character set, path shape, length.

    Gym emits names MLflow rejects, e.g. `pass@1` from pass@k metrics. Returns None for a name
    made only of separators, leaving nothing to sanitize into. Dropping that one entry is
    deliberate: MLflow rejects an entire `log_batch` when any single name is invalid.
    """
    # `@` is spelled out rather than blanked to `_`, so `pass@1` stays readable as `pass_at_1`.
    sanitized = _normalize_key_path(_INVALID_KEY_CHARS.sub("_", key.replace("@", "_at_")))
    # Length is enforced last because `_at_` lengthens a name. Re-normalize: splicing two ends
    # together can reintroduce a path shape MLflow rejects.
    sanitized = _normalize_key_path(_elide_key(sanitized))
    if not sanitized:
        logger.warning(f"Dropping entry {key!r}: nothing is left of the name once sanitized.")
        return None
    return sanitized


class MLflowExporter(BaseExporter):
    """MLflow backend.

    Configured by `mlflow_tracking_uri`, `mlflow_experiment_name` and `mlflow_run_name` in the
    global config, plus `mlflow_tracking_token` for servers that require authentication.

    Uses `MlflowClient` with an explicit run id rather than the fluent `mlflow.*` API, which
    tracks the active run in thread-local state that Gym's async call sites don't share.
    """

    name: ClassVar[str] = "mlflow"

    CONFIG_ARTIFACT_FILE: ClassVar[str] = "global_config.json"
    # Where a reopened run records each later segment's config; the first segment's stays in CONFIG_ARTIFACT_FILE.
    SEGMENT_CONFIG_ARTIFACT_DIR: ClassVar[str] = "config"
    # Can't write .jsonl: `log_table` accepts only .json/.parquet, and writes one table document.
    ROLLOUTS_ARTIFACT_FILE: ClassVar[str] = "rollouts.json"
    RESUME_COUNT_TAG: ClassVar[str] = "gym_resume_count"

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.config = MLFlowConfig.model_validate(global_config_dict)
        self.client: Optional[MlflowClient] = None
        self.run_id: Optional[str] = None
        # The reopened run's status and resume count before this job reopened it, restored by `start_over`.
        self._prior_status: Optional[str] = None
        self._prior_resumes: Optional[str] = None

    def setup(self) -> None:
        # The MLflow SDK reads the bearer token from the environment, not from client kwargs.
        # Left unset for unauthenticated servers so no stray Authorization header is sent.
        if self.config.mlflow_tracking_token:
            environ["MLFLOW_TRACKING_TOKEN"] = self.config.mlflow_tracking_token
        self.client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
        experiment_id = self._experiment_id(self.config.mlflow_experiment_name)
        # create_run takes tags as a dict; mlflow.note.content is the key MLflow renders as the run description.
        tags = dict(self.config.mlflow_run_tags)
        if self.config.mlflow_run_description:
            tags["mlflow.note.content"] = self.config.mlflow_run_description
        # A resumed or resubmitted run reopens the MLflow run recorded next to its output, unless that run is gone.
        recorded = self._recorded_run_id()
        if recorded is not None and self._is_active(recorded):
            self.run_id = recorded
            run = self.client.get_run(self.run_id)
            self._prior_status = run.info.status
            self._prior_resumes = run.data.tags.get(self.RESUME_COUNT_TAG)
            self.client.update_run(self.run_id, status="RUNNING")
            # Counts the times this run was reopened, so a reader knows its metrics span several jobs.
            self.client.set_tag(self.run_id, self.RESUME_COUNT_TAG, str(int(self._prior_resumes or 0) + 1))
            self.reopened = True
            return
        self.run_id = self.client.create_run(
            experiment_id, run_name=self.config.mlflow_run_name, tags=tags or None
        ).info.run_id
        self._record_run_id(self.run_id)

    def teardown(self) -> None:
        self._flush_pending_config()
        if self.client is not None and self.run_id is not None:
            self.client.set_terminated(self.run_id, status="FINISHED")
        self.client = None
        self.run_id = None
        self.reopened = False

    def _close_reopened(self) -> None:
        # Leave the reopened run as the earlier job left it: no rollout or metric of this job reached it. Let go of
        # it first, so the new output never logs to it, even when this fails.
        client, run_id = self._active()
        self.client, self.run_id, self.reopened = None, None, False
        client.update_run(run_id, status=self._prior_status)
        if self._prior_resumes is None:
            client.delete_tag(run_id, self.RESUME_COUNT_TAG)
        else:
            client.set_tag(run_id, self.RESUME_COUNT_TAG, self._prior_resumes)

    def _is_active(self, run_id: str) -> bool:
        """Whether the recorded run still exists and is not deleted; a new run takes the place of one that is not."""
        try:
            return self.client.get_run(run_id).info.lifecycle_stage == "active"
        except MlflowException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                return False
            raise

    def _experiment_id(self, name: str) -> str:
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self.client.create_experiment(name)
        except MlflowException:
            # Lost a create race against a concurrent shard; the experiment now exists.
            return self.client.get_experiment_by_name(name).experiment_id

    def _active(self) -> tuple[MlflowClient, str]:
        if self.client is None or self.run_id is None:
            raise RuntimeError("MLflow run is not open; call setup() before logging.")
        return self.client, self.run_id

    def _log_config(self, config_dict: DictConfig) -> None:
        client, run_id = self._active()
        container = OmegaConf.to_container(config_dict)

        params = [
            Param(name, str(value)[:MAX_PARAM_VAL_LENGTH])
            for key, value in _flatten_config(container)
            if (name := _sanitize_key(key)) is not None
        ]

        if self.reopened:
            # A later segment's config differs (e.g. the policy server's host and port), and MLflow refuses to
            # change a logged param. So the first segment's params and config artifact stay as they are, only new
            # keys become params, and the full config goes to a file of its own. Its name needs no read of the
            # run's artifacts: the start time orders segments, and the content digest keeps two segments that
            # start in the same second from overwriting each other.
            logged = client.get_run(run_id).data.params
            params = [param for param in params if param.key not in logged]
            digest = hashlib.sha256(json.dumps(container, sort_keys=True, default=str).encode()).hexdigest()[:12]
            started = strftime("%Y%m%dT%H%M%SZ", gmtime())
            client.log_dict(run_id, container, f"{self.SEGMENT_CONFIG_ARTIFACT_DIR}/segment_{started}_{digest}.json")
        else:
            # Log the whole config first: params are lossy (flat, truncated, length-capped), so the
            # artifact is the record of what actually ran.
            client.log_dict(run_id, container, self.CONFIG_ARTIFACT_FILE)

        for batch in batched(params, MAX_PARAMS_TAGS_PER_BATCH):
            client.log_batch(run_id, params=batch)

    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        client, run_id = self._active()
        timestamp = int(time() * 1000)

        # MLflow metrics must be numeric, so strings become tags rather than being lost and None is
        # dropped.
        numeric: list[Metric] = []
        tags: list[RunTag] = []
        for key, value in metrics.items():
            name = _sanitize_key(key)
            if name is None:
                continue
            if isinstance(value, bool) or isinstance(value, (int, float)):
                numeric.append(Metric(name, float(value), timestamp, step or 0))
            elif isinstance(value, str):
                tags.append(RunTag(name, value[:MAX_PARAM_VAL_LENGTH]))

        for batch in batched(numeric, MAX_METRICS_PER_BATCH):
            client.log_batch(run_id, metrics=batch)
        for batch in batched(tags, MAX_PARAMS_TAGS_PER_BATCH):
            client.log_batch(run_id, tags=batch)

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        client, run_id = self._active()
        if self.reopened:
            # A resumed run is handed every rollout that landed in its output, and `log_table` appends to the table
            # an earlier job logged: skip the landed rollouts that table holds. A failed or no-persist attempt in the
            # table does not count, so a rollout that lands on resume is logged next to it.
            landed = self._landed_rollout_keys(client, run_id)
            rollouts = [rollout for rollout in rollouts if not (_landed(rollout) and _rollout_key(rollout) in landed)]
            if not rollouts:
                return
        # One JSON blob per row, matching the W&B rollouts table: rollouts have no stable column
        # set across environments. `log_table` appends when the artifact already exists.
        rows = [orjson.dumps(rollout).decode() for rollout in rollouts]
        client.log_table(run_id, data={"Rollout": rows}, artifact_file=self.ROLLOUTS_ARTIFACT_FILE)

    def _landed_rollout_keys(self, client: MlflowClient, run_id: str) -> set[tuple[Any, Any]]:
        # Read from this run, as `log_table` does when it appends: `load_table` searches the fluent API's global
        # tracking URI, not this client's.
        if self.ROLLOUTS_ARTIFACT_FILE not in {artifact.path for artifact in client.list_artifacts(run_id)}:
            return set()
        with tempfile.TemporaryDirectory() as tmpdir:
            table = json.loads(
                Path(client.download_artifacts(run_id, self.ROLLOUTS_ARTIFACT_FILE, tmpdir)).read_text()
            )
        column = table["columns"].index("Rollout")
        rollouts = (orjson.loads(row[column]) for row in table["data"])
        return {_rollout_key(rollout) for rollout in rollouts if _landed(rollout)}


def _rollout_key(rollout: dict[str, Any]) -> tuple[Any, Any]:
    return rollout.get(TASK_INDEX_KEY_NAME), rollout.get(ROLLOUT_INDEX_KEY_NAME)


def _landed(rollout: dict[str, Any]) -> bool:
    """Whether the rollout landed in the output: a failed attempt carries its failure class, and a no-persist one is
    written nowhere."""
    # Imported here: rollout collection imports the exporters.
    from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY

    return rollout.get(NG_FAILURE_CLASS_KEY) is None and not rollout.get(NG_NO_PERSIST_KEY)
