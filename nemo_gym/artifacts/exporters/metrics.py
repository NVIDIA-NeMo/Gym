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
import hashlib
import json
import math
import os
import re
from typing import Any, Iterable

from nemo_gym.config.keys import AGENT_REF_KEY_NAME


_PRIMITIVE_TYPES = (bool, int, float, str, type(None))


# MLflow's metric, parameter and tag key limit; checked against the optional SDK in its exporter tests.
_MLFLOW_KEY_LENGTH_LIMIT = 250


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


def _elide_key(key: str, limit: int = _MLFLOW_KEY_LENGTH_LIMIT) -> str:
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


def _canonical_mlflow_key(key: str, *, limit: int = _MLFLOW_KEY_LENGTH_LIMIT) -> str | None:
    """The backend key shared by health preflight and MLflow delivery, without importing the optional SDK."""
    sanitized = _normalize_key_path(_INVALID_KEY_CHARS.sub("_", key.replace("@", "_at_")))
    sanitized = _normalize_key_path(_elide_key(sanitized, limit))
    return sanitized or None


def health_source_artifacts(artifacts, version: dict[str, Any]) -> dict[str, Any]:
    """Bind derived health to the same saved inputs, Store selection and underlying health report."""
    result = {k: v for k, v in version.items() if k != "aggregate_metrics_json_fpath"}
    output = artifacts.rollouts_jsonl_fpath
    for suffix in ("manifest.json", "attempts.jsonl"):
        path = output.with_name(f"{output.stem}_{suffix}")
        try:
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
            current = path.stat()
            identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            if identity(before) != identity(after) or identity(after) != identity(current):
                raise RuntimeError("Store evidence changed while binding health metrics.")
            result[suffix] = digest
        except FileNotFoundError:
            result[suffix] = None
    path = output.parent / "quality_summary.json"
    try:
        summary = json.loads(path.read_bytes())
        # Exclude only this derived section, avoiding a self-referential digest.
        stats = summary.get("run", {}).get("stats", {})
        if isinstance(stats, dict):
            stats.pop("acceptance_health", None)
        result["quality_summary"] = hashlib.sha256(
            json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except FileNotFoundError:
        result["quality_summary"] = None
    except (ValueError, AttributeError) as exc:
        raise ValueError("Existing health summary cannot be bound to health metrics.") from exc
    return result


def flatten_aggregate_metrics(
    agent_entries: Iterable[dict[str, Any]], *, source_artifacts: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The scalar metrics a tracking backend gets from per-agent aggregate-metrics entries.

    Live runs and post-run export both build their metrics here, so a run looks the same in the backend
    either way: `<agent>/<metric>` and `key_metrics/<agent>/<metric>`, primitive values only.
    """
    metrics: dict[str, Any] = {}
    health_seen = False
    health_metrics: dict[str, Any] = {}
    for entry in agent_entries:
        if "_ng_run_health_metrics" in entry:
            if health_seen:
                raise ValueError("Duplicate run health metric snapshots.")
            health_seen = True
            payload = entry["_ng_run_health_metrics"]
            if source_artifacts is not None:
                expected = {k: v for k, v in source_artifacts.items() if k != "aggregate_metrics_json_fpath"}
                if not isinstance(payload, dict) or payload.get("source_artifacts") != expected:
                    raise ValueError("Run health metrics do not match the current source artifacts.")
                values = payload.get("metrics")
                if (
                    not isinstance(values, dict)
                    or not values
                    or any(
                        not isinstance(k, str)
                        or not k.startswith("health/")
                        or type(v) not in (int, float)
                        or not math.isfinite(v)
                        for k, v in values.items()
                    )
                ):
                    raise ValueError("Run health metrics are not finite scalar observations.")
                health_metrics.update(values)
        agent_name = entry[AGENT_REF_KEY_NAME]["name"]
        metrics.update(
            {
                f"{agent_name}/{k}": v
                for k, v in entry.get("agent_metrics", {}).items()
                if isinstance(v, _PRIMITIVE_TYPES)
            }
        )
        metrics.update(
            {
                f"key_metrics/{agent_name}/{k}": v
                for k, v in entry.get("key_metrics", {}).items()
                if isinstance(v, _PRIMITIVE_TYPES)
            }
        )
    if {_canonical_mlflow_key(key) for key in health_metrics} & {_canonical_mlflow_key(key) for key in metrics}:
        raise ValueError("Run health metric names collide with scientific metric names.")
    return {**metrics, **health_metrics}
