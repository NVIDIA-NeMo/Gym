# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved run failure accounting without scanning rollout records or attempt history."""

from pathlib import Path

import orjson
from pydantic import ValidationError

from nemo_gym.eval.comparison.suite.schema import SuiteFailureCounts, SuiteFailureSummary
from nemo_gym.rollouts.rollout_records import coverage_path_for


_COUNT_KEYS = {
    "total": "failed",
    "final": "failed_final",
    "not_final": "failed_not_final",
    "by_class": "failed_by_class",
}


def read_saved_failures(path: Path) -> SuiteFailureSummary | None:
    """Read this rollout file's existing coverage snapshot; absent accounting remains unknown.

    The saved counts cover the entire rollout file, including every agent and benchmark.
    This reader neither refreshes the snapshot nor interprets historical failure rows.
    """
    source = coverage_path_for(path)
    try:
        payload = orjson.loads(source.read_bytes())
        if not isinstance(payload, dict):
            raise ValueError("coverage report must be a JSON object")
        version = payload.get("schema_version")
        if type(version) is not int or version not in (1, 2):
            raise ValueError(f"unsupported coverage schema version {version!r}")
        if any(payload.get(name) is None for name in _COUNT_KEYS.values()):
            raise ValueError("failure counts were not recorded")
        counts = SuiteFailureCounts.model_validate({field: payload[name] for field, name in _COUNT_KEYS.items()})
    except FileNotFoundError:
        return None
    except ValidationError as error:
        detail = error.errors(include_url=False, include_input=False)[0]
        parts = list(detail["loc"])
        if parts:
            parts[0] = _COUNT_KEYS.get(parts[0], parts[0])
        location = ".".join(map(str, parts))
        reason = f"invalid failure counts{f' at {location}' if location else ''}: {detail['msg']}"
        return SuiteFailureSummary(source=source, unavailable_reason=reason)
    except (OSError, ValueError) as error:
        reason = " ".join(str(error).split())
        return SuiteFailureSummary(source=source, unavailable_reason=reason)
    return SuiteFailureSummary(source=source, counts=counts)


def saved_failure_warning(summary: SuiteFailureSummary) -> str | None:
    """Summarize positive saved counts or unavailable accounting in one advisory line."""
    prefix = "Saved failures (entire rollout file): "
    if summary.unavailable_reason is not None:
        reason = " ".join(summary.unavailable_reason.split())
        if len(reason) > 240:
            reason = reason[:237] + "..."
        return prefix + f"unavailable ({reason})."
    counts = summary.counts
    if counts is None or counts.total == 0:
        return None
    details = f"{counts.final} final, {counts.not_final} not final"
    classes = sorted(
        ((name, count) for name, count in counts.by_class.items() if count), key=lambda item: (-item[1], item[0])
    )
    if classes:
        details += "; " + ", ".join(f"{name}={count}" for name, count in classes[:3])
        if len(classes) > 3:
            details += f", +{len(classes) - 3} more classes"
    return prefix + f"{counts.total} ({details})."
