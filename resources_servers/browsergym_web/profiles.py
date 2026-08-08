# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-specific BrowserGym launch profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_gym.web.models import (
    WebActionProfile,
    WebBenchmark,
    WebObservationProfile,
    WebRuntimeProfile,
    WebTask,
)


@dataclass(frozen=True, slots=True)
class BrowserGymLaunchSpec:
    module: str
    env_id: str
    action_subsets: tuple[str, ...]
    observation_profile: WebObservationProfile
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    external_verifier: bool = False
    verifier_version: str = "browsergym-v0.14.3"


_SAFE_ENV_OVERRIDES = frozenset(
    {
        "locale",
        "pre_observation_delay",
        "pw_chromium_kwargs",
        "pw_context_kwargs",
        "slow_mo",
        "timeout",
        "timezone_id",
        "viewport",
    }
)


def _safe_env_kwargs(task: WebTask) -> dict[str, Any]:
    supplied = task.task_kwargs.get("env_kwargs") or {}
    if not isinstance(supplied, dict):
        raise ValueError("task_kwargs.env_kwargs must be an object")
    unknown = set(supplied) - _SAFE_ENV_OVERRIDES
    if unknown:
        raise ValueError(f"unsupported BrowserGym environment override(s): {sorted(unknown)}")
    return dict(supplied)


def _gym_id(task: WebTask, prefix: str) -> str:
    explicit = task.task_kwargs.get("gym_id")
    if explicit:
        return str(explicit)
    if task.task_id.startswith("browsergym/"):
        return task.task_id
    return f"browsergym/{prefix}.{task.task_id}"


def resolve_browsergym_profile(task: WebTask) -> BrowserGymLaunchSpec:
    if task.runtime_profile != WebRuntimeProfile.BROWSERGYM:
        raise ValueError(f"task requests unsupported runtime profile: {task.runtime_profile}")

    env_kwargs = _safe_env_kwargs(task)
    if task.benchmark == WebBenchmark.WEBARENA:
        if task.action_profile != WebActionProfile.BROWSERGYM_HIGHLEVEL:
            raise ValueError("WebArena requires the browsergym_highlevel action profile")
        return BrowserGymLaunchSpec(
            module="browsergym.webarena",
            env_id=_gym_id(task, "webarena"),
            action_subsets=("webarena",),
            observation_profile=task.observation_profile or WebObservationProfile.A11Y,
            env_kwargs=env_kwargs,
            verifier_version="browsergym-v0.14.3:webarena",
        )

    if task.benchmark == WebBenchmark.VISUALWEBARENA:
        if task.action_profile != WebActionProfile.BROWSERGYM_HIGHLEVEL:
            raise ValueError("VisualWebArena requires the browsergym_highlevel action profile")
        return BrowserGymLaunchSpec(
            module="browsergym.visualwebarena",
            env_id=_gym_id(task, "visualwebarena"),
            action_subsets=("visualwebarena",),
            observation_profile=task.observation_profile or WebObservationProfile.SOM,
            env_kwargs=env_kwargs,
            verifier_version="browsergym-v0.14.3:visualwebarena",
        )

    if task.benchmark == WebBenchmark.WEBVOYAGER:
        start_url = next((url for url in task.start_urls if url), None)
        if start_url is None:
            start_url = task.task_kwargs.get("start_url")
        if not start_url:
            raise ValueError("WebVoyager requires at least one start URL")
        task_kwargs = dict(task.task_kwargs.get("browsergym_task_kwargs") or {})
        task_kwargs.update({"start_url": str(start_url), "goal": task.intent})
        return BrowserGymLaunchSpec(
            module="browsergym.core",
            env_id="browsergym/openended",
            action_subsets=("webarena",),
            observation_profile=task.observation_profile or WebObservationProfile.SOM,
            task_kwargs=task_kwargs,
            env_kwargs=env_kwargs,
            external_verifier=True,
            verifier_version="webvoyager-llm-judge-v1",
        )

    raise ValueError(f"unsupported web benchmark: {task.benchmark}")
