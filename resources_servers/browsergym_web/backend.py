# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BrowserGym implementation of the common web-environment protocol."""

from __future__ import annotations

import importlib
import json
from collections import deque
from pathlib import Path
from typing import Any

from nemo_gym.web.models import (
    WebAction,
    WebArtifactRef,
    WebObservation,
    WebObservationProfile,
    WebStepResult,
    WebTab,
    WebTask,
    WebVerifierResult,
)
from resources_servers.browsergym_web.artifacts import WebArtifactStore
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.profiles import BrowserGymLaunchSpec, resolve_browsergym_profile
from resources_servers.browsergym_web.visualwebarena_compat import configure_evaluator_model


def _json_safe(value: Any) -> Any:
    """Convert NumPy and other BrowserGym values into JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _first_number(value: Any, default: float = 0.0) -> float:
    """Normalize scalar or one-element vector values emitted by BrowserGym."""

    normalized = _json_safe(value)
    while isinstance(normalized, list):
        if not normalized:
            return default
        normalized = normalized[0]
    if normalized is None:
        return default
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return default


class BrowserGymBackend:
    """Own one BrowserGym environment and retain evaluator/evidence state."""

    def __init__(
        self,
        config: BrowserGymWebResourcesServerConfig,
        session_id: str,
        artifacts: WebArtifactStore,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.artifacts = artifacts
        self.env: Any = None
        self.task: WebTask | None = None
        self.spec: BrowserGymLaunchSpec | None = None
        self._observation: WebObservation | None = None
        self._step_index = 0
        self._latest_score = 0.0
        self._terminated = False
        self._truncated = False
        self._evidence: deque[WebArtifactRef] = deque(maxlen=config.max_evidence_screenshots)

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]:
        self.close()
        spec = resolve_browsergym_profile(task)
        env = self._make_environment(spec)
        try:
            raw_observation, raw_info = env.reset(seed=task.seed)
        except Exception:
            env.close()
            raise
        if spec.module == "browsergym.visualwebarena":
            # BrowserGym defers construction of the upstream task until
            # env.reset(). Only then has VisualWebArena installed its legacy
            # DATASET/site mapping and imported the evaluator modules.
            configure_evaluator_model(self.config.visualwebarena_evaluator_model)

        self.env = env
        self.task = task
        self.spec = spec
        self._step_index = 0
        self._latest_score = 0.0
        self._terminated = False
        self._truncated = False
        self._evidence.clear()
        self._observation = self._convert_observation(raw_observation)
        info = _json_safe(raw_info)
        if not isinstance(info, dict):
            info = {"value": info}
        info.update(
            {
                "env_id": spec.env_id,
                "runtime_profile": task.runtime_profile.value,
                "observation_profile": spec.observation_profile.value,
                "verifier_version": self._verifier_version(),
            }
        )
        if task.benchmark.value == "visualwebarena" and self.config.visualwebarena_evaluator_model:
            info["evaluator_model"] = self.config.visualwebarena_evaluator_model
        return self._observation, info

    def observe(self) -> WebObservation:
        if self._observation is None:
            raise RuntimeError("BrowserGym environment has not been reset")
        return self._observation

    def step(self, action: WebAction) -> WebStepResult:
        if self.env is None or self.task is None:
            raise RuntimeError("BrowserGym environment has not been reset")
        if self._terminated or self._truncated:
            raise RuntimeError("BrowserGym episode has already finished")

        try:
            raw_observation, reward, terminated, truncated, raw_info = self.env.step(action.script)
        except ValueError as exc:
            # BrowserGym's strict high-level action mapper raises ValueError for
            # a syntactically valid but non-executable call (for example a bad
            # argument shape or element reference).  This is an agent action
            # failure, not a resource-server outage: return it in the next
            # observation so the policy can correct itself instead of turning
            # the whole rollout into an HTTP 400 infrastructure sidecar.
            self._step_index += 1
            error = f"{type(exc).__name__}: {exc}"
            assert self._observation is not None
            self._observation = self._observation.model_copy(
                update={
                    "last_action": action.script,
                    "last_action_error": error,
                }
            )
            return WebStepResult(
                observation=self._observation,
                execution_ok=False,
                benchmark_reward=0.0,
                terminated=False,
                truncated=False,
                info={"action_error": error},
            )
        self._step_index += 1
        self._latest_score = max(self._latest_score, float(reward or 0.0))
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._observation = self._convert_observation(raw_observation)
        info = _json_safe(raw_info)
        if not isinstance(info, dict):
            info = {"value": info}
        return WebStepResult(
            observation=self._observation,
            execution_ok=not bool(self._observation.last_action_error),
            benchmark_reward=float(reward or 0.0),
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        )

    def evaluate(self, final_answer: str | None = None) -> WebVerifierResult:
        if self.env is None or self.task is None or self.spec is None:
            raise RuntimeError("BrowserGym environment has not been reset")

        if self.spec.external_verifier:
            return WebVerifierResult(
                valid_sample=False,
                failure_kind="external_judge_required",
                evidence=list(self._evidence),
                verifier_version=self.spec.verifier_version,
                metadata={"benchmark": self.task.benchmark.value},
            )

        if not self._terminated and not self._truncated:
            action = WebAction(
                name="send_msg_to_user",
                script=f"send_msg_to_user({(final_answer or '')!r})",
                arguments={"args": [final_answer or ""], "kwargs": {}},
                terminal=True,
                answer=final_answer or "",
            )
            self.step(action)

        score = float(self._latest_score)
        return WebVerifierResult(
            reward=score,
            raw_score=score,
            task_success=score >= 1.0,
            valid_sample=True,
            evidence=list(self._evidence),
            verifier_version=self._verifier_version(),
            metadata={
                "benchmark": self.task.benchmark.value,
                "terminated": self._terminated,
                "truncated": self._truncated,
            },
        )

    def close(self) -> None:
        env, self.env = self.env, None
        if env is not None:
            env.close()
        self.task = None
        self.spec = None
        self._observation = None

    def _make_environment(self, spec: BrowserGymLaunchSpec) -> Any:
        try:
            gymnasium = importlib.import_module("gymnasium")
            importlib.import_module(spec.module)
            action_module = importlib.import_module("browsergym.core.action.highlevel")
        except ImportError as exc:
            raise RuntimeError(
                "BrowserGym runtime dependencies are missing; install this resource server's optional dependencies"
            ) from exc
        action_set = action_module.HighLevelActionSet(
            subsets=list(spec.action_subsets),
            multiaction=True,
            strict=True,
        )
        env_kwargs: dict[str, Any] = {
            "action_mapping": action_set.to_python_code,
            "headless": self.config.headless,
            "pre_observation_delay": self.config.pre_observation_delay,
            "tags_to_mark": self.config.tags_to_mark,
        }
        env_kwargs.update(spec.env_kwargs)
        if self.config.record_video:
            video_dir = Path(self.artifacts.session_dir(self.session_id)) / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            env_kwargs["record_video_dir"] = str(video_dir)
        if spec.task_kwargs:
            env_kwargs["task_kwargs"] = spec.task_kwargs
        return gymnasium.make(spec.env_id, **env_kwargs)

    def _verifier_version(self) -> str:
        if self.spec is None:
            raise RuntimeError("cannot identify verifier without an active task")
        model = self.config.visualwebarena_evaluator_model
        if self.spec.module == "browsergym.visualwebarena" and model:
            return f"{self.spec.verifier_version}:judge={model}"
        return self.spec.verifier_version

    def _convert_observation(self, raw: dict[str, Any]) -> WebObservation:
        if self.task is None or self.spec is None:
            raise RuntimeError("cannot convert an observation without an active task")
        try:
            obs_utils = importlib.import_module("browsergym.utils.obs")
            extra_properties = raw.get("extra_element_properties") or {}
            axtree_text = obs_utils.flatten_axtree_to_str(
                raw.get("axtree_object") or {"nodes": []},
                extra_properties=extra_properties,
                with_som=self.spec.observation_profile == WebObservationProfile.SOM,
            )
        except Exception as exc:  # noqa: BLE001 - preserve a usable observation if AX flattening changes upstream.
            axtree_text = json.dumps(_json_safe(raw.get("axtree_object") or {}), ensure_ascii=False)
            flatten_error = f"{type(exc).__name__}: {exc}"
        else:
            flatten_error = ""

        screenshot = None
        raw_screenshot = raw.get("screenshot")
        if raw_screenshot is not None:
            rendered_screenshot = raw_screenshot
            if self.spec.observation_profile == WebObservationProfile.SOM:
                try:
                    rendered_screenshot = obs_utils.overlay_som(raw_screenshot, extra_properties)
                except Exception:  # noqa: BLE001 - raw screenshot remains valid evidence.
                    rendered_screenshot = raw_screenshot
            screenshot = self.artifacts.save_screenshot(
                self.session_id,
                self._step_index,
                rendered_screenshot,
            )
            if screenshot.artifact is not None:
                self._evidence.append(screenshot.artifact)

        urls = list(raw.get("open_pages_urls") or [])
        titles = list(raw.get("open_pages_titles") or [])
        active_index = int(_first_number(raw.get("active_page_index")))
        tabs = [
            WebTab(
                index=index,
                url=str(url),
                title=str(titles[index]) if index < len(titles) else "",
                active=index == active_index,
            )
            for index, url in enumerate(urls)
        ]
        raw_goal = raw.get("goal_object") or []
        if isinstance(raw_goal, (list, tuple)):
            goal = list(raw_goal)
        else:
            goal = [{"type": "text", "text": str(raw.get("goal") or self.task.intent)}]
        metadata = {
            "benchmark": self.task.benchmark.value,
            "observation_profile": self.spec.observation_profile.value,
            "chat_messages": _json_safe(raw.get("chat_messages") or []),
        }
        if flatten_error:
            metadata["axtree_flatten_error"] = flatten_error

        element_map = _json_safe(raw.get("extra_element_properties") or {})
        if not isinstance(element_map, dict):
            element_map = {}
        return WebObservation(
            goal=_json_safe(goal),
            axtree_text=axtree_text,
            screenshot=screenshot,
            url=str(raw.get("url") or ""),
            tabs=tabs,
            active_tab_index=max(0, active_index),
            element_map=element_map,
            focused_element_id=str(raw.get("focused_element_bid") or ""),
            last_action=str(raw.get("last_action") or ""),
            last_action_error=str(raw.get("last_action_error") or ""),
            elapsed_time=max(0.0, _first_number(raw.get("elapsed_time"))),
            metadata=metadata,
        )
