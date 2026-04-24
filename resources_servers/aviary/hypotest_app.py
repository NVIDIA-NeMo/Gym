# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from typing import Any

from fastapi import Request
from hypotest.dataset_server import HypotestDataset, HypotestDatasetConfig
from hypotest.env.interpreter_env import InterpreterEnv
from pydantic import Field, model_validator

from resources_servers.aviary.app import AviaryResourcesServer
from resources_servers.aviary.schemas import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryCloseRequest,
    AviaryCloseResponse,
    AviaryResourcesServerConfig,
)


class HypotestServerConfig(AviaryResourcesServerConfig):
    # dataset config
    dataset: HypotestDatasetConfig


class HypotestResourcesServer(AviaryResourcesServer[InterpreterEnv, HypotestDataset]):
    config: HypotestServerConfig
    dataset: HypotestDataset
    # Aviary's agent calls /close BEFORE /verify (inside responses().finally),
    # so by the time verify() runs, the env is already popped from env_id_to_env.
    # Snapshot the per-rollout diagnostics here during close() and read them in verify().
    env_id_to_diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def load_dataset(cls, data: dict) -> dict:
        if "dataset" not in data:
            config = data["config"] = HypotestServerConfig.model_validate(data.get("config", {}))
            data["dataset"] = HypotestDataset(config.dataset)
        return data

    async def close(self, request: Request, body: AviaryCloseRequest) -> AviaryCloseResponse:
        # Snapshot diagnostics before the parent pops the env.
        env = self.env_id_to_env.get(body.env_id)
        if env is not None:
            self.env_id_to_diagnostics[body.env_id] = _env_wandb_extras(env)
        return await super().close(request, body)

    async def verify(self, request: Request, body: AviaryAgentVerifyRequest) -> AviaryAgentVerifyResponse:
        env_id = body.response.env_id
        total_reward = self.env_id_to_total_reward[env_id]
        # Prefer the close()-time snapshot; fall back to the live env if close hasn't
        # fired yet (defensive — the agent's order means this path is rarely hit).
        extras = self.env_id_to_diagnostics.pop(env_id, None)
        if extras is None:
            extras = _env_wandb_extras(self.env_id_to_env.get(env_id))
        return AviaryAgentVerifyResponse(
            **body.model_dump(),
            reward=total_reward,
            **extras,
        )


def _env_wandb_extras(env: InterpreterEnv | None) -> dict[str, Any]:
    """Pull per-rollout diagnostics off the env state for wandb logging.

    Numeric fields auto-become scalar metrics in nemo_rl/experience/rollouts.py;
    string/list fields land in the `full_result` wandb Table (guarded by
    env.should_log_nemo_gym_responses: true in the training YAML).
    """
    if env is None or getattr(env, "state", None) is None:
        return {}

    s = env.state
    hm: dict[str, Any] = s.hybrid_metadata or {}
    try:
        max_score = int(env.problem.max_score)
    except Exception:
        max_score = 0

    strip_amount = float(max(0.0, s.rubric_reward_raw - s.hybrid_reward_value))
    # Spec §12.3: PASS/FAIL decomposition. A "PASS" rollout is one the rubric grader
    # awarded full credit to; a "FAIL" rollout is anything else. PASS strip is the
    # false-positive-prone signal — if it drifts high, the gate is stripping honest
    # solves. FAIL strip is the intended work (reward-hack suppression).
    EPS = 1e-6
    is_pass = int(s.rubric_reward_raw >= 1.0 - EPS)
    is_stripped = int(strip_amount > EPS)

    return {
        # --- scalar metrics ---
        "rubric_reward_raw": float(s.rubric_reward_raw),
        "hybrid_reward_value": float(s.hybrid_reward_value),
        "strip_amount": strip_amount,
        "strip_amount_on_pass_rollout": strip_amount if is_pass else 0.0,
        "strip_amount_on_fail_rollout": 0.0 if is_pass else strip_amount,
        "is_pass_rollout": is_pass,
        "is_stripped": is_stripped,
        "is_stripped_pass": int(is_pass and is_stripped),
        "is_stripped_fail": int(not is_pass and is_stripped),
        "raw_score": int(s.raw_score),
        "max_score": max_score,
        "proc_max_pts": int(hm.get("proc_max_pts", 0)),
        "proc_pts_awarded_by_rubric": int(hm.get("proc_pts_awarded_by_rubric", 0)),
        "proc_pts_credited": int(hm.get("proc_pts_credited", 0)),
        "concl_max_pts": int(hm.get("concl_max_pts", 0)),
        "concl_pts_awarded_by_rubric": int(hm.get("concl_pts_awarded_by_rubric", 0)),
        "concl_pts_credited": int(hm.get("concl_pts_credited", 0)),
        "judge_call_failed": int(bool(hm.get("judge_call_failed", False))),
        "parse_failed": int(bool(hm.get("parse_failed", False))),
        "weights_mismatch": int(bool(hm.get("weights_mismatch", False))),
        "n_items_parsed": int(len(hm.get("per_item", []))),
        "binary_faithfulness_passed": _tri_int(s.faithfulness_passed),
        # --- Scheme D (wager) diagnostics ---
        **_wager_extras(env, s),
        # --- install-shim counters (per-rollout) ---
        **_shim_extras(env),
        # --- cell-timeout override counters ---
        **_cell_timeout_extras(env, s),
        # --- strings / lists (full_result table only) ---
        "faithfulness_mode": str(env.config.faithfulness_mode),
        "strip_reason": str(hm.get("strip_reason", "")),
        "hybrid_prompt": str(hm.get("prompt", "")),
        "hybrid_response": str(hm.get("response", "")),
        "hybrid_per_item": list(hm.get("per_item", [])),
        "hybrid_per_item_strip": list(hm.get("per_item_strip", [])),
        "wager_mode": str(env.config.wager_mode),
        "wager_skipped_reason": str((s.wager_metadata or {}).get("skipped_reason", "")),
        "cell_timeout_override_mode": str(getattr(env.config, "cell_timeout_override_mode", "off")),
    }


def _cell_timeout_extras(env: InterpreterEnv, s: Any) -> dict[str, Any]:
    """Per-rollout cell-timeout override diagnostics. Scalar keys are stable
    across override_mode=off vs on runs; off-mode trivially reports zeros."""
    reqs = list(getattr(s, "cell_timeout_override_requests", None) or [])
    return {
        "cell_timeout_override_count": len(reqs),
        "cell_timeout_max_requested": float(max(reqs)) if reqs else 0.0,
        "cell_timeout_mean_requested": float(sum(reqs) / len(reqs)) if reqs else 0.0,
    }


def _wager_extras(env: InterpreterEnv, s: Any) -> dict[str, Any]:
    """Scalar wager diagnostics for wandb. Always emitted so scalar keys are
    stable across wager_mode=off vs shadow/active runs; off-mode rollouts
    trivially report zeros."""
    wager = float(getattr(s, "wager", 0.0) or 0.0)
    wm = getattr(s, "wager_metadata", None) or {}
    correct_sig = wm.get("correct")
    correct_int = -1 if correct_sig is None else int(bool(correct_sig))
    rubric_reward_raw = float(getattr(s, "rubric_reward_raw", 0.0) or 0.0)
    wager_reward_shadow = float(getattr(s, "wager_reward_shadow", 0.0) or 0.0)
    return {
        "wager": wager,
        "wager_is_honest": int(wager < 0.3),
        "wager_is_high_conf": int(wager > 0.7),
        "wager_correct_signal": correct_int,
        "wager_is_high_conf_correct": int(wager > 0.7 and correct_int == 1),
        "wager_is_overconfident": int(wager > 0.7 and correct_int == 0),
        "wager_bonus_applied": float(wm.get("bonus_applied", 0.0) or 0.0),
        "wager_penalty_applied": float(wm.get("penalty_applied", 0.0) or 0.0),
        "wager_reward_shadow": wager_reward_shadow,
        "wager_reward_delta": wager_reward_shadow - rubric_reward_raw,
    }


def _tri_int(v: bool | None) -> int:
    """Encode tri-state bool for scalar wandb logging: -1=unset, 0=False, 1=True."""
    if v is None:
        return -1
    return int(bool(v))


_SHIM_OUTCOMES = (
    "skipped",
    "skipped_version_mismatch",
    "installed",
    "install_timeout",
    "install_failed",
    "passthrough",
    "passthrough_force",
)


def _shim_extras(env: InterpreterEnv) -> dict[str, Any]:
    """Parse the per-rollout install-shim JSONL log and emit scalar counters.

    Log path: `$WORKDIR/.install_shim/log`. Missing file (or off-path kernel
    setups) returns zeros for every outcome so the scalar keys are stable
    across rollouts regardless of whether the shim ran.
    """
    counts = {outcome: 0 for outcome in _SHIM_OUTCOMES}
    total = 0
    work_dir = getattr(env, "work_dir", None)
    if work_dir is not None:
        log_path = Path(work_dir) / ".install_shim" / "log"
        if log_path.exists():
            try:
                with log_path.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except Exception:
                            continue
                        total += 1
                        outcome = entry.get("outcome")
                        if outcome in counts:
                            counts[outcome] += 1
            except Exception:
                pass
    return {
        "install_shim_total": total,
        **{f"install_shim_{k}": v for k, v in counts.items()},
    }


if __name__ == "__main__":
    HypotestResourcesServer.run_webserver()
