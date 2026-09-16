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
#
# Reward-correctness oracle for the replay backend: scripted policies of
# known quality must land in the right order (better play reaches higher
# return, the floor is reachable), and per-step rewards must move in the
# right direction when a KPI improves or an action is rejected.
#
# A reward normalization is pending, so every assertion here is a relative
# ordering or a directional property -- never an absolute threshold. A
# rescale of the reward must not invalidate this file.
#
# The scripted policies are test oracles, not shipped baselines; they live
# here on purpose.
import json
import random
from typing import Callable

from openair_congestion.rewards import compute, compute_breakdown, compute_terms
from openair_congestion.schemas import Observation, ToolCall

from resources_servers.openair_congestion.backends import ReplayBackend


# Fixed replay seed/difficulty pairs for the ladder. The offline backend is
# deterministic, so the orderings asserted below are stable, not statistical.
_LADDER_TASKS = ((42, 0.9), (123, 0.5), (555, 0.95), (888, 0.95))
# Pinned high-difficulty congested episode for the floor test.
_FLOOR_SEED, _FLOOR_DIFFICULTY = 555, 0.95
_MAX_STEPS = 16
# Trials per (policy, task): only the random policy varies across trials, but
# every policy is averaged the same way so the ladder compares like with like.
_N_TRIALS = 8


def _make_backend() -> ReplayBackend:
    return ReplayBackend(pool_size=64, max_steps_default=60)


def _task_params(seed: int, difficulty: float) -> dict:
    return {
        "seed": seed,
        "difficulty": difficulty,
        "regime_mix": {"prb_exhaustion": 1.0},
        "scenario_id": "prb_exhaustion",
        "tier": "replay",
        "max_steps": _MAX_STEPS,
    }


# --- Scripted policies -------------------------------------------------------
# A policy is (obs, step_idx, rng) -> ToolCall; factories give the random
# policy fresh state per episode.


def _noop_policy(obs: Observation, step_idx: int, rng: random.Random) -> ToolCall:
    return ToolCall(name="noop", arguments={})


def _relief_policy(obs: Observation, step_idx: int, rng: random.Random) -> ToolCall:
    # Scripted congestion relief: raise UL power on the most-loaded cell every
    # step. On the synthetic action-effect model set_ul_power_control lifts
    # SINR -> delivered_mbps and drains buffers for every UE in the cell. The
    # A high p0/alpha setting provides relief in the deterministic replay
    # model for this PRB-exhaustion-heavy oracle. The p0_dbm sweep keeps
    # consecutive calls distinct, so the guardrail's
    # identical-action rate limit (2 s window, 1 s logical step) never fires.
    cell = max(obs.cells, key=lambda c: (c.sla_violations_last_window, c.prb_util_dl_p99))
    return ToolCall(
        name="set_ul_power_control",
        arguments={"cell_id": cell.cell_id, "p0_dbm": 8 + (step_idx % 8), "alpha": 1.0},
    )


def _catastrophic_policy(obs: Observation, step_idx: int, rng: random.Random) -> ToolCall:
    # Worst case: max_prb=0 starves the target and is guardrail-rejected as
    # catastrophic every step, charging w_reject and suppressing delta gains.
    return ToolCall(name="set_prb_cap", arguments={"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 0})


def _make_random_valid_policy() -> Callable[[Observation, int, random.Random], ToolCall]:
    # Uniform over valid actions: every sample passes the guardrail's static
    # range checks, and the last two actions are excluded so the 2 s
    # identical-action rate limit (1 s logical step) never rejects either.
    recent: list[tuple] = []

    def _sample(obs: Observation, rng: random.Random) -> ToolCall:
        choice = rng.randrange(5)
        cell_id = rng.randrange(len(obs.cells))
        cell = obs.cells[cell_id]
        if choice == 0:
            return ToolCall(
                name="set_scheduler_policy",
                arguments={"cell_id": cell_id, "policy": rng.choice(["PF", "RR", "MaxCI"])},
            )
        if choice == 1:
            ue = rng.choice(cell.ues)
            return ToolCall(
                name="set_prb_cap",
                arguments={
                    "cell_id": cell_id,
                    "target": "ue",
                    "target_id": ue.ue_id,
                    "max_prb": rng.randrange(10, 273),
                },
            )
        if choice == 2:
            return ToolCall(
                name="set_mcs_bounds",
                arguments={"cell_id": cell_id, "mcs_min": 0, "mcs_max": rng.randrange(5, 28), "target_bler": 0.1},
            )
        if choice == 3:
            return ToolCall(
                name="set_admission_policy",
                arguments={
                    "cell_id": cell_id,
                    "accept_threshold_pct": rng.randrange(10, 100),
                    "slice_reservation": {},
                },
            )
        return ToolCall(
            name="set_ul_power_control",
            arguments={
                "cell_id": cell_id,
                "p0_dbm": rng.randrange(-120, 20),
                "alpha": rng.choice([0.4, 0.7, 0.8, 1.0]),
            },
        )

    def _key(tool_call: ToolCall) -> tuple:
        return (tool_call.name, json.dumps(tool_call.arguments, sort_keys=True))

    def policy(obs: Observation, step_idx: int, rng: random.Random) -> ToolCall:
        tool_call = _sample(obs, rng)
        for _ in range(20):
            if _key(tool_call) not in recent:
                break
            tool_call = _sample(obs, rng)
        recent.append(_key(tool_call))
        del recent[:-2]
        return tool_call

    return policy


def _episode_return(backend: ReplayBackend, policy, seed: int, difficulty: float, rng_seed: int = 0) -> float:
    rng = random.Random(rng_seed)
    obs, meta = backend.reset(_task_params(seed, difficulty))
    try:
        total = 0.0
        for step_idx in range(_MAX_STEPS):
            obs, reward, done, _ = backend.step(meta.episode_id, policy(obs, step_idx, rng))
            total += float(reward)
            if done:
                break
        return total
    finally:
        backend.close(meta.episode_id)


def _mean_return(backend: ReplayBackend, make_policy, seed: int, difficulty: float) -> float:
    returns = [_episode_return(backend, make_policy(), seed, difficulty, rng_seed=trial) for trial in range(_N_TRIALS)]
    return sum(returns) / len(returns)


class TestPolicyLadder:
    def test_relief_and_catastrophic_bound_noop_and_random_valid(self):
        # Empirical partial order on the fixed tasks. Unguided valid control
        # can sometimes beat either scripted relief or noop by accident. The
        # stable gate is that intentional relief beats noop and catastrophic
        # guardrail rejections lose to both valid policies.
        backend = _make_backend()
        for seed, difficulty in _LADDER_TASKS:
            relief = _mean_return(backend, lambda: _relief_policy, seed, difficulty)
            random_valid = _mean_return(backend, _make_random_valid_policy, seed, difficulty)
            noop = _mean_return(backend, lambda: _noop_policy, seed, difficulty)
            catastrophic = _mean_return(backend, lambda: _catastrophic_policy, seed, difficulty)
            assert relief > noop > catastrophic and random_valid > catastrophic, (
                f"ladder broken on seed={seed} difficulty={difficulty}: "
                f"relief={relief:.4f} random={random_valid:.4f} noop={noop:.4f} catastrophic={catastrophic:.4f}"
            )


class TestRewardFloor:
    def test_noop_on_pinned_congested_seed_scores_below_scripted_relief(self):
        # Floor: standing pat on a pinned high-difficulty congested episode
        # loses to scripted relief on the same episode. Same seed, same
        # trajectory, only the actions differ.
        backend = _make_backend()
        noop = _episode_return(backend, _noop_policy, _FLOOR_SEED, _FLOOR_DIFFICULTY)
        relief = _episode_return(backend, _relief_policy, _FLOOR_SEED, _FLOOR_DIFFICULTY)
        assert noop < relief


class TestPerStepRewardProperties:
    """Directional checks on compute_breakdown over real replay observations.

    Each pair shares the same prev_obs; the 'improved' curr_obs strictly
    improves exactly one KPI relative to the 'unimproved' one. The reward math
    is not re-derived -- only orderings are asserted.
    """

    _ACTION = ToolCall(name="set_prb_cap", arguments={"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 120})

    @staticmethod
    def _first_obs() -> Observation:
        backend = _make_backend()
        obs, meta = backend.reset(_task_params(_FLOOR_SEED, _FLOOR_DIFFICULTY))
        backend.close(meta.episode_id)
        return obs

    @staticmethod
    def _cell_variant(obs: Observation, *, ue_buffer_kb: float | None = None, **cell_fields) -> Observation:
        # Observation models are frozen; rebuild through dump/validate.
        data = obs.model_dump(by_alias=True)
        data["cells"][0].update(cell_fields)
        if ue_buffer_kb is not None:
            for ue in data["cells"][0]["ues"]:
                ue["buffer_occupancy_kb"] = ue_buffer_kb
        return Observation.model_validate(data)

    def test_clearing_sla_violations_never_scores_lower(self):
        prev = self._first_obs()
        violated = self._cell_variant(prev, sla_violations_last_window=2)
        cleared = self._cell_variant(prev, sla_violations_last_window=0)
        r_violated = compute_breakdown(prev, violated, self._ACTION)["total"]
        r_cleared = compute_breakdown(prev, cleared, self._ACTION)["total"]
        assert r_cleared > r_violated

    def test_prb_dropping_below_pressure_threshold_never_scores_lower(self):
        # The PRB level term charges pressure above the 0.85 p99 threshold;
        # p50 moves with p99 to satisfy the schema's p99 >= p50 invariant.
        prev = self._first_obs()
        pressured = self._cell_variant(prev, prb_util_dl_p50=0.90, prb_util_dl_p99=0.97)
        relieved = self._cell_variant(prev, prb_util_dl_p50=0.25, prb_util_dl_p99=0.30)
        r_pressured = compute_breakdown(prev, pressured, self._ACTION)["total"]
        r_relieved = compute_breakdown(prev, relieved, self._ACTION)["total"]
        assert r_relieved > r_pressured

    def test_draining_buffers_never_scores_lower(self):
        prev = self._first_obs()
        backed_up = self._cell_variant(prev, ue_buffer_kb=900.0)
        drained = self._cell_variant(prev, ue_buffer_kb=100.0)
        r_backed_up = compute_breakdown(prev, backed_up, self._ACTION)["total"]
        r_drained = compute_breakdown(prev, drained, self._ACTION)["total"]
        assert r_drained > r_backed_up

    def test_rejected_scores_below_accepted_on_the_same_pair(self):
        # Same prev/curr pair, same action: rejection charges w_reject and
        # clamps positive deltas, so it can never score >= the accepted step.
        prev = self._first_obs()
        curr = self._cell_variant(prev, prb_util_dl_p50=0.45, prb_util_dl_p99=0.50)
        accepted = compute_breakdown(prev, curr, self._ACTION, rejected=False)["total"]
        rejected = compute_breakdown(prev, curr, self._ACTION, rejected=True)["total"]
        assert rejected < accepted

    def test_reward_wrappers_match_breakdown(self):
        prev = self._first_obs()
        action = ToolCall(name="noop", arguments={})
        expected = compute_breakdown(prev, prev, action)

        assert compute_terms(prev, prev, action) == expected["terms"]
        assert compute(prev, prev, action) == expected["total"]

    def test_rejected_step_scores_below_accepted_step_through_the_env(self):
        # Same seed, two fresh episodes, one step each: an accepted relief
        # action must out-score a guardrail-rejected one on the same
        # trajectory position.
        backend = _make_backend()
        _, meta_a = backend.reset(_task_params(_FLOOR_SEED, _FLOOR_DIFFICULTY))
        _, r_accepted, _, info_a = backend.step(
            meta_a.episode_id,
            ToolCall(name="set_ul_power_control", arguments={"cell_id": 0, "p0_dbm": -90, "alpha": 0.8}),
        )
        backend.close(meta_a.episode_id)
        _, meta_r = backend.reset(_task_params(_FLOOR_SEED, _FLOOR_DIFFICULTY))
        _, r_rejected, _, info_r = backend.step(
            meta_r.episode_id,
            ToolCall(name="set_scheduler_policy", arguments={"cell_id": 99, "policy": "PF"}),
        )
        backend.close(meta_r.episode_id)
        assert info_a["guardrail_accepted"] is True
        assert info_r["guardrail_accepted"] is False
        assert r_rejected < r_accepted
