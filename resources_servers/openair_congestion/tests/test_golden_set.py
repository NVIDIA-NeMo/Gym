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
# The golden set is only a benchmark if it is recomputable and its scoring is
# well-formed: generation must be deterministic, the golden action must beat
# coasting by construction, and the oracle/noop endpoints of the recovery
# metric must pin to 1 and 0. The interference regime is used where a positive
# margin is needed, since that is where single actions matter.
from resources_servers.openair_congestion import golden_set
from resources_servers.openair_congestion.backends import ReplayBackend
from resources_servers.openair_congestion.golden_set import (
    _oracle_policy,
    generate_golden_set,
    score_policy_against_golden,
)


def _small_set():
    # A fast, opportunity-bearing slice: interference is the regime where a
    # single action beats coasting.
    return generate_golden_set(seeds=(33001, 33002, 33003), regimes=("interference",), decision_steps=(6, 9))


def test_generation_is_deterministic_and_well_formed():
    a = generate_golden_set(seeds=(33001, 33002), regimes=("interference",), decision_steps=(6,))
    b = generate_golden_set(seeds=(33001, 33002), regimes=("interference",), decision_steps=(6,))
    assert [r.to_json() for r in a] == [r.to_json() for r in b]  # recomputable

    for row in a:
        # The golden action is the argmax over the grid, so it can never score
        # below coasting; noop is always in the grid, so margin >= 0 exactly.
        assert row.golden_value >= row.noop_value
        assert row.margin >= 0.0
        assert row.state_hash and row.n_candidates > 1
        assert row.ranked[0]["name"] == row.golden_action["name"]


def test_oracle_pins_to_one_and_noop_to_zero():
    rows = _small_set()
    assert any(r.margin > 0.05 for r in rows), "interference should yield opportunities"

    oracle = score_policy_against_golden(rows, _oracle_policy(rows))
    assert oracle["exact_match_rate"] == 1.0
    assert oracle["mean_margin_recovered"] > 0.95  # replaying golden recovers ~all margin

    noop = score_policy_against_golden(rows, lambda obs: {"name": "noop", "arguments": {}})
    assert noop["mean_margin_recovered"] == 0.0  # coasting recovers none, by definition
    assert noop["scored_rows"] == oracle["scored_rows"] > 0


def test_cli_validation_runs_compact_local_benchmark(monkeypatch, capsys):
    import sys

    rows = _small_set()
    monkeypatch.setattr(golden_set, "generate_golden_set", lambda: rows)
    monkeypatch.setattr(sys, "argv", ["golden_set.py"])

    golden_set.main()

    assert "benchmark validity (oracle recovers ~1, noop ~0): PASS" in capsys.readouterr().out


def test_recovery_is_bounded_for_an_adversarial_policy():
    rows = _small_set()
    # A policy that always makes a catastrophic (guardrail-rejected) call must
    # not push recovery below 0 or above 1 -- the metric is clipped.
    catastrophic = score_policy_against_golden(
        rows,
        lambda obs: {"name": "set_prb_cap", "arguments": {"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 0}},
    )
    assert 0.0 <= catastrophic["mean_margin_recovered"] <= 1.0


def test_generation_and_scoring_close_every_reconstructed_episode(monkeypatch):
    class TrackingReplayBackend(ReplayBackend):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reset_calls = 0
            self.close_calls = 0
            self.instances.append(self)

        def reset(self, *args, **kwargs):
            self.reset_calls += 1
            return super().reset(*args, **kwargs)

        def close(self, *args, **kwargs):
            self.close_calls += 1
            return super().close(*args, **kwargs)

    monkeypatch.setattr(golden_set, "ReplayBackend", TrackingReplayBackend)
    rows = generate_golden_set(seeds=(33001,), regimes=("interference",), decision_steps=(6,))
    generation_backend = TrackingReplayBackend.instances[-1]
    assert generation_backend.close_calls == generation_backend.reset_calls

    TrackingReplayBackend.instances.clear()
    score_policy_against_golden(rows, lambda obs: {"name": "noop", "arguments": {}})
    scoring_backend = TrackingReplayBackend.instances[-1]
    assert scoring_backend.close_calls == scoring_backend.reset_calls
