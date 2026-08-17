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
# Loader and backend tests for the nested snapshot JSONL ingestion path.
# Fully offline: fixtures live under data/fixtures/.
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from openair_congestion.schemas import Observation, ToolCall

from resources_servers.openair_congestion import dataset_backend
from resources_servers.openair_congestion.backends import select_backend
from resources_servers.openair_congestion.dataset_backend import (
    DATASET_DYNAMICS_MODE,
    DatasetReplayBackend,
    load_provided_dataset,
)


FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"
SNAPSHOT_FIXTURE = FIXTURES / "sample_provided.jsonl"

NOOP = ToolCall(name="noop", arguments={})


def _make_backend(**overrides) -> DatasetReplayBackend:
    kwargs = dict(dataset_path=str(SNAPSHOT_FIXTURE), pool_size=4, max_steps_default=60)
    kwargs.update(overrides)
    return DatasetReplayBackend(**kwargs)


class TestSnapshotLoader:
    def test_fixture_parses_into_two_valid_episodes(self):
        episodes = load_provided_dataset(SNAPSHOT_FIXTURE)
        assert sorted(episodes) == ["lab_run_a", "lab_run_b"]
        assert len(episodes["lab_run_a"].observations) == 4
        assert len(episodes["lab_run_b"].observations) == 3
        for source in episodes.values():
            for obs in source.observations:
                assert isinstance(obs, Observation)  # fully schema-validated

    def test_missing_optional_fields_are_synthesized(self):
        # lab_run_a rows only carry the required fields; everything else is
        # derived with the env's own heuristics (env.py::_build_observation).
        obs = load_provided_dataset(SNAPSHOT_FIXTURE)["lab_run_a"].observations[0]
        cell = obs.cells[0]
        assert cell.prb_util_dl_p99 == pytest.approx(
            max(cell.prb_util_dl_p50, min(1.0, cell.prb_util_dl_p50 * 1.15 + 0.02))
        )
        assert cell.prb_util_ul_p50 == pytest.approx(cell.prb_util_dl_p50 * 0.4)
        assert cell.sched_latency_ms_p99 == pytest.approx(5.0 + 20.0 * cell.prb_util_dl_p99)
        assert cell.rrc_connected_ues == len(cell.ues) == 2
        ue1 = cell.ues[1]  # offered 30, delivered 14, sinr 6
        assert ue1.mcs_mean == pytest.approx((6.0 + 5.0) * 1.2)
        assert ue1.buffer_occupancy_kb == pytest.approx((30.0 - 14.0) * 50.0)
        assert ue1.pdb_violations == 1  # buffer 800 kB > 500 kB
        assert ue1.qos_5qi == 9  # default
        assert 0.0 <= cell.fairness_jain <= 1.0
        assert cell.sla_violations_last_window == 1

    def test_provided_fields_pass_through_unchanged(self):
        # lab_run_b row 0 cell 0 provides the full KPI set; no synthesis.
        obs = load_provided_dataset(SNAPSHOT_FIXTURE)["lab_run_b"].observations[0]
        cell = obs.cells[0]
        assert cell.prb_util_dl_p99 == pytest.approx(0.52)
        assert cell.fairness_jain == pytest.approx(0.93)
        assert cell.ues[0].mcs_mean == pytest.approx(20.0)
        assert obs.global_.difficulty == pytest.approx(0.7)

    def test_malformed_row_fails_fast_with_line_number(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        rows = [
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]},
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": "not_a_number"}]}]},
        ]
        bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            load_provided_dataset(bad)

    def test_row_missing_required_field_names_it(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        row = {"episode_id": "e", "cells": [{"ues": [{"delivered_mbps": 1.0}]}]}
        bad.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="prb_util_dl_p50"):
            load_provided_dataset(bad)

    def test_single_row_episode_is_rejected(self, tmp_path):
        # One observation cannot form a (prev, curr) reward pair.
        short = tmp_path / "short.jsonl"
        row = {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]}
        short.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="need >= 2"):
            load_provided_dataset(short)

    def test_missing_file_error_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dataset_path"):
            load_provided_dataset(tmp_path / "nope.jsonl")

    def test_wrong_scalar_type_fails_fast_with_line_number(self, tmp_path):
        # A list where a number belongs raises TypeError inside float();
        # the loader must still wrap it with file:line context.
        bad = tmp_path / "bad.jsonl"
        rows = [
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]},
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": [1, 2]}]}]},
        ]
        bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            load_provided_dataset(bad)

    @pytest.mark.parametrize(
        ("steps", "message"),
        [
            ([0, 0], "duplicate 'step' value 0"),
            ([0, 1.5], "'step' must be an integer"),
            ([0, None], "'step' must be present on every row"),
        ],
    )
    def test_explicit_step_indices_fail_closed(self, tmp_path, steps, message):
        bad = tmp_path / "bad_steps.jsonl"
        rows = []
        for step in steps:
            row = {
                "episode_id": "e",
                "cells": [
                    {
                        "prb_util_dl_p50": 0.5,
                        "ues": [{"delivered_mbps": 1.0}],
                    }
                ],
            }
            if step is not None:
                row["step"] = step
            rows.append(row)
        bad.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        with pytest.raises(ValueError, match=message):
            load_provided_dataset(bad)

    @pytest.mark.parametrize(
        ("container", "field", "value"),
        [
            ("cell", "prb_util_dl_p50", 1.2),
            ("cell", "fairness_jain", -0.1),
            ("ue", "delivered_mbps", -1.0),
            ("ue", "delivered_mbps", "1e999"),
            ("ue", "bler", 1.2),
            ("ue", "sinr_db", 50.0),
        ],
    )
    def test_supplied_kpis_are_rejected_instead_of_clamped(
        self,
        tmp_path,
        container,
        field,
        value,
    ):
        bad = tmp_path / "bad_kpi.jsonl"
        rows = []
        for step in range(2):
            ue = {
                "delivered_mbps": 1.0,
                "bler": 0.1,
                "sinr_db": 10.0,
            }
            cell = {
                "prb_util_dl_p50": 0.5,
                "fairness_jain": 0.9,
                "ues": [ue],
            }
            if step == 1:
                (cell if container == "cell" else ue)[field] = value
            rows.append({"episode_id": "e", "step": step, "cells": [cell]})
        bad.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        with pytest.raises(ValueError, match=rf"bad_kpi\.jsonl:2.*{field}"):
            load_provided_dataset(bad)

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_json_constant_fails_with_line_number(self, tmp_path, constant):
        bad = tmp_path / "non_finite.jsonl"
        bad.write_text(
            '{"episode_id":"e","cells":[{"prb_util_dl_p50":' + constant + ',"ues":[{"delivered_mbps":1.0}]}]}\n'
        )

        with pytest.raises(ValueError, match=r"non_finite\.jsonl:1.*non-finite"):
            load_provided_dataset(bad)

    def test_duplicate_json_key_fails_with_line_number(self, tmp_path):
        bad = tmp_path / "duplicate.jsonl"
        bad.write_text(
            '{"episode_id":"e","episode_id":"other","cells":'
            '[{"prb_util_dl_p50":0.5,"ues":[{"delivered_mbps":1.0}]}]}\n'
        )

        with pytest.raises(ValueError, match=r"duplicate\.jsonl:1.*duplicate JSON key 'episode_id'"):
            load_provided_dataset(bad)

    def test_long_episode_key_is_accepted(self, tmp_path):
        # Placeholder episode_id is clamped ('src_' + key[:56]) so long run
        # names don't trip the schema's episode_id max_length=64 at boot.
        key = "run_" + "x" * 100
        path = tmp_path / "long.jsonl"
        row = {"episode_id": key, "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]}
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        episodes = load_provided_dataset(path)
        assert list(episodes) == [key]  # full key preserved for scenario_id matching
        assert len(episodes[key].observations[0].episode_id) <= 64

    def test_csv_input_is_rejected(self, tmp_path):
        csv_path = tmp_path / "provided.csv"
        csv_path.write_text("episode_id,step,cell_id,prb_util_dl_p50,ue_id,delivered_mbps\ne0,0,0,0.5,0,9\n")

        with pytest.raises(ValueError, match="nested snapshot JSONL"):
            load_provided_dataset(csv_path)

    def test_grpo_trace_rows_are_rejected(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        rows = [
            {
                "episode_id": "e0",
                "step": step,
                "tool_sent": {"name": "noop", "arguments": {}},
                "reward_measurements": {
                    "aggregate_delivered_mbps": 10.0,
                    "n_ues": 2,
                },
            }
            for step in range(2)
        ]
        trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        with pytest.raises(ValueError, match="nested KPI snapshot"):
            load_provided_dataset(trace_path)

    def test_malformed_recorded_action_fails_with_line_number(self, tmp_path):
        rows = [json.loads(line) for line in SNAPSHOT_FIXTURE.open() if line.strip()][:2]
        rows[0]["recorded_action"] = {"name": "not_a_tool", "arguments": {}}
        path = tmp_path / "recorded_action.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        with pytest.raises(ValueError, match=r"recorded_action\.jsonl:1") as exc_info:
            load_provided_dataset(path)
        assert "unknown tool" in str(exc_info.value)


class TestDatasetReplayBackend:
    @pytest.mark.parametrize("capacity", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_invalid_cell_capacity_fails_at_startup(self, capacity):
        with pytest.raises(ValueError, match="cell_capacity_mbps"):
            _make_backend(cell_capacity_mbps=capacity)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "bad", True])
    def test_invalid_reward_weight_fails_at_startup(self, value):
        with pytest.raises((TypeError, ValueError), match="reward_weights.*w_sla"):
            _make_backend(reward_weights={"w_sla": value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [("pool_size", 0), ("pool_size", True), ("max_steps_default", 0), ("max_steps_default", True)],
    )
    def test_invalid_positive_integer_config_fails_at_startup(self, field, value):
        with pytest.raises((TypeError, ValueError), match=field):
            _make_backend(**{field: value})

    def test_reset_serves_provided_first_observation(self):
        backend = _make_backend()
        first_obs, meta = backend.reset({"scenario_id": "lab_run_a", "seed": 7})
        assert meta.scenario_id == "lab_run_a"
        assert meta.seed == 7
        assert meta.max_steps == 3  # 4 provided observations -> 3 steps
        assert first_obs.episode_id == meta.episode_id
        # Values come from the dataset, not seed-driven synthesis.
        assert first_obs.cells[0].prb_util_dl_p50 == pytest.approx(0.55)
        assert first_obs.cells[0].ues[1].delivered_mbps == pytest.approx(14.0)
        backend.close(meta.episode_id)

    def test_seed_maps_deterministically_when_no_scenario_id(self):
        backend = _make_backend()
        _, meta0 = backend.reset({"seed": 0})
        _, meta1 = backend.reset({"seed": 1})
        _, meta2 = backend.reset({"seed": 2})
        assert meta0.scenario_id == "lab_run_a"  # sorted keys[0 % 2]
        assert meta1.scenario_id == "lab_run_b"  # sorted keys[1 % 2]
        assert meta2.scenario_id == "lab_run_a"  # wraps
        for meta in (meta0, meta1, meta2):
            backend.close(meta.episode_id)

    def test_step_replays_provided_data_and_computes_reward(self):
        backend = _make_backend()
        _, meta = backend.reset({"scenario_id": "lab_run_a"})
        for expected_idx, expected_p50 in ((1, 0.70), (2, 0.92), (3, 0.80)):
            obs, reward, done, info = backend.step(meta.episode_id, NOOP)
            assert math.isfinite(reward)
            assert info["step_idx"] == expected_idx
            assert info["kpi_source"] == "dataset_replay"
            assert info["dynamics_mode"] == DATASET_DYNAMICS_MODE
            assert info["causal_action_effects"] is False
            assert info["training_usable"] is False
            assert info["diagnostic_only"] is True
            assert info["guardrail_accepted"] is True
            assert "service_accounting" not in info
            # Pass-through: KPIs are the recorded row, untouched by the action.
            assert obs.cells[0].prb_util_dl_p50 == pytest.approx(expected_p50)
            ue_payload = obs.cells[0].ues[0].model_dump()
            assert "requested_mbps" not in ue_payload
            assert "admitted_mbps" not in ue_payload
            # agent_aux is stamped like the other backends.
            assert obs.agent_aux.step_idx == expected_idx
            assert obs.agent_aux.last_action.name == "noop"
            assert obs.agent_aux.last_reward == pytest.approx(reward)
            assert done is (expected_idx == 3)
        summary = backend.close(meta.episode_id)
        assert summary == {"ok": True, "n_steps": 3}

    def test_recorded_action_is_inert_diagnostic_metadata(self, tmp_path):
        rows = [json.loads(line) for line in SNAPSHOT_FIXTURE.open() if line.strip()][:3]
        recorded_action = {
            "name": "set_scheduler_policy",
            "arguments": {"cell_id": 0, "policy": "RR"},
        }
        rows[0]["recorded_action"] = recorded_action
        path = tmp_path / "recorded_action.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        backend = _make_backend(dataset_path=str(path))
        _, meta = backend.reset({"scenario_id": "lab_run_a"})

        obs, _, _, info = backend.step(meta.episode_id, NOOP)
        assert info["recorded_action"] == recorded_action
        assert info["training_usable"] is False
        assert obs.cells[0].prb_util_dl_p50 == pytest.approx(0.70)

        _, _, _, info = backend.step(meta.episode_id, NOOP)
        assert info["recorded_action"] is None
        backend.close(meta.episode_id)

    def test_reward_exception_leaves_episode_state_unchanged(self, monkeypatch):
        backend = _make_backend()
        _, meta = backend.reset({"scenario_id": "lab_run_a"})
        episode = backend._episodes[meta.episode_id]
        before = {
            "step_idx": episode.step_idx,
            "history": list(episode.history),
            "trajectory": [obs.model_dump(by_alias=True) for obs in episode.trajectory],
        }

        def _raise_reward_error(*args, **kwargs):
            raise RuntimeError("reward failure")

        monkeypatch.setattr(dataset_backend._rewards, "compute_breakdown", _raise_reward_error)

        with pytest.raises(RuntimeError, match="reward failure"):
            backend.step(meta.episode_id, NOOP)

        after = {
            "step_idx": episode.step_idx,
            "history": list(episode.history),
            "trajectory": [obs.model_dump(by_alias=True) for obs in episode.trajectory],
        }
        assert after == before

    def test_close_waits_for_inflight_step(self, monkeypatch):
        backend = _make_backend(pool_size=1)
        _, meta = backend.reset({"scenario_id": "lab_run_a"})
        reward_entered = threading.Event()
        release_reward = threading.Event()
        step_done = threading.Event()
        close_done = threading.Event()
        errors: list[BaseException] = []
        original = dataset_backend._rewards.compute_breakdown

        def _blocking_reward(*args, **kwargs):
            reward_entered.set()
            assert release_reward.wait(timeout=5.0)
            return original(*args, **kwargs)

        def _step():
            try:
                backend.step(meta.episode_id, NOOP)
            except BaseException as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)
            finally:
                step_done.set()

        def _close():
            try:
                backend.close(meta.episode_id)
            except BaseException as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)
            finally:
                close_done.set()

        monkeypatch.setattr(dataset_backend._rewards, "compute_breakdown", _blocking_reward)
        step_thread = threading.Thread(target=_step)
        close_thread = threading.Thread(target=_close)
        step_thread.start()
        assert reward_entered.wait(timeout=5.0)
        close_thread.start()

        assert not close_done.wait(timeout=0.1)
        release_reward.set()
        step_thread.join(timeout=5.0)
        close_thread.join(timeout=5.0)

        assert step_done.is_set()
        assert close_done.is_set()
        assert not errors
        with pytest.raises(KeyError):
            backend.step(meta.episode_id, NOOP)

    def test_orphan_reaper_waits_for_inflight_step(self, monkeypatch):
        backend = _make_backend(pool_size=1)
        _, meta = backend.reset({"scenario_id": "lab_run_a"})
        reward_entered = threading.Event()
        release_reward = threading.Event()
        reset_done = threading.Event()
        errors: list[BaseException] = []
        replacement_ids: list[str] = []
        original = dataset_backend._rewards.compute_breakdown

        def _blocking_reward(*args, **kwargs):
            reward_entered.set()
            assert release_reward.wait(timeout=5.0)
            return original(*args, **kwargs)

        def _step():
            try:
                backend.step(meta.episode_id, NOOP)
            except BaseException as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)

        def _reset():
            try:
                _, replacement = backend.reset(
                    {"scenario_id": "lab_run_b"},
                    live_episode_ids=set(),
                )
                replacement_ids.append(replacement.episode_id)
            except BaseException as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)
            finally:
                reset_done.set()

        monkeypatch.setattr(dataset_backend._rewards, "compute_breakdown", _blocking_reward)
        step_thread = threading.Thread(target=_step)
        reset_thread = threading.Thread(target=_reset)
        step_thread.start()
        assert reward_entered.wait(timeout=5.0)
        reset_thread.start()

        assert not reset_done.wait(timeout=0.1)
        release_reward.set()
        step_thread.join(timeout=5.0)
        reset_thread.join(timeout=5.0)

        assert reset_done.is_set()
        assert not errors
        assert replacement_ids
        with pytest.raises(KeyError):
            backend.step(meta.episode_id, NOOP)
        backend.close(replacement_ids[0])

    def test_reward_breakdown_matches_rewards_module(self):
        # The reward must be rewards.compute_breakdown over the served
        # (prev, curr) pair, nothing else.
        from openair_congestion import rewards

        backend = _make_backend()
        episodes = load_provided_dataset(SNAPSHOT_FIXTURE)
        first_obs, meta = backend.reset({"scenario_id": "lab_run_a"})
        _, reward, _, info = backend.step(meta.episode_id, NOOP)
        expected = rewards.compute_breakdown(
            prev_obs=episodes["lab_run_a"].observations[0],
            curr_obs=episodes["lab_run_a"].observations[1],
            action=NOOP,
            rejected=False,
            cell_capacity_mbps=60.0,
        )
        assert reward == pytest.approx(float(expected["total"]))
        assert info["reward_terms"]["total"] == pytest.approx(float(expected["total"]))
        backend.close(meta.episode_id)

    def test_out_of_range_action_rejected_not_crashed(self):
        # Guardrail semantics survive: cell_id 3 does not exist in a 1-cell
        # episode -> rejected with the standard penalty, env intact.
        backend = _make_backend()
        _, meta = backend.reset({"scenario_id": "lab_run_a"})
        obs, reward, done, info = backend.step(
            meta.episode_id,
            ToolCall(name="set_scheduler_policy", arguments={"cell_id": 3, "policy": "PF"}),
        )
        assert info["guardrail_accepted"] is False
        assert info["rejection_reason"]
        assert math.isfinite(reward)
        assert done is False
        backend.close(meta.episode_id)

    def test_pool_exhaustion_reaps_orphans_then_raises(self):
        backend = _make_backend(pool_size=2)
        _, meta_a = backend.reset({"seed": 0})
        _, meta_b = backend.reset({"seed": 1})
        # Pool full; meta_a is not in live_episode_ids -> reaped, reset works.
        _, meta_c = backend.reset({"seed": 2}, live_episode_ids={meta_b.episode_id})
        assert meta_c.episode_id != meta_a.episode_id
        with pytest.raises(KeyError):
            backend.step(meta_a.episode_id, NOOP)  # reaped
        # Both slots live now -> exhausted.
        with pytest.raises(RuntimeError, match="pool exhausted"):
            backend.reset(
                {"seed": 3},
                live_episode_ids={meta_b.episode_id, meta_c.episode_id},
            )

    def test_task_max_steps_clamped_to_provided_length(self):
        backend = _make_backend()
        _, meta = backend.reset({"scenario_id": "lab_run_b", "max_steps": 50})
        assert meta.max_steps == 2  # 3 provided observations -> 2 steps max
        backend.close(meta.episode_id)

    def test_unknown_scenario_id_lists_available(self):
        backend = _make_backend()
        with pytest.raises(KeyError, match="lab_run_a"):
            backend.reset({"scenario_id": "does_not_exist"})


class TestSelectBackend:
    def test_default_dataset_replay_path_is_checked_in(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        resource_dir = Path(__file__).resolve().parent.parent
        monkeypatch.chdir(resource_dir)

        backend = select_backend(SimpleNamespace(backend="dataset_replay"))

        assert backend.dataset_path == Path("data/fixtures/sample_provided.jsonl")
        assert backend.dataset_path.is_file()

    def test_config_only_switch(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = SimpleNamespace(
            backend="dataset_replay",
            dataset_path=str(SNAPSHOT_FIXTURE),
            pool_size=4,
            max_steps_default=60,
            cell_capacity_mbps=60.0,
        )
        backend = select_backend(config)
        assert isinstance(backend, DatasetReplayBackend)
        assert backend.dataset_path == SNAPSHOT_FIXTURE

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPENAIR_CONGESTION_BACKEND", "dataset_replay")
        config = SimpleNamespace(backend="replay", dataset_path=str(SNAPSHOT_FIXTURE))
        assert isinstance(select_backend(config), DatasetReplayBackend)
