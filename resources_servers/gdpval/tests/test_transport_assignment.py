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
"""Transport-aware reference assignment for multistage stage plans."""

import pytest

from resources_servers.gdpval.multistage_orchestrator import (
    MultiStageRunConfig,
    compute_fingerprint,
    parse_multistage_config,
)
from resources_servers.gdpval.transport_assignment import PairCost, _solve_capacity_assignment, make_assignment_repair


def _global_config(candidate_root, references, *, agent_extra=None) -> dict:
    """A resolved config shaped like benchmarks/gdpval/config.yaml."""
    agent = {"resources_server": {"type": "resources_servers", "name": "gdpval_resources_server"}}
    if candidate_root is not None:
        agent["persist_deliverables_dir"] = str(candidate_root)
    agent.update(agent_extra or {})
    return {
        "gdpval_resources_server": {
            "resources_servers": {
                "gdpval": {
                    "reference_models": {name: {"deliverables_dir": str(root)} for name, root in references.items()},
                }
            }
        },
        "gdpval_stirrup_agent": {"responses_api_agents": {"stirrup_agent": agent}},
    }


def _write_task(root, task_id: str, *, payload: bytes = b"x", finished: bool = True) -> None:
    repeat = root / f"task_{task_id}" / "repeat_0"
    repeat.mkdir(parents=True)
    if finished:
        (repeat / "finish_params.json").write_text("{}")
    (repeat / "evidence.mp4").write_bytes(payload)


class TestSolveCapacityAssignment:
    def test_minimum_change_reassignment_respects_capacities(self) -> None:
        tasks = ["t0", "t1", "t2"]
        references = ["ra", "rb"]
        original = {"t0": "ra", "t1": "ra", "t2": "rb"}

        def cost(compatible: bool, wire: int = 1) -> PairCost:
            return PairCost(compatible=compatible, wire_bytes=wire, raw_bytes=wire, max_file_bytes=wire, reasons=())

        # t1/ra is incompatible; the only capacity-preserving repair swaps t1 and t2.
        costs = {
            ("t0", "ra"): cost(True),
            ("t0", "rb"): cost(True),
            ("t1", "ra"): cost(False),
            ("t1", "rb"): cost(True),
            ("t2", "ra"): cost(True),
            ("t2", "rb"): cost(True),
        }

        result = _solve_capacity_assignment(tasks, references, original, costs)

        assert result == {"t0": "ra", "t1": "rb", "t2": "ra"}
        assert sorted(result.values()) == sorted(original.values())

    @pytest.mark.parametrize("candidate_present", [True, False])
    @pytest.mark.parametrize(
        ("marker", "available"),
        [(None, False), ("", False), ("{", False), ("[]", False), ("null", True), ("{}", True)],
        ids=["missing-task", "missing-marker", "corrupt-marker", "wrong-shape", "null-finish", "explicit-finish"],
    )
    def test_availability_only_preserves_counts_and_completion_contract(
        self, tmp_path, marker, available, candidate_present
    ) -> None:
        roots = {name: tmp_path / name for name in ("candidate", "ra", "rb")}
        for name, root in roots.items():
            for task in ("t0", "t1", "t2"):
                if name == "candidate" and task == "t0" and not candidate_present:
                    continue
                if name == "ra" and task == "t0" and marker is None:
                    continue
                repeat = root / f"task_{task}" / "repeat_0"
                repeat.mkdir(parents=True)
                value = marker if name == "ra" and task == "t0" else "{}"
                if value:
                    (repeat / "finish_params.json").write_text(value)
                (repeat / "evidence.mp4").write_bytes(b"over-cap")
        global_config = _global_config(roots["candidate"], {name: roots[name] for name in ("ra", "rb")})
        limits = {"max_file_bytes": 1, "max_raw_bytes": 1, "max_wire_bytes": 1, "max_section_raw_bytes": 1}
        original = {"t0": "ra", "t1": "rb", "t2": "ra"}
        # Default repair still considers transport; availability-only is opt-in.
        # Every pair is over the 1-byte caps, so no task has a route: each keeps
        # its original draw instead of aborting the stage.
        transport_result, transport_receipt = make_assignment_repair(global_config, limits)(0, ["ra", "rb"], original)
        assert transport_result == original
        assert transport_receipt["changes"] == []
        assert transport_receipt["unrepairable_tasks"] == ([] if candidate_present else ["t0"])
        assert transport_receipt["unroutable_tasks"] == (["t0", "t1", "t2"] if candidate_present else ["t1", "t2"])
        assert transport_receipt["infeasible"] is None
        repair = make_assignment_repair(global_config, {**limits, "reference_availability_only": True})

        result, receipt = repair(0, ["ra", "rb"], original)

        assert result == (original if available else {"t0": "rb", "t1": "ra", "t2": "ra"})
        assert sorted(result.values()) == sorted(original.values())
        assert receipt["reference_availability_only"] is True
        assert len(receipt["changes"]) == (0 if available else 2)
        assert receipt["initially_incompatible"] == (
            [] if available else [{"task_id": "t0", "reference_id": "ra", "reasons": ["reference_incomplete"]}]
        )


class TestFingerprintTransportRepair:
    REF_ELOS = {"ref_a": 1000.0}
    DIST = {"grp": {"task_ids": ["t0"], "percentage": 100.0}}

    def _cfg(self) -> MultiStageRunConfig:
        return MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": [{"num_tasks": 1}]}).stages,
            seed=0,
        )

    def test_repair_settings_invalidate_fingerprint(self) -> None:
        cfg = self._cfg()
        base = compute_fingerprint(cfg, self.REF_ELOS, self.DIST, resolved_global_config={"multistage": {}})
        repair_a = compute_fingerprint(
            cfg,
            self.REF_ELOS,
            self.DIST,
            resolved_global_config={"multistage": {"transport_assignment_repair": {"max_file_bytes": 1}}},
        )
        repair_b = compute_fingerprint(
            cfg,
            self.REF_ELOS,
            self.DIST,
            resolved_global_config={"multistage": {"transport_assignment_repair": {"max_file_bytes": 2}}},
        )
        assert repair_a != base
        assert repair_b != base
        assert repair_a != repair_b

    def test_absent_repair_config_adds_nothing(self) -> None:
        cfg = self._cfg()
        without_block = compute_fingerprint(cfg, self.REF_ELOS, self.DIST, resolved_global_config={})
        empty_block = compute_fingerprint(cfg, self.REF_ELOS, self.DIST, resolved_global_config={"multistage": {}})
        assert without_block == empty_block


class TestAssignmentRepairConfig:
    def test_candidate_root_is_the_linked_agents_persist_dir(self, tmp_path) -> None:
        candidate, ra, rb = (tmp_path / name for name in ("candidate", "ra", "rb"))
        for root in (candidate, ra, rb):
            _write_task(root, "t0")
        repair = make_assignment_repair(_global_config(candidate, {"ra": ra, "rb": rb}), {})

        _, receipt = repair(0, ["ra", "rb"], {"t0": "ra"})

        # The candidate deliverable was found, so the task was costed, not pinned.
        assert receipt["unrepairable_tasks"] == []

    def test_resources_server_persist_dir_is_not_a_candidate_root(self, tmp_path) -> None:
        references = {"ra": tmp_path / "ra"}
        references["ra"].mkdir()
        config = _global_config(None, references)
        config["gdpval_resources_server"]["resources_servers"]["gdpval"]["persist_deliverables_dir"] = str(tmp_path)

        with pytest.raises(ValueError, match="exactly one persist_deliverables_dir"):
            make_assignment_repair(config, {})

    def test_agent_of_another_resources_server_is_ignored(self, tmp_path) -> None:
        references = {"ra": tmp_path / "ra"}
        references["ra"].mkdir()
        config = _global_config(
            tmp_path / "candidate",
            references,
            agent_extra={"resources_server": {"type": "resources_servers", "name": "other_server"}},
        )

        with pytest.raises(ValueError, match="exactly one persist_deliverables_dir"):
            make_assignment_repair(config, {})

    def test_relative_candidate_root_is_rejected(self, tmp_path) -> None:
        references = {"ra": tmp_path / "ra"}
        references["ra"].mkdir()

        with pytest.raises(ValueError, match="absolute path"):
            make_assignment_repair(_global_config("output/gdpval/deliverables", references), {})

    def test_candidate_root_that_is_a_file_is_rejected(self, tmp_path) -> None:
        references = {"ra": tmp_path / "ra"}
        references["ra"].mkdir()
        (tmp_path / "candidate").write_text("")

        with pytest.raises(ValueError, match="not a directory"):
            make_assignment_repair(_global_config(tmp_path / "candidate", references), {})

    def test_fresh_run_pins_every_task(self, tmp_path) -> None:
        references = {"ra": tmp_path / "ra", "rb": tmp_path / "rb"}
        for root in references.values():
            _write_task(root, "t0")
            _write_task(root, "t1")
        # Nothing has been persisted yet, so the candidate root does not exist.
        repair = make_assignment_repair(_global_config(tmp_path / "not_yet_created", references), {})

        original = {"t0": "ra", "t1": "rb"}
        result, receipt = repair(0, ["ra", "rb"], original)

        assert result == original
        assert receipt["unrepairable_tasks"] == ["t0", "t1"]
        assert receipt["changes"] == []

    def test_capacity_infeasible_stage_keeps_its_original_draw(self, tmp_path) -> None:
        # Both tasks can only be judged against rb, which holds one slot.
        references = {"ra": tmp_path / "ra", "rb": tmp_path / "rb"}
        references["ra"].mkdir()
        _write_task(references["rb"], "t0")
        _write_task(references["rb"], "t1")
        repair = make_assignment_repair(
            _global_config(tmp_path / "candidate", references), {"reference_availability_only": True}
        )

        original = {"t0": "ra", "t1": "rb"}
        result, receipt = repair(0, ["ra", "rb"], original)

        assert result == original
        assert receipt["unroutable_tasks"] == []
        assert "no count-preserving" in receipt["infeasible"]
        assert receipt["changes"] == []
