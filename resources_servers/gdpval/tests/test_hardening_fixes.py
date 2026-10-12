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
"""Resume safety and transport-repair hardening."""

import zipfile
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from omegaconf import OmegaConf

from resources_servers.gdpval.multistage_orchestrator import (
    MultiStageRunConfig,
    _prepare_resume,
    _prune_downstream_files,
    compute_fingerprint,
    failures_path_for,
    parse_multistage_config,
)
from resources_servers.gdpval.transport_assignment import PairCost, _footprint, _pair_cost


REF_ELOS = {"ref_a": 1000.0}
DIST = {"grp": {"task_ids": ["t0"], "percentage": 100.0}}


def _cfg() -> MultiStageRunConfig:
    return MultiStageRunConfig(
        enabled=True,
        stages=parse_multistage_config({"enabled": True, "stages": [{"num_tasks": 1}]}).stages,
        seed=0,
    )


class TestFingerprintConnectionFields:
    def test_endpoint_and_credential_changes_do_not_invalidate(self) -> None:
        def runtime(base_url: str, api_key: str) -> dict:
            return {
                "judge": {
                    "responses_api_models": {
                        "minimax": {
                            "openai_base_url": base_url,
                            "openai_api_key": api_key,
                            "openai_model": "minimax-m3",
                        }
                    }
                }
            }

        cfg = _cfg()
        baseline = compute_fingerprint(
            cfg, REF_ELOS, DIST, resolved_global_config=runtime("http://10.0.0.1:8000/v1", "sk-a")
        )
        rotated = compute_fingerprint(
            cfg, REF_ELOS, DIST, resolved_global_config=runtime("http://10.0.9.9:8000/v1", "sk-b")
        )
        assert rotated == baseline

        changed_model = dict(runtime("http://10.0.0.1:8000/v1", "sk-a"))
        changed_model["judge"]["responses_api_models"]["minimax"]["openai_model"] = "other-model"
        assert compute_fingerprint(cfg, REF_ELOS, DIST, resolved_global_config=changed_model) != baseline

    def test_bare_connection_fields_and_headers_do_not_invalidate(self) -> None:
        def runtime(**overrides) -> dict:
            server = {
                "base_url": ["http://10.0.0.1:8000/v1"],
                "api_key": "sk-a",
                "default_headers": {"Authorization": "Bearer a"},
                "openai_default_headers": {"X-Key": "a"},
                "model": "policy",
            }
            server.update(overrides)
            return {"policy_model": {"responses_api_models": {"vllm_model": server}}}

        cfg = _cfg()
        baseline = compute_fingerprint(cfg, REF_ELOS, DIST, resolved_global_config=runtime())
        rotated = runtime(
            base_url=["http://10.0.9.9:8000/v1", "http://10.0.9.10:8000/v1"],
            api_key="sk-b",
            default_headers={"Authorization": "Bearer b"},
            openai_default_headers={"X-Key": "b"},
        )
        assert compute_fingerprint(cfg, REF_ELOS, DIST, resolved_global_config=rotated) == baseline
        assert compute_fingerprint(cfg, REF_ELOS, DIST, resolved_global_config=runtime(model="other")) != baseline


class TestFingerprintBenchmarkConfig:
    """Resume across allocations with the shipped GDPVal benchmark and policy configs."""

    REPO_ROOT = Path(__file__).resolve().parents[3]

    def _fingerprint(self, monkeypatch: pytest.MonkeyPatch, *, rerun_incomplete: str = "false", **policy) -> str:
        monkeypatch.setenv("RERUN_INCOMPLETE", rerun_incomplete)
        policy_values = {
            "policy_base_url": "http://10.0.0.1:8000/v1",
            "policy_api_key": "key-a",
            "policy_model_name": "policy-model",
            **policy,
        }
        config = OmegaConf.merge(
            OmegaConf.load(self.REPO_ROOT / "benchmarks/gdpval/config.yaml"),
            OmegaConf.load(self.REPO_ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
            OmegaConf.create(policy_values),
        )
        resolved = OmegaConf.to_container(config, resolve=True)
        return compute_fingerprint(_cfg(), REF_ELOS, DIST, resolved_global_config=resolved)

    def test_new_allocation_and_resume_knobs_keep_the_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = self._fingerprint(monkeypatch)

        assert self._fingerprint(monkeypatch, policy_base_url="http://10.0.9.9:8000/v1") == baseline
        assert self._fingerprint(monkeypatch, policy_api_key="key-b") == baseline
        assert self._fingerprint(monkeypatch, rerun_incomplete="true") == baseline

    def test_policy_model_change_invalidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = self._fingerprint(monkeypatch)

        assert self._fingerprint(monkeypatch, policy_model_name="other-model") != baseline

    def test_agent_concurrency_is_operational_but_turn_cap_is_not(self) -> None:
        def runtime(**agent) -> dict:
            return {
                "gdpval_stirrup_agent": {
                    "responses_api_agents": {"stirrup_agent": {"concurrency": 32, "agent_max_turns": 250, **agent}}
                }
            }

        cfg = _cfg()
        baseline = compute_fingerprint(cfg, REF_ELOS, DIST, resolved_global_config=runtime())
        assert compute_fingerprint(cfg, REF_ELOS, DIST, resolved_global_config=runtime(concurrency=8)) == baseline
        assert (
            compute_fingerprint(cfg, REF_ELOS, DIST, resolved_global_config=runtime(agent_max_turns=100)) != baseline
        )


class TestStaleResumeQuarantine:
    def test_stale_state_is_quarantined_not_deleted(self, tmp_path: Path) -> None:
        output = tmp_path / "rollouts.jsonl"
        journal = tmp_path / "rollouts_multistage_state.jsonl"
        payload = b'{"stage_index": 0, "reward": 1.0}\n'
        output.write_bytes(payload)
        journal.write_bytes(b'{"stage_index": 0, "status": "planned", "fingerprint": "OLD"}\n')

        config = SimpleNamespace(resume_from_cache=True)
        _prepare_resume(config, output, journal, "NEW")

        assert not journal.exists()
        quarantined = sorted(tmp_path.glob("rollouts.jsonl.stale.*"))
        assert quarantined and quarantined[0].read_bytes() == payload


class TestPruneQuarantine:
    def test_pruned_rows_are_preserved(self, tmp_path: Path) -> None:
        output = tmp_path / "rollouts.jsonl"
        rows = [
            {"stage_index": 0, "task_id": "keep"},
            {"stage_index": 1, "task_id": "pruned"},
        ]
        output.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
        failures_path_for(output).write_bytes(orjson.dumps({"stage_index": 2, "task_id": "pruned_failure"}) + b"\n")

        _prune_downstream_files(output, restart_stage=0)

        assert orjson.loads(output.read_bytes())["task_id"] == "keep"
        main_quarantine = sorted(tmp_path.glob("rollouts.jsonl.pruned.*"))
        assert main_quarantine and orjson.loads(main_quarantine[0].read_bytes())["task_id"] == "pruned"
        sidecar_quarantine = sorted(tmp_path.glob("*failures*.pruned.*"))
        assert sidecar_quarantine


class TestTransportRepairHardening:
    def _cost(self, candidate: object, reference: object) -> PairCost:
        return _pair_cost(
            candidate,
            reference,
            max_file_bytes=250 * 1024 * 1024,
            max_raw_bytes=300 * 1024 * 1024,
            max_wire_bytes=420 * 1024 * 1024,
            max_section_raw_bytes=96 * 1024 * 1024,
            framing_reserve_bytes=4 * 1024 * 1024,
        )

    def test_corrupt_zip_is_costed_incompatible_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "broken.zip").write_bytes(b"this is not a zip archive")
        footprint = _footprint(tmp_path)
        assert footprint.defective is True

        from resources_servers.gdpval.transport_assignment import Footprint

        clean = Footprint(raw_bytes=1, max_file_bytes=1, has_av=False, file_count=1)
        cost = self._cost(footprint, clean)
        assert cost.compatible is False and "corrupt_archive" in cost.reasons

    def test_valid_zip_members_still_enumerated(self, tmp_path: Path) -> None:
        with zipfile.ZipFile(tmp_path / "ok.zip", "w") as archive:
            archive.writestr("page.pdf", b"%PDF-fake")
        footprint = _footprint(tmp_path)
        assert footprint.defective is False and footprint.file_count == 1

    def test_section_budget_marks_pair_incompatible(self) -> None:
        from resources_servers.gdpval.transport_assignment import Footprint

        big_section = Footprint(
            raw_bytes=150 * 1024 * 1024, max_file_bytes=150 * 1024 * 1024, has_av=False, file_count=1
        )
        small = Footprint(raw_bytes=1, max_file_bytes=1, has_av=False, file_count=1)
        cost = self._cost(big_section, small)
        assert cost.compatible is False and "section_raw_over_cap" in cost.reasons

    def test_new_av_extensions_are_recognized(self, tmp_path: Path) -> None:
        (tmp_path / "voice.opus").write_bytes(b"0" * 10)
        footprint = _footprint(tmp_path)
        assert footprint.has_av is True and footprint.file_count == 1
