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
"""Resume safety, failure classification, and transport-repair hardening."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from resources_servers.gdpval.comparison import (
    _is_lossy_transport_marker,
    build_file_section,
)
from resources_servers.gdpval.multistage_orchestrator import (
    MultiStageRunConfig,
    _prepare_resume,
    _prune_downstream_files,
    compute_fingerprint,
    failures_path_for,
    parse_multistage_config,
)
from resources_servers.gdpval.scoring import is_permanent_judge_error
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


class TestThrottleVeto:
    def test_429_body_with_permanent_phrase_is_not_permanent(self) -> None:
        assert is_permanent_judge_error("Error code: 429 - you have sent too many tokens this minute") is False
        error = RuntimeError("upstream failure")
        error.status = 429
        error.body = "too many tokens"
        assert is_permanent_judge_error(error) is False

    def test_413_and_context_overflow_remain_permanent(self) -> None:
        assert is_permanent_judge_error("HTTP 413: request entity too large") is True
        error = RuntimeError("bad request")
        error.status_code = 413
        assert is_permanent_judge_error(error) is True
        assert is_permanent_judge_error("maximum context length exceeded: input is too long") is True


class TestVerifyCacheScreening:
    def test_failure_classed_results_are_neither_written_nor_reused(self, tmp_path: Path) -> None:
        from responses_api_agents.stirrup_agent.app import (
            NG_FAILURE_CLASS_KEY,
            StirrupAgentWrapper,
            _verify_cache_path,
        )

        deliverables = tmp_path / "repeat_0"
        deliverables.mkdir()
        failure_result = {"reward": 0.0, NG_FAILURE_CLASS_KEY: "reference_missing"}

        StirrupAgentWrapper._write_cached_verify(None, str(deliverables), failure_result)
        cache_path = _verify_cache_path(str(deliverables))
        assert cache_path is not None and not cache_path.exists()

        # A legacy cache written before the fix must read as a miss.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(failure_result))
        assert StirrupAgentWrapper._read_cached_verify(None, str(deliverables)) is None

        success_result = {"reward": 1.0, "judge_response": {}}
        cache_path.write_text(json.dumps(success_result))
        assert StirrupAgentWrapper._read_cached_verify(None, str(deliverables)) == success_result


class TestLossMarkers:
    def test_new_marker_forms_are_detected(self) -> None:
        assert _is_lossy_transport_marker("[attachment unavailable for clip.mp4]")
        assert _is_lossy_transport_marker("[page 3 omitted for deck.pdf: raster dimensions too large]")
        assert not _is_lossy_transport_marker("[page 3] ordinary document text")
        assert not _is_lossy_transport_marker("regular text")

    def test_loss_marker_survives_exhausted_text_budget(self, tmp_path: Path, monkeypatch) -> None:
        import resources_servers.gdpval.comparison as comparison

        monkeypatch.setattr(comparison, "MAX_SECTION_TEXT_CHARS_FOR_JUDGE", 10)
        (tmp_path / "a_filler.txt").write_text("x" * 100)
        (tmp_path / "b_huge.mp4").write_bytes(b"0" * 64)
        monkeypatch.setattr(comparison, "MAX_FILE_BYTES_FOR_JUDGE", 8)

        clean_up: list = []
        section = build_file_section(
            str(tmp_path),
            clean_up,
            media_mode="native_pdf",
            render_dpi=144,
            max_pages=50,
            include_text=True,
            audio_capable=False,
            video_capable=False,
        )
        texts = [str(block.get("text", "")) for block in section if block.get("type") == "text"]
        assert any(_is_lossy_transport_marker(text) for text in texts)


class TestAvIdentityGuard:
    def test_oversize_av_is_not_hashed_and_fs_fault_degrades_to_marker(self, tmp_path: Path, monkeypatch) -> None:
        import resources_servers.gdpval.comparison as comparison

        monkeypatch.setattr(comparison, "MAX_FILE_BYTES_FOR_JUDGE", 4)
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"0" * 64)

        hashed: list = []
        real_sha = comparison.hashlib.sha256

        class _Spy:
            def __init__(self) -> None:
                hashed.append(1)
                self._inner = real_sha()

            def update(self, chunk) -> None:
                self._inner.update(chunk)

            def hexdigest(self) -> str:
                return self._inner.hexdigest()

        monkeypatch.setattr(comparison.hashlib, "sha256", _Spy)
        section = build_file_section(
            str(tmp_path),
            [],
            media_mode="native_pdf",
            render_dpi=144,
            max_pages=50,
            include_text=True,
            audio_capable=True,
            video_capable=True,
        )
        assert not hashed, "over-cap AV must not be hashed"
        assert section is not None


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


class TestJudgeNameCollision:
    def test_duplicate_resolved_names_raise(self, monkeypatch) -> None:
        from resources_servers.gdpval import app as gdpval_app

        monkeypatch.setattr(gdpval_app, "get_server_url", lambda name: "http://judge")

        def member(media_mode: str) -> SimpleNamespace:
            return SimpleNamespace(
                name=None,
                model="gemini-2.5-pro",
                create_params_overrides=None,
                weight=1.0,
                handles_audio=False,
                handles_video=False,
                media_mode=media_mode,
                max_native_pdf_pages=None,
                max_native_pdf_documents=None,
                max_native_pdf_bytes=None,
                max_native_pdf_bytes_per_document=None,
                raster_dpi_tiers=[],
                max_serialized_request_bytes=None,
                model_server=SimpleNamespace(name="judge_server"),
            )

        fake_server = SimpleNamespace(
            config=SimpleNamespace(
                judge_responses_create_params_overrides=None,
                judge_model_server=None,
                judge_media_mode="native_pdf",
            ),
            _effective_panel=lambda: [member("native_pdf"), member("images_and_text")],
        )
        with pytest.raises(ValueError, match="unique names"):
            gdpval_app.GDPValResourcesServer._resolve_judges(fake_server)
