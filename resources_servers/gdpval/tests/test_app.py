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
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.gdpval.app import (
    GDPValResourcesServer,
    GDPValResourcesServerConfig,
    GDPValVerifyRequest,
    _iter_ref_repeat_dirs,
)


def _server(reward_mode: str = "rubric", **extra) -> GDPValResourcesServer:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        reward_mode=reward_mode,
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        # Default off in tests: avoids triggering the host-libreoffice install
        # check in model_post_init. Tests that exercise the preconvert path
        # set this back to True explicitly.
        preconvert_office_to_pdf=False,
    )
    kwargs.update(extra)
    config = GDPValResourcesServerConfig(**kwargs)
    return GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _verify_request(**fields) -> GDPValVerifyRequest:
    deliverable_text = fields.pop("deliverable_text", "A text deliverable.")
    return GDPValVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=NeMoGymResponse(
            id="resp-1",
            created_at=0.0,
            model="model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg-1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[NeMoGymResponseOutputText(type="output_text", text=deliverable_text, annotations=[])],
                )
            ],
            status="completed",
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        task_id="task-1",
        prompt="Write a report on X.",
        rubric_json=fields.pop("rubric_json", None),
        rubric_pretty=fields.pop("rubric_pretty", ""),
        **fields,
    )


class TestIterRefRepeatDirs:
    def test_returns_all_repeats_sorted(self, tmp_path) -> None:
        td = tmp_path / "task_x"
        (td / "repeat_1").mkdir(parents=True)
        (td / "repeat_0").mkdir()
        (td / "repeat_2").mkdir()
        assert _iter_ref_repeat_dirs(td) == [
            td / "repeat_0",
            td / "repeat_1",
            td / "repeat_2",
        ]

    def test_falls_back_to_flat_layout(self, tmp_path) -> None:
        td = tmp_path / "task_x"
        td.mkdir()
        (td / "deliverable.docx").write_text("x")
        assert _iter_ref_repeat_dirs(td) == [td]

    def test_missing_dir_returns_empty(self, tmp_path) -> None:
        assert _iter_ref_repeat_dirs(tmp_path / "does-not-exist") == []


class TestApp:
    def test_sanity_rubric(self) -> None:
        _server(reward_mode="rubric")

    def test_sanity_comparison(self) -> None:
        _server(reward_mode="comparison", reference_deliverables_dir="/tmp/fork-deliverables")

    def test_comparison_requires_reference_dir(self) -> None:
        import pytest as _pytest

        with _pytest.raises(ValueError, match="reference_deliverables_dir"):
            _server(reward_mode="comparison")

    def test_comparison_fails_fast_when_libreoffice_unavailable(self) -> None:
        with patch("resources_servers.gdpval.setup_libreoffice.ensure_libreoffice", return_value=False):
            with pytest.raises(RuntimeError, match="libreoffice"):
                _server(
                    reward_mode="comparison",
                    reference_deliverables_dir="/tmp/fork-deliverables",
                    preconvert_office_to_pdf=True,
                )

    def test_rubric_does_not_fail_when_libreoffice_unavailable(self) -> None:
        with patch("resources_servers.gdpval.setup_libreoffice.ensure_libreoffice", return_value=False):
            # Rubric mode tolerates missing libreoffice; the rubric path has its own
            # text-extraction fallback. Should not raise.
            _server(reward_mode="rubric", preconvert_office_to_pdf=True)

    def test_comparison_passes_when_libreoffice_available(self) -> None:
        with patch("resources_servers.gdpval.setup_libreoffice.ensure_libreoffice", return_value=True):
            _server(
                reward_mode="comparison",
                reference_deliverables_dir="/tmp/fork-deliverables",
                preconvert_office_to_pdf=True,
            )

    @pytest.mark.asyncio
    async def test_verify_rubric_no_rubric_returns_zero(self) -> None:
        server = _server(reward_mode="rubric")
        body = _verify_request(rubric_json=None, rubric_pretty="")
        resp = await server.verify(body)
        assert resp.reward == 0.0
        assert resp.verify_mode == "rubric"
        assert resp.invalid_judge_response is True

    @pytest.mark.asyncio
    async def test_verify_rubric_with_canned_judge(self) -> None:
        server = _server(reward_mode="rubric")

        canned_result = {"overall_score": 0.7, "criteria_scores": [{"score": 0.7}]}

        async def fake_score_with_rubric(**_kwargs):
            return 0.7, canned_result

        body = _verify_request(
            rubric_json=[{"criterion": "clarity", "score": 1}],
            deliverable_text="Deliverable body text.",
        )

        with (
            patch("resources_servers.gdpval.scoring.score_with_rubric", side_effect=fake_score_with_rubric),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
        ):
            resp = await server.verify(body)

        assert resp.reward == 0.7
        assert resp.verify_mode == "rubric"
        assert resp.invalid_judge_response is False
        assert resp.judge_response == canned_result

    @pytest.mark.asyncio
    async def test_verify_rubric_passes_create_overrides_through(self) -> None:
        """``judge_responses_create_params_overrides`` must reach the scoring fn.

        ``model`` and ``api_key`` are pulled out as their own kwargs; everything
        else (e.g. ``max_tokens``, ``temperature``) flows through as
        ``create_overrides`` and gets merged into ``client.chat.completions.create``.
        """
        server = _server(
            reward_mode="rubric",
            judge_responses_create_params_overrides={
                "model": "custom-judge",
                "api_key": "sk-custom",  # pragma: allowlist secret
                "max_tokens": 16384,
                "temperature": 0.0,
            },
        )

        captured: dict = {}

        async def fake_score_with_rubric(**kwargs):
            captured.update(kwargs)
            return 0.5, {"overall_score": 0.5}

        body = _verify_request(rubric_json=[{"criterion": "clarity", "score": 1}])

        with (
            patch("resources_servers.gdpval.scoring.score_with_rubric", side_effect=fake_score_with_rubric),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
        ):
            await server.verify(body)

        assert captured["model_name"] == "custom-judge"
        assert captured["api_key"] == "sk-custom"  # pragma: allowlist secret
        assert captured["create_overrides"] == {"max_tokens": 16384, "temperature": 0.0}

    @pytest.mark.asyncio
    async def test_verify_comparison_missing_reference(self, tmp_path) -> None:
        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir=str(tmp_path / "no-such-dir"),
        )
        body = _verify_request(rubric_json=[{"criterion": "clarity", "score": 1}])
        resp = await server.verify(body)
        assert resp.reward == 0.0
        assert resp.verify_mode == "comparison"
        assert resp.judge_response == {"error": "reference_missing"}

    @pytest.mark.asyncio
    async def test_verify_comparison_iterates_all_ref_repeats(self, tmp_path) -> None:
        """Each eval rollout is judged against every reference repeat and the
        raw vote counts are summed — not just one matchup against repeat_0."""
        ref_root = tmp_path / "ref"
        task_dir = ref_root / "task_task-1"
        for i in range(3):
            r = task_dir / f"repeat_{i}"
            r.mkdir(parents=True)
            (r / "finish_params.json").write_text("{}")
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")

        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir=str(ref_root),
            preconvert_office_to_pdf=False,
            num_comparison_trials=4,
        )

        seen_ref_dirs: list[str] = []

        def fake_run_trials(*, submission_a, **_kwargs):
            # ``build_file_section`` includes a ``"role": "user"`` text block
            # whose text is "Submission:\n" followed by the dir contents — we
            # just need to record which ref dir was passed.
            seen_ref_dirs.append(str(submission_a))
            # 3 eval wins (B), 1 ref win (A), 0 ties per ref repeat.
            return {
                "winner": "[[B]]",
                "win_count_a": 1,
                "win_count_b": 3,
                "tie_count": 0,
                "task_count": 4,
            }

        body = _verify_request(deliverables_dir=str(eval_dir))

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=fake_run_trials),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("resources_servers.gdpval.app.OpenAI" if False else "openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        # All three reference repeats must be judged.
        assert len(seen_ref_dirs) == 3
        # Vote totals: 3 ref repeats × (3 wins, 1 loss, 0 ties).
        assert resp.total_wins == 9
        assert resp.total_losses == 3
        assert resp.total_ties == 0
        assert resp.reward == 1.0
        assert resp.win is True
        assert resp.judge_response["ref_repeat_count"] == 3
        assert len(resp.judge_response["per_ref_repeat"]) == 3

    @pytest.mark.asyncio
    async def test_verify_comparison_flat_layout_back_compat(self, tmp_path) -> None:
        """Old ``task_<id>/`` flat reference layouts still work — one matchup."""
        ref_root = tmp_path / "ref"
        task_dir = ref_root / "task_task-1"
        task_dir.mkdir(parents=True)
        (task_dir / "finish_params.json").write_text("{}")
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")

        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir=str(ref_root),
            preconvert_office_to_pdf=False,
            num_comparison_trials=4,
        )

        call_count = {"n": 0}

        def fake_run_trials(**_kwargs):
            call_count["n"] += 1
            return {
                "winner": "[[A]]",
                "win_count_a": 4,
                "win_count_b": 0,
                "tie_count": 0,
                "task_count": 4,
            }

        body = _verify_request(deliverables_dir=str(eval_dir))

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=fake_run_trials),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        assert call_count["n"] == 1
        assert resp.total_wins == 0
        assert resp.total_losses == 4
        assert resp.reward == 0.0
        assert resp.loss is True

    @pytest.mark.asyncio
    async def test_persist_raw_judge_responses_comparison(self, tmp_path) -> None:
        """When persist_raw_judge_responses=True, raw judge text per trial flows
        through ``run_trials`` and lands on ``per_ref_repeat[i].raw_responses``."""
        ref_root = tmp_path / "ref"
        task_dir = ref_root / "task_task-1" / "repeat_0"
        task_dir.mkdir(parents=True)
        (task_dir / "finish_params.json").write_text("{}")
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")

        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir=str(ref_root),
            preconvert_office_to_pdf=False,
            num_comparison_trials=2,
            persist_raw_judge_responses=True,
        )

        captured_kwargs: dict = {}
        canned_raw = ["Trial 0 verdict: BOXED[B]", "Trial 1 (swapped) verdict: BOXED[A]"]

        def fake_run_trials(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "winner": "[[B]]",
                "win_count_a": 1,
                "win_count_b": 1,
                "tie_count": 0,
                "task_count": 2,
                "raw_responses": canned_raw,
            }

        body = _verify_request(deliverables_dir=str(eval_dir))

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=fake_run_trials),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        assert captured_kwargs["return_raw_responses"] is True
        assert resp.judge_response["per_ref_repeat"][0]["raw_responses"] == canned_raw

    @pytest.mark.asyncio
    async def test_persist_raw_judge_responses_rubric(self) -> None:
        """When persist_raw_judge_responses=True, the structured-rubric scorer
        gets ``include_raw_responses=True`` and the resulting metadata reaches
        ``judge_response``."""
        server = _server(reward_mode="rubric", rubric_scoring_mode="structured", persist_raw_judge_responses=True)

        captured_kwargs: dict = {}

        async def fake_score_structured(**kwargs):
            captured_kwargs.update(kwargs)
            return 0.7, {
                "scoring_method": "structured_rubric",
                "raw_responses": ["FINAL_SCORE[7]\nMAX_POSSIBLE_SCORE[10]"],
            }

        body = _verify_request(rubric_json=[{"criterion": "clarity", "score": 1}])

        with (
            patch("resources_servers.gdpval.scoring.score_with_rubric_structured", side_effect=fake_score_structured),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
        ):
            resp = await server.verify(body)

        assert captured_kwargs["include_raw_responses"] is True
        assert resp.judge_response["raw_responses"] == ["FINAL_SCORE[7]\nMAX_POSSIBLE_SCORE[10]"]

    def test_aggregate_metrics_comparison_elo(self) -> None:
        from nemo_gym.config_types import AggregateMetricsRequest

        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir="/tmp/fork-deliverables",
            reference_elo=1000.0,
        )

        def _row(task_idx, reward, win, loss, tie):
            return {
                "_ng_task_index": task_idx,
                "_ng_rollout_index": 0,
                "reward": reward,
                "win": win,
                "loss": loss,
                "tie": tie,
                "response": {},
            }

        responses = (
            [_row(i, 1.0, True, False, False) for i in range(7)]
            + [_row(7 + i, 0.0, False, True, False) for i in range(2)]
            + [_row(9, 0.5, False, False, True)]
        )
        import asyncio as _asyncio

        body = AggregateMetricsRequest(verify_responses=responses)
        result = _asyncio.run(server.aggregate_metrics(body))
        assert result.agent_metrics["comparison/wins"] == 7
        assert result.agent_metrics["comparison/losses"] == 2
        assert result.agent_metrics["comparison/ties"] == 1
        assert result.agent_metrics["comparison/judged"] == 10
        assert abs(result.agent_metrics["comparison/win_rate"] - 0.75) < 1e-6
        # win_rate=0.75 → ELO = 1000 - 400 * (log10(0.25) - log10(0.75)) ≈ 1190.85
        assert 1180 < result.agent_metrics["comparison/eval_elo"] < 1200

    def test_aggregate_metrics_uses_raw_vote_counts(self) -> None:
        """When verify responses carry ``total_wins``/``total_losses``/
        ``total_ties`` (multi-ref-repeat path), they're summed as raw judge
        votes rather than treated as one matchup each."""
        from nemo_gym.config_types import AggregateMetricsRequest

        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir="/tmp/fork-deliverables",
            reference_elo=1000.0,
        )
        # Two verify responses, each representing one eval_repeat × 3 ref
        # repeats × 4 trials = 12 judge votes.
        responses = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "reward": 1.0,
                "win": True,
                "loss": False,
                "tie": False,
                "total_wins": 9,
                "total_losses": 2,
                "total_ties": 1,
                "response": {},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "reward": 0.0,
                "win": False,
                "loss": True,
                "tie": False,
                "total_wins": 3,
                "total_losses": 8,
                "total_ties": 1,
                "response": {},
            },
        ]
        import asyncio as _asyncio

        body = AggregateMetricsRequest(verify_responses=responses)
        result = _asyncio.run(server.aggregate_metrics(body))
        assert result.agent_metrics["comparison/wins"] == 12
        assert result.agent_metrics["comparison/losses"] == 10
        assert result.agent_metrics["comparison/ties"] == 2
        assert result.agent_metrics["comparison/judged"] == 24
        # win_rate = (12 + 0.5*2) / 24 = 13/24 ≈ 0.5417
        assert abs(result.agent_metrics["comparison/win_rate"] - (13.0 / 24.0)) < 1e-6


class TestComparisonPayloadHardening:
    """Three protections against multi-hour /verify stalls observed on the
    multimodal-heavy long-tail tasks (task_a941b6d8 video, task_4b894ae3
    multi-stem audio): tmpdir zip extraction, per-file size cap, and
    APITimeoutError treated as non-retryable."""

    def test_maybe_unzip_extracts_to_tempdir_not_parent(self, tmp_path) -> None:
        """``_maybe_unzip`` must never write back into the zip's parent dir —
        the reference deliverables tree is read-only on the production bind
        mount, and ``extractall(path.parent)`` raised PermissionError."""
        import zipfile

        from resources_servers.gdpval.comparison import _maybe_unzip

        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        zip_path = ref_dir / "stems.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("stem_a.txt", "hello")
            zf.writestr("nested/stem_b.txt", "world")

        before = sorted(p.name for p in ref_dir.iterdir())
        extract_dir, members = _maybe_unzip(zip_path)
        after = sorted(p.name for p in ref_dir.iterdir())

        assert extract_dir is not None
        assert extract_dir.is_dir()
        assert extract_dir != ref_dir
        # Original ref dir must be unchanged.
        assert before == after == ["stems.zip"]
        # Extracted members live under the tmpdir.
        assert (extract_dir / "stem_a.txt").read_text() == "hello"
        assert (extract_dir / "nested" / "stem_b.txt").read_text() == "world"
        assert {p.name for p in members} >= {"stem_a.txt"}

    def test_maybe_unzip_handles_bad_zip(self, tmp_path) -> None:
        from resources_servers.gdpval.comparison import _maybe_unzip

        bad = tmp_path / "broken.zip"
        bad.write_bytes(b"not a zip")
        extract_dir, members = _maybe_unzip(bad)
        assert extract_dir is None
        assert members == []

    def test_get_file_content_block_rejects_oversize(self, tmp_path) -> None:
        """Files > MAX_FILE_BYTES_FOR_JUDGE are returned as a one-line text
        marker, not base64-encoded into the judge payload."""
        import os as _os

        from resources_servers.gdpval.comparison import (
            MAX_FILE_BYTES_FOR_JUDGE,
            get_file_content_block,
        )

        big = tmp_path / "huge.mp4"
        # Sparse file (truncate to size, no real disk/RAM cost) so this
        # stays cheap even when the cap is hundreds of MB.
        big.touch()
        _os.truncate(big, MAX_FILE_BYTES_FOR_JUDGE + 1)

        block = get_file_content_block(str(tmp_path), "huge.mp4")
        assert block == {
            "type": "text",
            "text": f"[oversize: huge.mp4 {(MAX_FILE_BYTES_FOR_JUDGE + 1) / (1024 * 1024):.1f}MB — not included]",
        }

    def test_get_file_content_block_includes_under_threshold(self, tmp_path) -> None:
        from resources_servers.gdpval.comparison import get_file_content_block

        small = tmp_path / "note.txt"
        small.write_text("hello world")
        block = get_file_content_block(str(tmp_path), "note.txt")
        assert block == {"type": "text", "text": "hello world"}

    def test_build_file_section_emits_zip_members_from_tempdir_and_cleans_up(self, tmp_path) -> None:
        """End-to-end: a zip in the source dir is extracted to tmp, its members
        appear as content blocks, and the tmpdir is registered for cleanup."""
        import zipfile

        from resources_servers.gdpval.comparison import build_file_section, clean_up_paths

        src = tmp_path / "src"
        src.mkdir()
        (src / "top_level.txt").write_text("top")
        with zipfile.ZipFile(src / "bundle.zip", "w") as zf:
            zf.writestr("inside.txt", "inner")

        clean_up_list: list = []
        section = build_file_section(str(src), clean_up_list)

        # both files appear (top_level.txt directly, inside.txt from zip).
        texts = [b.get("text", "") for b in section if b.get("type") == "text"]
        assert any("top_level.txt" in t for t in texts)
        assert any("inside.txt" in t for t in texts)
        assert any(t == "top" for t in texts)
        assert any(t == "inner" for t in texts)

        assert len(clean_up_list) == 1
        tmp_extract = clean_up_list[0]
        assert tmp_extract.exists()
        # Source dir is untouched.
        assert sorted(p.name for p in src.iterdir()) == ["bundle.zip", "top_level.txt"]

        clean_up_paths(clean_up_list)
        assert not tmp_extract.exists()

    def test_is_retryable_treats_apitimeout_as_non_retryable(self) -> None:
        """APITimeoutError must NOT be retried — multimodal payload timeouts
        are deterministic, not transient, and retrying burns 5×120s = 10 min
        per /verify with no chance of recovery."""
        from openai import APITimeoutError

        from resources_servers.gdpval.comparison import _is_retryable

        # Construct a minimal APITimeoutError. SDK signature is
        # ``APITimeoutError(request)`` since v1.x.
        try:
            err = APITimeoutError(request=object())
        except TypeError:
            # Older SDKs allow positional/no-arg.
            err = APITimeoutError("Request timed out.")
        assert _is_retryable(err) is False

    def test_is_retryable_still_retries_502_and_rate_limit(self) -> None:
        from resources_servers.gdpval.comparison import _is_retryable

        assert _is_retryable(RuntimeError("502 Bad Gateway")) is True
        assert _is_retryable(RuntimeError("429 Too Many Requests")) is True
        assert _is_retryable(RuntimeError("rate limit exceeded")) is True
        # ``timeout`` substring no longer triggers a blind retry.
        assert _is_retryable(RuntimeError("Request timed out")) is False
