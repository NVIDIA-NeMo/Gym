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

        With no ``judge_panel`` the server resolves a single judge from
        ``judge_model_server`` + the legacy overrides: ``model`` and ``api_key``
        become the judge's own fields; everything else (``max_tokens``,
        ``temperature``, …) is its ``create_overrides``, merged into
        ``client.chat.completions.create``.
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

        judges = captured["judges"]
        assert len(judges) == 1
        judge = judges[0]
        assert judge.model == "custom-judge"
        assert judge.api_key == "sk-custom"  # pragma: allowlist secret
        assert judge.base_url == "http://localhost:9999/v1"
        assert judge.create_overrides == {"max_tokens": 16384, "temperature": 0.0}

    @pytest.mark.asyncio
    async def test_verify_rubric_samples_from_judge_panel(self) -> None:
        """A configured ``judge_panel`` reaches the scoring fn as multiple judges.

        Each member resolves to its own model + reasoning knobs; members without
        an explicit ``model_server`` share the server-level ``judge_model_server``
        proxy (same base_url).
        """
        server = _server(
            reward_mode="rubric",
            judge_panel=[
                {
                    "name": "gpt-5.5",
                    "model": "openai/gpt-5.5",
                    "create_params_overrides": {"reasoning_effort": "medium"},
                },
                {"name": "gemini-3.1-pro", "model": "gcp/google/gemini-3.1-pro-preview"},
                {"name": "claude-opus-4.8", "model": "anthropic/claude-opus-4.8", "weight": 2.0},
            ],
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

        judges = captured["judges"]
        assert [j.name for j in judges] == ["gpt-5.5", "gemini-3.1-pro", "claude-opus-4.8"]
        assert [j.model for j in judges] == [
            "openai/gpt-5.5",
            "gcp/google/gemini-3.1-pro-preview",
            "anthropic/claude-opus-4.8",
        ]
        assert judges[0].create_overrides == {"reasoning_effort": "medium"}
        assert judges[2].weight == 2.0
        # All share the single proxy base_url.
        assert {j.base_url for j in judges} == {"http://localhost:9999/v1"}
        # A seeded rng is threaded through for reproducible sampling.
        assert captured["rng"] is not None

    @pytest.mark.asyncio
    async def test_verify_rubric_routes_audio_video_to_capable_judge(self, tmp_path) -> None:
        """Rubric mode routes a task with audio/video deliverables to the
        AV-capable panel member(s) before scoring."""
        deliv = tmp_path / "task_task-1" / "repeat_0"
        deliv.mkdir(parents=True)
        (deliv / "narration.mp3").write_bytes(b"\x00")

        server = _server(
            reward_mode="rubric",
            judge_panel=[
                {"name": "gpt-5.5", "model": "openai/gpt-5.5"},
                {"name": "gemini-3.1-pro", "model": "gcp/google/gemini", "handles_audio_video": True},
            ],
        )

        captured: dict = {}

        async def fake_score_with_rubric(**kwargs):
            captured.update(kwargs)
            return 0.5, {"overall_score": 0.5}

        body = _verify_request(rubric_json=[{"criterion": "clarity", "score": 1}], deliverables_dir=str(deliv))

        with (
            patch("resources_servers.gdpval.scoring.score_with_rubric", side_effect=fake_score_with_rubric),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            # Avoid pulling in real file-reader conversion for the .mp3 stub.
            patch("responses_api_agents.stirrup_agent.file_reader.read_deliverable_files", return_value=""),
            patch(
                "responses_api_agents.stirrup_agent.file_reader.convert_deliverables_to_content_blocks",
                return_value=[],
            ),
        ):
            await server.verify(body)

        assert [j.name for j in captured["judges"]] == ["gemini-3.1-pro"]

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
    async def test_verify_comparison_panel_metadata_and_per_judge(self, tmp_path) -> None:
        """Comparison mode passes the resolved panel to ``run_trials`` and folds
        each matchup's ``per_judge`` counts into eval-perspective panel totals,
        surfacing ``judge_panel`` + ``per_judge`` on the response."""
        ref_root = tmp_path / "ref"
        task_dir = ref_root / "task_task-1"
        for i in range(2):
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
            num_comparison_trials=2,
            judge_panel=[
                {"name": "gpt-5.5", "model": "openai/gpt-5.5"},
                {"name": "gemini-3.1-pro", "model": "gcp/google/gemini-3.1-pro-preview"},
            ],
        )

        captured: dict = {}

        def fake_run_trials(**kwargs):
            captured.update(kwargs)
            # eval (B) wins both trials; attribute one to each panel member.
            return {
                "winner": "[[B]]",
                "win_count_a": 0,
                "win_count_b": 2,
                "tie_count": 0,
                "task_count": 2,
                "per_judge": {
                    "gpt-5.5": {"win_count_a": 0, "win_count_b": 1, "tie_count": 0, "trials": 1},
                    "gemini-3.1-pro": {"win_count_a": 0, "win_count_b": 1, "tie_count": 0, "trials": 1},
                },
                "trial_judges": ["gpt-5.5", "gemini-3.1-pro"],
            }

        body = _verify_request(deliverables_dir=str(eval_dir))

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=fake_run_trials),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        # The panel (both members) is threaded into run_trials with a seeded rng.
        judges = captured["judges"]
        assert [j.name for j in judges] == ["gpt-5.5", "gemini-3.1-pro"]
        assert captured["rng"] is not None
        # Panel summary + pooled per-judge tally (over 2 ref repeats).
        assert [m["name"] for m in resp.judge_response["judge_panel"]] == ["gpt-5.5", "gemini-3.1-pro"]
        per_judge = resp.judge_response["per_judge"]
        assert per_judge["gpt-5.5"] == {"wins": 2, "losses": 0, "ties": 0, "trials": 2}
        assert per_judge["gemini-3.1-pro"] == {"wins": 2, "losses": 0, "ties": 0, "trials": 2}
        assert resp.total_wins == 4
        assert resp.reward == 1.0
        assert resp.judge_response["av_routed"] is False

    @pytest.mark.asyncio
    async def test_verify_comparison_routes_audio_video_to_capable_judge(self, tmp_path) -> None:
        """A task with an audio/video file is graded only by the AV-capable panel
        member(s) — the whole comparison is routed to that judge subset."""
        ref_root = tmp_path / "ref"
        ref_dir = ref_root / "task_task-1" / "repeat_0"
        ref_dir.mkdir(parents=True)
        (ref_dir / "finish_params.json").write_text("{}")
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")
        # Eval deliverable contains a video file → AV routing kicks in.
        (eval_dir / "walkthrough.mp4").write_bytes(b"\x00")

        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir=str(ref_root),
            preconvert_office_to_pdf=False,
            num_comparison_trials=2,
            judge_panel=[
                {"name": "gpt-5.5", "model": "openai/gpt-5.5"},
                {"name": "gemini-3.1-pro", "model": "gcp/google/gemini", "handles_audio_video": True},
                {"name": "claude-opus-4.8", "model": "anthropic/claude"},
            ],
        )

        captured: dict = {}

        def fake_run_trials(**kwargs):
            captured.update(kwargs)
            return {
                "winner": "[[B]]",
                "win_count_a": 0,
                "win_count_b": 2,
                "tie_count": 0,
                "task_count": 2,
                "per_judge": {"gemini-3.1-pro": {"win_count_a": 0, "win_count_b": 2, "tie_count": 0, "trials": 2}},
                "trial_judges": ["gemini-3.1-pro", "gemini-3.1-pro"],
            }

        body = _verify_request(deliverables_dir=str(eval_dir))

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=fake_run_trials),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        # Only the AV-capable member is passed to run_trials and recorded.
        assert [j.name for j in captured["judges"]] == ["gemini-3.1-pro"]
        assert resp.judge_response["av_routed"] is True
        assert [m["name"] for m in resp.judge_response["judge_panel"]] == ["gemini-3.1-pro"]

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


class TestMleElo:
    def test_single_reference_matches_closed_form(self) -> None:
        """For one reference the anchored MLE reduces exactly to ``calculate_elo``."""
        from resources_servers.gdpval.comparison import calculate_elo, calculate_mle_elo

        # 7 wins, 2 losses, 1 tie vs a single reference at ELO 1000 → win_rate 0.75.
        closed_elo, closed_norm = calculate_elo(0.75, 1000.0)
        mle = calculate_mle_elo([(1000.0, 7, 2, 1)])
        assert mle is not None
        mle_elo, mle_norm = mle
        assert abs(mle_elo - closed_elo) < 0.5
        assert abs(mle_norm - closed_norm) < 1e-3

    def test_two_references_fits_midpoint(self) -> None:
        """50% win rate vs two references at 1000 and 1400 ⇒ eval ELO ≈ 1200."""
        from resources_servers.gdpval.comparison import calculate_mle_elo

        mle = calculate_mle_elo([(1000.0, 5, 5, 0), (1400.0, 5, 5, 0)])
        assert mle is not None
        eval_elo, _ = mle
        assert abs(eval_elo - 1200.0) < 1.0

    def test_stronger_eval_gets_higher_rating(self) -> None:
        from resources_servers.gdpval.comparison import calculate_mle_elo

        weak = calculate_mle_elo([(1200.0, 2, 8, 0)])
        strong = calculate_mle_elo([(1200.0, 8, 2, 0)])
        assert weak is not None and strong is not None
        assert strong[0] > 1200.0 > weak[0]

    def test_degenerate_all_wins_is_finite_and_high(self) -> None:
        from resources_servers.gdpval.comparison import calculate_mle_elo

        mle = calculate_mle_elo([(1000.0, 10, 0, 0), (1100.0, 10, 0, 0)])
        assert mle is not None
        eval_elo, _ = mle
        import math as _math

        assert _math.isfinite(eval_elo)
        assert eval_elo > 1100.0

    def test_no_games_returns_none(self) -> None:
        from resources_servers.gdpval.comparison import calculate_mle_elo

        assert calculate_mle_elo([]) is None
        assert calculate_mle_elo([(1000.0, 0, 0, 0)]) is None


class TestMultiReference:
    @pytest.mark.asyncio
    async def test_verify_judges_every_reference_model(self, tmp_path) -> None:
        """Each eval rollout is judged against every configured reference model,
        and per-reference vote tallies are recorded on the response."""
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")

        ref_roots = {}
        for ref_id in ("kimi", "gpt5"):
            root = tmp_path / ref_id
            td = root / "task_task-1"
            td.mkdir(parents=True)
            (td / "finish_params.json").write_text("{}")
            ref_roots[ref_id] = root

        server = _server(
            reward_mode="comparison",
            reference_models={
                "kimi": {"deliverables_dir": str(ref_roots["kimi"]), "elo": 1290.0},
                "gpt5": {"deliverables_dir": str(ref_roots["gpt5"]), "elo": 1320.0},
            },
            preconvert_office_to_pdf=False,
            num_comparison_trials=4,
        )

        def fake_run_trials(**_kwargs):
            # 3 eval wins (B), 1 ref win (A) per matchup.
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
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        assert set(resp.per_reference) == {"kimi", "gpt5"}
        assert resp.per_reference["kimi"] == {
            "wins": 3,
            "losses": 1,
            "ties": 0,
            "reference_elo": 1290.0,
            "ref_repeat_count": 1,
        }
        assert resp.per_reference["gpt5"]["reference_elo"] == 1320.0
        # Totals are pooled across both references.
        assert resp.total_wins == 6
        assert resp.total_losses == 2
        assert resp.judge_response["reference_count"] == 2

    @pytest.mark.asyncio
    async def test_reference_ids_filter_judges_subset(self, tmp_path) -> None:
        """``reference_ids`` on the verify request restricts judging to the named
        references; unknown ids are ignored."""
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")

        ref_roots = {}
        for ref_id in ("kimi", "gpt5"):
            root = tmp_path / ref_id
            td = root / "task_task-1"
            td.mkdir(parents=True)
            (td / "finish_params.json").write_text("{}")
            ref_roots[ref_id] = root

        server = _server(
            reward_mode="comparison",
            reference_models={
                "kimi": {"deliverables_dir": str(ref_roots["kimi"]), "elo": 1290.0},
                "gpt5": {"deliverables_dir": str(ref_roots["gpt5"]), "elo": 1320.0},
            },
            preconvert_office_to_pdf=False,
            num_comparison_trials=4,
        )

        def fake_run_trials(**_kwargs):
            return {"winner": "[[B]]", "win_count_a": 1, "win_count_b": 3, "tie_count": 0, "task_count": 4}

        # Only judge against gpt5 (and an unknown id, which is ignored).
        body = _verify_request(deliverables_dir=str(eval_dir), reference_ids=["gpt5", "nonexistent"])

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=fake_run_trials),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        assert set(resp.per_reference) == {"gpt5"}
        assert resp.total_wins == 3
        assert resp.total_losses == 1
        assert resp.judge_response["reference_count"] == 1

    @pytest.mark.asyncio
    async def test_reference_ids_empty_yields_no_references(self, tmp_path) -> None:
        """An empty ``reference_ids`` list judges against nothing → reference_missing."""
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")
        root = tmp_path / "kimi"
        (root / "task_task-1").mkdir(parents=True)
        (root / "task_task-1" / "finish_params.json").write_text("{}")

        server = _server(
            reward_mode="comparison",
            reference_models={"kimi": {"deliverables_dir": str(root), "elo": 1290.0}},
            preconvert_office_to_pdf=False,
        )
        body = _verify_request(deliverables_dir=str(eval_dir), reference_ids=[])

        with patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"):
            resp = await server.verify(body)

        assert resp.reward == 0.0
        assert resp.judge_response == {"error": "reference_missing"}

    @staticmethod
    def _two_ref_server_and_body(tmp_path):
        eval_dir = tmp_path / "eval" / "task_task-1" / "repeat_0"
        eval_dir.mkdir(parents=True)
        (eval_dir / "finish_params.json").write_text("{}")
        ref_roots = {}
        for ref_id in ("kimi", "gpt5"):
            root = tmp_path / ref_id
            td = root / "task_task-1"
            td.mkdir(parents=True)
            (td / "finish_params.json").write_text("{}")
            ref_roots[ref_id] = root
        server = _server(
            reward_mode="comparison",
            reference_models={
                "kimi": {"deliverables_dir": str(ref_roots["kimi"]), "elo": 1284.0},
                "gpt5": {"deliverables_dir": str(ref_roots["gpt5"]), "elo": 1320.0},
            },
            preconvert_office_to_pdf=False,
            num_comparison_trials=4,
        )
        return server, _verify_request(deliverables_dir=str(eval_dir))

    @pytest.mark.asyncio
    async def test_verify_skips_failed_reference_keeps_others(self, tmp_path) -> None:
        """A judge failure on one reference is isolated: that reference is
        skipped (recorded under ``ref_errors``) while the others still score."""
        server, body = self._two_ref_server_and_body(tmp_path)

        calls = {"n": 0}

        def flaky_run_trials(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # First matchup (kimi) blows up the way a judge timeout would.
                raise RuntimeError("openai.APITimeoutError: Request timed out.")
            return {
                "winner": "[[B]]",
                "win_count_a": 1,
                "win_count_b": 3,
                "tie_count": 0,
                "task_count": 4,
            }

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=flaky_run_trials),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            resp = await server.verify(body)

        # Only the surviving reference contributes votes.
        assert set(resp.per_reference) == {"gpt5"}
        assert resp.total_wins == 3
        assert resp.total_losses == 1
        assert resp.reward == 1.0
        # The failed reference is recorded but does not abort the rollout.
        assert "kimi" in resp.judge_response["ref_errors"]
        assert resp.judge_response["reference_count"] == 1

    @pytest.mark.asyncio
    async def test_verify_all_references_fail_raises(self, tmp_path) -> None:
        """When every matchup fails the rollout is genuinely unjudgeable and
        verify raises (surfacing as a failure) rather than faking a reward."""
        server, body = self._two_ref_server_and_body(tmp_path)

        def always_fail(**_kwargs):
            raise RuntimeError("500 Internal Server Error")

        with (
            patch("resources_servers.gdpval.comparison.run_trials", side_effect=always_fail),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
            patch("resources_servers.gdpval.comparison.build_file_section", return_value=[]),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="all .* judge matchup"):
                await server.verify(body)

    def test_aggregate_metrics_mle_and_per_reference_stats(self) -> None:
        from nemo_gym.config_types import AggregateMetricsRequest

        server = _server(
            reward_mode="comparison",
            reference_models={
                "low": {"deliverables_dir": "/tmp/low", "elo": 1000.0},
                "high": {"deliverables_dir": "/tmp/high", "elo": 1400.0},
            },
        )

        # One verify response carrying a per-reference breakdown: 50% vs each
        # reference ⇒ MLE eval ELO ≈ 1200 (midpoint).
        responses = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "reward": 0.5,
                "total_wins": 10,
                "total_losses": 10,
                "total_ties": 0,
                "per_reference": {
                    "low": {"wins": 5, "losses": 5, "ties": 0, "reference_elo": 1000.0},
                    "high": {"wins": 5, "losses": 5, "ties": 0, "reference_elo": 1400.0},
                },
                "response": {},
            }
        ]
        import asyncio as _asyncio

        body = AggregateMetricsRequest(verify_responses=responses)
        result = _asyncio.run(server.aggregate_metrics(body))
        m = result.agent_metrics

        # Total win stats.
        assert m["comparison/wins"] == 10
        assert m["comparison/losses"] == 10
        assert m["comparison/judged"] == 20

        # Per-reference win stats.
        assert m["comparison/ref/low/wins"] == 5
        assert m["comparison/ref/low/judged"] == 10
        assert abs(m["comparison/ref/low/win_rate"] - 0.5) < 1e-9
        assert m["comparison/ref/low/reference_elo"] == 1000.0
        assert m["comparison/ref/high/reference_elo"] == 1400.0

        # MLE ELO over both references.
        assert m["comparison/num_references"] == 2
        assert abs(m["comparison/eval_elo"] - 1200.0) < 1.0
        # Every emitted metric value must be numeric (downstream coerces each
        # into a float Score — a string value fails parsing and fails the run).
        assert all(isinstance(v, (int, float)) for v in m.values())

    def test_single_stage_matches_unstaged_full_run(self) -> None:
        """A one-stage run is a special case of the full run: tagging the same
        rollouts with a single ``stage_index`` must yield the same headline
        eval_elo as the untagged (legacy full-220-vs-all-refs) aggregation."""
        from nemo_gym.config_types import AggregateMetricsRequest

        server = _server(
            reward_mode="comparison",
            reference_models={
                "low": {"deliverables_dir": "/tmp/low", "elo": 1000.0},
                "high": {"deliverables_dir": "/tmp/high", "elo": 1400.0},
            },
        )
        per_reference = {
            "low": {"wins": 7, "losses": 3, "ties": 0, "reference_elo": 1000.0},
            "high": {"wins": 3, "losses": 7, "ties": 0, "reference_elo": 1400.0},
        }
        base_row = {
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "task_id": "t0",
            "reward": 0.5,
            "total_wins": 10,
            "total_losses": 10,
            "total_ties": 0,
            "per_reference": per_reference,
            "response": {},
        }
        import asyncio as _asyncio

        unstaged = _asyncio.run(
            server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[dict(base_row)]))
        ).agent_metrics
        staged = _asyncio.run(
            server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[{**base_row, "stage_index": 0}]))
        ).agent_metrics

        # Headline ELO identical; the staged run just adds stage_* extras.
        assert staged["comparison/eval_elo"] == unstaged["comparison/eval_elo"]
        assert staged["comparison/num_references"] == unstaged["comparison/num_references"]
        assert staged["comparison/num_stages"] == 1
        assert staged["comparison/stage_0/eval_elo"] == unstaged["comparison/eval_elo"]
        # Untagged run carries no stage_* keys at all.
        assert not any(k.startswith("comparison/stage_") for k in unstaged)

    def test_aggregate_metrics_stage_aware_headline_is_last_stage(self) -> None:
        """When rollouts are tagged with ``stage_index`` the headline eval_elo is
        the LAST stage's fit, and every stage's estimate is emitted as an extra."""
        from nemo_gym.config_types import AggregateMetricsRequest

        server = _server(
            reward_mode="comparison",
            reference_models={
                "low": {"deliverables_dir": "/tmp/low", "elo": 1000.0},
                "high": {"deliverables_dir": "/tmp/high", "elo": 1400.0},
            },
        )

        # Stage 0: broad — vs both refs, 50/50 ⇒ stage ELO ≈ 1200 (midpoint).
        # Stage 1: refined — vs the nearby "low" ref only, strong 8/2 win ⇒
        # stage ELO well above 1000. Headline must equal the stage-1 fit, NOT
        # the pooled fit over all rollouts.
        responses = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "stage_index": 0,
                "task_id": "t0",
                "reward": 0.5,
                "total_wins": 5,
                "total_losses": 5,
                "total_ties": 0,
                "per_reference": {
                    "low": {"wins": 5, "losses": 5, "ties": 0, "reference_elo": 1000.0},
                    "high": {"wins": 5, "losses": 5, "ties": 0, "reference_elo": 1400.0},
                },
                "response": {},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "stage_index": 1,
                "task_id": "t1",
                "reward": 1.0,
                "total_wins": 8,
                "total_losses": 2,
                "total_ties": 0,
                "per_reference": {
                    "low": {"wins": 8, "losses": 2, "ties": 0, "reference_elo": 1000.0},
                },
                "response": {},
            },
        ]
        import asyncio as _asyncio

        body = AggregateMetricsRequest(verify_responses=responses)
        m = _asyncio.run(server.aggregate_metrics(body)).agent_metrics

        assert m["comparison/num_stages"] == 2
        # Per-stage estimates emitted.
        assert abs(m["comparison/stage_0/eval_elo"] - 1200.0) < 1.0
        assert m["comparison/stage_0/num_references"] == 2
        assert m["comparison/stage_0/num_tasks"] == 1
        assert m["comparison/stage_1/num_references"] == 1
        assert m["comparison/stage_1/num_tasks"] == 1
        # Stage 1 (0.8 win rate vs a 1000 anchor) must be above 1000.
        assert m["comparison/stage_1/eval_elo"] > 1000.0
        # Headline == last stage's fit, not the pooled midpoint.
        assert m["comparison/eval_elo"] == m["comparison/stage_1/eval_elo"]
        assert m["comparison/num_references"] == 1
        # Pooled descriptive win stats still cover every stage.
        assert m["comparison/wins"] == 13
        assert m["comparison/judged"] == 20
        assert all(isinstance(v, (int, float)) for v in m.values())

    def test_aggregate_metrics_handles_repeated_task_across_stages(self) -> None:
        """The same ``(task_index, rollout_index)`` may recur across stages (one
        rollout judged against a different reference subset per stage). The
        stage-aware aggregation must NOT trip RewardProfiler's duplicate-key
        guard — stages are distinguished by ``stage_index`` alone, with no
        per-stage index offset."""
        from nemo_gym.config_types import AggregateMetricsRequest

        server = _server(
            reward_mode="comparison",
            reference_models={
                "low": {"deliverables_dir": "/tmp/low", "elo": 1000.0},
                "high": {"deliverables_dir": "/tmp/high", "elo": 1400.0},
            },
        )

        # Identical (task_index=0, rollout_index=0) in both stages; only the
        # references judged and the stage_index differ.
        responses = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "stage_index": 0,
                "task_id": "t0",
                "reward": 0.5,
                "total_wins": 5,
                "total_losses": 5,
                "total_ties": 0,
                "per_reference": {
                    "low": {"wins": 5, "losses": 5, "ties": 0, "reference_elo": 1000.0},
                    "high": {"wins": 5, "losses": 5, "ties": 0, "reference_elo": 1400.0},
                },
                "response": {},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "stage_index": 1,
                "task_id": "t0",
                "reward": 1.0,
                "total_wins": 8,
                "total_losses": 2,
                "total_ties": 0,
                "per_reference": {
                    "low": {"wins": 8, "losses": 2, "ties": 0, "reference_elo": 1000.0},
                },
                "response": {},
            },
        ]
        import asyncio as _asyncio

        # Must not raise "Duplicate result row for rollout key".
        m = _asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))).agent_metrics

        assert m["comparison/num_stages"] == 2
        # Headline tracks the last stage; pooled descriptive stats span both.
        assert m["comparison/eval_elo"] == m["comparison/stage_1/eval_elo"]
        assert m["comparison/wins"] == 13
        assert m["comparison/judged"] == 20


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
