# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
    AABriefcaseLiteVerifyRequest,
)
from resources_servers.gdpval.judge_panel import ResolvedJudge, sample_judge


async def test_binary_checks_select_only_requested_artifacts_and_capable_judges(monkeypatch, tmp_path):
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    server = AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(
            dataset_dir=str(tmp_path), preconvert_office_to_pdf=False
        )
    )
    subtitles = "1\n00:00:00,000 --> 00:00:01,000\nHello.\n"
    (tmp_path / "captions.srt").write_text(subtitles)
    (tmp_path / "clip.mp4").write_bytes(b"video")
    (tmp_path / "unrequested.txt").write_text("Unrelated content")
    checks = [
        {
            "task_id": "w1_t4",
            "scoring_type": "binary",
            "check_id": name,
            "check_type": "format",
            "taskdoer_output_file": filename,
        }
        for name, filename in [("subtitles", "captions.srt"), ("video", "clip.mp4"), ("missing", "absent.srt")]
    ]
    server._aa_checks = checks
    judges = [
        ResolvedJudge(name=name, model=name, base_url="http://judge.invalid/v1", handles_video=name == "gemini")
        for name in ("gpt", "claude", "gemini")
    ]
    eligible_panels = []

    def choose(panel, rng):
        eligible_panels.append([judge.name for judge in panel])
        return sample_judge(panel, rng)

    call = AsyncMock(return_value=({"passed": True, "reasoning": "test response"}, "{}"))
    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.sample_judge", choose)
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "_binary_call", call)
    reward, results = await server._verify_binary(
        AABriefcaseLiteVerifyRequest.model_construct(task_id="w1_t4", deliverables_dir=str(tmp_path)),
        "Create a video and captions.",
        judges,
    )

    # The check whose file was never submitted is the model's zero and does not reach a judge.
    assert eligible_panels == [["gpt", "claude", "gemini"], ["gemini"]]
    subtitle_blocks, video_blocks = [entry.args[3] for entry in call.call_args_list]
    assert {"type": "text", "text": subtitles} in subtitle_blocks
    assert all(block["type"] == "text" for block in subtitle_blocks)
    assert any(block["type"] != "text" for block in video_blocks)
    assert subtitles not in str(video_blocks)
    assert "Unrelated content" not in str(call.call_args_list)
    assert reward == pytest.approx(2 / 3)
    missing = next(item for item in results if item["check_id"] == "missing")
    assert missing["passed"] is False
    assert missing["judge_name"] is None
    assert missing["missing_deliverables"] == ["absent.srt"]

    # A required artifact that disappears is a zero for its check, and the other checks keep their judges.
    (tmp_path / "clip.mp4").unlink()
    reward, results = await server._verify_binary(
        AABriefcaseLiteVerifyRequest.model_construct(task_id="w1_t4", deliverables_dir=str(tmp_path)),
        "Create a video and captions.",
        judges,
    )
    assert reward == pytest.approx(1 / 3)
    assert eligible_panels[2:] == [["gpt", "claude", "gemini"]]
    assert call.call_args_list[2].args[0].name == call.call_args_list[0].args[0].name
    assert [item["passed"] for item in results] == [True, False, False]


async def test_a_deliverable_in_another_encoding_is_judged_not_an_evaluation_error(monkeypatch, tmp_path):
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    server = AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(
            dataset_dir=str(tmp_path), preconvert_office_to_pdf=False
        )
    )
    # Windows-1252 smart quotes, as an agent's script can write them.
    (tmp_path / "captions.srt").write_bytes("1\n00:00:00,000 --> 00:00:01,000\n\x93Hello\x94\n".encode("latin-1"))
    (tmp_path / "report.tex").write_bytes("\\section{Caf\xe9}".encode("latin-1"))
    server._aa_checks = [
        {
            "task_id": "w1_t1",
            "scoring_type": "binary",
            "check_id": name,
            "check_type": "format",
            "taskdoer_output_file": filename,
        }
        for name, filename in [("subtitles", "captions.srt"), ("source", "report.tex")]
    ]
    judge = ResolvedJudge(name="gpt", model="gpt", base_url="http://judge.invalid/v1")
    call = AsyncMock(return_value=({"passed": True, "reasoning": "test response"}, "{}"))
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "_binary_call", call)

    reward, _results = await server._verify_binary(
        AABriefcaseLiteVerifyRequest.model_construct(task_id="w1_t1", deliverables_dir=str(tmp_path)),
        "Write the captions and the report.",
        [judge],
    )

    assert reward == 1.0
    subtitle_blocks, source_blocks = [entry.args[3] for entry in call.call_args_list]
    assert "\ufffdHello\ufffd" in str(subtitle_blocks)
    assert "Caf\ufffd" in str(source_blocks)
