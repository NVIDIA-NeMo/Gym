# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

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
    await server._verify_binary(
        AABriefcaseLiteVerifyRequest.model_construct(task_id="w1_t4", deliverables_dir=str(tmp_path)),
        "Create a video and captions.",
        judges,
    )

    assert eligible_panels == [["gpt", "claude", "gemini"], ["gemini"], ["gpt", "claude", "gemini"]]
    subtitle_blocks, video_blocks, missing_blocks = [entry.args[3] for entry in call.call_args_list]
    assert {"type": "text", "text": subtitles} in subtitle_blocks
    assert all(block["type"] == "text" for block in subtitle_blocks)
    assert any(block["type"] != "text" for block in video_blocks)
    assert subtitles not in str(video_blocks)
    assert "Unrelated content" not in str(call.call_args_list)
    assert {"type": "text", "text": "[required submitted file missing: absent.srt]"} in missing_blocks
    assert subtitles not in str(missing_blocks)

    # Removing a required artifact changes the evidence, not the assigned judge.
    (tmp_path / "clip.mp4").unlink()
    await server._verify_binary(
        AABriefcaseLiteVerifyRequest.model_construct(task_id="w1_t4", deliverables_dir=str(tmp_path)),
        "Create a video and captions.",
        judges,
    )
    assert eligible_panels[3:] == eligible_panels[:3]
    first_judges = [entry.args[0].name for entry in call.call_args_list[:3]]
    second_judges = [entry.args[0].name for entry in call.call_args_list[3:]]
    assert first_judges == second_judges
    assert {"type": "text", "text": "[required submitted file missing: clip.mp4]"} in call.call_args_list[4].args[3]
