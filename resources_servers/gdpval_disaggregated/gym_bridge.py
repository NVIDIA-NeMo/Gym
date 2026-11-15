# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade saved GDPVal artifacts in an isolated process using the shared rubric scorer."""

import argparse
import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import BaseVerifyResponse
from resources_servers.gdpval.file_reader import (
    OFFICE_EXTS,
    convert_deliverables_to_content_blocks,
    read_deliverable_files,
)
from resources_servers.gdpval.judge_panel import ResolvedJudge, make_rng
from resources_servers.gdpval.scoring import SCORING_ERROR_KEY, score_with_rubric_structured


class JudgeConfig(BaseModel):
    """Coordinates and media settings passed by the resources server."""

    judge_url: str
    judge_model: str
    judge_timeout: float = Field(gt=0)
    judge_media_mode: Literal["native_pdf", "images_and_text"] = "images_and_text"
    judge_pdf_render_dpi: int = Field(default=144, ge=1)
    judge_pdf_max_pages: int = Field(default=120, ge=1)
    judge_handles_audio: bool = False
    judge_handles_video: bool = False
    judge_create_params: dict[str, Any] = Field(default_factory=dict)
    libreoffice_command: list[str] | None = None
    office_rendering_available: bool = True
    policy_failure_reason: Literal["agent_context_limit", "agent_iteration_limit", "agent_stuck_in_loop"] | None = None
    empty_submission_zero_reward: bool = False


def valid_reward(reward: object, judgement: object) -> bool:
    """Accept a completed structured-rubric trial, including a genuine zero."""
    return (
        isinstance(judgement, dict)
        and not judgement.get(SCORING_ERROR_KEY)
        and judgement.get("scoring_method") == "structured_rubric"
        and judgement.get("num_trials_completed", 0) >= 1
        and bool(judgement.get("scores"))
        and isinstance(reward, (float, int))
        and not isinstance(reward, bool)
        and math.isfinite(reward)
        and 0 <= reward <= 1
    )


def read_artifacts(root: Path, config: JudgeConfig) -> tuple[str, list[dict[str, Any]]]:
    """Read every submission directory with the shared GDPVal media converter.

    Keep directory labels: submissions may contain two files with the same basename.
    A converter error is an inconclusive measurement, never an invisible missing file.
    """
    if not config.office_rendering_available and any(
        path.is_file() and path.suffix.lower() in OFFICE_EXTS for path in root.rglob("*")
    ):
        raise ValueError("LibreOffice is unavailable; refusing to judge Office files without rendering")
    text_sections = []
    blocks = []
    directories = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    for directory in directories:
        label = str(directory.relative_to(root))
        text = read_deliverable_files(str(directory))
        if text:
            text_sections.append(f"Directory {label}:\n{text}")
        converted = convert_deliverables_to_content_blocks(
            str(directory),
            media_mode=config.judge_media_mode,
            render_dpi=config.judge_pdf_render_dpi,
            max_pages=config.judge_pdf_max_pages,
            include_text=True,
            audio_capable=config.judge_handles_audio,
            video_capable=config.judge_handles_video,
            libreoffice_command=config.libreoffice_command,
        )
        for block in converted:
            if block.get("type") == "text" and re.match(r"\s*[^\n]+: \[Error:", block.get("text", "")):
                raise ValueError("Unreadable GDPVal deliverable; refusing partial grading")
        if converted:
            blocks.extend([{"type": "text", "text": f"Directory {label}:"}, *converted])
    return "\n\n".join(text_sections), blocks


async def judge(task: Path, config: JudgeConfig) -> None:
    """Write a rollout only after a valid judgement or a completed policy failure."""
    trajectory = json.loads((task / "trajectory.json").read_text())
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(trajectory["responses_create_params"])
    response = NeMoGymResponse.model_validate(trajectory["response"])
    result = BaseVerifyResponse(responses_create_params=params, response=response, reward=0.0).model_dump(mode="json")
    failure_reason = config.policy_failure_reason
    if config.empty_submission_zero_reward or failure_reason:
        manifest = json.loads((task / "artifact-manifest.json").read_text())
        if not failure_reason and (
            any(entry["bytes"] for entry in manifest)
            or any(path.is_file() and path.stat().st_size for path in (task / "artifacts").rglob("*"))
        ):
            raise ValueError("Empty-submission flag disagrees with saved artifacts")
        result.update(reward_reason=failure_reason or "no_nonempty_deliverables", judge_skipped=True)
    else:
        metadata = json.loads((task / "judge.json").read_text())
        try:
            text, blocks = await asyncio.to_thread(read_artifacts, task / "artifacts", config)
        except Exception as error:
            (task / "judge-failure.json").write_text(
                json.dumps({"kind": "artifact_processing", "error_type": type(error).__name__}) + "\n"
            )
            raise
        resolved = ResolvedJudge(
            name=config.judge_model,
            base_url=config.judge_url,
            model=config.judge_model,
            create_overrides=config.judge_create_params,
            handles_audio=config.judge_handles_audio,
            handles_video=config.judge_handles_video,
            media_mode=config.judge_media_mode,
        )
        async with asyncio.timeout(config.judge_timeout):
            reward, judgement = await score_with_rubric_structured(
                deliverable_text=text,
                rubric_json=metadata.get("rubric_json"),
                rubric_pretty=metadata.get("rubric_pretty", ""),
                task_prompt=metadata["prompt"],
                judges=[resolved],
                rng=make_rng(None, metadata["task_id"], "rubric"),
                num_trials=1,
                formatting_retries=2,
                deliverable_content_blocks=blocks or None,
                include_raw_responses=True,
            )
        (task / "judge-response.json").write_text(json.dumps({"reward": reward, "judge_response": judgement}) + "\n")
        if not valid_reward(reward, judgement):
            raise ValueError("Invalid GDPVal judgement: no valid reward exported")
        result.update(reward=reward, verify_mode="rubric", judge_response=judgement)
    (task / "rollout.json").write_text(json.dumps(result) + "\n")
    (task / "reward.json").write_text(
        json.dumps({"valid": True, "reward": result["reward"], "judge_skipped": result.get("judge_skipped", False)})
        + "\n"
    )


def main() -> None:
    """Run one bounded judgement of a saved task directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    config = JudgeConfig.model_validate_json(args.config.read_text())
    asyncio.run(judge(args.task.resolve(), config))


if __name__ == "__main__":
    main()
