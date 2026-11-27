# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-Index v1.0 / v1.1 release pins, cohort schema, and evaluation-only annotation loader."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resources_servers.video_mcqa.dataset import (
    AnnotatedQuestion,
    VideoSource,
    parse_options,
    text,
)
from resources_servers.video_mcqa.task import Setting, VerifierMetadata, VideoQuestion
from resources_servers.videoindex.scoring import (
    LETTERS,
    MAX_OPTIONS,
    VIDEOINDEX_RULE_POLICY,
    VideoIndexAnswerPolicy,
)


VideoIndexBenchmark = Literal["videoindex_v1_0", "videoindex_v1_1"]
CAPABILITY_GROUPS: tuple[str, ...] = (
    "perception",
    "temporal",
    "spatial_physical",
    "reasoning_knowledge",
)
AXES = {"capability_group"}


@dataclass(frozen=True, slots=True)
class VideoIndexRelease:
    """Pinned upstream Hugging Face release metadata for one Video-Index version."""

    benchmark: VideoIndexBenchmark
    repo_id: str
    dataset_source: str
    dataset_revision: str
    items_filename: str
    items_sha256: str
    num_questions: int
    metric_name: str
    items_env_var: str


RELEASES: dict[VideoIndexBenchmark, VideoIndexRelease] = {
    "videoindex_v1_0": VideoIndexRelease(
        benchmark="videoindex_v1_0",
        repo_id="Video-Index/Video-Index",
        dataset_source="https://huggingface.co/datasets/Video-Index/Video-Index",
        dataset_revision="fb63b6c93948e78e76aee9c87745bf78358d3011",  # pragma: allowlist secret
        items_filename="items/test.jsonl",
        items_sha256="6ebffa6ee53dccad983ab44f1a8e1a95cb77db6e04c1a440c384efea93d3635e",  # pragma: allowlist secret
        num_questions=840,
        metric_name="Video_Index_v1_0",
        items_env_var="VIDEOINDEX_V1_0_ITEMS",
    ),
    "videoindex_v1_1": VideoIndexRelease(
        benchmark="videoindex_v1_1",
        repo_id="Video-Index/Video-Index-v1-1",
        dataset_source="https://huggingface.co/datasets/Video-Index/Video-Index-v1-1",
        dataset_revision="e7404c468cd0ad83d7c96834c6eb4767fe831a72",  # pragma: allowlist secret
        items_filename="items/test.jsonl",
        items_sha256="b2b63478b548b5ad86d169693c56bb5844adbac8ba6ffeac73dd124b4846e20e",  # pragma: allowlist secret
        num_questions=144,
        metric_name="Video_Index_v1_1",
        items_env_var="VIDEOINDEX_V1_1_ITEMS",
    ),
}


def validate_videoindex_subsets(subsets: Mapping[str, tuple[str, ...]]) -> None:
    """Validate that each task belongs to exactly one of the four official Video-Index capability groups."""
    if set(subsets) != AXES:
        raise ValueError(f"Expected subset axes {sorted(AXES)}")
    labels = subsets["capability_group"]
    if len(labels) != 1:
        raise ValueError("Each Video-Index task must have exactly one capability_group label")
    label = labels[0]
    if text(label, field="capability_group") != label or label not in CAPABILITY_GROUPS:
        raise ValueError(f"Unknown Video-Index capability_group {label!r}")


class VideoIndexCohort(BaseModel):
    """Frozen Video-Index cohort: version identity, denominator, and capability-group membership."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    benchmark: VideoIndexBenchmark
    dataset_source: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    setting: Setting
    answer_policy: VideoIndexAnswerPolicy = VIDEOINDEX_RULE_POLICY
    task_ids: tuple[str, ...] = Field(min_length=1)
    task_subsets: dict[str, dict[str, tuple[str, ...]]]
    num_repeats: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def valid_cohort(self) -> Self:
        release = RELEASES[self.benchmark]
        if self.dataset_source != release.dataset_source:
            raise ValueError("Dataset source does not match this Video-Index release")
        if len(set(self.task_ids)) != len(self.task_ids) or any(
            not identity.startswith(f"{self.benchmark}:") or not identity.split(":", 1)[1].strip()
            for identity in self.task_ids
        ):
            raise ValueError("Expected unique, benchmark-scoped task IDs")
        if set(self.task_subsets) != set(self.task_ids):
            raise ValueError("Every frozen task must have complete subset membership")
        for subsets in self.task_subsets.values():
            validate_videoindex_subsets(subsets)
        return self


def normalize_video_relpath(value: object) -> str:
    """Validate an item's relative video path as ``videos/<filename>.mp4``."""
    raw = text(value, field="video")
    if any(char in raw for char in "\r\n\\"):
        raise ValueError(f"Invalid video path {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.parts[0] != "videos" or len(path.parts) != 2 or not path.name.endswith(".mp4"):
        raise ValueError(f"Expected video path 'videos/<name>.mp4', got {raw!r}")
    if ".." in path.parts or path.name in (".", ".."):
        raise ValueError(f"Invalid video path {raw!r}")
    return path.as_posix()


def load_videoindex(
    records: Iterable[Mapping[str, object]],
    *,
    benchmark: VideoIndexBenchmark,
    dataset_revision: str,
    media: Mapping[str, VideoSource],
) -> list[AnnotatedQuestion]:
    """Load Video-Index ``items/test.jsonl`` records without decoding media.

    ``media`` maps each item's relative ``video`` path (``videos/<name>.mp4``) to its explicit ``VideoSource``.
    """
    if benchmark not in RELEASES:
        raise ValueError(f"Unknown Video-Index benchmark {benchmark!r}")
    release = RELEASES[benchmark]
    result: list[AnnotatedQuestion] = []
    for record in records:
        if not isinstance(record, Mapping) or not all(isinstance(key, str) for key in record):
            raise ValueError("item record must be an object")
        item_id = text(record.get("item_id"), field="item_id")
        video_rel = normalize_video_relpath(record.get("video"))
        video_id = text(record.get("video_id", PurePosixPath(video_rel).stem), field="video_id")
        binding = media.get(video_rel)
        if binding is None:
            raise ValueError(f"Missing explicit media binding for video {video_rel!r}; no questions were dropped")
        options = parse_options(record.get("options"), strip_labels=False)
        if len(options) > MAX_OPTIONS:
            raise ValueError(f"{item_id}: Video-Index supports at most {MAX_OPTIONS} options, got {len(options)}")
        answer = text(record.get("answer"), field="answer")
        if "answer_idx" in record and record["answer_idx"] is not None:
            answer_idx = record["answer_idx"]
            if type(answer_idx) is not int or not 0 <= answer_idx < len(options):
                raise ValueError(f"{item_id}: invalid answer_idx {answer_idx!r} for {len(options)} options")
            if answer != LETTERS[answer_idx]:
                raise ValueError(f"{item_id}: answer {answer!r} disagrees with answer_idx {answer_idx}")
        group = text(record.get("capability_group"), field="capability_group")
        subsets = {"capability_group": (group,)}
        validate_videoindex_subsets(subsets)
        question = VideoQuestion(
            task_id=f"{benchmark}:{item_id}",
            dataset_source=release.dataset_source,
            dataset_revision=dataset_revision,
            video_id=video_id,
            video_source=binding.source_url,
            video_sha256=binding.sha256,
            question=text(record.get("question"), field="question"),
            options=options,
            allow_duplicate_options=len({option.strip() for option in options}) != len(options),
        )
        VerifierMetadata.from_question(question, expected_answer=answer)
        warnings = ("duplicate_option_text",) if question.allow_duplicate_options else ()
        result.append(AnnotatedQuestion(question, answer, subsets, warnings))
    if not result or len({item.question.task_id for item in result}) != len(result):
        raise ValueError("Annotations must contain nonempty, unique question IDs")
    return result
