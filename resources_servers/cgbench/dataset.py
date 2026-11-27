# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CG-Bench (full / mini / hard) pinned slice metadata, cohort schema, and long_acc annotation loader."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resources_servers.cgbench.scoring import (
    CGBENCH_RULE_POLICY,
    LETTERS,
    MAX_OPTIONS,
    CGBenchAnswerPolicy,
)
from resources_servers.video_mcqa.dataset import (
    AnnotatedQuestion,
    VideoSource,
    parse_options,
    text,
)
from resources_servers.video_mcqa.task import Setting, VerifierMetadata, VideoQuestion


CGBenchBenchmark = Literal["cgbench", "cgbench_mini", "cgbench_hard"]

DATASET_REPO_ID = "CG-Bench/CG-Bench"
DATASET_SOURCE = "https://huggingface.co/datasets/CG-Bench/CG-Bench"
DATASET_REVISION = "aa63c0b7f74b5b43b5e0c1f012ea96bc9f460c57"  # pragma: allowlist secret

CATEGORIES: tuple[str, ...] = (
    "Perception",
    "Cognition",
    "Hallucination",
)

SUB_CATEGORY_TO_CATEGORY: dict[str, str] = {
    "Entity Perception": "Perception",
    "Event Perception": "Perception",
    "Text Perception": "Perception",
    "Time Perception": "Perception",
    "2D Spatial Perception": "Perception",
    "Scene Perception": "Perception",
    "Entity Cognition": "Cognition",
    "Event Cognition": "Cognition",
    "Time Cognition": "Cognition",
    "Text Cognition": "Cognition",
    "Scene Cognition": "Cognition",
    "Hallucination": "Hallucination",
}
SUB_CATEGORIES: tuple[str, ...] = tuple(SUB_CATEGORY_TO_CATEGORY)

# Official domain labels from CG-Bench/CG-Bench JSON annotations (retaining upstream spelling).
DOMAINS: tuple[str, ...] = (
    "Life Record",
    "Music & TV show",
    "Instruction & Knowledge",
    "Driving",
    "Embodied Expert",
    "Humor/funny",
    "Electonic/Social Gaming",
    "Security & Health",
    "Sports & Exercise",
    "Special Scenes",
    "Art & Culture",
    "GUI",
    "News",
    "Animal & Pet",
)

# Official CG-Bench 10-minute video duration buckets (right-closed minute intervals matching VLMEvalKit/CG-Bench).
DURATION_BUCKETS: tuple[str, ...] = (
    "0-10m",
    "10-20m",
    "20-30m",
    "30-40m",
    "40-50m",
    "50-60m",
    "60m+",
)

AXES = {"category", "sub_category", "domain", "duration"}


@dataclass(frozen=True, slots=True)
class CGBenchSlice:
    """Pinned upstream metadata for one CG-Bench evaluation slice."""

    benchmark: CGBenchBenchmark
    json_filename: str
    json_sha256: str
    num_questions: int
    num_videos: int
    metric_name: str
    json_env_var: str


SLICES: dict[CGBenchBenchmark, CGBenchSlice] = {
    "cgbench": CGBenchSlice(
        benchmark="cgbench",
        json_filename="cgbench.json",
        json_sha256="f2db9d2e8144e30806c3eab3ea50e7e1681bf9dd31d0dd6ce2871f960559de12",  # pragma: allowlist secret
        num_questions=12129,
        num_videos=1219,
        metric_name="CG_Bench",
        json_env_var="CGBENCH_JSON",
    ),
    "cgbench_mini": CGBenchSlice(
        benchmark="cgbench_mini",
        json_filename="cgbench_mini.json",
        json_sha256="907a2b4cc8849b852a928e7339794fb2bc43b13ee0fd561af5d697fd0e03da90",  # pragma: allowlist secret
        num_questions=3000,
        num_videos=1118,
        metric_name="CG_Bench_mini",
        json_env_var="CGBENCH_MINI_JSON",
    ),
    "cgbench_hard": CGBenchSlice(
        benchmark="cgbench_hard",
        json_filename="cgbench_hard.json",
        json_sha256="4a1218847060e8608ea99e891961be2a384bd6e19c746807f3742423149f099a",  # pragma: allowlist secret
        num_questions=2417,
        num_videos=1010,
        metric_name="CG_Bench_hard",
        json_env_var="CGBENCH_HARD_JSON",
    ),
}


def duration_bucket(seconds: object) -> str:
    """Map a video duration in seconds to its official CG-Bench minute bucket (right-closed intervals)."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError(f"duration must be a positive number of seconds, got {seconds!r}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"duration must be a positive finite number of seconds, got {seconds!r}")
    minutes = float(seconds) / 60.0
    if minutes <= 10.0:
        return "0-10m"
    if minutes <= 20.0:
        return "10-20m"
    if minutes <= 30.0:
        return "20-30m"
    if minutes <= 40.0:
        return "30-40m"
    if minutes <= 50.0:
        return "40-50m"
    if minutes <= 60.0:
        return "50-60m"
    return "60m+"


def normalize_video_uid(value: object) -> str:
    """Validate a CG-Bench ``video_uid`` as a safe single filename stem."""
    uid = text(value, field="video_uid")
    if uid in (".", "..") or ".." in uid or any(char in uid for char in "/\\ \r\n\t"):
        raise ValueError(f"Invalid CG-Bench video_uid {uid!r}")
    return uid


def validate_cgbench_subsets(subsets: Mapping[str, tuple[str, ...]]) -> None:
    """Validate official CG-Bench category, sub_category, domain, and duration bucket membership."""
    if set(subsets) != AXES:
        raise ValueError(f"Expected subset axes {sorted(AXES)}")
    for axis, labels in subsets.items():
        if len(labels) != 1:
            raise ValueError(f"Each CG-Bench task must have exactly one {axis} label")
        label = labels[0]
        if text(label, field=axis) != label or any(char in label for char in "\r\n\t"):
            raise ValueError(f"Invalid CG-Bench {axis} label {label!r}")
    category = subsets["category"][0]
    sub_category = subsets["sub_category"][0]
    domain = subsets["domain"][0]
    duration = subsets["duration"][0]
    if sub_category not in SUB_CATEGORY_TO_CATEGORY:
        raise ValueError(f"Unknown CG-Bench sub_category {sub_category!r}")
    if category != SUB_CATEGORY_TO_CATEGORY[sub_category]:
        raise ValueError(f"CG-Bench category {category!r} disagrees with sub_category {sub_category!r}")
    if domain not in DOMAINS:
        raise ValueError(f"Unknown CG-Bench domain {domain!r}")
    if duration not in DURATION_BUCKETS:
        raise ValueError(f"Unknown CG-Bench duration bucket {duration!r}")


class CGBenchCohort(BaseModel):
    """Frozen CG-Bench long_acc cohort: slice identity, denominator, and official dimension tags."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    benchmark: CGBenchBenchmark
    dataset_source: str = DATASET_SOURCE
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    setting: Setting
    answer_policy: CGBenchAnswerPolicy = CGBENCH_RULE_POLICY
    task_ids: tuple[str, ...] = Field(min_length=1)
    task_subsets: dict[str, dict[str, tuple[str, ...]]]
    num_repeats: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def valid_cohort(self) -> Self:
        if self.dataset_source != DATASET_SOURCE:
            raise ValueError("Dataset source does not match CG-Bench/CG-Bench")
        if len(set(self.task_ids)) != len(self.task_ids) or any(
            not identity.startswith(f"{self.benchmark}:") or not identity.split(":", 1)[1].strip()
            for identity in self.task_ids
        ):
            raise ValueError("Expected unique, benchmark-scoped task IDs")
        if set(self.task_subsets) != set(self.task_ids):
            raise ValueError("Every frozen task must have complete subset membership")
        for subsets in self.task_subsets.values():
            validate_cgbench_subsets(subsets)
        return self


def load_cgbench(
    records: Iterable[Mapping[str, object]],
    *,
    benchmark: CGBenchBenchmark,
    dataset_revision: str,
    media: Mapping[str, VideoSource],
) -> list[AnnotatedQuestion]:
    """Load CG-Bench JSON records for full-video multiple-choice ``long_acc`` (no subtitles, no clue intervals).

    ``media`` maps each item's ``video_uid`` to its explicit ``VideoSource``. ``clue_intervals`` are
    intentionally ignored and never exposed in prompts or metadata.
    """
    if benchmark not in SLICES:
        raise ValueError(f"Unknown CG-Bench slice {benchmark!r}")
    result: list[AnnotatedQuestion] = []
    for record in records:
        if not isinstance(record, Mapping) or not all(isinstance(key, str) for key in record):
            raise ValueError("annotation record must be an object")
        raw_qid = record.get("qid")
        if type(raw_qid) is int and raw_qid >= 0:
            qid = str(raw_qid)
        else:
            qid = text(raw_qid, field="qid")
        video_uid = normalize_video_uid(record.get("video_uid"))
        binding = media.get(video_uid)
        if binding is None:
            raise ValueError(f"Missing explicit media binding for video {video_uid!r}; no questions were dropped")
        # Preserve literal option text (e.g. "J. CREW", "S. R. Nathan") without stripping prefixes.
        options = parse_options(record.get("choices"), strip_labels=False)
        if len(options) > MAX_OPTIONS:
            raise ValueError(f"{qid}: CG-Bench supports at most {MAX_OPTIONS} options, got {len(options)}")
        right_answer = text(record.get("right_answer"), field="right_answer")
        if right_answer not in tuple(LETTERS[: len(options)]):
            raise ValueError(f"{qid}: right_answer {right_answer!r} is outside A-{LETTERS[len(options) - 1]}")
        raw_answer = record.get("answer")
        if raw_answer is not None:
            answer_text = text(raw_answer, field="answer")
            expected_choice_text = options[LETTERS.index(right_answer)]
            if answer_text != expected_choice_text:
                raise ValueError(
                    f"{qid}: answer {answer_text!r} disagrees with choices[{right_answer}]={expected_choice_text!r}"
                )
        sub_category = text(record.get("sub_category"), field="sub_category")
        if sub_category not in SUB_CATEGORY_TO_CATEGORY:
            raise ValueError(f"{qid}: unknown CG-Bench sub_category {sub_category!r}")
        domain = text(record.get("domain"), field="domain")
        bucket = duration_bucket(record.get("duration"))
        subsets = {
            "category": (SUB_CATEGORY_TO_CATEGORY[sub_category],),
            "sub_category": (sub_category,),
            "domain": (domain,),
            "duration": (bucket,),
        }
        validate_cgbench_subsets(subsets)
        question = VideoQuestion(
            task_id=f"{benchmark}:{qid}",
            dataset_source=DATASET_SOURCE,
            dataset_revision=dataset_revision,
            video_id=video_uid,
            video_source=binding.source_url,
            video_sha256=binding.sha256,
            question=text(record.get("question"), field="question"),
            options=options,
            allow_duplicate_options=len({option.strip() for option in options}) != len(options),
        )
        VerifierMetadata.from_question(question, expected_answer=right_answer)
        warnings = ("duplicate_option_text",) if question.allow_duplicate_options else ()
        result.append(AnnotatedQuestion(question, right_answer, subsets, warnings))
    if not result or len({item.question.task_id for item in result}) != len(result):
        raise ValueError("Annotations must contain nonempty, unique question IDs")
    return result
