# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local, evaluation-only annotation adapters; never download or decode media."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resources_servers.video_mcqa.task import VerifierMetadata, VideoQuestion


Benchmark = Literal["videomme_v1", "lvbench"]
SOURCES = {
    "videomme_v1": "https://huggingface.co/datasets/lmms-lab/Video-MME",
    "lvbench": "https://huggingface.co/datasets/THUDM/LVBench",
}
LV_CATEGORIES = (
    "entity recognition",
    "event understanding",
    "key information retrieval",
    "temporal grounding",
    "reasoning",
    "summarization",
)
AXES = {
    "videomme_v1": {"duration", "domain", "sub_category", "task_type"},
    "lvbench": {"domain", "task_type"},
}


def text(value: object, *, field: str) -> str:
    """Reject silent coercion, empty strings and control characters."""
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise ValueError(f"{field} must be nonempty text without control characters")
    return value.strip()


def validate_subsets(benchmark: Benchmark, subsets: Mapping[str, tuple[str, ...]]) -> None:
    """Validate complete category membership without turning overlap into a macro average."""
    if set(subsets) != AXES[benchmark]:
        raise ValueError(f"Expected subset axes {sorted(AXES[benchmark])}")
    for axis, labels in subsets.items():
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Subset labels must be nonempty and unique")
        for label in labels:
            if text(label, field=axis) != label or any(char in label for char in "\r\n\t"):
                raise ValueError("Subset labels must be trimmed, single-line text")
        if (benchmark != "lvbench" or axis != "task_type") and len(labels) != 1:
            raise ValueError("Only LVBench task_type may have multiple labels")
    if benchmark == "videomme_v1" and subsets["duration"][0] not in {"short", "medium", "long"}:
        raise ValueError("VideoMME-v1 duration must be short, medium or long")
    if benchmark == "lvbench" and not set(subsets["task_type"]).issubset(LV_CATEGORIES):
        raise ValueError("Unknown LVBench capability category")


class VideoSource(BaseModel):
    """An explicit binding, not a filename inferred from a question or answer.

    video_url is produced by the single-turn VideoAnyRes preparation layer.
    source and sandbox_path seed the sandboxed agent's video transport.
    The caller supplies the SHA256 of the original video, not of an NPZ window.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_url: str | None = None
    source: str | None = None
    sandbox_path: str | None = None

    @model_validator(mode="after")
    def valid_strings(self) -> Self:
        for name in ("source_url", "video_url", "source", "sandbox_path"):
            value = getattr(self, name)
            if value is not None:
                text(value, field=name)
                if any(char in value for char in "\r\n"):
                    raise ValueError(f"{name} must be a single-line value")
        return self


@dataclass(frozen=True, slots=True)
class AnnotatedQuestion:
    """Normalized question, private answer, and reporting-only category tags."""

    question: VideoQuestion
    expected_answer: str
    subsets: dict[str, tuple[str, ...]]
    annotation_warnings: tuple[str, ...] = ()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    return list(value)


def _options(value: object) -> tuple[str, ...]:
    values = [text(item, field="option") for item in _sequence(value, field="options")]
    if len(values) != 4:
        raise ValueError("VideoMME-v1 and LVBench require exactly four choices")
    matches = [re.match(r"^(?:([A-D])[.)]|\(([A-D])\))\s+", option) for option in values]
    if any(matches):
        if not all(matches) or [match[1] or match[2] for match in matches] != list("ABCD"):
            raise ValueError("Choice labels must occur once in canonical A-D order")
        values = [option[match.end() :].strip() for option, match in zip(values, matches)]
    return tuple(values)


def _embedded_options(value: object) -> tuple[str, tuple[str, ...]]:
    question = text(value, field="question")
    matches = list(re.finditer(r"(?m)^\s*\(([A-H])\)\s*", question))
    if [match[1] for match in matches] != list("ABCD"):
        raise ValueError("LVBench question must end with canonical (A)-(D) options")
    stem = text(question[: matches[0].start()], field="question stem")
    options = tuple(
        text(question[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None], field="option")
        for index, match in enumerate(matches)
    )
    return stem, options


def _normalized(
    *,
    benchmark: Benchmark,
    dataset_revision: str,
    identity: object,
    video_id: object,
    question: object,
    options: tuple[str, ...],
    answer: object,
    subsets: dict[str, tuple[str, ...]],
    media: Mapping[str, VideoSource],
) -> AnnotatedQuestion:
    video_id = text(video_id, field="video ID")
    binding = media.get(video_id)
    if binding is None:
        raise ValueError(f"Missing explicit media binding for video {video_id!r}; no questions were dropped")
    validate_subsets(benchmark, subsets)
    normalized = VideoQuestion(
        task_id=f"{benchmark}:{text(identity, field='question ID')}",
        dataset_source=SOURCES[benchmark],
        dataset_revision=dataset_revision,
        video_id=video_id,
        video_source=binding.source_url,
        video_sha256=binding.sha256,
        question=text(question, field="question"),
        options=options,
        allow_duplicate_options=len({option.strip() for option in options}) != len(options),
    )
    answer = text(answer, field="answer")
    VerifierMetadata.from_question(normalized, expected_answer=answer)
    warnings = ("duplicate_option_text",) if normalized.allow_duplicate_options else ()
    return AnnotatedQuestion(normalized, answer, subsets, warnings)


def _unique(rows: Iterable[AnnotatedQuestion]) -> list[AnnotatedQuestion]:
    result = list(rows)
    if not result or len({row.question.task_id for row in result}) != len(result):
        raise ValueError("Annotations must contain nonempty, unique question IDs")
    return result


def load_videomme_v1(
    records: Iterable[Mapping[str, object]], *, dataset_revision: str, media: Mapping[str, VideoSource]
) -> list[AnnotatedQuestion]:
    """Read official HF flat rows or the official nested per-video JSON layout."""
    result = []
    for record in records:
        record = _mapping(record, field="video record")
        questions = _sequence(record["questions"], field="questions") if "questions" in record else [record]
        for value in questions:
            row = _mapping(value, field="question")
            result.append(
                _normalized(
                    benchmark="videomme_v1",
                    dataset_revision=dataset_revision,
                    identity=row.get("question_id"),
                    video_id=record.get("videoID", record.get("video_id")),
                    question=row.get("question"),
                    options=_options(row.get("options")),
                    answer=row.get("answer"),
                    subsets={
                        "duration": (text(record.get("duration"), field="duration"),),
                        "domain": (text(record.get("domain"), field="domain"),),
                        "sub_category": (text(record.get("sub_category"), field="sub_category"),),
                        "task_type": (text(row.get("task_type"), field="task_type"),),
                    },
                    media=media,
                )
            )
    return _unique(result)


def load_lvbench(
    records: Iterable[Mapping[str, object]], *, dataset_revision: str, media: Mapping[str, VideoSource]
) -> list[AnnotatedQuestion]:
    """Read THUDM's official video_info.meta.jsonl, or a lossless flattened export.

    time_reference, source captions and any pre-existing model response are
    intentionally ignored. Multiple question_type labels remain overlapping.
    """
    result = []
    for record in records:
        record = _mapping(record, field="video record")
        questions = _sequence(record["qa"], field="qa") if "qa" in record else [record]
        for value in questions:
            row = _mapping(value, field="question")
            stem, options = _embedded_options(row.get("question"))
            identity = row.get("uid")
            # The official JSONL mixes integer and string UIDs; its scorer
            # explicitly converts them to strings before looking up answers.
            if type(identity) is int and identity >= 0:
                identity = str(identity)
            result.append(
                _normalized(
                    benchmark="lvbench",
                    dataset_revision=dataset_revision,
                    identity=identity,
                    video_id=record.get("key"),
                    question=stem,
                    options=options,
                    answer=row.get("answer"),
                    subsets={
                        "domain": (text(record.get("type"), field="type"),),
                        "task_type": tuple(
                            text(label, field="question_type")
                            for label in _sequence(row.get("question_type"), field="question_type")
                        ),
                    },
                    media=media,
                )
            )
    return _unique(result)
