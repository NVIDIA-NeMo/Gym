# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-independent separation of model inputs, gold answers and provenance."""

import re
from dataclasses import asdict, dataclass
from typing import Literal, NotRequired, TypedDict


Setting = Literal["single_turn", "agentic"]
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string without NUL bytes")


def _digest(value: str, length: int, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{name} must be a fixed {length}-character lowercase hex digest")


@dataclass(frozen=True, slots=True)
class VideoQuestion:
    """A question and immutable source identity, with no gold-answer field.

    Media preparation resolves the source separately. Neither this contract nor
    its renderers decode videos, select frames, extend context, or call a model.
    """

    task_id: str
    dataset_source: str
    dataset_revision: str
    video_id: str
    video_source: str
    video_sha256: str
    question: str
    options: tuple[str, ...]
    level: int | None = None
    group_id: str | None = None
    allow_duplicate_options: bool = False

    def __post_init__(self) -> None:
        for name in ("task_id", "dataset_source", "video_id", "video_source", "question"):
            _text(getattr(self, name), name)
        _digest(self.dataset_revision, 40, "dataset_revision")
        _digest(self.video_sha256, 64, "video_sha256")
        if not isinstance(self.options, tuple) or not 2 <= len(self.options) <= len(LABELS):
            raise ValueError(f"options must be an immutable tuple of two to {len(LABELS)} choices")
        for option in self.options:
            _text(option, "option")
        if type(self.allow_duplicate_options) is not bool:
            raise ValueError("allow_duplicate_options must be an explicit boolean")
        if not self.allow_duplicate_options and len({option.strip() for option in self.options}) != len(self.options):
            raise ValueError("option texts must be distinct")
        if self.level is not None and (type(self.level) is not int or self.level < 1):
            raise ValueError("level must be a positive integer or None")
        if self.group_id is not None:
            _text(self.group_id, "group_id")


@dataclass(frozen=True, slots=True)
class VerifierMetadata:
    """Server-side answer and provenance; never forwarded as model input."""

    task_id: str
    expected_answer: str
    option_count: int
    dataset_source: str
    dataset_revision: str
    video_id: str
    video_source: str
    video_sha256: str
    level: int | None = None
    group_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "dataset_source", "video_id", "video_source"):
            _text(getattr(self, name), name)
        _digest(self.dataset_revision, 40, "dataset_revision")
        _digest(self.video_sha256, 64, "video_sha256")
        if type(self.option_count) is not int or not 2 <= self.option_count <= len(LABELS):
            raise ValueError(f"option_count must be an integer from two to {len(LABELS)}")
        if not isinstance(self.expected_answer, str) or self.expected_answer not in tuple(LABELS[: self.option_count]):
            raise ValueError("expected_answer must be one available uppercase choice label")
        if self.level is not None and (type(self.level) is not int or self.level < 1):
            raise ValueError("level must be a positive integer or None")
        if self.group_id is not None:
            _text(self.group_id, "group_id")

    @classmethod
    def from_question(cls, question: VideoQuestion, *, expected_answer: str) -> "VerifierMetadata":
        """Bind a gold label to the exact question identity and choice range."""
        return cls(
            task_id=question.task_id,
            expected_answer=expected_answer,
            option_count=len(question.options),
            dataset_source=question.dataset_source,
            dataset_revision=question.dataset_revision,
            video_id=question.video_id,
            video_source=question.video_source,
            video_sha256=question.video_sha256,
            level=question.level,
            group_id=question.group_id,
        )


class InputText(TypedDict):
    """Model-visible Responses-compatible text part."""

    type: Literal["input_text"]
    text: str


class InputVideo(TypedDict):
    """A video already materialized by the media-preparation layer."""

    type: Literal["input_video"]
    video_url: str


class UserMessage(TypedDict):
    """One user turn; no verifier metadata or benchmark system instructions."""

    role: Literal["user"]
    content: list[InputText | InputVideo]


class ResponsesCreateParams(TypedDict):
    """The allowlisted model-facing part of a prepared task."""

    input: list[UserMessage]
    tools: NotRequired[list[dict[str, object]]]
    tool_choice: NotRequired[Literal["none"]]


class PreparedTask(TypedDict):
    """Gym row with explicitly separated model and verifier payloads."""

    task_id: str
    setting: Setting
    responses_create_params: ResponsesCreateParams
    verifier_metadata: dict[str, str | int | None]


def _question_text(question: VideoQuestion, *, video_path: str | None = None) -> str:
    text = "Watch the video and answer the question.\n"
    text += r"Put the selected option letter in \boxed{...} as your final answer."
    if video_path is not None:
        text += f"\n\nVideo: {video_path}"
    text += f"\n\nQuestion: {question.question}\n"
    return text + "\n".join(f"{label}. {option}" for label, option in zip(LABELS, question.options))


def prepare_single_turn(question: VideoQuestion, *, expected_answer: str, video_url: str) -> PreparedTask:
    """Attach prepared video once; sampling remains an explicit external step."""
    _text(video_url, "video_url")
    metadata = VerifierMetadata.from_question(question, expected_answer=expected_answer)
    return {
        "task_id": question.task_id,
        "setting": "single_turn",
        "responses_create_params": {
            "tools": [],
            "tool_choice": "none",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": video_url},
                        {"type": "input_text", "text": _question_text(question)},
                    ],
                }
            ],
        },
        "verifier_metadata": asdict(metadata),
    }


def prepare_agentic(question: VideoQuestion, *, expected_answer: str, video_path: str) -> PreparedTask:
    """Expose the bound sandbox path without source URLs, gold, or tool hints."""
    _text(video_path, "video_path")
    if any(character in video_path for character in "\r\n"):
        raise ValueError("video_path must be a single-line sandbox path")
    metadata = VerifierMetadata.from_question(question, expected_answer=expected_answer)
    return {
        "task_id": question.task_id,
        "setting": "agentic",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _question_text(question, video_path=video_path)},
                    ],
                }
            ]
        },
        "verifier_metadata": asdict(metadata),
    }
