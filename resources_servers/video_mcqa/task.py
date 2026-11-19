# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-independent separation of model inputs, gold answers and provenance."""

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, NotRequired, TypedDict, get_args


Setting = Literal["single_turn", "agentic"]
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# How a single-turn question asks for its answer. boxed (the default) asks for \boxed{letter}, which the rule-based
# answer policies read. Each other style asks for the bare letter in the words of that benchmark's reference
# evaluation (sources in the video_mcqa README); answers to it are read with answer_policy llm_extraction_v1.
PromptStyle = Literal["boxed", "videomme_v2", "videomme_v1", "lvbench"]


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
    """Server-side answer, question and provenance; never forwarded as model input.

    ``question`` and ``options`` are the question as asked: its text and its option texts in label order, without
    labels. A judge model that reads an answer (answer policy llm_extraction_v1) gets the question from them, never
    from the model's request. Prepared rows carry both. Rule-scored results saved before rows carried them have
    neither; the rule policies and aggregation do not read them.
    """

    task_id: str
    expected_answer: str
    option_count: int
    dataset_source: str
    dataset_revision: str
    video_id: str
    video_source: str
    video_sha256: str
    question: str | None = None
    options: tuple[str, ...] | None = None
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
        if (self.question is None) != (self.options is None):
            raise ValueError("question and options are given together or not at all")
        if self.question is not None:
            _text(self.question, "question")
            if not isinstance(self.options, (tuple, list)) or len(self.options) != self.option_count:
                raise ValueError("options must hold one text per choice, in label order")
            for option in self.options:
                _text(option, "option")
            # A row read back from JSON holds a list.
            object.__setattr__(self, "options", tuple(self.options))
        if self.level is not None and (type(self.level) is not int or self.level < 1):
            raise ValueError("level must be a positive integer or None")
        if self.group_id is not None:
            _text(self.group_id, "group_id")

    @classmethod
    def from_question(cls, question: VideoQuestion, *, expected_answer: str) -> "VerifierMetadata":
        """Bind a gold label to the exact question, its identity and choice range."""
        return cls(
            task_id=question.task_id,
            expected_answer=expected_answer,
            option_count=len(question.options),
            dataset_source=question.dataset_source,
            dataset_revision=question.dataset_revision,
            video_id=question.video_id,
            video_source=question.video_source,
            video_sha256=question.video_sha256,
            question=question.question,
            options=question.options,
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


class InputImage(TypedDict):
    """One frame of a video sent as frames, already materialized by the media-preparation layer."""

    type: Literal["input_image"]
    image_url: str
    detail: Literal["auto", "low", "high"]


class UserMessage(TypedDict):
    """One user turn; no verifier metadata or benchmark system instructions."""

    role: Literal["user"]
    content: list[InputText | InputVideo | InputImage]


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
    verifier_metadata: dict[str, str | int | list[str] | None]


def _verifier_metadata(question: VideoQuestion, *, expected_answer: str) -> dict[str, str | int | list[str] | None]:
    """The row's verifier metadata, JSON-shaped as a row reads back: its options are a list."""
    metadata = asdict(VerifierMetadata.from_question(question, expected_answer=expected_answer))
    return {**metadata, "options": list(metadata["options"])}


def _question_text(question: VideoQuestion, *, video_path: str | None = None) -> str:
    text = "Watch the video and answer the question.\n"
    text += r"Put the selected option letter in \boxed{...} as your final answer."
    if video_path is not None:
        text += f"\n\nVideo: {video_path}"
    text += f"\n\nQuestion: {question.question}\n"
    return text + "\n".join(f"{label}. {option}" for label, option in zip(LABELS, question.options))


def _letter_question_text(question: VideoQuestion, prompt: PromptStyle) -> str:
    options = [f"{label}. {option}" for label, option in zip(LABELS, question.options)]
    if prompt == "videomme_v2":
        letters = LABELS[: len(question.options)]
        return "\n".join(
            (
                question.question,
                *options,
                f"Respond with only the letter ({', '.join(letters[:-1])}, or {letters[-1]}) of the correct option.",
                "Answer: ",
            )
        )
    if prompt == "videomme_v1":
        if len(question.options) != 4:
            raise ValueError("The videomme_v1 prompt names exactly four options")
        instruction = (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option."
        )
        return "\n".join((instruction, question.question, *options, "The best answer is:"))
    options = [f"({label}) {option}" for label, option in zip(LABELS, question.options)]
    return "\n".join((question.question, *options, "Answer with the option's letter from the given choices directly."))


def video_parts(
    *, video_url: str | None = None, frames: Sequence[tuple[str, str]] | None = None
) -> list[InputText | InputVideo | InputImage]:
    """A prepared video as user-turn parts: one ``input_video`` part, or the video as frames.

    ``frames`` are ``(label, image_url)`` pairs in playback order. Each frame is its label as an ``input_text`` part
    followed by the frame as an ``input_image`` part. A label holds text only, such as the frame's time; the model's
    chat template places its image token at the image part. Give exactly one of ``video_url`` and ``frames``.
    """
    if (video_url is None) == (frames is None):
        raise ValueError("Give the video either as video_url or as frames")
    if video_url is not None:
        _text(video_url, "video_url")
        return [{"type": "input_video", "video_url": video_url}]
    if isinstance(frames, (str, bytes)) or not frames:
        raise ValueError("frames must be a nonempty sequence of (label, image_url) pairs")
    parts: list[InputText | InputVideo | InputImage] = []
    for label, image_url in frames:
        _text(label, "frame label")
        _text(image_url, "frame image_url")
        parts.append({"type": "input_text", "text": label})
        parts.append({"type": "input_image", "image_url": image_url, "detail": "auto"})
    return parts


def prepare_single_turn(
    question: VideoQuestion,
    *,
    expected_answer: str,
    video_url: str | None = None,
    frames: Sequence[tuple[str, str]] | None = None,
    prompt: PromptStyle = "boxed",
) -> PreparedTask:
    """Attach the prepared video once, as one video part or as labelled frames (``video_parts``), then the question.

    Sampling remains an explicit external step.
    """
    if prompt not in get_args(PromptStyle):
        raise ValueError(f"Unknown prompt style {prompt!r}")
    metadata = _verifier_metadata(question, expected_answer=expected_answer)
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
                        *video_parts(video_url=video_url, frames=frames),
                        {
                            "type": "input_text",
                            "text": _question_text(question)
                            if prompt == "boxed"
                            else _letter_question_text(question, prompt),
                        },
                    ],
                }
            ],
        },
        "verifier_metadata": metadata,
    }


def prepare_agentic(
    question: VideoQuestion, *, expected_answer: str, video_path: str, prompt: PromptStyle = "boxed"
) -> PreparedTask:
    """Expose the bound sandbox path without source URLs, gold, or tool hints.

    A letter prompt is the benchmark's own single-turn text after a first line that names the video,
    ``Video: <video_path>``; its answers need an answer policy that reads a bare letter.
    """
    _text(video_path, "video_path")
    if any(character in video_path for character in "\r\n"):
        raise ValueError("video_path must be a single-line sandbox path")
    if prompt not in get_args(PromptStyle):
        raise ValueError(f"Unknown prompt style {prompt!r}")
    text = (
        _question_text(question, video_path=video_path)
        if prompt == "boxed"
        else f"Video: {video_path}\n{_letter_question_text(question, prompt)}"
    )
    metadata = _verifier_metadata(question, expected_answer=expected_answer)
    return {
        "task_id": question.task_id,
        "setting": "agentic",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                    ],
                }
            ]
        },
        "verifier_metadata": metadata,
    }
