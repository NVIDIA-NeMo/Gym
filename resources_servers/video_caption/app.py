# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caption recall/precision/F1, length, and degeneration judging through NeMo Gym."""

import asyncio
import json
import logging
import random
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, PrivateAttr
from tokenizers import Tokenizer

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming, NgFailureEvent
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.verification import JUDGE_FAILURES, JudgeError, call_judge, is_judge_failure
from resources_servers.video_caption.reward import (
    VideoCaptionRewardResult,
    VisualRewardType,
    compute_video_caption_reward,
    degeneration_response_format,
    empty_video_caption_reward,
    format_visual_judge_input,
    normalize_atomic_events,
    parse_degeneration_judgment,
    parse_visual_events,
    parse_visual_hits,
    visual_events_response_format,
    visual_hits_response_format,
)


logger = logging.getLogger(__name__)

# A row the verifier cannot score (blank reference events, or precision/F1 without a reference caption).
# The sample is masked rather than raised: a 5xx from /verify makes the environment server replay the
# whole episode, which costs a new policy rollout for a row that can never be scored.
UNSCORABLE_ROW = "video_caption:unscorable_row"

# Judge replies that fail the same way again: every 4xx except a timeout (408) or rate limit (429).
RETRYABLE_JUDGE_STATUSES = frozenset({408, 429})


# The wire form of the row's task fields. task_data.py declares the same fields for data validation with a
# stricter blank check; here a blank entry still reaches verify(), which masks the row instead of answering 422.
# tests/test_task_data.py pins the two declarations to the same fields and required set.
class VisualAtomicEvents(BaseModel):
    visual_events: list[str] = Field(min_length=1)


class CaptionVerifierMetadata(BaseModel):
    atomic_events: VisualAtomicEvents
    reference_caption: str | None = None


class VideoCaptionResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_chat_completion_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_prompt_fpath: str = "prompt_templates/visual_event_judge.txt"
    visual_reward_type: VisualRewardType = "recall"
    event_extraction_prompt_fpath: str = "prompt_templates/visual_event_extraction.txt"
    # Opt-in zero-total-reward gate; the recall prompt and schema stay the default.
    penalize_degeneration: bool = False
    degeneration_judge_prompt_fpath: str = "prompt_templates/degeneration_judge.txt"
    caption_tokenizer_path: str
    judge_endpoint_max_concurrency: PositiveInt = 256
    judge_max_retries: NonNegativeInt = 2
    judge_attempt_timeout_seconds: PositiveFloat = 1200.0
    judge_retry_base_delay_seconds: NonNegativeFloat = 1.0
    judge_retry_max_delay_seconds: NonNegativeFloat = 30.0
    min_caption_tokens: NonNegativeInt = 100
    max_caption_tokens: PositiveInt = 4000
    # A caption longer than this skips every judge and scores its visual half 0: scored, not masked. The judge's
    # context is finite, and a runaway caption is the policy's failure, not a judge outage. At least
    # max_caption_tokens, so no caption the length reward accepts goes unjudged.
    max_judged_caption_tokens: PositiveInt = 16000


class VideoCaptionVerifyRequest(BaseVerifyRequest):
    verifier_metadata: CaptionVerifierMetadata
    source_index: int | None = None


class VideoCaptionVerifyResponse(VideoCaptionVerifyRequest, BaseVerifyResponse):
    is_valid_caption: bool
    caption_length_tokens: int
    length_reward: float
    visual_hits: list[int]
    visual_recall_reward: float | None
    precision_visual_hits: list[int]
    visual_precision_reward: float | None
    visual_f1_reward: float | None
    visual_reward_type: VisualRewardType
    visual_reward: float
    predicted_visual_events: list[str] | None
    reward_before_degeneration: float
    degeneration_checked: bool
    degeneration_detected: bool
    degeneration_reward: float
    judge_attempts: int
    judge_response_text: str | None = None
    # True when the caption exceeded max_judged_caption_tokens: no judge ran and the visual half is 0.
    caption_over_judge_cap: bool
    event_extraction_attempts: int
    event_extraction_response_text: str | None = None
    precision_judge_attempts: int
    precision_judge_response_text: str | None = None
    degeneration_judge_attempts: int
    degeneration_judge_response_text: str | None = None


class VideoCaptionResourcesServer(SimpleResourcesServer):
    config: VideoCaptionResourcesServerConfig
    _tokenizer: Tokenizer = PrivateAttr()
    _judge_prompt: str = PrivateAttr()
    _event_extraction_prompt: str | None = PrivateAttr(default=None)
    _degeneration_judge_prompt: str | None = PrivateAttr(default=None)
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()
    _judge_converter: ChatCompletionsConverter = PrivateAttr(
        default_factory=lambda: ChatCompletionsConverter(return_token_id_information=False, uses_reasoning_parser=True)
    )

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.max_caption_tokens < self.config.min_caption_tokens:
            raise ValueError("max_caption_tokens must be at least min_caption_tokens")
        if self.config.max_judged_caption_tokens < self.config.max_caption_tokens:
            raise ValueError("max_judged_caption_tokens must be at least max_caption_tokens")
        self._tokenizer = Tokenizer.from_file(self.config.caption_tokenizer_path)
        self._judge_prompt = self._read_prompt(self.config.judge_prompt_fpath)
        if self.config.visual_reward_type in ("precision", "f1"):
            self._event_extraction_prompt = self._read_prompt(self.config.event_extraction_prompt_fpath)
        if self.config.penalize_degeneration:
            self._degeneration_judge_prompt = self._read_prompt(self.config.degeneration_judge_prompt_fpath)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)

    @staticmethod
    def _read_prompt(prompt_fpath: str) -> str:
        prompt_path = Path(prompt_fpath)
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).parent / prompt_path
        return prompt_path.read_text().rstrip()

    async def _judge(self, caption: str, atomic_events: dict[str, list[str]]) -> tuple[str, int]:
        return await self._judge_coverage(name="recall", caption=caption, atomic_events=atomic_events)

    async def _judge_coverage(
        self, *, name: str, caption: str, atomic_events: dict[str, list[str]]
    ) -> tuple[str, int]:
        event_count = len(atomic_events["visual_events"])
        return await self._request_judgment(
            name=name,
            prompt=f"{self._judge_prompt}\n\n{format_visual_judge_input(caption, atomic_events)}",
            response_format=visual_hits_response_format(event_count),
            validate_output=lambda text: parse_visual_hits(text, event_count),
        )

    async def _extract_prediction_events(self, caption: str) -> tuple[str, int]:
        assert self._event_extraction_prompt is not None
        visual_caption = json.dumps({"visual_caption": caption}, ensure_ascii=False)
        return await self._request_judgment(
            name="event extraction",
            prompt=f"{self._event_extraction_prompt}\n\n# Input Visual Caption\n{visual_caption}",
            response_format=visual_events_response_format(),
            validate_output=parse_visual_events,
        )

    async def _judge_degeneration(self, caption: str) -> tuple[str, int]:
        assert self._degeneration_judge_prompt is not None
        return await self._request_judgment(
            name="degeneration",
            prompt=f"{self._degeneration_judge_prompt}\n\nFINAL CAPTION:\n{caption}",
            response_format=degeneration_response_format(),
            validate_output=parse_degeneration_judgment,
        )

    async def _request_judgment(
        self, *, name: str, prompt: str, response_format: dict, validate_output: Callable[[str], Any]
    ) -> tuple[str, int]:
        params = self.config.judge_chat_completion_create_params.model_copy(deep=True)
        params.messages = [{"role": "user", "content": prompt}]
        params.response_format = response_format
        total_attempts = self.config.judge_max_retries + 1
        for attempt in range(total_attempts):
            try:
                # Queue waiting does not consume the attempt's serving deadline.
                async with self._judge_semaphore:
                    async with asyncio.timeout(self.config.judge_attempt_timeout_seconds):
                        # This loop owns the retry budget, so the shared client makes one attempt per
                        # iteration; otherwise its own three attempts nest inside ours.
                        response = await call_judge(
                            self.server_client,
                            server_name=self.config.judge_model_server.name,
                            url_path="/v1/chat/completions",
                            json=params,
                            response_model=NeMoGymChatCompletion,
                            max_attempts=1,
                        )
                if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
                    raise ValueError("Judge did not return one completed choice")
                # Gym wraps the judge's separate reasoning field in think tags.
                # Its converter separates reasoning from the final JSON answer.
                output = self._judge_converter.postprocess_chat_response(response.choices[0])
                text = "".join(
                    part.text
                    for item in output
                    if item.type == "message"
                    for part in item.content
                    if part.type == "output_text"
                ).strip()
                validate_output(text)
                return text, attempt + 1
            # JUDGE_FAILURES covers the transport failure call_judge raises as a judge_failed event.
            except (*JUDGE_FAILURES, TimeoutError, ValueError) as error:
                if isinstance(error, NgFailureEvent) and not is_judge_failure(error):
                    raise  # another Gym failure, such as a judge provider stop: not this loop's to retry
                if isinstance(error, NgFailureEvent) and not _is_retryable(error):
                    raise  # an auth or bad-request reply fails the same way again
                if attempt + 1 == total_attempts:
                    # judge_failsafe turns this into a masked zero: reward 0, mask_sample true,
                    # failure_kind judge_failed, and no caption length bonus.
                    raise JudgeError(
                        f"Caption {name} judge exhausted {total_attempts} attempts: {type(error).__name__}: {error}"
                    ) from error
                await asyncio.sleep(self._retry_delay(attempt))
        raise AssertionError("At least one judge attempt is required")

    def _retry_delay(self, attempt: int) -> float:
        """Full-jitter exponential backoff, so concurrent verifies do not retry a rate-limited judge in lockstep."""
        config = self.config
        return random.uniform(
            0, min(config.judge_retry_max_delay_seconds, config.judge_retry_base_delay_seconds * 2**attempt)
        )

    async def verify(self, body: VideoCaptionVerifyRequest) -> VideoCaptionVerifyResponse:
        # Gym's parsed output_text contains assistant message text, not reasoning items.
        caption = body.response.output_text.strip()
        caption_length = len(self._tokenizer.encode(caption, add_special_tokens=False).ids) if caption else 0
        reward_kwargs = {
            "visual_reward_type": self.config.visual_reward_type,
            "min_caption_tokens": self.config.min_caption_tokens,
            "max_caption_tokens": self.config.max_caption_tokens,
        }
        judge_text = None
        judge_attempts = 0
        predicted_events = None
        extraction_text = None
        extraction_attempts = 0
        precision_text = None
        precision_attempts = 0
        degeneration_judge_text = None
        degeneration_judge_attempts = 0
        try:
            atomic_events = normalize_atomic_events(body.verifier_metadata.model_dump())
        except ValueError as error:
            return self._unscorable(body, caption, caption_length, reward_kwargs, reason=str(error))
        reference_caption = body.verifier_metadata.reference_caption
        needs_precision = self.config.visual_reward_type in ("precision", "f1")
        if needs_precision and (reference_caption is None or not reference_caption.strip()):
            return self._unscorable(
                body,
                caption,
                caption_length,
                reward_kwargs,
                reason="Precision/F1 requires a nonempty verifier_metadata.reference_caption",
            )
        over_cap = caption_length > self.config.max_judged_caption_tokens
        if caption and over_cap:
            logger.info(
                "Caption of %d tokens exceeds max_judged_caption_tokens=%d (source_index=%s): scored 0 without a judge",
                caption_length,
                self.config.max_judged_caption_tokens,
                body.source_index,
            )
        if caption and not over_cap:
            if self.config.visual_reward_type in ("recall", "f1"):
                judge_text, judge_attempts = await self._judge(caption, atomic_events)
            if needs_precision:
                assert reference_caption is not None
                extraction_text, extraction_attempts = await self._extract_prediction_events(caption)
                predicted_events = parse_visual_events(extraction_text)
                if predicted_events:
                    precision_text, precision_attempts = await self._judge_coverage(
                        name="precision", caption=reference_caption, atomic_events={"visual_events": predicted_events}
                    )
            if self.config.penalize_degeneration:
                degeneration_judge_text, degeneration_judge_attempts = await self._judge_degeneration(caption)
            result = compute_video_caption_reward(
                caption,
                caption_length,
                atomic_events,
                judge_text,
                predicted_visual_events=predicted_events,
                precision_judge_output_text=precision_text,
                penalize_degeneration=self.config.penalize_degeneration,
                degeneration_judge_output_text=degeneration_judge_text,
                **reward_kwargs,
            )
        else:
            # Nothing to judge: an empty caption, or one over the judge cap (which the length window also rejects).
            result = empty_video_caption_reward(caption, caption_length, **reward_kwargs)
        return VideoCaptionVerifyResponse(
            **body.model_dump(),
            **asdict(result),
            judge_attempts=judge_attempts,
            judge_response_text=judge_text,
            caption_over_judge_cap=bool(caption) and over_cap,
            predicted_visual_events=predicted_events,
            event_extraction_attempts=extraction_attempts,
            event_extraction_response_text=extraction_text,
            precision_judge_attempts=precision_attempts,
            precision_judge_response_text=precision_text,
            degeneration_judge_attempts=degeneration_judge_attempts,
            degeneration_judge_response_text=degeneration_judge_text,
        )

    @staticmethod
    def _unscorable(
        body: VideoCaptionVerifyRequest,
        caption: str,
        caption_length: int,
        reward_kwargs: dict[str, Any],
        *,
        reason: str,
    ) -> VideoCaptionVerifyResponse:
        """A masked zero for a row the verifier cannot score; no judge is called and no length bonus is paid."""
        logger.warning("Masking an unscorable video_caption row (source_index=%s): %s", body.source_index, reason)
        zero: VideoCaptionRewardResult = empty_video_caption_reward(caption, caption_length, **reward_kwargs)
        # Keep the caption's length for diagnostics, but pay nothing for it: the sample is masked. The per-metric
        # scores stay null (uncomputed), so aggregates that ignore the mask do not count them as measured zeros.
        result = replace(
            zero,
            reward=0.0,
            length_reward=0.0,
            reward_before_degeneration=0.0,
            visual_recall_reward=None,
            visual_precision_reward=None,
            visual_f1_reward=None,
        )
        return VideoCaptionVerifyResponse(
            **body.model_dump(),
            **asdict(result),
            mask_sample=True,
            failure_kind=UNSCORABLE_ROW,
            failure_reason=reason,
            judge_attempts=0,
            caption_over_judge_cap=False,
            predicted_visual_events=None,
            event_extraction_attempts=0,
            precision_judge_attempts=0,
            degeneration_judge_attempts=0,
        )


def _is_retryable(failure: NgFailureEvent) -> bool:
    """Whether another attempt can succeed: no reply (a timeout or lost connection), a 408/429, or a 5xx."""
    status = failure.http_status
    return status is None or status in RETRYABLE_JUDGE_STATUSES or status >= 500


if __name__ == "__main__":
    VideoCaptionResourcesServer.run_webserver()
