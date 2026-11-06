# Copyright (c) 2025 Mohammad Tavakoli
# Originally MIT License: https://github.com/mohammadtavakoli78/BEAM/blob/b2da22eac88bb0874c64665f13457eb99835774a/LICENSE
# Adapted from https://github.com/mohammadtavakoli78/BEAM, src/beam/main.py (answer_generation,
# ConversationSummaryBuffer) at commit b2da22eac88bb0874c64665f13457eb99835774a.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of the original software
# and associated documentation files, to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, subject to the following conditions: the above copyright notice and this permission
# notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS
# PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. The original BEAM-authored portions of this
# file remain subject to the MIT license referenced above.

"""BEAM's long-conversation simulation, independent of Gym's server plumbing.

A port of ``answer_generation`` and ``ConversationSummaryBuffer`` from BEAM (see the header):
for each pre-generated user message an assistant model answers from a summary-plus-recent-turns
memory; a yes/no check decides whether the assistant asked the user something, in which case a
user model answers (at most twice); another check decides whether the user follows up (at most
twice). Model calls go through an async ``call`` function so the logic runs without servers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal


LOG = logging.getLogger(__name__)

CallRole = Literal["assistant", "user", "helper"]
# call(role, messages) -> text. "assistant" answers the conversation, "user" simulates the user,
# "helper" writes memory summaries and answers the yes/no checks.
Call = Callable[[CallRole, list[dict[str, str]]], Awaitable[str]]

ENGLISH_SUFFIX = "Please respond only in English."


class TokenCounter:
    """BEAM's estimator: tiktoken when an encoding loads, otherwise characters // 3."""

    def __init__(self, encoding: str | None = "cl100k_base") -> None:
        self.encoder = None
        if encoding:
            try:
                import tiktoken

                self.encoder = tiktoken.get_encoding(encoding)
            except Exception:  # missing package or no network for the encoding file
                LOG.warning("tiktoken encoding %s unavailable; estimating tokens as characters // 3", encoding)

    def __call__(self, text: Any) -> int:
        if not text:
            return 0
        text = str(text)
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(text, disallowed_special=()))
            except Exception:
                return len(text) // 3
        return len(text) // 3


def truncate_to_tokens(text: str, max_tokens: int, count: TokenCounter) -> str:
    """Longest prefix (plus a marker) within max_tokens, found by binary search."""
    if count(text) <= max_tokens:
        return text
    left, right, result = 0, len(text), text
    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid] + "... [truncated]"
        if count(truncated) <= max_tokens:
            result, left = truncated, mid
        else:
            right = mid - 1
    return result


def truncate_smart(text: str, max_tokens: int, count: TokenCounter) -> str:
    """Keep the beginning and end of text, about 40% of the budget each."""
    current = count(text)
    if current <= max_tokens:
        return text
    chars_each_side = int(int(max_tokens * 0.4) * (len(text) / current))
    return f"{text[:chars_each_side]}... [content truncated] ...{text[-chars_each_side:]}"


PRIORITY_ORDER = (
    "current_batch_messages",
    "ai_last_message",
    "current_plan",
    "topic",
    "theme",
    "previous_plans_summary",
    "previous_batches",
)


def prepare_prompt_within_budget(
    template: str, replacements: dict[str, str], count: TokenCounter, max_tokens: int
) -> str:
    """Fill <key> placeholders in priority order, shortening values that would exceed the budget."""
    prompt = template
    token_count = count(prompt)
    for key in PRIORITY_ORDER:
        if f"<{key}>" not in prompt or key not in replacements:
            continue
        value = replacements[key]
        if token_count + count(value) > max_tokens - 1000:
            available = max_tokens - token_count - 1000
            if key in ("previous_batches", "previous_plans_summary"):
                value = (
                    truncate_to_tokens(value, available, count)
                    if available > 100
                    else "[Content omitted due to length]"
                )
            elif key == "current_plan":
                value = (
                    truncate_to_tokens(value, available, count)
                    if available > 500
                    else f"[Plan summary: {value[:200]}...]"
                )
            elif key == "ai_last_message":
                value = (
                    truncate_smart(value, available, count)
                    if available > 300
                    else f"{value[:100]}... [truncated] ...{value[-100:]}"
                )
            elif key in ("topic", "theme"):
                if count(value) > 200:
                    value = value[:200] + "..."
            else:  # current_batch_messages
                value = (
                    "...[earlier messages truncated]...\n" + value[-(available * 3) :]
                    if available > 1000
                    else "[Messages too long - see summary]"
                )
        prompt = prompt.replace(f"<{key}>", value)
        token_count = count(prompt)
    return prompt


class SummaryMemory:
    """BEAM's ConversationSummaryBuffer: recent messages plus an LLM-written running summary,
    compressed whenever messages and summary together exceed max_tokens."""

    def __init__(
        self,
        call: Call,
        count: TokenCounter,
        max_tokens: int,
        recent_messages_count: int,
        max_summary_tokens: int,
    ) -> None:
        self.call = call
        self.count = count
        self.max_tokens = max_tokens
        self.recent_messages_count = recent_messages_count
        self.max_summary_tokens = max_summary_tokens
        self.messages: list[dict[str, str]] = []
        self.summary = ""
        # Token counts by text, so each message is encoded once rather than on every add().
        self._token_counts: dict[str, int] = {}

    def tokens(self, text: str) -> int:
        if text not in self._token_counts:
            self._token_counts[text] = self.count(text)
        return self._token_counts[text]

    def total_tokens(self) -> int:
        return sum(self.tokens(m["content"]) for m in self.messages) + self.tokens(self.summary)

    def snapshot(self) -> tuple[list[dict[str, str]], str]:
        return list(self.messages), self.summary

    def restore(self, state: tuple[list[dict[str, str]], str]) -> None:
        self.messages, self.summary = list(state[0]), state[1]

    async def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        for _ in range(10):
            if self.total_tokens() <= self.max_tokens:
                return
            if self.messages:
                if not await self._compress_messages():
                    # Only the newest message is left and it alone exceeds the budget: shrink the
                    # summary once instead of looping on a budget that cannot be met.
                    if self.summary:
                        await self._compress_summary_further()
                    return
            elif self.summary:
                await self._compress_summary_further()
            else:
                return
        LOG.warning("could not reduce memory below %d tokens after 10 attempts", self.max_tokens)

    async def _compress_messages(self) -> bool:
        """Summarize all but the recent messages that fit; False if there was nothing to summarize.
        The newest message is always kept verbatim (when any are kept), even if it alone exceeds
        the budget, so the model always sees the turn it is answering."""
        available = self.max_tokens - self.max_summary_tokens - 400
        kept: list[dict[str, str]] = []
        summarize: list[dict[str, str]] = []
        cumulative = 0
        for message in reversed(self.messages):
            tokens = self.tokens(message["content"])
            newest = not kept and not summarize and self.recent_messages_count > 0
            if newest or (cumulative + tokens <= available and len(kept) < self.recent_messages_count):
                kept.insert(0, message)
                cumulative += tokens
            else:
                summarize.insert(0, message)
        if summarize:
            new_summary = await self._summarize(summarize)
            if self.summary:
                combined = f"{self.summary}\n\nAdditional context: {new_summary}"
                if self.count(combined) > self.max_summary_tokens:
                    self.summary = await self._meta_summary(self.summary, new_summary)
                else:
                    self.summary = combined
            else:
                self.summary = new_summary
            if self.tokens(self.summary) > self.max_summary_tokens:
                self.summary = self._truncate_summary(self.summary)
        self.messages = kept
        self._token_counts = {
            text: self._token_counts[text]
            for text in [m["content"] for m in kept] + [self.summary]
            if text in self._token_counts
        }
        return bool(summarize)

    async def _ask(self, prompt: str) -> str:
        return (await self.call("helper", [{"role": "user", "content": prompt}])).strip()

    async def _summarize(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            return ""
        conversation = "".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n\n" for m in messages
        )
        max_chars = self.max_summary_tokens * 3
        prompt = f"""Summarize the following conversation in NO MORE than {max_chars} characters. Focus on key decisions, important information, and progress made. Be concise but preserve important context:
        {conversation}
        Summary (keep under {max_chars} characters):"""
        try:
            summary = await self._ask(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOG.warning("summary failed: %s", error)
            return self._truncate_summary(f"Previous conversation covered {len(messages)} exchanges.")
        return self._truncate_summary(summary) if self.count(summary) > self.max_summary_tokens else summary

    async def _meta_summary(self, old_summary: str, new_summary: str) -> str:
        max_chars = self.max_summary_tokens * 3
        prompt = f"""Combine these two summaries into one concise summary of NO MORE than {max_chars} characters:
        Previous Summary: {old_summary}
        Recent Summary: {new_summary}
        Combined Summary (under {max_chars} characters):"""
        try:
            summary = await self._ask(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOG.warning("meta summary failed: %s", error)
            return self._truncate_summary(new_summary)
        return self._truncate_summary(summary) if self.count(summary) > self.max_summary_tokens else summary

    async def _compress_summary_further(self) -> None:
        target = self.max_summary_tokens // 2
        prompt = f"""Compress this summary to NO MORE than {target * 3} characters while keeping the most important information:
        {self.summary}
        Compressed summary (under {target * 3} characters):"""
        try:
            compressed = await self._ask(prompt)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.summary = self._truncate_summary(self.summary, target)
            return
        if self.count(compressed) < self.tokens(self.summary):
            self.summary = compressed
        else:
            self.summary = self._truncate_summary(self.summary, target)

    def _truncate_summary(self, summary: str, max_tokens: int | None = None) -> str:
        target = max_tokens or self.max_summary_tokens
        if self.count(summary) <= target:
            return summary
        char_limit = target * 3
        return summary[: char_limit - 3] + "..." if len(summary) > char_limit else summary

    def messages_for_llm(self, system_prompt: str | None = None) -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.summary:
            messages.append({"role": "system", "content": f"Previous conversation summary: {self.summary}"})
        return messages + list(self.messages)

    def history_text(self) -> str:
        if not self.messages:
            return ""
        text = f"Previous Context: {self.summary}\n\n" if self.summary else ""
        text += "Recent Conversations:\n"
        for message in self.messages:
            text += f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}\n\n"
        return text.strip()


@dataclass
class Prompts:
    """BEAM templates (``src/prompts.py``) with their <placeholder> slots."""

    assistant_system: str  # ai_assistant_llm_template
    check_include_question: str  # check_include_question_template
    check_need_followup: str  # check_need_followup_template
    user_answer_question: str  # user_llm_answer_question_template
    user_followup_question: str  # user_llm_ask_followup_question_template


@dataclass
class Settings:
    """BEAM's answer_generation constants."""

    assistant_context_tokens: int = 28000
    assistant_recent_messages: int = 10
    assistant_summary_tokens: int = 1000
    # The assistant memory gets assistant_context_tokens minus the system prompt, which grows with the
    # previous plans; this floor keeps late batches of the 500K/1M sizes from a budget near zero.
    # BEAM's 100K settings stay above it, so they behave as in BEAM.
    assistant_min_memory_tokens: int = 8000
    user_memory_tokens: int = 19000
    user_recent_messages: int = 10
    user_summary_tokens: int = 2000
    user_prompt_max_tokens: int = 27000
    check_max_tokens: int = 25000
    max_clarifications: int = 2
    max_followups: int = 2
    batch_shortfall_allowance: int = 10


@dataclass
class Batch:
    messages: list[str]
    current_plan: str
    previous_plans: str
    time_anchor: str | None = None


@dataclass
class SimulationResult:
    chat: list[list[dict[str, Any]]] = field(default_factory=list)
    batches_requested: int = 0
    skipped_questions: list[dict[str, Any]] = field(default_factory=list)
    # Questions whose main answer was recorded but whose clarification or follow-up failed.
    interrupted_turns: list[dict[str, Any]] = field(default_factory=list)
    stopped_at_batch: int | None = None


def describe(error: BaseException) -> str:
    """`Type: message`, even for exceptions whose own __str__ fails."""
    try:
        return f"{type(error).__name__}: {error}"
    except Exception:
        return type(error).__name__


def bullet_index(question: str) -> str:
    """BEAM's `batch,bullet` index from a message's `->->` tag."""
    if "->->" not in question:
        return "N/A,N/A"
    index = question.split("->->")[1].strip()
    if "," in index:
        batch, bullet = index.split(",")[0].strip(), index.split(",")[1].strip()
        if bullet == "":
            index = batch + "," + "N/A"
    return index


async def includes_question(
    call: Call, prompts: Prompts, response: str, count: TokenCounter, settings: Settings
) -> bool:
    if count(response) > settings.check_max_tokens:
        response = truncate_to_tokens(response, settings.check_max_tokens, count)
    prompt = prompts.check_include_question.replace("<assistant_response>", response)
    return "yes" in (await call("helper", [{"role": "user", "content": prompt}])).strip().lower()


async def needs_followup(
    call: Call,
    prompts: Prompts,
    response: str,
    history: list[dict[str, str]],
    topic: str,
    theme: str,
    count: TokenCounter,
    settings: Settings,
) -> bool:
    """BEAM's check_need_followup_safe, including its history formatting (each history entry is
    rendered as a Python dict and labelled USER/ASSISTANT by position)."""
    max_tokens = settings.check_max_tokens
    max_response_tokens = (max_tokens - 1000) // 2
    if count(response) > max_response_tokens:
        response = truncate_to_tokens(response, max_response_tokens, count)
    available = max_tokens - (count(response) + count(f"{topic}{theme}")) - 1000
    if available <= 0:
        response, history = response[:1000] + "...[truncated]", []
        topic, theme = topic[:100], theme[:100]
    elif history and sum(count(m.get("content", "")) for m in history) > available:
        kept: list[dict[str, str]] = []
        used = 0
        for message in reversed(history):
            tokens = count(message.get("content", ""))
            if used + tokens < available:
                kept.insert(0, message)
                used += tokens
            else:
                if available - used > 100:
                    kept.insert(
                        0,
                        {
                            **message,
                            "content": truncate_to_tokens(message.get("content", ""), available - used, count),
                        },
                    )
                break
        history = kept
    formatted = "\n".join(f"USER: {m}" if i % 2 == 0 else f"ASSISTANT: {m}" for i, m in enumerate(history))
    prompt = (
        prompts.check_need_followup.replace("<topic>", topic)
        .replace("<theme>", theme)
        .replace("<formatted_history>", formatted)
        .replace("<assistant_response>", response)
    )
    return "yes" in (await call("helper", [{"role": "user", "content": prompt}])).strip().lower()


async def simulate(
    call: Call,
    prompts: Prompts,
    batches: list[Batch],
    topic: str,
    theme: str,
    *,
    previous_plans_summary: str = "",
    settings: Settings | None = None,
    count: TokenCounter | None = None,
) -> SimulationResult:
    """Play the whole conversation batch by batch, as BEAM's answer_generation does. A question
    whose turn raises is skipped; a batch that ends up more than `batch_shortfall_allowance`
    messages short of two per question stops the conversation (BEAM keeps earlier batches)."""
    settings = settings or Settings()
    count = count or TokenCounter()
    result = SimulationResult(batches_requested=len(batches))
    message_id = 0

    for batch_index, batch in enumerate(batches):
        if batch_index == 0:
            system_prompt = (
                prompts.assistant_system.replace("<topic>", "N/A")
                .replace("<theme>", "N/A")
                .replace("<previous_plans_summary>", previous_plans_summary)
                .replace("<previous_batches>", "N/A")
            )
        else:
            system_prompt = (
                prompts.assistant_system.replace("<topic>", topic)
                .replace("<theme>", theme)
                .replace("<previous_plans_summary>", previous_plans_summary)
                .replace("<previous_batches>", batch.previous_plans)
            )
        assistant_memory = SummaryMemory(
            call,
            count,
            max(settings.assistant_context_tokens - count(system_prompt), settings.assistant_min_memory_tokens),
            settings.assistant_recent_messages,
            settings.assistant_summary_tokens,
        )
        user_memory = SummaryMemory(
            call, count, settings.user_memory_tokens, settings.user_recent_messages, settings.user_summary_tokens
        )
        history: list[dict[str, Any]] = []

        async def assistant_turn() -> str:
            return await call("assistant", assistant_memory.messages_for_llm(system_prompt))

        async def record_assistant(text: str) -> None:
            # Clarification and follow-up replies: user memory first, as in BEAM.
            nonlocal message_id
            await user_memory.add("assistant", text)
            await assistant_memory.add("assistant", text)
            history.append({"role": "assistant", "id": message_id, "content": text})
            message_id += 1

        async def user_turn(template: str, last_answer: str, question_type: str) -> None:
            nonlocal message_id
            replacements = {
                "current_batch_messages": user_memory.history_text(),
                "topic": topic,
                "theme": theme,
                "previous_plans_summary": previous_plans_summary,
                "previous_batches": batch.previous_plans,
                "current_plan": batch.current_plan,
                "ai_last_message": last_answer,
            }
            prompt = prepare_prompt_within_budget(template, replacements, count, settings.user_prompt_max_tokens)
            text = await call("user", [{"role": "user", "content": prompt}])
            await user_memory.add("user", text)
            await assistant_memory.add("user", text + ENGLISH_SUFFIX)
            history.append({"role": "user", "id": message_id, "question_type": question_type, "content": text})
            message_id += 1

        def checkpoint() -> tuple[Any, ...]:
            return assistant_memory.snapshot(), user_memory.snapshot(), len(history), message_id

        def rollback(state: tuple[Any, ...]) -> None:
            nonlocal message_id
            assistant_memory.restore(state[0])
            user_memory.restore(state[1])
            del history[state[2] :]
            message_id = state[3]

        for question_index, question in enumerate(batch.messages):
            # A failed turn is rolled back to the last complete exchange, so memories and the
            # transcript never hold a question without its answer (unlike BEAM, which kept them).
            start, committed = checkpoint(), None
            try:
                index = bullet_index(question)
                await user_memory.add("user", question)
                await assistant_memory.add("user", question + ENGLISH_SUFFIX)
                answer = await assistant_turn()
                await assistant_memory.add("assistant", answer)
                await user_memory.add("assistant", answer)
                entry: dict[str, Any] = {"role": "user", "id": message_id}
                if question_index == 0:
                    entry["time_anchor"] = batch.time_anchor
                entry.update(index=index, question_type="main_question", content=question)
                history.append(entry)
                message_id += 1
                history.append({"role": "assistant", "id": message_id, "content": answer})
                message_id += 1
                committed = checkpoint()

                asked = await includes_question(call, prompts, answer, count, settings)
                rounds = 0
                while asked and rounds < settings.max_clarifications:
                    await user_turn(prompts.user_answer_question, answer, "answer_ai_question")
                    answer = await assistant_turn()
                    await record_assistant(answer)
                    committed = checkpoint()
                    asked = await includes_question(call, prompts, answer, count, settings)
                    rounds += 1

                follow = await needs_followup(
                    call, prompts, answer, user_memory.messages_for_llm(), topic, theme, count, settings
                )
                rounds = 0
                while follow and rounds < settings.max_followups:
                    await user_turn(prompts.user_followup_question, answer, "followup_question")
                    answer = await assistant_turn()
                    await record_assistant(answer)
                    committed = checkpoint()
                    rounds += 1
                    follow = await needs_followup(
                        call, prompts, answer, user_memory.messages_for_llm(), topic, theme, count, settings
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                reason = describe(error)
                record = {"batch": batch_index, "question": question_index, "error": reason}
                if committed is None:
                    rollback(start)
                    LOG.warning("batch %d question %d skipped: %s", batch_index, question_index, reason)
                    result.skipped_questions.append(record)
                else:
                    rollback(committed)
                    LOG.warning("batch %d question %d cut short: %s", batch_index, question_index, reason)
                    result.interrupted_turns.append(record)

        needed = len(batch.messages) * 2 - settings.batch_shortfall_allowance
        if history and len(history) >= needed:
            result.chat.append(history)
        else:
            result.stopped_at_batch = batch_index
            break
    return result
