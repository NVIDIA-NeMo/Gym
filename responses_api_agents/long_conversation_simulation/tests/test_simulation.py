# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest

from responses_api_agents.long_conversation_simulation.simulation import (
    ENGLISH_SUFFIX,
    Batch,
    Prompts,
    Settings,
    SummaryMemory,
    TokenCounter,
    bullet_index,
    needs_followup,
    prepare_prompt_within_budget,
    simulate,
    truncate_smart,
    truncate_to_tokens,
)


COUNT = TokenCounter(None)  # characters // 3, no tiktoken download in tests

PROMPTS = Prompts(
    assistant_system="SYS topic=<topic> theme=<theme> summary=<previous_plans_summary> previous=<previous_batches>",
    check_include_question="ASKED? <assistant_response>",
    check_need_followup="FOLLOW? <topic>|<theme>|<formatted_history>|<assistant_response>",
    user_answer_question="ANSWER <ai_last_message> plan=<current_plan>",
    user_followup_question="FOLLOWUP <ai_last_message> history=<current_batch_messages>",
)


class Script:
    """A fake model: answers by role and prompt kind, recording every call."""

    def __init__(self, asks: int = 0, follows: int = 0, fail_on: str | None = None) -> None:
        self.asks, self.follows, self.fail_on = asks, follows, fail_on
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self.answers = 0

    async def __call__(self, role, messages):
        self.calls.append((role, messages))
        text = messages[-1]["content"]
        if self.fail_on and self.fail_on in text and role == "assistant":
            raise RuntimeError("model down")
        if role == "assistant":
            self.answers += 1
            return f"answer {self.answers}"
        if role == "user":
            return "user says " + text.split()[0].lower()
        if text.startswith("ASKED?"):
            self.asks -= 1
            return "Yes" if self.asks >= 0 else "No"
        if text.startswith("FOLLOW?"):
            self.follows -= 1
            return "yes" if self.follows >= 0 else "no"
        return "a summary"


def roles(script: Script) -> list[str]:
    return [role for role, _ in script.calls]


def batch(*messages: str, anchor: str | None = "March-03-2025") -> Batch:
    return Batch(messages=list(messages), current_plan="PLAN", previous_plans="EARLIER", time_anchor=anchor)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("q ->-> 2,7", "2,7"),
        ("q ->-> 2,", "2,N/A"),
        ("q ->-> 3", "3"),
        ("no tag", "N/A,N/A"),
    ],
)
def test_bullet_index(question: str, expected: str) -> None:
    assert bullet_index(question) == expected


def test_token_counter_falls_back_to_characters() -> None:
    assert COUNT("") == 0
    assert COUNT("x" * 30) == 10
    assert TokenCounter("no-such-encoding")("x" * 9) == 3


def test_token_counter_uses_encoder_and_falls_back_on_encode_errors() -> None:
    class Encoder:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def encode(self, text, disallowed_special=()):
            if self.fail:
                raise ValueError("bad text")
            return text.split()

    counter = TokenCounter(None)
    counter.encoder = Encoder(fail=False)
    assert counter("one two three") == 3
    counter.encoder = Encoder(fail=True)
    assert counter("x" * 9) == 3


def test_truncation_helpers() -> None:
    text = "abcdefghij" * 30
    assert truncate_to_tokens("short", 100, COUNT) == "short"
    cut = truncate_to_tokens(text, 20, COUNT)
    assert cut.endswith("... [truncated]") and COUNT(cut) <= 20
    assert truncate_smart("short", 100, COUNT) == "short"
    smart = truncate_smart(text, 20, COUNT)
    assert "[content truncated]" in smart and smart.startswith("abc") and smart.endswith("hij")


def test_prepare_prompt_fills_and_shortens_by_priority() -> None:
    template = "<current_batch_messages>|<ai_last_message>|<current_plan>|<topic>|<theme>|<previous_plans_summary>|<previous_batches>"
    fits = prepare_prompt_within_budget(template, {key: key.upper() for key in ("topic", "theme")}, COUNT, 10000)
    assert "TOPIC" in fits and "<current_plan>" in fits  # missing keys are left as they are
    long = {
        "current_batch_messages": "m" * 15000,  # 5,000 tokens at characters // 3
        "ai_last_message": "a" * 9000,
        "current_plan": "p" * 9000,
        "topic": "t" * 900,
        "theme": "h" * 900,
        "previous_plans_summary": "s" * 9000,
        "previous_batches": "b" * 9000,
    }
    roomy = prepare_prompt_within_budget(template, long, COUNT, 5000)
    assert roomy.startswith("...[earlier messages truncated]...")
    assert "... [truncated] ..." in roomy and "t" * 200 + "..." in roomy
    tight = prepare_prompt_within_budget(template, long, COUNT, 1200)
    assert "[Messages too long - see summary]" in tight
    assert "[Content omitted due to length]" in tight and "[Plan summary: " in tight
    # With room left, the last assistant message keeps its beginning and end.
    smart = prepare_prompt_within_budget("<ai_last_message>", {"ai_last_message": "a" * 9000}, COUNT, 3500)
    assert "[content truncated]" in smart


def test_prepare_prompt_truncates_plan_and_previous_batches_when_room_remains() -> None:
    template = "<current_plan>|<previous_batches>|<previous_plans_summary>"
    out = prepare_prompt_within_budget(
        template,
        {"current_plan": "p" * 6000, "previous_batches": "b" * 6000, "previous_plans_summary": "s"},
        COUNT,
        3200,
    )
    assert out.count("... [truncated]") >= 1 and COUNT(out) <= 3200


@pytest.mark.asyncio
async def test_memory_keeps_recent_messages_and_summarizes_the_rest() -> None:
    script = Script()
    memory = SummaryMemory(script, COUNT, max_tokens=700, recent_messages_count=2, max_summary_tokens=100)
    for index in range(4):
        await memory.add("user" if index % 2 == 0 else "assistant", f"{index}" * 600)
    assert len(memory.messages) <= 2 and memory.summary
    assert all(role == "helper" for role in roles(script))
    formatted = memory.messages_for_llm("SYSTEM")
    assert formatted[0] == {"role": "system", "content": "SYSTEM"}
    assert formatted[1]["content"].startswith("Previous conversation summary: ")
    assert memory.history_text().startswith("Previous Context: ")
    assert SummaryMemory(script, COUNT, 10, 1, 5).history_text() == ""


@pytest.mark.asyncio
async def test_memory_meta_summary_and_fallbacks() -> None:
    async def failing(role, messages):
        raise RuntimeError("down")

    memory = SummaryMemory(failing, COUNT, max_tokens=200, recent_messages_count=1, max_summary_tokens=20)
    memory.summary = "old " * 40
    await memory.add("user", "u" * 400)
    await memory.add("assistant", "a" * 400)
    assert memory.summary  # fallbacks keep a (truncated) summary without the model
    assert await memory._summarize([]) == ""

    long_summary = Script()

    async def wordy(role, messages):
        await long_summary(role, messages)
        return "w" * 900

    wordy_memory = SummaryMemory(wordy, COUNT, max_tokens=300, recent_messages_count=1, max_summary_tokens=30)
    wordy_memory.summary = "previous summary " * 4
    await wordy_memory.add("user", "x" * 600)
    await wordy_memory.add("assistant", "y" * 600)
    assert COUNT(wordy_memory.summary) <= 30
    assert any("Combine these two summaries" in m[-1]["content"] for _, m in long_summary.calls)


@pytest.mark.asyncio
async def test_memory_compresses_an_oversized_summary() -> None:
    script = Script()
    memory = SummaryMemory(script, COUNT, max_tokens=50, recent_messages_count=1, max_summary_tokens=40)
    memory.summary = "s" * 600
    await memory._compress_summary_further()
    assert memory.summary == "a summary"

    async def same(role, messages):
        return "s" * 900

    stubborn = SummaryMemory(same, COUNT, max_tokens=50, recent_messages_count=1, max_summary_tokens=40)
    stubborn.summary = "s" * 600
    await stubborn._compress_summary_further()
    assert stubborn.summary.endswith("...")

    async def down(role, messages):
        raise RuntimeError("down")

    broken = SummaryMemory(down, COUNT, max_tokens=50, recent_messages_count=1, max_summary_tokens=40)
    broken.summary = "s" * 600
    await broken._compress_summary_further()
    assert broken.summary.endswith("...")

    # With no messages left, add() falls back to compressing the summary itself.
    empty = SummaryMemory(script, COUNT, max_tokens=10, recent_messages_count=0, max_summary_tokens=4)
    empty.summary = "z" * 300
    await empty.add("user", "hi")
    assert COUNT(empty.summary) <= 10 or empty.summary == "a summary"


@pytest.mark.asyncio
async def test_memory_gives_up_after_ten_rounds() -> None:
    async def bloated(role, messages):
        return "b" * 3000

    memory = SummaryMemory(bloated, COUNT, max_tokens=5, recent_messages_count=0, max_summary_tokens=1000)
    await memory.add("user", "x" * 300)
    assert memory.messages == []


@pytest.mark.asyncio
async def test_memory_with_nothing_left_to_compress_stops() -> None:
    async def silent(role, messages):
        return ""

    # Every message moves into an empty summary, leaving nothing to compress.
    memory = SummaryMemory(silent, COUNT, max_tokens=-1, recent_messages_count=0, max_summary_tokens=10)
    await memory.add("user", "hello")
    assert memory.messages == [] and memory.summary == ""


@pytest.mark.asyncio
async def test_simulate_records_a_plain_conversation_like_beam() -> None:
    script = Script()
    result = await simulate(
        script,
        PROMPTS,
        [batch("q1 ->-> 1,1", "q2 ->-> 1,2")],
        "Topic",
        "Theme",
        settings=Settings(batch_shortfall_allowance=0),
        count=COUNT,
    )
    assert result.stopped_at_batch is None and len(result.chat) == 1
    history = result.chat[0]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert [m["id"] for m in history] == [0, 1, 2, 3]
    assert history[0] == {
        "role": "user",
        "id": 0,
        "time_anchor": "March-03-2025",
        "index": "1,1",
        "question_type": "main_question",
        "content": "q1 ->-> 1,1",
    }
    assert "time_anchor" not in history[2]
    # First batch: BEAM fills the assistant system prompt with N/A.
    assistant_messages = next(m for r, m in script.calls if r == "assistant")
    assert assistant_messages[0]["content"] == "SYS topic=N/A theme=N/A summary= previous=N/A"
    assert assistant_messages[-1]["content"] == "q1 ->-> 1,1" + ENGLISH_SUFFIX


@pytest.mark.asyncio
async def test_simulate_clarifications_and_followups_are_capped_at_two() -> None:
    script = Script(asks=5, follows=5)
    result = await simulate(script, PROMPTS, [batch("q ->-> 1,1")], "T", "H", count=COUNT)
    types = [m.get("question_type") for m in result.chat[0] if m["role"] == "user"]
    assert types == [
        "main_question",
        "answer_ai_question",
        "answer_ai_question",
        "followup_question",
        "followup_question",
    ]
    user_prompts = [m[-1]["content"] for r, m in script.calls if r == "user"]
    assert user_prompts[0].startswith("ANSWER answer 1 plan=PLAN")
    assert user_prompts[2].startswith("FOLLOWUP ")


@pytest.mark.asyncio
async def test_later_batches_use_topic_and_previous_plans() -> None:
    script = Script()
    await simulate(
        script,
        PROMPTS,
        [batch("a ->-> 1,1"), batch("b ->-> 2,1", anchor=None)],
        "T",
        "H",
        previous_plans_summary="S",
        count=COUNT,
    )
    systems = [m[0]["content"] for r, m in script.calls if r == "assistant"]
    assert systems == [
        "SYS topic=N/A theme=N/A summary=S previous=N/A",
        "SYS topic=T theme=H summary=S previous=EARLIER",
    ]


@pytest.mark.asyncio
async def test_failed_question_is_skipped_and_short_batch_stops_the_conversation() -> None:
    script = Script(fail_on="bad")
    settings = Settings(batch_shortfall_allowance=0)
    result = await simulate(
        script,
        PROMPTS,
        [batch("ok ->-> 1,1", "bad ->-> 1,2"), batch("later ->-> 2,1")],
        "T",
        "H",
        settings=settings,
        count=COUNT,
    )
    assert result.skipped_questions == [{"batch": 0, "question": 1, "error": "RuntimeError: model down"}]
    assert result.chat == [] and result.stopped_at_batch == 0
    lenient = await simulate(
        Script(fail_on="bad"), PROMPTS, [batch("ok ->-> 1,1", "bad ->-> 1,2")], "T", "H", count=COUNT
    )
    assert len(lenient.chat) == 1 and lenient.stopped_at_batch is None


@pytest.mark.asyncio
async def test_a_failed_turn_leaves_no_trace_in_memory_or_transcript() -> None:
    class FailingFollowup(Script):
        async def __call__(self, role, messages):
            if role == "assistant" and messages[-1]["content"].startswith("user says followup"):
                raise RuntimeError("model down")
            return await super().__call__(role, messages)

    # The main answer is kept; the follow-up whose answer failed is rolled back and not double counted.
    script = FailingFollowup(follows=1)
    result = await simulate(script, PROMPTS, [batch("q ->-> 1,1", "next ->-> 1,2")], "T", "H", count=COUNT)
    history = result.chat[0]
    assert [m.get("question_type") for m in history if m["role"] == "user"] == ["main_question", "main_question"]
    assert [m["id"] for m in history] == [0, 1, 2, 3]
    assert result.skipped_questions == []
    assert result.interrupted_turns == [{"batch": 0, "question": 0, "error": "RuntimeError: model down"}]
    # The next question's assistant call sees alternating roles: no dangling follow-up message.
    last_call = [m for r, m in script.calls if r == "assistant"][-1]
    assert [m["role"] for m in last_call[1:]] == ["user", "assistant", "user"]

    # A failed main question is skipped entirely, and the next question starts from a clean memory.
    script = Script(fail_on="bad")
    result = await simulate(script, PROMPTS, [batch("bad ->-> 1,1", "ok ->-> 1,2")], "T", "H", count=COUNT)
    assert [m["content"] for m in result.chat[0]] == ["ok ->-> 1,2", "answer 1"]
    assert result.skipped_questions[0]["question"] == 0 and result.interrupted_turns == []
    ok_call = [m for r, m in script.calls if r == "assistant"][-1]
    assert [m["content"] for m in ok_call[1:]] == ["ok ->-> 1,2" + ENGLISH_SUFFIX]


@pytest.mark.asyncio
async def test_memory_keeps_the_newest_message_when_the_budget_is_too_small() -> None:
    script = Script()
    # max_tokens below max_summary_tokens + 400: BEAM's budget for recent messages is negative.
    memory = SummaryMemory(script, COUNT, max_tokens=300, recent_messages_count=10, max_summary_tokens=100)
    await memory.add("user", "a" * 600)
    await memory.add("assistant", "b" * 600)
    await memory.add("user", "the question " * 80)
    assert memory.messages == [{"role": "user", "content": "the question " * 80}]
    assert memory.summary == "a summary"
    # Nothing is left to summarize, so add() stops instead of making ten more summary calls.
    calls = len(script.calls)
    await memory.add("assistant", "z" * 3000)
    assert memory.messages[-1]["content"] == "z" * 3000 and len(script.calls) - calls <= 2


@pytest.mark.asyncio
async def test_assistant_memory_budget_has_a_floor() -> None:
    script = Script()
    long_plans = Batch(messages=["q ->-> 2,1"], current_plan="PLAN", previous_plans="p" * 120000)
    settings = Settings(assistant_context_tokens=28000, assistant_min_memory_tokens=8000)
    await simulate(script, PROMPTS, [batch("a ->-> 1,1"), long_plans], "T", "H", settings=settings, count=COUNT)
    # The system prompt alone is ~40K tokens, yet the question is still sent verbatim.
    last = [m for r, m in script.calls if r == "assistant"][-1]
    assert last[-1]["content"] == "q ->-> 2,1" + ENGLISH_SUFFIX


@pytest.mark.asyncio
async def test_memory_encodes_each_message_once() -> None:
    encoded: list[str] = []

    class Counting(TokenCounter):
        def __call__(self, text):
            encoded.append(text)
            return super().__call__(text)

    memory = SummaryMemory(
        Script(), Counting(None), max_tokens=10**6, recent_messages_count=10, max_summary_tokens=100
    )
    for index in range(20):
        await memory.add("user", f"message {index}")
    assert sorted(encoded) == sorted({*(f"message {i}" for i in range(20)), ""})


@pytest.mark.asyncio
async def test_needs_followup_formats_history_like_beam_and_trims_to_budget() -> None:
    script = Script(follows=1)
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    assert await needs_followup(script, PROMPTS, "resp", history, "T", "H", COUNT, Settings())
    prompt = script.calls[-1][1][-1]["content"]
    assert "USER: {'role': 'user', 'content': 'hello'}" in prompt and "ASSISTANT: {'role': 'assistant'" in prompt

    tight = Settings(check_max_tokens=1300)
    long_history = [{"role": "user", "content": "h" * 600} for _ in range(4)]
    await needs_followup(script, PROMPTS, "r" * 3000, long_history, "T", "H", COUNT, tight)
    trimmed = script.calls[-1][1][-1]["content"]
    assert "... [truncated]" in trimmed

    starved = Settings(check_max_tokens=900)
    await needs_followup(script, PROMPTS, "r" * 5000, long_history, "T" * 300, "H", COUNT, starved)
    assert "...[truncated]" in script.calls[-1][1][-1]["content"]


@pytest.mark.asyncio
async def test_includes_question_truncates_long_answers() -> None:
    from responses_api_agents.long_conversation_simulation.simulation import includes_question

    script = Script(asks=1)
    assert await includes_question(script, PROMPTS, "x" * 3000, COUNT, Settings(check_max_tokens=100))
    assert "... [truncated]" in script.calls[-1][1][-1]["content"]
