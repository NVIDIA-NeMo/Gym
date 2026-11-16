# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for benchmarks/beam/prepare.py: a fake HF tokenizer and a monkeypatched load_dataset."""

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

from benchmarks.beam import prepare as prepare_module


class _FakeTokenizer:
    """Whitespace tokens; the chat template adds 3 header tokens, 2 per user / 3 per assistant turn,
    and 1 generation-prompt token."""

    chat_template = "fake"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[str]:
        return text.split()

    def apply_chat_template(
        self, messages: List[Dict[str, str]], tokenize: bool = True, add_generation_prompt: bool = True
    ) -> List[int]:
        ids = [0] * 3
        for m in messages:
            ids += [0] * (2 if m["role"] == "user" else 3) + [1] * len(m["content"].split())
        return ids + [0] * add_generation_prompt


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> _FakeTokenizer:
    tokenizer = _FakeTokenizer()
    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", module)
    return tokenizer


def _conversation(conversation_id: str, n_turns: int) -> Dict[str, Any]:
    chat = [
        [
            {"role": "user", "content": f"question {i} about the budget app"},
            {"role": "assistant", "content": "an answer"},
        ]
        for i in range(n_turns)
    ]
    probing = {
        "abstention": [{"question": "Who reviewed it?", "rubric": ["says unknown"], "ideal_response": "Not stated."}],
        "event_ordering": [
            {
                "question": "In what order?",
                "rubric": ["setup", "deploy"],
                "answer": "setup, deploy",
                "difficulty": "easy",
            }
        ],
    }
    return {"conversation_id": conversation_id, "chat": chat, "probing_questions": repr(probing)}


@pytest.fixture
def fake_dataset(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Dict[str, Any]]]:
    splits = {"100K": [_conversation("1", 2), _conversation("2", 40)]}
    monkeypatch.setattr(prepare_module, "load_dataset", lambda name, split: splits[split])
    return splits


def _rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_is_tiktoken_encoding_name(tmp_path: Path) -> None:
    assert prepare_module.is_tiktoken_encoding_name("cl100k_base")
    assert not prepare_module.is_tiktoken_encoding_name("meta-models/Muse-Glimmer-30B")
    assert not prepare_module.is_tiktoken_encoding_name(str(tmp_path))


def test_prompt_counter_calibrates_per_role_overheads(fake_transformers: _FakeTokenizer) -> None:
    counter = prepare_module.PromptCounter("org/fake-model")
    assert counter.role_overhead == {"user": 2, "assistant": 3}
    assert counter.fixed_overhead == 4  # 3 header tokens + 1 generation-prompt token
    messages = [
        {"role": "user", "content": "one two"},
        {"role": "assistant", "content": "three"},
        {"role": "user", "content": "four five six"},
    ]
    additive = counter.fixed_overhead + sum(counter.message_cost(m) for m in messages)
    assert additive == counter.exact_count(messages) == len(fake_transformers.apply_chat_template(messages))


def test_prepare_single_turn_rows(fake_transformers: _FakeTokenizer, fake_dataset: Any, tmp_path: Path) -> None:
    out = prepare_module.prepare(tokenizer_name="org/fake-model", splits=["100K"], output_fpath=tmp_path / "b.jsonl")
    rows = _rows(out)
    assert len(rows) == 4  # 2 conversations x 2 probing questions

    row = rows[0]
    (message,) = row["responses_create_params"]["input"]
    assert message["role"] == "user"
    assert message["content"].startswith("Below is the full history of my earlier conversation with you.")
    assert "User: question 0 about the budget app\n\nAssistant: an answer" in message["content"]
    assert message["content"].endswith("answer my question:\nWho reviewed it?")
    assert row["verifier_metadata"] == {
        "conversation_id": "1",
        "chat_size": "100K",
        "ability": "abstention",
        "question_index": 0,
        "question": "Who reviewed it?",
        "rubric": ["says unknown"],
        "reference_answer": "Not stated.",
        "difficulty": None,
    }
    assert rows[1]["verifier_metadata"]["reference_answer"] == "setup, deploy"
    # The additive count used for every row matches the chat template exactly.
    for r in rows:
        assert r["prompt_tokens"] == len(fake_transformers.apply_chat_template(r["responses_create_params"]["input"]))


def test_prepare_multi_turn_with_answer_only_note(
    fake_transformers: _FakeTokenizer, fake_dataset: Any, tmp_path: Path
) -> None:
    out = prepare_module.prepare(
        tokenizer_name="org/fake-model",
        splits=["100K"],
        output_fpath=tmp_path / "b.jsonl",
        prompt_format="multi_turn",
        answer_only_note=True,
    )
    messages = _rows(out)[0]["responses_create_params"]["input"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"] == prepare_module.QUESTION_TEMPLATE.format(question="Who reviewed it?")


def test_prepare_drops_prompts_over_max_context_tokens(
    fake_transformers: _FakeTokenizer, fake_dataset: Any, tmp_path: Path
) -> None:
    every = _rows(
        prepare_module.prepare(tokenizer_name="org/fake-model", splits=["100K"], output_fpath=tmp_path / "a.jsonl")
    )
    short_max = max(r["prompt_tokens"] for r in every if r["verifier_metadata"]["conversation_id"] == "1")
    kept = _rows(
        prepare_module.prepare(
            tokenizer_name="org/fake-model",
            splits=["100K"],
            output_fpath=tmp_path / "b.jsonl",
            max_context_tokens=short_max,
        )
    )
    assert {r["verifier_metadata"]["conversation_id"] for r in kept} == {"1"}
    assert len(kept) == 2 and all(r["prompt_tokens"] <= short_max for r in kept)


def test_prepare_rejects_unknown_prompt_format(fake_transformers: _FakeTokenizer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt_format"):
        prepare_module.prepare(tokenizer_name="org/fake-model", output_fpath=tmp_path / "b.jsonl", prompt_format="x")
