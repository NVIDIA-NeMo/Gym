# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from pytest import raises

from nemo_gym.decontamination import (
    DecontaminationConfig,
    _check_contamination_async,
    _content_to_text,
    _filter_split,
    _read_problem_texts,
    extract_problem_text,
    run_decontamination,
)


def test_extract_problem_text_responses_string_input():
    row = {"responses_create_params": {"input": "What is 2 + 2?"}}
    assert extract_problem_text(row) == "What is 2 + 2?"


def test_extract_problem_text_responses_user_message():
    row = {"responses_create_params": {"input": [{"role": "user", "content": "Solve x + 1 = 2"}]}}
    assert extract_problem_text(row) == "Solve x + 1 = 2"


def test_extract_problem_text_prefers_user_over_system():
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "the real problem"},
            ]
        }
    }
    assert extract_problem_text(row) == "the real problem"


def test_extract_problem_text_content_parts():
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "part text"}]}]
        }
    }
    assert extract_problem_text(row) == "part text"


def test_extract_problem_text_flat_fallback():
    assert extract_problem_text({"problem": "flat question"}) == "flat question"
    assert extract_problem_text({"q": "custom key"}, problem_text_key="q") == "custom key"


def test_extract_problem_text_returns_none_when_missing():
    assert extract_problem_text({}) is None
    assert extract_problem_text({"responses_create_params": {"input": []}}) is None


def test_extract_problem_text_non_user_message_fallback():
    # No user message, but a developer message with content -> use it.
    row = {"responses_create_params": {"input": [{"role": "developer", "content": "system-ish text"}]}}
    assert extract_problem_text(row) == "system-ish text"


def test_content_to_text_variants():
    assert _content_to_text("hello") == "hello"
    assert _content_to_text([{"type": "input_text", "text": "a"}, {"text": "b"}]) == "a\nb"
    assert _content_to_text(["x", "y"]) == "x\ny"
    assert _content_to_text(123) is None
    assert _content_to_text([]) is None


def test_read_problem_texts_dedup(tmp_path: Path):
    f = tmp_path / "test.jsonl"
    rows = [
        {"problem": "dup"},
        {"problem": "dup"},
        {"responses_create_params": {"input": "unique"}},
        {"problem": ""},  # skipped
    ]
    f.write_text("".join(f"{json.dumps(r)}\n" for r in rows) + "\n")  # trailing blank line ignored
    texts = _read_problem_texts([str(f)], "problem")
    assert texts == ["dup", "unique"]


def test_filter_split_removes_contaminated(tmp_path: Path):
    f = tmp_path / "train.jsonl"
    rows = [
        {"responses_create_params": {"input": "keep me"}},
        {"responses_create_params": {"input": "Remove ME"}},  # case-insensitive match
        {"problem": "also keep"},
    ]
    # Trailing blank line exercises the empty-line skip in _filter_split.
    f.write_text("".join(f"{json.dumps(r)}\n" for r in rows) + "\n")

    kept, removed = _filter_split(f, {"remove me"}, "problem")

    assert (kept, removed) == (2, 1)
    remaining = [json.loads(line) for line in f.read_text().splitlines()]
    texts = [extract_problem_text(r) for r in remaining]
    assert texts == ["keep me", "also keep"]


def test_config_defaults():
    cfg = DecontaminationConfig(test_set_jsonls=["a.jsonl"])
    assert cfg.decontaminate_types == ["train"]
    assert cfg.top_k == 5
    assert cfg.judge_model == "gpt-4o-mini"
    assert cfg.check_both_ways is False


def test_config_requires_test_set_jsonls():
    with raises(Exception):
        DecontaminationConfig()


class _FakeClient:
    """Stand-in for NeMoGymAsyncOpenAI that returns canned judge verdicts."""

    def __init__(self, verdict_for):
        self.verdict_for = verdict_for
        self.calls = 0

    async def create_chat_completion(self, **kwargs):
        self.calls += 1
        user_content = kwargs["messages"][0]["content"]
        verdict = "True" if self.verdict_for in user_content else "False"
        return {"choices": [{"message": {"content": verdict}}]}


def test_check_contamination_exact_match_short_circuits(monkeypatch):
    # Exact (case-insensitive) match must be flagged without any LLM call.
    fake = _FakeClient(verdict_for="never")
    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", lambda **kw: fake)

    candidates = [{"problem": "Find X", "similar_items": ["find x"], "similarity_scores": [0.99]}]
    cfg = DecontaminationConfig(test_set_jsonls=["a.jsonl"])
    results = asyncio.run(_check_contamination_async(candidates, cfg, "key"))

    assert results[0]["contaminated"] is True
    assert fake.calls == 0


def test_check_contamination_llm_verdict(monkeypatch):
    fake = _FakeClient(verdict_for="leaked problem")
    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", lambda **kw: fake)

    candidates = [
        {"problem": "leaked problem", "similar_items": ["a paraphrase"], "similarity_scores": [0.8]},
        {"problem": "clean problem", "similar_items": ["unrelated"], "similarity_scores": [0.3]},
    ]
    cfg = DecontaminationConfig(test_set_jsonls=["a.jsonl"])
    results = asyncio.run(_check_contamination_async(candidates, cfg, "key"))

    assert results[0]["contaminated"] is True
    assert results[1]["contaminated"] is False
    assert fake.calls == 2


def test_check_contamination_check_both_ways_doubles_queries(monkeypatch):
    fake = _FakeClient(verdict_for="never")
    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", lambda **kw: fake)

    candidates = [{"problem": "q", "similar_items": ["a", "b"], "similarity_scores": [0.5, 0.4]}]
    cfg = DecontaminationConfig(test_set_jsonls=["a.jsonl"], check_both_ways=True)
    results = asyncio.run(_check_contamination_async(candidates, cfg, "key"))

    assert results[0]["contaminated"] is False
    # 2 similar items x both directions = 4 judge calls.
    assert fake.calls == 4


def _patch_retrieve_all_pairs(monkeypatch):
    """Make _retrieve_similar pair every train problem with every test problem (no torch)."""

    def fake_retrieve(train_texts, test_texts, config):
        return [
            {"problem": t, "similar_items": list(test_texts), "similarity_scores": [1.0] * len(test_texts)}
            for t in train_texts
        ]

    monkeypatch.setattr("nemo_gym.decontamination._retrieve_similar", fake_retrieve)


def test_run_decontamination_noop_when_none():
    # Should return immediately without touching anything.
    run_decontamination(SimpleNamespace(decontamination=None), Path("/nonexistent"))


def test_run_decontamination_no_test_problems(tmp_path: Path):
    empty = tmp_path / "test.jsonl"
    empty.write_text("")
    cfg = SimpleNamespace(decontamination=DecontaminationConfig(test_set_jsonls=[str(empty)], judge_api_key="k"))
    # No test problems -> early return, no error even though train.jsonl is absent.
    run_decontamination(cfg, tmp_path)


def test_run_decontamination_missing_and_empty_splits(tmp_path: Path, monkeypatch):
    _patch_retrieve_all_pairs(monkeypatch)
    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", lambda **kw: _FakeClient("zzz"))

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(json.dumps({"problem": "leaked"}) + "\n")

    # 'train' split is absent; 'validation' split exists but has only a blank line.
    (tmp_path / "validation.jsonl").write_text("\n")

    cfg = SimpleNamespace(
        decontamination=DecontaminationConfig(
            test_set_jsonls=[str(test_path)],
            decontaminate_types=["train", "validation"],
            judge_api_key="k",
        )
    )
    run_decontamination(cfg, tmp_path)  # both splits skipped, no error


def test_run_decontamination_end_to_end_filters_and_reports(tmp_path: Path, monkeypatch):
    _patch_retrieve_all_pairs(monkeypatch)
    # Judge always says False; the contaminated row is caught by the exact-match short-circuit.
    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", lambda **kw: _FakeClient("zzz"))

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(json.dumps({"problem": "alpha"}) + "\n")

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps({"responses_create_params": {"input": "alpha"}})
        + "\n"  # exact dup -> removed
        + json.dumps({"responses_create_params": {"input": "beta"}})
        + "\n"  # clean -> kept
    )

    cfg = SimpleNamespace(
        decontamination=DecontaminationConfig(
            test_set_jsonls=[str(test_path)],
            judge_api_key="k",
            report_dirpath=str(tmp_path),
        )
    )
    run_decontamination(cfg, tmp_path)

    remaining = [extract_problem_text(json.loads(line)) for line in train_path.read_text().splitlines()]
    assert remaining == ["beta"]

    report = [json.loads(line) for line in (tmp_path / "train_contamination.jsonl").read_text().splitlines()]
    flagged = {r["problem"] for r in report if r["contaminated"]}
    assert flagged == {"alpha"}


def test_run_decontamination_dry_run_keeps_rows(tmp_path: Path, monkeypatch):
    _patch_retrieve_all_pairs(monkeypatch)
    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", lambda **kw: _FakeClient("zzz"))

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(json.dumps({"problem": "alpha"}) + "\n")
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps({"problem": "alpha"}) + "\n")

    cfg = SimpleNamespace(
        decontamination=DecontaminationConfig(test_set_jsonls=[str(test_path)], judge_api_key="k", dry_run=True)
    )
    run_decontamination(cfg, tmp_path)

    # dry_run: row is reported but NOT removed.
    assert [json.loads(line) for line in train_path.read_text().splitlines()] == [{"problem": "alpha"}]
    assert (tmp_path / "train_contamination.jsonl").exists()


def test_run_decontamination_accepts_dict_and_reads_global_api_key(tmp_path: Path, monkeypatch):
    _patch_retrieve_all_pairs(monkeypatch)
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return _FakeClient("zzz")

    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", fake_client)
    # judge_api_key unset -> resolved from global config.
    monkeypatch.setattr("nemo_gym.global_config.get_global_config_dict", lambda: {"openai_api_key": "k"})

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(json.dumps({"problem": "alpha"}) + "\n")
    (tmp_path / "train.jsonl").write_text(json.dumps({"problem": "beta"}) + "\n")

    # Pass decontamination as a plain dict to exercise model_validate coercion.
    cfg = SimpleNamespace(decontamination={"test_set_jsonls": [str(test_path)], "report_dirpath": str(tmp_path)})
    run_decontamination(cfg, tmp_path)

    assert captured["api_key"] == "k"


def test_run_decontamination_warns_when_no_api_key(tmp_path: Path, monkeypatch):
    _patch_retrieve_all_pairs(monkeypatch)
    monkeypatch.setattr("nemo_gym.decontamination.NeMoGymAsyncOpenAI", lambda **kw: _FakeClient("zzz"))
    monkeypatch.setattr("nemo_gym.global_config.get_global_config_dict", lambda: {})

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(json.dumps({"problem": "alpha"}) + "\n")
    (tmp_path / "train.jsonl").write_text(json.dumps({"problem": "beta"}) + "\n")

    cfg = SimpleNamespace(decontamination=DecontaminationConfig(test_set_jsonls=[str(test_path)]))
    run_decontamination(cfg, tmp_path)  # no key anywhere -> warns but proceeds
