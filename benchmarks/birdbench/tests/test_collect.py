# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/birdbench/collect.py``."""

import asyncio
import json
import sqlite3
from pathlib import Path

import aiohttp
import pytest

from benchmarks.birdbench import collect
from benchmarks.birdbench.prepare import DIALECTS, prompt_config_fpath_for


def _choice(content="```sql\nSELECT 1\n```", reasoning=None, finish_reason="stop", token_logprobs=(-0.5, -0.25)):
    message = {"content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    logprobs = {"content": [{"token": t, "logprob": lp} for t, lp in zip(["a", "<|im_end|>"], token_logprobs)]}
    return {"message": message, "finish_reason": finish_reason, "logprobs": logprobs}


class TestTextReconstruction:
    def test_reasoning_then_content_and_stop_token_on_clean_stop(self):
        choice = _choice(content="ans", reasoning="think")
        assert collect._reconstruct_generated_text(choice) == "think</think>ans<|im_end|>"

    def test_length_cutoff_mid_reasoning_does_not_invent_closing_tag_or_stop_token(self):
        choice = _choice(content="", reasoning="partial", finish_reason="length")
        assert collect._reconstruct_generated_text(choice) == "partial"

    def test_no_reasoning_is_just_content(self):
        assert collect._reconstruct_generated_text(_choice(content="x", finish_reason="length")) == "x"

    def test_closing_tag_without_logprobs_still_added_on_stop(self):
        choice = {
            "message": {"content": "ans", "reasoning_content": "think"},
            "finish_reason": "stop",
            "logprobs": None,
        }
        assert collect._reconstruct_generated_text(choice) == "think</think>ans"

    def test_think_prefix_only_when_reasoning_present(self):
        assert collect._add_think_prefix_for_saving("t", _choice(reasoning="r")) == "<think>t"
        assert collect._add_think_prefix_for_saving("t", _choice()) == "t"

    def test_logprob_sum_and_missing_logprobs(self):
        assert collect._sum_chat_completion_logprobs(_choice()) == pytest.approx(-0.75)
        assert collect._sum_chat_completion_logprobs({"logprobs": None}) is None
        assert collect._sum_chat_completion_logprobs({"logprobs": {"content": [{"token": "a"}]}}) is None


@pytest.fixture
def sqlite_db(tmp_path):
    db_dir = tmp_path / "databases" / "db1"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "db1.sqlite"))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()
    conn.close()
    return tmp_path / "databases"


class TestScorer:
    @pytest.fixture
    def scorer(self, sqlite_db):
        return collect.make_scorer("sqlite", sqlite_db, sql_concurrency=2, spark_warehouse_dir=Path("unused"))

    @pytest.mark.asyncio
    async def test_matching_result_scores_one(self, scorer):
        assert await scorer("db1", "SELECT COUNT(*) FROM t", "```sql\nSELECT COUNT(x) FROM t\n```") == 1.0

    @pytest.mark.asyncio
    async def test_wrong_result_no_codeblock_and_pred_error_score_zero(self, scorer):
        assert await scorer("db1", "SELECT COUNT(*) FROM t", "```sql\nSELECT 99\n```") == 0.0
        assert await scorer("db1", "SELECT COUNT(*) FROM t", "SELECT COUNT(*) FROM t") == 0.0
        assert await scorer("db1", "SELECT COUNT(*) FROM t", "```sql\nSELECT * FROM missing\n```") == 0.0

    @pytest.mark.asyncio
    async def test_gold_error_scores_zero_rather_than_matching_an_erroring_prediction(self, scorer):
        assert await scorer("db1", "SELECT * FROM missing", "```sql\nSELECT * FROM missing\n```") == 0.0


class _FakeResponse:
    def __init__(self, status, payload=None):
        self.status, self._payload = status, payload or {}
        self.ok = status < 400

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return "body"

    async def json(self):
        return self._payload

    def raise_for_status(self):
        raise aiohttp.ClientResponseError(None, (), status=self.status)


class _FakeSession:
    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), []

    def post(self, url, headers, json, timeout):
        self.calls.append(json)
        return _FakeResponse(self.statuses.pop(0), {"choices": [_choice()]})


async def _noop_latency(_elapsed):
    pass


class TestCallPolicyModel:
    @pytest.fixture(autouse=True)
    def no_backoff_sleep(self, monkeypatch):
        async def instant(_s):
            pass

        monkeypatch.setattr(collect.asyncio, "sleep", instant)

    async def _call(self, session, **kw):
        return await collect._call_policy_model(
            session, asyncio.Semaphore(1), "http://x/v1", "k", "m", [], 1.0, 10, kw.get("effort"), _noop_latency
        )

    @pytest.mark.asyncio
    async def test_retries_transient_status_then_succeeds_and_requests_logprobs(self):
        session = _FakeSession([429, 503, 200])
        assert (await self._call(session))["finish_reason"] == "stop"
        assert len(session.calls) == 3
        assert session.calls[0]["logprobs"] is True and session.calls[0]["top_logprobs"] == 1
        assert "reasoning_effort" not in session.calls[0]

    @pytest.mark.asyncio
    async def test_reasoning_effort_is_sent_when_set(self):
        session = _FakeSession([200])
        await self._call(session, effort="low")
        assert session.calls[0]["reasoning_effort"] == "low"

    @pytest.mark.asyncio
    async def test_non_retryable_status_raises_immediately(self):
        session = _FakeSession([400])
        with pytest.raises(aiohttp.ClientResponseError):
            await self._call(session)
        assert len(session.calls) == 1

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        session = _FakeSession([503] * collect._MAX_REQUEST_RETRIES)
        with pytest.raises(aiohttp.ClientResponseError):
            await self._call(session)
        assert len(session.calls) == collect._MAX_REQUEST_RETRIES


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestCollectAndReprocess:
    @pytest.mark.asyncio
    async def test_collect_writes_grouped_and_raw_and_reprocess_reproduces_rewards(
        self, sqlite_db, tmp_path, monkeypatch
    ):
        rows = [
            {"id": 0, "db_id": "db1", "question": "q0", "sql_context": "ctx", "gt_sql": "SELECT COUNT(*) FROM t"},
            {"id": 7, "db_id": "db1", "question": "q7", "sql_context": "ctx", "gt_sql": "SELECT MAX(x) FROM t"},
        ]
        answers = iter(
            [
                _choice(content="```sql\nSELECT COUNT(x) FROM t\n```"),  # id 0, repeat 0: right
                _choice(content="```sql\nSELECT 99\n```"),  # id 0, repeat 1: wrong
                _choice(content="```sql\nSELECT MAX(x) FROM t\n```", reasoning="hm"),  # id 7, repeat 0: right
                None,  # id 7, repeat 1: fails after retries -> dropped
            ]
        )

        async def fake_call(*_args):
            answer = next(answers)
            if answer is None:
                raise aiohttp.ClientResponseError(None, (), status=500)
            return answer

        monkeypatch.setattr(collect, "_call_policy_model", fake_call)
        scorer = collect.make_scorer("sqlite", sqlite_db, 2, Path("unused"))
        grouped, raw = tmp_path / "out" / "grouped.jsonl", tmp_path / "out" / "raw_choices.jsonl"
        await collect.collect_direct(
            rows=rows,
            prompt_config_fpath=prompt_config_fpath_for("sqlite"),
            scorer=scorer,
            grouped_output_fpath=grouped,
            raw_choices_output_fpath=raw,
            num_repeats=2,
            temperature=1.0,
            max_tokens=10,
            reasoning_effort=None,
            api_concurrency=1,
            base_url="http://x",
            api_key="k",
            model_name="m",
        )

        records = {r["id"]: r for r in _read_jsonl(grouped)}
        assert sorted(records) == [0, 7]
        assert [c["reward"] for c in records[0]["completions"]] == [1.0, 0.0]
        assert [c["reward"] for c in records[7]["completions"]] == [1.0]  # the failed repeat is dropped
        assert records[7]["completions"][0]["completion"][0]["content"].startswith("<think>hm</think>")
        assert records[0]["completions"][0]["reference_logprob"] == pytest.approx(-0.75)
        assert "q0" in json.dumps(records[0]["context"]) and "ctx" in json.dumps(records[0]["context"])
        assert len(_read_jsonl(raw)) == 3  # only successful generations are persisted

        reprocessed = tmp_path / "out" / "grouped_reprocessed.jsonl"
        await collect.reprocess_raw_choices(raw, scorer, reprocessed)
        again = {r["id"]: r for r in _read_jsonl(reprocessed)}
        assert {i: [c["reward"] for c in r["completions"]] for i, r in again.items()} == {0: [1.0, 0.0], 7: [1.0]}
        assert again[0]["context"] == records[0]["context"]


class TestGroupGymRollouts:
    def test_groups_by_task_orders_repeats_and_rebuilds_reasoning_text(self, tmp_path):
        def row(task, rollout, reward, text, logprobs=None):
            assistant = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
            if logprobs is not None:
                assistant["generation_log_probs"] = logprobs
            reasoning = {"type": "reasoning", "summary": [{"text": "why"}]}
            return {
                "_ng_task_index": task,
                "_ng_rollout_index": rollout,
                "id": 10 + task,
                "db_id": "db1",
                "reward": reward,
                "responses_create_params": {"input": [{"role": "user", "content": f"q{task}"}]},
                "response": {"output": [reasoning, assistant]},
            }

        path = tmp_path / "rollouts.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(r) for r in [row(1, 0, 0.0, "b"), row(0, 1, 0.0, "a1"), row(0, 0, 1.0, "a0", [-1, -2])]
            )
        )
        grouped = collect.group_gym_rollouts(path)
        assert [(g["id"], g["db_id"]) for g in grouped] == [(10, "db1"), (11, "db1")]
        assert [c["reward"] for c in grouped[0]["completions"]] == [1.0, 0.0]
        assert grouped[0]["completions"][0]["reference_logprob"] == -3
        assert grouped[0]["completions"][1]["reference_logprob"] is None
        content = grouped[0]["completions"][0]["completion"][0]["content"]
        assert content.endswith("a0") and "why" in content and "</think>" in content


class TestSelectHard:
    def _rows(self):
        return [
            {"id": i, "db_id": "db1", "question": f"q{i}", "sql_context": "ctx", "gt_sql": "SELECT 1"}
            for i in range(4)
        ]

    def _run(self, tmp_path, grouped_records, *, threshold=1.0):
        prepared, out = tmp_path / "prepared.jsonl", tmp_path / "hard.jsonl"
        prepared.write_text("\n".join(json.dumps(r) for r in self._rows()))
        grouped = tmp_path / "grouped.jsonl"
        grouped.write_text("\n".join(json.dumps(r) for r in grouped_records))
        collect.select_hard_examples(prepared, [grouped], out, threshold, prompt_config_fpath_for("sqlite"))
        return [r["id"] for r in _read_jsonl(out)]

    def test_unsolved_and_never_collected_rows_are_hard_matched_by_id(self, tmp_path):
        records = [
            {"id": 0, "context": [], "completions": [{"reward": 1.0}, {"reward": 0.0}]},
            {"id": 1, "context": [], "completions": [{"reward": 0.0}]},
            {"id": 2, "context": [], "completions": []},
        ]
        assert self._run(tmp_path, records) == [1, 2, 3]  # 3 was never collected

    def test_records_without_id_fall_back_to_matching_the_prompt(self, tmp_path):
        from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config

        cfg = load_prompt_config(str(prompt_config_fpath_for("sqlite")))
        context = apply_prompt_to_row(self._rows()[0], cfg)["responses_create_params"]["input"]
        records = [{"context": context, "completions": [{"reward": 1.0}]}]
        assert self._run(tmp_path, records) == [1, 2, 3]


class TestCli:
    def test_every_dialect_has_a_prompt_config_file(self):
        for dialect in DIALECTS:
            assert prompt_config_fpath_for(dialect).is_file()

    def test_collect_defaults_and_choices(self):
        args = collect._build_argparser().parse_args(["collect"])
        assert (args.split, args.dialect, args.num_repeats, args.sql_concurrency) == ("train", "sqlite", 8, 16)
        with pytest.raises(SystemExit):
            collect._build_argparser().parse_args(["collect", "--split", "val"])

    def test_default_output_dir_is_train_data_split_dialect_then_sanitized_model(self):
        args = collect._build_argparser().parse_args(["collect", "--dialect", "spark"])
        out = collect._output_dir(args, "nvidia/zai-org/glm 5")
        assert out == collect.BENCHMARK_DIR / "train_data" / "train_spark" / "nvidia_zai-org_glm_5"
