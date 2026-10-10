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
"""Preparation pipeline, with the network stubbed.

This module is the one that produces a silently wrong dataset rather than a
crash, so the fetch, the join, the count gate and the argument handling are all
exercised against local fixtures.

``fixtures/golden_prompts.json`` pins the rendered prompt for one instance of
every task family, in both protocols where they exist. The goldens were captured
from a rendering verified byte-for-byte against upstream's ``predict.py``
builders over all 10,648 rows; this file is what keeps them that way.
"""

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).absolute().parents[3] / "benchmarks" / "chemreason_bench"))

import prepare as P  # noqa: E402


_FIXTURES = Path(__file__).absolute().parent / "fixtures"


def _load(name):
    return [json.loads(line) for line in (_FIXTURES / name).open(encoding="utf-8")]


@pytest.fixture
def stub_fetch(monkeypatch):
    """Serve the fixtures instead of GitHub, and fail loudly if a real fetch is attempted."""
    payloads = {P.PROMPTS_URL: _load("prompts.jsonl"), P.ANSWERS_URL: _load("answers.jsonl")}

    def _fake(url):
        if url not in payloads:
            raise AssertionError(f"unexpected fetch: {url}")
        return [dict(row) for row in payloads[url]]

    monkeypatch.setattr(P, "_fetch_jsonl", _fake)
    # The fixture holds one row per task family, not the full corpus.
    monkeypatch.setattr(P, "EXPECTED_TOTAL", 6)
    monkeypatch.setattr(P, "EXPECTED_BY_TASK", {t: 1 for t in P.EXPECTED_BY_TASK})
    monkeypatch.setattr(P, "EXPECTED_LM_ROWS", 3)
    return payloads


def _prepare(tmp_path, **kwargs):
    out = tmp_path / "out.jsonl"
    P.prepare(output_fpath=out, **kwargs)
    return [json.loads(line) for line in out.open(encoding="utf-8")]


class TestPromptRendering:
    def test_rendered_prompts_are_byte_identical_to_the_goldens(self, stub_fetch, tmp_path):
        """The claim `benchmarks/prompts/eval/chemreason_bench/paper.yaml` makes."""
        golden = json.loads((_FIXTURES / "golden_prompts.json").read_text(encoding="utf-8"))
        rows = {r["task_id"]: r for r in _prepare(tmp_path)}
        assert set(rows) == set(golden), "fixture and golden task_id sets diverged"
        for task_id, expected in golden.items():
            assert rows[task_id]["question"] == expected, f"prompt drifted for {task_id}"

    def test_every_task_family_is_covered_by_a_golden(self, stub_fetch, tmp_path):
        covered = {r["task_type"] for r in _prepare(tmp_path)}
        assert covered == set(P.PROMPT_BUILDERS)

    def test_gold_is_never_rendered_into_the_prompt(self, stub_fetch, tmp_path):
        """A leaked answer would make the benchmark unscoreable and is silent."""
        for row in _prepare(tmp_path):
            gold = row["ground_truth"]
            for key in ("gold_explanation", "explanation_private", "description_private", "gold_rationale"):
                secret = gold.get(key)
                if isinstance(secret, str) and len(secret) > 20:
                    assert secret not in row["question"]


class TestJoinAndGates:
    def test_joins_prompts_to_answers_on_task_id(self, stub_fetch, tmp_path):
        for row in _prepare(tmp_path):
            assert row["ground_truth"] == next(
                a["ground_truth"]
                for a in _load("answers.jsonl")
                if a["task_id"] == row["task_id"].removesuffix("::lm")
            )

    def test_missing_gold_refuses_to_write(self, stub_fetch, tmp_path, monkeypatch):
        trimmed = _load("answers.jsonl")[:-1]
        monkeypatch.setattr(P, "_fetch_jsonl", lambda url: _load("prompts.jsonl") if url == P.PROMPTS_URL else trimmed)
        with pytest.raises(ValueError, match="no gold for task_id"):
            _prepare(tmp_path)

    def test_duplicate_task_ids_are_rejected(self, stub_fetch, tmp_path, monkeypatch):
        answers = _load("answers.jsonl")
        monkeypatch.setattr(
            P, "_fetch_jsonl", lambda url: _load("prompts.jsonl") if url == P.PROMPTS_URL else answers + [answers[0]]
        )
        with pytest.raises(ValueError, match="duplicate task_ids"):
            _prepare(tmp_path)

    def test_short_corpus_refuses_to_write(self, stub_fetch, tmp_path, monkeypatch):
        """A truncated download exits 0 and silently changes every denominator."""
        monkeypatch.setattr(P, "EXPECTED_TOTAL", 7)
        with pytest.raises(ValueError, match="expected 7 instances"):
            _prepare(tmp_path)

    def test_per_task_count_change_refuses_to_write(self, stub_fetch, tmp_path, monkeypatch):
        monkeypatch.setattr(P, "EXPECTED_BY_TASK", {**P.EXPECTED_BY_TASK, "ordering": 99})
        with pytest.raises(ValueError, match="per-task counts changed"):
            _prepare(tmp_path)

    def test_task_type_mismatch_between_files_is_rejected(self, stub_fetch, tmp_path, monkeypatch):
        answers = _load("answers.jsonl")
        answers[0]["task_type"] = "ordering" if answers[0]["task_type"] != "ordering" else "rationalization"
        monkeypatch.setattr(P, "_fetch_jsonl", lambda url: _load("prompts.jsonl") if url == P.PROMPTS_URL else answers)
        with pytest.raises(ValueError, match="task_type mismatch"):
            _prepare(tmp_path)

    def test_nothing_is_written_when_a_gate_trips(self, stub_fetch, tmp_path, monkeypatch):
        """Load, validate, then write -- so a failure leaves no truncated file."""
        monkeypatch.setattr(P, "EXPECTED_TOTAL", 7)
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError):
            P.prepare(output_fpath=out)
        assert not out.exists()


class TestProtocolsAndLimit:
    def test_both_protocols_are_emitted_with_lm_trailing(self, stub_fetch, tmp_path):
        rows = _prepare(tmp_path)
        assert [r["protocol"] for r in rows] == ["gen"] * 6 + ["lm"] * 3
        assert all(r["task_id"].endswith("::lm") for r in rows if r["protocol"] == "lm")

    def test_only_discriminative_tasks_get_an_lm_row(self, stub_fetch, tmp_path):
        lm = {r["task_type"] for r in _prepare(tmp_path) if r["protocol"] == "lm"}
        assert lm == set(P.DUAL_PROTOCOL_TASKS)

    def test_protocol_filter(self, stub_fetch, tmp_path):
        assert {r["protocol"] for r in _prepare(tmp_path, protocol="lm")} == {"lm"}
        assert {r["protocol"] for r in _prepare(tmp_path, protocol="gen")} == {"gen"}

    def test_limit_takes_gen_rows_first(self, stub_fetch, tmp_path):
        rows = _prepare(tmp_path, limit=2)
        assert len(rows) == 2
        assert {r["protocol"] for r in rows} == {"gen"}

    def test_limit_reaches_lm_rows_only_with_the_protocol_filter(self, stub_fetch, tmp_path):
        rows = _prepare(tmp_path, limit=2, protocol="lm")
        assert len(rows) == 2
        assert {r["protocol"] for r in rows} == {"lm"}

    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_non_positive_limit_is_rejected_at_parse_time(self, bad):
        """Truthiness once made --limit 0 mean 'no limit' and --limit -1 keep one row."""
        with pytest.raises(SystemExit):
            P.main(["--limit", bad])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"limit": 0}, "limit"),
            ({"limit": -1}, "limit"),
            ({"protocol": "Gen"}, "protocol"),
            ({"protocol": ""}, "protocol"),
        ],
    )
    def test_prepare_validates_its_own_arguments(self, monkeypatch, tmp_path, kwargs, message):
        """Gym calls prepare() directly from prepare_script_args; argparse never runs.

        Without this a `protocol="Gen"` typo silently writes the full corpus, and
        `limit=0` writes it too, both reading as if a subset had been requested.
        """

        def _explode(url):
            raise AssertionError("fetch attempted before argument validation")

        monkeypatch.setattr(P, "_fetch_jsonl", _explode)
        with pytest.raises(ValueError, match=message):
            P.prepare(output_fpath=tmp_path / "out.jsonl", **kwargs)

    def test_arguments_are_validated_before_any_fetch(self, monkeypatch):
        def _explode(url):
            raise AssertionError("fetch attempted before argument validation")

        monkeypatch.setattr(P, "_fetch_jsonl", _explode)
        with pytest.raises(SystemExit):
            P.main(["--limit", "0"])


class TestRowContract:
    def test_question_side_vocabulary_is_carried_per_task(self, stub_fetch, tmp_path):
        by_type = {r["task_type"]: r for r in _prepare(tmp_path) if r["protocol"] == "gen"}
        source = {r["task_type"]: r for r in _load("prompts.jsonl")}
        # The carried values must equal the source row's, not merely be the right type.
        assert by_type["ordering"]["expected_step_ids"] == [
            str(s["step_id"]) for s in source["ordering"]["steps_to_order"]
        ]
        assert by_type["contrastive_choice"]["options"] == source["contrastive_choice"]["options"]
        # canonicalize_slots resolves reagent names through this.
        assert by_type["step_completion"]["legend"] == source["step_completion"]["legend"]
        assert by_type["step_completion"]["legend"], "legend must be non-empty for this fixture"
        # and it is not attached where it has no meaning
        assert "legend" not in by_type["ordering"]
        assert "options" not in by_type["rationalization"]

    def test_rows_carry_dataset_name_and_split(self, stub_fetch, tmp_path):
        """Declared in task_data.py, so they must actually be written."""
        for row in _prepare(tmp_path):
            assert row["dataset_name"] == "chemreason_bench"
            assert row["split"] == "test"

    def test_lm_rows_ask_for_token_probabilities(self, stub_fetch, tmp_path):
        """Upstream decides an lm row from token mass, so the row must request it.

        Both keys are required: `top_logprobs` alone is inert, because vLLM computes
        `logprobs = top_logprobs if logprobs else None`. They travel on
        `metadata.extra_body` since the Responses model forbids unknown fields.
        """
        for row in _prepare(tmp_path, protocol="lm"):
            extra = json.loads(row["responses_create_params"]["metadata"]["extra_body"])
            assert extra["logprobs"] is True
            assert extra["top_logprobs"] >= 2

    def test_gen_rows_ask_for_nothing_extra(self, stub_fetch, tmp_path):
        """Only the lm protocol reads probabilities; gen rows must stay untouched."""
        for row in _prepare(tmp_path, protocol="gen"):
            assert "responses_create_params" not in row

    def test_pinned_revision_is_a_full_sha(self):
        assert len(P.GITHUB_REVISION) == 40
        assert P.PROMPTS_URL.startswith(f"https://raw.githubusercontent.com/{P.GITHUB_REPO}/{P.GITHUB_REVISION}/")
