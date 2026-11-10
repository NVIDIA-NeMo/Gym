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
"""Document preparation and scoring that follow Artificial Analysis's GDP.pdf method."""

import json
import sys
import types
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest
from pytest import approx

from benchmarks.gdp_pdf import prepare
from benchmarks.gdp_pdf.report_aa import failures_path, load_run, read_jsonl, summarize
from nemo_gym.rollouts import RolloutStore, RunManifest
from nemo_gym.rollouts.rollout_records import rejudge_predecessor_id


# ---------------------------------------------------------------------------
# Preparation: LiteParse 2.5.0 with English OCR, recorded in the manifest
# ---------------------------------------------------------------------------


class _FakeLiteParse:
    """LiteParse 2.5.0's surface: no extract_screenshots option, pages rendered by screenshot()."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeLiteParse.calls.append(kwargs)

    def parse(self, path: str) -> Any:
        pages = [types.SimpleNamespace(page_num=n, text=f"text of page {n}") for n in (1, 2)]
        return types.SimpleNamespace(pages=pages)

    def screenshot(self, path: str) -> list[Any]:
        return [types.SimpleNamespace(page_num=n, image_bytes=b"png", width=10, height=20) for n in (1, 2)]


@pytest.fixture
def fake_liteparse(monkeypatch: pytest.MonkeyPatch):
    _FakeLiteParse.calls = []
    monkeypatch.setitem(sys.modules, "liteparse", types.SimpleNamespace(LiteParse=_FakeLiteParse))
    prepare._liteparse_version.cache_clear()
    yield _FakeLiteParse
    prepare._liteparse_version.cache_clear()


def _pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake")
    return pdf


class TestPrepareDocument:
    def test_uses_aa_liteparse_settings_and_records_them(self, tmp_path: Path, fake_liteparse) -> None:
        with patch.object(prepare, "version", return_value="2.5.0"):
            manifest_path = prepare.prepare_document(_pdf(tmp_path), tmp_path / "out")

        assert fake_liteparse.calls == [{"ocr_enabled": True, "ocr_language": "eng", "dpi": 150, "quiet": True}]
        manifest = json.loads(manifest_path.read_text())
        assert manifest["parser"] == "liteparse"
        assert manifest["parser_version"] == "2.5.0"
        assert manifest["ocr_language"] == "eng"
        assert [page["page_number"] for page in manifest["pages"]] == [1, 2]
        assert all((manifest_path.parent / page["image"]).is_file() for page in manifest["pages"])

    def test_reuses_a_manifest_from_the_same_version(self, tmp_path: Path, fake_liteparse) -> None:
        pdf = _pdf(tmp_path)
        with patch.object(prepare, "version", return_value="2.5.0"):
            prepare.prepare_document(pdf, tmp_path / "out")
            prepare.prepare_document(pdf, tmp_path / "out")
        assert len(fake_liteparse.calls) == 1

    def test_rerenders_a_manifest_from_another_version(self, tmp_path: Path, fake_liteparse) -> None:
        pdf = _pdf(tmp_path)
        with patch.object(prepare, "version", return_value="2.17.0"):
            prepare.prepare_document(pdf, tmp_path / "out")
        prepare._liteparse_version.cache_clear()
        with patch.object(prepare, "version", return_value="2.5.0"):
            manifest_path = prepare.prepare_document(pdf, tmp_path / "out")
        assert len(fake_liteparse.calls) == 2
        assert json.loads(manifest_path.read_text())["parser_version"] == "2.5.0"

    def test_warns_once_about_a_non_aa_version(self, tmp_path: Path, fake_liteparse, capsys) -> None:
        with patch.object(prepare, "version", return_value="2.17.0"):
            prepare.prepare_document(_pdf(tmp_path), tmp_path / "a")
            prepare.prepare_document(_pdf(tmp_path), tmp_path / "b")
        assert capsys.readouterr().out.count("LiteParse 2.17.0 is installed") == 1


# ---------------------------------------------------------------------------
# Scoring: AA's fixed task-by-attempt denominator
# ---------------------------------------------------------------------------

CRITERIA = [{"index": 1, "criterion": "Says A."}, {"index": 2, "criterion": "Says B."}]


def _expected(task_id: str, domain: str = "Legal") -> dict[str, Any]:
    return {"verifier_metadata": {"task_id": task_id, "domain": domain, "criteria": CRITERIA}}


def _rollout(
    task_id: str, repeat: int, passed: Optional[list[bool]], *, answer: str = "An answer.", **extra: Any
) -> dict[str, Any]:
    """A scored rollout; ``passed=None`` leaves the second criterion without a verdict."""
    evaluations = []
    for criterion, verdict in zip(CRITERIA, passed or [True, None]):
        item = {**criterion, "passed": bool(verdict) if answer else False}
        if answer and verdict is not None:
            item["score"] = "1" if verdict else "0"
        evaluations.append(item)
    return {
        "_ng_task_index": int(task_id[1:]),
        "_ng_rollout_index": repeat,
        "verifier_metadata": {"task_id": task_id, "criteria": CRITERIA},
        "response": {
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": answer}]}]
        },
        "criterion_evaluations": evaluations,
        **extra,
    }


class TestReportAA:
    def test_fully_graded_run(self) -> None:
        expected = [_expected("t0"), _expected("t1", domain="Finance")]
        rollouts = [
            _rollout("t0", 0, [True, True]),
            _rollout("t0", 1, [True, False]),
            _rollout("t1", 0, [False, False]),
            _rollout("t1", 1, [True, True]),
        ]
        report = summarize(expected, rollouts, repeats=2)
        assert report["expected_attempts"] == 4
        assert report["all_pass"] == approx(0.5)
        assert report["mean_pass"] == approx((1 + 0.5 + 0 + 1) / 4)
        assert report["domain_mean_pass"] == {"Finance": approx(0.5), "Legal": approx(0.75)}
        assert report["attempt_outcomes"] == {"graded": 4}
        assert report["judging_complete"] and report["all_attempts_present"]
        assert not report["aa_shape"]

    def test_judge_failed_attempt_stays_in_the_denominator(self) -> None:
        failure = {"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_failure_class": "judge_failed"}
        report = summarize([_expected("t0")], [_rollout("t0", 0, [True, True])], [failure], repeats=2)
        assert report["all_pass"] == approx(0.5)
        assert report["attempt_outcomes"] == {"failed_judge_failed": 1, "graded": 1}
        assert not report["judging_complete"]

    def test_incomplete_verdicts_score_zero(self) -> None:
        report = summarize([_expected("t0")], [_rollout("t0", 0, None)], repeats=1)
        assert report["all_pass"] == 0.0 and report["mean_pass"] == 0.0
        assert report["attempt_outcomes"] == {"incomplete_judging": 1}
        assert not report["judging_complete"]

    def test_missing_attempt_scores_zero(self) -> None:
        report = summarize([_expected("t0")], [_rollout("t0", 0, [True, True])], repeats=2)
        assert report["all_pass"] == approx(0.5)
        assert report["attempt_outcomes"] == {"graded": 1, "missing": 1}
        assert not report["judging_complete"] and not report["all_attempts_present"]

    def test_empty_answer_and_terminal_input_failure_are_graded_zeros(self) -> None:
        rollouts = [
            _rollout("t0", 0, [False, False], answer=""),
            _rollout("t0", 1, [False, False], answer="", failure_reason="GDP.pdf terminal input limit: context"),
        ]
        report = summarize([_expected("t0")], rollouts, repeats=2)
        assert report["all_pass"] == 0.0
        assert report["attempt_outcomes"] == {"empty_answer": 1, "terminal_input_failure": 1}
        assert report["judging_complete"]

    def test_an_earlier_failed_try_does_not_replace_a_graded_attempt(self) -> None:
        failure = {"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_failure_class": "agent_run_error"}
        report = summarize([_expected("t0")], [_rollout("t0", 0, [True, True])], [failure], repeats=1)
        assert report["all_pass"] == 1.0
        assert report["attempt_outcomes"] == {"graded": 1}
        assert report["judging_complete"]

    @pytest.mark.parametrize(
        ("score", "passed"),
        [
            ("1", True),
            ("1.0", True),
            ("0.99996", True),
            ("1.5", True),
            ("0", False),
            ("0.0", False),
            ("0.99994", False),
        ],
    )
    def test_decimal_scores_follow_the_verifier(self, score: str, passed: bool) -> None:
        # The verifier saves str() of the judge's decimal score and passes it when it rounds to at least 1.
        rollout = _rollout("t0", 0, [passed, passed])
        for item in rollout["criterion_evaluations"]:
            item["score"] = score
        report = summarize([_expected("t0")], [rollout], repeats=1)
        assert report["all_pass"] == float(passed)
        assert report["attempt_outcomes"] == {"graded": 1}
        assert report["judging_complete"]

    @pytest.mark.parametrize(
        ("score", "passed"), [("1.0", False), ("0.5", True), ("NaN", False), ("abc", False), (1, True)]
    )
    def test_a_verdict_the_verifier_could_not_have_saved_is_incomplete(self, score: Any, passed: bool) -> None:
        rollout = _rollout("t0", 0, [True, True])
        rollout["criterion_evaluations"][0].update(score=score, passed=passed)
        report = summarize([_expected("t0")], [rollout], repeats=1)
        assert report["all_pass"] == 0.0
        assert report["attempt_outcomes"] == {"incomplete_judging": 1}
        assert not report["judging_complete"]

    def test_a_masked_verification_failure_is_not_judged(self) -> None:
        # The agent masks an attempt whose /verify call failed and keeps its answer for re-verification.
        rollout = _rollout("t0", 0, [True, True], mask_sample=True, failure_kind="agent_route_failed")
        rollout["criterion_evaluations"] = []
        report = summarize([_expected("t0")], [rollout], repeats=1)
        assert report["all_pass"] == 0.0
        assert report["attempt_outcomes"] == {"failed_agent_route_failed": 1}
        assert not report["judging_complete"]

    def test_inputs_attribute_the_failure_of_a_task_without_a_successful_rollout(self) -> None:
        inputs = [{"_ng_task_index": 42, "_ng_rollout_index": 0, "verifier_metadata": {"task_id": "t42"}}]
        failure = {"_ng_task_index": 42, "_ng_rollout_index": 0, "_ng_failure_class": "agent_run_error"}
        report = summarize([_expected("t42")], [], [failure], repeats=1, inputs=inputs)
        assert report["attempt_outcomes"] == {"failed_agent_run_error": 1}
        assert report["all_attempts_present"] and not report["judging_complete"]
        assert summarize([_expected("t42")], [], [failure], repeats=1)["attempt_outcomes"] == {"missing": 1}

    def test_a_task_index_of_two_tasks_is_rejected(self) -> None:
        inputs = [{"_ng_task_index": 0, "verifier_metadata": {"task_id": "t1"}}]
        with pytest.raises(ValueError, match="more than one task"):
            summarize([_expected("t0"), _expected("t1")], [_rollout("t0", 0, [True, True])], repeats=1, inputs=inputs)

    def test_empty_answer_with_credit_is_rejected(self) -> None:
        rollout = _rollout("t0", 0, [False, False], answer="")
        rollout["criterion_evaluations"][0]["passed"] = True
        with pytest.raises(ValueError, match="empty answer"):
            summarize([_expected("t0")], [rollout], repeats=1)

    def test_duplicate_or_unexpected_rollouts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            summarize([_expected("t0")], [_rollout("t0", 0, [True, True])] * 2, repeats=1)
        with pytest.raises(ValueError, match="outside"):
            summarize([_expected("t0")], [_rollout("t0", 1, [True, True])], repeats=1)


class TestLoadRun:
    def test_a_reverified_result_replaces_the_masked_one(self, tmp_path: Path) -> None:
        rows = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "agent_ref": {"name": "agent"},
                "verifier_metadata": {"task_id": "t0", "criteria": CRITERIA},
            }
        ]
        source = tmp_path / "source.jsonl"
        source.write_text(json.dumps(rows[0]) + "\n")
        output = tmp_path / "rollouts.jsonl"

        def prepare_run() -> tuple[list[dict[str, Any]], RunManifest]:
            return rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})

        graded = _rollout("t0", 0, [True, True])
        graded["response"] |= {"object": "response", "id": "resp-1", "status": "completed"}
        with RolloutStore.start_or_resume(output, prepare_run, resume=False) as store:
            row = store.pending(3)[0]
            store.record_dispatch(row)
            store.record_outcome(
                row
                | {
                    "response": graded["response"],
                    "reward": 0.0,
                    "mask_sample": True,
                    "failure_kind": "agent_route_failed",
                }
            )
        (masked,) = RolloutStore.read(output).selected("success")
        with RolloutStore.open_for_rejudgment(output) as store:
            pending = store.admit_rejudgment(masked)
            reverified = {key: value for key, value in masked.items() if key not in ("mask_sample", "failure_kind")}
            assert store.record_rejudgment(
                masked,
                reverified
                | {
                    "reward": 1.0,
                    "criterion_evaluations": graded["criterion_evaluations"],
                    "_ng_rejudged_failure_id": rejudge_predecessor_id(masked),
                    "_ng_rejudge_operation_id": pending.operation_id,
                    "_ng_rejudge_admitted_at": pending.admitted_at,
                },
            )

        # Both results stay in the file; read as they are, they are two rows for one attempt.
        assert len(read_jsonl(output)) == 2
        with pytest.raises(ValueError, match="Duplicate"):
            summarize([_expected("t0")], *load_run(output, failures_path(output))[:2], repeats=1)
        rollouts, failures, inputs = load_run(output)
        report = summarize([_expected("t0")], rollouts, failures, repeats=1, inputs=inputs)
        assert report["all_pass"] == 1.0
        assert report["attempt_outcomes"] == {"graded": 1}
        assert report["judging_complete"]

    def test_rollouts_without_a_manifest_are_read_with_their_failures_file(self, tmp_path: Path) -> None:
        output = tmp_path / "rollouts.jsonl"
        output.write_text(json.dumps(_rollout("t0", 0, [True, True])) + "\n")
        failure = {"_ng_task_index": 0, "_ng_rollout_index": 1, "_ng_failure_class": "judge_failed"}
        failures_path(output).write_text(json.dumps(failure) + "\n")
        rollouts, failures, inputs = load_run(output)
        assert len(rollouts) == 1 and failures == [failure] and inputs == []
