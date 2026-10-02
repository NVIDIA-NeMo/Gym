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
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from resources_servers.chemreason_bench import metrics as M
from resources_servers.chemreason_bench import response_parsing
from resources_servers.chemreason_bench.app import (
    ChemReasonBenchResourcesServer,
    ChemReasonBenchResourcesServerConfig,
    ChemReasonBenchVerifyRequest,
)
from resources_servers.chemreason_bench.response_parsing import (
    MAX_RESPONSE_CHARS,
    to_prediction,
    to_prediction_lm,
)


_SERVER_DIR = Path(__file__).absolute().parents[1]

GOLD = {
    "ordering": {"correct_order": ["1", "2", "0"]},
    "contrastive_choice": {"correct_option_idx": 1},
    "step_validation": {"label": True},
    "condition_validation": {"label": False},
    "step_completion": {"action": "WASH", "slots": {"reagent": "$7$"}},
    "rationalization": {"gold_rationale": "Calcium activates the carbonyl."},
}
# Question-side vocabulary the post-processors need; upstream reads it from the
# prompt row, prepare.py carries it on the prepared row.
QUESTION_SIDE = {
    "ordering": {"expected_step_ids": ["0", "1", "2"]},
    "contrastive_choice": {"options": ["$4$", "$5$", "$6$", "$7$"]},
}

GOLD_REPLY = {
    "ordering": '{"predicted_order": ["1","2","0"]}',
    "contrastive_choice": '{"predicted_option_idx": 1}',
    "step_validation": '{"score": 1.0}',
    "condition_validation": '{"score": 0.0}',
    "step_completion": '{"action":"WASH","slots":{"reagent":"$7$"}}',
    "rationalization": '{"gold_rationale":"Calcium activates the carbonyl."}',
}


def _make_server() -> ChemReasonBenchResourcesServer:
    return ChemReasonBenchResourcesServer(
        config=ChemReasonBenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )


def _make_request(output_text: str, **fields) -> ChemReasonBenchVerifyRequest:
    response = NeMoGymResponse(
        id="test-id",
        created_at=1234.5,
        model="test-model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg-id",
                content=[NeMoGymResponseOutputText(annotations=[], text=output_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    return ChemReasonBenchVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        **fields,
    )


def _verify(server, output_text, **fields):
    """Fills in the question-side vocabulary for the task under test.

    Explicit values in `fields` win, so a test can still exercise the
    missing-vocabulary path.
    """
    task_type = fields.get("task_type")
    # Guarded: wrong-type tests pass unhashable values here on purpose.
    defaults = QUESTION_SIDE.get(task_type, {}) if isinstance(task_type, str) else {}
    for key, value in defaults.items():
        fields.setdefault(key, value)
    return asyncio.run(server.verify(_make_request(output_text, **fields)))


class TestRowShapes:
    """Prepared rows are flat; the committed example nests under verifier_metadata."""

    @pytest.mark.parametrize("task_type", sorted(GOLD))
    def test_flat_row_scores_gold_at_ceiling(self, task_type):
        result = _verify(
            _make_server(),
            GOLD_REPLY[task_type],
            task_id=f"{task_type}_001_1",
            task_type=task_type,
            ground_truth=GOLD[task_type],
        )
        assert result.harness_failure is False
        assert result.status == "ok"
        assert result.reward == pytest.approx(1.0)

    def test_nested_row_is_lifted(self):
        result = _verify(
            _make_server(),
            GOLD_REPLY["ordering"],
            verifier_metadata={
                "task_id": "ordering_001_1",
                "task_type": "ordering",
                "ground_truth": GOLD["ordering"],
                **QUESTION_SIDE["ordering"],
            },
        )
        assert result.task_type == "ordering"
        assert result.reward == pytest.approx(1.0)

    def test_top_level_wins_over_nested(self):
        result = _verify(
            _make_server(),
            GOLD_REPLY["ordering"],
            task_type="ordering",
            ground_truth=GOLD["ordering"],
            verifier_metadata={"task_type": "rationalization", "ground_truth": {"gold_rationale": "x"}},
        )
        assert result.task_type == "ordering"


class TestHarnessFailures:
    """A malformed row is a status, never a 500 -- a 500 ends the whole run."""

    def test_missing_task_type(self):
        result = _verify(_make_server(), "{}")
        assert result.status == "bad_task_type"
        assert result.harness_failure is True
        assert result.reward == 0.0

    def test_unknown_task_type(self):
        result = _verify(_make_server(), "{}", task_type="not_a_task", ground_truth={})
        assert result.status == "bad_task_type"
        assert result.harness_failure is True

    @pytest.mark.parametrize(
        "fields",
        [
            {"task_type": "not_a_task", "ground_truth": {}},
            {"task_type": "ordering", "ground_truth": None},
            {"task_type": "ordering", "ground_truth": {}, "protocol": "lm"},
        ],
    )
    def test_harness_failures_are_masked(self, fields):
        """The 0.0 measures the harness, not the model, so it must not reach the score.

        compute_metrics drops these, but mask_sample is what keeps them out of the
        downstream reward statistics Gym computes on its own.
        """
        result = _verify(_make_server(), "{}", **fields)
        assert result.harness_failure is True
        assert result.mask_sample is True

    def test_a_scored_row_is_not_masked(self):
        result = _verify(_make_server(), GOLD_REPLY["ordering"], task_type="ordering", ground_truth=GOLD["ordering"])
        assert result.mask_sample is False

    @pytest.mark.parametrize("bad", [None, [], "a string", 7, True])
    def test_ground_truth_wrong_type(self, bad):
        result = _verify(_make_server(), "{}", task_type="ordering", ground_truth=bad)
        assert result.status == "bad_ground_truth"
        assert result.harness_failure is True

    @pytest.mark.parametrize("bad", [7, [], {}, True])
    def test_task_type_wrong_type(self, bad):
        result = _verify(_make_server(), "{}", ground_truth={}, task_type=bad)
        assert result.status == "bad_task_type"
        assert result.harness_failure is True

    @pytest.mark.parametrize("bad_id", [123, ["a"], {"k": "v"}, 1.5, True])
    def test_non_string_task_id_does_not_crash(self, bad_id):
        """Fields are declared Any so a wrong-typed row is a status, not a 422.

        That widening is what lets a non-string reach _sanitize, so it must coerce
        rather than assume str -- an int previously raised AttributeError inside
        verify(), which is a 500 and aborts the run.
        """
        result = _verify(
            _make_server(),
            GOLD_REPLY["ordering"],
            task_id=bad_id,
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.harness_failure is False
        assert isinstance(result.task_id, str)
        json.dumps(result.model_dump())

    @pytest.mark.parametrize(
        "field,bad",
        [("expected_step_ids", "not a list"), ("options", 7), ("legend", ["a"]), ("benchmark_id", "x")],
    )
    def test_wrong_typed_question_side_vocabulary_degrades(self, field, bad):
        """Costs the assist for that row; must not fail it."""
        result = _verify(
            _make_server(),
            GOLD_REPLY["step_validation"],
            task_type="step_validation",
            ground_truth=GOLD["step_validation"],
            **{field: bad},
        )
        assert result.harness_failure is False
        assert result.reward == pytest.approx(1.0)

    def test_surrogate_in_task_id_does_not_break_the_response(self):
        result = _verify(
            _make_server(),
            GOLD_REPLY["ordering"],
            task_id="ordering_\udcff_1",
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        # Must survive being encoded for the wire.
        json.dumps(result.model_dump())


class TestModelOutputHandling:
    def test_empty_output_scores_zero_not_crash(self):
        result = _verify(_make_server(), "", task_type="ordering", ground_truth=GOLD["ordering"])
        assert result.status == "empty_output"
        assert result.reward == 0.0
        assert result.harness_failure is False

    def test_think_block_is_stripped(self):
        result = _verify(
            _make_server(),
            "<think>weighing options</think>" + GOLD_REPLY["ordering"],
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.reward == pytest.approx(1.0)

    def test_rightmost_object_wins(self):
        """A corrected answer further down the reply is the model's conclusion."""
        text = '{"predicted_order": ["0","1","2"]} on reflection: {"predicted_order": ["1","2","0"]}'
        result = _verify(_make_server(), text, task_type="ordering", ground_truth=GOLD["ordering"])
        assert result.reward == pytest.approx(1.0)

    def test_prose_only_reply_is_not_excused(self):
        result = _verify(
            _make_server(), "I think step 1 comes first.", task_type="ordering", ground_truth=GOLD["ordering"]
        )
        assert result.status == "no_json_found"
        assert result.reward == 0.0
        assert result.harness_failure is False

    def test_prose_only_rationalization_is_scored_not_zeroed(self):
        """Upstream falls through to _raw, so a non-JSON rationale still scores."""
        result = _verify(
            _make_server(),
            "Calcium activates the carbonyl.",
            task_type="rationalization",
            ground_truth=GOLD["rationalization"],
        )
        assert result.status == "no_json_found"
        assert result.reward == pytest.approx(1.0)

    def test_reasoning_trace_is_not_scored_as_the_rationale(self):
        """The fallback must use CLEANED text, not output_text.

        Scoring a whole <think> block through token_f1 measures the trace, not
        the answer, and matches neither upstream nor any published number.
        """
        answer = "Calcium activates the carbonyl."
        traced = f"<think>{'borohydride reduces esters slowly ' * 20}</think>{answer}"
        plain = _verify(_make_server(), answer, task_type="rationalization", ground_truth=GOLD["rationalization"])
        with_trace = _verify(_make_server(), traced, task_type="rationalization", ground_truth=GOLD["rationalization"])
        assert with_trace.reward == pytest.approx(plain.reward)

    @pytest.mark.parametrize("reply", ["YES", '{"note":"unsure"}', "{broken", "Sure: [1, 2]"])
    def test_unparseable_binary_reply_scores_the_0_5_default(self, reply):
        """A FAILED parse is {"_raw": ...} upstream: no score, 0.5 >= 0.5, positive."""
        server = _make_server()
        assert _verify(server, reply, task_type="step_validation", ground_truth={"label": True}).reward == 1.0
        assert _verify(server, reply, task_type="step_validation", ground_truth={"label": False}).reward == 0.0

    @pytest.mark.parametrize(
        "reply", ["[1, 2]", "null", '"YES"', "```json\n[1, 2]\n```", "```\nnull\n```", '[{"score": 0.9}]']
    )
    def test_valid_non_object_json_is_negative(self, reply):
        """Valid JSON that is not an object reaches post_binary as a non-dict: label False.

        Differential against the test above: `Sure: [1, 2]` fails to parse and
        defaults positive; bare `[1, 2]` parses to a list and is negative.
        """
        server = _make_server()
        positive = _verify(server, reply, task_type="step_validation", ground_truth={"label": True})
        assert positive.status == "non_object_json"
        assert positive.reward == 0.0
        assert _verify(server, reply, task_type="step_validation", ground_truth={"label": False}).reward == 1.0

    def test_binary_score_below_threshold_is_negative(self):
        """The 0.5 default is a default, not a constant: a real score still decides."""
        result = _verify(_make_server(), '{"score": 0.2}', task_type="step_validation", ground_truth={"label": True})
        assert result.reward == 0.0

    def test_unknown_amount_unit_is_scrubbed_not_fatal(self):
        """Canonicalization drops an out-of-set amount_unit before scoring.

        Upstream's _sanitize_units removes amount_* when the unit is neither a
        mass/volume/mole unit nor migratable to time/temperature, so the fatal
        format-error flag never sees it and the reagent still matches.
        """
        result = _verify(
            _make_server(),
            '{"action":"WASH","slots":{"reagent":"$7$","amount_value":1,"amount_unit":"furlongs"}}',
            task_type="step_completion",
            ground_truth=GOLD["step_completion"],
        )
        assert result.contributions["format_error"] == 0.0
        assert result.reward == pytest.approx(1.0)

    def test_illegal_duration_unit_is_fatal_and_zeroes_the_task(self):
        """duration_unit and temperature_unit are normalized but never dropped.

        So they remain the only route to the fatal flag after canonicalization --
        which is exactly how the six gold rows carrying duration_unit "day" trip
        upstream's own legality check.
        """
        result = _verify(
            _make_server(),
            '{"action":"WASH","slots":{"reagent":"$7$","duration_value":2,"duration_unit":"day"}}',
            task_type="step_completion",
            ground_truth=GOLD["step_completion"],
        )
        assert result.contributions["format_error"] == 1.0
        assert result.reward == 0.0

    @pytest.mark.parametrize(
        "junk",
        ["{", '{"a":', "é" * 50, '{"slots": [1,2]}', '{"predicted_order": {"a":1}}', "<think>unterminated"],
    )
    @pytest.mark.parametrize("task_type", sorted(GOLD))
    def test_malformed_output_is_charged_to_the_model_not_the_harness(self, junk, task_type):
        """A raise here would be a 500, and a 500 aborts the whole run.

        Beyond not raising, the row must land as a scored answer: harness_failure
        False so it stays in the denominator, and a status that names what
        happened rather than a silent zero.
        """
        result = _verify(_make_server(), junk, task_type=task_type, ground_truth=GOLD[task_type])
        assert result.harness_failure is False
        assert result.status in ("ok", "no_json_found", "empty_output")
        assert result.contributions is not None


class TestMetrics:
    def test_corpus_reduction_and_key_metrics(self):
        server = _make_server()
        rollouts = [
            [_verify(server, GOLD_REPLY[t], task_type=t, ground_truth=GOLD[t]).model_dump()] for t in sorted(GOLD)
        ]
        # f1_positive is undefined over an all-negative corpus, and GOLD's
        # condition_validation instance is negative. Add a positive one so the
        # task has a defined ceiling; see test_all_negative_corpus_scores_zero.
        rollouts.append(
            [
                _verify(
                    server, '{"score": 1.0}', task_type="condition_validation", ground_truth={"label": True}
                ).model_dump()
            ]
        )
        computed = server.compute_metrics(rollouts)
        assert computed["primary_overall"] == pytest.approx(100.0)
        assert "harness_failure" not in computed
        # Every task is represented; condition_validation carries the extra positive row.
        # Counts are keyed per protocol now that gen and lm are reduced separately.
        for task_type in M.TASK_TYPES:
            expected = 2.0 if task_type == "condition_validation" else 1.0
            assert computed[f"{task_type}/count[gen]"] == expected
            assert computed[f"{task_type}/protocols"] == 1.0

        key = server.get_key_metrics(computed)
        assert "primary_overall" in key
        # mean/reward matches no published quantity and must not be a headline.
        assert not any(name.startswith("mean/reward") for name in key)
        for task_type in M.TASK_TYPES:
            assert f"{task_type}/{M.PRIMARY_METRIC_BY_TASK[task_type]}" in key

    def test_all_negative_corpus_scores_zero_f1_positive(self):
        """Pins a real property of the published metric, not a bug.

        f1_positive has no positives to find when every gold label is negative,
        so tp=fp=fn=0 and the task scores 0 even on a perfect prediction. It is
        why a per-task denominator matters when reading a sliced report.
        """
        server = _make_server()
        perfect_negative = _verify(
            server, '{"score": 0.0}', task_type="condition_validation", ground_truth={"label": False}
        )
        assert perfect_negative.reward == pytest.approx(1.0)
        computed = server.compute_metrics([[perfect_negative.model_dump()]])
        assert computed["condition_validation/f1_positive"] == 0.0

    def test_harness_failures_are_masked_through_the_aggregator(self):
        """Through compute_aggregate_metrics, a harness failure is excluded from the
        score and reported as coverage -- not as a zero and not as a custom headline.
        """
        server = _make_server()
        good = _verify(server, GOLD_REPLY["ordering"], task_type="ordering", ground_truth=GOLD["ordering"])
        bad = _verify(server, "{}")
        rows = [
            {**good.model_dump(), "_ng_task_index": 0, "_ng_rollout_index": 0},
            {**bad.model_dump(), "_ng_task_index": 1, "_ng_rollout_index": 0},
        ]
        result = compute_aggregate_metrics(
            rows, compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
        )
        assert result.agent_metrics["mean/reward"] == pytest.approx(1.0)
        assert result.agent_metrics["coverage/masked_rollouts"] == 1
        assert result.agent_metrics["coverage/measured_rollouts"] == 1
        assert "harness_failure" not in result.agent_metrics
        assert "harness_failure" not in result.key_metrics
        assert "primary_overall" in result.key_metrics

    def test_absent_task_scores_zero_rather_than_shrinking_the_denominator(self):
        server = _make_server()
        only_one = _verify(server, GOLD_REPLY["ordering"], task_type="ordering", ground_truth=GOLD["ordering"])
        computed = server.compute_metrics([[only_one.model_dump()]])
        # One task at 100, five absent at 0 -> 100/6.
        assert computed["primary_overall"] == pytest.approx(100.0 / 6)

    def test_f1_positive_is_corpus_level_not_a_row_average(self):
        """Always answering positive must not score 1.0 when gold is mixed."""
        server = _make_server()
        rollouts = [
            [
                _verify(
                    server, '{"score": 1.0}', task_type="step_validation", ground_truth={"label": True}
                ).model_dump()
            ],
            [
                _verify(
                    server, '{"score": 1.0}', task_type="step_validation", ground_truth={"label": False}
                ).model_dump()
            ],
        ]
        computed = server.compute_metrics(rollouts)
        # precision 1/2, recall 1/1 -> F1 = 2/3.
        assert computed["step_validation/f1_positive"] == pytest.approx(200.0 / 3)


class TestExampleData:
    def test_example_rows_are_scoreable(self):
        server = _make_server()
        rows = [json.loads(line) for line in (_SERVER_DIR / "data" / "example.jsonl").open(encoding="utf-8")]
        assert len(rows) == 5
        for row in rows:
            meta = row["verifier_metadata"]
            reply = json.dumps(_gold_reply_for(meta["task_type"], meta["ground_truth"]))
            result = _verify(server, reply, verifier_metadata=meta)
            assert result.harness_failure is False
            assert result.reward > 0.0


def _gold_reply_for(task_type, gt):
    if task_type == "ordering":
        return {"predicted_order": gt["correct_order"]}
    if task_type == "contrastive_choice":
        return {"predicted_option_idx": gt["correct_option_idx"]}
    if task_type in ("step_validation", "condition_validation"):
        return {"score": 1.0 if gt["label"] else 0.0}
    if task_type == "step_completion":
        return {"action": gt["action"], "slots": gt.get("slots") or {}}
    return {"gold_rationale": gt["gold_rationale"]}


class TestParsingContract:
    def test_to_prediction_rejects_unknown_task(self):
        with pytest.raises(ValueError, match="unknown task_type"):
            to_prediction("nope", {})

    def test_score_row_rejects_unknown_task(self):
        with pytest.raises(ValueError, match="unknown task_type"):
            M.score_row("nope", {}, {})

    def test_reply_is_truncated_at_the_length_cap(self, monkeypatch):
        """The cap must bound the input, not merely avoid raising.

        The cap is monkeypatched rather than read from the module: building the
        filler from the real constant would make the test self-referential and
        unable to detect the cap being raised or removed.
        """
        monkeypatch.setattr(response_parsing, "MAX_RESPONSE_CHARS", 50)
        payload = '{"predicted_order":["1"]}'
        filler = "x" * 100
        assert response_parsing.extract_json(payload + filler)[1] == "ok"
        assert response_parsing.extract_json(filler + payload) == (None, "no_json_found")

    @pytest.mark.parametrize(
        "reply",
        [
            'He said "maybe. {"predicted_option_idx": 1}',
            'I am 5" tall. {"score": 1.0}',
        ],
    )
    def test_a_prose_quote_does_not_hide_the_object(self, reply):
        """An unbalanced quote outside any object must not swallow the reply.

        The brace scanner tracks strings so a `}` inside one is not an end; before
        the first `{` there is no string to track, and treating prose as one made
        every following brace invisible.
        """
        assert response_parsing.extract_json(reply)[1] == "ok"

    def test_an_enormous_json_number_is_malformed_not_a_crash(self):
        """float() of a 400-digit int raises OverflowError, not ValueError.

        An unhandled one leaves verify() as a 500, which ends the whole run -- the
        outcome every other malformed-input path here exists to avoid.
        """
        huge = '{"score": 1' + "0" * 400 + "}"
        result = _verify(_make_server(), huge, task_type="step_validation", ground_truth={"label": True})
        assert result.harness_failure is False
        # Falls back to the 0.5 default, which is >= 0.5 and therefore positive.
        assert result.reward == pytest.approx(1.0)

    def test_shipped_cap_is_bounded(self):
        """Separate from the mechanism: the value actually shipped must be finite."""
        assert 0 < MAX_RESPONSE_CHARS <= 1_000_000


class TestLmProtocol:
    """The lm protocol asks for one bare decision token instead of JSON."""

    @pytest.mark.parametrize(
        "reply,expected_reward",
        [("YES", 1.0), ("yes", 1.0), ("NO", 0.0), (" Yes.", 1.0), ("<think>hmm</think>YES", 1.0)],
    )
    def test_binary_token_decides_the_label(self, reply, expected_reward):
        result = _verify(
            _make_server(), reply, task_type="step_validation", ground_truth={"label": True}, protocol="lm"
        )
        assert result.reward == pytest.approx(expected_reward)

    def test_binary_first_token_wins_when_both_appear(self):
        result = _verify(
            _make_server(), "NO, not YES", task_type="step_validation", ground_truth={"label": False}, protocol="lm"
        )
        assert result.reward == pytest.approx(1.0)

    def test_binary_without_a_decision_token_falls_back_to_negative(self):
        result = _verify(
            _make_server(), "maybe", task_type="step_validation", ground_truth={"label": True}, protocol="lm"
        )
        assert result.status == "no_decision_token"
        assert result.reward == 0.0

    def test_index_is_read_from_a_bare_integer(self):
        result = _verify(
            _make_server(), "1", task_type="contrastive_choice", ground_truth={"correct_option_idx": 1}, protocol="lm"
        )
        assert result.reward == pytest.approx(1.0)

    def test_index_without_a_number_scores_zero(self):
        result = _verify(
            _make_server(),
            "the second one",
            task_type="contrastive_choice",
            ground_truth={"correct_option_idx": 1},
            protocol="lm",
        )
        assert result.status == "no_decision_token"
        assert result.reward == 0.0

    def test_lm_is_rejected_for_gen_only_tasks(self):
        result = _verify(_make_server(), "YES", task_type="ordering", ground_truth=GOLD["ordering"], protocol="lm")
        assert result.status == "no_lm_protocol"
        assert result.harness_failure is True

    def test_primary_metric_averages_the_two_protocols(self):
        """The published metric is (m_gen + m_lm) / 2 for the discriminative tasks."""
        server = _make_server()
        gold = {"label": True}
        gen_right = _verify(server, '{"score": 1.0}', task_type="step_validation", ground_truth=gold)
        lm_wrong = _verify(server, "NO", task_type="step_validation", ground_truth=gold, protocol="lm")
        computed = server.compute_metrics([[gen_right.model_dump()], [lm_wrong.model_dump()]])
        assert computed["step_validation/f1_positive[gen]"] == pytest.approx(100.0)
        assert computed["step_validation/f1_positive[lm]"] == pytest.approx(0.0)
        assert computed["step_validation/f1_positive"] == pytest.approx(50.0)
        assert computed["step_validation/protocols"] == 2.0

    def test_single_protocol_falls_back_rather_than_halving(self):
        """A gen-only dataset must not be silently scored as if lm were zero."""
        server = _make_server()
        gen_right = _verify(server, '{"score": 1.0}', task_type="step_validation", ground_truth={"label": True})
        computed = server.compute_metrics([[gen_right.model_dump()]])
        assert computed["step_validation/f1_positive"] == pytest.approx(100.0)
        assert computed["step_validation/protocols"] == 1.0


class TestUpstreamPostProcessing:
    """Behaviours ported from upstream's post_ordering / post_contrastive."""

    def test_id_prefixed_tokens_are_canonicalized(self):
        result = _verify(
            _make_server(),
            '{"predicted_order": ["id1","id2","id0"]}',
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.reward == pytest.approx(1.0)

    def test_step_prefixed_tokens_are_canonicalized(self):
        result = _verify(
            _make_server(),
            '{"predicted_order": ["step_1","step_2","step_0"]}',
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.reward == pytest.approx(1.0)

    def test_duplicates_are_dropped(self):
        result = _verify(
            _make_server(),
            '{"predicted_order": ["1","1","2","0"]}',
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.reward == pytest.approx(1.0)

    def test_partial_answer_is_completed_in_presentation_order(self):
        """Upstream's 'critical behavior': fill remaining ids once one matched.

        Without this a partial answer with a single legal id scores 0; upstream
        completes it to ["1","0","2"] and scores the pairs. Affects 71 of 1266
        Qwen ordering rows in the first full run.
        """
        result = _verify(
            _make_server(),
            '{"predicted_order": ["1"]}',
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        # Completed to ["1","0","2"] vs gold ["1","2","0"]: 2 of 3 pairs right.
        assert result.reward == pytest.approx(2 / 3)

    def test_wholly_unmatched_answer_is_not_fabricated(self):
        """Upstream is explicit: no valid id means an empty list, not the gold order."""
        result = _verify(
            _make_server(),
            '{"predicted_order": ["zzz","qqq"]}',
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.reward == 0.0

    def test_choice_string_recovers_the_index(self):
        result = _verify(
            _make_server(),
            '{"predicted_choice": "$5$"}',
            task_type="contrastive_choice",
            ground_truth=GOLD["contrastive_choice"],
        )
        assert result.reward == pytest.approx(1.0)

    def test_out_of_range_index_does_not_fall_back_to_zero(self):
        """Upstream refuses to default to option 0; it must score wrong."""
        result = _verify(
            _make_server(),
            '{"predicted_option_idx": 99}',
            task_type="contrastive_choice",
            ground_truth={"correct_option_idx": 0},
        )
        assert result.reward == 0.0


class TestLmLogprobs:
    """lm labels come from summed token mass, as upstream computes them."""

    @staticmethod
    def _with_logprobs(server, text, top, **fields):
        response = NeMoGymResponse(
            id="t",
            created_at=1.0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="i",
                    content=[
                        NeMoGymResponseOutputText(
                            annotations=[],
                            text=text,
                            type="output_text",
                            logprobs=[
                                {
                                    "token": top[0][0],
                                    "bytes": [],
                                    "logprob": top[0][1],
                                    "top_logprobs": [{"token": t, "bytes": [], "logprob": lp} for t, lp in top],
                                }
                            ],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        request = ChemReasonBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=response,
            protocol="lm",
            **fields,
        )
        return asyncio.run(server.verify(request))

    def test_logprobs_override_a_json_shaped_reply(self):
        """The 1,103 Llama lm replies that came back as JSON are now irrelevant.

        The reply opens with '{', but the restricted argmax over the first
        position still decides YES vs NO -- which is what upstream computes.
        """
        result = self._with_logprobs(
            _make_server(),
            '{"result": "NO"}',
            [("{", -0.1), ("YES", -0.5), ("NO", -2.0)],
            task_type="step_validation",
            ground_truth={"label": True},
        )
        assert result.status == "ok_logprobs"
        assert result.reward == pytest.approx(1.0)

    def test_tokenizer_variants_match(self):
        result = self._with_logprobs(
            _make_server(),
            " Yes",
            [(" Yes", -0.2), ("No", -1.0)],
            task_type="step_validation",
            ground_truth={"label": True},
        )
        assert result.reward == pytest.approx(1.0)

    def test_index_argmax_restricted_to_digits(self):
        result = self._with_logprobs(
            _make_server(),
            "The",
            [("The", -0.1), ("2", -0.4), ("1", -3.0)],
            task_type="contrastive_choice",
            ground_truth={"correct_option_idx": 2},
            options=["a", "b", "c", "d"],
        )
        assert result.status == "ok_logprobs"
        assert result.reward == pytest.approx(1.0)

    def test_falls_back_to_text_without_logprobs(self):
        result = _verify(
            _make_server(),
            "YES",
            task_type="step_validation",
            ground_truth={"label": True},
            protocol="lm",
        )
        assert result.status == "ok"
        assert result.reward == pytest.approx(1.0)

    def test_summed_mass_beats_the_single_largest_token(self):
        """Summed mass, not argmax: NO is the single likeliest token, but two YES
        spellings outweigh it (2*exp(-1.5)=0.446 > exp(-1.0)=0.368)."""
        result = self._with_logprobs(
            _make_server(),
            "NO",
            [("NO", -1.0), ("YES", -1.5), (" YES", -1.5)],
            task_type="step_validation",
            ground_truth={"label": True},
        )
        assert result.status == "ok_logprobs"
        assert result.reward == pytest.approx(1.0)

    def test_suffix_tokens_are_not_decision_tokens(self):
        """Only Y = {YES, NO} counts (paper eq. 1); predict.py's suffix rule would
        let MONO and PORNO bury the real NO and flip the label."""
        result = self._with_logprobs(
            _make_server(),
            "YES",
            [("YES", -1.2), ("NO", -1.4), ("MONO", -0.2), ("PORNO", -0.3)],
            task_type="step_validation",
            ground_truth={"label": True},
        )
        assert result.status == "ok_logprobs"
        # exp(-1.2)=0.301 > exp(-1.4)=0.247, so YES wins once the suffixes are out.
        assert result.reward == pytest.approx(1.0)

    def test_one_sided_mass_abstains_positive(self):
        """Upstream: None -> post_binary -> 0.5 -> label True. Abstention is positive."""
        server = _make_server()
        positive = self._with_logprobs(
            server,
            "YES",
            [("YES", -0.1), ("MAYBE", -2.0)],
            task_type="step_validation",
            ground_truth={"label": True},
        )
        negative = self._with_logprobs(
            server,
            "YES",
            [("YES", -0.1), ("MAYBE", -2.0)],
            task_type="step_validation",
            ground_truth={"label": False},
        )
        assert positive.status == "ok_logprobs"
        assert positive.reward == pytest.approx(1.0)
        assert negative.reward == pytest.approx(0.0)

    def test_confident_no_with_yes_outside_the_window_stays_negative(self):
        """Upstream sees YES in the full vocabulary and labels this False; abstaining
        here would hand a confident NO the positive default."""
        window = [("NO", -0.01)] + [(f"t{i}", -7.5) for i in range(19)]
        negative = self._with_logprobs(
            _make_server(), "NO", window, task_type="step_validation", ground_truth={"label": False}
        )
        assert negative.status == "ok_logprobs"
        assert negative.reward == pytest.approx(1.0)

    def test_neither_candidate_visible_abstains_positive(self):
        result = self._with_logprobs(
            _make_server(),
            "ok",
            [("ok", -0.1), ("the", -2.0)],
            task_type="step_validation",
            ground_truth={"label": True},
        )
        assert result.status == "lm_abstained"
        assert result.reward == pytest.approx(1.0)

    def test_a_single_visible_index_still_decides(self):
        """predict.py's >=2-index guard zeroed every Phi-3-mini contrastive row
        over a top-k window; the paper's eq. (2) has no such condition."""
        result = self._with_logprobs(
            _make_server(),
            "2",
            [("2", -0.2), ("the", -1.0), ("an", -2.0)],
            task_type="contrastive_choice",
            ground_truth={"correct_option_idx": 2},
            options=["a", "b", "c", "d"],
        )
        assert result.status == "ok_logprobs"
        assert result.reward == pytest.approx(1.0)

    def test_contrastive_abstains_only_when_no_index_is_visible(self):
        result = self._with_logprobs(
            _make_server(),
            "the",
            [("the", -0.2), ("an", -1.0)],
            task_type="contrastive_choice",
            ground_truth={"correct_option_idx": 2},
            options=["a", "b", "c", "d"],
        )
        assert result.status == "lm_abstained"
        assert result.reward == pytest.approx(0.0)

    def test_the_sampled_token_is_not_counted_twice(self):
        """vLLM repeats the sampled token inside top_logprobs; it must count once."""
        result = self._with_logprobs(
            _make_server(),
            "YES",
            # Sampled YES repeats in the alternatives. Counted once, NO wins
            # (0.368 > 0.223); counted twice, YES would (0.446).
            [("YES", -1.5), ("NO", -1.0)],
            task_type="step_validation",
            ground_truth={"label": False},
        )
        assert result.status == "ok_logprobs"
        assert result.reward == pytest.approx(1.0)


class TestRationalizationFallbackScope:
    """Upstream sets _raw only on a parse FAILURE (predict.py:303)."""

    def test_parsed_dict_without_a_rationale_key_scores_empty(self):
        """It must not be scored on its own JSON text; upstream gives it "".

        The wrong key deliberately carries text OVERLAPPING the gold rationale.
        A non-overlapping payload scores 0 whether or not the fallback is scoped,
        so it cannot tell the branches apart.
        """
        result = _verify(
            _make_server(),
            '{"note":"Calcium activates the carbonyl."}',
            task_type="rationalization",
            ground_truth=GOLD["rationalization"],
        )
        assert result.status == "ok"
        assert result.reward == 0.0

    def test_unparsed_reply_still_falls_back(self):
        result = _verify(
            _make_server(),
            "Calcium activates the carbonyl.",
            task_type="rationalization",
            ground_truth=GOLD["rationalization"],
        )
        assert result.status == "no_json_found"
        assert result.reward == pytest.approx(1.0)


class TestOneCleaningRule:
    """Every raw-text fallback receives the think-stripped reply, not output_text.

    The case must be an UNPARSED reply: a parsed dict gets no recovery text at all
    (see TestRawRecoveryScope), so the cleaning rule would never be reached.
    Ordering has no testable case -- see TestRawRecoveryScope for why its recovery
    path is unreachable.
    """

    def test_contrastive_recovery_ignores_a_reasoning_trace(self):
        """An option named only inside a trace must not become the answer."""
        traced = "<think>the answer is surely $5$</think>I cannot decide."
        result = _verify(
            _make_server(),
            traced,
            task_type="contrastive_choice",
            ground_truth=GOLD["contrastive_choice"],
        )
        assert result.reward == 0.0


class TestRawRecoveryScope:
    """Recovery text exists only where upstream sets _raw: on a parse FAILURE.

    Upstream's three recovery paths all read ``obj.get("_raw", "")``, which the
    extractor writes only when parsing fails. A dict that parsed but lacks the
    requested key therefore scores empty. Scanning it as text instead would award
    credit upstream never gives.
    """

    def test_parsed_contrastive_dict_naming_the_option_is_not_rescanned(self):
        """The wrong key holds the gold option string, so a rescan would score 1.0."""
        result = _verify(
            _make_server(),
            '{"note": "I pick $5$"}',
            task_type="contrastive_choice",
            ground_truth=GOLD["contrastive_choice"],
        )
        assert result.status == "ok"
        assert result.reward == 0.0

    def test_unparsed_contrastive_reply_still_falls_back(self):
        result = _verify(
            _make_server(), "I pick $5$", task_type="contrastive_choice", ground_truth=GOLD["contrastive_choice"]
        )
        assert result.status == "no_json_found"
        assert result.reward == pytest.approx(1.0)

    def test_ordering_recovery_is_unreachable_as_upstream(self):
        """Both ordering branches score empty, and upstream's does too.

        Upstream reaches its raw scan only when `predicted_order` is present and
        not a list -- but that means the reply parsed, so `_raw` is absent and the
        scan reads "". With no JSON at all, `got` defaults to [], already a list,
        so the scan never runs. Recorded because the dead path is easy to mistake
        for a porting bug.
        """
        server = _make_server()
        for reply in ('{"predicted_order": "id1 id2 id0"}', "id1 id2 id0"):
            assert _verify(server, reply, task_type="ordering", ground_truth=GOLD["ordering"]).reward == 0.0


class TestLmIndexRecovery:
    """An lm contrastive index is a bare small integer, not any digit run."""

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("2", 2),
            ("Option 2", 2),
            ("$2$", -1),  # a reagent placeholder is not an option index
            ("0" * 5000, -1),  # int() raises above 4,300 digits; must not escape verify()
        ],
    )
    def test_index_is_read_only_from_a_bare_integer(self, reply, expected):
        assert to_prediction_lm("contrastive_choice", reply)["predicted_option_idx"] == expected


class TestProseQuoteBeforeObject:
    """A quote in prose must not swallow a valid trailing object.

    The scanner enters string mode only at depth > 0, so an unmatched prose
    quote ahead of the first `{` leaves the real object findable.
    """

    def test_unmatched_prose_quote_does_not_hide_the_object(self):
        result = _verify(
            _make_server(),
            'Step "1 goes first. {"predicted_order": ["1","2","0"]}',
            task_id="ordering_001_1",
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.status == "ok"
        assert result.reward == pytest.approx(1.0)

    def test_a_quote_inside_the_object_still_masks_braces(self):
        # The guard must not disable string mode: a brace inside a JSON string
        # is not structure, so the rightmost *valid* object still wins.
        result = _verify(
            _make_server(),
            '{"note": "not an object: {"} {"predicted_order": ["1","2","0"]}',
            task_id="ordering_001_1",
            task_type="ordering",
            ground_truth=GOLD["ordering"],
        )
        assert result.reward == pytest.approx(1.0)


class TestNumericConversionLimits:
    """Oversized numbers are malformed answers, not failed rollouts.

    float() raises OverflowError -- not ValueError -- on a several-hundred-digit
    JSON integer; unhandled it escapes verify() and aborts the run.
    """

    def test_an_oversized_score_falls_back_to_the_default(self):
        # The object parses; only the value overflows, so this is the
        # missing-score path, not a parse failure. Asserted differentially
        # against a reply that simply omits the score.
        server = _make_server()
        fields = dict(
            task_id="step_validation_001_1",
            task_type="step_validation",
            ground_truth=GOLD["step_validation"],
        )
        oversized = _verify(server, '{"score": %s}' % ("9" * 400), **fields)
        omitted = _verify(server, '{"other": 1}', **fields)
        assert oversized.harness_failure is False
        assert oversized.status == omitted.status
        assert oversized.reward == pytest.approx(omitted.reward)

    def test_an_oversized_amount_value_is_scored_not_raised(self):
        result = _verify(
            _make_server(),
            '{"action":"WASH","slots":{"reagent":"$7$","amount_value":%s}}' % ("9" * 400),
            task_id="step_completion_001_1",
            task_type="step_completion",
            ground_truth=GOLD["step_completion"],
        )
        assert result.harness_failure is False
        assert result.reward is not None
