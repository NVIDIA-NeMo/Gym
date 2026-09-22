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
from nemo_gym.server_utils import ServerClient
from resources_servers.chemreason_bench import metrics as M
from resources_servers.chemreason_bench.app import (
    ChemReasonBenchResourcesServer,
    ChemReasonBenchResourcesServerConfig,
    ChemReasonBenchVerifyRequest,
)
from resources_servers.chemreason_bench.response_parsing import extract_json, to_prediction


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
    for key, value in QUESTION_SIDE.get(task_type, {}).items():
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

    def test_ground_truth_wrong_type(self):
        result = _verify(_make_server(), "{}", task_type="ordering", ground_truth=None)
        assert result.status == "bad_ground_truth"
        assert result.harness_failure is True

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

    def test_non_dict_reply_forces_negative_label(self):
        """Upstream's conservative fallback: a non-JSON reply must not count positive.

        Labelling unparseable replies positive is worth ~0.63-0.73 f1_positive on
        its own, given the 46-57% positive gold rate.
        """
        result = _verify(_make_server(), "YES", task_type="step_validation", ground_truth={"label": True})
        assert result.reward == 0.0

    def test_dict_without_score_follows_upstream_threshold(self):
        """A dict missing `score` falls through to 0.5 >= 0.5 and counts positive."""
        result = _verify(
            _make_server(), '{"note":"unsure"}', task_type="step_validation", ground_truth={"label": True}
        )
        assert result.reward == pytest.approx(1.0)

    def test_illegal_unit_zeroes_step_completion(self):
        result = _verify(
            _make_server(),
            '{"action":"WASH","slots":{"reagent":"$7$","amount_unit":"furlongs"}}',
            task_type="step_completion",
            ground_truth=GOLD["step_completion"],
        )
        assert result.reward == 0.0

    @pytest.mark.parametrize(
        "junk",
        ["{", '{"a":', "é" * 50, '{"slots": [1,2]}', '{"predicted_order": {"a":1}}', "<think>unterminated"],
    )
    @pytest.mark.parametrize("task_type", sorted(GOLD))
    def test_malformed_output_never_raises(self, junk, task_type):
        result = _verify(_make_server(), junk, task_type=task_type, ground_truth=GOLD[task_type])
        assert 0.0 <= result.reward <= 1.0


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
        assert computed["harness_failure"] == 0.0
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

    def test_harness_failures_are_published_not_hidden(self):
        server = _make_server()
        good = _verify(server, GOLD_REPLY["ordering"], task_type="ordering", ground_truth=GOLD["ordering"])
        bad = _verify(server, "{}")
        computed = server.compute_metrics([[good.model_dump()], [bad.model_dump()]])
        assert computed["harness_failure"] == pytest.approx(0.5)

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

    def test_oversized_reply_is_bounded_before_parsing(self):
        obj, status = extract_json("x" * 500_000 + '{"predicted_order":["1"]}')
        # The tail is past the cap, so nothing parses -- but it must not hang or raise.
        assert status in ("no_json_found", "ok")
        assert obj is None or isinstance(obj, dict)


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
    """lm labels come from an argmax restricted to the decision tokens."""

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
