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
"""Metric internals, exercised against the formulas upstream publishes."""

import pytest

from resources_servers.chemreason_bench import metrics as M
from resources_servers.chemreason_bench import slot_canonicalization as SC
from resources_servers.chemreason_bench.task_data import TaskData


class TestPairwiseAccuracy:
    def test_perfect_order(self):
        assert M.pairwise_accuracy(["1", "2", "0"], ["1", "2", "0"]) == 1.0

    def test_reversed_order_scores_zero(self):
        assert M.pairwise_accuracy(["0", "2", "1"], ["1", "2", "0"]) == 0.0

    def test_one_swapped_pair(self):
        # 3 pairs, one inverted.
        assert M.pairwise_accuracy(["2", "1", "0"], ["1", "2", "0"]) == pytest.approx(2 / 3)

    def test_illegal_ids_are_dropped_not_filled(self):
        """Upstream is conservative: unknown ids are discarded, never imputed."""
        assert M.pairwise_accuracy(["1", "99", "2"], ["1", "2", "0"]) == 1.0

    def test_fewer_than_two_legal_ids_scores_zero(self):
        assert M.pairwise_accuracy(["99", "98"], ["1", "2", "0"]) == 0.0
        assert M.pairwise_accuracy(["1"], ["1", "2", "0"]) == 0.0

    def test_integer_ids_compare_as_strings(self):
        assert M.pairwise_accuracy([1, 2, 0], ["1", "2", "0"]) == 1.0


class TestTokenF1:
    def test_identical_text(self):
        assert M.token_f1("the quick brown fox", "the quick brown fox", M.EN_STOP) == 1.0

    def test_both_empty_after_stopword_removal(self):
        assert M.token_f1("the and of", "a an is", M.EN_STOP) == 1.0

    def test_one_side_empty(self):
        assert M.token_f1("", "calcium chloride", M.EN_STOP) == 0.0
        assert M.token_f1("calcium chloride", "", M.EN_STOP) == 0.0

    def test_no_overlap(self):
        assert M.token_f1("sodium borohydride", "quantum chromodynamics", M.EN_STOP) == 0.0

    def test_punctuation_and_case_are_normalized(self):
        assert M.token_f1("Calcium, chloride!", "calcium chloride", M.EN_STOP) == 1.0

    def test_partial_overlap_is_symmetric_f1(self):
        # pred {a,b}, gold {b,c}: overlap 1, precision 1/2, recall 1/2 -> F1 1/2.
        assert M.token_f1("alpha beta", "beta gamma", M.EN_STOP) == pytest.approx(0.5)

    def test_none_is_treated_as_empty(self):
        assert M.normalize_text(None) == ""


class TestSlotF1:
    def test_matching_reagent(self):
        f1, fatal = M.slot_f1({"reagent": "$7$"}, {"reagent": "$7$"})
        assert (f1, fatal) == (1.0, False)

    def test_empty_both_sides_scores_zero(self):
        """An upstream property, not a port bug.

        With no reagents, numeric groups or token fields there are no true
        positives to find, so precision and recall are both 0. This is why the
        benchmark's own gold answers cannot reach 1.0 on step_completion.
        """
        f1, fatal = M.slot_f1({}, {})
        assert (f1, fatal) == (0.0, False)

    def test_reagents_match_as_a_multiset(self):
        pred = {"reagent": "$1$", "reagent_1": "$2$"}
        assert M.slot_f1(pred, pred)[0] == 1.0
        # One right, one wrong: precision 1/2, recall 1/2.
        f1, _ = M.slot_f1({"reagent": "$1$", "reagent_1": "$9$"}, pred)
        assert f1 == pytest.approx(0.5)

    def test_illegal_unit_is_fatal(self):
        _, fatal = M.slot_f1({"amount_unit": "furlongs"}, {"amount_unit": "mL"})
        assert fatal is True

    def test_day_is_not_a_legal_time_unit(self):
        """Six gold rows carry duration_unit 'day'/'days' and trip this."""
        assert M.units_legal("day") is False
        assert M.units_legal("h") is True
        assert M.units_legal(None) is False

    def test_unit_aliases_normalize(self):
        assert M.norm_unit("HOURS") == "h"
        assert M.norm_unit("mins") == "min"
        assert M.norm_unit("µL") == "uL"
        assert M.norm_unit(7) == 7

    def test_token_fields_need_exact_match(self):
        assert M.slot_f1({"temperature_token": "#8#"}, {"temperature_token": "#8#"})[0] == 1.0
        assert M.slot_f1({"temperature_token": "#9#"}, {"temperature_token": "#8#"})[0] == 0.0


class TestNumericSlotComparison:
    def test_temperature_absolute_tolerance(self):
        assert M.compare_numeric_slot("temperature", 25.5, "C", 25.0, "C") is True
        assert M.compare_numeric_slot("temperature", 27.0, "C", 25.0, "C") is False

    def test_time_relative_tolerance(self):
        assert M.compare_numeric_slot("time", 2.6, "h", 2.5, "h") is True
        assert M.compare_numeric_slot("time", 3.5, "h", 2.5, "h") is False

    def test_amount_relative_tolerance(self):
        assert M.compare_numeric_slot("amount", 10.4, "mL", 10.0, "mL") is True
        assert M.compare_numeric_slot("amount", 12.0, "mL", 10.0, "mL") is False

    def test_mismatched_units_never_match(self):
        assert M.compare_numeric_slot("time", 2.5, "min", 2.5, "h") is False
        assert M.compare_numeric_slot("time", 2.5, None, 2.5, "h") is False

    def test_non_numeric_values_do_not_raise(self):
        assert M.compare_numeric_slot("amount", "lots", "mL", 10.0, "mL") is False

    def test_unknown_kind_returns_false(self):
        assert M.compare_numeric_slot("mass", 1.0, "g", 1.0, "g") is False

    def test_zero_gold_uses_unit_denominator(self):
        assert M.compare_numeric_slot("time", 0.05, "h", 0.0, "h") is True


class TestStepCompletionScore:
    def test_formula_matches_paper(self):
        # Raw = 0.8*ActionEM + 0.2*SlotF1, SCS = Raw*(1-FER).
        assert M.step_completion_score(1.0, 1.0, 0.0) == pytest.approx(1.0)
        assert M.step_completion_score(1.0, 0.0, 0.0) == pytest.approx(0.8)
        assert M.step_completion_score(0.0, 1.0, 0.0) == pytest.approx(0.2)

    def test_format_error_rate_penalizes(self):
        assert M.step_completion_score(1.0, 1.0, 1.0) == 0.0
        assert M.step_completion_score(1.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_penalty_is_clamped(self):
        assert M.step_completion_score(1.0, 1.0, 2.0) == 0.0
        assert M.step_completion_score(1.0, 1.0, -1.0) == pytest.approx(1.0)


class TestCorpusReduction:
    def test_f1_positive_from_labels(self):
        assert M.f1_positive_from_labels([1, 1, 0], [1, 1, 0]) == 1.0
        assert M.f1_positive_from_labels([1, 0], [1, 1]) == pytest.approx(2 / 3)
        assert M.f1_positive_from_labels([0, 0], [0, 0]) == 0.0

    def test_empty_rows_score_zero(self):
        for task_type in M.TASK_TYPES:
            assert M.reduce_task(task_type, []) == 0.0

    def test_reduce_rejects_unknown_task(self):
        with pytest.raises(ValueError, match="unknown task_type"):
            M.reduce_task("nope", [{"x": 1}])

    def test_primary_overall_scores_absent_tasks_zero(self):
        assert M.primary_overall({t: 1.0 for t in M.TASK_TYPES}) == pytest.approx(1.0)
        assert M.primary_overall({"ordering": 1.0}) == pytest.approx(1 / 6)
        assert M.primary_overall({}) == 0.0

    def test_safe_div_guards_zero(self):
        assert M.safe_div(1, 0) == 0.0
        assert M.safe_div(1, 2) == 0.5


class TestScoreRow:
    def test_contrastive_non_integer_index_scores_zero(self):
        assert (
            M.score_row("contrastive_choice", {"predicted_option_idx": None}, {"correct_option_idx": 1})["reward"]
            == 0.0
        )

    def test_step_completion_row_carries_its_contributions(self):
        scored = M.score_row(
            "step_completion",
            {"action": "wash", "slots": {"reagent": "$7$"}},
            {"action": "WASH", "slots": {"reagent": "$7$"}},
        )
        # Action match is case-insensitive upstream (both sides upper-cased).
        assert scored["action_em"] == 1.0
        assert scored["slot_f1"] == 1.0
        assert scored["format_error"] == 0.0


class TestTaskDataSchema:
    def test_required_fields(self):
        data = TaskData(task_type="ordering", ground_truth={"correct_order": ["1"]})
        assert data.task_type == "ordering"
        assert data.task_id is None

    def test_extra_fields_allowed(self):
        data = TaskData(task_type="ordering", ground_truth={}, question="text")
        assert data.question == "text"


class TestSlotCanonicalization:
    """Upstream runs canonicalize_slots(slots, legend) before scoring (predict.py:953).

    Paired agreement against upstream's own implementation was 1483/1483 on real
    model outputs for both evaluated models; these pin the individual behaviours.
    """

    LEGEND = {"$4$": "calcium chloride", "$5$": "sodium borohydride", "$7$": "water"}

    def test_alias_keys_collapse_to_reagent(self):
        for alias in ("solvent", "base", "wash_with", "drying_agent", "chemical"):
            out = SC.canonicalize_slots({alias: "$7$"}, self.LEGEND)
            assert out == {"reagent": "$7$"}, alias

    def test_reagent_name_resolves_to_its_placeholder(self):
        assert SC.canonicalize_slots({"reagent": "sodium borohydride"}, self.LEGEND) == {"reagent": "$5$"}

    def test_reagent_name_matching_ignores_case_and_parentheses(self):
        out = SC.canonicalize_slots({"reagent": "Calcium Chloride (anhydrous)"}, self.LEGEND)
        assert out == {"reagent": "$4$"}

    def test_unknown_reagent_name_is_left_alone(self):
        assert SC.canonicalize_slots({"reagent": "unobtainium"}, self.LEGEND) == {"reagent": "unobtainium"}

    def test_amount_blob_splits_into_value_and_unit(self):
        out = SC.canonicalize_slots({"reagent": "$7$", "amount": "10 mL"}, self.LEGEND)
        assert out == {"reagent": "$7$", "amount_value": 10.0, "amount_unit": "mL"}

    def test_placeholder_used_as_a_key_becomes_the_reagent(self):
        out = SC.canonicalize_slots({"$4$": "5 g"}, self.LEGEND)
        assert out == {"reagent": "$4$", "amount_value": 5.0, "amount_unit": "g"}

    def test_time_valued_amount_migrates_to_duration(self):
        out = SC.canonicalize_slots({"amount_value": 2, "amount_unit": "hours"}, self.LEGEND)
        assert out == {"duration_value": 2.0, "duration_unit": "h"}

    def test_temperature_valued_amount_migrates_to_temperature(self):
        out = SC.canonicalize_slots({"amount_value": 25, "amount_unit": "°C"}, self.LEGEND)
        assert out == {"temperature_value": 25.0, "temperature_unit": "C"}

    def test_unknown_amount_unit_drops_the_amount_pair(self):
        assert SC.canonicalize_slots({"amount_value": 1, "amount_unit": "furlongs"}, self.LEGEND) == {}

    def test_unit_spellings_normalize(self):
        # These six were missing from a hand-copied map and cost 59 rows of disagreement.
        for spelling, expected in (
            ("liter", "L"),
            ("litre", "L"),
            ("ltrs", "L"),
            ("mole", "mol"),
            ("moles", "mol"),
            ("ul.", "uL"),
        ):
            out = SC.canonicalize_slots({"amount_value": 1, "amount_unit": spelling}, self.LEGEND)
            assert out.get("amount_unit") == expected, spelling

    def test_tokens_are_lifted_from_a_surviving_slot_value(self):
        """The scan runs after the blob/unit passes, over whatever keys survive."""
        out = SC.canonicalize_slots({"duration_unit": "@9@"}, self.LEGEND)
        assert out == {"duration_token": "@9@"}
        # A token sitting in the reagent value is lifted too, and the reagent stays.
        assert SC.canonicalize_slots({"reagent": "#8#"}, self.LEGEND) == {
            "reagent": "#8#",
            "temperature_token": "#8#",
        }

    def test_blob_keys_that_parse_to_nothing_are_dropped(self):
        """`duration: "@9@"` has no number+unit, so the key is popped before the token scan."""
        assert SC.canonicalize_slots({"duration": "@9@", "temperature": "#8#"}, self.LEGEND) == {}

    def test_token_fields_pass_through_untouched(self):
        pair = {"duration_token": "@9@", "temperature_token": "#8#"}
        assert SC.canonicalize_slots(pair, self.LEGEND) == pair

    def test_keys_outside_the_whitelist_are_dropped(self):
        out = SC.canonicalize_slots({"reagent": "$7$", "vibe": "energetic", "notes": 3}, self.LEGEND)
        assert out == {"reagent": "$7$"}

    def test_unit_given_as_a_list_takes_the_first_element(self):
        out = SC.canonicalize_slots({"amount_value": 1, "amount_unit": ["mL", "L"]}, self.LEGEND)
        assert out.get("amount_unit") == "mL"

    def test_non_numeric_value_is_dropped_rather_than_raising(self):
        out = SC.canonicalize_slots({"amount_value": "lots", "amount_unit": "mL"}, self.LEGEND)
        assert "amount_value" not in out

    def test_empty_and_none_inputs_are_safe(self):
        assert SC.canonicalize_slots(None, None) == {}
        assert SC.canonicalize_slots({}, {}) == {}

    def test_unit_map_matches_upstreams_exactly(self):
        """Every entry, not a spot check.

        Six were missing from a hand-copied version (liter, litre, ltrs, mole,
        moles, ul.) and cost 59 rows of disagreement with upstream before anyone
        noticed, so the whole table is pinned against predict.py's DEFAULT_UCUM.
        """
        expected = {
            "μl": "uL",
            "µl": "uL",
            "ul": "uL",
            "μL": "uL",
            "µL": "uL",
            "uL": "uL",
            "ml": "mL",
            "mL": "mL",
            "l": "L",
            "L": "L",
            "ul.": "uL",
            "liter": "L",
            "litre": "L",
            "ltrs": "L",
            "mg": "mg",
            "g": "g",
            "kg": "kg",
            "µg": "ug",
            "μg": "ug",
            "ug": "ug",
            "ng": "ng",
            "mol": "mol",
            "mmol": "mmol",
            "µmol": "umol",
            "μmol": "umol",
            "umol": "umol",
            "mole": "mol",
            "moles": "mol",
            "s": "s",
            "sec": "s",
            "secs": "s",
            "second": "s",
            "seconds": "s",
            "min": "min",
            "mins": "min",
            "minute": "min",
            "minutes": "min",
            "h": "h",
            "hr": "h",
            "hour": "h",
            "hours": "h",
            "°c": "C",
            "c": "C",
            "°f": "F",
            "f": "F",
            "k": "K",
            "C": "C",
            "F": "F",
            "K": "K",
            "m": "M",
            "M": "M",
            "n": "N",
            "N": "N",
        }
        assert SC._UCUM == expected

    def test_canonicalization_and_scoring_tables_stay_separate(self):
        """predict.py and eval.py ship DIFFERENT tables; merging them would change scores."""
        assert "M" not in M.DEFAULT_UCUM and "n" not in M.DEFAULT_UCUM
        assert SC._UCUM.keys() - M.DEFAULT_UCUM.keys() == {
            "ul.",
            "liter",
            "litre",
            "ltrs",
            "mole",
            "moles",
            "m",
            "M",
            "n",
            "N",
        }
