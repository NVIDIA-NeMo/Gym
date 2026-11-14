# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual parsing and fixed-cohort accounting using invented inputs."""

import unittest
from dataclasses import replace

from resources_servers.videomme_v2.scoring import aggregate_scores, grade_answer
from resources_servers.videomme_v2.task import VerifierMetadata
from resources_servers.videomme_v2.tests.test_task import question


def metadata(task_id: str = "synthetic/one") -> VerifierMetadata:
    return VerifierMetadata.from_question(replace(question(), task_id=task_id), expected_answer="B")


class TestScoring(unittest.TestCase):
    def test_correct_wrong_and_boxed_wrappers(self) -> None:
        for text in (r"\boxed{B}", r"\boxed{\text{B}}", r"\boxed{\mathrm{B}}", r"Therefore: \boxed { B }"):
            with self.subTest(text=text):
                result = grade_answer(text, metadata(), setting="single_turn")
                self.assertEqual((result.outcome, result.predicted_answer, result.reward), ("correct", "B", 1.0))
        self.assertEqual(grade_answer(r"\boxed{A}", metadata(), setting="single_turn").outcome, "incorrect")

    def test_both_settings_use_the_same_grader(self) -> None:
        for text in (r"\boxed{B}", r"\boxed{A}", "", "B", r"\boxed{H}", r"\boxed{A} \boxed{B}"):
            with self.subTest(text=text):
                single = grade_answer(text, metadata(), setting="single_turn")
                multi = grade_answer(text, metadata(), setting="agentic")
                self.assertEqual(replace(single, setting="agentic"), multi)

    def test_no_answer_is_not_the_incorrect_category(self) -> None:
        for text in (None, "", " \n "):
            with self.subTest(text=text):
                result = grade_answer(text, metadata(), setting="agentic")
                self.assertEqual(result.outcome, "no_answer")
                self.assertEqual(result.reward, 0.0)
                self.assertTrue(result.evaluation_completed)

    def test_bare_letter_is_explicit_opt_in(self) -> None:
        strict = grade_answer(" B\n", metadata(), setting="agentic")
        compatible = grade_answer(" B\n", metadata(), setting="agentic", answer_policy="boxed_or_letter_v1")
        self.assertEqual(strict.outcome, "format_error")
        self.assertEqual(compatible.outcome, "correct")

    def test_compatibility_does_not_guess_from_prose(self) -> None:
        for text in ("Answer B", "The answer is B.", "B or C", "b", "BC", r"\boxed{H}"):
            with self.subTest(text=text):
                result = grade_answer(text, metadata(), setting="single_turn", answer_policy="boxed_or_letter_v1")
                self.assertEqual(result.outcome, "format_error")

    def test_option_range_is_enforced(self) -> None:
        for text in (r"\boxed{H}", r"\boxed{Z}", r"\boxed{D}", "H"):
            with self.subTest(text=text):
                result = grade_answer(text, metadata(), setting="agentic", answer_policy="boxed_or_letter_v1")
                self.assertEqual((result.outcome, result.predicted_answer), ("format_error", None))

    def test_questions_with_more_than_eight_options(self) -> None:
        options = tuple(f"choice {number}" for number in range(11))
        eleven = VerifierMetadata.from_question(replace(question(), options=options), expected_answer="K")
        self.assertEqual(grade_answer(r"\boxed{K}", eleven, setting="agentic").outcome, "correct")
        self.assertEqual(grade_answer(r"\boxed{J}", eleven, setting="agentic").outcome, "incorrect")
        self.assertEqual(grade_answer(r"\boxed{L}", eleven, setting="agentic").outcome, "format_error")
        # Up to eight options the counted letters stay A-H: a box past them is no answer.
        self.assertEqual(grade_answer(r"\boxed{B} \boxed{I}", metadata(), setting="agentic").outcome, "correct")

    def test_conflicting_boxed_answers_are_ambiguous(self) -> None:
        result = grade_answer(r"\boxed{B} then \boxed{A}", metadata(), setting="agentic")
        self.assertEqual((result.outcome, result.reward), ("ambiguous_answer", 0.0))

    def test_repeated_same_box_is_unambiguous(self) -> None:
        result = grade_answer(r"\boxed{B} again \boxed{B}", metadata(), setting="single_turn")
        self.assertEqual(result.outcome, "correct")

    def test_reasoning_boxes_are_not_final_answers(self) -> None:
        for tag in ("think", "thinking", "THINK"):
            with self.subTest(tag=tag):
                text = rf"<{tag}>Try \boxed{{A}}</{tag}> Final: \boxed{{B}}"
                self.assertEqual(grade_answer(text, metadata(), setting="agentic").outcome, "correct")
                only_reasoning = rf"<{tag}>Try \boxed{{B}}</{tag}>"
                self.assertEqual(grade_answer(only_reasoning, metadata(), setting="agentic").outcome, "no_answer")

    def test_unclosed_reasoning_is_not_graded(self) -> None:
        result = grade_answer(r"<think>Maybe \boxed{B}", metadata(), setting="agentic")
        self.assertEqual(result.outcome, "no_answer")

    def test_orphaned_reasoning_close_uses_only_later_text(self) -> None:
        result = grade_answer(r"\boxed{A}</think>\boxed{B}", metadata(), setting="agentic")
        self.assertEqual(result.outcome, "correct")

    def test_infrastructure_failure_is_inconclusive(self) -> None:
        result = grade_answer(r"\boxed{B}", metadata(), setting="single_turn", failure="infrastructure_failure")
        self.assertEqual(result.outcome, "infrastructure_failure")
        self.assertIsNone(result.reward)
        self.assertIsNone(result.predicted_answer)
        self.assertFalse(result.evaluation_completed)

    def test_execution_failure_does_not_grade_partial_correct_text(self) -> None:
        result = grade_answer(r"\boxed{B}", metadata(), setting="agentic", failure="execution_failure")
        self.assertEqual((result.outcome, result.reward, result.predicted_answer), ("execution_failure", 0.0, None))
        self.assertTrue(result.evaluation_completed)

    def test_unknown_protocol_and_invalid_text_are_rejected(self) -> None:
        for kwargs in (
            {"setting": "unknown"},
            {"setting": "single_turn", "answer_policy": "guess"},
            {"setting": "single_turn", "failure": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                grade_answer("B", metadata(), **kwargs)
        with self.assertRaises(ValueError):
            grade_answer(7, metadata(), setting="single_turn")

    def test_score_records_reject_inconsistent_fields(self) -> None:
        result = grade_answer(r"\boxed{B}", metadata(), setting="single_turn")
        for kwargs in (
            {"task_id": ""},
            {"dataset_revision": "main"},
            {"setting": "unknown"},
            {"answer_policy": "guess"},
            {"outcome": "unknown"},
            {"reward": 0.0},
            {"reward": True},
            {"evaluation_completed": False},
            {"predicted_answer": None},
            {"predicted_answer": "AB"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(result, **kwargs)


class TestAccuracy(unittest.TestCase):
    def summarize(self, scores, identities=("synthetic/one", "synthetic/two"), **kwargs):
        return aggregate_scores(
            scores, expected_task_ids=identities, dataset_revision="a" * 40, setting="single_turn", **kwargs
        )

    def test_complete_question_level_average(self) -> None:
        scores = [
            grade_answer(r"\boxed{B}", metadata(), setting="single_turn"),
            grade_answer(r"\boxed{A}", metadata("synthetic/two"), setting="single_turn"),
        ]
        result = self.summarize(scores)
        self.assertEqual((result.expected, result.returned, result.completed), (2, 2, 2))
        self.assertEqual((result.avg_accuracy, result.coverage, result.is_complete), (0.5, 1.0, True))
        self.assertEqual((result.outcome_counts["correct"], result.outcome_counts["incorrect"]), (1, 1))

    def test_missing_rows_stay_in_denominator(self) -> None:
        result = self.summarize([grade_answer(r"\boxed{B}", metadata(), setting="single_turn")])
        self.assertEqual((result.avg_accuracy, result.missing, result.is_complete), (0.5, 1, False))
        self.assertEqual(result.coverage, 0.5)

    def test_infrastructure_and_missing_are_not_silently_dropped(self) -> None:
        scores = [
            grade_answer(r"\boxed{B}", metadata(), setting="single_turn"),
            grade_answer(None, metadata("synthetic/two"), setting="single_turn", failure="infrastructure_failure"),
            grade_answer(None, metadata("synthetic/three"), setting="single_turn"),
        ]
        result = self.summarize(scores, ("synthetic/one", "synthetic/two", "synthetic/three", "synthetic/four"))
        self.assertEqual((result.avg_accuracy, result.coverage, result.missing), (0.25, 0.5, 1))
        self.assertEqual(result.outcome_counts["infrastructure_failure"], 1)
        self.assertEqual(result.outcome_counts["no_answer"], 1)
        self.assertFalse(result.is_complete)

    def test_all_missing_is_provisional_zero_not_nan(self) -> None:
        result = self.summarize([])
        self.assertEqual((result.avg_accuracy, result.coverage, result.missing, result.is_complete), (0, 0, 2, False))

    def test_all_completed_wrong_is_still_complete(self) -> None:
        scores = [
            grade_answer(r"\boxed{A}", metadata(identity), setting="single_turn")
            for identity in ("synthetic/one", "synthetic/two")
        ]
        result = self.summarize(scores)
        self.assertEqual((result.avg_accuracy, result.is_complete), (0, True))

    def test_duplicate_and_unexpected_results_are_rejected(self) -> None:
        one = grade_answer(r"\boxed{B}", metadata(), setting="single_turn")
        other = grade_answer(r"\boxed{B}", metadata("unexpected"), setting="single_turn")
        for scores in ([one, one], [other]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                self.summarize(scores)

    def test_settings_and_policies_cannot_be_mixed(self) -> None:
        for result in (
            grade_answer(r"\boxed{B}", metadata(), setting="agentic"),
            grade_answer("B", metadata(), setting="single_turn", answer_policy="boxed_or_letter_v1"),
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.summarize([result])

    def test_source_revisions_cannot_be_mixed(self) -> None:
        result = grade_answer(r"\boxed{B}", replace(metadata(), dataset_revision="c" * 40), setting="single_turn")
        with self.assertRaises(ValueError):
            self.summarize([result])

    def test_invalid_cohort_is_rejected(self) -> None:
        for identities in ((), ("same", "same"), ("",), (7,), "not-a-cohort"):
            with self.subTest(identities=identities), self.assertRaises(ValueError):
                self.summarize([], identities)

    def test_invalid_summary_protocol_is_rejected(self) -> None:
        for kwargs in (
            {"setting": "unknown", "dataset_revision": "a" * 40},
            {"setting": "single_turn", "dataset_revision": "main"},
            {"setting": "single_turn", "dataset_revision": "a" * 40, "answer_policy": "guess"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                aggregate_scores([], expected_task_ids=["synthetic/one"], **kwargs)

    def test_order_does_not_change_aggregate(self) -> None:
        scores = [
            grade_answer(r"\boxed{B}", metadata(), setting="single_turn"),
            grade_answer("", metadata("synthetic/two"), setting="single_turn"),
        ]
        self.assertEqual(self.summarize(scores), self.summarize(reversed(scores)))


if __name__ == "__main__":
    unittest.main()
