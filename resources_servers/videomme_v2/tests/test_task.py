# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-only fixtures: no benchmark questions, model, decoder or credentials."""

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from resources_servers.videomme_v2.task import (
    VerifierMetadata,
    VideoQuestion,
    prepare_agentic,
    prepare_single_turn,
)


def question() -> VideoQuestion:
    """Return an invented question, never a sample from a benchmark split."""
    return VideoQuestion(
        task_id="synthetic/one",
        dataset_source="https://example.invalid/synthetic-dataset",
        dataset_revision="a" * 40,
        video_id="synthetic-video",
        video_source="https://example.invalid/private-origin.mp4",
        video_sha256="b" * 64,
        question="Which shape moves to the right?",
        options=("A circle", "A triangle", "A square"),
        level=1,
        group_id="synthetic-group",
    )


class TestQuestionContract(unittest.TestCase):
    def test_question_is_immutable_and_contains_no_gold(self) -> None:
        item = question()
        self.assertFalse(hasattr(item, "expected_answer"))
        with self.assertRaises(FrozenInstanceError):
            item.task_id = "changed"

    def test_required_strings_are_validated(self) -> None:
        for field in ("task_id", "dataset_source", "video_id", "video_source", "question"):
            for value in ("", " \n", "bad\x00value", None, 7):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(question(), **{field: value})

    def test_revision_must_be_pinned(self) -> None:
        for value in ("main", "a" * 39, "A" * 40, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(question(), dataset_revision=value)

    def test_video_hash_must_be_exact(self) -> None:
        for value in ("b" * 63, "B" * 64, "z" * 64, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(question(), video_sha256=value)

    def test_options_require_two_to_twenty_six_immutable_choices(self) -> None:
        for value in ((), ("one",), tuple(str(x) for x in range(27)), ["one", "two"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(question(), options=value)
        self.assertEqual(len(replace(question(), options=tuple(str(x) for x in range(26))).options), 26)

    def test_empty_duplicate_and_nonstr_choices_are_rejected(self) -> None:
        for value in (("one", " "), ("one", " one "), ("one", 2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(question(), options=value)

    def test_level_and_group_are_validated(self) -> None:
        for value in (0, -1, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(question(), level=value)
        with self.assertRaises(ValueError):
            replace(question(), group_id="")
        self.assertIsNone(replace(question(), level=None, group_id=None).level)

    def test_question_text_is_not_rewritten(self) -> None:
        item = replace(question(), question="  Preserve this text?  ", options=(" one ", "two"))
        self.assertEqual(item.question, "  Preserve this text?  ")
        self.assertEqual(item.options[0], " one ")


class TestPrivateMetadata(unittest.TestCase):
    def test_gold_is_bound_to_choice_range_and_provenance(self) -> None:
        item = question()
        metadata = VerifierMetadata.from_question(item, expected_answer="B")
        self.assertEqual((metadata.task_id, metadata.video_sha256), (item.task_id, item.video_sha256))
        self.assertEqual(metadata.dataset_revision, item.dataset_revision)
        self.assertEqual(metadata.option_count, 3)
        self.assertEqual(metadata.expected_answer, "B")

    def test_invalid_gold_is_not_normalized_or_guessed(self) -> None:
        for value in ("", "AB", " B", "b", "H", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                VerifierMetadata.from_question(question(), expected_answer=value)

    def test_metadata_validates_its_own_boundary(self) -> None:
        metadata = VerifierMetadata.from_question(question(), expected_answer="B")
        for values in (
            {"option_count": True},
            {"option_count": 1},
            {"option_count": 27},
            {"task_id": ""},
            {"dataset_revision": "latest"},
            {"video_sha256": "bad"},
            {"level": False},
            {"group_id": ""},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(metadata, **values)


class TestModelProjection(unittest.TestCase):
    def test_single_turn_has_one_video_and_one_user_turn(self) -> None:
        row = prepare_single_turn(question(), expected_answer="B", video_url="data:video/mp4;base64,c3ludGhldGlj")
        messages = row["responses_create_params"]["input"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual([x["type"] for x in messages[0]["content"]], ["input_video", "input_text"])
        self.assertEqual(row["setting"], "single_turn")

    def test_agentic_has_only_question_and_bound_path(self) -> None:
        row = prepare_agentic(question(), expected_answer="B", video_path="/testbed/videos/synthetic.mp4")
        messages = row["responses_create_params"]["input"]
        self.assertEqual(len(messages), 1)
        self.assertEqual([x["type"] for x in messages[0]["content"]], ["input_text"])
        text = messages[0]["content"][0]["text"]
        self.assertIn("Video: /testbed/videos/synthetic.mp4", text)
        self.assertIn("B. A triangle", text)
        self.assertNotIn("watch_video", text)
        self.assertNotIn("view_image", text)
        self.assertEqual(row["setting"], "agentic")

    def test_changing_gold_does_not_change_either_model_input(self) -> None:
        for prepare, kwargs in (
            (prepare_single_turn, {"video_url": "https://example.invalid/prepared.mp4"}),
            (prepare_agentic, {"video_path": "/testbed/videos/synthetic.mp4"}),
        ):
            with self.subTest(prepare=prepare.__name__):
                first = prepare(question(), expected_answer="B", **kwargs)
                second = prepare(question(), expected_answer="C", **kwargs)
                self.assertEqual(first["responses_create_params"], second["responses_create_params"])
                self.assertNotEqual(first["verifier_metadata"], second["verifier_metadata"])

    def test_provenance_and_gold_are_outside_model_allowlist(self) -> None:
        row = prepare_agentic(question(), expected_answer="B", video_path="video/synthetic.mp4")
        model = row["responses_create_params"]
        self.assertEqual(set(model), {"input"})
        self.assertEqual(set(model["input"][0]), {"role", "content"})
        encoded = json.dumps(model)
        for private_value in (
            "expected_answer",
            "verifier_metadata",
            question().video_source,
            question().dataset_source,
            question().dataset_revision,
            question().video_sha256,
        ):
            self.assertNotIn(private_value, encoded)
        self.assertEqual(row["verifier_metadata"]["expected_answer"], "B")

    def test_prepared_rows_are_json_serializable(self) -> None:
        for row in (
            prepare_single_turn(question(), expected_answer="B", video_url="https://example.invalid/clip.mp4"),
            prepare_agentic(question(), expected_answer="B", video_path="video/synthetic.mp4"),
        ):
            self.assertEqual(json.loads(json.dumps(row)), row)
            self.assertEqual(VerifierMetadata(**row["verifier_metadata"]).expected_answer, "B")

    def test_empty_media_and_multiline_path_are_rejected(self) -> None:
        for value in ("", None, "\x00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_single_turn(question(), expected_answer="B", video_url=value)
        for value in ("", "a\nb", "a\rb", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_agentic(question(), expected_answer="B", video_path=value)


if __name__ == "__main__":
    unittest.main()
