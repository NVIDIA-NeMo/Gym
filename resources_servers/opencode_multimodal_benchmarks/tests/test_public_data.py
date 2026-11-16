# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib

import pytest

from benchmarks.opencode_multimodal_benchmarks import canonical
from resources_servers.opencode_multimodal_benchmarks.mcq import infer_option


@pytest.mark.parametrize(
    "prediction",
    [
        "The correct answer is **B**. Here is a long explanation of the image contents.",
        "The answer is C. A is incorrect, B is incorrect and so is D.",
    ],
)
def test_verbose_option_does_not_require_judge(prediction):
    assert infer_option(prediction, {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}) == (
        "B" if "**B**" in prediction else "C"
    )


def test_verbose_option_must_be_available_and_api_failure_not_salvaged():
    assert infer_option("The answer is D. More explanation follows here.", {"A": "cat", "B": "dog"}) is None
    assert infer_option("Failed to obtain answer via API. The answer is B.", {"A": "cat", "B": "dog"}) is None


@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K"])
def test_mcq_pins_ids_labels_cycles_and_media(monkeypatch, tmp_path, benchmark):
    image = tmp_path / (hashlib.sha256(b"original").hexdigest() + ".png")
    image.write_bytes(b"original")
    replacement = tmp_path / (hashlib.sha256(b"different").hexdigest() + ".png")
    replacement.write_bytes(b"different")
    task = {
        "index": "0",
        "question": "color?",
        "answer": "A",
        "options": {"A": "red", "B": "blue"},
        "category": "single",
        "l2_category": "",
        "cycle_category": "0",
        "split": "none",
        "images": [str(image)],
    }
    ref = {"tasks": {"0": canonical.fingerprint(canonical.mcq_identity(task))}}
    monkeypatch.setattr(canonical, "manifest", lambda _: ref)
    canonical.validate_mcq_tasks([task], benchmark)
    for field, value in [("answer", "B"), ("cycle_category", "1"), ("images", [str(replacement)]), ("index", "1")]:
        changed = copy.deepcopy(task)
        changed[field] = value
        with pytest.raises(ValueError):
            canonical.validate_mcq_tasks([changed], benchmark)
    with pytest.raises(ValueError):
        canonical.validate_mcq_tasks([task, task], benchmark)

    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="image checksum"):
        canonical.validate_mcq_tasks([task], benchmark)


@pytest.mark.parametrize(
    "prediction,expected",
    [("The answer is dog.", "B"), ("The correct answer is bird.", "C"), ("The answer is a tricky one", None)],
)
def test_prose_is_not_an_option_letter(prediction, expected):
    assert infer_option(prediction, {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}) == expected
