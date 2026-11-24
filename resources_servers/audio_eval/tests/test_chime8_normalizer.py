# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored CHiME-8 text normalizer used by the default cpWER normalization."""

import pytest

from resources_servers.audio_eval._scoring.chime8_normalizer import (
    _EnglishNumberNormalizer,
    _EnglishReverseNumberNormalizer,
    _windowed,
)
from resources_servers.audio_eval._scoring.chime8_normalizer import (
    normalize as chime8_normalize,
)


def test_chime8_normalizer_is_idempotent_on_numbers_and_fillers() -> None:
    normalized = chime8_normalize("Hmm, I have 21 wi-fi routers")
    assert normalized == "i have twenty one wifi routers"
    assert chime8_normalize(normalized) == normalized


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("twenty one", "21"),
        ("one hundred and five dollars", "$105"),
        ("three point five percent", "3.5%"),
        ("two thousand twenty five", "2025"),
        ("fifty cents", "¢50"),
        ("twenty first", "21st"),
        ("two and a half", "2.5"),
        ("forty two thousand", "42000"),
        ("sixty percent", "60%"),
        ("minus seven degrees", "-7 degrees"),
        ("plus five", "+5"),
        ("fifteen dollars and fifty cents", "$15.50"),
        ("one billion", "1000000000"),
        ("the nineteen nineties", "the 1990s"),
        ("six thousand seven hundred eighty nine", "6789"),
        ("five hundred pounds and ten pence", "£500 and 10 pence"),
        ("twelve hundred", "1200"),
        ("two dozen eggs", "2 dozen eggs"),
        ("twenty twenty", "2020"),
        ("twenty twenties", "2020s"),
        ("one point five million", "1500000"),
        ("one and a half billion dollars", "$1500000000"),
        ("eighty first and ninetieth", "81st and 90th"),
        ("first second third", "1st 2nd 3rd"),
        ("zero point five", "0.5"),
        ("two hundred and fifty three thousand", "253000"),
        ("one two three", "123"),
        ("thirty cents", "¢30"),
    ],
)
def test_chime8_forward_number_normalizer_writes_spoken_numbers_as_digits(spoken: str, written: str) -> None:
    assert _EnglishNumberNormalizer()(spoken) == written


def test_chime8_reverse_number_normalizer_leaves_spoken_numbers_spoken() -> None:
    reverse = _EnglishReverseNumberNormalizer()
    assert reverse("twenty one") == "twenty one"
    assert reverse("$5") == "five dollars"
    assert reverse("50%") == "fifty percent"
    assert reverse("12th") == "twelfth"


def test_chime8_normalizer_applies_spelling_and_filler_rules() -> None:
    assert chime8_normalize("$5 and 50%") == "five dollars and fifty percent"
    assert chime8_normalize("Mr. Smith, the colour is grey") == "mister smith the color is gray"
    assert chime8_normalize("hmm uh okay") == "ok"
    assert chime8_normalize("it's 3 o'clock") == "it is three o clock"


def test_windowed_yields_every_window_and_rejects_nonpositive_sizes() -> None:
    assert list(_windowed(["a", "b", "c"], 2)) == [("a", "b"), ("b", "c")]
    with pytest.raises(ValueError):
        list(_windowed(["a"], 0))
