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

"""Unit tests for VerifIF keyword validators."""

from resources_servers.verifif.vif_validators.validator import validate_instruction


def test_italian_keyword_frequency_matches_multiword_phrase() -> None:
    result = validate_instruction(
        "Il pretesto segnaposto di contatto e un testo segnaposto di contatto.",
        "keywords:frequency",
        {"keyword": "testo segnaposto di contatto", "relation": "equal to", "frequency": 1},
        language="it",
    )

    assert result == (True, "No error")


def test_italian_forbidden_words_matches_multiword_phrase() -> None:
    result = validate_instruction(
        "La risposta e limitato a tre elementi.",
        "keywords:forbidden_words",
        {"forbidden_words": ["limitato a"]},
        language="it",
    )

    assert result == (False, "Forbidden words found: ['limitato a']")


def test_italian_keyword_frequency_preserves_elision_matching() -> None:
    result = validate_instruction(
        "L'amico risponde.",
        "keywords:frequency",
        {"keyword": "amico", "relation": "equal to", "frequency": 1},
        language="it",
    )

    assert result == (True, "No error")


def test_german_forbidden_words_accepts_accented_keyword_when_absent() -> None:
    result = validate_instruction(
        "Der Text enthalt keine relevanten Einschrankungen.",
        "keywords:forbidden_words",
        {"forbidden_words": ["beschränkt auf"]},
        language="de",
    )

    assert result == (True, "No error")


def test_german_forbidden_words_detects_accented_phrase() -> None:
    result = validate_instruction(
        "Der Inhalt ist beschränkt auf drei Abschnitte.",
        "keywords:forbidden_words",
        {"forbidden_words": ["beschränkt auf"]},
        language="de",
    )

    assert result == (False, "Forbidden words found: ['beschränkt auf']")


def test_german_forbidden_words_preserves_ascii_hyphenated_keywords() -> None:
    result = validate_instruction(
        "Die E-Mail ist gueltig.",
        "keywords:forbidden_words",
        {"forbidden_words": ["E-Mail"]},
        language="de",
    )

    assert result == (False, "Forbidden words found: ['E-Mail']")


def test_german_forbidden_words_rejects_leading_hyphen() -> None:
    valid, reason = validate_instruction(
        "Ein harmloser Text.",
        "keywords:forbidden_words",
        {"forbidden_words": ["-beschränkt"]},
        language="de",
    )

    assert not valid
    assert reason.startswith("Validation error: Invalid token")
