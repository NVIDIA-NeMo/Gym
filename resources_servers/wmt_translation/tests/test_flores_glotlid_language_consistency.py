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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _load_module_with_fake_model():
    fake_model = MagicMock()
    fake_model.get_output_matrix.return_value = np.zeros((6, 2))
    fake_model.get_labels.return_value = [
        "__label__eng_Latn",
        "__label__deu_Latn",
        "__label__acm_Arab",
        "__label__arb_Arab",
        "__label__ary_Arab",
        "__label__cmn_Hani",
    ]

    fake_fasttext = ModuleType("fasttext")
    fake_fasttext.load_model = MagicMock(return_value=fake_model)
    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.hf_hub_download = MagicMock(return_value="model_v3.bin")

    module_path = Path(__file__).parents[1] / "flores_glotlid_language_consistency.py"
    spec = importlib.util.spec_from_file_location("_flores_glotlid_language_consistency_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "fasttext": fake_fasttext,
            "huggingface_hub": fake_huggingface_hub,
        },
    ):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lid_module():
    return _load_module_with_fake_model()


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_text_returns_zero_without_inference(lid_module, monkeypatch, text: str) -> None:
    predict = MagicMock()
    monkeypatch.setattr(lid_module, "_predict_all_language_probabilities", predict)

    assert lid_module.flores_glotlid_language_consistency_score(text, "eng_Latn") == 0.0
    predict.assert_not_called()


def test_text_is_collapsed_to_one_line_before_inference(lid_module, monkeypatch) -> None:
    predict = MagicMock(return_value=np.array([0.8, 0.2, 0.0, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(lid_module, "_predict_all_language_probabilities", predict)

    score = lid_module.flores_glotlid_language_consistency_score(" Hello\n  world\t", "eng_Latn")

    assert score == pytest.approx(0.8)
    predict.assert_called_once_with("Hello world")


def test_unlisted_flores_code_uses_matching_glotlid_code(lid_module, monkeypatch) -> None:
    monkeypatch.setattr(
        lid_module,
        "_predict_all_language_probabilities",
        MagicMock(return_value=np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0])),
    )

    assert lid_module.flores_glotlid_language_consistency_score("English text", "eng_Latn") == pytest.approx(0.7)


def test_dialect_suffix_is_removed(lid_module, monkeypatch) -> None:
    monkeypatch.setattr(
        lid_module,
        "_predict_all_language_probabilities",
        MagicMock(return_value=np.array([0.65, 0.35, 0.0, 0.0, 0.0, 0.0])),
    )

    assert lid_module.flores_glotlid_language_consistency_score(
        "English text",
        "eng_Latn_dial1234",
    ) == pytest.approx(0.65)


def test_override_probabilities_are_summed(lid_module, monkeypatch) -> None:
    monkeypatch.setattr(
        lid_module,
        "_predict_all_language_probabilities",
        MagicMock(return_value=np.array([0.1, 0.1, 0.2, 0.3, 0.25, 0.05])),
    )

    assert lid_module.flores_glotlid_language_consistency_score(
        "Arabic text",
        "acm_Arab",
    ) == pytest.approx(0.75)


def test_renamed_override_uses_replacement_code(lid_module, monkeypatch) -> None:
    monkeypatch.setattr(
        lid_module,
        "_predict_all_language_probabilities",
        MagicMock(return_value=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])),
    )

    assert lid_module.flores_glotlid_language_consistency_score(
        "Chinese text",
        "cmn_Hans",
    ) == pytest.approx(0.95)


def test_missing_glotlid_label_returns_zero_without_inference(lid_module, monkeypatch) -> None:
    predict = MagicMock()
    monkeypatch.setattr(lid_module, "_predict_all_language_probabilities", predict)

    assert lid_module.flores_glotlid_language_consistency_score("text", "xxx_Latn") == 0.0
    predict.assert_not_called()


def test_partially_missing_override_sums_available_labels(lid_module, monkeypatch) -> None:
    monkeypatch.delitem(lid_module._LABEL_INDICES, "__label__ary_Arab")
    monkeypatch.setattr(
        lid_module,
        "_predict_all_language_probabilities",
        MagicMock(return_value=np.array([0.1, 0.1, 0.2, 0.3, 0.25, 0.05])),
    )

    assert lid_module.flores_glotlid_language_consistency_score(
        "Arabic text",
        "acm_Arab",
    ) == pytest.approx(0.5)


def test_all_language_probabilities_match_reference_softmax(lid_module, monkeypatch) -> None:
    monkeypatch.setattr(
        lid_module,
        "_OUTPUT_MATRIX",
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    )
    lid_module._MODEL.get_sentence_vector.return_value = np.array([2.0, 1.0])

    probabilities = lid_module._predict_all_language_probabilities("text")
    expected = np.exp(np.array([2.0, 1.0, -2.0]))
    expected /= expected.sum()

    assert probabilities == pytest.approx(expected)
    assert probabilities.sum() == pytest.approx(1.0)


def test_all_language_softmax_is_stable_for_large_logits(lid_module, monkeypatch) -> None:
    monkeypatch.setattr(
        lid_module,
        "_OUTPUT_MATRIX",
        np.array([[10_000.0], [9_999.0], [-10_000.0]]),
    )
    lid_module._MODEL.get_sentence_vector.return_value = np.array([1.0])

    probabilities = lid_module._predict_all_language_probabilities("text")

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert probabilities.sum() == pytest.approx(1.0)


def test_override_table_has_well_formed_unique_codes(lid_module) -> None:
    for flores_code, glotlid_codes in lid_module._LANG_CODE_OVERRIDES.items():
        assert len(flores_code.split("_")) == 2
        assert glotlid_codes
        assert len(glotlid_codes) == len(set(glotlid_codes))
        assert all(len(code.split("_")) == 2 for code in glotlid_codes)
