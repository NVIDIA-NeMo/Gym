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

import fasttext
import numpy as np
from huggingface_hub import hf_hub_download


GLOTLID_REPO_ID = "cis-lmu/glotlid"
GLOTLID_MODEL_FILENAME = "model_v3.bin"


_MODEL_PATH = hf_hub_download(repo_id=GLOTLID_REPO_ID, filename=GLOTLID_MODEL_FILENAME)
_MODEL = fasttext.load_model(_MODEL_PATH)
_OUTPUT_MATRIX = _MODEL.get_output_matrix()
_LABEL_INDICES = {label: index for index, label in enumerate(_MODEL.get_labels())}


_LANG_CODE_OVERRIDES = {
    "acm_Arab": ["acm_Arab", "arb_Arab", "ary_Arab"],
    "acq_Arab": ["arb_Arab", "ars_Arab", "arz_Arab"],
    "aeb_Arab": ["aeb_Arab", "arb_Arab"],
    "apc_Arab": ["arb_Arab", "apc_Arab", "arz_Arab", "ajp_Arab"],
    "apd_Arab": ["arz_Arab", "arb_Arab", "ajp_Arab", "apc_Arab"],
    "ars_Arab": ["arb_Arab"],
    "arz_Arab": ["arz_Arab", "arb_Arab"],
    "bos_Latn": ["hrv_Latn", "bos_Latn"],
    "cmn_Hans": ["cmn_Hani"],
    "cmn_Hant": ["cmn_Hani"],
    "dgo_Deva": ["doi_Deva"],
    "dyu_Latn": ["dyu_Latn", "bam_Latn"],
    "kaa_Latn": ["kaa_Latn", "crh_Latn", "uig_Latn"],
    "khk_Mong": ["und_Mong"],
    "ktu_Latn": ["kng_Latn"],
    "lld_Latn": ["lld_Latn", "lmo_Latn"],
    "pes_Arab": ["fas_Arab"],
    "prs_Arab": ["fas_Arab"],
    "quy_Latn": ["quy_Latn", "quz_Latn"],
    "wuu_Hans": ["wuu_Hani"],
    "yue_Hant": ["yue_Hani"],
    "zgh_Tfng": ["zgh_Tfng", "taq_Tfng"],
}


def _predict_all_language_probabilities(text: str) -> np.ndarray:
    """
    Return normalized GlotLID probabilities aligned with `_LABEL_INDICES`.

    This follows GlotLID's documented all-language inference path to produce
    properly normalized output.
    """
    sentence_vector = _MODEL.get_sentence_vector(text)
    logits = _OUTPUT_MATRIX @ sentence_vector
    unnormalized_probabilities = np.exp(logits.astype(np.float64) - np.max(logits))
    return unnormalized_probabilities / unnormalized_probabilities.sum()


def flores_glotlid_language_consistency_score(text: str, expected_language_code: str) -> float:
    """
    Return the probability GlotLID assigns to the expected FLORES+ language.

    FLORES+ dialect suffixes are removed because GlotLID does not represent
    them. For known ambiguous or differently named languages, probabilities
    for all configured GlotLID codes are summed.
    """
    text = " ".join(text.split())
    if not text:
        return 0.0

    flores_code = "_".join(expected_language_code.split("_")[:2])
    glotlid_codes = _LANG_CODE_OVERRIDES.get(flores_code, [flores_code])
    expected_indices = [
        _LABEL_INDICES[label] for code in glotlid_codes if (label := f"__label__{code}") in _LABEL_INDICES
    ]
    if not expected_indices:
        return 0.0

    probabilities = _predict_all_language_probabilities(text)
    return float(probabilities[expected_indices].sum())
