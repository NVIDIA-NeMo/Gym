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

"""
Validate the FLORES+ GlotLID language-consistency scorer against FLORES+ text.

For every FLORES+ language config, score each sentence in devtest when
available, otherwise dev, and raise an error if the average score is too low.
This check is too expensive for the default unit-test suite.
"""

import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


_WMT_TRANSLATION_DIR = Path(__file__).resolve().parents[1]
if str(_WMT_TRANSLATION_DIR) not in sys.path:
    sys.path.insert(0, str(_WMT_TRANSLATION_DIR))

from flores_glotlid_language_consistency import flores_glotlid_language_consistency_score


FLORES_REPO_ID = "openlanguagedata/flores_plus"
MINIMUM_AVERAGE_LANGUAGE_CONSISTENCY_SCORE = 0.60


def _validate_language_consistency_scores() -> None:
    flores_configs = [config for config in sorted(get_dataset_config_names(FLORES_REPO_ID)) if config != "default"]

    for flores_config in flores_configs:
        splits = load_dataset(FLORES_REPO_ID, flores_config)
        split = "devtest" if "devtest" in splits else "dev"
        scores = [flores_glotlid_language_consistency_score(text, flores_config) for text in splits[split]["text"]]
        if not scores:
            raise AssertionError(f"{flores_config}: no sentences found in {split}")

        average_score = sum(scores) / len(scores)
        if average_score < MINIMUM_AVERAGE_LANGUAGE_CONSISTENCY_SCORE:
            raise AssertionError(
                f"{flores_config}: average language-consistency score {average_score:.5f} "
                f"is below minimum {MINIMUM_AVERAGE_LANGUAGE_CONSISTENCY_SCORE:.2f}"
            )

        print(f"{flores_config} ({split}): pass ({average_score:.5f})")


if __name__ == "__main__":
    _validate_language_consistency_scores()
