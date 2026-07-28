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
Build a mapping from FLORES+ to GlotLID language codes,
using a data-driven approach to overcome both inconsistencies
in language-code naming and languages/dialects that GlotLID is not
able to distinguish accurately.
"""

from collections import Counter
from pprint import pprint

import fasttext
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download


FLORES_REPO_ID = "openlanguagedata/flores_plus"
GLOTLID_REPO_ID = "cis-lmu/glotlid"
GLOTLID_MODEL_FILENAME = "model_v3.bin"
MASS_THRESHOLD = 0.80


model_path = hf_hub_download(repo_id=GLOTLID_REPO_ID, filename=GLOTLID_MODEL_FILENAME)
model = fasttext.load_model(model_path)
counts_by_flores_code: dict[str, Counter[str]] = {}

for flores_config in sorted(get_dataset_config_names(FLORES_REPO_ID)):
    # "default" combines every language dataset and is not itself a language code.
    if flores_config == "default":
        continue

    flores_code = "_".join(flores_config.split("_")[:2]) # drop optional dialect
    label_counts = counts_by_flores_code.setdefault(flores_code, Counter())
    splits = load_dataset(FLORES_REPO_ID, flores_config)
    dataset = splits["devtest" if "devtest" in splits else "dev"] # naming is inconsistent

    for text in dataset["text"]:
        text = " ".join(text.split())
        # The internal binding avoids fasttext-wheel's NumPy 2-incompatible public wrapper.
        _probability, label = model.f.predict(text + "\n", 1, 0.0, "strict")[0]
        label_counts[label.removeprefix("__label__")] += 1

exceptions = {}
for flores_code, label_counts in sorted(counts_by_flores_code.items()):
    # Here mass is the fraction of sentence-level top-1 labels, not model confidence.
    top_codes = []
    cumulative_count = 0
    for glotlid_code, count in label_counts.most_common():
        top_codes.append(glotlid_code)
        cumulative_count += count
        if cumulative_count / label_counts.total() > MASS_THRESHOLD:
            break

    if top_codes != [flores_code]: # only track exceptions
        exceptions[flores_code] = top_codes

pprint(exceptions, sort_dicts=True)
