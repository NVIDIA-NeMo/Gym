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
"""MRCR variant: Gemma-4-31B-it tokenizer with a 128K token cap.

Same data + grading as ``prepare.py``, but counts ``n_tokens`` with the
``google/gemma-4-31B-it`` HuggingFace tokenizer and drops samples whose
tokenized conversation exceeds 131072 tokens (the model's 128K context window).

Paired with ``config_gemma4_128k.yaml``. Requires HF auth / license acceptance
for the gated Google repo (``HF_TOKEN`` env or ``huggingface-cli login``).
"""

import os
from pathlib import Path

from benchmarks.mrcr.prepare import prepare as _prepare


TOKENIZER_NAME = os.environ.get("GEMMA4_TOKENIZER", "google/gemma-4-31B-it")
MAX_CONTEXT_TOKENS = 131072
OUTPUT_FPATH = Path(__file__).parent / "data" / "mrcr_gemma4_128k_benchmark.jsonl"
# Which n_needles buckets to keep (MRCR ships 2, 4, 8). Default 8-needle only; override e.g. MRCR_N_NEEDLES=2,4,8.
N_NEEDLES = tuple(int(x) for x in os.environ["MRCR_N_NEEDLES"].split(",")) if os.environ.get("MRCR_N_NEEDLES") else (8,)


def prepare() -> Path:
    return _prepare(
        tokenizer_name=TOKENIZER_NAME,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        output_fpath=OUTPUT_FPATH,
        n_needles=N_NEEDLES,
    )


if __name__ == "__main__":
    prepare()
