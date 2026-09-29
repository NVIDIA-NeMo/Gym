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
"""MRCR variant: Nemotron-3-Super tokenizer with a 128k token cap.

Same data + grading as ``prepare.py``, but counts ``n_tokens`` with
the ``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`` HuggingFace
tokenizer and drops samples whose tokenized conversation exceeds
131072 tokens.

Paired with ``config_n3_128k.yaml``. Requires HF auth for the gated
NVIDIA repo (``HF_TOKEN`` env or ``huggingface-cli login``).
"""

from pathlib import Path

from benchmarks.mrcr.prepare import prepare as _prepare


TOKENIZER_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"  # pragma: allowlist secret
MAX_CONTEXT_TOKENS = 131072
OUTPUT_FPATH = Path(__file__).parent / "data" / "mrcr_n3_128k_benchmark.jsonl"


def prepare() -> Path:
    return _prepare(
        tokenizer_name=TOKENIZER_NAME,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        output_fpath=OUTPUT_FPATH,
    )


if __name__ == "__main__":
    prepare()
