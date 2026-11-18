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
"""BEAM variant: Muse-Glimmer-30B tokenizer, questions filtered to a 131072-token prompt.

Same rows as ``prepare.py``, but a question is dropped when its full prompt
(whole chat plus question under the Muse chat template) exceeds 131072 tokens;
nothing is truncated. Only 100K chats qualify (260 of 1800 questions). Serve
with a window of at least 131072 + max output tokens (vLLM
``--max-model-len 147456`` with 16384 output tokens).

Uses ``prepare.py``'s default harness: the chat as a single-turn transcript
and the bare question (no upstream "answer only" NOTE).

Paired with ``config_muse_128k.yaml``.
"""

from pathlib import Path

from benchmarks.beam.prepare import prepare as _prepare


TOKENIZER_NAME = "meta-models/Muse-Glimmer-30B"
MAX_CONTEXT_TOKENS = 131072
OUTPUT_FPATH = Path(__file__).parent / "data" / "beam_muse_128k_benchmark.jsonl"


def prepare() -> Path:
    return _prepare(
        tokenizer_name=TOKENIZER_NAME,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        output_fpath=OUTPUT_FPATH,
    )


if __name__ == "__main__":
    prepare()
