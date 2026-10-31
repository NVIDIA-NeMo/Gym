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
"""BEAM variant: full set, Nemotron-3 tokenizer, questions filtered to a 1048576-token prompt.

Builds all three splits (100K / 500K / 1M; 90 chats, 1800 questions) with
``prepare.py``'s default harness (single-turn transcript, bare question) and
drops a question when its full prompt -- whole chat plus question under the
Nemotron-3 chat template -- exceeds 1048576 tokens. Nothing is truncated; every
100K and 500K question fits, and 440 of the 700 1M questions do. Serve each
policy with a window of at least its own token count of the longest prompt plus
the max output tokens.

Paired with ``config_n3_1m.yaml``. Requires HF access to the NVIDIA repo
(``HF_TOKEN`` env or ``huggingface-cli login``).
"""

from pathlib import Path

from benchmarks.beam.prepare import prepare as _prepare


TOKENIZER_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"  # pragma: allowlist secret
MAX_CONTEXT_TOKENS = 1048576
OUTPUT_FPATH = Path(__file__).parent / "data" / "beam_n3_1m_benchmark.jsonl"


def prepare() -> Path:
    return _prepare(
        tokenizer_name=TOKENIZER_NAME,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        output_fpath=OUTPUT_FPATH,
    )


if __name__ == "__main__":
    prepare()
