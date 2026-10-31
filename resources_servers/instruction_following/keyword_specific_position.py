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
from verifiable_instructions import instructions, instructions_util


class KeywordSpecificPositionChecker(instructions.KeywordSpecificPositionChecker):
    """`keywords:keyword_specific_position` counted in words only, not in punctuation.

    The upstream checker in verifiable-instructions takes the m-th raw `nltk.word_tokenize` token of the
    n-th sentence, and punctuation marks are tokens, so a comma, quote or colon before the keyword shifts
    its position: with keyword `cat`, n=1 and m=3, `Yes, the cat sat.` fails although `cat` is the third
    word. This subclass keeps the upstream instruction text and only replaces `check_following`, with the
    same rule as `count_unique.py`, which allenai/IFBench#25 also applied to this checker: a word is a token
    with at least one letter or digit. Case, contraction splitting and sentence splitting are unchanged on
    purpose, so `Cat` does not match the keyword `cat`, `It's` still tokenizes to `It` and `'s`, and the
    sentences are still the ones `instructions_util.split_into_sentences` returns.
    """

    def check_following(self, value: str) -> bool:
        sentences = instructions_util.split_into_sentences(value)
        if len(sentences) < self._n:
            return False
        # Same tokenizer as upstream, minus tokens that are only punctuation.
        words = [
            token
            for token in instructions_util.nltk.word_tokenize(sentences[self._n - 1])
            if any(char.isalnum() for char in token)
        ]
        if len(words) < self._m:
            return False
        return words[self._m - 1] == self._keyword
