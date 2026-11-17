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


class CountUniqueChecker(instructions.CountUniqueChecker):
    """`count:count_unique` scored on words only, not on punctuation.

    The upstream checker in verifiable-instructions compares the raw `nltk.word_tokenize` tokens with
    their set, and punctuation marks are tokens, so two sentences that both end in a period repeat the
    `.` token and the response fails however well it follows the instruction. This subclass keeps the
    upstream instruction text and only replaces `check_following`, with the same rule as the fix in
    allenai/open-instruct#1718: a word is a token with at least one letter or digit. Case and contraction
    splitting are unchanged on purpose, so `Dogs` and `dogs` stay different words and `Don't` still
    tokenizes to `Do` and `n't`.
    """

    def check_following(self, value: str) -> bool:
        # Same tokenizer as upstream, minus tokens that are only punctuation.
        words = [
            token for token in instructions_util.nltk.word_tokenize(value) if any(char.isalnum() for char in token)
        ]
        return len(words) == len(set(words))
