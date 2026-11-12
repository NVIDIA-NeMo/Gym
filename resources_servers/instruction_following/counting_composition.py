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
import re

from verifiable_instructions import instructions, instructions_util


# The spaced divider the instruction asks for, on a line of its own. Inline asterisks (for example
# emphasis inside a paragraph) are not dividers. Spaces or tabs around the stars and CRLF are allowed.
_DIVIDER_LINE = re.compile(r"^[ \t]*\*[ \t]+\*[ \t]+\*[ \t]*\r?$", re.MULTILINE)


class CountingCompositionChecker(instructions.CountingCompositionChecker):
    """`count:counting_composition` scored the way its instruction is worded.

    The upstream checker in verifiable-instructions splits paragraphs on `***` although the instruction
    asks for `* * *`, and counts the punctuation that ends each sentence as a word, so a response that
    follows the instruction always fails. This subclass keeps the upstream instruction text and arguments
    and only replaces `check_following`, with the same rules as the fix in allenai/open-instruct#1718.
    """

    def check_following(self, value: str) -> bool:
        args = self.get_instruction_args()
        paragraphs = _DIVIDER_LINE.split(value)
        # A divider before the first or after the last paragraph is tolerated; an empty paragraph
        # between two dividers is not.
        if paragraphs and not paragraphs[0].strip():
            paragraphs = paragraphs[1:]
        if paragraphs and not paragraphs[-1].strip():
            paragraphs = paragraphs[:-1]
        if len(paragraphs) != 3:
            return False

        for paragraph in paragraphs:
            sentences = instructions_util.split_into_sentences(paragraph)
            if len(sentences) != args["n_sent"]:
                return False
            for sentence in sentences:
                # Same tokenizer as upstream, minus tokens that are only punctuation.
                words = [
                    token
                    for token in instructions_util.nltk.word_tokenize(sentence)
                    if any(char.isalnum() for char in token)
                ]
                if len(words) != args["n_words"]:
                    return False
        return True
