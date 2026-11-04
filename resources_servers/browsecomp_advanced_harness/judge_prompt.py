# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Derived from: https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/browsecomp_eval.py#L26-L45
# and https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/simpleqa_eval.py#L12-L104.
# We keep the output format of browsecomp.
#
# WHY THE GRADING INSTRUCTION IS A RULE LIST
#
# The original instruction asks the judge to explain whether the extracted answer is
# correct "focusing only on if there are meaningful differences". Whether a difference
# is meaningful is exactly the judgement being delegated, so surface-form mismatches --
# spelling, diacritics, transliteration -- are left to the judge's taste. The failure
# that motivated this: when the reference answer itself carries a typo, a response that
# spells the entity correctly gets marked wrong.
#
# The instruction below is five ordered rules, first match wins, and the judge must name
# the rule it used. Order is load-bearing. R1 (a requested component is missing) and R2
# (a different entity) are evaluated BEFORE R3 (same entity, surface-form difference), so
# accepting a spelling variant cannot smuggle in a partial or wrong answer.
#
# The `correct:` field is required: the response parser keys off `correct:\s*(yes|no)`,
# and a judge reply without that line does not parse and grades the row wrong. It is
# worded to report the verdict of the rule that fired rather than to re-decide it.

JUDGE_PROMPT_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Apply these rules IN ORDER. The FIRST rule that applies decides the verdict.

R1. If the extracted answer omits any component the [question] explicitly requested \
(a surname, a middle name, one part of a multi-part question, or a qualifier that \
belongs to the requested name), answer 'no'.

R2. If the extracted answer names a different real-world entity than \
[correct_answer], answer 'no'.

R3. If the extracted answer and [correct_answer] denote the SAME real-world entity, \
and they differ ONLY in surface form — spelling, diacritics, transliteration, case, \
punctuation, spacing, singular/plural, or an added honorific or parenthetical gloss \
that does not contradict [correct_answer] — answer 'yes'. This holds EVEN IF \
[correct_answer] contains the misspelling and the [response] gives the correct \
spelling of that same entity.

R4. For a numerical answer, answer 'yes' if it is within a small margin of error.

R5. Otherwise, if you are not confident that the two denote the same entity, \
answer 'no'.

Your judgement must be in the format and criteria specified below:

answer_type: The kind of value the [question] asks for. Name every component it \
requests, for example "full birth name of a person", "exact title of a publication", \
or "two parts: the politician's name AND the event".

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to \
extract from the response.

reasoning: Name the first rule above that applies, and explain in one or two \
sentences why it applies. Do not comment on any background to the problem, do not \
attempt to solve the problem, and do not argue for any answer different than \
[correct_answer].

correct: The verdict of the first rule that applied above — 'yes' or 'no'.

confidence: The extracted confidence score between 0% and 100% from \
[response]. Put 100 if there is no confidence score available."""
