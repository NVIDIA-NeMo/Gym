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
"""The MM-BrowseComp judge prompt and its parser.

The prompt is upstream's ``construct_judge_prompt`` from src/eval.py at
https://github.com/MMBrowseComp/MM-BrowseComp/blob/cf82a2390aa7a7ed3d96f72969d22548eff3e101/src/eval.py
(MIT), kept byte for byte, typos included, so a score stays comparable with the paper's.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# Upstream judges only the last 25,000 characters of the answer.
ANSWER_TAIL_CHARS = 25000


def build_judge_prompt(question: str, answer: str, reference_answer: str, checklist: List[str]) -> str:
    """Upstream's judge prompt. ``question`` is the decrypted question; like upstream, only the text
    after its last "Question: " reaches the judge."""
    question = question.split("Question: ")[-1]
    answer = answer[-ANSWER_TAIL_CHARS:]
    intro = (
        "You are an AI evaluator. Your task is to evaluate the quality of an answer. "
        "I will provide you with the user's question, the reference answer (ground truth), "
        "a checklist, and the answer to be evaluated."
    )
    question_section = f"--- USER QUESTION ---\n{question}"
    reference_answer_section = (
        f"\n\n--- REFERENCE ANSWER (Ground Truth) ---\n{reference_answer}\n"
        "This reference answer is considered the correct and ideal response content-wise."
    )
    total = len(checklist)
    checklist_items = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(checklist))
    checklist_section = f"\n\n--- REFERENCE CHECKLIST ---\n{checklist_items}"
    checklist_instruction = (
        f"1. Checklist Score: First, determine how many of the {total} items in the 'REFERENCE CHECKLIST' "
        f"have been correctly and completely addressed by the 'MODEL'S GENERATED ANSWER TO EVALUATE'.\n"
        f"lease remember that for any item in the checklist, the model’s generated answer to evaluate must "
        f"fully comply in order for that item to be considered complete.\n"
        f"   State this as 'CHECKLIST_SCORE: [correct_items]/{total}' "
        f"(e.g., CHECKLIST_SCORE: {max(0, total - 1 if total > 0 else 0)}/{total})."
        f"2. Checklist Result Vector:"
        f"Next, please provide a 0-1 vector to indicate whether each checklist item passed."
        f"Output the vector in the order of the items in the checklist, for example, [1,0,1]."
        f"'1' means the item is 'fully satisfied,' and '0' means 'not fully satisfied.'"
        f"If there is no checklist for this question, please return N/A."
        f"Output in the format 'CHECKLIST_RESULT: ...' (e.g., CHECKLIST_RESULT: [1,0,1])."
    )
    answer_section = f"\n\n--- MODEL'S GENERATED ANSWER TO EVALUATE ---\n{answer}"
    correctness_instruction = (
        "2. Overall Correctness: Next, you need to judge whether the 'MODEL'S GENERATED ANSWER TO EVALUATE' "
        "is consistent with the 'REFERENCE ANSWER (Ground Truth)' in terms of its core content and information.\n"
        "   - Content consistency is key. Differences in formatting or minor wording variations are acceptable "
        "as long as the essential information and meaning conveyed by the generated answer align with the "
        "reference answer.\n"
        "   - If the generated answer accurately reflects the information in the reference answer, it should be "
        "considered correct.\n"
        "   State your judgment as 'OVERALL_CORRECTNESS: [YES/NO]' (e.g., OVERALL_CORRECTNESS: YES)."
    )
    return f"""{intro}

{question_section}{reference_answer_section}{checklist_section}{answer_section}

--- EVALUATION INSTRUCTIONS ---
Please provide your evaluation strictly in the following format on separate lines:
{checklist_instruction}
{correctness_instruction}

Example 1 (Checklist provided, generated answer consistent with reference, some checklist items missed):
CHECKLIST_SCORE: 1/3
CHECKLIST_RESULT: [1,0,1]
OVERALL_CORRECTNESS: YES

Example 2 (Checklist provided, generated answer NOT consistent with reference, even if checklist is met):
CHECKLIST_SCORE: 4/4
CHECKLIST_RESULT: [1,1,1,1]
OVERALL_CORRECTNESS: NO

Provide only these formatted lines (CHECKLIST_SCORE, CHECKLIST_RESULT, OVERALL_CORRECTNESS) as your response.
"""


@dataclass(frozen=True)
class JudgeVerdict:
    correct: bool
    # None when the judge gave no parseable "k/N" with N > 0 (upstream drops such rows from AVG CS).
    checklist_correct: Optional[int]
    checklist_total: Optional[int]

    @property
    def checklist_score(self) -> Optional[float]:
        if self.checklist_total is None:
            return None
        return min(self.checklist_correct, self.checklist_total) / self.checklist_total

    @property
    def strict_correct(self) -> bool:
        """Upstream's SA: correct, and the judge's own CHECKLIST_SCORE is full marks."""
        return self.correct and self.checklist_total is not None and self.checklist_correct == self.checklist_total


_CORRECTNESS_RE = re.compile(r"OVERALL_CORRECTNESS:\s*\**\s*(YES|NO)\b", re.IGNORECASE)
_CHECKLIST_RE = re.compile(r"CHECKLIST_SCORE:\s*\**\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def parse_judge_response(text: str) -> Optional[JudgeVerdict]:
    """None when there is no OVERALL_CORRECTNESS verdict. Upstream takes the first match; the last one is
    used here, as in the BrowseComp judge, so an echoed example line cannot decide the grade."""
    verdicts = _CORRECTNESS_RE.findall(text or "")
    if not verdicts:
        return None
    scores = _CHECKLIST_RE.findall(text)
    correct, total = (int(scores[-1][0]), int(scores[-1][1])) if scores else (None, None)
    if not total:
        correct, total = None, None
    return JudgeVerdict(correct=verdicts[-1].upper() == "YES", checklist_correct=correct, checklist_total=total)
