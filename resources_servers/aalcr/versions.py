# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream AA-LCR versions: dataset revision and its matching judge protocol.

Upstream versions the answer keys and the scoring protocol *together*, so they are selected together
here. Mixing them silently mis-grades: several v1.1 keys carry grading instructions addressed to the
judge (question 26 offers two acceptable values, question 78 permits either alphabetical ordering), and
nine more are notation rewrites (`0.16` to `16%`) whose equivalence only the v1.1 system prompt declares.
Graded by the v1.0 protocol those keys produce a plausible but meaningless number, so `revision` is
deliberately not exposed on its own -- callers pick a `version`, and the protocol follows it.
"""

from dataclasses import dataclass
from typing import Dict, Optional


# Verbatim from the upstream dataset card ("Scoring Approach") at revision REVISION_V1_1.
_JUDGE_SYSTEM_PROMPT_V1_1 = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""

_JUDGE_USER_PROMPT_V1_1 = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""

# The v1.0 dataset card documents this compact prompt, and it is what this server has always sent. The
# v1.1 card retroactively describes v1.0 as having used the delimited blocks above; it did not, so this
# is kept as-is to reproduce results actually produced under v1.0 rather than the later description.
_JUDGE_USER_PROMPT_V1_0 = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT."""


@dataclass(frozen=True)
class AalcrVersion:
    """A published AA-LCR version: the data and the protocol that grades it."""

    revision: str
    judge_user_prompt: str
    judge_system_prompt: Optional[str]
    judge_replies_json: bool


VERSIONS: Dict[str, AalcrVersion] = {
    "1.0": AalcrVersion(
        revision="bdae010bbce259820c0e34c1d7cce210d966fb75",  # pragma: allowlist secret
        judge_user_prompt=_JUDGE_USER_PROMPT_V1_0,
        judge_system_prompt=None,
        judge_replies_json=False,
    ),
    "1.1": AalcrVersion(
        revision="9a77ef56b717057ade24ceab4d273712a0b4f19e",  # pragma: allowlist secret
        judge_user_prompt=_JUDGE_USER_PROMPT_V1_1,
        judge_system_prompt=_JUDGE_SYSTEM_PROMPT_V1_1,
        judge_replies_json=True,
    ),
}

DEFAULT_VERSION = "1.1"


def get_version(version: str) -> AalcrVersion:
    if version not in VERSIONS:
        raise ValueError(f"Unknown AA-LCR version {version!r}. Known versions: {sorted(VERSIONS)}")
    return VERSIONS[version]
