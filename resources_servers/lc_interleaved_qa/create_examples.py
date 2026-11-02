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

"""Write ``data/example.jsonl``: a few hand-written interleaved long-context QA tasks.

The example prefixes are short so the file stays readable and the smoke test is cheap;
real tasks carry a prefix of tens of thousands of characters. Run with:

    python resources_servers/lc_interleaved_qa/create_examples.py
"""

import json
from pathlib import Path


AGENT_REF = {"type": "responses_api_agents", "name": "lc_interleaved_qa_simple_agent"}

PREAMBLE = (
    "The input is a document presented in parts; questions are asked between the parts. Your job is to locate the "
    "parts of the document each question depends on and derive the answer from them.\n\n"
)

CHUNK_1 = (
    "Annual Report 2015\n\n"
    "Item 1. Business\n\n"
    "Meridian Capital Group is a business development company incorporated in Delaware in 1997. Our asset "
    "management business is conducted through our wholly-owned portfolio company, Meridian Asset Management "
    '("MAM"). As of December 31, 2015, MAM managed the following funds:\n\n'
    "  * Meridian Senior Loan Fund I\n"
    "  * Meridian Senior Loan Fund II\n"
    "  * Meridian Opportunity Fund\n\n"
    "Total assets under management were $4.1 billion at year end, compared with $3.6 billion a year earlier.\n"
)

CHUNK_2 = (
    "Qualifying Assets\n\n"
    "As a business development company, we may not acquire any asset other than certain qualifying assets "
    "described in the 1940 Act, unless, at the time the acquisition is made, the value of such qualifying assets "
    "represents at least 70% of the value of our total assets. Meridian Opportunity Fund invests primarily in "
    "senior floating rate loans to middle-market companies.\n"
)

ANSWER_INSTRUCTION = (
    "\n\nIdentify the information the question needs, work out the answer from it, and then give the answer."
)


def _task(
    *,
    turns: list[tuple[str, str]],
    question: str,
    family: str,
    evidence: list[str],
    answer_key: str | None = None,
    constraint: dict | None = None,
    verbatim: bool = False,
) -> dict:
    """Build one row.

    ``turns`` is the (chunk, question asked after it) sequence. The final turn is the one
    being verified, so its question is always ``question``: whatever the caller passed
    there is replaced. Letting the two drift apart asks the model one thing and grades
    another, which is how two of these three examples came to score 0 in the smoke run.
    """
    turns = [*turns[:-1], (turns[-1][0], question)]
    messages = []
    prefix_parts = []
    for position, (chunk, asked) in enumerate(turns):
        body = (PREAMBLE if position == 0 else "") + chunk + f"\n\nQuestion: {asked}" + ANSWER_INSTRUCTION
        messages.append({"role": "user", "content": body})
        prefix_parts.append(chunk)
    prefix = "".join(prefix_parts)
    row = {
        "responses_create_params": {"input": messages},
        "question": question,
        "prefix": prefix,
        "family": family,
        "evidence": [{"quote": quote} for quote in evidence],
        "verbatim": verbatim,
        "agent_ref": AGENT_REF,
    }
    if answer_key is not None:
        row["answer_key"] = answer_key
    if constraint is not None:
        row["constraint"] = constraint
    return row


def build_examples() -> list[dict]:
    first_turn_question = (
        "List the funds managed by MAM as of December 31, 2015, in the order they are presented in the report."
    )
    return [
        # One chunk read; an enumeration question answerable from it alone.
        _task(
            turns=[(CHUNK_1, first_turn_question)],
            question=first_turn_question,
            family="enumeration",
            evidence=["MAM managed the following funds"],
            answer_key="Meridian Senior Loan Fund I, Meridian Senior Loan Fund II, Meridian Opportunity Fund",
        ),
        # Two chunks read; the answer needs a fact from each, so it only works once the
        # second chunk has arrived.
        _task(
            turns=[(CHUNK_1, first_turn_question), (CHUNK_2, "")],
            question=(
                "Meridian Capital Group holds a direct investment in Meridian Opportunity Fund through MAM. What "
                "type of assets does that fund invest in?"
            ),
            family="multi_hop",
            evidence=[
                "our wholly-owned portfolio company, Meridian Asset Management",
                "Meridian Opportunity Fund invests primarily in senior floating rate loans",
            ],
            answer_key="senior floating rate loans",
        ),
        # A response requirement the verifier can check without a judge.
        _task(
            turns=[(CHUNK_1, first_turn_question)],
            question=(
                "What were total assets under management at the end of 2015? Answer with the figure only, in "
                "lowercase."
            ),
            family="recall",
            evidence=["Total assets under management were $4.1 billion at year end"],
            answer_key="$4.1 billion",
            constraint={"kind": "lowercase"},
        ),
    ]


def main() -> None:
    destination = Path(__file__).parent / "data/example.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for row in build_examples():
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
