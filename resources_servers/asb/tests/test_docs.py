# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deviation list must not drift between the two documents that publish it.

METRICS.md and README.md each listed four deviations once -- but different fours, because a
deviation discovered mid-run was added to one file in place of an existing entry rather than
alongside it. Each file looked internally consistent, so nothing caught it until a consumer
diffed them. These tests make that failure loud.

Reports are generated from METRICS.md, so a missing entry there ships a package that
understates how far the run departs from upstream.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def flat(path: Path) -> str:
    """Document text with runs of whitespace collapsed.

    Markdown wraps prose at the column limit, so a phrase like "Observation Prompt
    Injection" can straddle a newline. Searching the raw text for it then fails on a
    document that says exactly the right thing.
    """
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).lower()


BENCH = Path(__file__).resolve().parents[3] / "benchmarks" / "asb"
METRICS = BENCH / "METRICS.md"
README = BENCH / "README.md"

NUMBER_WORDS = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
    "Six": 6,
    "Seven": 7,
    "Eight": 8,
    "Nine": 9,
    "Ten": 10,
}


def metrics_deviations() -> list[str]:
    """Headings of the numbered deviations in METRICS.md."""
    return re.findall(r"^### \d+\.\s+(.+)$", METRICS.read_text(encoding="utf-8"), re.MULTILINE)


def readme_deviations() -> list[str]:
    """Bold titles of the numbered deviations in README.md's Deviations section."""
    text = README.read_text(encoding="utf-8")
    section = text.split("## Deviations", 1)[1]
    section = section.split("\n## ", 1)[0]
    return re.findall(r"^\d+\.\s+\*\*(.+?)\.?\*\*", section, re.MULTILINE)


def test_both_documents_list_the_same_number_of_deviations():
    assert len(metrics_deviations()) == len(readme_deviations()), (
        f"METRICS.md lists {metrics_deviations()}, README.md lists {readme_deviations()}. "
        "They must describe the same set."
    )


def test_metrics_header_count_matches_its_own_list():
    """The prose count ('Five, all forced') must match the headings beneath it."""
    text = METRICS.read_text(encoding="utf-8")
    match = re.search(r"^(\w+), all forced", text, re.MULTILINE)
    assert match, "METRICS.md lost its '<N>, all forced' deviation header"
    stated = NUMBER_WORDS[match.group(1)]
    assert stated == len(metrics_deviations()), (
        f"METRICS.md says '{match.group(1)}' but lists {len(metrics_deviations())} deviations"
    )


def test_readme_header_count_matches_its_own_list():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Deviations", 1)[1]
    match = re.search(r"^(\w+), all forced", section, re.MULTILINE)
    assert match, "README.md lost its '<N>, all forced' deviation header"
    assert NUMBER_WORDS[match.group(1)] == len(readme_deviations())


@pytest.mark.parametrize(
    "topic",
    [
        # Each deviation must be discoverable by the thing a reader would search for.
        ("4,096", "output cap"),
        ("ada-002", "memory ranking"),
        ("json.loads", "plan-parse salvage"),
        ("system message", "system-message merge"),
        ("parameters", "tool schema envelope"),
    ],
)
def test_every_deviation_is_substantively_described_in_metrics(topic):
    needle, label = topic
    assert needle.lower() in flat(METRICS), (
        f"METRICS.md no longer describes the {label} deviation (searched for {needle!r})"
    )


def test_system_message_merge_is_documented_because_it_changes_model_input():
    """The merge alters what every model reads, so silence about it is a real omission.

    This is the specific entry that went missing: it was added to README.md in place of the
    tool-schema entry rather than alongside it, leaving METRICS.md -- the file reports are
    generated from -- without it.
    """
    text = flat(METRICS)
    assert "system message" in text
    assert "qwen" in text, "the merge's cause (Qwen3.5's template) must be stated"
    assert "every model" in text, "must state the merge applies to all models, not just Qwen"


def test_judge_loss_is_described_as_non_random():
    """A denominator of 3,177 is not self-explaining; the bias has to be stated."""
    text = flat(METRICS)
    assert "non-random" in text or "not random" in text
    assert "observation prompt injection" in text
