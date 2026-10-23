# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed output checks reject concrete violations without interpreting semantic criteria."""

import json

import pytest

from resources_servers.partner_if import fixed_checks
from resources_servers.partner_if.checks import REGISTRY, CheckContext, EmptyParams, resolve
from resources_servers.partner_if.engine import evaluate
from resources_servers.partner_if.evidence import Artifact, EvidenceBundle, EvidenceUnavailable, VisibleEvent
from resources_servers.partner_if.grading_contract import GradingContract, RawJudgment, VerifierSpec


FIXED_CHECKS = [
    "scratchpad_comments_v1",
    "review_sections_v1",
    "max_100_words_v1",
    "short_headings_v1",
    "headed_section_v1",
    "max_five_sections_v1",
    "no_plan_tool_v1",
    "made_project_edits_v1",
]


def run(identifier, answer="", artifacts=None):
    checker, params = resolve(VerifierSpec(id=identifier))
    return checker.run(params, CheckContext(EvidenceBundle(answer=answer, artifacts=artifacts or {})))


@pytest.mark.parametrize("identifier", FIXED_CHECKS)
def test_fixed_checks_are_built_in_and_take_no_dataset_parameters(identifier):
    assert REGISTRY[identifier].params is EmptyParams and REGISTRY[identifier].run is not None
    with pytest.raises(ValueError, match="Extra inputs"):
        resolve(VerifierSpec(id=identifier, params={"limit": 100000}))


# Scratchpad comments.

SCRATCHPAD = "<!-- [scratchpad_start]\n[task 1/1]\n[scratchpad_end] -->"


@pytest.mark.parametrize(
    "answer,expected",
    [
        (SCRATCHPAD, True),
        (SCRATCHPAD + "\nReview.\n" + SCRATCHPAD, True),
        ("", False),
        ("[scratchpad_start]\nTasks\n[scratchpad_end]", False),
        (SCRATCHPAD.replace("<!-- ", "<!--"), False),
        (SCRATCHPAD.replace(" -->", "-->"), False),
        ("<!-- [scratchpad_start]\n" + SCRATCHPAD + "\n[scratchpad_end] -->", False),
        ("[scratchpad_end] -->\n" + SCRATCHPAD, False),
        (SCRATCHPAD + "\n<!-- [scratchpad_start]", False),
        (SCRATCHPAD.replace("[task 1/1]", "Premature close -->"), False),
        ("```text\n" + SCRATCHPAD + "\n```", False),
        ("`" + SCRATCHPAD.replace("\n", " ") + "`", False),
        (SCRATCHPAD + "\nExample: `[scratchpad_start]`", True),
        ("\\" + SCRATCHPAD, False),
        (SCRATCHPAD.replace("<!--", "&lt;!--"), False),
        (SCRATCHPAD.replace("<!--", "<!`unused`--"), False),
        ("<!-- other comment\n" + SCRATCHPAD, False),
        ("<!-- Other comment -->\n" + SCRATCHPAD, True),
        ("Review complete." + SCRATCHPAD.replace("[task 1/1]", "[task]\n- Check --flags\n- Finish"), True),
        (SCRATCHPAD.replace("[task 1/1]", "Literal `backticks` and --flags"), True),
        (SCRATCHPAD.replace("[task 1/1]", "`<!-- nested -->`"), False),
        ("Example: ``<!-- [scratchpad_start] [scratchpad_end] -->``\n" + SCRATCHPAD, True),
        ("Literal `\n" + SCRATCHPAD.replace("[task 1/1]", "Task with ` character"), True),
        ("Literal `\n## Heading\n" + SCRATCHPAD.replace("[task 1/1]", "Task with ` character"), True),
        ("Literal `\r\n\r\n" + SCRATCHPAD.replace("[task 1/1]", "Task with ` character"), True),
        (SCRATCHPAD + "\nLiteral `\n<!-- [scratchpad_start] bad [scratchpad_end]-->\n`", False),
    ],
)
def test_scratchpad_comments_need_exact_paired_raw_wrappers(answer, expected):
    assert run("scratchpad_comments_v1", answer)[0] is expected


# Review sections.

SECTIONS = (
    "## Automated change summary\n\nSummary.\n\n"
    "## Changes to exported or public declarations\n\nNone.\n\n"
    "## Dependency manifest review\n\nNo manifests.\n"
)


@pytest.mark.parametrize(
    "answer,expected",
    [
        (SECTIONS, True),
        (SECTIONS.replace("## ", "##  "), True),
        (SECTIONS.replace("## Dependency manifest review", "Dependency manifest review\n---"), True),
        (SECTIONS.replace("Automated change summary", "**Automated change summary**"), True),
        (SECTIONS.replace("## Automated change summary", "Automated change\nsummary\n---"), True),
        (SECTIONS.replace("Automated change summary", "Automated   change summary"), True),
        (SECTIONS.replace("## Dependency", "### Dependency"), False),
        (SECTIONS.replace("## Dependency manifest review", "Dependency manifest review"), False),
        ("```markdown\n" + SECTIONS + "```", False),
        ("\n".join("> " + line for line in SECTIONS.splitlines()), False),
        (SECTIONS.replace("Automated change summary", "Wrong title"), False),
        ("## Dependency manifest review\n" + SECTIONS, False),
    ],
)
def test_review_sections_require_actual_h2_titles_in_order(answer, expected):
    assert run("review_sections_v1", answer)[0] is expected


# Word limit.


@pytest.mark.parametrize(
    "answer,expected,count",
    [
        ("word " * 100, True, 100),
        ("word " * 101, False, 101),
        ("word " * 100 + "[web:3] [web:2][web:4]", True, 100),
        ("word " * 100 + "[doc:3] [internal:5]", True, 100),
        ("word " * 99 + "state-of-the-art 3.5", False, 101),
        ("# Heading\n- café 東京 ٣\n*** [web:3]", True, 4),
        ("word " * 100 + "[ordinary words]", False, 102),
    ],
)
def test_word_limit_counts_alphanumeric_tokens_without_citation_markers(answer, expected, count):
    passed, detail = run("max_100_words_v1", answer)
    assert passed is expected
    assert f"{count} words" in detail


# Headings and sections.


@pytest.mark.parametrize(
    "title,passed,count",
    [
        ("One two three four five", True, 5),
        ("One two three four five six", False, 6),
        ("**Back up** [the site](https://example.org/a/b/8.3) safely", True, 5),
        ("Use `git pull --rebase` safely", True, 5),
        ("State-of-the-art upgrades aren’t risk-free", True, 4),
        ("Préparer 東京 ٣", True, 3),
        ("Back up &amp; verify", True, 3),
        ("Check backups [web:3]", True, 2),
        ("🚀", True, 0),
        ("![One two three four five six](https://example.org/image)", False, 6),
    ],
)
def test_heading_words_use_visible_text_and_the_word_limit_convention(title, passed, count):
    verdict, detail = run("short_headings_v1", f"## {title}\n\nBody.")
    assert verdict is passed
    assert str(count) in detail


@pytest.mark.parametrize(
    "answer,passed",
    [
        ("## Short\nBody.\n\n### This heading has six separate words\nBody.", False),
        ("This heading has six separate words\n---\nBody.", False),
        ("```md\n## This heading has six separate words\n```\n## Good\nBody.", True),
        ("<!-- ## This heading has six separate words -->\n## Good\nBody.", True),
        ("## Good\n\n> ### This heading has six separate words\n> Quoted body.", False),
        ("Body with no heading.", False),
        ("## ***\nBody.", True),
        ("## Good\n\n    ## This heading has six separate words\n", True),
    ],
)
def test_short_headings_parse_real_headings_and_do_not_pass_vacuously(answer, passed):
    assert run("short_headings_v1", answer)[0] is passed


@pytest.mark.parametrize(
    "answer,passed",
    [
        ("## Backups\nBody.", True),
        ("### Backups\nBody.", True),
        ("Backups\n---\nBody.", True),
        ("## Backups\n\n1. Back up the site.", True),
        ("Intro.\n\n1. ## Back up\n\n   Save files. Verify restoration.", True),
        ("1. ## Backups\n\n   > Quoted body.", True),
        ("## Backups\n\n```sh\nbackup\n```", True),
        ("## Backups\n<!-- no body -->", False),
        ("## Backups\n### Empty too\n", False),
        ("# Backups\nBody.", False),
        ("**Backups**\nBody.", False),
        ("```md\n## Backups\nBody.\n```", False),
        ("> ## Backups\n> Quoted body.", False),
        ("1. > ## Backups\n   > Quoted body.", False),
        ("> 1. ## Backups\n>\n>    Quoted body.", False),
        ("## \nBody.", False),
    ],
)
def test_headed_section_needs_a_real_substantive_h2_or_h3_section(answer, passed):
    assert run("headed_section_v1", answer)[0] is passed


@pytest.mark.parametrize(
    "answer,passed,count",
    [
        ("## A\nBody.\n\n## B\nBody.\n\n## C\nBody.\n\n## D\nBody.\n\n## E\nBody.", True, 5),
        ("Intro.\n\n## A\nBody.\n\n## B\nBody.\n\n## C\nBody.\n\n## D\nBody.\n\n## E\nBody.", False, 6),
        ("Intro one.\n\nIntro two.\n\n## A\nBody.\n\nAnother paragraph.\n\n### Subsection\nBody.", True, 2),
        ("### A\nBody.\n\n### B\nBody.\n\n## C\nBody.\n\n### Nested\nBody.", True, 3),
        ("<!-- hidden intro -->\n\n---\n\n## A\nBody.", True, 1),
        ("Intro.\n\n```md\n## Not a section\n```\n\n## A\nBody.\n\n> ## Not top-level\n> Quote.", True, 2),
        ("Only one unheaded answer.\n\nAnother paragraph.", True, 1),
        ("## A\nBody.\n\n## B\nBody.\n\n## C\nBody.\n\n## D\nBody.\n\n## E\nBody.\n\n## F\nBody.", False, 6),
        ("Intro.\n\n1. ## Back up\n\n   Save files.\n\n2. ## Audit\n\n   Check settings.", True, 3),
        ("\n\n".join(f"{i}. ## Section\n\n   Content." for i in range(1, 7)), False, 6),
        ("## Main\nBody.\n\n1. ### Nested\n\n   Detail.\n\n## Next\nBody.", True, 2),
        ("", False, 0),
    ],
)
def test_sections_count_outline_roots_and_one_introduction_not_paragraphs(answer, passed, count):
    verdict, detail = run("max_five_sections_v1", answer)
    assert verdict is passed
    assert f"{count} top-level" in detail


# Trajectory and environment receipts.


def receipts(events=(), edits=None, artifact=True):
    artifacts = {}
    if artifact:
        artifacts["project_edits"] = Artifact(status="present", text=json.dumps(edits if edits is not None else []))
    return CheckContext(evidence=EvidenceBundle(trajectory=list(events), artifacts=artifacts))


def tool_call(name, index=0, arguments=None):
    return VisibleEvent(source_index=index, kind="tool_call", tool_name=name, call_id=f"c{index}", arguments=arguments)


def test_no_plan_tool_passes_when_the_agent_reads_and_edits_without_planning():
    passed, detail = fixed_checks.no_plan_tool(receipts([tool_call("read_file"), tool_call("write_file", 1)]))
    assert passed and detail.startswith("0 plan--draft")


@pytest.mark.parametrize("arguments", [{"content": "## Plan"}, "{not json", None])
def test_no_plan_tool_fails_on_any_attempt_even_malformed_or_late(arguments):
    events = [tool_call("write_file"), tool_call("plan--draft", 1, arguments)]
    assert fixed_checks.no_plan_tool(receipts(events)) == (False, "1 plan--draft call(s) in the episode")


def test_no_plan_tool_ignores_plans_mentioned_in_text():
    text = VisibleEvent(source_index=0, kind="assistant", text="I will call plan--draft next.")
    assert fixed_checks.no_plan_tool(receipts([text]))[0]


def test_project_edits_require_at_least_one_receipted_edit():
    passed, detail = fixed_checks.made_project_edits(
        receipts(edits=[{"tool": "write_file", "path": "src/main.ts", "created": False, "deleted": False}])
    )
    assert passed and "src/main.ts" in detail
    assert fixed_checks.made_project_edits(receipts(edits=[])) == (
        False,
        "0 successful project edit(s) across 0 file(s): ",
    )


def test_tool_calls_without_receipts_are_not_project_edits():
    assert not fixed_checks.made_project_edits(receipts([tool_call("write_file")], edits=[]))[0]


@pytest.mark.parametrize("artifact", [None, "absent", '[{"tool": "write_file"}]', "not json"])
def test_missing_or_malformed_receipts_are_capture_gaps(artifact):
    if artifact is None:
        context = receipts(artifact=False)
    elif artifact == "absent":
        context = CheckContext(evidence=EvidenceBundle(artifacts={"project_edits": Artifact(status="absent")}))
    else:
        context = CheckContext(
            evidence=EvidenceBundle(artifacts={"project_edits": Artifact(status="present", text=artifact)})
        )
    with pytest.raises(EvidenceUnavailable):
        fixed_checks.made_project_edits(context)


# Grading.


@pytest.mark.asyncio
async def test_fixed_checks_share_the_engine_and_do_not_send_their_items_to_the_judge():
    contract = GradingContract(
        task_id="synthetic",
        origin="test",
        source="test",
        rubric_version="test",
        audit_signoff=False,
        data_kind="synthetic_smoke",
        rubrics=[
            {"id": "format", "constraint": "Wrappers", "verifier": {"id": "scratchpad_comments_v1"}},
            {"id": "meaning", "constraint": "Meaning", "verifier": {"id": "llm_judge_v1"}},
        ],
    )

    async def judge(items, evidence):
        assert [item.id for item in items] == ["meaning"]
        return [RawJudgment(id="meaning", passed=False, reason="Missing action", evidence="No action")]

    result = await evaluate(contract, EvidenceBundle(answer=SCRATCHPAD), judge=judge)
    assert result.dimensions["if"].points_earned == 1
    assert result.dimensions["if"].points_possible == 2
    assert result.dimensions["if"].partial_score == 0.5
    assert result.reward == 0
