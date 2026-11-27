# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document identity corrections are verifier-only and preserve rubric requirements."""

import json
from pathlib import Path

import pytest

from resources_servers.legal_agent_bench import verifier
from resources_servers.legal_agent_bench.lab_harbor.document_ids import CORRECTION_VERSION, SOURCE_REVISION, TASK_ID


@pytest.mark.parametrize(
    "profile, target",
    [
        ("single", "matching"),
        ("official_dual", "matching"),
        ("single", "other-task"),
        ("single", "other-revision"),
        ("single", "no-metadata"),
    ],
)
def test_verifier_resolves_only_verified_task_identities(
    tmp_path: Path, monkeypatch, profile: str, target: str
) -> None:
    output = tmp_path / "run" / "output"
    output.mkdir(parents=True)
    text = "D003 | chandrasekaran-burrell-litrisik-email.eml | March 3, 2023"
    (output / "privilege-log.md").write_text(text)
    metadata = {"lab_task_id": TASK_ID, "lab_source_revision": SOURCE_REVISION}
    if target == "other-task":
        metadata["lab_task_id"] = "another-task"
    elif target == "other-revision":
        metadata["lab_source_revision"] = "another-revision"
    elif target == "no-metadata":
        metadata = {}
    criteria = [
        {
            "id": "C-007",
            "title": "DOC_008 privilege log entry includes date",
            "match_criteria": "PASS if the entry for DOC_008 includes March 3, 2023. FAIL if missing.",
            "deliverables": ["privilege-log.md"],
        },
        {
            "id": "C-051",
            "title": "All documents covered",
            "match_criteria": "Cover DOC_001 through DOC_025, DOC_004, and unknown DOC_0080.",
            "deliverables": ["privilege-log.md"],
        },
    ]
    task = tmp_path / "task.json"
    task_data = {"title": "Review", "instructions": "Review documents.", "criteria": criteria, "metadata": metadata}
    task.write_text(json.dumps(task_data))
    original_task = task.read_bytes()
    prompts = []

    class Judge:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, prompt, *_args, **_kwargs):
            prompts.append(prompt)
            return {"verdict": "fail", "reasoning": "A verdict is still required."}

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    models = ["gpt-5.5"] if profile == "single" else ["claude-sonnet-4-6", "gpt-5.5"]
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": model} for model in models]))
    monkeypatch.setenv("LAB_JUDGE_PARALLELISM", "1")
    args = verifier.build_parser().parse_args(
        [
            "--task-json",
            str(task),
            "--run-dir",
            str(output.parent),
            "--verifier-dir",
            str(tmp_path / "verifier"),
            "--reward-json",
            str(tmp_path / "verifier" / "reward.json"),
            "--judge-base-url",
            "http://judge.test/v1",
        ]
    )
    result = verifier.run_verifier(args)
    judges = 1 if profile == "single" else 2
    assert len(prompts) == 2 * judges
    date_prompts = prompts[::2]
    for prompt in date_prompts:
        assert f"## Agent's Output\n## Agent Output: privilege-log.md\n{text}" in prompt
        assert "D003" in prompt
        if target == "matching":
            assert (
                "entry for chandrasekaran-burrell-litrisik-email.eml includes March 3, 2023. FAIL if missing."
                in prompt
            )
            assert "DOC_008" not in prompt
        else:
            assert "entry for DOC_008 includes March 3, 2023. FAIL if missing." in prompt
    assert all("DOC_001 through DOC_025, DOC_004, and unknown DOC_0080" in p for p in prompts[1::2])
    if judges == 2:
        assert prompts[:2] == prompts[2:]
    expected_versions = [CORRECTION_VERSION] if target == "matching" else []
    reports = result["per_judge"].values() if judges == 2 else [result]
    assert result["grading_corrections"] == expected_versions
    for report in reports:
        assert report["grading_corrections"] == expected_versions
        assert report["criteria_results"][0]["title"] == criteria[0]["title"]
        assert report["criteria_results"][0]["id"] == "C-007"
        if target == "matching":
            assert report["document_id_aliases"]["DOC_008"] == "chandrasekaran-burrell-litrisik-email.eml"
        else:
            assert report["document_id_aliases"] == {}
    # Identity context cannot grant credit: even the mapped task retains both
    # independent failing verdicts and the all-criteria task reward remains zero.
    assert result["score"] == 0.0
    assert result["n_passed"] == 0
    assert task.read_bytes() == original_task
    assert (output / "privilege-log.md").read_text() == text
    transcript = tmp_path / "verifier" / ("judge-1" if judges == 2 else "") / "transcript.jsonl"
    events = [json.loads(line) for line in transcript.read_text().splitlines()]
    completed = [event for event in events if event["type"] == "criterion_complete"]
    assert completed[0]["criterion_title"] == criteria[0]["title"]
    assert completed[0]["prompt"] == prompts[0]


def test_missing_deliverable_stays_a_failure_with_identity_context(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "run" / "output").mkdir(parents=True)
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "title": "Review",
                "instructions": "Write privilege-log.md.",
                "metadata": {"lab_task_id": TASK_ID, "lab_source_revision": SOURCE_REVISION},
                "criteria": [
                    {
                        "id": "C-007",
                        "title": "DOC_008 date",
                        "match_criteria": "Date exists.",
                        "deliverables": ["privilege-log.md"],
                    }
                ],
            }
        )
    )

    class Judge:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            pytest.fail("A missing deliverable must not be sent to the judge")

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": "gpt-5.5"}]))
    args = verifier.build_parser().parse_args(
        [
            "--task-json",
            str(task),
            "--run-dir",
            str(tmp_path / "run"),
            "--verifier-dir",
            str(tmp_path / "verifier"),
            "--reward-json",
            str(tmp_path / "reward.json"),
            "--judge-base-url",
            "http://judge.test/v1",
        ]
    )
    result = verifier.run_verifier(args)
    assert result["score"] == 0.0
    assert result["judge_error_count"] == 0
    assert result["criteria_results"][0]["verdict"] == "fail"
    assert "missing deliverable" in result["criteria_results"][0]["reasoning"]
