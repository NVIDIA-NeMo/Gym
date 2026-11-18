# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation regressions use invented tasks, never private question-bank data."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.data import load_task_data_schema, validate_jsonl_rows
from resources_servers.question_bank_judge.prepare import family_splits, prepare
from resources_servers.question_bank_judge.prompts import SOLVER_INSTRUCTION


def _row(task_id: str, **overrides) -> dict:
    return {
        "id": task_id,
        "version_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "original_question_sha256": hashlib.sha256((task_id + " original").encode()).hexdigest(),
        "version": "original_valid_pair",
        "domain": "Mathematics",
        "problem": "What is 1 + 1?",
        "expected_answer": "2",
        "reference": {"kind": "qa"},
        "upstream": {},
        **overrides,
    }


def _dataset(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _inventory(dataset: Path, groups: dict[str, str] | None = None, **overrides) -> Path:
    records = []
    with gzip.open(dataset, "rb") as handle:
        for index, line in enumerate(handle, start=1):
            row = json.loads(line)
            records.append(
                {
                    "id": row["id"],
                    "dataset_line": str(index),
                    "version_sha256": row["version_sha256"],
                    "version": row["version"],
                    "domain": row["domain"],
                    "source_row_sha256": hashlib.sha256(line).hexdigest(),
                    "review_group": (groups or {}).get(row["id"], "answer_comparison_review"),
                    "declared_methods": "EXACT_VALUE",
                    "upstream_dataset": "",
                    **overrides,
                }
            )
    path = dataset.parent / "inventory.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_family_transitive_closure_forces_all_variants_into_development() -> None:
    rows = [
        _row("a", upstream={"dataset": "source", "original_record_id": 0}),
        _row("b", upstream={"dataset": "source", "original_record_id": 0}, qa_variant_ids=["variant"]),
        _row("c", qa_variant_ids=["variant"]),
    ]
    first = family_splits(rows, development_ids={"c"})
    second = family_splits(list(reversed(rows)), development_ids={"c"})
    assert first == second
    assert len({item["family_id"] for item in first.values()}) == 1
    assert all(item["split"] == "development" and item["forced_development"] for item in first.values())


def test_source_identifiers_are_namespaced_and_missing_ids_do_not_merge() -> None:
    rows = [
        _row("a", upstream={"dataset": "source1", "original_record_id": 3}),
        _row("b", upstream={"dataset": "source2", "original_record_id": 3}),
        _row("c", upstream={"dataset": "source2", "original_record_id": None}),
        _row("d", upstream={"dataset": "source2", "original_record_id": None}),
    ]
    splits = family_splits(rows, development_ids=set())
    assert len({item["family_id"] for item in splits.values()}) == 4


def test_routes_answers_without_per_row_contracts_and_keeps_exclusions(tmp_path: Path) -> None:
    rows = [
        _row("math"),
        _row("semantic", domain="Biology"),
        _row("humanities", domain="Humanities / Social Sciences"),
        _row("unlabeled", domain="Other / Unlabeled"),
        _row("unknown", domain="unrecognized domain"),
        _row("rubric", domain="Humanities / Social Sciences"),
        _row("native", domain="Other / Unlabeled", reference={"kind": "native_grader"}, expected_answer=None),
        _row("chemistry_native", domain="Chemistry / materials"),
    ]
    source = _dataset(tmp_path, rows)
    inventory = _inventory(
        source,
        groups={
            "rubric": "construction_or_rubric_review",
            "native": "competitive_coding_native",
            "chemistry_native": "noncoding_native_tests_unresolved",
        },
    )
    output = tmp_path / "prepared"
    receipt = prepare(source, output, development_ids={r["id"] for r in rows}, inventory_path=inventory)
    assert receipt["counts"] == {
        "evaluation_candidate": 4,
        "excluded_unsupported_domain": 1,
        "excluded_rubric": 1,
        "excluded_native_pending_runner": 2,
    }
    assert receipt["route_counts"] == {"math": 1, "semantic": 3}
    assert receipt["training_admitted"] == 0
    ledger = _lines(output / "ledger.jsonl")
    assert len(ledger) == len(rows)
    assert not any(row["training_admitted"] for row in ledger)
    emitted = _lines(output / "development.gym.jsonl")
    assert {row["id"]: row["agent_ref"]["name"] for row in emitted} == {
        "math": "question_bank_math_simple_agent",
        "semantic": "question_bank_semantic_simple_agent",
        "humanities": "question_bank_semantic_simple_agent",
        "unlabeled": "question_bank_semantic_simple_agent",
    }
    config_path = Path(__file__).resolve().parents[1] / "configs/question_bank.yaml"
    config = OmegaConf.load(config_path)
    for row in emitted:
        agent = config[row["agent_ref"]["name"]].responses_api_agents.simple_agent
        component = "math_with_judge" if row["id"] == "math" else "question_bank_judge"
        assert list(config[agent.resources_server.name].resources_servers) == [component]
    assert config.question_bank_math.resources_servers.math_with_judge.should_use_judge is True
    assert all("task_contract" not in row for row in emitted)
    with pytest.raises(FileExistsError):
        prepare(source, output, development_ids=set(), inventory_path=inventory)


def test_native_method_overrides_answer_triage(tmp_path: Path) -> None:
    source = _dataset(tmp_path, [_row("a", domain="Physics")])
    receipt = prepare(
        source,
        tmp_path / "prepared",
        development_ids={"a"},
        inventory_path=_inventory(source, declared_methods="EXACT_VALUE|NATIVE_CODE_TESTS"),
    )
    assert receipt["counts"] == {"excluded_native_pending_runner": 1}


def test_policy_contains_only_selected_problem_and_shared_instructions(tmp_path: Path) -> None:
    from resources_servers.question_bank_judge.app import QuestionBankJudgeRunRequest
    from responses_api_agents.simple_agent.app import SimpleAgentRunRequest

    row = _row(
        "a",
        domain="Physics",
        expected_answer="PRIVATE_REFERENCE",
        original={"problem": "ORIGINAL_QUESTION", "expected_answer": "ORIGINAL_REFERENCE"},
        version="deepseek_repaired",
        kimi_profile={"n_correct": 6},
        rl_suitability={"decision": "RL_SUITABLE"},
    )
    source = _dataset(tmp_path, [row])
    output = tmp_path / "prepared"
    prepare(source, output, development_ids={"a"}, inventory_path=_inventory(source))
    gym_row = _lines(output / "development.gym.jsonl")[0]
    prompt = gym_row["responses_create_params"]["input"][0]["content"]
    assert prompt == row["problem"] + "\n\n" + SOLVER_INSTRUCTION
    assert gym_row["question"] == row["problem"]
    assert gym_row["id"] == gym_row["uuid"] == "a"
    assert gym_row["metadata"]["repaired"] is True
    assert gym_row["metadata"]["reference_sha256"] == hashlib.sha256(b"PRIVATE_REFERENCE").hexdigest()
    assert gym_row["metadata"]["review_group"] == "answer_comparison_review"
    assert gym_row["metadata"]["declared_methods"] == ["EXACT_VALUE"]
    assert gym_row["metadata"]["reference_kind"] == "qa"
    assert gym_row["metadata"]["policy_overrides_applied"] is False
    assert gym_row["metadata"]["policy_extra_instructions"] == []
    assert all(text not in prompt for text in ("PRIVATE_REFERENCE", "ORIGINAL_REFERENCE", "ORIGINAL_QUESTION"))
    assert "kimi_profile" not in json.dumps(gym_row)
    assert "RL_SUITABLE" not in json.dumps(gym_row)
    forwarded = SimpleAgentRunRequest.model_validate(gym_row).model_dump()
    verifier_row = QuestionBankJudgeRunRequest.model_validate(forwarded)
    assert verifier_row.expected_answer == row["expected_answer"]

    schema = load_task_data_schema(Path(__file__).resolve().parents[1])
    assert schema is not None
    report = validate_jsonl_rows("question_bank_judge", schema, "development.gym.jsonl", [json.dumps(gym_row)])
    assert report.clean, report.summary()

    for validate in (schema.validate_python, QuestionBankJudgeRunRequest.model_validate):
        with pytest.raises(ValidationError, match="answer_format"):
            validate({**gym_row, "answer_format": "misspelled"})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"expected_answer": None}, "excluded_missing_reference"),
        ({"expected_answer": "  "}, "excluded_missing_reference"),
        ({"reference": {"expected_answer": "3"}}, "excluded_reference_mismatch"),
        ({"reference": {"expected_answer_sha256": "0" * 64}}, "excluded_reference_mismatch"),
    ],
)
def test_missing_or_conflicting_selected_reference_is_excluded(tmp_path: Path, overrides: dict, expected: str) -> None:
    source = _dataset(tmp_path, [_row("a", **overrides)])
    output = tmp_path / "prepared"
    receipt = prepare(source, output, development_ids={"a"}, inventory_path=_inventory(source))
    assert receipt["counts"] == {expected: 1}
    assert not (output / "development.gym.jsonl").read_text()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version_sha256": "wrong-version"}, "version_sha256 mismatch"),
        ({"source_row_sha256": "wrong-row"}, "source row binding mismatch"),
        ({"review_group": "unknown"}, "unsupported triage group"),
        ({"id": "wrong-id"}, "membership"),
    ],
)
def test_inventory_must_bind_the_exact_selected_rows(tmp_path: Path, overrides: dict, message: str) -> None:
    source = _dataset(tmp_path, [_row("a")])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=message):
        prepare(source, output, development_ids={"a"}, inventory_path=_inventory(source, **overrides))
    assert not output.exists()


def test_unknown_source_version_is_not_treated_as_keep(tmp_path: Path) -> None:
    source = _dataset(tmp_path, [_row("a", version="unreviewed_draft")])
    with pytest.raises(ValueError, match="unsupported source version"):
        prepare(source, tmp_path / "prepared", development_ids={"a"}, inventory_path=_inventory(source))
    assert not (tmp_path / "prepared").exists()


def test_frozen_families_preserved_before_exclusion_and_prompt_changes(tmp_path: Path) -> None:
    rows = [_row("a"), _row("b", qa_variant_ids=["variant"]), _row("c", qa_variant_ids=["variant"])]
    source = _dataset(tmp_path, rows)
    inventory = _inventory(source, groups={"c": "construction_or_rubric_review"})
    first = tmp_path / "first"
    prepare(source, first, development_ids={"a"}, inventory_path=inventory)
    override = tmp_path / "overrides.json"
    override.write_text(json.dumps({"domains": {"mathematics": "State any necessary assumptions."}}))
    second = tmp_path / "second"
    prepare(
        source,
        second,
        development_ids={"a"},
        inventory_path=inventory,
        frozen_families_path=first / "families.jsonl",
        prompt_overrides_path=override,
    )
    assert (first / "families.jsonl").read_bytes() == (second / "families.jsonl").read_bytes()
    altered = json.loads((first / "families.jsonl").read_text().splitlines()[0])
    altered["family_id"] = "different"
    bad = tmp_path / "bad-families.jsonl"
    bad.write_text(json.dumps(altered) + "\n")
    with pytest.raises(ValueError, match="differ from the frozen"):
        prepare(
            source,
            tmp_path / "bad",
            development_ids={"a"},
            inventory_path=inventory,
            frozen_families_path=bad,
        )
    assert not (tmp_path / "bad").exists()


def test_prompt_overrides_select_canonical_domain_and_source(tmp_path: Path) -> None:
    source = _dataset(tmp_path, [_row("a", domain="phys", upstream={"dataset": "source-a"})])
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"domains": {"physics": "State units."}, "sources": {"source-a": "Label parts."}}))
    output = tmp_path / "prepared"
    prepare(
        source,
        output,
        development_ids={"a"},
        inventory_path=_inventory(source),
        prompt_overrides_path=overrides,
    )
    row = _lines(output / "development.gym.jsonl")[0]
    assert row["metadata"]["domain"] == "physics"
    assert row["metadata"]["source"] == "source-a"
    assert row["metadata"]["policy_overrides_applied"] is True
    assert row["metadata"]["policy_extra_instructions"] == ["State units.", "Label parts."]
    assert row["responses_create_params"]["input"][0]["content"].endswith(
        "State units.\n\nLabel parts.\n\n" + SOLVER_INSTRUCTION
    )
