# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.data import load_task_data_schema, validate_jsonl_rows
from resources_servers.agentic_turn_simulation_judge.app import AgenticTurnSimulationJudgeRunRequest
from resources_servers.agentic_turn_simulation_judge.scripts.validate_rows import validate_files


ENV_DIR = Path(__file__).resolve().parents[1]
EXAMPLE = ENV_DIR / "data" / "example.jsonl"
ROWS = [json.loads(line) for line in EXAMPLE.read_text().splitlines()]


def write(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class TestExampleData:
    def test_example_rows_validate_clean_against_task_data(self) -> None:
        report = validate_jsonl_rows(
            "agentic_turn_simulation_judge",
            load_task_data_schema(ENV_DIR),
            str(EXAMPLE),
            EXAMPLE.read_text().splitlines(),
        )
        assert report.rows == 5 and report.clean, report.summary()

    def test_example_rows_pass_validate_rows(self) -> None:
        report = validate_files([EXAMPLE], agent="")
        counts = report["all"]
        assert counts["rows"] == 5
        assert counts["gold_ladder"] == {"yes": 3, "no": 2}
        assert counts["by_tier"] == {"A": 4, "B": 1}
        assert counts["by_family"] == {"ordinary": 4, "obstacle": 1}
        assert counts["by_split"] == {"example": 5}
        assert counts["gold_ladder_by_split"] == {"example": {"yes": 3, "no": 2}}
        assert counts["duplicate_prompts"] == 0 and counts["groups_in_multiple_splits"] == 0
        assert counts["p_yes_disagrees_with_gold"] == 0 and counts["gold_fields_ladder_disagrees_with_gold"] == 0

    def test_example_prompts_are_the_ats_judge_prompt(self) -> None:
        for row in ROWS:
            system, user = row["responses_create_params"]["input"]
            assert system["content"].startswith("You are evaluating what an AI agent did")
            assert "<criteria>" in system["content"] and '"discard_reason"' in system["content"]
            assert "<step_under_evaluation>" in user["content"] and "<conversation>" in user["content"]


def broken(mutate) -> dict:
    row = copy.deepcopy(ROWS[0])
    mutate(row)
    return row


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda r: r.pop("gold_ladder"), "gold_ladder"),
        (lambda r: r.update(gold_ladder="maybe"), "gold_ladder"),
        (lambda r: r.update(gold_p_yes=1.5), "gold_p_yes"),
        (lambda r: r.update(task="pairwise"), "task"),
        (lambda r: r.update(gold_ladr="yes"), "undeclared top-level keys"),
        (lambda r: r.pop("group_id"), "group_id is missing"),
        (lambda r: r.update(row_id=" "), "row_id is missing"),
        (lambda r: r["gold_fields"].update(blocked_by="user"), "gold_fields.blocked_by"),
        (lambda r: r["responses_create_params"].update(tools=[{"type": "function", "name": "x"}]), "tools"),
        (lambda r: r["responses_create_params"]["input"].pop(0), "must be [system message, user message]"),
        (lambda r: r["responses_create_params"]["input"][1].update(content="no tag"), "<step_under_evaluation>"),
        (lambda r: r["responses_create_params"]["input"][0].update(content=""), "non-empty string"),
        (lambda r: r.pop("responses_create_params"), "responses_create_params"),
    ],
)
def test_bad_rows_stop_with_their_line(tmp_path: Path, mutate, message: str) -> None:
    path = write(tmp_path, "rows.jsonl", [ROWS[1], broken(mutate)])
    with pytest.raises(SystemExit, match=re.escape(message)) as exc:
        validate_files([path], agent="")
    assert f"{path}:2:" in str(exc.value)


def test_agent_ref_is_checked(tmp_path: Path) -> None:
    path = write(tmp_path, "rows.jsonl", ROWS)
    with pytest.raises(SystemExit, match="agent_ref"):
        validate_files([path])
    tagged = [{**r, "agent_ref": {"name": "agentic_turn_simulation_judge_simple_agent"}} for r in ROWS]
    assert validate_files([write(tmp_path, "tagged.jsonl", tagged)])["all"]["rows"] == 5


def test_duplicate_row_id_in_a_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="duplicate row_id"):
        validate_files([write(tmp_path, "rows.jsonl", [ROWS[0], ROWS[0]])], agent="")


def test_cross_file_counts(tmp_path: Path) -> None:
    train = [{**r, "meta": {**r["meta"], "split": "train"}} for r in ROWS[:3]]
    # The same transcript group in validation, with a relabelled copy of one prompt and no split recorded.
    leaked = {**ROWS[0], "row_id": "v-0", "meta": {"source": "x"}, "gold_ladder": "no", "gold_p_yes": 0.75}
    val = [leaked, {**ROWS[3], "meta": {**ROWS[3]["meta"], "split": "val"}}]
    report = validate_files([write(tmp_path, "train.jsonl", train), write(tmp_path, "val.jsonl", val)], agent="")
    assert report["train.jsonl"]["rows"] == 3 and report["val.jsonl"]["rows"] == 2
    assert report["val.jsonl"]["gold_ladder_by_split"] == {"none": {"no": 1}, "val": {"no": 1}}
    totals = report["all"]
    assert totals["rows"] == 5 and totals["gold_ladder"] == {"yes": 2, "no": 3}
    assert totals["duplicate_prompts"] == 1
    assert totals["groups_in_multiple_splits"] == 1
    assert totals["groups_in_multiple_splits_examples"] == [ROWS[0]["group_id"]]
    assert totals["p_yes_disagrees_with_gold"] == 1
    assert totals["gold_fields_ladder_disagrees_with_gold"] == 1


def test_report_file_is_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from resources_servers.agentic_turn_simulation_judge.scripts import validate_rows

    out = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["validate_rows.py", str(EXAMPLE), "--agent", "", "--report", str(out)])
    validate_rows.main()
    assert json.loads(out.read_text())["all"]["rows"] == 5
    assert out.stat().st_mode & 0o777 == 0o600
    assert capsys.readouterr().out.splitlines()[0].startswith("example.jsonl {")


@pytest.mark.parametrize("extra", [123, ["a"], {"k": "v"}])
def test_gold_field_extras_must_be_strings(tmp_path: Path, extra) -> None:
    """The file validator and the server's request model agree: extra gold_fields values are strings or null."""
    row = copy.deepcopy(ROWS[0])
    row["gold_fields"]["analysis"] = extra
    with pytest.raises(SystemExit, match="gold_fields"):
        validate_files([write(tmp_path, "rows.jsonl", [row])], agent="")
    with pytest.raises(ValidationError):
        AgenticTurnSimulationJudgeRunRequest.model_validate(row)


def test_gold_field_string_and_null_extras_validate(tmp_path: Path) -> None:
    row = copy.deepcopy(ROWS[0])
    row["gold_fields"].update(reason="free text", analysis=None)
    assert validate_files([write(tmp_path, "rows.jsonl", [row])], agent="")["all"]["rows"] == 1
    AgenticTurnSimulationJudgeRunRequest.model_validate(row)


def test_same_basename_files_stay_apart(tmp_path: Path) -> None:
    def unsplit(row: dict, row_id: str) -> dict:
        meta = {k: v for k, v in row["meta"].items() if k != "split"}
        return {**row, "row_id": row_id, "group_id": "shared", "meta": meta}

    yes, no = unsplit(ROWS[0], "a"), unsplit(ROWS[1], "b")
    for d in ("train", "validation"):
        (tmp_path / d).mkdir()
    paths = [write(tmp_path, "train/rows.jsonl", [yes]), write(tmp_path, "validation/rows.jsonl", [no])]
    report = validate_files(paths, agent="")
    assert set(report) == {str(p.resolve()) for p in paths} | {"all"}
    assert [report[str(p.resolve())]["rows"] for p in paths] == [1, 1]
    assert report["all"]["rows"] == 2 and report["all"]["groups_in_multiple_splits"] == 1
    # The same explicit split on both rows: the group is not in two splits.
    for p, row in zip(paths, (yes, no)):
        write(tmp_path, str(p.relative_to(tmp_path)), [{**row, "meta": {**row["meta"], "split": "train"}}])
    assert validate_files(paths, agent="")["all"]["groups_in_multiple_splits"] == 0
