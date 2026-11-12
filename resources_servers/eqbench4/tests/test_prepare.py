# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation against a small stand-in for the upstream repository (no upstream content is used)."""

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from resources_servers.eqbench4 import prepare as P
from resources_servers.eqbench4.judging import PromptParts
from resources_servers.eqbench4.task_data import TaskData


STAND_IN_PROMPTS = """
def build_persona_system_prompt(doc, num_turns=10):
    return f"persona of {doc['id']} for {num_turns} turns"

def build_evaluated_system_prompt(doc, current_turn, num_turns=10):
    return f"evaluated {doc['id']} turn {current_turn}/{num_turns}"

def _get_respondent_presenting_description(doc):
    return doc["presenting_issue"]["description"]

def _format_relationship_label(relationship):
    return relationship.title()

def _extract_modifier_sections(modifiers):
    return ("demo", "background", "personality")

def _extract_sensitivities_and_preferences(doc):
    return ("sensitive", "prefers")

def _format_chatbot_relationship_context(relationship, *, for_judge=False):
    return f"context:{relationship}:{for_judge}"

def _format_hand_authored_judge_context(doc, persona_system_prompt=None):
    return f"hand:{persona_system_prompt}"

def _build_initial_state_str(state, exclude):
    return f"state:{sorted(state)}:{exclude}"

def _state_exclude_fields(doc):
    return None

def build_performance_judge_prompt(scenario, transcript, final_state, persona_system_prompt=None,
                                   include_initial_state=True, state_trajectory=None):
    body = "".join(f"{m['role'].capitalize()}: {m['content']}\\n\\n" for m in transcript)
    return f"traits for {scenario['id']}\\n[T]\\n{body}[/T] final={final_state} p={persona_system_prompt}"
"""
PAIRWISE_TEMPLATE = (
    "{assistant_relationship}|{relationship_context}|{presenting_issue}|{core_issue_str}|{adjacent_str}|"
    "{trust_style_name}|{initial_state_str}|{hand_authored_judge_context}|{not_supplied}\n"
    "{code_a}: {transcript_a} [{persona_a_final_state}]\n{code_b}: {transcript_b} [{persona_b_final_state}]\n"
    "{pairwise_dimensions}\n{{\n{pairwise_dimensions_json}\n}} ends {code_a}"
)


def _doc(scenario_id: str) -> dict:
    return {
        "id": scenario_id,
        "assistant_relationship": "friend",
        "presenting_issue": {"description": f"issue {scenario_id}"},
        "core_issue": {"label": "Core", "description": "deep"},
        "core_adjacent_attributes": [{"label": "clue", "content": "detail"}],
        "persona": {"trust_style": {"name": "Slow"}},
        "initial_state": {"trusting": 3.0},
    }


def _conversation(scenario_id: str, *, completed: bool = True) -> dict:
    return {
        "scenario_id": scenario_id,
        "convo_index": 0,
        "completed": completed,
        "doc": _doc(scenario_id),
        "prompts": {"persona_system_prompt": f"persona of {scenario_id} for 8 turns"},
        "state_history": [],
        "transcript": [
            {"role": "assistant", "content": f"hello {scenario_id}"},
            {"role": "user", "content": "hi", "raw_response": "<internal>\nwary\n</internal><dialogue>hi</dialogue>"},
        ],
    }


def _run_file(model: str, comparisons: list[dict]) -> dict:
    return {
        "r1": {
            "__meta__": {"evaluated_model": model},
            "s1": [_conversation("s1")],
            "s2": [_conversation("s2"), _conversation("s2", completed=False)],
            "__pairwise__": comparisons,
        },
        "__summary__": "not a run",
    }


def _comparison(a: str, b: str, scenario: str, fraction: float | None, **extra) -> dict:
    return {
        "scenario_id": scenario,
        "iteration_index": 0,
        "pair": {"model_a": a, "model_b": b},
        "aggregated": {} if fraction is None else {"fraction_for_model_a": fraction},
        **extra,
    }


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake upstream checkout with pinned hashes patched to match it."""
    files = {
        P.DATASET_FILE: json.dumps({"docs": [_doc("s1"), _doc("s2")]}).encode(),
        "src/eqbench/__init__.py": b"",
        "src/eqbench/core/prompts.py": STAND_IN_PROMPTS.encode(),
        "src/eqbench/prompts/pairwise_judge_prompt.txt": PAIRWISE_TEMPLATE.encode(),
        "src/eqbench/rubrics/pairwise_dimensions.txt": b"# dims\nbond_and_rapport: Rapport.\n\nattunement: Reading.\n",
        "src/eqbench/rubrics/pairwise_dimensions_weights.txt": b"bond_and_rapport: 2.0\n",
        "results/r1.json": json.dumps(
            _run_file(
                "alpha",
                [
                    _comparison("beta", "alpha", "s1", 0.25),
                    _comparison("alpha", "beta", "s1", 0.9),  # same matchup, kept once
                    _comparison("alpha", "gamma", "s2", 1.4),
                    _comparison("alpha", "gamma", "s1", None),
                    _comparison("alpha", "gamma", "s2", 0.5, error="judge failed"),
                    _comparison("alpha", "alpha", "s2", 0.5),
                ],
            )
        ).encode(),
        "results/r2.json": json.dumps(_run_file("beta", [_comparison("beta", "gamma", "s2", 0.6)])).encode(),
        "results/r3.json": json.dumps(_run_file("gamma", [])).encode(),
    }
    for path, data in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.replace(b"\n", b"\r\n") if path.endswith(".txt") else data)
    monkeypatch.setattr(P, "SOURCE_SHA256", {path: hashlib.sha256(data).hexdigest() for path, data in files.items()})
    monkeypatch.setattr(P, "RESULT_FILES", ["results/r1.json", "results/r2.json", "results/r3.json"])
    return tmp_path


def test_load_source_checks_hashes_and_restores_line_endings(stand_in: Path) -> None:
    text = P.load_source("src/eqbench/rubrics/pairwise_dimensions.txt", source_dir=stand_in)
    assert b"\r\n" not in text and text.startswith(b"# dims\n")
    (stand_in / P.DATASET_FILE).write_bytes(b"{}")
    with pytest.raises(ValueError, match="does not match"):
        P.load_source(P.DATASET_FILE, source_dir=stand_in)


def test_load_source_downloads_once_into_the_cache(
    stand_in: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []
    data = (stand_in / "results/r3.json").read_bytes()

    def urlopen(url: str, timeout: int) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr(P.urllib.request, "urlopen", urlopen)
    cache = tmp_path / "cache"
    assert P.load_source("results/r3.json", cache_dir=cache) == data
    assert P.load_source("results/r3.json", cache_dir=cache) == data
    assert requested == [P.SOURCE_URL + "results/r3.json"]
    assert (cache / P.REVISION / "results/r3.json").read_bytes() == data


def test_split_prompt_and_rubric_lines() -> None:
    assert P.split_prompt("a\x00x\x00b\x00y\x00") == {"literals": ["a", "b", ""], "slots": ["x", "y"]}
    with pytest.raises(ValueError, match="unpaired"):
        P.split_prompt("a\x00x")
    assert P.parse_rubric_lines("# c\n\nname: value: more\nno colon\n") == {"name": "value: more"}


def test_final_state_and_reference_transcript() -> None:
    assert P.final_state_text([]) == P.NO_FINAL_STATE
    assert P.final_state_text([{"trusting": 4.25, "progress_on_core_issue": 2}]) == (
        "- Trusting: 4.2\n- Progress On Core Issue: 2"
    )
    reduced = P.reference_transcript(
        [
            {"role": "assistant", "content": "hey", "raw_response": "hey"},
            {"role": "user", "content": None, "raw_response": "[INTERNAL] guarded [/INTERNAL]"},
            {"role": "user", "content": "x"},
        ]
    )
    assert reduced == [
        {"role": "assistant", "content": "hey"},
        {"role": "user", "content": "", "internal": "guarded"},
        {"role": "user", "content": "x", "internal": None},
    ]


def test_reference_runs_dedupe_and_ladder(stand_in: Path) -> None:
    references = P.ReferenceRuns()
    for path in P.RESULT_FILES:
        references.add_run_file(json.loads(P.load_source(path, source_dir=stand_in)))
    assert sorted(references.transcripts) == ["alpha", "beta", "gamma"]
    assert sorted(references.transcripts["alpha"]) == ["s1", "s2"]
    assert references.comparisons == [
        {"model_a": "alpha", "model_b": "beta", "fraction_a": 0.75, "scenario_id": "s1"},
        {"model_a": "alpha", "model_b": "gamma", "fraction_a": 1.0, "scenario_id": "s2"},
        {"model_a": "beta", "model_b": "gamma", "fraction_a": 0.6, "scenario_id": "s2"},
    ]
    assert references.ladder() == ["alpha", "beta", "gamma"]


def test_select_opponents_spreads_and_rotates() -> None:
    ladder = [f"m{index}" for index in range(8)]
    available = set(ladder) - {"m3"}
    assert P.select_opponents(ladder, available, 0, 2) == ["m0", "m4"]
    assert P.select_opponents(ladder, available, 1, 2) == ["m1", "m5"]
    assert P.select_opponents(ladder, available, 6, 2) == ["m7", "m2"]
    with pytest.raises(ValueError, match="need 9"):
        P.select_opponents(ladder, available, 0, 9)


def test_prepare_writes_rows_and_comparisons(stand_in: Path, tmp_path: Path) -> None:
    before = set(sys.modules)
    output = P.prepare_eqbench4(
        tmp_path / "out" / "rows.jsonl",
        tmp_path / "out" / "comparisons.jsonl",
        source_dir=stand_in,
        opponents_per_scenario=2,
        log=lambda message: None,
    )
    assert not {name for name in set(sys.modules) - before if name.split(".")[0] == "eqbench"}
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    comparisons = [json.loads(line) for line in (tmp_path / "out" / "comparisons.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and len(comparisons) == 3
    first = TaskData.model_validate(rows[0])
    assert rows[0]["responses_create_params"]["input"] == [
        {"role": "system", "content": "evaluated s1 turn 1/8"},
        {"role": "user", "content": '[Presenting issue (what begins this conversation): "issue s1"]'},
    ]
    assert first.evaluated_system_prompts[-1] == "evaluated s1 turn 8/8"
    assert first.persona_system_prompt == "persona of s1 for 8 turns"
    assert first.pairwise_dimensions == {"bond_and_rapport": 2.0, "attunement": 1.0}
    assert [opponent.model for opponent in first.opponents] == ["alpha", "beta"]
    assert [opponent.model for opponent in TaskData.model_validate(rows[1]).opponents] == ["beta", "gamma"]
    assert [opponent.comparison_index for opponent in TaskData.model_validate(rows[1]).opponents] == [2, 3]
    assert first.opponents[0].transcript[1].internal == "wary"

    pairwise = PromptParts(first.pairwise_prompt.literals, first.pairwise_prompt.slots).fill(
        {
            "first_code": "A1",
            "second_code": "B2",
            "first_transcript": "T1",
            "second_transcript": "T2",
            "first_final_state": "F1",
            "second_final_state": "F2",
        }
    )
    assert pairwise == (
        "friend|context:friend:True|issue s1|Label: Core\nDescription: deep|- clue: detail|Slow|"
        "state:['trusting']:None|hand:persona of s1 for 8 turns|\n"
        "A1: T1 [F1]\nB2: T2 [F2]\n"
        "1. **Bond And Rapport**: Rapport.\n\n2. **Attunement**: Reading.\n"
        '{\n  "bond_and_rapport": "winner_code+disparity or tie",\n'
        '  "attunement": "winner_code+disparity or tie"\n} ends A1'
    )
    traits = PromptParts(first.trait_prompt.literals, first.trait_prompt.slots).fill({"transcript": "<T>"})
    assert traits == "traits for s1\n[T]\n<T>[/T] final={} p=persona of s1 for 8 turns"
    assert first.trait_assistant_label == "Friend"


def test_prepare_rejects_bad_settings_and_scenarios(stand_in: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        P.prepare_eqbench4(tmp_path / "a", tmp_path / "b", source_dir=stand_in, opponents_per_scenario=0)
    files = {path: P.load_source(path, source_dir=stand_in) for path in P.SOURCE_SHA256 if "results/" not in path}
    with P.upstream_prompt_module(files) as prompts:
        renderer = P.ScenarioRenderer(prompts, files)
        with pytest.raises(ValueError, match="no core issue"):
            renderer.pairwise_prompt(_doc("s1") | {"core_issue": {}}, "persona")
        prompts.build_performance_judge_prompt = lambda **_: "no transcript here"
        with pytest.raises(ValueError, match="could not place"):
            renderer.trait_prompt(_doc("s1"), "persona")
