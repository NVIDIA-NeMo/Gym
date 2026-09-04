#!/usr/bin/env python3
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
"""Tests for harness patch 0008: in-gym grading of instruction-following constraints (swe_if_agents/if_constraints).

Run with a plain interpreter (no gym dependencies are needed by the grader):

    python3 responses_api_agents/swe_if_agents/tests/test_if_constraints_grading.py

or under pytest. The gym checkout is located from this file's own position when the file lives inside one, else from
the environment variable IFCD_GYM_DIR, else from the default path below. The parity checks read the two rolled-out
batches and the offline scorer's reports from the owner's run directory; when those files are absent the parity checks
report SKIP (and the run counts as failed, because the parity evidence is the point of the test).

Checks:
  (a) parity with the offline scorer `score_if.py` on the sdg5 and sdg10 batches: for every rolled-out row, embed the
      constraints of its params item into the row's metadata (the rows were built before constraints were embedded)
      and compare grade_row's output with the offline `if_scores.json` item: same number of constraints, same ids in
      order, same n_steps, n_pass, all_pass and the same ordered list of (turn, reward); the batch total of
      constraints must equal the offline aggregate. Also graded_turns == the offline n_graded_turns and
      continuation_only agrees with the offline prefix note.
  (b) grade_row returns None when metadata has no sdg_item and when sdg_item has no constraints;
  (c) grade_row returns the error record (not an exception) on malformed sdg_item JSON and on a constraint without a
      verifier_parameter;
  (d) the vendored verifier.py is byte-identical to the canonical logbook copy;
  (e) app.py compiles and carries the if_constraints field and the grade_row call.
"""
import copy
import json
import os
import sys
from pathlib import Path

DEFAULT_GYM_DIR = "/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/gym_workdir/gym_swe_if_agents"
LOGBOOK_RUN_DIR = (
    "/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/work/logbook/problems/P0000-one-off-task/experiments/"
    "E260823-agentic-if-understand-lin-work/runs/2026-08-31-constraint-design"
)
R7_RUN_DIR = "/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/work/data/runs/P0000-one-off-task/2026-09-02_r7-sdg-turn-output-samples"

RECIPE_DIR = os.environ.get("IFCD_RECIPE_DIR") or "/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/agentic-if/recipes/if-constraint-design"
CANONICAL_VERIFIER = f"{RECIPE_DIR}/verifier_impl/template_verifiers.py"   # the recipe IS the implementation (owner rule 2026-09-03); the logbook copy is deprecated

# (tag, params file, rolled-out results, offline scorer report)
BATCHES = [
    (
        "sdg5",
        f"{LOGBOOK_RUN_DIR}/constraints_trace_map/samples_turn_output_5.params.json",
        f"{R7_RUN_DIR}/scores_sdg5/results_sdg5.jsonl",
        f"{R7_RUN_DIR}/scores_sdg5/if_scores.json",
    ),
    (
        "sdg10",
        f"{LOGBOOK_RUN_DIR}/constraints_trace_map/samples_turn_output_10.params.json",
        f"{R7_RUN_DIR}/scores_sdg10/results_sdg10.jsonl",
        f"{R7_RUN_DIR}/scores_sdg10/if_scores.json",
    ),
]


def _gym_dir() -> Path:
    here = Path(os.path.abspath(__file__))
    for parent in here.parents:
        if (parent / "responses_api_agents" / "swe_if_agents" / "if_constraints" / "grader.py").exists():
            return parent
    env = os.environ.get("IFCD_GYM_DIR")
    return Path(env) if env else Path(DEFAULT_GYM_DIR)


GYM_DIR = _gym_dir()
if str(GYM_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_DIR))

from responses_api_agents.swe_if_agents.if_constraints import grade_row  # noqa: E402
from responses_api_agents.swe_if_agents.if_constraints.grader import GRADING_ERROR_ID  # noqa: E402

VENDORED_VERIFIER = GYM_DIR / "responses_api_agents" / "swe_if_agents" / "if_constraints" / "verifier.py"
APP_PY = GYM_DIR / "responses_api_agents" / "swe_if_agents" / "app.py"


# ------------------------------------------------------------------ helpers
def _embed_constraints(metadata: dict, param: dict) -> dict:
    """Return a copy of the row's metadata whose sdg_item follows the 2.0 row contract (type, phrasing_source, seed,
    prefix, constraint_ids, constraints), taking the constraints from the params item. The rolled-out rows were built
    before constraints were embedded, so their sdg_item carries only constraint_ids (and the legacy data_type)."""
    md = copy.deepcopy(metadata)
    legacy = json.loads(md["sdg_item"]) if md.get("sdg_item") else {}
    legacy_ids = legacy.get("constraint_ids")
    param_ids = [c["id"] for c in param["constraints"]]
    assert legacy_ids is None or legacy_ids == param_ids, (legacy_ids, param_ids)
    sdg_item = dict(legacy)
    sdg_item.pop("data_type", None)
    sdg_item.update(
        {
            "type": param["type"],
            "phrasing_source": param.get("phrasing_source", "template"),
            "seed": param["seed"],
            "prefix": param["prefix"],
            "constraint_ids": param_ids,
            "constraints": param["constraints"],
        }
    )
    md["sdg_item"] = json.dumps(sdg_item)
    return md


def _load_batch(tag, params_path, results_path, scores_path):
    if not (os.path.exists(params_path) and os.path.exists(results_path) and os.path.exists(scores_path)):
        return None
    params = {p["instance_id"]: p for p in json.load(open(params_path))["items"]}
    with open(results_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    scores = json.load(open(scores_path))
    offline = {it["instance_id"]: it for it in scores["items"]}
    return params, rows, scores, offline


def _steps_signature(constraint_record):
    return [(s["turn"], s["reward"]) for s in constraint_record["steps"]]


# ------------------------------------------------------------------ (a) parity with the offline scorer
def _check_parity(tag, params_path, results_path, scores_path):
    loaded = _load_batch(tag, params_path, results_path, scores_path)
    if loaded is None:
        return None, f"{tag}: SKIP (batch files not found)"
    params, rows, scores, offline = loaded
    n_constraints_total = 0
    n_rows = 0
    lines = []
    for row in rows:
        md = row["responses_create_params"]["metadata"]
        iid = md.get("instance_id")
        if iid not in params:
            sdg = json.loads(md["sdg_item"]) if md.get("sdg_item") else {}
            ids = sdg.get("constraint_ids") or []
            iid = ids[0].split("#")[0] if ids else None
        assert iid in params, f"{tag}: row without params item ({iid!r})"
        param = params[iid]
        # the offline scorer resolved tool identifiers with param['tool_names']; the gym uses the row's binding
        assert json.loads(md["tool_name_overrides"]) == param["tool_names"], f"{tag}/{iid}: row binding differs from params"
        metadata = _embed_constraints(md, param)
        records = grade_row(metadata, row["responses_create_params"]["input"], row["response"]["output"])
        assert isinstance(records, list), f"{tag}/{iid}: grade_row returned {type(records).__name__}"
        assert not (records and records[0].get("id") == GRADING_ERROR_ID), f"{tag}/{iid}: {records[0]}"
        expected = offline[iid]
        assert len(records) == len(expected["constraints"]) == len(param["constraints"]), (
            f"{tag}/{iid}: {len(records)} records vs {len(expected['constraints'])} offline constraints"
        )
        for got, exp in zip(records, expected["constraints"]):
            where = f"{tag}/{iid}/{exp['id']}"
            assert got["id"] == exp["id"], f"{where}: id {got['id']!r}"
            assert got["n_steps"] == exp["n_steps"], f"{where}: n_steps {got['n_steps']} vs {exp['n_steps']}"
            assert got["n_pass"] == exp["n_pass"], f"{where}: n_pass {got['n_pass']} vs {exp['n_pass']}"
            if exp["n_steps"]:
                assert got["all_pass"] == exp["all_pass"], f"{where}: all_pass {got['all_pass']} vs {exp['all_pass']}"
                assert got["step_avg"] == exp["step_avg"], f"{where}: step_avg {got['step_avg']} vs {exp['step_avg']}"
            else:
                # interface 3 defines all_pass as a boolean (False when the trigger never fired); the offline scorer
                # reports None for the same case. step_avg is None in both.
                assert got["all_pass"] is False and exp["all_pass"] is None, f"{where}: not-applicable encoding"
                assert got["step_avg"] is None and exp["step_avg"] is None, f"{where}: step_avg for no steps"
            assert _steps_signature(got) == _steps_signature(exp), (
                f"{where}: steps {_steps_signature(got)} vs {_steps_signature(exp)}"
            )
            assert got["trigger"] == exp["trigger"] and got["match"] == exp["match"], f"{where}: trigger/match"
            assert got["graded_turns"] == expected["n_graded_turns"], (
                f"{where}: graded_turns {got['graded_turns']} vs {expected['n_graded_turns']}"
            )
            offline_continuation = expected["n_prefix_turns_skipped"] > 0 or "continuation only" in expected["prefix_note"]
            assert got["continuation_only"] == offline_continuation, (
                f"{where}: continuation_only {got['continuation_only']} vs offline note {expected['prefix_note']!r}"
            )
            assert set(got) - {"error"} == {   # `error` is present only for a retired or unknown matcher (not applicable, row kept)
                "id", "trigger", "match", "instruction", "no_answer", "n_steps", "n_pass", "n_silent", "step_avg", "all_pass",
                "graded_turns", "continuation_only", "steps",
            }, f"{where}: record keys {sorted(got)}"
        n_constraints_total += len(records)
        n_rows += 1
        lines.append(
            f"    {iid:32s} {param['type']:10s} graded_turns={records[0]['graded_turns']:3d} "
            f"continuation_only={str(records[0]['continuation_only']):5s} "
            + " ".join(f"{c['n_pass']}/{c['n_steps']}" for c in records)
        )
    assert n_constraints_total == scores["aggregate"]["n_constraints"], (
        f"{tag}: {n_constraints_total} constraints vs offline aggregate {scores['aggregate']['n_constraints']}"
    )
    assert n_rows == len(offline), f"{tag}: {n_rows} rows graded vs {len(offline)} offline items"
    summary = f"{tag}: {n_rows} rows, {n_constraints_total} constraints identical to the offline scorer\n" + "\n".join(lines)
    return True, summary


def test_parity_sdg5():
    ok, msg = _check_parity(*BATCHES[0])
    assert ok, msg


def test_parity_sdg10():
    ok, msg = _check_parity(*BATCHES[1])
    assert ok, msg


# ------------------------------------------------------------------ (b) None when there is nothing to grade
def test_none_without_sdg_item():
    assert grade_row({}, [], []) is None
    assert grade_row(None, [], []) is None
    assert grade_row({"instance_id": "x", "tool_name_overrides": "{}"}, [], []) is None


def test_none_without_constraints():
    md = {"sdg_item": json.dumps({"type": "fresh", "constraint_ids": [], "seed": 1, "prefix": None})}
    assert grade_row(md, [], []) is None
    md = {"sdg_item": json.dumps({"type": "fresh", "constraints": []})}
    assert grade_row(md, [], []) is None


# ------------------------------------------------------------------ (c) error record instead of an exception
def test_error_record_on_malformed_sdg_item():
    out = grade_row({"sdg_item": "{not valid json"}, [], [])
    assert isinstance(out, list) and len(out) == 1, out
    assert out[0]["id"] == GRADING_ERROR_ID and "error" in out[0] and out[0]["error"], out


def test_error_record_on_constraint_without_verifier_parameter():
    md = {"sdg_item": json.dumps({"type": "fresh", "constraints": [{"id": "x#c1"}]})}
    out = grade_row(md, [], [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}])
    assert isinstance(out, list) and len(out) == 1 and out[0]["id"] == GRADING_ERROR_ID, out


# ------------------------------------------------------------------ small synthetic checks of the record shape
def _synthetic_row():
    constraints = [
        {"id": "t#c1", "verifier_parameter": {"template": "turn_output", "trigger": {"position": "final"},
                                              "obligation": {"match": "exact", "value": "DONE"}},
         "reference_instruction": "End your final message with exactly DONE.", "surface": "system_prompt", "position": None},
        {"id": "t#c2", "verifier_parameter": {"template": "turn_output", "trigger": {"tool": "GREP_TOOL_NAME"},
                                              "obligation": {"match": "prefix", "value": "[SEARCH]"}},
         "reference_instruction": "Start every message that calls the search tool with [SEARCH].", "surface": "system_prompt", "position": None},
    ]
    md = {
        "tool_name_overrides": json.dumps({"GREP_TOOL_NAME": "find_pattern", "BASH_TOOL_NAME": "shell"}),
        "sdg_item": json.dumps({"type": "fresh", "phrasing_source": "template", "seed": 1, "prefix": None,
                                "constraint_ids": [c["id"] for c in constraints], "constraints": constraints}),
    }
    output = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "[SEARCH] looking"}]},
        {"type": "function_call", "name": "find_pattern", "arguments": json.dumps({"pattern": "x"}), "call_id": "a"},
        {"type": "function_call_output", "call_id": "a", "output": "found"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "not tagged"}]},
        {"type": "function_call", "name": "find_pattern", "arguments": json.dumps({"pattern": "y"}), "call_id": "b"},
        {"type": "function_call_output", "call_id": "b", "output": "found"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "DONE"}]},
    ]
    return md, output


def test_synthetic_record_shape_and_binding():
    md, output = _synthetic_row()
    out = grade_row(md, [{"type": "message", "role": "system", "content": "s"}, {"type": "message", "role": "user", "content": "u"}], output)
    assert [c["id"] for c in out] == ["t#c1", "t#c2"], out
    final, grep = out
    assert final["n_steps"] == 1 and final["n_pass"] == 1 and final["all_pass"] is True and final["step_avg"] == 1.0, final
    assert final["graded_turns"] == 3 and final["continuation_only"] is False, final
    # the search tool is bound to `find_pattern`; both calls fire the trigger, only the first message is tagged
    assert grep["n_steps"] == 2 and grep["n_pass"] == 1 and grep["all_pass"] is False and grep["step_avg"] == 0.5, grep
    assert [(s["turn"], s["reward"]) for s in grep["steps"]] == [(0, 1), (1, 0)], grep["steps"]
    # without the binding the identifier resolves to the default name `grep`, which never appears: not applicable
    md_unbound = dict(md)
    md_unbound.pop("tool_name_overrides")
    out2 = grade_row(md_unbound, [], output)
    assert out2[1]["n_steps"] == 0 and out2[1]["step_avg"] is None and out2[1]["all_pass"] is False, out2[1]


def test_continuation_turn_indices_relative_to_continuation():
    """A prefix item whose output is the continuation only: the first graded turn has index 0, and a prefix that is
    reproduced at the head of the output is skipped, leaving the same continuation."""
    md, output = _synthetic_row()
    sdg = json.loads(md["sdg_item"])
    sdg["type"] = "interject"
    md["sdg_item"] = json.dumps(sdg)
    prefix = [
        {"type": "message", "role": "system", "content": "s"},
        {"type": "message", "role": "user", "content": "u"},
        {"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "p1"},
        {"type": "function_call_output", "call_id": "p1", "output": "ok"},
    ]
    out = grade_row(md, prefix, output)
    assert all(c["continuation_only"] is True and c["graded_turns"] == 3 for c in out), out
    assert [(s["turn"], s["reward"]) for s in out[1]["steps"]] == [(0, 1), (1, 0)], out[1]["steps"]
    # same continuation, but the recorded output reproduces the replayed prefix at its head
    out_with_prefix = grade_row(md, prefix, prefix[2:] + output)
    assert all(c["continuation_only"] is True and c["graded_turns"] == 3 for c in out_with_prefix), out_with_prefix
    assert [(s["turn"], s["reward"]) for s in out_with_prefix[1]["steps"]] == [(0, 1), (1, 0)], out_with_prefix[1]["steps"]


# ------------------------------------------------------------------ (d) vendored verifier parity
def test_vendored_verifier_is_byte_identical():
    assert VENDORED_VERIFIER.exists(), VENDORED_VERIFIER
    assert os.path.exists(CANONICAL_VERIFIER), CANONICAL_VERIFIER
    assert VENDORED_VERIFIER.read_bytes() == Path(CANONICAL_VERIFIER).read_bytes(), "vendored template_verifiers.py differs from the canonical copy"


# ------------------------------------------------------------------ (e) app.py wiring (static)
def test_app_py_wiring():
    """The swe_if_agents wrapper grades in run() and attaches ONLY if_constraints; the outcome reward is untouched."""
    src = APP_PY.read_text()
    assert "class SWEIFVerifyResponse(swe.SWEBenchVerifyResponse)" in src
    assert "if_constraints: Optional[List[Dict[str, Any]]] = None" in src
    assert "records = grade_row(" in src and "if_constraints=records" in src
    assert "reward=" not in src.split("async def run(")[1], "run() must not recompute the reward"
    assert "resolved_agent_env" in src and "write_row_templates(" in src and "tag_replay_observation_suffix(" in src



# ------------------------------------------------------------------ plain-python runner
def main() -> int:
    checks = [
        ("(a) parity sdg5", lambda: _check_parity(*BATCHES[0])),
        ("(a) parity sdg10", lambda: _check_parity(*BATCHES[1])),
        ("(b) None without sdg_item", test_none_without_sdg_item),
        ("(b) None without constraints", test_none_without_constraints),
        ("(c) error record on malformed sdg_item", test_error_record_on_malformed_sdg_item),
        ("(c) error record on constraint without verifier_parameter", test_error_record_on_constraint_without_verifier_parameter),
        ("synthetic record shape and tool binding", test_synthetic_record_shape_and_binding),
        ("continuation turn indices", test_continuation_turn_indices_relative_to_continuation),
        ("(d) vendored verifier byte-identical", test_vendored_verifier_is_byte_identical),
        ("(e) app.py wiring", test_app_py_wiring),
    ]
    n_pass = n_fail = 0
    print(f"gym checkout: {GYM_DIR}")
    for name, fn in checks:
        try:
            result = fn()
            if isinstance(result, tuple):
                ok, msg = result
                if ok is None:
                    print(f"FAIL {name}: {msg}")
                    n_fail += 1
                    continue
                print(f"PASS {name}: {msg}")
            else:
                print(f"PASS {name}")
            n_pass += 1
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}")
            n_fail += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed" + ("  -> ALL PASS" if n_fail == 0 else ""))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
