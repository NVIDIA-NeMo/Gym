# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise frozen-tool semantics using invented public data only."""

import copy
import hashlib
import json
from uuid import uuid4

import pytest

from benchmarks.partner_if.tests.fixture_data import BUNDLE, PARAMS, RUNTIME, SOURCE
from responses_api_agents.ngc_recorded_context.readonly_tools import CoverageGap, ReadonlyTools
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, load_inputs, load_runtime
from responses_api_agents.ngc_recorded_context.replay_fixtures import Fixtures
from responses_api_agents.ngc_recorded_context.runtime_contract import RuntimeContract


def state(**options):
    _, bundle = load_inputs(
        ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir="unused", coverage_profile="expanded-v4", **options),
        PARAMS,
    )
    return Fixtures(bundle, session_id=uuid4().hex)


@pytest.mark.parametrize("status", [200, 404])
def test_wrapper_and_response_namespace_are_data_not_runner_constants(status):
    wrapper = "/opt/fictional/lookup.sh"
    prefix = "GET:service.example.invalid"
    path = "/api/v1/items/widget"
    key = hashlib.sha256((prefix + path).encode()).hexdigest()[:12]
    tools = ReadonlyTools({}, {key: {"status": status, "body": "recorded body"}}, get_wrappers={wrapper: prefix})
    stdout, stderr, code = tools.command([wrapper, path], "")
    expected = (
        ("recorded body", "", 0) if status == 200 else ("", "curl: (22) The requested URL returned error: 404\n", 22)
    )
    assert (stdout, stderr, code) == expected
    with pytest.raises(CoverageGap):
        tools.command(["/opt/unregistered/lookup.sh", path], "")


def test_unknown_http_response_remains_a_coverage_gap():
    tools = ReadonlyTools({}, {}, get_wrappers={"/tool/get": "GET:example.invalid"})
    with pytest.raises(CoverageGap):
        tools.command(["/tool/get", "/api/v1/missing"], "")


def test_optional_missing_runtime_paths_are_declared_not_invented():
    tools = ReadonlyTools({}, {}, optional_runtime_paths=["/workspace/optional.json"], missing_runtime_reads=True)
    assert tools.command(["cat", "/workspace/optional.json"], "")[2] == 1
    with pytest.raises(CoverageGap):
        tools.command(["cat", "/workspace/uncaptured.json"], "")


@pytest.mark.parametrize(
    "command,expected",
    [
        ("false && echo hidden; echo visible", "visible\n"),
        ("true || echo hidden; echo visible", "visible\n"),
        ("false || echo recovered", "recovered\n"),
    ],
)
def test_readonly_conditionals_preserve_exit_status(command, expected):
    from responses_api_agents.ngc_recorded_context.replay_fixtures import shell_tokens

    assert state(shell_control_flow=True).readonly.run(shell_tokens(command))[0] == expected


def test_source_search_uses_actual_frozen_bytes_and_rejects_shell_options():
    tool = state().readonly
    text, stderr, code = tool.search(["-n", "generate_manifest", RUNTIME["source_root"]], "")
    assert code == 0 and not stderr and "service.py" in text and "generate_manifest" in text
    with pytest.raises(CoverageGap):
        tool.search(["--pre=touch /tmp/escape", "x", RUNTIME["source_root"]], "")
    with pytest.raises(CoverageGap):
        tool.search(["x", "/etc/passwd"], "")


def test_search_requires_complete_content_but_inventory_listing_does_not():
    inventory = {"root": "/repo", "paths": ["a.py", "b.py", "ignored.md"]}
    tool = ReadonlyTools({"/repo/a.py": "no match"}, {}, source_inventory=inventory, workspace_listing=True)
    with pytest.raises(CoverageGap, match="not captured"):
        tool.search(["needle", "/repo"], "")
    with pytest.raises(CoverageGap, match="not captured"):
        tool.search(["needle", "/repo/*.py"], "")
    assert tool.search(["needle", "/repo/a.py"], "")[2] == 1
    assert set(tool.search(["--files", "/repo"], "")[0].splitlines()) == {
        "/repo/a.py",
        "/repo/b.py",
        "/repo/ignored.md",
    }
    tool.files["/repo/b.py"] = "needle"
    # The missing markdown file is excluded by rg's actual glob semantics.
    stdout, stderr, code = tool.search(["-g", "*.py", "needle", "/repo"], "")
    assert code == 0 and not stderr and "needle" in stdout
    with pytest.raises(CoverageGap, match="not captured"):
        tool.search(["needle", "/repo"], "")
    tool.files["/repo/ignored.md"] = "nothing"
    assert tool.search(["needle", "/repo"], "")[2] == 0


def test_generated_transcript_contains_actual_messages_only():
    fixture = state()
    messages = [{"role": "user", "content": "synthetic request"}]
    fixture.sync_transcript(messages)
    record = json.loads(fixture.files[fixture.transcript_path])
    assert record["message"] == messages[0] and record["type"] == "api_replay_message"
    assert fixture.files[RUNTIME["session_id_path"]] == fixture.session_id + "\n"


def test_unavailable_source_and_uncaptured_existing_file_are_distinct():
    tool = state().readonly
    assert tool.known_missing_source(RUNTIME["source_root"] + "/not-present.py")
    assert not tool.known_missing_source(RUNTIME["source_root"] + "/frontend/app.ts")
    assert tool.list_files([RUNTIME["source_root"] + "/frontend"]) == "app.ts\n"


def test_workflow_replay_preserves_recorded_partial_history():
    fixture = state(temporal_history=True)
    text, kind = fixture.bash(
        "temporal --address workflow.example.invalid:7233 --namespace public-example workflow show --workflow-id example-workflow --run-id example-run --output json"
    )
    data = json.loads(text)
    assert kind == "temporal_fixture_result"
    assert data["_replay"]["captured_event_count"] == 1 and data["_replay"]["reported_history_length"] == 2
    assert data["_replay"]["complete_history"] is False
    assert fixture.bash("temporal --address other.example.invalid workflow show --output json")[1] == "fixture_gap"


def test_forked_skill_is_not_falsely_reported_as_executed():
    fixture = state()
    path = fixture.bundle["skills"]["registry-access"]
    fixture.files[path] = "---\ncontext: fork\n---\nRun a private reviewer."
    result, kind = fixture.dispatch(
        {"function": {"name": "Skill", "arguments": json.dumps({"skill": "registry-access"})}},
        {"Skill": {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]}},
    )
    assert kind == "fixture_gap" and "not modeled" in result


@pytest.mark.parametrize("fault", ["traversal", "digest", "schema", "unknown", "answer", "template"])
def test_runtime_contract_rejects_ambiguous_or_unsafe_configuration(fault):
    value = copy.deepcopy(RUNTIME)
    if fault == "traversal":
        value["input_hashes"]["../escape"] = "0" * 64
    elif fault == "digest":
        value["input_hashes"]["case.json"] = "not-a-digest"
    elif fault == "schema":
        value["source_schemas"]["invented"] = "x"
    elif fault == "unknown":
        value["unknown_control"] = True
    elif fault == "answer":
        value["answer_paths"] = ["/outside/answer.txt"]
    else:
        value["transcript_template"] = "/workspace/missing-placeholder.jsonl"
    with pytest.raises(ValueError):
        RuntimeContract.model_validate(value)


def test_external_contract_requires_a_pin_and_checks_it_before_inputs(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(RUNTIME))
    config = ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir=tmp_path, runtime_contract_path=path)
    with pytest.raises(ValueError, match="requires runtime_contract_sha256"):
        load_runtime(config)
    config = config.model_copy(update={"runtime_contract_sha256": "0" * 64})
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_runtime(config)
    config = config.model_copy(update={"runtime_contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert load_runtime(config).source_root == RUNTIME["source_root"]


def test_source_checkout_is_not_a_supported_runtime_dependency():
    with pytest.raises(ValueError):
        ReplayConfig(package_dir=SOURCE / "assets", artifacts_dir="unused", agentic_if_root="../private")


@pytest.mark.parametrize("identifier", ["../outside", "", "a" * 33, "g" * 32])
def test_session_identifier_cannot_select_host_paths(identifier):
    with pytest.raises(ValueError, match="Invalid replay session ID"):
        Fixtures(BUNDLE, session_id=identifier)


@pytest.mark.parametrize("identifier", [None, "", "different_criterion"])
def test_ordering_result_must_match_the_declared_criterion(identifier):
    from responses_api_agents.ngc_recorded_context.prefix import ordering_outcome

    graded = {"reward": 1.0, "mask_sample": False, "if_result": {"items": [{"id": identifier, "passed": True}]}}
    with pytest.raises(ValueError):
        ordering_outcome(graded, [], expected_criterion_id="check_before_advice")


def test_alternative_criterion_id_works_without_case_specific_source():
    from responses_api_agents.ngc_recorded_context.prefix import ordering_outcome

    graded = {
        "reward": 1.0,
        "mask_sample": False,
        "if_result": {"items": [{"id": "check_before_advice", "passed": True}]},
    }
    events = [{"verified_probe": {"namespace": "neighbor"}}]
    assert (
        ordering_outcome(graded, events, probe_namespace="neighbor", expected_criterion_id="check_before_advice")
        == "ordering_passed"
    )
