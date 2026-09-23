# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic in-memory ASGI tests. No source execution, provider, sockets or model calls.

The grader is mocked, so these tests exercise HTTP policy, not sandbox authenticity
or reference-before-candidate allocation order. Those require execution qualification.
Contributed YAML/JSONL checks validate schemas and opaque tagged structure only, not
source/value semantics or effective runtime privacy settings.
"""

import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi.exceptions import RequestValidationError
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym import failure_kinds, server_utils
from nemo_gym.config_types import DatasetConfig, ResourcesServerTypeConfig, ResponsesAPIAgentServerTypeConfig
from nemo_gym.server_metadata import visit_resources_server
from nemo_gym.server_utils import ServerClient
from nemo_gym.task_data import normalize_task_fields
from resources_servers.critpt_custom_grader import app as adapter
from resources_servers.critpt_custom_grader import task_data as task_schema
from resources_servers.critpt_custom_grader.execution import (
    DOMAINS,
    LABEL_DOMAIN,
    LABEL_JOB,
    LABEL_OWNER,
    ExecutionPolicy,
    Grader,
    GradeResult,
    OwnershipJournal,
    SandboxHandle,
    SandboxLookup,
)
from resources_servers.critpt_custom_grader.server_support import (
    ServerRequestPolicy,
    ServerRequestPrivacy,
)
from resources_servers.critpt_custom_grader.task_data import TaskData
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


SOURCE = "def solve(value):\n    return value\n"
REFERENCE_MARKER = "synthetic_reference_marker"
EXPECTED_MARKER = "synthetic_expected_marker"
NONTERMINAL_CATEGORIES = (
    "reference_error",
    "reference_timeout",
    "comparator_uncertain",
    "provider_create",
    "provider_exec",
    "transfer_limit",
    "result_invalid",
    "candidate_no_result",
    "reference_no_result",
    "comparator_no_result",
    "job_deadline",
    "ownership_unresolved",
    "busy",
)


def task_fields():
    return {
        "problem_id": "synthetic-1",
        "reference_source": f"# {REFERENCE_MARKER}\n{SOURCE}",
        "entrypoint": "solve",
        "test_cases": [{"input": [1], "output": EXPECTED_MARKER}],
    }


def saved_response(text=SOURCE):
    return {
        "id": "synthetic-response",
        "created_at": 1.0,
        "model": "synthetic-model",
        "object": "response",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        "output": [
            {
                "id": "synthetic-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def payload(**task_overrides):
    return {
        "responses_create_params": {"input": "Return a Python function named solve."},
        **task_fields(),
        **task_overrides,
        "response": saved_response(),
    }


def verdict(**overrides):
    result = GradeResult(
        outcome="passed",
        reward=1.0,
        category="passed",
        failure_class=None,
        terminal=None,
        job_id="synthetic-job",
        case_count=1,
        cases_attempted=1,
        cases_equal=1,
        first_failed_case=None,
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "deleted"},
    )
    return replace(result, **overrides)


def unscorable_verdict(**overrides):
    fields = {
        "outcome": "unscorable",
        "reward": None,
        "category": "comparator_uncertain",
        "failure_class": failure_kinds.PROVIDER_UNAVAILABLE,
        "terminal": False,
        "cases_attempted": 0,
        "cases_equal": 0,
    }
    fields.update(overrides)
    return verdict(**fields)


@pytest.fixture(autouse=True)
def no_runtime_work(monkeypatch):
    blocked = MagicMock(side_effect=AssertionError("host schema/dispatch check attempted runtime work"))
    assert not hasattr(task_schema, "_input_literal")
    assert not hasattr(task_schema, "ast")
    # Runtime modules may already be loaded by controller imports. Do not import them for this check.
    targets = {
        "resources_servers.critpt_custom_grader.runner": (
            "parse_source",
            "_input_literal",
            "_decode_input",
            "_decode_cases",
            "materialize_input",
            "_symbolic_input",
            "adapt_output",
            "_invoke",
            "_run_worker",
            "_compare_worker",
            "execute_run",
            "execute_compare",
        ),
        "resources_servers.critpt_custom_grader.codec": ("decode_value", "encode_value", "load_value", "dump_value"),
        "resources_servers.critpt_custom_grader.comparator": ("compare_request",),
        "resources_servers.critpt_custom_grader.symbolic": ("parse_expression",),
    }
    for name, functions in targets.items():
        module = sys.modules.get(name)
        if module is not None:
            for function in functions:
                monkeypatch.setattr(module, function, blocked)
            if hasattr(module, "ast"):
                # Guard grader lookups, including saved function aliases, without breaking inspect or pytest.
                grader_ast = SimpleNamespace(**vars(module.ast))
                grader_ast.parse = grader_ast.literal_eval = blocked
                monkeypatch.setattr(module, "ast", grader_ast)
    yield
    blocked.assert_not_called()


@pytest.fixture
def make_server(monkeypatch, tmp_path):
    def make(**overrides):
        policy = ExecutionPolicy(
            snapshot="synthetic-pinned-snapshot",
            os_user="grader",
            owner_id="synthetic",
            journal_dir=str(tmp_path / "unused-journal"),
        )
        overrides.setdefault("request_policy", {"no_resubmission": True, "deadline_seconds": 20_000.0})
        overrides.setdefault("request_privacy", {"private_requests": True})
        config = adapter.CritPtCustomGraderConfig(
            host="127.0.0.1",
            port=8000,
            name="synthetic-grader",
            entrypoint="app.py",
            execution=policy,
            **overrides,
        )
        fake = MagicMock()
        fake.grade = AsyncMock(return_value=verdict())
        fake.shutdown = AsyncMock(return_value=[])
        # A resolved (empty) reconcile by default: a bare MagicMock attribute is not awaitable.
        fake.reconcile = AsyncMock(return_value=[])
        factory = MagicMock(return_value=fake)
        monkeypatch.setattr(adapter, "Grader", factory)
        server = adapter.CritPtCustomGraderServer(config=config, server_client=MagicMock(spec=ServerClient))
        return server, server.setup_webserver(), fake, factory

    return make


async def post(app, path, value=None, *, raw=None):
    """Exercise the ASGI parser/router/serializer without an HTTP client or sockets."""
    body = raw if raw is not None else json.dumps(value, allow_nan=False).encode()
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "http.request", "body": body, "more_body": False})
    messages = []

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("in-memory", 1),
            "server": ("in-memory", 80),
        },
        incoming.get,
        send,
    )
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    output = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, json.loads(output)


async def until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    pytest.fail("mocked operation did not reach its expected state")


def assert_unscorable(result, *, terminal=False):
    assert result["reward"] == 0.0
    assert result["scored"] is False
    assert result["_ng_failure_class"] == ("verifier_error" if terminal else "provider_unavailable")
    if terminal:
        assert result["_ng_failure_terminal"] is True
    else:
        assert "_ng_failure_terminal" not in result


@pytest.mark.asyncio
async def test_direct_verify_and_frozen_replay(make_server, capsys, caplog):
    server, app, fake, factory = make_server()
    request = payload()
    request["response"]["metadata"] = {"synthetic": "replay"}
    original = deepcopy(request)
    factory.assert_not_called()

    status, result = await post(app, "/verify", request)
    assert status == 200
    assert result["reward"] == 1.0 and result["scored"] is True
    assert result["response"] == original["response"]
    assert result["responses_create_params"] == original["responses_create_params"]
    assert "_ng_failure_class" not in result
    assert result["cleanup_status"] == "resolved"
    assert result["problem_id"] == "synthetic-1"
    assert request == original
    validated, source = fake.grade.await_args.args
    assert isinstance(validated, TaskData)
    assert validated.entrypoint == "solve"
    assert validated.reference_source == original["reference_source"]
    assert source == SOURCE.strip()
    assert not server.server_client.mock_calls

    # Recovery uses the materialized input plus the saved response, not response-only state.
    fresh, fresh_app, fresh_fake, _ = make_server()
    replay = {**original, "response": result["response"]}
    status, recovered = await post(fresh_app, "/verify", replay)
    assert status == 200
    assert recovered == result
    assert fresh_fake.grade.await_args.args[1] == source
    assert not fresh.server_client.mock_calls
    rendered = json.dumps(result) + capsys.readouterr().out + caplog.text
    assert REFERENCE_MARKER not in rendered
    assert EXPECTED_MARKER not in rendered
    assert "synthetic-pinned-snapshot" not in rendered
    assert "synthetic-job" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["flat", "verifier_metadata", "task_data"])
async def test_seed_validates_without_allocating_or_storing_private_data(make_server, placement):
    server, app, fake, factory = make_server()
    row = {"responses_create_params": {"input": "Public synthetic prompt"}}
    row.update(task_fields() if placement == "flat" else {placement: task_fields()})
    row.update({"agent_ref": {"name": "synthetic-agent"}, "task_source": "synthetic", "_ng_task_index": 4})
    status, result = await post(app, "/seed_session", row)
    assert (status, result) == (200, {})
    factory.assert_not_called()
    fake.grade.assert_not_awaited()
    assert server._grader is None
    assert not server.server_client.mock_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"entrypoint": ""},
        {"test_cases": []},
        {"entry": "solve"},
        {"execution": {"provider": "local"}},
        {"snapshot": "task-owned"},
        {"unexpected": REFERENCE_MARKER},
        {"_ng_failure_class": "judge_failed", "_ng_failure_terminal": False, "reward": 1.0},
        {"verifier_metadata": {"entrypoint": "different"}},
        {"verifier_metadata": {"responses_create_params": {"input": "nested runtime field"}}},
    ],
)
async def test_invalid_task_is_terminal_before_allocation(make_server, changes):
    _, app, fake, factory = make_server()
    status, result = await post(app, "/verify", payload(**changes))
    assert status == 200
    assert_unscorable(result, terminal=True)
    assert result["category"] == "task_invalid"
    assert result["response"] == saved_response()
    assert "unexpected" not in result and "execution" not in result
    fake.grade.assert_not_awaited()
    factory.assert_not_called()
    seed = payload(**changes)
    seed.pop("response")
    assert await post(app, "/seed_session", seed) == (422, {"error": {"category": "task_invalid"}})


@pytest.mark.asyncio
async def test_equal_flat_and_legacy_values_use_gym_normalization(make_server):
    _, app, fake, _ = make_server()
    row = payload(verifier_metadata=task_fields())
    assert (await post(app, "/verify", row))[1]["scored"] is True
    assert fake.grade.await_args.args[0].problem_id == "synthetic-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/verify", "/seed_session"])
@pytest.mark.parametrize("malformation", ["json", "model", "tools"])
async def test_pre_endpoint_privacy_survives_late_handler_registration(
    make_server, path, malformation, capsys, caplog
):
    server, app, fake, factory = make_server()
    # Match the important run_webserver order, including its late global handler.
    server.setup_exception_middleware(app)
    server.setup_cancellation_middleware(app)
    server.instrument_app_for_telemetry(app)
    late_handler = AsyncMock(side_effect=AssertionError("late handler must not receive the body"))
    app.add_exception_handler(RequestValidationError, late_handler)
    row = payload()
    if path == "/seed_session":
        row.pop("response")
    if malformation == "json":
        status, result = await post(app, path, raw=b'{"private":"synthetic_reference_marker",')
    else:
        if malformation == "model":
            row["responses_create_params"] = {"input": {"private": REFERENCE_MARKER}}
        else:
            row["responses_create_params"]["tools"] = [
                {"type": "function", "name": "verify", "description": REFERENCE_MARKER, "parameters": {}}
            ]
        status, result = await post(app, path, row)
    assert (status, result) == (422, {"error": {"category": "request_invalid"}})
    late_handler.assert_not_awaited()
    fake.grade.assert_not_awaited()
    factory.assert_not_called()
    captured = capsys.readouterr()
    assert REFERENCE_MARKER not in captured.out + captured.err + caplog.text + json.dumps(result)


@pytest.mark.asyncio
async def test_malformed_saved_response_and_ingress_limit(make_server, monkeypatch):
    _, app, fake, _ = make_server()
    row = payload()
    row["response"] = {"private": REFERENCE_MARKER}
    assert await post(app, "/verify", row) == (422, {"error": {"category": "request_invalid"}})
    monkeypatch.setattr(adapter, "MAX_REQUEST_BYTES", 64)
    assert await post(app, "/verify", raw=b" " * 65) == (413, {"error": {"category": "request_limit"}})
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_number_materialization_and_http_round_trip(make_server):
    _, app, fake, _ = make_server()
    large = 10**100 + 3
    fields = task_fields()
    fields["test_cases"] = [{"input": [large], "output": large}]
    normalized, conflicts = normalize_task_fields({"verifier_metadata": fields, "_ng_task_index": 0})
    assert not conflicts
    materialized = TaskData.model_validate(normalized).model_dump(mode="json")
    # No comparator/codec execution: schema normalization alone creates the tags.
    persisted = orjson.loads(orjson.dumps(materialized))
    assert persisted["test_cases"][0]["args"][0]["value"] == ["int", str(large)]
    for task in (fields, persisted):
        row = {**payload(), **task}
        status, result = await post(app, "/verify", row)
        assert status == 200 and result["scored"] is True
        assert result["response"] == row["response"]
        delivered = fake.grade.await_args.args[0].model_dump(mode="json")
        assert delivered == materialized

    tags = [
        ["decimal", "1.0000000000000000000000000000000001"],
        ["fraction", "7", "19"],
        ["int", str(large)],
        ["nonfinite", "inf"],
    ]
    for node in tags:
        tagged = {"format": "critpt-value-v1", "value": node}
        case = {"args": [tagged], "expected": {"kind": "value", "value": tagged}}
        row = payload(test_cases=[case])
        assert (await post(app, "/verify", row))[0] == 200
        assert fake.grade.await_args.args[0].test_cases[0].args[0].value == node


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [" 1.25\n", "[True, False]", "'green'", "(2, 3)", "pi", "[", "(lambda: 1)()", "1/7", "1_000", "1e-999"],
)
async def test_keyword_input_text_reaches_only_mocked_grader(make_server, text):
    server, app, fake, factory = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    expected = f"  {EXPECTED_MARKER}: [1, 2]\n"
    row = payload(test_cases=[{"inputs": {"value": text}, "expected_output": expected}])
    original = deepcopy(row)
    seed = {key: value for key, value in row.items() if key != "response"}
    assert await post(app, "/seed_session", seed) == (200, {})
    factory.assert_not_called()
    fake.grade.assert_not_awaited()
    status, result = await post(app, "/verify", row)
    assert status == 200
    assert_unscorable(result)  # Dispatch alone establishes no value semantics.
    validated, source = fake.grade.await_args.args
    assert validated.test_cases[0].kwargs["value"].value == ["legacy", text]
    assert validated.test_cases[0].expected.value.value == ["legacy", expected]
    assert source == SOURCE.strip() and row == original
    assert not server.server_client.mock_calls
    assert EXPECTED_MARKER not in json.dumps(result) and REFERENCE_MARKER not in json.dumps(result)


@pytest.mark.asyncio
async def test_keyword_shape_comparison_type_reaches_only_mocked_grader(make_server):
    # The extra comparison_type key is accepted, dropped, and never a value signal.
    _, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    row = payload(test_cases=[{"inputs": {"value": "1"}, "expected_output": "2", "comparison_type": "numeric"}])
    status, result = await post(app, "/verify", row)
    assert status == 200
    validated = fake.grade.await_args.args[0]
    assert validated.test_cases[0].kwargs["value"].value == ["legacy", "1"]
    assert validated.test_cases[0].expected.value.value == ["legacy", "2"]


@pytest.mark.asyncio
async def test_complex_carrier_reaches_grader_as_a_complex_node(make_server):
    # A stored {"__complex__": [re, im]} carrier on input and expected becomes the complex wire node.
    _, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    row = payload(test_cases=[{"input": [{"__complex__": [1, 2.0]}], "output": {"__complex__": [3, 4]}}])
    status, result = await post(app, "/verify", row)
    assert status == 200
    validated = fake.grade.await_args.args[0]
    assert validated.test_cases[0].args[0].value == ["complex", ["int", "1"], ["float", (2.0).hex()]]
    assert validated.test_cases[0].expected.value.value == ["complex", ["int", "3"], ["int", "4"]]


@pytest.mark.asyncio
async def test_input_conversions_reach_the_grader_task(make_server):
    _, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    row = payload(input_conversions=[None, "symbol", "function"])
    status, result = await post(app, "/verify", row)
    assert status == 200
    assert fake.grade.await_args.args[0].input_conversions == [None, "symbol", "function"]


@pytest.mark.asyncio
@pytest.mark.parametrize("node", [["unknown", "bounded"], ["fraction", "1"], ["int", "not-an-integer"]])
async def test_bounded_wire_transport_does_not_establish_semantic_acceptance(make_server, node):
    _, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    tagged = {"format": task_schema.WIRE_FORMAT, "value": node}
    row = payload(test_cases=[{"args": [tagged], "expected": {"kind": "exception"}}])
    status, result = await post(app, "/verify", row)
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == "reference_error"
    assert fake.grade.await_args.args[0].test_cases[0].args[0].value == node


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["candidate_mismatch", "candidate_invalid_output"])
async def test_established_failed_verdict_is_scored_without_failure_keys(make_server, category):
    _, app, fake, _ = make_server()
    fake.grade.return_value = verdict(outcome="failed", reward=0.0, category=category, cases_equal=0)
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["scored"] is True and result["reward"] == 0.0
    assert "_ng_failure_class" not in result
    assert result["category"] == category


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["task_invalid", "reference_mismatch"])
async def test_explicit_failed_preflight_is_terminal(make_server, category):
    _, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        outcome="unscorable",
        reward=None,
        category=category,
        failure_class=failure_kinds.VERIFIER_ERROR,
        terminal=True,
        cases_attempted=0,
        cases_equal=0,
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "not_created"},
    )
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result, terminal=True)
    assert result["category"] == "reference_invalid"
    assert fake.grade.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("category", NONTERMINAL_CATEGORIES)
async def test_nonterminal_execution_category_is_preserved_without_fault(make_server, category):
    server, app, fake, factory = make_server()
    fake.grade.return_value = unscorable_verdict(category=category)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == category
    assert result["cleanup_status"] == "resolved"
    assert result["response"] == saved_response()
    factory.assert_called_once_with(
        server.config.execution,
        default_rtol=server.config.comparison.default_rtol,
        default_atol=server.config.comparison.default_atol,
    )
    fake.grade.assert_awaited_once()
    await until(lambda: not server._jobs)
    assert server._pending == 0 and not server._lock.locked() and not server._blocked
    assert not server.server_client.mock_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category,attempted,cleanup",
    [
        (
            "reference_error",
            0,
            {"reference": "deleted", "comparator": "not_created", "candidate": "not_created"},
        ),
        (
            "provider_create",
            0,
            {"reference": "create_failed_clean", "comparator": "not_created", "candidate": "not_created"},
        ),
        (
            "comparator_uncertain",
            2,
            {"reference": "absent", "comparator": "deleted", "candidate": "deleted"},
        ),
        (
            "comparator_no_result",
            2,
            {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"},
        ),
        (
            "provider_exec",
            1,
            {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"},
        ),
        ("busy", 0, {}),
    ],
)
async def test_nonterminal_counts_and_resolved_cleanup_shapes(make_server, category, attempted, cleanup):
    server, app, fake, _ = make_server()
    row = payload(test_cases=[{"input": [1], "output": 1}, {"input": [2], "output": 2}])
    fake.grade.return_value = unscorable_verdict(
        category=category, case_count=2, cases_attempted=attempted, cleanup=cleanup
    )
    status, result = await post(app, "/verify", row)
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == category
    assert result["cleanup_status"] == ("resolved" if cleanup else "not_started")
    assert not server._blocked
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", NONTERMINAL_CATEGORIES)
@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "passed"},
        {"outcome": "failed"},
        {"reward": 0.0},
        {"reward": 1.0},
        {"reward": False},
        {"failure_class": None},
        {"failure_class": failure_kinds.VERIFIER_ERROR},
        {"failure_class": "forged"},
        {"terminal": None},
        {"terminal": True},
        {"terminal": 0},
        {"terminal": "false"},
    ],
)
async def test_nonterminal_category_requires_exact_unscorable_contract(make_server, category, changes):
    server, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(category=category, **changes)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"
    assert result["cleanup_status"] == "resolved"
    assert not server._blocked
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", NONTERMINAL_CATEGORIES)
@pytest.mark.parametrize("contract", ["passed", "failed", "terminal"])
async def test_nonterminal_categories_cannot_enter_scored_or_terminal_allowlists(make_server, category, contract):
    _, app, fake, _ = make_server()
    if contract == "terminal":
        fake.grade.return_value = unscorable_verdict(
            category=category, failure_class=failure_kinds.VERIFIER_ERROR, terminal=True
        )
    elif contract == "failed":
        fake.grade.return_value = verdict(outcome="failed", reward=0.0, category=category, cases_equal=0)
    else:
        fake.grade.return_value = verdict(category=category)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"case_count": True},
        {"case_count": 1.0},
        {"case_count": 0},
        {"case_count": 2},
        {"case_count": None},
        {"cases_attempted": False},
        {"cases_attempted": 0.0},
        {"cases_attempted": "0"},
        {"cases_attempted": -1},
        {"cases_attempted": 2},
        {"cases_equal": False},
        {"cases_equal": 0.0},
        {"cases_equal": None},
        {"cases_equal": -1},
        {"cases_equal": 1},
    ],
)
async def test_nonterminal_result_requires_valid_counts(make_server, changes):
    _, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(**changes)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"
    assert result["cleanup_status"] == "resolved"
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category",
    [
        "passed",
        "candidate_mismatch",
        "candidate_invalid_output",
        "candidate_source_limit",
        "task_invalid",
        "reference_mismatch",
        REFERENCE_MARKER,
        None,
        [],
        {},
    ],
)
async def test_nonterminal_contract_does_not_expand_other_allowlists(make_server, category):
    _, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(category=category)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"
    assert REFERENCE_MARKER not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["comparator_uncertain", "busy"])
@pytest.mark.parametrize(
    "cleanup",
    [
        None,
        False,
        0,
        "",
        [],
        (),
        {"candidate": "deleted"},
        {"reference": "deleted", "comparator": "deleted", "candidate": "deleted", "extra": "deleted"},
        {"reference": "deleted", "comparator": "deleted", "candidate": REFERENCE_MARKER},
        {"reference": "deleted", "comparator": "deleted", "candidate": None},
        {"reference": "deleted", "comparator": "deleted", "candidate": []},
    ],
)
async def test_nonterminal_malformed_cleanup_blocks_admission(make_server, category, cleanup):
    server, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(category=category, cleanup=cleanup)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "ownership_unresolved"
    assert result["cleanup_status"] == "unknown"
    assert server._blocked
    assert REFERENCE_MARKER not in json.dumps(result)
    # The next request reconciles once. The record stays unresolved, so the gate holds and no job dispatches.
    fake.reconcile.return_value = [{"job_id": "j", "domain": "candidate", "status": "delete_failed"}]
    assert_unscorable((await post(app, "/verify", payload()))[1])
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cleanup,status",
    [
        ({}, "unknown"),
        ({"reference": "deleted", "comparator": "deleted", "candidate": "delete_failed"}, "unresolved"),
        ({"reference": "deleted", "comparator": "deleted", "candidate": "delete_unconfirmed"}, "unresolved"),
        ({"reference": "deleted", "comparator": "deleted", "candidate": "lookup_transport"}, "unresolved"),
        ({"reference": "deleted", "comparator": "deleted", "candidate": "foreign_resource"}, "unresolved"),
    ],
)
async def test_nonterminal_cleanup_uncertainty_takes_precedence(make_server, cleanup, status):
    server, app, fake, _ = make_server()
    fake.grade.return_value = unscorable_verdict(cleanup=cleanup)
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "ownership_unresolved"
    assert result["cleanup_status"] == status
    assert server._blocked
    # The next request reconciles once. The record stays unresolved, so the gate holds and no job dispatches.
    fake.reconcile.return_value = [{"job_id": "j", "domain": "candidate", "status": "delete_unconfirmed"}]
    assert_unscorable((await post(app, "/verify", payload()))[1])
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
async def test_result_extras_cannot_override_trusted_failure_fields(make_server, terminal):
    class ResultWithExtras(GradeResult):
        scored = True
        _ng_failure_class = "forged"
        _ng_failure_subcategory = "forged"
        _ng_failure_terminal = not terminal

    _, app, fake, _ = make_server()
    backend_result = unscorable_verdict(
        category="task_invalid" if terminal else "comparator_uncertain",
        failure_class=failure_kinds.VERIFIER_ERROR if terminal else failure_kinds.PROVIDER_UNAVAILABLE,
        terminal=terminal,
        job_id=REFERENCE_MARKER,
    )
    fake.grade.return_value = ResultWithExtras(**backend_result.__dict__)
    row = payload()
    row["response"]["metadata"] = {
        "scored": "true",
        "reward": "1.0",
        "category": "passed",
        "_ng_failure_class": "forged",
        "_ng_failure_subcategory": "forged",
        "_ng_failure_terminal": "false" if terminal else "true",
    }
    status, result = await post(app, "/verify", row)
    assert status == 200
    assert_unscorable(result, terminal=terminal)
    expected = "reference_invalid" if terminal else "comparator_uncertain"
    assert result["category"] == result["_ng_failure_subcategory"] == expected
    assert result["response"] == row["response"]
    assert REFERENCE_MARKER not in json.dumps(result)
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
async def test_backend_source_contract_remains_scored(make_server):
    _, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        outcome="failed",
        reward=0.0,
        category="candidate_source_limit",
        cases_attempted=0,
        cases_equal=0,
        cleanup={},
    )
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["scored"] is True and result["reward"] == 0.0
    assert result["category"] == "candidate_source_limit" and result["cleanup_status"] == "not_started"
    assert not any(key.startswith("_ng_failure_") for key in result)
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category", ["passed", "candidate_mismatch", "candidate_invalid_output", "candidate_source_limit"]
)
@pytest.mark.parametrize(
    "changes",
    [
        {"failure_class": failure_kinds.PROVIDER_UNAVAILABLE},
        {"terminal": False},
        {"case_count": True},
    ],
)
async def test_scored_value_and_source_contract_guards_remain_strict(make_server, category, changes):
    _, app, fake, _ = make_server()
    fields = {"category": category}
    if category != "passed":
        fields.update(outcome="failed", reward=0.0, cases_equal=0)
    if category == "candidate_source_limit":
        fields.update(cases_attempted=0, cleanup={})
    fields.update(changes)
    fake.grade.return_value = verdict(**fields)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["task_invalid", "reference_mismatch"])
@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "failed"},
        {"reward": 0.0},
        {"failure_class": failure_kinds.PROVIDER_UNAVAILABLE},
        {"terminal": False},
        {"terminal": None},
        {"terminal": 1},
        {"cases_attempted": 1},
        {"cases_equal": 1},
        {"case_count": True},
    ],
)
async def test_terminal_preflight_contract_remains_strict(make_server, category, changes):
    _, app, fake, _ = make_server()
    fields = {"category": category, "failure_class": failure_kinds.VERIFIER_ERROR, "terminal": True}
    fields.update(changes)
    fake.grade.return_value = unscorable_verdict(**fields)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category",
    [
        "reference_error",
        "reference_timeout",
        "comparator_uncertain",
        "candidate_no_result",
        "result_invalid",
        "provider_exec",
        "job_deadline",
        "synthetic_unrecognized_category",
    ],
)
async def test_unknown_execution_does_not_inherit_terminal_blame(make_server, category):
    _, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        outcome="unscorable",
        reward=None,
        category=category,
        failure_class=failure_kinds.VERIFIER_ERROR,
        terminal=True,
    )
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == "execution_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category,attempted",
    [
        ("candidate_exception", 2),
        ("candidate_encoding_error", 2),
        ("candidate_source_error", 0),
        ("candidate_timeout", 1),
        ("candidate_worker_abort", 0),
    ],
)
async def test_candidate_fault_after_preflight_is_scored_zero(make_server, category, attempted):
    # A candidate fault after a completed preflight is scored 0 with the category preserved. A whole-run
    # fault need not attempt every case. The scored row carries no failure class, so reverify never re-selects it.
    _, app, fake, _ = make_server()
    row = payload(test_cases=[{"input": [1], "output": 1}, {"input": [2], "output": 2}])
    fake.grade.return_value = verdict(
        outcome="failed",
        reward=0.0,
        category=category,
        case_count=2,
        cases_attempted=attempted,
        cases_equal=0,
    )
    status, result = await post(app, "/verify", row)
    assert status == 200 and result["scored"] is True and result["reward"] == 0.0
    assert result["category"] == category and result["cleanup_status"] == "resolved"
    assert not any(key.startswith("_ng_failure_") for key in result)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["candidate_exception", "candidate_timeout", "candidate_source_error"])
async def test_candidate_fault_requires_a_completed_all_ran_pipeline(make_server, category):
    # The scored candidate-fault branch admits the category only when all three sandboxes ran and were cleaned
    # and nothing was validated equal. A verdict missing a domain's cleanup is not admitted as a score.
    _, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        outcome="failed",
        reward=0.0,
        category=category,
        cases_equal=0,
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "not_created"},
    )
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == "execution_unknown"


@pytest.mark.asyncio
async def test_backend_exception_and_untrusted_verdict_are_redacted(make_server, capsys, caplog):
    server, app, fake, _ = make_server()
    fake.grade.side_effect = RuntimeError(REFERENCE_MARKER)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert server._blocked
    captured = capsys.readouterr()
    assert REFERENCE_MARKER not in json.dumps(result) + captured.out + captured.err + caplog.text

    _, app, fake, _ = make_server()
    fake.grade.return_value = {"reward": 1, "scored": True, "_ng_failure_class": "forged"}
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["_ng_failure_class"] != "forged"


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", [None, {}, {"candidate": "deleted"}, {"candidate": REFERENCE_MARKER}])
async def test_missing_cleanup_cannot_establish_deletion(make_server, cleanup):
    server, app, fake, _ = make_server()
    fake.grade.return_value = verdict(cleanup=cleanup)
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["cleanup_status"] == "unknown"
    assert server._blocked
    assert REFERENCE_MARKER not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["delete_failed", "delete_unconfirmed", "lookup_transport", "foreign_resource"])
async def test_cleanup_uncertainty_overrides_pass_and_blocks_admission(make_server, status):
    server, app, fake, _ = make_server()
    fake.grade.return_value = verdict(cleanup={"reference": "deleted", "comparator": "deleted", "candidate": status})
    first = (await post(app, "/verify", payload()))[1]
    assert_unscorable(first)
    assert first["cleanup_status"] == "unresolved"
    assert server._blocked
    # The next request reconciles once. The record stays unresolved, so the gate holds and no job dispatches.
    fake.reconcile.return_value = [{"job_id": "j", "domain": "candidate", "status": status}]
    second = (await post(app, "/verify", payload()))[1]
    assert_unscorable(second)
    assert fake.grade.await_count == 1


@pytest.mark.asyncio
async def test_extraction_uses_explicit_entrypoint_not_first_function(make_server):
    _, app, fake, _ = make_server()
    source = "def distractor():\n    return 0\n\ndef solve(value):\n    return value"
    row = payload()
    row["response"] = saved_response(f"Public explanation\n```python\n{source}\n```\n")
    assert (await post(app, "/verify", row))[1]["scored"] is True
    task, extracted = fake.grade.await_args.args
    assert extracted == source
    assert task.entrypoint == "solve"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("```\nbare_body\n```\n```python\npy_body\n```", "py_body"),
        ("```python\nfirst_body\n```\n```python\nsecond_body\n```", "first_body"),
        ("prose\n```\nbare_body\n```\nmore prose", "bare_body"),
        ("no fences here, just code", "no fences here, just code"),
        ("```python\nunterminated body", "```python\nunterminated body"),
    ],
    ids=["python_over_bare", "first_of_several", "bare_fallback", "raw_fallback", "unterminated_fence"],
)
def test_source_extraction_matches_the_official_harness_rule(text, expected):
    assert adapter._source(saved_response(text)) == expected


@pytest.mark.asyncio
async def test_first_fence_reaches_grading_when_several_are_present(make_server):
    """A response with several fences grades the first ```python fence."""
    _, app, fake, _ = make_server()
    source = "def solve(value):\n    return value"
    row = payload()
    row["response"] = saved_response(
        f"Reasoning...\n```python\n{source}\n```\nAlternative:\n```python\ndef solve(value):\n    return 0\n```"
    )
    assert (await post(app, "/verify", row))[1]["scored"] is True
    _, extracted = fake.grade.await_args.args
    assert extracted == source


@pytest.mark.asyncio
async def test_raw_text_fallback_still_honors_source_bounds(make_server):
    """The raw-text fallback stays inside the encoding and size bounds: an undecodable body is scored, not graded."""
    _, app, fake, factory = make_server()
    row = payload()
    row["response"] = saved_response("\x00")  # no fence -> raw text -> fails the source bound
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is True
    assert result["category"] == "candidate_source_limit"
    assert "_ng_failure_class" not in result
    fake.grade.assert_not_awaited()
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["incomplete", "incomplete_details", "error"])
async def test_uncompleted_response_needs_regeneration(make_server, mode):
    # An uncompleted, incomplete, or error response has no usable source. It carries needs_regeneration, not
    # verifier_unavailable, because reverify would reach the same verdict. The class is not terminal.
    _, app, fake, _ = make_server()
    row = payload()
    if mode == "incomplete":
        row["response"]["status"] = "incomplete"
    elif mode == "incomplete_details":
        row["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        row["response"]["error"] = {"code": "server_error", "message": "synthetic"}
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is False
    assert result["category"] == "response_incomplete"
    assert result["_ng_failure_class"] == "critpt_custom_grader:response_incomplete"
    assert "_ng_failure_terminal" not in result
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_final_message_in_a_completed_envelope_needs_regeneration(make_server):
    # A completed envelope whose last assistant message did not itself complete carries truncated code, not a
    # finished empty answer. It needs regeneration, not a scored zero as a candidate source fault.
    _, app, fake, _ = make_server()
    row = payload()
    row["response"]["output"][-1]["status"] = "incomplete"
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is False
    assert result["category"] == "response_incomplete"
    assert result["_ng_failure_class"] == "critpt_custom_grader:response_incomplete"
    assert "_ng_failure_terminal" not in result
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty_text", "no_message"])
async def test_completed_response_with_empty_source_is_a_candidate_fault(make_server, mode):
    # A completed response with no usable source is an empty submission. Like the historical grader it is scored
    # 0 as a candidate source fault, before any sandbox is created and without grading.
    _, app, fake, _ = make_server()
    row = payload()
    if mode == "empty_text":
        row["response"] = saved_response("")
    else:
        row["response"]["output"] = []
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is True
    assert result["category"] == "candidate_source_limit"
    assert "_ng_failure_class" not in result
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_bound_and_queued_cancellation(make_server):
    server, app, fake, _ = make_server(max_queued_jobs=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def grade(*args):
        entered.set()
        await release.wait()
        return verdict()

    fake.grade.side_effect = grade
    first = asyncio.create_task(post(app, "/verify", payload()))
    await asyncio.wait_for(entered.wait(), 1)
    queued = asyncio.create_task(post(app, "/verify", payload()))
    await until(lambda: server._pending == 2)
    overflow = (await post(app, "/verify", payload()))[1]
    assert_unscorable(overflow)
    assert overflow["category"] == "busy"
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await until(lambda: server._pending == 1)
    assert server._lock.locked() and not server._blocked
    release.set()
    assert (await first)[1]["scored"] is True
    await until(lambda: not server._jobs)
    assert fake.grade.await_count == 1


@pytest.mark.asyncio
async def test_second_verify_waits_then_completes_when_first_finishes(make_server):
    """A second concurrent verify waits in the queue and completes once the first job finishes
    within the queue window, rather than a busy refusal."""
    server, app, fake, _ = make_server(max_queued_jobs=1)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def grade(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        return verdict()

    fake.grade.side_effect = grade
    first = asyncio.create_task(post(app, "/verify", payload()))
    await asyncio.wait_for(entered.wait(), 1)
    second = asyncio.create_task(post(app, "/verify", payload()))
    await until(lambda: server._pending == 2)
    # The second request is queued on the lock, not refused, and has not completed yet.
    assert not second.done()
    release.set()
    assert (await first)[1]["scored"] is True
    assert (await second)[1]["scored"] is True
    await until(lambda: not server._jobs)
    assert fake.grade.await_count == 2


@pytest.mark.asyncio
async def test_queue_deadline_does_not_cancel_active_job(make_server):
    server, app, fake, _ = make_server(max_queued_jobs=1, queue_timeout_s=0.01)
    entered, release = asyncio.Event(), asyncio.Event()

    async def grade(*args):
        entered.set()
        await release.wait()
        return verdict()

    fake.grade.side_effect = grade
    first = asyncio.create_task(post(app, "/verify", payload()))
    await asyncio.wait_for(entered.wait(), 1)
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["category"] == "queue_timeout"
    assert server._lock.locked()
    release.set()
    assert (await first)[1]["scored"] is True
    assert fake.grade.await_count == 1


@pytest.mark.asyncio
async def test_reconcile_honors_a_deadline_instead_of_waiting_the_full_queue_timeout(make_server):
    # On the latched-gate path one queue budget is shared across reconcile and execute. reconcile honors that
    # deadline: with the budget already spent it returns busy at once, never waiting the full queue timeout again.
    server, _app, fake, _ = make_server(queue_timeout_s=30.0)
    server._grader = fake
    loop = asyncio.get_running_loop()
    await server._lock.acquire()  # a running job holds the single job slot
    try:
        start = loop.time()
        report = await asyncio.wait_for(server.reconcile(deadline=loop.time()), timeout=5)
        assert report == [{"status": "busy"}]
        assert loop.time() - start < 5  # bounded by the spent deadline, not the 30 s queue timeout
    finally:
        server._lock.release()


@pytest.mark.asyncio
async def test_blocked_verify_shares_one_queue_wait_across_reconcile_and_execute(make_server, monkeypatch):
    # After the cleanup latch, verify reconciles then dispatches _execute under ONE shared queue-wait deadline, so
    # a request never waits the full queue timeout twice. Both waits receive the same deadline.
    server, app, fake, _ = make_server(queue_timeout_s=30.0)
    server._grader = fake
    server._blocked = True
    fake.reconcile = AsyncMock(return_value=[])  # a resolved reconcile clears the latched gate
    seen: dict[str, float | None] = {}
    server_type = type(server)
    real_reconcile, real_execute = server_type.reconcile, server_type._execute

    async def spy_reconcile(self, deadline=None):
        seen["reconcile"] = deadline
        return await real_reconcile(self, deadline)

    async def spy_execute(self, body, task, source, deadline=None):
        seen["execute"] = deadline
        return await real_execute(self, body, task, source, deadline)

    monkeypatch.setattr(server_type, "reconcile", spy_reconcile)
    monkeypatch.setattr(server_type, "_execute", spy_execute)
    result = (await post(app, "/verify", payload()))[1]
    assert result["scored"] is True
    assert seen["reconcile"] is not None
    assert seen["reconcile"] == seen["execute"]


def _cancellable_grade():
    """A grade stand-in that blocks until cancelled, then holds inside its cleanup until released."""
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def grade(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    return grade, entered, cleaning, release


async def _cancel_one_job(server, app, fake):
    """Start one /verify, let its grade begin, cancel the caller, then let the grader's cleanup finish. Returns once
    the cancelled job has fully settled. `fake.reconcile` decides whether that cleanup left ownership resolved."""
    grade, entered, cleaning, release = _cancellable_grade()
    fake.grade.side_effect = grade
    request = asyncio.create_task(post(app, "/verify", payload()))
    await asyncio.wait_for(entered.wait(), 1)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.wait_for(cleaning.wait(), 1)
    # While the cancelled job still holds the gate, a second request is refused rather than racing the cleanup.
    assert server._pending == 1 and server._lock.locked() and server._jobs
    assert_unscorable((await post(app, "/verify", payload()))[1])
    release.set()
    await until(lambda: not server._jobs)
    assert not server._lock.locked()


@pytest.mark.asyncio
async def test_disconnect_with_resolved_cleanup_keeps_accepting_jobs(make_server):
    """A caller disconnect whose cancelled job resolved its own cleanup must not latch the gate."""
    server, app, fake, _ = make_server()
    fake.reconcile.return_value = []  # every owned domain resolved during the cancelled job's own cleanup

    await _cancel_one_job(server, app, fake)

    fake.reconcile.assert_awaited_once()
    assert not server._blocked
    # The gate stayed open: the next job runs and scores.
    fake.grade.side_effect = None
    fake.grade.return_value = verdict()
    assert (await post(app, "/verify", payload()))[1]["scored"] is True


@pytest.mark.asyncio
async def test_disconnect_with_unresolved_cleanup_blocks(make_server):
    """A caller disconnect whose cancelled job could not resolve every owned resource latches the gate."""
    server, app, fake, _ = make_server()
    fake.reconcile.return_value = [{"job_id": "j", "domain": "candidate", "status": "delete_unconfirmed"}]

    await _cancel_one_job(server, app, fake)

    assert server._blocked
    # Blocked: a later request is refused as ownership_unresolved without dispatching another grading job.
    fake.grade.reset_mock()
    fake.grade.side_effect = None
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["category"] == "ownership_unresolved"
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_later_reconcile_that_resolves_everything_unblocks(make_server):
    """A gate latched by an unresolved disconnect is cleared by a later reconcile that resolves every resource."""
    server, app, fake, _ = make_server()
    fake.reconcile.side_effect = [
        [{"job_id": "j", "domain": "candidate", "status": "delete_unconfirmed"}],  # cancellation path: unresolved
        [{"job_id": "j", "domain": "candidate", "status": "deleted"}],  # later reconcile: resolved
    ]

    await _cancel_one_job(server, app, fake)
    assert server._blocked

    report = await server.reconcile()
    assert report == [{"job_id": "j", "domain": "candidate", "status": "deleted"}]
    assert not server._blocked
    assert fake.reconcile.await_count == 2

    # Unblocked: the next job runs again.
    fake.grade.side_effect = None
    fake.grade.return_value = verdict()
    assert (await post(app, "/verify", payload()))[1]["scored"] is True


@pytest.mark.asyncio
async def test_reconcile_clears_a_latch_set_by_an_ownership_unresolved_verdict(make_server):
    """A gate latched by a completed job whose cleanup was unresolved clears once a later reconcile resolves it."""
    server, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "delete_unconfirmed"}
    )
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["category"] == "ownership_unresolved"
    assert server._blocked
    fake.reconcile.assert_not_awaited()  # the latch came from the verdict, not from a reconcile

    fake.reconcile.return_value = []  # the journal is clean now
    assert await server.reconcile() == []
    assert not server._blocked

    fake.grade.return_value = verdict()
    assert (await post(app, "/verify", payload()))[1]["scored"] is True


@pytest.mark.asyncio
async def test_reconcile_clears_a_latch_set_by_a_failed_cancellation_reconcile(make_server):
    """A gate latched because the cancellation-path reconcile itself failed clears once a later reconcile succeeds."""
    server, app, fake, _ = make_server()
    fake.reconcile.side_effect = [RuntimeError("reconcile unavailable"), []]

    await _cancel_one_job(server, app, fake)
    assert server._blocked

    assert await server.reconcile() == []
    assert not server._blocked
    assert fake.reconcile.await_count == 2

    fake.grade.side_effect = None
    fake.grade.return_value = verdict()
    assert (await post(app, "/verify", payload()))[1]["scored"] is True


@pytest.mark.asyncio
async def test_blocked_server_admits_when_a_reconcile_resolves(make_server):
    """A gate latched by a transient fault clears on the next request when a reconcile resolves every record."""
    server, app, fake, _ = make_server()
    # The first verdict leaves one domain unresolved and latches the gate.
    fake.grade.return_value = verdict(
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "delete_unconfirmed"}
    )
    first = (await post(app, "/verify", payload()))[1]
    assert_unscorable(first)
    assert first["category"] == "ownership_unresolved"
    assert server._blocked
    fake.reconcile.assert_not_awaited()  # the latch came from the verdict, not from a reconcile
    # The transient fault cleared. The next request reconciles clean and then grades a passing verdict.
    fake.reconcile.return_value = []
    fake.grade.return_value = verdict()
    second = (await post(app, "/verify", payload()))[1]
    assert second["scored"] is True and second["reward"] == 1.0
    assert not server._blocked
    fake.reconcile.assert_awaited_once()
    assert fake.grade.await_count == 2


@pytest.mark.asyncio
async def test_blocked_server_stays_blocked_when_a_reconcile_cannot_resolve(make_server):
    """A gate latched by a fault stays closed when the next request's reconcile still finds an owned resource
    unresolved. The request is refused and no grading job runs."""
    server, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "delete_unconfirmed"}
    )
    assert_unscorable((await post(app, "/verify", payload()))[1])
    assert server._blocked
    fake.reconcile.return_value = [{"job_id": "j", "domain": "candidate", "status": "delete_failed"}]
    fake.grade.reset_mock()
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["category"] == "ownership_unresolved"
    assert server._blocked
    fake.reconcile.assert_awaited_once()
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocked_server_stays_blocked_when_a_reconcile_raises(make_server):
    """A reconcile that raises does not clear the gate. The request is refused and no grading job runs."""
    server, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "delete_unconfirmed"}
    )
    assert_unscorable((await post(app, "/verify", payload()))[1])
    assert server._blocked
    fake.reconcile.side_effect = RuntimeError(REFERENCE_MARKER)
    fake.grade.reset_mock()
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["category"] == "ownership_unresolved"
    assert server._blocked
    fake.reconcile.assert_awaited_once()
    fake.grade.assert_not_awaited()
    assert REFERENCE_MARKER not in json.dumps(result)


@pytest.mark.asyncio
async def test_resolved_shutdown_report_clears_a_latch_and_lifespan_does_not_raise(make_server):
    """A latch set during serving must not fail a clean shutdown. When the shutdown report resolves every
    resource, the lifespan clears the gate and does not raise."""
    server, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        cleanup={"reference": "deleted", "comparator": "deleted", "candidate": "delete_unconfirmed"}
    )
    async with app.router.lifespan_context(app):
        assert_unscorable((await post(app, "/verify", payload()))[1])
        assert server._blocked
        fake.shutdown.return_value = [{"job_id": "j", "domain": "candidate", "status": "deleted"}]
    assert app.state.cleanup_report == [{"status": "deleted"}]
    assert not server._blocked


@pytest.mark.asyncio
async def test_unresolved_shutdown_report_still_raises(make_server):
    """A shutdown report with an unresolved resource keeps the gate closed, so the lifespan raises."""
    server, app, fake, _ = make_server()
    fake.shutdown.return_value = [{"job_id": "j", "domain": "candidate", "status": "delete_unconfirmed"}]
    with pytest.raises(RuntimeError, match="^grader cleanup unresolved$"):
        async with app.router.lifespan_context(app):
            await post(app, "/verify", payload())
    assert app.state.cleanup_report == [{"status": "delete_unconfirmed"}]
    assert server._blocked


@pytest.mark.asyncio
async def test_shutdown_closes_only_owned_grader_and_reports_uncertainty(make_server):
    server, app, fake, _ = make_server()
    await post(app, "/verify", payload())
    fake.shutdown.return_value = [{"job_id": REFERENCE_MARKER, "domain": "candidate", "status": "delete_unconfirmed"}]
    await server.shutdown()
    fake.shutdown.assert_awaited_once()
    assert server._shutdown_report == [{"status": "delete_unconfirmed"}]
    assert server._blocked and not server.server_client.mock_calls
    assert_unscorable((await post(app, "/verify", payload()))[1])

    external, _, external_fake, _ = make_server()
    external._grader = external_fake
    await external.shutdown()
    external_fake.shutdown.assert_not_awaited()


class _FakeReconcileBackend:
    """Minimal backend for the reconcile path only: exact-key lookup and delete against a shared present-set.

    The present-set maps an exact id or stable name to a live ``SandboxLookup``. A create, exec, write or download
    is a test error, because reconcile must never run task code."""

    def __init__(self, present):
        self._present = present

    async def lookup(self, key):
        return self._present.get(key)

    async def close(self, handle):
        for key in [k for k, v in self._present.items() if v.handle.sandbox_id == handle.sandbox_id]:
            del self._present[key]

    async def aclose(self):
        pass

    async def create(self, spec):
        raise AssertionError("reconcile must not create")

    async def exec(self, *args, **kwargs):
        raise AssertionError("reconcile must not exec")

    async def write_file(self, *args, **kwargs):
        raise AssertionError("reconcile must not write")

    async def download_bounded(self, *args, **kwargs):
        raise AssertionError("reconcile must not download")


def _seed_unresolved_record(tmp_path, sandbox_id):
    """Write one journal record whose candidate domain is ``delete_failed`` with a learned id. Returns the journal
    and the exact stable name and labels of that domain, so a test can register a matching live sandbox."""
    journal = OwnershipJournal(str(tmp_path / "journal"))
    names = {domain: f"ngcg-crashjob-{domain}" for domain in DOMAINS}
    labels = {domain: {LABEL_OWNER: "synthetic", LABEL_JOB: "crashjob", LABEL_DOMAIN: domain} for domain in DOMAINS}
    record = journal.open_job("crashjob", "synthetic", names, labels)
    journal.update(record, "candidate", "delete_failed", sandbox_id)
    return journal, names["candidate"], labels["candidate"]


def _inject_real_grader(server, policy, journal, present):
    grader = Grader(policy, backend_factory=lambda p: _FakeReconcileBackend(present), journal=journal)
    server._grader = grader
    server._owns_grader = True
    return grader


@pytest.mark.asyncio
async def test_startup_reconcile_clears_an_absent_crash_record_and_serves(make_server, tmp_path):
    # A crash left an unresolved record whose sandbox is already gone. Startup reconciles the journal before
    # serving, so the record clears and the first verify is served rather than wedging on ownership_unresolved.
    server, app, _, _ = make_server()
    journal, _, _ = _seed_unresolved_record(tmp_path, "sb-gone")
    assert journal.unresolved()  # precondition: the record wedges the first request without a startup reconcile
    grader = _inject_real_grader(server, server.config.execution, journal, present={})
    grader.grade = AsyncMock(return_value=verdict())

    async with app.router.lifespan_context(app):
        assert journal.unresolved() == []  # startup reconcile resolved the absent record
        assert server._blocked is False
        status, body = await post(app, "/verify", payload())
        assert status == 200 and body["scored"] is True and body["category"] == "passed"
        grader.grade.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_reconcile_deletes_a_still_present_sandbox_and_serves(make_server, tmp_path):
    # A crash left an unresolved record whose sandbox still exists. Startup reconcile finds it by its exact stable
    # identity, deletes it, and then serves the first request.
    server, app, _, _ = make_server()
    journal, name, labels = _seed_unresolved_record(tmp_path, "sb-live")
    handle = SandboxHandle(sandbox_id="sb-live", provider_name="daytona", raw=None)
    present = {"sb-live": SandboxLookup(handle=handle, name=name, labels=labels, state="started")}
    grader = _inject_real_grader(server, server.config.execution, journal, present)
    grader.grade = AsyncMock(return_value=verdict())

    async with app.router.lifespan_context(app):
        assert present == {}  # the live sandbox was deleted during startup reconcile
        assert journal.unresolved() == [] and server._blocked is False
        status, body = await post(app, "/verify", payload())
        assert status == 200 and body["scored"] is True
        grader.grade.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_close_failure_is_visible_without_provider_text(make_server):
    server, app, fake, _ = make_server()
    fake.shutdown.side_effect = RuntimeError(REFERENCE_MARKER)
    with pytest.raises(RuntimeError, match="^grader cleanup unresolved$"):
        async with app.router.lifespan_context(app):
            await post(app, "/verify", payload())
    assert app.state.cleanup_report == [{"status": "unknown"}]
    assert server._blocked


@pytest.mark.asyncio
async def test_mixed_aggregation_is_scoped_to_supplied_rows(make_server):
    _, app, _, _ = make_server()
    rows = [
        {"_ng_task_index": 0, "reward": 1.0, "scored": True, "private_numeric": 9001},
        {"_ng_task_index": 1, "reward": 0.0, "scored": True},
        {"_ng_task_index": 2, "reward": 0.0, "scored": False, "_ng_failure_class": "verifier_unavailable"},
        {"_ng_task_index": 3, "reward": 0.0, "scored": False, "_ng_failure_class": "reference_failed"},
    ]
    status, result = await post(app, "/aggregate_metrics", {"verify_responses": rows})
    assert status == 200
    metrics = result["agent_metrics"]
    assert metrics["supplied_attempted"] == 4
    assert metrics["supplied_scored"] == 2
    assert metrics["supplied_failed"] == 1
    assert metrics["supplied_unscorable"] == 2
    assert metrics["mean/reward"] == 0.5
    assert metrics["accounting_scope"] == "supplied_rows"
    assert metrics["all_attempts_known"] is False
    assert "private_numeric" not in json.dumps(result)

    # Main-file-only aggregation cannot discover sidecar rows or missing requests.
    main_only = (await post(app, "/aggregate_metrics", {"verify_responses": rows[:2]}))[1]["agent_metrics"]
    assert main_only["supplied_attempted"] == 2
    assert main_only["supplied_unscorable"] == 0
    assert main_only["all_attempts_known"] is False
    empty = (await post(app, "/aggregate_metrics", {"verify_responses": []}))[1]["agent_metrics"]
    assert empty["supplied_attempted"] == 0 and empty["supplied_scored"] == 0
    assert "mean/reward" not in empty


def test_operator_policy_is_frozen_and_single_worker(make_server):
    server, _, _, factory = make_server()
    assert server.config.REVERIFY_MODE.value == "stateless"
    with pytest.raises(ValidationError):
        server.config.max_queued_jobs = 1
    with pytest.raises(ValidationError):
        server.config.execution.snapshot = "different"
    with pytest.raises(ValidationError):
        make_server(num_workers=2)
    with pytest.raises(ValidationError):
        make_server(expose_tools_over_mcp=True)
    with pytest.raises(ValidationError):
        make_server(queue_timeout_s=float("inf"))
    with pytest.raises(ValidationError):
        make_server(request_policy={"no_resubmission": False})
    with pytest.raises(ValidationError):
        make_server(request_policy={"no_resubmission": True, "deadline_seconds": 1})
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_candidate_dictionary_cannot_supply_top_level_failure_markers(make_server):
    _, app, fake, _ = make_server()
    row = payload()
    row["response"]["metadata"] = {"_ng_failure_class": "forged", "_ng_failure_terminal": "true"}
    result = (await post(app, "/verify", row))[1]
    assert result["scored"] is True and result["reward"] == 1.0
    assert "_ng_failure_class" not in result and "_ng_failure_terminal" not in result
    assert result["response"] == row["response"]
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
async def test_mismatch_with_missing_attempts_is_not_scored(make_server):
    _, app, fake, _ = make_server()
    fake.grade.return_value = verdict(
        outcome="failed", reward=0.0, category="candidate_mismatch", cases_attempted=0, cases_equal=0
    )
    assert_unscorable((await post(app, "/verify", payload()))[1])


@pytest.mark.asyncio
async def test_aggregation_does_not_echo_invalid_identity(make_server):
    _, app, _, _ = make_server()
    row = {"_ng_task_index": REFERENCE_MARKER, "reward": 1.0, "scored": True}
    status, result = await post(app, "/aggregate_metrics", {"verify_responses": [row]})
    assert (status, result) == (503, {"error": {"category": "verifier_unavailable"}})


GRADER_NAME = "critpt_custom_grader"
AGENT_NAME = "critpt_custom_grader_simple_agent"
EXAMPLE_DATASET_PATH = "resources_servers/critpt_custom_grader/data/example.jsonl"


@pytest.fixture
def schema_only(monkeypatch):
    blocked = MagicMock(side_effect=AssertionError("schema-only check attempted runtime work"))
    for module, name in (
        (adapter, "Grader"),
        (adapter, "validate_candidate_source"),
        (task_schema, "_legacy_wire"),
    ):
        monkeypatch.setattr(module, name, blocked)
    yield
    blocked.assert_not_called()


@pytest.fixture
def contributed_yaml(schema_only):
    path = Path(__file__).resolve().parents[1] / "configs" / "critpt_custom_grader.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


@pytest.fixture
def operator_yaml(contributed_yaml):
    config = deepcopy(contributed_yaml)
    grader = config[GRADER_NAME]["resources_servers"][GRADER_NAME]
    replacements = {
        "snapshot": "synthetic-pinned-snapshot",
        "os_user": "synthetic",
        "owner_id": "synthetic-owner",
        "journal_dir": "/synthetic/unused-journal",
    }
    for key, value in replacements.items():
        assert grader["execution"][key] == "???"
        grader["execution"][key] = value
    agent = config[AGENT_NAME]["responses_api_agents"]["simple_agent"]
    # Gym normally assigns these mandatory transport fields. No sockets or journal are opened.
    for port, block in enumerate((grader, agent), start=8000):
        assert "host" not in block and "port" not in block
        block.update(host="127.0.0.1", port=port)
    return config


@pytest.fixture
def contributed_rows(contributed_yaml):
    declarations = contributed_yaml[AGENT_NAME]["responses_api_agents"]["simple_agent"]["datasets"]
    assert len(declarations) == 1
    declaration = declarations[0]
    assert set(declaration) == {"name", "type", "jsonl_fpath", "num_repeats", "license"}
    dataset = DatasetConfig.model_validate(declaration)
    assert dataset.name == dataset.type == "example"
    assert dataset.jsonl_fpath == EXAMPLE_DATASET_PATH
    assert dataset.license == "Apache 2.0" and dataset.num_repeats == 1
    assert dataset.source is None
    path = Path(__file__).resolve().parents[3] / dataset.jsonl_fpath
    with path.open("rb") as stream:
        raw = stream.read(5 * task_schema.MAX_TASK_BYTES + 1)
    assert 0 < len(raw) <= 5 * task_schema.MAX_TASK_BYTES
    lines = raw.splitlines()
    assert len(lines) == 5
    rows = []
    for line in lines:
        assert 0 < len(line) <= task_schema.MAX_TASK_BYTES
        row = json.loads(line)
        assert set(row) == {"responses_create_params", "task_data"}
        # Check canonical cases before TaskData: these fixtures must not need delivered-value normalization.
        for case in row["task_data"]["test_cases"]:
            assert {"args", "expected"} <= set(case) <= {"args", "kwargs", "expected", "tolerances"}
        rows.append(row)
    return rows


def test_contributed_yaml_resource_metadata_is_discoverable(contributed_yaml):
    # Metadata discovery descends into the first dictionary-valued root entry.
    assert next(name for name, value in contributed_yaml.items() if isinstance(value, dict)) == GRADER_NAME
    metadata = visit_resources_server(contributed_yaml)
    assert metadata.domain == "coding"
    assert metadata.description == (
        "Stored-expectation Python grading with exact, tolerance, composite, and symbolic values."
    )
    assert metadata.verified is False


def test_contributed_yaml_keeps_mandatory_operator_values(contributed_yaml):
    prefix = f"{GRADER_NAME}.resources_servers.{GRADER_NAME}.execution"
    assert OmegaConf.missing_keys(OmegaConf.create(contributed_yaml)) == {
        f"{prefix}.{key}" for key in ("snapshot", "os_user", "owner_id", "journal_dir")
    }
    assert "policy_model" not in contributed_yaml


def test_contributed_yaml_loads_actual_server_schemas(operator_yaml, monkeypatch):
    ResourcesServerTypeConfig.model_validate(operator_yaml[GRADER_NAME])
    ResponsesAPIAgentServerTypeConfig.model_validate(operator_yaml[AGENT_NAME])
    reader = MagicMock(return_value=OmegaConf.create(operator_yaml))
    monkeypatch.setattr(server_utils, "get_global_config_dict", reader)
    loaded = {}
    for name, server_class, config_class in (
        (GRADER_NAME, adapter.CritPtCustomGraderServer, adapter.CritPtCustomGraderConfig),
        (AGENT_NAME, SimpleAgent, SimpleAgentConfig),
    ):
        selector = MagicMock(return_value=name)
        monkeypatch.setattr(server_utils, "getenv", selector)
        loaded[name] = server_class.load_config_from_global_config()
        selector.assert_called_once_with(server_utils.NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME)
        assert isinstance(loaded[name], config_class)
        assert loaded[name].name == name and loaded[name].entrypoint == "app.py"
        assert loaded[name].num_workers == 1
    assert reader.call_count == 2
    grader = loaded[GRADER_NAME]
    block = operator_yaml[GRADER_NAME]["resources_servers"][GRADER_NAME]
    assert grader.description == block["description"] and grader.description
    assert grader.domain.value == "coding" and grader.verified is False
    assert grader.request_privacy == ServerRequestPrivacy(private_requests=True)
    assert grader.model_config["extra"] == "forbid" and grader.model_config["frozen"] is True
    assert grader.expose_tools_over_mcp is False and grader.max_queued_jobs == 4
    agent = loaded[AGENT_NAME]
    assert agent.resources_server.model_dump() == {"type": "resources_servers", "name": GRADER_NAME}
    assert agent.model_server.model_dump() == {"type": "responses_api_models", "name": "policy_model"}
    assert agent.max_steps == 1 and agent.skip_verification is False and agent.token_id_capture is False


def test_contributed_yaml_private_policy_and_deadlines(operator_yaml):
    grader = operator_yaml[GRADER_NAME]["resources_servers"][GRADER_NAME]
    agent = operator_yaml[AGENT_NAME]["responses_api_agents"]["simple_agent"]
    # The grader owns the request-policy and request-privacy config, backed by its own local models.
    # The agent block is the stock simple_agent and carries no private-mode keys.
    assert ServerRequestPrivacy.model_validate(grader["request_privacy"]).private_requests is True
    assert ServerRequestPolicy.model_validate(grader["request_policy"]).no_resubmission is True
    assert "request_privacy" not in agent and "request_policy" not in agent
    execution = ExecutionPolicy.model_validate(grader["execution"])
    verify_policy = ServerRequestPolicy.model_validate(grader["request_policy"])
    assert execution.minimum_job_timeout_s() == 5705.0
    assert execution.effective_job_timeout_s() == 5900.0
    minimum = grader["queue_timeout_s"] + execution.effective_job_timeout_s() + 6 * execution.cleanup_timeout_s
    # The verify deadline strictly exceeds queue + one full job + six cleanup bounds.
    assert minimum == 12280.0 < verify_policy.deadline_seconds == 12310.0
    for flag in (
        "global_aiohttp_client_request_debug",
        "observability_enabled",
        "profiling_enabled",
        "uvicorn_logging_show_200_ok",
        "upload_rollouts",
    ):
        assert operator_yaml[flag] is False
    assert operator_yaml["disable_aggregation"] is True
    assert operator_yaml["num_samples_in_parallel"] == operator_yaml["num_repeats"] == 1
    assert operator_yaml["token_id_capture"] == {"enabled": False, "all_agents": False}
    assert operator_yaml["telemetry"] == {
        "enabled": False,
        "traces_enabled": False,
        "metrics_enabled": False,
        "logs_enabled": False,
        "instrument_aiohttp": False,
        "span_groups": "job",
    }
    assert not server_utils.GlobalAIOHTTPAsyncClientConfig.model_validate(
        operator_yaml
    ).global_aiohttp_client_request_debug
    assert not server_utils.UvicornLoggingConfig.model_validate(operator_yaml).uvicorn_logging_show_200_ok
    # The grader's requirement must not change the shared schemas' intentional public defaults.
    assert ServerRequestPrivacy().private_requests is False
    assert ServerRequestPolicy().no_resubmission is False


def test_contributed_yaml_requires_grader_privacy(operator_yaml):
    fields = deepcopy(operator_yaml[GRADER_NAME]["resources_servers"][GRADER_NAME]) | {"name": GRADER_NAME}
    del fields["request_privacy"]
    with pytest.raises(ValidationError) as caught:
        adapter.CritPtCustomGraderConfig.model_validate(fields)
    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [(("request_privacy",), "missing")]


@pytest.mark.parametrize(
    "changes,location,error_type",
    [
        ({"description": {}}, ("description",), "string_type"),
        ({"request_privacy": None}, ("request_privacy",), "model_type"),
        ({"request_privacy": {}}, ("request_privacy",), "value_error"),
        ({"request_privacy": {"private_requests": False}}, ("request_privacy",), "value_error"),
        (
            {"request_privacy": {"private_requests": True, "task_override": True}},
            ("request_privacy", "task_override"),
            "extra_forbidden",
        ),
        ({"task_override": True}, ("task_override",), "extra_forbidden"),
        ({"request_policy": {"no_resubmission": False, "deadline_seconds": 1020.0}}, (), "value_error"),
        ({"request_policy": {"no_resubmission": True, "deadline_seconds": 6480.0}}, (), "value_error"),
    ],
)
def test_contributed_yaml_rejects_incompatible_policy(operator_yaml, changes, location, error_type):
    fields = operator_yaml[GRADER_NAME]["resources_servers"][GRADER_NAME] | {"name": GRADER_NAME} | changes
    with pytest.raises(ValidationError) as caught:
        adapter.CritPtCustomGraderConfig.model_validate(fields)
    assert (location, error_type) in [(error["loc"], error["type"]) for error in caught.value.errors()]


def test_contributed_execution_deadline_bounds(operator_yaml):
    fields = deepcopy(operator_yaml[GRADER_NAME]["resources_servers"][GRADER_NAME]) | {"name": GRADER_NAME}
    fields["execution"]["job_timeout_s"] = 5704.0
    with pytest.raises(ValidationError, match="job_timeout_s must be at least 5705"):
        adapter.CritPtCustomGraderConfig.model_validate(fields)
    fields["execution"]["job_timeout_s"] = 5705.0
    assert adapter.CritPtCustomGraderConfig.model_validate(fields).execution.minimum_job_timeout_s() == 5705.0
    fields["execution"]["job_timeout_s"] = 6000.0
    with pytest.raises(ValidationError, match="deadline covering queue, job and cleanup"):
        adapter.CritPtCustomGraderConfig.model_validate(fields)


def assert_fixture_wire_structure(wire):
    """Check this fixture's bounded JSON containers, never interpret numeric or symbolic leaf text."""
    assert type(wire) is dict and set(wire) == {"format", "value"}
    assert wire["format"] == task_schema.WIRE_FORMAT
    task_schema.bounded_json(wire)
    pending = [(wire["value"], 0)]
    count = 0
    tags = set()
    while pending:
        node, depth = pending.pop()
        count += 1
        assert count <= task_schema.MAX_ELEMENTS and depth <= task_schema.MAX_DEPTH
        assert type(node) is list and node and type(node[0]) is str
        tag = node[0]
        tags.add(tag)
        if tag in {"int", "fraction", "decimal", "str", "symbolic"}:
            assert len(node) == (3 if tag == "fraction" else 2)
            for text in node[1:]:
                assert type(text) is str and len(text.encode("utf-8")) <= task_schema.MAX_TEXT_BYTES
        elif tag == "bool":
            assert len(node) == 2 and type(node[1]) is bool
        elif tag == "tuple":
            assert len(node) == 2 and type(node[1]) is list
            pending.extend((child, depth + 1) for child in node[1])
        elif tag == "map":
            assert len(node) == 2 and type(node[1]) is list
            keys = set()
            for pair in node[1]:
                assert type(pair) is list and len(pair) == 2 and type(pair[0]) is str
                assert len(pair[0].encode("utf-8")) <= task_schema.MAX_TEXT_BYTES and pair[0] not in keys
                keys.add(pair[0])
                pending.append((pair[1], depth + 1))
        else:
            pytest.fail("unexpected fixture tag")
    return tags


def test_contributed_dataset_schema_only(contributed_rows):
    expected_entries = {
        "original-integer-step-v1": "add_one",
        "original-rational-half-v1": "halve_fraction",
        "original-reciprocal-tolerance-v1": "reciprocal",
        "original-pair-summary-v1": "summarize_pair",
        "original-symbolic-offset-square-v1": "offset_square",
    }
    ids = []
    tags = set()
    for row in contributed_rows:
        original = deepcopy(row)
        fields, conflicts = normalize_task_fields(row)
        assert not conflicts and fields == row["task_data"]
        assert {"schema_version", "entrypoint", "reference_entrypoint"} <= fields.keys()
        task = TaskData.model_validate(fields)
        assert task.schema_version == 1
        assert task.entrypoint == task.reference_entrypoint == expected_entries[task.problem_id]
        assert task.reference_source == fields["reference_source"]
        assert task.model_dump(mode="json", exclude_unset=True) == fields
        request = adapter.CritPtRunRequest.model_validate(row)
        assert request.responses_create_params["input"] == [{"role": "user", "content": task.problem}]
        assert adapter._task(request) == task
        ids.append(task.problem_id)
        for case in fields["test_cases"]:
            assert case["expected"]["kind"] == "value"
            wires = [*case["args"], *case.get("kwargs", {}).values(), case["expected"]["value"]]
            for wire in wires:
                tags.update(assert_fixture_wire_structure(wire))
        assert row == original
    assert len(ids) == len(set(ids)) == 5 and set(ids) == expected_entries.keys()
    assert tags == {"int", "fraction", "decimal", "map", "tuple", "bool", "str", "symbolic"}


@pytest.mark.parametrize("placement", ["flat", "task_data", "verifier_metadata"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("description", "task-owned description"),
        ("request_privacy", {"private_requests": False}),
        ("request_policy", {"no_resubmission": False}),
        ("execution", {"snapshot": "task-owned"}),
    ],
)
def test_contributed_tasks_cannot_select_operator_policy(contributed_rows, placement, field, value):
    row = deepcopy(contributed_rows[0])
    target = row if placement == "flat" else row.setdefault(placement, {})
    target[field] = value
    with pytest.raises(ValidationError) as caught:
        adapter._task(adapter.CritPtRunRequest.model_validate(row))
    assert ((field,), "extra_forbidden") in [(error["loc"], error["type"]) for error in caught.value.errors()]


def test_contributed_task_schema_rejects_excess_cases(contributed_rows):
    fields = deepcopy(contributed_rows[0]["task_data"])
    fields["test_cases"] *= task_schema.MAX_CASES + 1
    with pytest.raises(ValidationError, match="testcase count limit"):
        TaskData.model_validate(fields)


@pytest.mark.parametrize(
    "node",
    [
        ["int", 1],
        ["symbolic", {}],
        ["bool", 1],
        ["fraction", "1"],
        ["tuple", "not-a-list"],
        ["map", [["duplicate", ["int", "1"]], ["duplicate", ["int", "2"]]]],
    ],
)
def test_fixture_structure_check_rejects_malformed_tags(schema_only, node):
    with pytest.raises(AssertionError):
        assert_fixture_wire_structure({"format": task_schema.WIRE_FORMAT, "value": node})


def test_comparison_defaults_validate_at_load_and_default_to_delivered_verdict_values():
    execution = ExecutionPolicy(
        snapshot="synthetic-pinned-snapshot",
        os_user="grader",
        owner_id="synthetic",
        journal_dir="/tmp/unused-journal",
    )
    base = dict(
        host="127.0.0.1",
        port=8000,
        name="synthetic-grader",
        entrypoint="app.py",
        execution=execution,
        request_policy={"no_resubmission": True, "deadline_seconds": 20_000.0},
        request_privacy={"private_requests": True},
    )
    default = adapter.CritPtCustomGraderConfig(**base)
    assert (default.comparison.default_rtol, default.comparison.default_atol) == ("5e-12", "0")
    explicit = adapter.CritPtCustomGraderConfig(**base, comparison={"default_rtol": "1e-6", "default_atol": "0"})
    assert (explicit.comparison.default_rtol, explicit.comparison.default_atol) == ("1e-6", "0")
    for bad in ({"default_rtol": "abc"}, {"default_atol": "-1e-8"}, {"default_rtol": None}, {"default_rtol": "nan"}):
        with pytest.raises(ValidationError):
            adapter.CritPtCustomGraderConfig(**base, comparison=bad)
