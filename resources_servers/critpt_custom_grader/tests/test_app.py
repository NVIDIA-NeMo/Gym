# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic in-memory ASGI tests. No source execution, provider, sockets or model calls: the grader is a fake."""

import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi.exceptions import RequestValidationError
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.config import DatasetConfig, visit_resources_server
from nemo_gym.config.config_types import ResourcesServerTypeConfig, ResponsesAPIAgentServerTypeConfig
from nemo_gym.core import failure_kinds
from nemo_gym.data import normalize_task_fields
from nemo_gym.server_runtime import server_utils
from resources_servers.critpt_custom_grader import app as adapter
from resources_servers.critpt_custom_grader import task_data as task_schema
from resources_servers.critpt_custom_grader.execution import (
    FAILED_CATEGORIES,
    UNSCORABLE_CATEGORIES,
    GradeResult,
    RuntimeProbeError,
)
from resources_servers.critpt_custom_grader.task_data import TaskData
from resources_servers.critpt_custom_grader.tests.support import ScriptedProvider
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


SOURCE = "def solve(value):\n    return value\n"
REFERENCE_MARKER = "synthetic_reference_marker"
EXPECTED_MARKER = "synthetic_expected_marker"
NONTERMINAL_CATEGORIES = tuple(name for name, (_, terminal) in UNSCORABLE_CATEGORIES.items() if not terminal)
TERMINAL_CATEGORIES = tuple(name for name, (_, terminal) in UNSCORABLE_CATEGORIES.items() if terminal)
CLEAN = {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"}
# Every category probe_runtime raises: each one refuses to serve. startup() itself raises a seventh, image_missing,
# before any probe; the image tests below cover it.
PROBE_CATEGORIES = (
    "exec_user_unsupported",
    "exec_user_is_root",
    "interpreter_missing_dependency",
    "provider_create",
    "upload_unsupported",
    "provider_exec",
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
        cleanup=dict(CLEAN),
        sandboxes={
            name: {
                "kind": "sandbox",
                "role": "verifier",
                "provider": "critpt_fake",
                "sandbox_id": f"fake-{name}-1",
                "outcome": "completed",
                "exit_code": 0,
                "wall_time_s": 0.5,
                "error_type": None,
            }
            for name in CLEAN
        },
    )
    return replace(result, **overrides)


def unscorable_verdict(category="comparator_uncertain", **overrides):
    failure_class, terminal = UNSCORABLE_CATEGORIES[category]
    fields = {
        "outcome": "unscorable",
        "reward": None,
        "category": category,
        "failure_class": failure_class,
        "terminal": terminal,
        "cases_attempted": 0,
        "cases_equal": 0,
    }
    fields.update(overrides)
    return verdict(**fields)


def failed_verdict(category="candidate_mismatch", **overrides):
    fields = {
        "outcome": "failed",
        "reward": 0.0,
        "category": category,
        "cases_attempted": 1,
        "cases_equal": 0,
        "first_failed_case": 0,
    }
    fields.update(overrides)
    return verdict(**fields)


@pytest.fixture(autouse=True)
def no_runtime_work(monkeypatch):
    blocked = MagicMock(side_effect=AssertionError("host schema/dispatch check attempted runtime work"))
    targets = {
        "resources_servers.critpt_custom_grader.runner": (
            "parse_source",
            "materialize_input",
            "adapt_output",
            "_invoke",
            "_run_worker",
            "_compare_worker",
            "execute_run",
            "execute_compare",
        ),
        "resources_servers.critpt_custom_grader.codec": ("decode_value", "encode_value"),
        "resources_servers.critpt_custom_grader.comparator": ("compare_request",),
        "resources_servers.critpt_custom_grader.symbolic": ("parse_expression",),
    }
    for name, functions in targets.items():
        module = sys.modules.get(name)
        if module is not None:
            for function in functions:
                if hasattr(module, function):
                    monkeypatch.setattr(module, function, blocked)
    yield
    blocked.assert_not_called()


@pytest.fixture
def make_server():
    def make(grade=None, **overrides):
        config = adapter.CritPtCustomGraderConfig(
            host="127.0.0.1",
            port=8000,
            name="synthetic-grader",
            entrypoint="app.py",
            sandbox_provider={"critpt_fake": {}},
            sandbox_spec={"image": "example/grader:test"},
            startup_probe=False,
            **overrides,
        )
        server = adapter.CritPtCustomGraderServer(config=config, server_client=MagicMock(spec=ServerClient))
        fake = MagicMock()
        fake.grade = AsyncMock(return_value=grade if grade is not None else verdict())
        fake.shutdown = AsyncMock(return_value=None)
        server._grader = fake
        return server, server.setup_webserver(), fake

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


def assert_unscorable(result, *, terminal=False):
    assert result["reward"] == 0.0 and result["scored"] is False and result["mask_sample"] is True
    assert result["_ng_failure_class"] == ("verifier_error" if terminal else "provider_unavailable")
    assert result["failure_kind"] == result["_ng_failure_class"] and result["failure_reason"] == result["category"]
    if terminal:
        assert result["_ng_failure_terminal"] is True
    else:
        assert "_ng_failure_terminal" not in result


def probing_server(provider, *, startup_probe=True, **spec_fields):
    """A server that resolves ``provider`` from its config and, unless told not to, probes at startup."""
    config = adapter.CritPtCustomGraderConfig(
        host="127.0.0.1",
        port=8000,
        name="synthetic-grader",
        entrypoint="app.py",
        sandbox_provider=provider,
        sandbox_spec={"image": "example/grader:test", **spec_fields},
        execution={"exec_user": "grader"},
        startup_probe=startup_probe,
    )
    return adapter.CritPtCustomGraderServer(config=config, server_client=MagicMock(spec=ServerClient))


# --- review regressions ------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stock_agent_defaults_tool_choice_auto_with_empty_tools_are_accepted(make_server):
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    server, app, fake = make_server()
    request = payload()
    # Exactly what the stock simple_agent posts: its body's model_dump(), stock defaults included.
    request["responses_create_params"] = NeMoGymResponseCreateParamsNonStreaming(
        input="Return a Python function named solve."
    ).model_dump()
    assert request["responses_create_params"]["tool_choice"] == "auto"
    assert request["responses_create_params"]["tools"] == []
    status, result = await post(app, "/verify", request)
    assert status == 200 and result["reward"] == 1.0 and result["scored"] is True and result["mask_sample"] is False
    status, result = await post(app, "/seed_session", {k: v for k, v in request.items() if k != "response"})
    assert status == 200
    request["responses_create_params"]["tools"] = [{"type": "function", "name": "f", "parameters": {}}]
    status, result = await post(app, "/verify", request)
    assert (status, result) == (422, {"error": {"category": "request_invalid"}})
    request["responses_create_params"] = {"input": "x", "tool_choice": {"type": "function", "name": "f"}, "tools": []}
    status, result = await post(app, "/verify", request)
    assert (status, result) == (422, {"error": {"category": "request_invalid"}})


@pytest.mark.asyncio
async def test_framework_bookkeeping_keys_are_stripped_before_task_validation(make_server):
    server, app, fake = make_server()
    request = payload()
    request.update(
        {
            "_ng_attempt_index": 1,
            "_ng_environment_server": "critpt_custom_grader_environment_server",
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
            "agent_ref": {"name": "critpt_custom_grader_simple_agent"},
            "task_source": "example",
            "skills_ref": None,
            "reference_ids": [],
            "deliverables_dir": None,
            "reuse_cached_deliverable": True,
        }
    )
    status, result = await post(app, "/verify", request)
    assert status == 200 and result["category"] == "passed"
    validated, source = fake.grade.await_args.args
    assert isinstance(validated, TaskData) and validated.problem_id == "synthetic-1" and source == SOURCE.strip()
    bad = payload(unexpected_task_field=REFERENCE_MARKER)
    status, result = await post(app, "/verify", bad)
    assert status == 200 and result["category"] == "task_invalid"
    assert_unscorable(result, terminal=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", NONTERMINAL_CATEGORIES + TERMINAL_CATEGORIES)
async def test_every_unscorable_category_masks_the_sample_and_keeps_its_name(make_server, category):
    server, app, fake = make_server(grade=unscorable_verdict(category))
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["category"] == category and result["_ng_failure_subcategory"] == category
    assert_unscorable(result, terminal=UNSCORABLE_CATEGORIES[category][1])


@pytest.mark.asyncio
@pytest.mark.parametrize("category", sorted(FAILED_CATEGORIES))
async def test_scored_zero_rows_are_not_masked_and_carry_no_failure_keys(make_server, category):
    extra = (
        {"first_failed_case": None, "cases_attempted": 0}
        if category
        in ("candidate_source_error", "candidate_timeout", "candidate_worker_abort", "candidate_source_limit")
        else {}
    )
    server, app, fake = make_server(grade=failed_verdict(category, **extra))
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["reward"] == 0.0 and result["scored"] is True and result["mask_sample"] is False
    assert result["category"] == category and "_ng_failure_class" not in result and result.get("failure_kind") is None


@pytest.mark.asyncio
async def test_sandbox_observations_and_cleanup_status_are_reported(make_server):
    server, app, fake = make_server(grade=verdict(cleanup={**CLEAN, "candidate": "delete_failed"}))
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["category"] == "passed" and result["cleanup_status"] == "unresolved"
    assert {obs["role"] for obs in result["sandbox_observations"]} == {"verifier"} and len(
        result["sandbox_observations"]
    ) == 3


@pytest.mark.asyncio
async def test_malformed_and_oversized_bodies_answer_category_only(make_server, caplog):
    server, app, fake = make_server()
    status, result = await post(app, "/verify", raw=b'{"not json')
    assert (status, result) == (422, {"error": {"category": "request_invalid"}})
    status, result = await post(app, "/verify", raw=b"x" * (adapter.MAX_REQUEST_BYTES + 1))
    assert (status, result) == (413, {"error": {"category": "request_limit"}})
    status, result = await post(app, "/verify", payload(entrypoint=""))
    assert status == 200 and result["category"] == "task_invalid"
    assert REFERENCE_MARKER not in caplog.text


@pytest.mark.asyncio
async def test_verify_before_startup_is_a_503_and_reverify_mode_is_stateless(make_server):
    server, app, fake = make_server()
    server._grader = None
    status, result = await post(app, "/verify", payload())
    assert (status, result) == (503, {"error": {"category": "server_not_ready"}})
    assert server.config.REVERIFY_MODE.value == "stateless"


@pytest.mark.asyncio
async def test_seed_session_reports_request_errors_as_request_invalid_and_task_errors_as_task_invalid(
    make_server, caplog
):
    server, app, fake = make_server()
    row = {key: value for key, value in payload().items() if key != "response"}
    assert await post(app, "/seed_session", row) == (200, {})
    request_invalid = (422, {"error": {"category": "request_invalid"}})
    tool = {"type": "function", "name": "f", "parameters": {}}
    for params in (
        {"input": "x", "tools": [tool]},
        {"input": "x", "tool_choice": {"type": "function", "name": "f"}},
        {"input": 5},
        "not an object",
    ):
        assert await post(app, "/seed_session", {**row, "responses_create_params": params}) == request_invalid
    no_params = {key: value for key, value in row.items() if key != "responses_create_params"}
    assert await post(app, "/seed_session", no_params) == request_invalid
    for change in ({"entrypoint": ""}, {"unexpected_task_field": REFERENCE_MARKER}):
        assert await post(app, "/seed_session", {**row, **change}) == (422, {"error": {"category": "task_invalid"}})
    fake.grade.assert_not_awaited()
    assert REFERENCE_MARKER not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,category",
    [
        ({"cleanup": {"candidate": []}}, "passed"),
        ({"cleanup": {**CLEAN, "candidate": []}}, "passed"),
        ({"sandboxes": {"candidate": "not an observation"}}, "execution_unknown"),
    ],
    ids=["partial_cleanup_map", "unhashable_cleanup_status", "observation_not_a_mapping"],
)
async def test_malformed_grader_results_answer_a_category_row_never_a_500(make_server, caplog, overrides, category):
    server, app, fake = make_server(grade=verdict(**overrides))
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["category"] == category and result["cleanup_status"] == "unknown"
    if category == "execution_unknown":
        assert_unscorable(result)
        assert result["sandbox_observations"] == []
        assert "verify projection failed (ValidationError)" in caplog.text  # the class name, nothing more
    assert "not an observation" not in json.dumps(result) + caplog.text


# --- startup, shutdown and end to end ----------------------------------------------------------- #


@pytest.mark.asyncio
async def test_startup_resolves_the_provider_probes_once_and_passes_the_whole_spec():
    ScriptedProvider.reset()
    files = {"/work/operator/notice.txt": "operator-supplied file"}
    # The probe's size check finds its one-byte upload through the ("result", "probe") entry.
    server = probing_server(
        {"critpt_fake": {"script": {("result", "probe"): b"x"}}}, files=files, metadata={"team": "synthetic"}
    )
    await server.startup()
    (provider,) = ScriptedProvider.instances
    (spec,) = provider.created
    assert spec.files == files and spec.image == "example/grader:test" and spec.workdir == "/work"
    assert spec.metadata["team"] == "synthetic" and spec.metadata["nemo_gym_server"] == "synthetic-grader"
    assert provider.sandboxes["fake-probe-1"]["files"]["/work/operator/notice.txt"] == b"operator-supplied file"
    assert provider.closed == ["fake-probe-1"] and server._probe_info["uid"] == "1000" and server._grader is not None
    await server.startup()
    assert len(provider.created) == 1
    await server.shutdown()
    assert provider.aclosed == 1


@pytest.mark.asyncio
async def test_startup_without_the_probe_starts_no_sandbox():
    ScriptedProvider.reset()
    server = probing_server({"critpt_fake": {}}, startup_probe=False)
    await server.startup()
    (provider,) = ScriptedProvider.instances
    assert server._grader is not None and server._probe_info == {}
    assert provider.created == [] and provider.exec_calls == []
    await server.shutdown()


@pytest.mark.asyncio
async def test_shutdown_closes_the_provider_even_when_the_grader_drain_is_cancelled():
    ScriptedProvider.reset()
    # The probe's size check finds its one-byte upload, so startup records what the probe saw.
    server = probing_server({"critpt_fake": {"script": {("result", "probe"): b"x"}}})
    await server.startup()
    first = server._grader
    assert server._probe_info["uid"] == "1000"
    first.shutdown = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await server.shutdown()
    (provider,) = ScriptedProvider.instances
    assert first.shutdown.await_count == 1 and provider.aclosed == 1
    assert server._grader is None and server._provider is None and server._probe_info == {}
    await server.shutdown()  # a second call is a no-op
    assert provider.aclosed == 1
    await server.startup()  # and the server can start again, probing afresh
    assert server._grader is not None and server._grader is not first and len(ScriptedProvider.instances) == 2
    assert server._probe_info["uid"] == "1000" and ScriptedProvider.instances[1].closed == ["fake-probe-1"]
    await server.shutdown()
    assert ScriptedProvider.instances[1].aclosed == 1 and server._probe_info == {}


@pytest.mark.asyncio
async def test_end_to_end_verify_through_the_real_grader_and_the_scripted_provider():
    from resources_servers.critpt_custom_grader.tests.support import CANDIDATE, happy_script, value_outcome, wire

    ScriptedProvider.reset()
    # Every scripted run and comparison succeeds; the probe's size check finds its one-byte upload.
    server = probing_server({"critpt_fake": {"script": {**happy_script(), ("result", "probe"): b"x"}}})
    app = server.setup_webserver()
    await server.startup()
    request = {
        "responses_create_params": {"input": "Return a Python function named f."},
        "problem_id": "p1",
        "reference_source": CANDIDATE,
        "entrypoint": "f",
        "test_cases": [
            {"args": [wire(["int", str(n)])], "expected": value_outcome(["int", str(n + 1)])} for n in (1, 2)
        ],
        "response": saved_response(f"```python\n{CANDIDATE}```"),
    }
    status, result = await post(app, "/verify", request)
    (provider,) = ScriptedProvider.instances
    assert status == 200 and result["category"] == "passed" and result["reward"] == 1.0 and result["scored"] is True
    assert result["mask_sample"] is False and "_ng_failure_class" not in result and result["problem_id"] == "p1"
    assert result["cleanup_status"] == "resolved"
    observations = result["sandbox_observations"]
    assert [obs["sandbox_id"].split("-")[1] for obs in observations] == ["reference", "comparator", "candidate"]
    for obs in observations:
        assert (obs["role"], obs["provider"], obs["outcome"], obs["exit_code"]) == (
            "verifier",
            "critpt_fake",
            "completed",
            0,
        )
    assert set(provider.closed) == {"fake-probe-1"} | {obs["sandbox_id"] for obs in observations}
    assert not any(box["live"] for box in provider.sandboxes.values())
    await server.shutdown()
    assert provider.aclosed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("category", PROBE_CATEGORIES)
async def test_startup_refuses_to_serve_on_every_probe_category(monkeypatch, category):
    ScriptedProvider.reset()
    probe = AsyncMock(side_effect=RuntimeProbeError(category, "synthetic detail"))
    monkeypatch.setattr(adapter, "probe_runtime", probe)
    server = probing_server({"critpt_fake": {}})
    with pytest.raises(RuntimeProbeError) as caught:
        await server.startup()
    (provider,) = ScriptedProvider.instances
    assert caught.value.category == category and probe.await_count == 1
    assert server._grader is None and server._provider is None and provider.aclosed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_probe", [True, False])
@pytest.mark.parametrize("image", [None, ""])  # an empty CRITPT_CUSTOM_GRADER_IMAGE resolves to "", not null
async def test_startup_without_an_image_refuses_to_serve_before_any_sandbox(startup_probe, image):
    # No image is shipped, and the server runs without one only on the local provider: checked before the probe,
    # and without it. Unset and empty are the same refusal.
    ScriptedProvider.reset()
    server = probing_server({"critpt_fake": {}}, startup_probe=startup_probe, image=image)
    with pytest.raises(RuntimeProbeError) as caught:
        await server.startup()
    (provider,) = ScriptedProvider.instances
    assert caught.value.category == "image_missing"
    assert provider.created == [] and provider.exec_calls == [] and provider.aclosed == 1
    assert server._grader is None and server._provider is None and server._probe_info == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_options", [{"snapshot_id": "grader-snapshot"}, {"template": "grader-template"}])
async def test_startup_refuses_a_snapshot_or_template_in_place_of_the_image(provider_options):
    # Deliberate: some providers could start from a snapshot or template without an image, but the reference cache
    # names the runtime by its image, not by provider_options, so such a setup is refused like any missing image.
    ScriptedProvider.reset()
    server = probing_server({"critpt_fake": {}}, image=None, provider_options=provider_options)
    with pytest.raises(RuntimeProbeError) as caught:
        await server.startup()
    (provider,) = ScriptedProvider.instances
    assert caught.value.category == "image_missing" and provider.created == [] and provider.aclosed == 1
    assert server._grader is None and server._provider is None


@pytest.mark.asyncio
async def test_startup_without_an_image_serves_on_a_provider_named_local(monkeypatch):
    # The exemption keys on the provider's name: the same double, named as the local provider is, starts and probes.
    ScriptedProvider.reset()
    monkeypatch.setattr(ScriptedProvider, "name", "local")
    server = probing_server({"critpt_fake": {"script": {("result", "probe"): b"x"}}}, image=None)
    await server.startup()
    (provider,) = ScriptedProvider.instances
    (spec,) = provider.created
    assert spec.image is None and provider.closed == ["fake-probe-1"] and server._probe_info["uid"] == "1000"
    assert server._grader is not None
    await server.shutdown()
    assert provider.aclosed == 1


@pytest.mark.asyncio
async def test_startup_without_an_image_serves_on_gyms_local_provider():
    # Gym's own local provider, selected by name as an operator's config selects it. Without the probe nothing runs.
    server = probing_server({"local": {}}, startup_probe=False, image=None)
    await server.startup()
    assert server._provider.name == "local" and server._grader is not None
    await server.shutdown()
    assert server._provider is None and server._grader is None


@pytest.mark.asyncio
async def test_startup_builds_the_grader_from_the_execution_policy_and_comparison_defaults(monkeypatch):
    ScriptedProvider.reset()
    grader_class = MagicMock()
    monkeypatch.setattr(adapter, "Grader", grader_class)
    config = adapter.CritPtCustomGraderConfig(
        host="127.0.0.1",
        port=8000,
        name="synthetic-grader",
        entrypoint="app.py",
        sandbox_provider={"critpt_fake": {}},
        sandbox_spec={"image": "example/grader:test"},
        execution={"exec_user": "grader", "max_concurrent_jobs": 2},
        comparison={"default_rtol": "1e-6", "default_atol": "1e-9"},
        startup_probe=False,
    )
    server = adapter.CritPtCustomGraderServer(config=config, server_client=MagicMock(spec=ServerClient))
    await server.startup()
    (provider,) = ScriptedProvider.instances
    grader_class.assert_called_once()
    args, kwargs = grader_class.call_args
    assert args == (server.config.execution,) and server.config.execution.max_concurrent_jobs == 2
    assert kwargs["provider"] is provider
    assert kwargs["spec"].image == "example/grader:test" and kwargs["spec"].workdir == "/work"
    assert (kwargs["default_rtol"], kwargs["default_atol"]) == ("1e-6", "1e-9")
    assert kwargs["reference_cache"] is not None and server._grader is grader_class.return_value


@pytest.mark.asyncio
async def test_lifespan_drains_the_grader_and_logs_a_provider_close_failure_by_class_only(make_server, caplog):
    server, app, fake = make_server()
    order = []

    async def drain():
        order.append("grader")

    async def failing_close():
        order.append("provider")
        raise RuntimeError(REFERENCE_MARKER)

    fake.shutdown.side_effect = drain
    server._provider = MagicMock(aclose=AsyncMock(side_effect=failing_close))
    async with app.router.lifespan_context(app):
        assert (await post(app, "/verify", payload()))[0] == 200
        assert order == []
    assert order == ["grader", "provider"]
    assert "RuntimeError" in caplog.text and REFERENCE_MARKER not in caplog.text


@pytest.mark.asyncio
async def test_direct_verify_and_frozen_replay(make_server, capsys, caplog):
    server, app, fake = make_server()
    request = payload()
    request["response"]["metadata"] = {"synthetic": "replay"}
    original = deepcopy(request)

    status, result = await post(app, "/verify", request)
    assert status == 200
    assert result["reward"] == 1.0 and result["scored"] is True
    assert result["response"] == original["response"]
    assert result["responses_create_params"] == original["responses_create_params"]
    assert result["mask_sample"] is False
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
    fresh, fresh_app, fresh_fake = make_server()
    replay = {**original, "response": result["response"]}
    status, recovered = await post(fresh_app, "/verify", replay)
    assert status == 200
    assert recovered == result
    assert fresh_fake.grade.await_args.args[1] == source
    assert not fresh.server_client.mock_calls
    rendered = json.dumps(result) + capsys.readouterr().out + caplog.text
    assert REFERENCE_MARKER not in rendered
    assert EXPECTED_MARKER not in rendered
    assert "synthetic-job" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["flat", "verifier_metadata", "task_data"])
async def test_seed_validates_without_allocating_or_storing_private_data(make_server, placement):
    server, app, fake = make_server()
    row = {"responses_create_params": {"input": "Public synthetic prompt"}}
    row.update(task_fields() if placement == "flat" else {placement: task_fields()})
    row.update({"agent_ref": {"name": "synthetic-agent"}, "task_source": "synthetic", "_ng_task_index": 4})
    status, result = await post(app, "/seed_session", row)
    assert (status, result) == (200, {})
    fake.grade.assert_not_awaited()
    assert server._grader is fake and server._provider is None
    assert not server.server_client.mock_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"entrypoint": ""},
        {"test_cases": []},
        {"entry": "solve"},
        {"execution": {"exec_user": "task-owned"}},
        {"sandbox_spec": {"image": "task-owned"}},
        {"unexpected": REFERENCE_MARKER},
        {"_ng_failure_class": "judge_failed", "_ng_failure_terminal": False, "reward": 1.0},
        {"verifier_metadata": {"entrypoint": "different"}},
        {"verifier_metadata": {"responses_create_params": {"input": "nested runtime field"}}},
    ],
)
async def test_invalid_task_is_terminal_before_allocation(make_server, changes):
    _, app, fake = make_server()
    status, result = await post(app, "/verify", payload(**changes))
    assert status == 200
    assert_unscorable(result, terminal=True)
    assert result["category"] == "task_invalid"
    assert result["response"] == saved_response()
    assert "unexpected" not in result and "execution" not in result and "sandbox_spec" not in result
    fake.grade.assert_not_awaited()
    seed = payload(**changes)
    seed.pop("response")
    assert await post(app, "/seed_session", seed) == (422, {"error": {"category": "task_invalid"}})


@pytest.mark.asyncio
async def test_equal_flat_and_legacy_values_use_gym_normalization(make_server):
    _, app, fake = make_server()
    row = payload(verifier_metadata=task_fields())
    assert (await post(app, "/verify", row))[1]["scored"] is True
    assert fake.grade.await_args.args[0].problem_id == "synthetic-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/verify", "/seed_session"])
@pytest.mark.parametrize("malformation", ["json", "model", "tools"])
async def test_pre_endpoint_privacy_survives_late_handler_registration(
    make_server, path, malformation, capsys, caplog
):
    server, app, fake = make_server()
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
    captured = capsys.readouterr()
    assert REFERENCE_MARKER not in captured.out + captured.err + caplog.text + json.dumps(result)


@pytest.mark.asyncio
async def test_malformed_saved_response_and_ingress_limit(make_server, monkeypatch):
    _, app, fake = make_server()
    row = payload()
    row["response"] = {"private": REFERENCE_MARKER}
    assert await post(app, "/verify", row) == (422, {"error": {"category": "request_invalid"}})
    monkeypatch.setattr(adapter, "MAX_REQUEST_BYTES", 64)
    assert await post(app, "/verify", raw=b" " * 65) == (413, {"error": {"category": "request_limit"}})
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_number_materialization_and_http_round_trip(make_server):
    _, app, fake = make_server()
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
    server, app, fake = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    expected = f"  {EXPECTED_MARKER}: [1, 2]\n"
    row = payload(test_cases=[{"inputs": {"value": text}, "expected_output": expected}])
    original = deepcopy(row)
    seed = {key: value for key, value in row.items() if key != "response"}
    assert await post(app, "/seed_session", seed) == (200, {})
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
    _, app, fake = make_server()
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
    _, app, fake = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    row = payload(test_cases=[{"input": [{"__complex__": [1, 2.0]}], "output": {"__complex__": [3, 4]}}])
    status, result = await post(app, "/verify", row)
    assert status == 200
    validated = fake.grade.await_args.args[0]
    assert validated.test_cases[0].args[0].value == ["complex", ["int", "1"], ["float", (2.0).hex()]]
    assert validated.test_cases[0].expected.value.value == ["complex", ["int", "3"], ["int", "4"]]


@pytest.mark.asyncio
async def test_input_conversions_reach_the_grader_task(make_server):
    _, app, fake = make_server()
    fake.grade.return_value = unscorable_verdict(category="reference_error")
    row = payload(input_conversions=[None, "symbol", "function"])
    status, result = await post(app, "/verify", row)
    assert status == 200
    assert fake.grade.await_args.args[0].input_conversions == [None, "symbol", "function"]


@pytest.mark.asyncio
@pytest.mark.parametrize("node", [["unknown", "bounded"], ["fraction", "1"], ["int", "not-an-integer"]])
async def test_bounded_wire_transport_does_not_establish_semantic_acceptance(make_server, node):
    _, app, fake = make_server()
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
    _, app, fake = make_server()
    fake.grade.return_value = verdict(outcome="failed", reward=0.0, category=category, cases_equal=0)
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["scored"] is True and result["reward"] == 0.0
    assert "_ng_failure_class" not in result
    assert result["category"] == category


@pytest.mark.asyncio
@pytest.mark.parametrize("category", NONTERMINAL_CATEGORIES)
async def test_nonterminal_execution_category_is_preserved_without_fault(make_server, category):
    server, app, fake = make_server()
    fake.grade.return_value = unscorable_verdict(category=category)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == category
    assert result["cleanup_status"] == "resolved"
    assert result["response"] == saved_response()
    fake.grade.assert_awaited_once()
    assert not server.server_client.mock_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category,attempted,cleanup,cleanup_status",
    [
        (
            "reference_error",
            0,
            {"reference": "deleted", "comparator": "not_created", "candidate": "not_created"},
            "resolved",
        ),
        (
            "provider_create",
            0,
            {"reference": "not_created", "comparator": "not_created", "candidate": "not_created"},
            "resolved",
        ),
        ("comparator_uncertain", 2, dict(CLEAN), "resolved"),
        ("comparator_no_result", 2, dict(CLEAN), "resolved"),
        ("provider_exec", 1, dict(CLEAN), "resolved"),
        ("provider_exec", 1, {**CLEAN, "candidate": "delete_failed"}, "unresolved"),
        ("server_draining", 0, {}, "not_started"),
    ],
)
async def test_nonterminal_counts_and_resolved_cleanup_shapes(
    make_server, category, attempted, cleanup, cleanup_status
):
    _, app, fake = make_server()
    row = payload(test_cases=[{"input": [1], "output": 1}, {"input": [2], "output": 2}])
    fake.grade.return_value = unscorable_verdict(
        category=category, case_count=2, cases_attempted=attempted, cleanup=cleanup
    )
    status, result = await post(app, "/verify", row)
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == category
    assert result["cleanup_status"] == cleanup_status
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
    _, app, fake = make_server()
    fake.grade.return_value = unscorable_verdict(category=category, **changes)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"
    # A rejected result vouches for nothing, its cleanup report included.
    assert result["cleanup_status"] == "unknown"
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", NONTERMINAL_CATEGORIES)
@pytest.mark.parametrize("contract", ["passed", "failed", "terminal"])
async def test_nonterminal_categories_cannot_enter_scored_or_terminal_allowlists(make_server, category, contract):
    _, app, fake = make_server()
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
        {"cases_equal": 2},
    ],
)
async def test_scored_result_requires_valid_counts(make_server, changes):
    _, app, fake = make_server()
    fake.grade.return_value = verdict(**changes)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"
    # A rejected result vouches for nothing, its cleanup report included.
    assert result["cleanup_status"] == "unknown"
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
    # A category outside the nonterminal allowlist is unknown whatever its type, an unhashable one included.
    _, app, fake = make_server()
    fake.grade.return_value = verdict(
        outcome="unscorable",
        reward=None,
        category=category,
        failure_class=failure_kinds.PROVIDER_UNAVAILABLE,
        terminal=False,
        cases_attempted=0,
        cases_equal=0,
    )
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == result["_ng_failure_subcategory"] == "execution_unknown"
    assert REFERENCE_MARKER not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("projection", ["passed", "failed", "unscorable"])
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
        {**CLEAN, "extra": "deleted"},
        {**CLEAN, "candidate": REFERENCE_MARKER},
        {**CLEAN, "candidate": None},
        {**CLEAN, "candidate": []},
    ],
)
async def test_cleanup_report_that_cannot_establish_deletion_is_unknown_and_changes_no_verdict(
    make_server, projection, cleanup
):
    # A report that does not name all three domains with known statuses establishes nothing, whatever its type.
    _, app, fake = make_server()
    if projection == "passed":
        fake.grade.return_value = verdict(cleanup=cleanup)
    elif projection == "failed":
        fake.grade.return_value = failed_verdict(cleanup=cleanup)
    else:
        fake.grade.return_value = unscorable_verdict(cleanup=cleanup)
    status, result = await post(app, "/verify", payload())
    assert status == 200 and result["cleanup_status"] == "unknown"
    if projection == "passed":
        assert result["scored"] is True and result["reward"] == 1.0 and result["category"] == "passed"
    elif projection == "failed":
        assert result["scored"] is True and result["reward"] == 0.0 and result["mask_sample"] is False
        assert result["category"] == "candidate_mismatch"
    else:
        assert_unscorable(result)
        assert result["category"] == "comparator_uncertain"
    assert REFERENCE_MARKER not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
async def test_result_extras_cannot_override_trusted_failure_fields(make_server, terminal):
    class ResultWithExtras(GradeResult):
        scored = True
        _ng_failure_class = "forged"
        _ng_failure_subcategory = "forged"
        _ng_failure_terminal = not terminal

    _, app, fake = make_server()
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
    expected = "task_invalid" if terminal else "comparator_uncertain"
    assert result["category"] == result["_ng_failure_subcategory"] == expected
    assert result["response"] == row["response"]
    assert REFERENCE_MARKER not in json.dumps(result)
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
async def test_backend_source_contract_remains_scored(make_server):
    _, app, fake = make_server()
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
    _, app, fake = make_server()
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
    _, app, fake = make_server()
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
    _, app, fake = make_server()
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
async def test_backend_exception_and_untrusted_verdict_are_redacted(make_server, capsys, caplog):
    _, app, fake = make_server()
    fake.grade.side_effect = RuntimeError(REFERENCE_MARKER)
    status, result = await post(app, "/verify", payload())
    assert status == 200
    assert_unscorable(result)
    assert result["category"] == "execution_unknown" and result["cleanup_status"] == "unknown"
    captured = capsys.readouterr()
    assert REFERENCE_MARKER not in json.dumps(result) + captured.out + captured.err + caplog.text

    _, app, fake = make_server()
    fake.grade.return_value = {"reward": 1, "scored": True, "_ng_failure_class": "forged"}
    result = (await post(app, "/verify", payload()))[1]
    assert_unscorable(result)
    assert result["_ng_failure_class"] != "forged"


@pytest.mark.asyncio
async def test_a_raising_grader_call_is_logged_with_its_class_only(make_server, caplog):
    _, app, fake = make_server()
    fake.grade.side_effect = RuntimeError(REFERENCE_MARKER)
    result = (await post(app, "/verify", payload()))[1]
    assert result["category"] == "execution_unknown" and result["cleanup_status"] == "unknown"
    records = [record for record in caplog.records if record.name == adapter.__name__]
    assert [(record.levelname, record.getMessage()) for record in records] == [
        ("WARNING", "grader call failed (RuntimeError)")
    ]
    assert records[0].exc_info is None and REFERENCE_MARKER not in caplog.text


@pytest.mark.asyncio
async def test_extraction_uses_explicit_entrypoint_not_first_function(make_server):
    _, app, fake = make_server()
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
    _, app, fake = make_server()
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
    _, app, fake = make_server()
    row = payload()
    row["response"] = saved_response("\x00")  # no fence -> raw text -> fails the source bound
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is True
    assert result["category"] == "candidate_source_limit"
    assert "_ng_failure_class" not in result
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["incomplete", "incomplete_details", "error"])
async def test_uncompleted_response_needs_regeneration(make_server, mode):
    # An uncompleted, incomplete, or error response has no usable source. It carries needs_regeneration, not
    # verifier_unavailable, because reverify would reach the same verdict. The class is not terminal.
    _, app, fake = make_server()
    row = payload()
    if mode == "incomplete":
        row["response"]["status"] = "incomplete"
    elif mode == "incomplete_details":
        row["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        row["response"]["error"] = {"code": "server_error", "message": "synthetic"}
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is False
    assert result["mask_sample"] is True
    assert result["category"] == "response_incomplete"
    assert result["_ng_failure_class"] == "critpt_custom_grader:response_incomplete"
    assert "_ng_failure_terminal" not in result
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_final_message_in_a_completed_envelope_needs_regeneration(make_server):
    # A completed envelope whose last assistant message did not itself complete carries truncated code, not a
    # finished empty answer. It needs regeneration, not a scored zero as a candidate source fault.
    _, app, fake = make_server()
    row = payload()
    row["response"]["output"][-1]["status"] = "incomplete"
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is False
    assert result["mask_sample"] is True
    assert result["category"] == "response_incomplete"
    assert result["_ng_failure_class"] == "critpt_custom_grader:response_incomplete"
    assert "_ng_failure_terminal" not in result
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty_text", "no_message"])
async def test_completed_response_with_empty_source_is_a_candidate_fault(make_server, mode):
    # A completed response with no usable source is an empty submission. It is scored 0 as a candidate source
    # fault, before any sandbox is created and without grading.
    _, app, fake = make_server()
    row = payload()
    if mode == "empty_text":
        row["response"] = saved_response("")
    else:
        row["response"]["output"] = []
    result = (await post(app, "/verify", row))[1]
    assert result["reward"] == 0.0 and result["scored"] is True
    assert result["mask_sample"] is False
    assert result["category"] == "candidate_source_limit"
    assert "_ng_failure_class" not in result
    fake.grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_aggregation_is_scoped_to_supplied_rows(make_server):
    _, app, _ = make_server()
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
    assert metrics["accuracy_scorable"] == 0.5 and metrics["accuracy_all"] == 0.25
    assert metrics["accounting_scope"] == "supplied_rows"
    assert metrics["all_attempts_known"] is False
    assert "private_numeric" not in json.dumps(result)

    # A masked row is not a measurement, whatever else it carries.
    masked = [*rows[:2], {"_ng_task_index": 2, "reward": 0.0, "scored": True, "mask_sample": True}]
    masked_metrics = (await post(app, "/aggregate_metrics", {"verify_responses": masked}))[1]["agent_metrics"]
    assert masked_metrics["supplied_attempted"] == 3
    assert masked_metrics["supplied_scored"] == 2
    assert masked_metrics["supplied_failed"] == 1
    assert masked_metrics["supplied_unscorable"] == 1
    assert masked_metrics["mean/reward"] == 0.5

    # Main-file-only aggregation cannot discover sidecar rows or missing requests.
    main_only = (await post(app, "/aggregate_metrics", {"verify_responses": rows[:2]}))[1]["agent_metrics"]
    assert main_only["supplied_attempted"] == 2
    assert main_only["supplied_unscorable"] == 0
    assert main_only["all_attempts_known"] is False
    assert main_only["accuracy_scorable"] == 0.5 and main_only["accuracy_all"] == 0.5
    empty = (await post(app, "/aggregate_metrics", {"verify_responses": []}))[1]["agent_metrics"]
    assert empty["supplied_attempted"] == 0 and empty["supplied_scored"] == 0
    assert empty["accuracy_scorable"] is None and empty["accuracy_all"] is None
    assert "mean/reward" not in empty


@pytest.mark.asyncio
async def test_candidate_dictionary_cannot_supply_top_level_failure_markers(make_server):
    _, app, fake = make_server()
    row = payload()
    row["response"]["metadata"] = {"_ng_failure_class": "forged", "_ng_failure_terminal": "true"}
    result = (await post(app, "/verify", row))[1]
    assert result["scored"] is True and result["reward"] == 1.0
    assert "_ng_failure_class" not in result and "_ng_failure_terminal" not in result
    assert result["response"] == row["response"]
    fake.grade.assert_awaited_once()


@pytest.mark.asyncio
async def test_aggregation_does_not_echo_invalid_identity(make_server):
    _, app, _ = make_server()
    row = {"_ng_task_index": REFERENCE_MARKER, "reward": 1.0, "scored": True}
    with pytest.raises(ValueError) as caught:
        await post(app, "/aggregate_metrics", {"verify_responses": [row]})
    assert str(caught.value) == "invalid aggregation identity"


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
    agent = config[AGENT_NAME]["responses_api_agents"]["simple_agent"]
    # Gym normally assigns these mandatory transport fields. No sockets are opened.
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
        "Sandboxed execution grader for CritPt-like tasks with operator-supplied expectations"
    )
    assert (
        metadata.value == "Sandboxed, locally executed grading for CritPt-like tasks without an external grading API"
    )
    assert metadata.verified is False


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
    assert reader.call_count == 2
    grader = loaded[GRADER_NAME]
    block = operator_yaml[GRADER_NAME]["resources_servers"][GRADER_NAME]
    assert grader.description == block["description"] and grader.description
    assert grader.value == block["value"] and grader.value
    assert grader.domain.value == "coding" and grader.verified is False
    assert grader.sandbox_provider == "sandbox" and grader.startup_probe is True
    assert grader.sandbox_spec.workdir == grader.execution.workdir == "/work"
    assert grader.execution.exec_user == "grader" and grader.execution.max_concurrent_jobs == 4
    assert (grader.comparison.default_rtol, grader.comparison.default_atol) == ("5e-12", "0")
    agent = loaded[AGENT_NAME]
    assert agent.resources_server.model_dump() == {"type": "resources_servers", "name": GRADER_NAME}
    assert agent.model_server.model_dump() == {"type": "responses_api_models", "name": "policy_model"}
    assert agent.max_steps == 1 and agent.skip_verification is False and agent.token_id_capture is False


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
        ("sandbox_provider", "task-owned"),
        ("sandbox_spec", {"image": "task-owned"}),
        ("execution", {"exec_user": "task-owned"}),
        ("comparison", {"default_rtol": "1"}),
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
    base = dict(
        host="127.0.0.1",
        port=8000,
        name="synthetic-grader",
        entrypoint="app.py",
        sandbox_spec={"image": "example/grader:test"},
    )
    default = adapter.CritPtCustomGraderConfig(**base)
    assert (default.comparison.default_rtol, default.comparison.default_atol) == ("5e-12", "0")
    explicit = adapter.CritPtCustomGraderConfig(**base, comparison={"default_rtol": "1e-6", "default_atol": "0"})
    assert (explicit.comparison.default_rtol, explicit.comparison.default_atol) == ("1e-6", "0")
    for bad in ({"default_rtol": "abc"}, {"default_atol": "-1e-8"}, {"default_rtol": None}, {"default_rtol": "nan"}):
        with pytest.raises(ValidationError):
            adapter.CritPtCustomGraderConfig(**base, comparison=bad)
