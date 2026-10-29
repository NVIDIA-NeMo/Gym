# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Instrumentation attached to Gym's own code: sandbox, servers, CLI supervisor."""

import pytest

from nemo_gym.telemetry import GymSpanGroup
from tests.unit_tests.telemetry.conftest import requires_lens


#: These exercise the telemetry-enabled path, which needs nemo-lens. The absent-lens path
#: is covered by test_fallbacks.py, which runs either way.
pytestmark = requires_lens


@pytest.fixture
def recorded_spans(monkeypatch):
    """Capture spans without installing a global provider."""
    from nemo.lens.state import set_enabled_span_groups
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    import nemo.lens.helpers as lens_helpers

    monkeypatch.setattr(lens_helpers, "get_tracer", lambda *a, **k: tracer)
    monkeypatch.setattr("opentelemetry.trace.get_tracer", lambda *a, **k: tracer)
    set_enabled_span_groups(GymSpanGroup.resolve("all"))
    yield exporter.get_finished_spans
    set_enabled_span_groups(frozenset())


@pytest.fixture
def rollout_attributes_registered():
    """What Gym's telemetry setup does: hand the sandbox API the rollout-id provider."""
    from nemo_gym.sandbox.observability import set_span_attribute_provider
    from nemo_gym.telemetry.setup import rollout_span_attributes

    set_span_attribute_provider(rollout_span_attributes)
    yield
    set_span_attribute_provider(None)


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #


class _StubProvider:
    """Minimal sandbox provider. Instrumentation lives in AsyncSandbox, above every
    real provider, so one stub covers docker, daytona, opensandbox and the rest."""

    name = "stub"

    def __init__(self, return_code: int = 0):
        self._return_code = return_code
        self.closed = False

    async def create(self, spec):
        from nemo_gym.sandbox import SandboxHandle

        return SandboxHandle(sandbox_id="handle-1", provider_name=self.name, raw=None)

    async def exec(self, handle, command, **kwargs):
        from nemo_gym.sandbox import SandboxExecResult

        return SandboxExecResult(stdout="out", stderr="", return_code=self._return_code)

    async def close(self, handle):
        self.closed = True

    async def aclose(self):
        pass


@pytest.fixture
def started_sandbox():
    from nemo_gym.sandbox import AsyncSandbox, SandboxSpec

    def build(return_code: int = 0):
        return AsyncSandbox(_StubProvider(return_code), SandboxSpec(image="ubuntu"))

    return build


async def test_sandbox_start_emits_a_span(recorded_spans, started_sandbox):
    await started_sandbox().start()
    spans = recorded_spans()
    assert "gym.sandbox.start" in [span.name for span in spans]
    start_span = next(span for span in spans if span.name == "gym.sandbox.start")
    assert start_span.attributes["nemo.gym.sandbox.provider"] == "stub"
    assert start_span.attributes["nemo.gym.sandbox.id"] == "handle-1"


async def test_sandbox_exec_emits_a_span_with_the_exit_code(recorded_spans, started_sandbox):
    sandbox = await started_sandbox(return_code=3).start()
    result = await sandbox.exec("echo hi")

    assert result.return_code == 3
    exec_span = next(span for span in recorded_spans() if span.name == "gym.sandbox.exec")
    assert exec_span.attributes["nemo.gym.sandbox.return_code"] == 3
    assert exec_span.attributes["nemo.gym.sandbox.provider"] == "stub"
    assert exec_span.attributes["nemo.gym.sandbox.id"] == "handle-1"


async def test_sandbox_stop_emits_a_span_for_the_release(recorded_spans, started_sandbox):
    """Start and exec were spans already; the release was invisible, so a trace could not
    pair a sandbox's start with its end."""
    sandbox = await started_sandbox().start()
    await sandbox.stop()

    stop_span = next(span for span in recorded_spans() if span.name == "gym.sandbox.stop")
    assert stop_span.attributes["nemo.gym.sandbox.provider"] == "stub"
    assert stop_span.attributes["nemo.gym.sandbox.id"] == "handle-1"


async def test_sandbox_spans_carry_the_rollout_id_inside_a_rollout(
    recorded_spans, started_sandbox, rollout_attributes_registered
):
    """The same correlation key the endpoint spans carry, so a rollout's sandbox time is filterable
    by attribute and not only through the parent chain. The sandbox API learns it through the
    provider Gym's telemetry setup registers, not by importing rollout correlation."""
    from nemo_gym.core import rollout_context

    with rollout_context("7-0"):
        sandbox = await started_sandbox().start()
        await sandbox.exec("echo hi")
        await sandbox.stop()

    by_name = {span.name: span for span in recorded_spans()}
    for name in ("gym.sandbox.start", "gym.sandbox.exec", "gym.sandbox.stop"):
        assert by_name[name].attributes["nemo.gym.rollout.id"] == "7-0", name


async def test_sandbox_spans_have_no_rollout_id_outside_a_rollout(
    recorded_spans, started_sandbox, rollout_attributes_registered
):
    sandbox = await started_sandbox().start()
    await sandbox.exec("echo hi")
    await sandbox.stop()
    names = set()
    for span in recorded_spans():
        if span.name.startswith("gym.sandbox."):
            names.add(span.name)
            assert "nemo.gym.rollout.id" not in span.attributes
    assert names == {"gym.sandbox.start", "gym.sandbox.exec", "gym.sandbox.stop"}


class _FailingProvider(_StubProvider):
    """A provider whose create or exec raises, so the span ends in error before any result exists."""

    def __init__(self, fail: str):
        super().__init__()
        self._fail = fail

    async def create(self, spec):
        if self._fail == "create":
            raise RuntimeError("boom")
        return await super().create(spec)

    async def exec(self, handle, command, **kwargs):
        if self._fail == "exec":
            raise RuntimeError("boom")
        return await super().exec(handle, command, **kwargs)


@pytest.mark.parametrize("fail", ["create", "exec"])
async def test_sandbox_spans_keep_the_rollout_id_when_the_provider_fails(
    recorded_spans, fail, rollout_attributes_registered
):
    """Correlation is attached before the awaited work, so a failed start or exec is still filterable."""
    from opentelemetry.trace import StatusCode

    from nemo_gym.core import rollout_context
    from nemo_gym.sandbox import AsyncSandbox, SandboxSpec

    sandbox = AsyncSandbox(_FailingProvider(fail), SandboxSpec(image="ubuntu"))
    with rollout_context("7-0"), pytest.raises(RuntimeError, match="boom"):
        if fail == "create":
            await sandbox.start()
        else:
            await (await sandbox.start()).exec("echo hi")

    name = "gym.sandbox.start" if fail == "create" else "gym.sandbox.exec"
    span = next(s for s in recorded_spans() if s.name == name)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["nemo.gym.rollout.id"] == "7-0"


async def test_sandbox_span_never_records_the_command(recorded_spans, started_sandbox):
    """In a code-execution environment the command is model output or task content.

    Attribute redaction works on key names, so a command stored under any key would be
    exported verbatim. The only safe handling is not to record it.
    """
    sandbox = await started_sandbox().start()
    await sandbox.exec('python -c "print(SECRET_FLAG)"')

    exec_span = next(span for span in recorded_spans() if span.name == "gym.sandbox.exec")
    for value in exec_span.attributes.values():
        assert "SECRET_FLAG" not in str(value)


async def test_sandbox_is_uninstrumented_when_the_group_is_disabled(recorded_spans, started_sandbox):
    from nemo.lens.state import set_enabled_span_groups

    set_enabled_span_groups(frozenset())
    sandbox = await started_sandbox().start()
    result = await sandbox.exec("echo hi")

    assert result.return_code == 0, "the sandbox must still work with telemetry off"
    assert recorded_spans() == ()


# --------------------------------------------------------------------------- #
# Server wiring
# --------------------------------------------------------------------------- #


def test_server_type_is_derived_from_the_class_hierarchy():
    """Used for the `nemo.gym.server.type` resource attribute.

    Matched against the MRO by name rather than by import, because the base server
    modules import `server_utils` and importing them back would be circular.
    """
    from nemo_gym.agents import SimpleResponsesAPIAgent
    from nemo_gym.models import SimpleResponsesAPIModel
    from nemo_gym.server_runtime import SimpleResourcesServer
    from nemo_gym.server_runtime.server_utils import _telemetry_server_type

    assert _telemetry_server_type(SimpleResourcesServer) == "resources_servers"
    assert _telemetry_server_type(SimpleResponsesAPIAgent) == "responses_api_agents"
    assert _telemetry_server_type(SimpleResponsesAPIModel) == "responses_api_models"


def test_server_type_survives_subclassing():
    """Every real Gym server is a subclass, so the lookup has to walk the MRO."""
    from nemo_gym.server_runtime import SimpleResourcesServer
    from nemo_gym.server_runtime.server_utils import _telemetry_server_type

    class MyWeatherServer(SimpleResourcesServer):
        pass

    assert _telemetry_server_type(MyWeatherServer) == "resources_servers"


def test_server_type_is_none_for_an_unrelated_class():
    from nemo_gym.server_runtime.server_utils import _telemetry_server_type

    assert _telemetry_server_type(dict) is None


def test_url_query_string_is_stripped_before_it_becomes_an_attribute():
    """Gym URLs carry API keys, and rollout-prefixed routes carry task identity."""
    from nemo_gym.client.http_client import _redacted_url

    assert _redacted_url("http://h:1/v1/responses?api_key=sk-abc") == "http://h:1/v1/responses"
    assert _redacted_url("http://h:1/v1/responses") == "http://h:1/v1/responses"


# --------------------------------------------------------------------------- #
# CLIENT-kind spans
# --------------------------------------------------------------------------- #


def test_client_span_is_client_kind(recorded_spans):
    """nemo-lens's managed_span cannot set SpanKind, so Gym creates this one itself.

    An INTERNAL span on a cross-service hop means a backend cannot draw the edge between
    the agent server and the model server.
    """
    from nemo_gym.telemetry.spans import client_span

    with client_span("HTTP POST"):
        pass

    span = recorded_spans()[0]
    assert span.kind.name == "CLIENT"
    assert span.name == "HTTP POST"


def test_client_span_records_and_reraises_exceptions(recorded_spans):
    from nemo_gym.telemetry.spans import client_span

    with pytest.raises(ValueError, match="connection reset"):
        with client_span("HTTP POST"):
            raise ValueError("connection reset")

    span = recorded_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert span.end_time is not None, "the span must still be ended when the body raises"


def test_client_span_sets_attributes(recorded_spans):
    from nemo_gym.telemetry.spans import client_span

    with client_span("HTTP POST", **{"http.request.method": "POST"}):
        pass

    assert recorded_spans()[0].attributes["http.request.method"] == "POST"


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


def test_supervisor_is_the_only_writer_of_the_active_servers_gauge():
    """`gym.servers.active` is a gauge; more than one writer makes it meaningless.

    Pins that the call lives in the CLI and nowhere else, so a later change that starts
    reporting it from a server process has to be deliberate.
    """
    import subprocess

    from nemo_gym import __file__ as pkg_file

    package_dir = pkg_file.rsplit("/", 1)[0]
    hits = (
        subprocess.run(
            ["grep", "-rn", "record_active_servers", package_dir],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )

    callers = {line.split(":")[0].rsplit("/", 1)[-1] for line in hits}
    assert callers == {"env.py", "metrics.py"}, (
        f"gym.servers.active must only be written by the CLI supervisor, found: {sorted(callers)}"
    )


# --------------------------------------------------------------------------- #
# Reaching the per-server venvs
# --------------------------------------------------------------------------- #


def test_server_requirements_contain_no_spaces():
    """`head_server_deps` is interpolated into a shell command line **unquoted**.

    `setup_env_command` builds `uv pip install ... {" ".join(head_server_deps)}`, so a
    requirement containing spaces is split into separate arguments by the shell. The
    spaced PEP 508 form `nemo-lens[sdk] @ git+https://...` is therefore unusable here.

    Quoting at that call site would be the better fix and is tracked separately; this side
    emits the equally valid space-free form.
    """
    pytest.importorskip("nemo.lens")

    import os

    os.environ["NEMO_GYM_OTEL_ENABLED"] = "1"
    try:
        from nemo_gym.telemetry.setup import server_venv_requirements

        requirements = server_venv_requirements()
        assert requirements, "expected telemetry requirements with lens installed"
        for requirement in requirements:
            assert " " not in requirement, (
                f"{requirement!r} contains a space and would be split by the shell in setup_env_command"
            )
    finally:
        os.environ.pop("NEMO_GYM_OTEL_ENABLED", None)


def test_server_requirements_survive_shell_splitting():
    """Assert against a real shell parse rather than eyeballing the string."""
    pytest.importorskip("nemo.lens")

    import os
    import shlex

    os.environ["NEMO_GYM_OTEL_ENABLED"] = "1"
    try:
        from nemo_gym.telemetry.setup import server_venv_requirements

        requirements = server_venv_requirements()
        command_line = "uv pip install -r requirements.txt " + " ".join(requirements)
        parsed = shlex.split(command_line)
        for requirement in requirements:
            assert requirement in parsed, f"{requirement!r} did not survive shell splitting"
    finally:
        os.environ.pop("NEMO_GYM_OTEL_ENABLED", None)


# --------------------------------------------------------------------------- #
# The nemo.lens import boundary
# --------------------------------------------------------------------------- #


def _modules_importing_lens():
    """Return `{module path -> [import lines]}` for every direct nemo.lens import."""
    import ast
    import pathlib

    import nemo_gym

    root = pathlib.Path(nemo_gym.__file__).parent
    found = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - would fail the linter first
            continue
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nemo.lens"):
                hits.append(f"from {node.module} import ...")
            elif isinstance(node, ast.Import):
                hits += [f"import {a.name}" for a in node.names if a.name.startswith("nemo.lens")]
        if hits:
            found[str(path.relative_to(root))] = hits
    return found


def test_nemo_lens_is_imported_only_inside_the_telemetry_package():
    """`nemo_gym.telemetry` is the only package allowed to import nemo-lens.

    nemo-lens is an optional extra, so every `nemo.lens` import has to be reachable only
    when it is installed. Confining them to one package makes that checkable by looking at
    one directory instead of auditing each import in context, and keeps the rest of
    `nemo_gym` importable with the extra absent.

    Call sites use `nemo_gym.telemetry._fallbacks` for instrumentation primitives and
    `nemo_gym.telemetry.contrib` for propagation and framework instrumentation.
    """
    offenders = {
        module: hits for module, hits in _modules_importing_lens().items() if not module.startswith("telemetry/")
    }
    assert not offenders, (
        "nemo.lens must only be imported inside nemo_gym/telemetry/. Import the telemetry "
        f"package instead. Offending modules: {offenders}"
    )


def test_the_boundary_check_actually_finds_lens_imports():
    """Guard the guard: a scan that silently finds nothing would assert nothing."""
    found = _modules_importing_lens()
    assert found, "the scanner found no nemo.lens imports at all, so the check above is vacuous"
    assert any(module.startswith("telemetry/") for module in found)


def test_lens_imports_are_function_local_outside_the_fallback_resolver():
    """Only `_fallbacks` and `span_groups` may import nemo-lens at module scope.

    Those two resolve at import time on purpose — one to bind the primitives, one to pick
    the SpanGroup base class — and both are guarded by `try/except ImportError`. Anywhere
    else, a module-scope import would make `import nemo_gym.telemetry` fail outright when
    the extra is absent.
    """
    import ast
    import pathlib

    import nemo_gym

    root = pathlib.Path(nemo_gym.__file__).parent / "telemetry"
    allowed_at_module_scope = {"_fallbacks.py", "span_groups.py"}
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:  # module scope only, not ast.walk
            targets = []
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nemo.lens"):
                targets.append(node.module)
            elif isinstance(node, ast.Try):
                for inner in node.body:
                    if isinstance(inner, ast.ImportFrom) and (inner.module or "").startswith("nemo.lens"):
                        targets.append(inner.module)
            if targets and path.name not in allowed_at_module_scope:
                offenders.append((path.name, targets))
    assert not offenders, f"module-scope nemo.lens imports outside the resolvers: {offenders}"


async def test_sandbox_spans_have_no_extra_attributes_without_a_provider(recorded_spans, started_sandbox):
    """The sandbox package on its own knows nothing about rollouts."""
    from nemo_gym.core import rollout_context

    with rollout_context("7-0"):
        sandbox = await started_sandbox().start()
        await sandbox.exec("echo hi")
    for span in recorded_spans():
        if span.name.startswith("gym.sandbox."):
            assert "nemo.gym.rollout.id" not in span.attributes


async def test_explicit_span_attributes_win_over_the_provider(
    recorded_spans, started_sandbox, rollout_attributes_registered
):
    from nemo_gym.core import rollout_context

    with rollout_context("7-0"):
        sandbox = await started_sandbox().start(span_attributes={"nemo.gym.rollout.id": "explicit", "extra": "x"})
        await sandbox.exec("echo hi", span_attributes={"nemo.gym.rollout.id": "explicit-exec"})

    by_name = {span.name: span for span in recorded_spans()}
    assert by_name["gym.sandbox.start"].attributes["nemo.gym.rollout.id"] == "explicit"
    assert by_name["gym.sandbox.start"].attributes["extra"] == "x"
    assert by_name["gym.sandbox.exec"].attributes["nemo.gym.rollout.id"] == "explicit-exec"


# --------------------------------------------------------------------------- #
# Driver dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"_ng_task_index": 3, "_ng_rollout_index": 0}, "3-0"),
        ({"_ng_task_index": 3, "_ng_rollout_index": 0, "_ng_attempt_index": 2}, "3-0-a2"),
        ({"_ng_rollout_id": "custom", "_ng_attempt_index": 2}, "custom-a2"),
        ({"task_id": {"taskset": "t"}, "_ng_task_index": 5, "_ng_rollout_index": 1}, "5-1"),
    ],
)
def test_dispatch_span_takes_its_identity_from_the_collector_row(recorded_spans, row, expected):
    """A materialized episode body carries the identity only as `episode_id`; the span reads the row."""
    from nemo_gym.rollout_collection import _dispatch_span

    with _dispatch_span(row, "alpha", server_name="alpha_environment_server"):
        pass

    (span,) = [s for s in recorded_spans() if s.name == "gym.rollout.dispatch"]
    assert span.attributes["nemo.gym.rollout.id"] == expected
    assert span.attributes["nemo.gym.rollout.dispatch.name"] == "alpha"
    assert span.attributes["nemo.gym.http.destination.server.name"] == "alpha_environment_server"


def test_dispatch_span_omits_the_id_for_a_row_without_identity(recorded_spans):
    from nemo_gym.rollout_collection import _dispatch_span

    with _dispatch_span({"input": "x"}, "agent", server_name="agent_environment_server"):
        pass

    (span,) = [s for s in recorded_spans() if s.name == "gym.rollout.dispatch"]
    assert "nemo.gym.rollout.id" not in span.attributes


def test_dispatch_span_survives_a_malformed_attempt_index(recorded_spans):
    """`int()` on a list raises TypeError; telemetry must not abort the rollout for it."""
    from nemo_gym.rollout_collection import _dispatch_span

    with _dispatch_span(
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_attempt_index": [1]}, "agent", server_name="s"
    ):
        pass

    (span,) = [s for s in recorded_spans() if s.name == "gym.rollout.dispatch"]
    assert "nemo.gym.rollout.id" not in span.attributes


@pytest.mark.parametrize(
    ("row", "task_id"),
    [
        ({"task_id": "terminal-bench/foodstuff", "task_name": "other"}, "terminal-bench/foodstuff"),
        ({"task_id": {"taskset": "swe_pro", "task_id": "instance"}}, "instance"),
        ({"instance_id": "django__django-1"}, "django__django-1"),
        ({"problem_id": 17}, "17"),
        ({"task_name": "terminal-bench/background-server"}, "terminal-bench/background-server"),
        ({"task_id": {"taskset": "swe_pro"}}, None),
        ({"input": "x"}, None),
    ],
)
def test_dispatch_span_names_the_task_and_the_repeat(recorded_spans, row, task_id):
    from nemo_gym.rollout_collection import _dispatch_span

    with _dispatch_span({**row, "_ng_task_index": 4, "_ng_rollout_index": 2}, "alpha", server_name="s") as context:
        pass

    (span,) = [s for s in recorded_spans() if s.name == "gym.rollout.dispatch"]
    assert context == span.context
    assert span.attributes.get("nemo.gym.task.id") == task_id
    assert span.attributes["nemo.gym.rollout.repeat"] == 2


def test_dispatch_span_omits_a_repeat_that_is_not_an_index(recorded_spans):
    from nemo_gym.rollout_collection import _dispatch_span

    with _dispatch_span({"_ng_rollout_index": True}, "alpha", server_name="s"):
        pass

    (span,) = [s for s in recorded_spans() if s.name == "gym.rollout.dispatch"]
    assert "nemo.gym.rollout.repeat" not in span.attributes


# --------------------------------------------------------------------------- #
# Driver wiring: run_from_config end to end against a fake /run
# --------------------------------------------------------------------------- #


def _write_rollout_input(tmp_path, n: int):
    import json as _json

    from nemo_gym.config import AGENT_REF_KEY_NAME, TASK_SOURCE_KEY_NAME

    path = tmp_path / "input.jsonl"
    path.write_text(
        "".join(
            _json.dumps(
                {
                    AGENT_REF_KEY_NAME: {"name": "alpha"},
                    TASK_SOURCE_KEY_NAME: "alpha",
                    "responses_create_params": {"input": []},
                    "source_index": i,
                }
            )
            + "\n"
            for i in range(n)
        )
    )
    return path


def _collection_config(tmp_path, input_path):
    from nemo_gym.rollout_collection import RolloutCollectionConfig

    return RolloutCollectionConfig(
        input_jsonl_fpath=str(input_path),
        output_jsonl_fpath=str(tmp_path / "output.jsonl"),
        num_samples_in_parallel=1,
        disable_aggregation=True,
        disable_health_check=True,
    )


def _install_recording_post(monkeypatch):
    """A fake `/run` that records the current span and the `traceparent` the client would inject."""
    from unittest.mock import AsyncMock

    from opentelemetry import trace as otel_trace

    from nemo_gym.telemetry.contrib import inject_trace_context
    from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client

    sent = []

    async def post(server_name, url_path, json):
        headers: dict = {}
        inject_trace_context(headers)
        sent.append(
            {"span": otel_trace.get_current_span().get_span_context(), "traceparent": headers.get("traceparent")}
        )
        return FakeResponse(200, {"response": {}})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post), agent_names=("alpha",))
    return sent


@pytest.fixture
def driver_telemetry(monkeypatch):
    """The driver as a standalone eval run sees it: empty global config, trace exporter active."""
    import nemo_gym.rollout_collection as rollout_collection

    monkeypatch.setattr(rollout_collection, "get_global_config_dict", lambda *a, **k: {})
    monkeypatch.setattr(rollout_collection, "is_trace_exporter_active", lambda: True)
    return rollout_collection


async def test_standalone_run_gives_each_rollout_its_own_trace_linked_to_the_job(
    tmp_path, monkeypatch, recorded_spans, driver_telemetry
):
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    sent = _install_recording_post(monkeypatch)
    await RolloutCollectionHelper().run_from_config(_collection_config(tmp_path, _write_rollout_input(tmp_path, 3)))

    spans = recorded_spans()
    (job,) = [s for s in spans if s.name == "gym.job"]
    dispatches = [s for s in spans if s.name == "gym.rollout.dispatch"]
    assert len(dispatches) == 3
    assert job.parent is None
    trace_ids = {d.context.trace_id for d in dispatches}
    assert len(trace_ids) == 3 and job.context.trace_id not in trace_ids
    for dispatch in dispatches:
        assert dispatch.parent is None
        (link,) = dispatch.links
        assert (link.context.trace_id, link.context.span_id) == (job.context.trace_id, job.context.span_id)
        assert dispatch.attributes["nemo.gym.rollout.dispatch.name"] == "alpha"
        assert dispatch.attributes["nemo.gym.http.destination.server.name"] == "alpha_environment_server"
    # Each /run is sent under its rollout's dispatch span, so the agent joins that trace.
    by_span_id = {d.context.span_id: d for d in dispatches}
    assert len(sent) == 3
    for request in sent:
        dispatch = by_span_id[request["span"].span_id]
        assert request["traceparent"].split("-")[1] == f"{dispatch.context.trace_id:032x}"
    assert driver_telemetry._OWNED_JOB_SPAN_CONTEXT.get() is None


async def test_embedded_run_keeps_nesting_under_the_callers_span(
    tmp_path, monkeypatch, recorded_spans, driver_telemetry
):
    from opentelemetry import trace as otel_trace

    from nemo_gym.rollout_collection import RolloutCollectionHelper

    sent = _install_recording_post(monkeypatch)
    tracer = otel_trace.get_tracer("embedder")
    with tracer.start_as_current_span("embedder.step") as step:
        await RolloutCollectionHelper().run_from_config(
            _collection_config(tmp_path, _write_rollout_input(tmp_path, 2))
        )

    spans = recorded_spans()
    (job,) = [s for s in spans if s.name == "gym.job"]
    assert job.parent.span_id == step.get_span_context().span_id
    dispatches = [s for s in spans if s.name == "gym.rollout.dispatch"]
    assert len(dispatches) == 2
    for dispatch in dispatches:
        assert dispatch.context.trace_id == step.get_span_context().trace_id
        assert dispatch.parent.span_id == job.context.span_id
        assert dispatch.links == ()
    for request in sent:
        assert request["traceparent"].split("-")[1] == f"{step.get_span_context().trace_id:032x}"
    assert driver_telemetry._OWNED_JOB_SPAN_CONTEXT.get() is None


async def test_dispatch_span_is_skipped_while_traces_are_off(tmp_path, monkeypatch, recorded_spans, driver_telemetry):
    """Metrics-only processes keep span groups enabled; the dispatch must not start a span for them."""
    import nemo_gym.rollout_collection as rollout_collection
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    monkeypatch.setattr(rollout_collection, "is_trace_exporter_active", lambda: False)

    def boom(*args, **kwargs):
        raise AssertionError("_dispatch_span must not run while traces are off")

    monkeypatch.setattr(rollout_collection, "_dispatch_span", boom)
    sent = _install_recording_post(monkeypatch)
    await RolloutCollectionHelper().run_from_config(_collection_config(tmp_path, _write_rollout_input(tmp_path, 2)))
    assert len(sent) == 2


@pytest.mark.asyncio
async def test_materialized_dispatch_takes_its_id_from_the_row_not_the_episode_body(
    monkeypatch, recorded_spans, driver_telemetry
):
    """The posted episode body carries the identity only as `episode_id`; the dispatch span still gets the id.

    Pins the call site, not the helper: `_dispatch_span` must receive the collector row, because
    `maybe_rollout_id_from_run_body` cannot read an id out of the posted body of a materialized row.
    """
    import json as _json
    from unittest.mock import AsyncMock

    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from tests.unit_tests import test_rollout_collection as collection_tests

    post = AsyncMock(return_value=collection_tests.FakeResponse(200, {"reward": 1.0}))
    client = collection_tests.install_fake_server_client(monkeypatch, post)
    client.global_config_dict = collection_tests.TestEnvironmentServerRouting._mixed_batch_config()
    row = collection_tests.TestEnvironmentServerRouting._materialized_row()
    config = RolloutCollectionConfig(
        input_jsonl_fpath="input.jsonl",
        output_jsonl_fpath="output.jsonl",
        environment_server_routes={"swe_pro": "environment"},
        num_repeats=1,
    )
    rows = RolloutCollectionHelper._preprocess_raw_rows([(0, _json.dumps(row), row)], config)

    for future in RolloutCollectionHelper().run_examples(rows):
        await future

    (dispatch,) = [s for s in recorded_spans() if s.name == "gym.rollout.dispatch"]
    assert dispatch.attributes["nemo.gym.rollout.id"] == "0-0"
    assert "episode_id" in post.call_args.kwargs["json"]


# --------------------------------------------------------------------------- #
# Rollout health span
# --------------------------------------------------------------------------- #


async def test_each_traced_rollout_gets_a_health_span_under_its_dispatch(
    tmp_path, monkeypatch, recorded_spans, driver_telemetry
):
    """The dispatch span has ended by the time health is evaluated; the health span still joins it."""
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    monkeypatch.setattr(driver_telemetry, "is_metrics_exporter_active", lambda: False)
    _install_recording_post(monkeypatch)
    await RolloutCollectionHelper().run_from_config(_collection_config(tmp_path, _write_rollout_input(tmp_path, 2)))

    spans = recorded_spans()
    dispatches = {s.context.span_id: s for s in spans if s.name == "gym.rollout.dispatch"}
    healths = [s for s in spans if s.name == "gym.rollout.health"]
    assert len(dispatches) == len(healths) == 2
    for health in healths:
        dispatch = dispatches[health.parent.span_id]
        assert health.context.trace_id == dispatch.context.trace_id
        assert health.attributes["nemo.gym.rollout.id"] == dispatch.attributes["nemo.gym.rollout.id"]
        assert health.attributes["nemo.gym.health.verdict"] in {"healthy", "unhealthy", "unobserved"}


async def test_no_health_span_while_traces_are_off(tmp_path, monkeypatch, recorded_spans, driver_telemetry):
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    monkeypatch.setattr(driver_telemetry, "is_trace_exporter_active", lambda: False)
    _install_recording_post(monkeypatch)
    await RolloutCollectionHelper().run_from_config(_collection_config(tmp_path, _write_rollout_input(tmp_path, 2)))

    assert [s for s in recorded_spans() if s.name == "gym.rollout.health"] == []


def _stub_health_digest(monkeypatch, verdict, checks):
    from types import SimpleNamespace

    import nemo_gym.rollout_health as rollout_health

    digest = SimpleNamespace(verdict=verdict, findings=[SimpleNamespace(check=check) for check in checks])
    monkeypatch.setattr(rollout_health, "evaluate_record", lambda result, ignored: digest)


def test_health_span_carries_the_verdict_and_the_distinct_flagged_checks(recorded_spans, monkeypatch):
    from opentelemetry import trace as otel_trace

    from nemo_gym.rollout_collection import _record_rollout_health

    _stub_health_digest(monkeypatch, "unhealthy", ["truncated", "empty_answer", "truncated"])
    with otel_trace.get_tracer("test").start_as_current_span("gym.rollout.dispatch") as dispatch:
        parent = dispatch.get_span_context()

    _record_rollout_health({"_ng_task_index": 1, "_ng_rollout_index": 0}, "scored", frozenset(), parent)

    (health,) = [s for s in recorded_spans() if s.name == "gym.rollout.health"]
    assert health.parent.span_id == parent.span_id
    assert health.attributes["nemo.gym.health.verdict"] == "unhealthy"
    assert tuple(health.attributes["nemo.gym.health.checks"]) == ("empty_answer", "truncated")
    assert health.attributes["nemo.gym.rollout.id"] == "1-0"


def test_a_healthy_rollout_has_no_checks_attribute(recorded_spans, monkeypatch):
    from opentelemetry import trace as otel_trace

    from nemo_gym.rollout_collection import _record_rollout_health

    _stub_health_digest(monkeypatch, "healthy", [])
    with otel_trace.get_tracer("test").start_as_current_span("gym.rollout.dispatch") as dispatch:
        parent = dispatch.get_span_context()

    _record_rollout_health({}, "scored", frozenset(), parent)

    (health,) = [s for s in recorded_spans() if s.name == "gym.rollout.health"]
    assert health.attributes["nemo.gym.health.verdict"] == "healthy"
    assert "nemo.gym.health.checks" not in health.attributes
    assert "nemo.gym.rollout.id" not in health.attributes


def test_health_metrics_are_recorded_without_a_span(recorded_spans, monkeypatch):
    import nemo_gym.rollout_collection as rollout_collection

    recorded = []
    _stub_health_digest(monkeypatch, "unhealthy", ["truncated"])
    monkeypatch.setattr(
        rollout_collection,
        "record_rollout_health",
        lambda verdict, *, outcome, flagged_checks: recorded.append((verdict, outcome, list(flagged_checks))),
    )

    rollout_collection._record_rollout_health({}, "failed", frozenset())

    assert recorded == [("unhealthy", "failed", ["truncated"])]
    assert [s for s in recorded_spans() if s.name == "gym.rollout.health"] == []
