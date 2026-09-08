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
"""Instrumentation attached to Gym's own code: sandbox, servers, CLI orchestrator."""

import pytest

from nemo_gym.telemetry.span_groups import GymSpanGroup
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

    monkeypatch.setattr(lens_helpers.trace, "get_tracer", lambda *a, **k: tracer)
    monkeypatch.setattr("opentelemetry.trace.get_tracer", lambda *a, **k: tracer)
    set_enabled_span_groups(GymSpanGroup.resolve("all"))
    yield exporter.get_finished_spans
    set_enabled_span_groups(frozenset())


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
        return "handle-1"

    async def exec(self, handle, command, **kwargs):
        from nemo_gym.sandbox.providers.base import SandboxExecResult

        return SandboxExecResult(stdout="out", stderr="", return_code=self._return_code)

    async def close(self, handle):
        self.closed = True

    async def aclose(self):
        pass


@pytest.fixture
def started_sandbox():
    from nemo_gym.sandbox.api import AsyncSandbox
    from nemo_gym.sandbox.providers.base import SandboxSpec

    def build(return_code: int = 0):
        return AsyncSandbox(_StubProvider(return_code), SandboxSpec(image="ubuntu"))

    return build


async def test_sandbox_start_emits_a_span(recorded_spans, started_sandbox):
    await started_sandbox().start()
    spans = recorded_spans()
    assert "gym.sandbox.start" in [span.name for span in spans]
    start_span = next(span for span in spans if span.name == "gym.sandbox.start")
    assert start_span.attributes["nemo.gym.sandbox.provider"] == "stub"


async def test_sandbox_exec_emits_a_span_with_the_exit_code(recorded_spans, started_sandbox):
    sandbox = await started_sandbox(return_code=3).start()
    result = await sandbox.exec("echo hi")

    assert result.return_code == 3
    exec_span = next(span for span in recorded_spans() if span.name == "gym.sandbox.exec")
    assert exec_span.attributes["nemo.gym.sandbox.return_code"] == 3
    assert exec_span.attributes["nemo.gym.sandbox.provider"] == "stub"


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
    from nemo_gym.base_resources_server import SimpleResourcesServer
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
    from nemo_gym.base_responses_api_model import SimpleResponsesAPIModel
    from nemo_gym.server_utils import _telemetry_server_type

    assert _telemetry_server_type(SimpleResourcesServer) == "resources_servers"
    assert _telemetry_server_type(SimpleResponsesAPIAgent) == "responses_api_agents"
    assert _telemetry_server_type(SimpleResponsesAPIModel) == "responses_api_models"


def test_server_type_survives_subclassing():
    """Every real Gym server is a subclass, so the lookup has to walk the MRO."""
    from nemo_gym.base_resources_server import SimpleResourcesServer
    from nemo_gym.server_utils import _telemetry_server_type

    class MyWeatherServer(SimpleResourcesServer):
        pass

    assert _telemetry_server_type(MyWeatherServer) == "resources_servers"


def test_server_type_is_none_for_an_unrelated_class():
    from nemo_gym.server_utils import _telemetry_server_type

    assert _telemetry_server_type(dict) is None


def test_url_query_string_is_stripped_before_it_becomes_an_attribute():
    """Gym URLs carry API keys, and rollout-prefixed routes carry task identity."""
    from nemo_gym.server_utils import _redacted_url

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
# Orchestrator
# --------------------------------------------------------------------------- #


def test_orchestrator_is_the_only_writer_of_the_active_servers_gauge():
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
        f"gym.servers.active must only be written by the CLI orchestrator, found: {sorted(callers)}"
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
