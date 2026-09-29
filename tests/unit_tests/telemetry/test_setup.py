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
"""Per-process telemetry lifecycle.

`kb/knowledge/conventions/shared-global-state.md` asks three questions of anything
touching a process-global: what happens on the second call, in a forked child, and on a
non-exporting rank. Each has a test here.
"""

import os
import sys

import pytest

from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry.setup import (
    get_telemetry,
    init_telemetry,
    is_telemetry_env_enabled,
    shutdown_telemetry,
)
from tests.unit_tests.telemetry.conftest import import_without_lens, no_lens, requires_lens


#: These exercise the telemetry-enabled path, which needs nemo-lens. The absent-lens path
#: is covered by test_fallbacks.py, which runs either way.
pytestmark = requires_lens


@pytest.fixture
def enabled_console_env(clean_otel_env):
    """Telemetry on, exporting to console — no network, no collector."""
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORTER", "console")
    clean_otel_env.setenv("NEMO_GYM_OTEL_SPAN_GROUPS", "all")
    return clean_otel_env


# --------------------------------------------------------------------------- #
# The off paths
# --------------------------------------------------------------------------- #


def test_init_returns_none_when_disabled(clean_otel_env):
    assert init_telemetry(server_name="x") is None
    assert get_telemetry() is None


def test_init_does_not_import_lens_when_disabled(clean_otel_env, monkeypatch):
    """A run with telemetry off must not pay for importing nemo-lens.

    The gate is `is_telemetry_env_enabled()`, which reads two env vars — so a disabled
    process never touches the SDK at all.
    """
    monkeypatch.delitem(sys.modules, "nemo.lens", raising=False)
    calls = []
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def tracking_import(name, *args, **kwargs):
        if name.startswith("nemo.lens"):
            calls.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)
    assert init_telemetry(server_name="x") is None
    assert calls == [], f"disabled telemetry imported nemo-lens: {calls}"


def test_init_returns_none_when_lens_is_absent(clean_otel_env):
    """Enabled + nemo-lens not installed must degrade, not explode.

    `init_telemetry` imports lens lazily, so the block has to stay in force across the
    call — importing the module under a block proves nothing on its own.
    """
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    assert is_telemetry_env_enabled() is True
    with no_lens():
        assert init_telemetry(server_name="x") is None
    assert get_telemetry() is None


def test_invalid_env_disables_rather_than_crashing_the_server(clean_otel_env):
    """A malformed telemetry env var must not take a Gym server process down."""
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORT_RANK", "not-an-int")
    assert init_telemetry(server_name="x") is None


def test_setup_failure_is_swallowed(clean_otel_env, monkeypatch):
    """If nemo-lens itself raises, Gym keeps serving."""
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")

    def boom(*args, **kwargs):
        raise RuntimeError("exporter unreachable")

    monkeypatch.setattr("nemo.lens.setup_telemetry", boom)
    assert init_telemetry(server_name="x") is None


# --------------------------------------------------------------------------- #
# The on path
# --------------------------------------------------------------------------- #


def test_init_builds_an_exporting_handle(enabled_console_env):
    handle = init_telemetry(server_name="weather", server_type="resources_servers")
    assert handle is not None
    assert handle.is_exporting is True
    assert get_telemetry() is handle


def test_enabled_span_groups_follow_the_config(enabled_console_env):
    from nemo_gym.telemetry._fallbacks import is_span_group_enabled

    enabled_console_env.setenv("NEMO_GYM_OTEL_SPAN_GROUPS", "default")
    init_telemetry(server_name="weather")

    assert is_span_group_enabled("server") is True
    assert is_span_group_enabled("rollout") is True
    assert is_span_group_enabled("sandbox") is False, "`default` must not enable fine-grained groups"


def test_service_name_is_suffixed_per_server(enabled_console_env, monkeypatch):
    """Three Gym processes sharing one service.name make a backend's service map useless."""
    captured = {}

    def capture(config, **kwargs):
        captured["service_name"] = config.service_name
        captured["attrs"] = kwargs.get("resource_attributes")
        raise RuntimeError("stop here — we only need the config")

    monkeypatch.setattr("nemo.lens.setup_telemetry", capture)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "nemo-gym")

    init_telemetry(server_name="policy_model", server_type="responses_api_models")

    assert captured["service_name"] == "nemo-gym/policy_model"
    assert captured["attrs"]["nemo.gym.service_group"] == "nemo-gym", (
        "the unsuffixed name must survive as a resource attribute so the fleet stays groupable"
    )
    assert captured["attrs"]["nemo.gym.server.name"] == "policy_model"
    assert captured["attrs"]["nemo.gym.server.type"] == "responses_api_models"


def test_service_name_suffixing_can_be_disabled(enabled_console_env, monkeypatch):
    captured = {}

    def capture(config, **kwargs):
        captured["service_name"] = config.service_name
        raise RuntimeError("stop here")

    monkeypatch.setattr("nemo.lens.setup_telemetry", capture)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "nemo-gym")
    monkeypatch.setenv("NEMO_GYM_OTEL_SERVICE_NAME_PER_SERVER", "0")

    init_telemetry(server_name="policy_model")
    assert captured["service_name"] == "nemo-gym"


def test_caller_resource_attributes_are_merged(enabled_console_env, monkeypatch):
    captured = {}

    def capture(config, **kwargs):
        captured.update(kwargs.get("resource_attributes") or {})
        raise RuntimeError("stop here")

    monkeypatch.setattr("nemo.lens.setup_telemetry", capture)
    init_telemetry(server_name="weather", resource_attributes={"custom.attr": "value"})

    from nemo_gym.package_info import __version__

    assert captured["custom.attr"] == "value"
    assert captured["nemo.gym.version"] == __version__


# --------------------------------------------------------------------------- #
# The three shared-global-state questions
# --------------------------------------------------------------------------- #


def test_init_is_idempotent(enabled_console_env):
    """Second call: returns the same handle rather than rebuilding providers.

    nemo-lens raises RuntimeError on a second `setup_telemetry`, so without this guard a
    server whose entrypoint is imported twice would crash on startup.
    """
    first = init_telemetry(server_name="weather")
    second = init_telemetry(server_name="weather")
    assert first is second is not None


def test_init_is_idempotent_even_when_disabled(clean_otel_env, monkeypatch):
    """A disabled process must not redo the (cheap, but repeated) setup on every call."""
    assert init_telemetry(server_name="x") is None
    monkeypatch.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    assert init_telemetry(server_name="x") is None, "the first call already settled this process"


def test_concurrent_init_calls_produce_exactly_one_setup(enabled_console_env, monkeypatch):
    """Prove `_INIT_LOCK` actually guards something.

    Without it, two threads can both observe `_INITIALISED is False` before either sets
    it and both call `setup_telemetry`, which nemo-lens raises on for a second call in
    the same process. Wrap `setup_telemetry` to count calls and race many threads through
    `init_telemetry` at once.
    """
    import threading

    from nemo.lens import setup_telemetry as real_setup_telemetry

    call_count = 0
    count_lock = threading.Lock()

    def counting_setup_telemetry(*args, **kwargs):
        nonlocal call_count
        with count_lock:
            call_count += 1
        return real_setup_telemetry(*args, **kwargs)

    monkeypatch.setattr("nemo.lens.setup_telemetry", counting_setup_telemetry)

    barrier = threading.Barrier(16)
    results = [None] * 16

    def worker(index):
        barrier.wait()
        results[index] = init_telemetry(server_name="weather")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count == 1, f"setup_telemetry must run exactly once under concurrent init, ran {call_count} times"
    assert all(r is not None for r in results)
    assert len({id(r) for r in results}) == 1, "every thread must observe the same handle"


def test_non_exporting_rank_gets_a_silent_handle(clean_otel_env):
    """Rank 1..N under `single_rank`: no-op providers and no enabled span groups."""
    from nemo_gym.telemetry._fallbacks import is_span_group_enabled

    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORTER", "console")
    clean_otel_env.setenv("NEMO_GYM_OTEL_SPAN_GROUPS", "all")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORT_STRATEGY", "single_rank")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORT_RANK", "0")

    handle = init_telemetry(server_name="weather", rank=3, world_size=4)
    assert handle is not None
    assert handle.is_exporting is False
    assert is_span_group_enabled("server") is False


def test_all_ranks_strategy_exports_from_every_process(clean_otel_env):
    """The Gym default: no server process is silenced, so traces have no holes."""
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORTER", "console")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORT_STRATEGY", "all_ranks")

    handle = init_telemetry(server_name="weather", rank=3, world_size=4)
    assert handle.is_exporting is True


def test_forked_child_inherits_working_telemetry(enabled_console_env):
    """Forked child: must not deadlock, double-init, or crash.

    Gym spawns servers with Popen (exec, not fork), but Ray and uvicorn's multi-worker
    supervisor can fork. nemo-lens registers at-fork hooks for its id generator and open
    span tracker; this asserts the composite actually survives a fork, which no
    single-process test can show.
    """
    init_telemetry(server_name="weather")

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the forked child
        status = 1
        try:
            os.close(read_fd)
            handle = init_telemetry(server_name="weather")
            with telemetry_setup.get_telemetry().tracer.start_as_current_span("child"):
                pass
            status = 0 if handle is not None else 2
            os.write(write_fd, b"ok" if status == 0 else b"no")
        finally:
            os._exit(status)

    os.close(write_fd)
    try:
        _, exit_status = os.waitpid(pid, 0)
        message = os.read(read_fd, 8)
    finally:
        os.close(read_fd)

    assert os.WIFEXITED(exit_status) and os.WEXITSTATUS(exit_status) == 0, "the forked child did not exit cleanly"
    assert message == b"ok"


# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #


def test_shutdown_without_init_is_a_no_op(clean_otel_env):
    shutdown_telemetry()


def test_shutdown_is_idempotent(enabled_console_env):
    init_telemetry(server_name="weather")
    shutdown_telemetry()
    shutdown_telemetry()


def test_shutdown_swallows_provider_errors(enabled_console_env, monkeypatch):
    """A failing flush at exit must not change Gym's exit code."""
    handle = init_telemetry(server_name="weather")

    def boom(*args, **kwargs):
        raise RuntimeError("flush failed")

    monkeypatch.setattr(handle, "shutdown", boom)
    shutdown_telemetry()


def test_is_telemetry_env_enabled_does_not_require_lens(clean_otel_env):
    clean_otel_env.setenv("NEMO_LENS_ENABLED", "1")
    module = import_without_lens("nemo_gym.telemetry.setup")
    assert module.is_telemetry_env_enabled() is True
    assert is_telemetry_env_enabled() is True


# --------------------------------------------------------------------------- #
# Defensive paths
# --------------------------------------------------------------------------- #


def test_a_config_object_without_get_yields_a_disabled_config():
    """`telemetry_config_from_global_config` is called with whatever the CLI holds.

    A caller that hands it a list, a string, or a stub without `.get` must get a disabled
    config back rather than an exception — telemetry resolution must never be the thing
    that stops a run from starting.
    """
    from nemo_gym.telemetry.setup import telemetry_config_from_global_config

    class _NoGet:
        pass

    assert telemetry_config_from_global_config(_NoGet()).enabled is False
    assert telemetry_config_from_global_config(["not", "a", "mapping"]).enabled is False


def test_env_enabled_but_config_disabled_returns_none(clean_otel_env, monkeypatch):
    """Belt and braces: the env gate and the resolved lens config are checked separately.

    They agree today, but the env gate exists to avoid importing lens and the config check
    is what lens itself acts on. If they ever disagree, the safe answer is 'off'.
    """
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")

    from nemo.lens import NemoLensConfig

    monkeypatch.setattr(
        "nemo.lens.NemoLensConfig.from_env",
        classmethod(lambda cls, **kwargs: NemoLensConfig(enabled=False)),
    )
    assert init_telemetry(server_name="x") is None


def test_logging_bridge_is_set_up_when_logs_are_enabled(clean_otel_env, monkeypatch):
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORTER", "console")
    clean_otel_env.setenv("NEMO_GYM_OTEL_LOGS_ENABLED", "1")

    calls = []
    monkeypatch.setattr("nemo.lens.logging_bridge.setup_logging_bridge", lambda *a, **k: calls.append(1))

    handle = init_telemetry(server_name="weather")
    assert handle is not None and handle.is_exporting
    assert calls == [1]


def test_a_failing_logging_bridge_does_not_break_startup(clean_otel_env, monkeypatch):
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")
    clean_otel_env.setenv("NEMO_GYM_OTEL_EXPORTER", "console")
    clean_otel_env.setenv("NEMO_GYM_OTEL_LOGS_ENABLED", "1")

    def boom(*args, **kwargs):
        raise RuntimeError("no log exporter")

    monkeypatch.setattr("nemo.lens.logging_bridge.setup_logging_bridge", boom)
    assert init_telemetry(server_name="weather") is not None
