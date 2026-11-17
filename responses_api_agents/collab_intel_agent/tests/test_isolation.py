# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.collab_intel_agent import app as app_module
from responses_api_agents.collab_intel_agent import isolation
from responses_api_agents.collab_intel_agent.app import CollabIntelAgent
from responses_api_agents.collab_intel_agent.pytest_bridge import run_isolated_pytest


def _runner(tmp_path, provider=None):
    return isolation.PytestIsolation(
        tmp_path,
        provider or {"docker": {"create": {"network": "host", "extra_run_args": ["--privileged"]}}},
        {},
        "pytest:prepared",
    )


def _request(directory):
    return {
        "directory": str(directory),
        "targets": ["test_code.py"],
        "exitfirst": True,
        "noconftest": False,
        "timeout_s": 30,
    }


def _sandbox(monkeypatch, *, failure=None, error_type=None, return_code=0):
    instances = []

    class Sandbox:
        def __init__(self, provider, spec):
            self.provider, self.spec = provider, spec
            self.calls = []
            self.stopped = False
            instances.append(self)

        async def start(self):
            if failure == "start":
                raise RuntimeError("provider unavailable")

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            if failure == "exec":
                raise RuntimeError("provider unavailable")
            if len(self.calls) == 1:
                return SimpleNamespace(
                    return_code=return_code, error_type=error_type, stdout="pytest output", stderr=None
                )
            return SimpleNamespace(return_code=0, error_type=None)

        async def download(self, remote_path, local_path):
            if failure == "download":
                raise RuntimeError("transfer failed")
            Path(local_path).write_text(json.dumps({"test_code.py": base64.b64encode(b"updated").decode()}))

        async def stop(self):
            self.stopped = True
            if failure == "stop":
                raise RuntimeError("cleanup failed")

    monkeypatch.setattr(isolation, "AsyncSandbox", Sandbox)
    return instances


def test_only_episode_files_enter_a_fresh_offline_sandbox(tmp_path, monkeypatch):
    root = tmp_path / "this_phase"
    root.mkdir()
    episode = root / "coding"
    episode.mkdir()
    (episode / "test_code.py").write_text("assert True")
    (root / "other_episode_secret").write_text("not this episode")
    (tmp_path / "server_secret").write_text("not this phase")
    monkeypatch.setenv("SECRET_TOKEN", "must not be inherited")
    instances = _sandbox(monkeypatch)
    runner = _runner(root)

    async def run():
        for _ in range(2):
            result = await runner.run(_request(episode))
            assert result == {"return_code": 0, "output": "pytest output"}

    asyncio.run(run())
    assert len(instances) == 2
    for sandbox in instances:
        assert set(sandbox.spec.files) == {"/workspace/episode/test_code.py"}
        assert sandbox.spec.env == {"HOME": "/workspace/episode", "PYTHONDONTWRITEBYTECODE": "1"}
        assert sandbox.spec.provider_options == {}
        assert sandbox.provider["docker"]["create"]["network"] == "none"
        assert sandbox.provider["docker"]["create"]["extra_run_args"] == []
        assert sandbox.provider["docker"]["create"]["cap_drop"] == ["ALL"]
        assert sandbox.spec.ttl_s == 150
        assert sandbox.stopped
        assert "python -I -m pytest -q -x --no-header -p no:cacheprovider test_code.py" == sandbox.calls[0][0]
    assert (episode / "test_code.py").read_text() == "updated"
    assert (root / "other_episode_secret").read_text() == "not this episode"


@pytest.mark.parametrize("failure", ["start", "exec", "download", "stop"])
def test_provider_failures_stop_sandbox_and_never_execute_on_host(tmp_path, monkeypatch, failure):
    (tmp_path / "test_code.py").write_text("assert True")
    instances = _sandbox(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError):
        asyncio.run(_runner(tmp_path).run(_request(tmp_path)))
    assert len(instances) == 1 and instances[0].stopped
    assert (tmp_path / "test_code.py").read_text() == ("updated" if failure == "stop" else "assert True")


@pytest.mark.parametrize("error_type, code", [("provider", 0), (None, None)])
def test_missing_process_exit_is_not_a_test_result(tmp_path, monkeypatch, error_type, code):
    instances = _sandbox(monkeypatch, error_type=error_type, return_code=code)
    with pytest.raises(RuntimeError, match="process exit"):
        asyncio.run(_runner(tmp_path).run(_request(tmp_path)))
    assert instances[0].stopped


def test_timeout_stops_sandbox_and_retains_harness_timeout_contract(tmp_path, monkeypatch):
    instances = _sandbox(monkeypatch, error_type="timeout", return_code=125)
    assert asyncio.run(_runner(tmp_path).run(_request(tmp_path))) == {"return_code": None, "output": ""}
    assert instances[0].stopped


def test_valid_test_failure_is_not_infrastructure_failure(tmp_path, monkeypatch):
    instances = _sandbox(monkeypatch, return_code=1)
    result = asyncio.run(_runner(tmp_path).run(_request(tmp_path)))
    assert result["return_code"] == 1 and instances[0].stopped


def test_hidden_tests_preserve_noconftest_and_explicit_targets(tmp_path, monkeypatch):
    instances = _sandbox(monkeypatch)
    request = _request(tmp_path) | {"noconftest": True, "exitfirst": False}
    asyncio.run(_runner(tmp_path).run(request))
    assert instances[0].calls[0][0] == (
        "python -I -m pytest -q --no-header -p no:cacheprovider --noconftest --rootdir /workspace/episode -c /dev/null test_code.py"
    )


@pytest.mark.parametrize("target", ["../other/test.py", "/server/test.py", "--override-ini=anything"])
def test_target_cannot_escape_episode(tmp_path, monkeypatch, target):
    instances = _sandbox(monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(_runner(tmp_path).run(_request(tmp_path) | {"targets": [target]}))
    assert not instances


def test_path_and_symlink_cannot_read_server_or_other_phase(tmp_path, monkeypatch):
    root = tmp_path / "phase"
    root.mkdir()
    outside = tmp_path / "server_secret"
    outside.write_text("not episode data")
    instances = _sandbox(monkeypatch)
    runner = _runner(root)
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(runner.run(_request(tmp_path)))
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        asyncio.run(runner.run(_request(root)))
    assert not instances


@pytest.mark.parametrize("name", ["../server_secret", "/server_secret", "a/../../escape", ".", "a//b"])
def test_returned_paths_are_validated_before_any_write(tmp_path, name):
    (tmp_path / "keep").write_text("original")
    with pytest.raises(ValueError):
        isolation.restore_files(tmp_path, {"keep": "Y2hhbmdlZA==", name: "eA=="})
    assert (tmp_path / "keep").read_text() == "original"


def test_returned_symlinks_fail_collection_and_cannot_be_followed(tmp_path, monkeypatch):
    (tmp_path / "keep").write_text("original")
    with pytest.raises(ValueError):
        isolation.restore_files(tmp_path, {"a": "eA==", "a/b": "eA=="})
    (tmp_path / "link").symlink_to(tmp_path / "keep")
    with pytest.raises(ValueError):
        isolation.restore_files(tmp_path, {"keep": "Y2hhbmdlZA=="})
    assert (tmp_path / "keep").read_text() == "original"


def test_opensandbox_disables_egress_without_volumes(tmp_path):
    runner = _runner(tmp_path, {"opensandbox": {}})
    assert runner.provider_options == {"network_policy": {"defaultAction": "deny", "egress": []}}


def test_unsupported_provider_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="isolation policy"):
        _runner(tmp_path, {"not_isolated": {}})


def test_bridge_runs_every_pytest_request_in_gym(tmp_path, monkeypatch):
    instances = _sandbox(monkeypatch)
    runner = _runner(tmp_path)

    async def run():
        async with runner.serve() as endpoint:
            return await asyncio.to_thread(run_isolated_pytest, endpoint, tmp_path, ["test_code.py"], True, False, 30)

    assert asyncio.run(run()) == (0, "pytest output")
    assert len(instances) == 1 and instances[0].stopped


def test_unavailable_bridge_never_falls_back_to_local_pytest(tmp_path):
    with pytest.raises(OSError):
        run_isolated_pytest(str(tmp_path / "absent.sock"), tmp_path, [], True, False, 1)


def test_bridge_failure_aborts_phase_even_if_harness_catches_error(tmp_path, monkeypatch):
    instances = _sandbox(monkeypatch, failure="start")
    runner = _runner(tmp_path)

    async def run():
        async with runner.serve() as endpoint:
            with pytest.raises(RuntimeError, match="local execution is disabled"):
                await asyncio.to_thread(run_isolated_pytest, endpoint, tmp_path, [], True, False, 30)

    with pytest.raises(RuntimeError, match="Isolated pytest failed"):
        asyncio.run(run())
    assert instances[0].stopped


def test_cancellation_stops_inflight_sandbox(tmp_path, monkeypatch):
    stopped = asyncio.Event()
    started = asyncio.Event()

    class Sandbox:
        def __init__(self, *args):
            pass

        async def start(self):
            started.set()

        async def exec(self, *args, **kwargs):
            await asyncio.Event().wait()

        async def stop(self):
            stopped.set()

    monkeypatch.setattr(isolation, "AsyncSandbox", Sandbox)

    async def run():
        task = asyncio.create_task(_runner(tmp_path).run(_request(tmp_path)))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(run())


def test_agent_fails_closed_without_isolation_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    agent = CollabIntelAgent.model_construct(config=SimpleNamespace(pytest_sandbox_image=None))
    agent._run_harness = AsyncMock()
    result = asyncio.run(agent._run(tmp_path, "solo", 1, tmp_path / "out"))
    assert result["pass"] is False and result["failure_kind"] == "agent_run_error"
    agent._run_harness.assert_not_awaited()


def test_agent_wraps_harness_in_phase_isolation(tmp_path, monkeypatch):
    seen = []

    class Isolation:
        def __init__(self, root, provider, named, image):
            seen.append((root, provider, named, image))

        @asynccontextmanager
        async def serve(self):
            yield "/private/control.sock"

    monkeypatch.setattr(app_module, "PytestIsolation", Isolation)
    monkeypatch.setattr(CollabIntelAgent, "model_post_init", lambda self, context: None)
    agent = CollabIntelAgent.model_construct(
        config=SimpleNamespace(pytest_sandbox_image="pytest:prepared", sandbox_provider="sandbox"),
        server_client=SimpleNamespace(global_config_dict={"sandbox": {"docker": {}}}),
    )
    agent._run_harness = AsyncMock(return_value={"pass": True})
    assert asyncio.run(agent._run(tmp_path, "solo", 1, tmp_path / "out")) == {"pass": True}
    assert seen[0][1:] == ("sandbox", {"sandbox": {"docker": {}}}, "pytest:prepared")
    args = agent._run_harness.await_args.args
    assert args[-2] == "/private/control.sock" and args[-1] == seen[0][0]
    assert not seen[0][0].exists()


def test_bridge_installs_hook_for_fresh_worker_contexts(monkeypatch):
    import contextvars
    import sys
    from types import ModuleType

    from responses_api_agents.collab_intel_agent import pytest_bridge

    package = ModuleType("collab_intel")
    harness = ModuleType("collab_intel.harness")
    upstream = ModuleType("collab_intel.harness.sandbox")
    original = contextvars.ContextVar("unprotected", default=None)
    upstream.PYTEST_RUNNER = original
    harness.sandbox = upstream
    for name, module in [
        ("collab_intel", package),
        ("collab_intel.harness", harness),
        ("collab_intel.harness.sandbox", upstream),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["bridge", "/private/runner.sock", "scenario", "--version", "solo"])
    calls = []
    monkeypatch.setattr(pytest_bridge, "run_isolated_pytest", lambda *args: calls.append(args) or (0, "ok"))

    def cli(name, run_name):
        assert (name, run_name) == ("collab_intel.harness.cli", "__main__")
        # New threads start with a fresh context. They must not see the upstream
        # default None, which would silently re-enable local pytest.
        runner = contextvars.Context().run(upstream.PYTEST_RUNNER.get)
        assert runner(Path("episode"), [], True, False, 30) == (0, "ok")

    monkeypatch.setattr(pytest_bridge.runpy, "run_module", cli)
    pytest_bridge.main()
    assert calls == [("/private/runner.sock", Path("episode"), [], True, False, 30)]
    assert upstream.PYTEST_RUNNER is original
