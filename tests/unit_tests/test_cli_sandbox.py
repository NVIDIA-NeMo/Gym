# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `gym sandbox`.

Everything runs against an in-process fake provider, so the suite exercises the
real resolution and runner code without a container runtime. Both server kinds
are covered: the design's whole claim is that an agent and a resources server
are treated alike, so that is asserted rather than assumed.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.sandbox as cli_sandbox
from nemo_gym.config_types import ConfigError
from nemo_gym.sandbox import SandboxSpec, register_provider
from nemo_gym.sandbox.hooks import (
    spec_from_mapping,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxStatus


########################################
# Fake provider
########################################


class RecordingProvider:
    """Provider that records what it was asked to do and never leaves the process."""

    name = "recording"
    instances: list["RecordingProvider"] = []

    def __init__(self, exit_code: int = 0, fail_create: bool = False, connectable: bool = True, **_: Any) -> None:
        # Real providers are constructed from their YAML config block, so accept
        # and ignore arbitrary provider settings the way they do.
        self.exit_code = exit_code
        self.fail_create = fail_create
        self.connectable = connectable
        self.created_specs: list[SandboxSpec] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.closed: list[str] = []
        RecordingProvider.instances.append(self)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if self.fail_create:
            raise RuntimeError("image pull failed")
        self.created_specs.append(spec)
        return SandboxHandle(sandbox_id="sbx-1", provider_name=self.name, raw={})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        self.exec_calls.append({"command": command, "cwd": cwd, "user": user, "timeout_s": timeout_s})
        return SandboxExecResult(stdout="hello", stderr=None, return_code=self.exit_code)

    async def upload_file(self, handle, source_path: Path, target_path: str) -> None:
        self.uploads.append((source_path, target_path))

    async def download_file(self, handle, source_path: str, target_path: Path) -> None:
        target_path.write_bytes(b"")

    async def status(self, handle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle) -> None:
        self.closed.append(handle.sandbox_id)

    async def aclose(self) -> None:
        return None

    # Connect capability, so --keep / exec / rm are exercisable.
    async def serialize_handle(self, handle, *, scope=None) -> dict[str, Any]:
        if not self.connectable:
            raise RuntimeError("not connectable")
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw={})


class UnconnectableProvider(RecordingProvider):
    """A provider with no way back to a sandbox once this process exits."""

    name = "unconnectable"

    serialize_handle = None  # type: ignore[assignment]
    connect = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _register_fakes():
    RecordingProvider.instances.clear()
    for name, cls in (("recording", RecordingProvider), ("unconnectable", UnconnectableProvider)):
        register_provider(name, cls, override=True)
    yield
    RecordingProvider.instances.clear()


########################################
# Config fixtures
########################################

AGENT_CONFIG = {
    "swe_like": {
        "responses_api_agents": {
            "swe_like": {
                "entrypoint": "app.py",
                "host": "127.0.0.1",
                "port": 10001,
                "sandbox_provider": {"recording": {}},
                "sandbox_spec": {
                    "ttl_s": 900,
                    "resources": {"cpu": 2, "memory_mib": 4096},
                    "metadata": {"benchmark": "demo"},
                },
                "sandbox_environment_kwargs": {"cwd": "/testbed", "user": "root", "conda_env": "testbed"},
                "sandbox_task": {
                    "id_from_row": "instance_id",
                    "image_from_row": "image_name",
                },
            }
        }
    }
}

RESOURCES_SERVER_CONFIG = {
    "tool_host": {
        "resources_servers": {
            "tool_host": {
                "entrypoint": "app.py",
                "domain": "knowledge",
                "host": "127.0.0.1",
                "port": 10002,
                "sandbox_provider": {"recording": {}},
                # Fully declarative: no hooks, no row — the Layer-0/1 path.
                "sandbox_spec": {"image": "docker.io/library/python:3.12-slim", "workdir": "/tmp"},
            }
        }
    }
}


def _config(*blocks: dict, **overrides: Any):
    merged: dict[str, Any] = {}
    for block in blocks:
        merged.update(block)
    merged.update(overrides)
    return OmegaConf.create(merged)


def _rows(tmp_path: Path, rows: list[dict]) -> str:
    fpath = tmp_path / "tasks.jsonl"
    fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(fpath)


def _run_debug(monkeypatch, config, **cli):
    """Drive `gym sandbox debug` with a merged config, as the router would."""
    merged = _config(config, **cli) if isinstance(config, dict) else config
    for key, value in cli.items():
        merged[key] = value
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    return cli_sandbox.debug()


def _expect_error(capsys, fragment: str, fn, *args, **kwargs) -> None:
    """Assert a command exits non-zero having printed `fragment`.

    The CLI entrypoints wrap `ConfigError` into a rendered message plus
    `SystemExit(1)`, so asserting on the message is both what the caller
    actually sees and the thing worth pinning. Rich hard-wraps its output, so
    whitespace is collapsed before matching.
    """
    with pytest.raises(SystemExit) as excinfo:
        fn(*args, **kwargs)
    assert excinfo.value.code == 1
    printed = " ".join(capsys.readouterr().out.split())
    assert fragment in printed, printed


########################################
# hooks: spec_from_mapping
########################################


def test_spec_from_mapping_reads_every_field() -> None:
    spec = spec_from_mapping(
        {
            "image": "img:1",
            "ttl_s": 60,
            "ready_timeout_s": 30,
            "workdir": "/w",
            "env": {"A": "1"},
            "files": {"/f": "x"},
            "metadata": {"m": "v"},
            "resources": {"cpu": 2, "memory_mib": 512},
            "entrypoint": ["sleep", "1"],
            "provider_options": {"p": 1},
        }
    )
    assert spec.image == "img:1"
    assert spec.ttl_s == 60 and spec.ready_timeout_s == 30 and spec.workdir == "/w"
    assert spec.env == {"A": "1"} and spec.files == {"/f": "x"} and spec.metadata == {"m": "v"}
    assert spec.resources.cpu == 2 and spec.resources.memory_mib == 512
    assert spec.entrypoint == ["sleep", "1"] and spec.provider_options == {"p": 1}


def test_spec_from_mapping_applies_image_rewrites() -> None:
    """A mirrored registry should be transparent to whoever named the image."""
    spec = spec_from_mapping(
        {"image": "docker.io/library/python:3.12", "image_rewrites": [{"from": "docker.io/", "to": "mirror/"}]}
    )
    assert spec.image == "mirror/library/python:3.12"


def test_spec_from_mapping_rejects_unknown_keys() -> None:
    """A typo'd spec key is otherwise invisible until the sandbox misbehaves."""
    with pytest.raises(ValueError, match="Unknown sandbox_spec keys: workdirr"):
        spec_from_mapping({"image": "img", "workdirr": "/w"})


def test_spec_from_mapping_handles_empty() -> None:
    assert spec_from_mapping(None).image is None


########################################
# hooks: load_hook
########################################


@pytest.mark.parametrize(
    "reference, match",
    [
        ("json.dumps", "must be of the form"),
        ("nemo_gym_missing_module_xyz:thing", "could not be imported"),
        ("json:not_a_real_attribute", "not found"),
        ("json:__doc__", "not callable"),
    ],
)
########################################
# hooks: id + spec resolution
########################################


def _resolver(row, server_config):
    return SandboxSpec(image=f"resolved-{(row or {}).get('instance_id', 'none')}")


def _boom(row, server_config):
    raise ValueError("nope")


########################################
# hooks: command wrapping
########################################


def _upper_wrapper(command, **kwargs):
    return f"WRAPPED({command})"


########################################
# Server discovery
########################################


def test_discovers_agents_and_resources_servers_alike() -> None:
    """`sandbox_provider` is the marker, not the server kind."""
    servers = cli_sandbox.discover_sandbox_servers(_config(AGENT_CONFIG, RESOURCES_SERVER_CONFIG))
    assert set(servers) == {"swe_like", "tool_host"}
    assert servers["swe_like"]["server_type"] == "responses_api_agents"
    assert servers["tool_host"]["server_type"] == "resources_servers"


def test_ignores_blocks_without_a_sandbox() -> None:
    config = _config({"plain": {"responses_api_models": {"plain": {"entrypoint": "app.py"}}}})
    assert cli_sandbox.discover_sandbox_servers(config) == {}


def test_select_server_infers_when_unambiguous() -> None:
    servers = cli_sandbox.discover_sandbox_servers(_config(AGENT_CONFIG))
    assert cli_sandbox.select_server(servers, None)[0] == "swe_like"


def test_select_server_requires_a_choice_when_ambiguous() -> None:
    servers = cli_sandbox.discover_sandbox_servers(_config(AGENT_CONFIG, RESOURCES_SERVER_CONFIG))
    with pytest.raises(ConfigError, match="Pick one with --server"):
        cli_sandbox.select_server(servers, None)


def test_select_server_reports_unknown_name() -> None:
    servers = cli_sandbox.discover_sandbox_servers(_config(AGENT_CONFIG))
    with pytest.raises(ConfigError, match="Unknown server"):
        cli_sandbox.select_server(servers, "nope")


def test_no_sandbox_server_is_actionable() -> None:
    with pytest.raises(ConfigError, match="declaring `sandbox_provider`"):
        cli_sandbox.select_server({}, None)


########################################
# debug: dry run
########################################


def test_dry_run_resolves_without_creating_a_sandbox(monkeypatch, tmp_path, capsys) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(monkeypatch, AGENT_CONFIG, input_jsonl_fpath=rows, dry_run=True)
    out = capsys.readouterr().out
    assert "img:a" in out and "task-a" in out
    assert not RecordingProvider.instances or not RecordingProvider.instances[0].created_specs


def test_dry_run_json_output(monkeypatch, tmp_path, capsys) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(monkeypatch, AGENT_CONFIG, input_jsonl_fpath=rows, dry_run=True, json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["image"] == "img:a"
    assert payload[0]["task_id"] == "task-a"
    assert payload[0]["server_type"] == "responses_api_agents"


def test_resources_server_needs_no_row_or_hooks(monkeypatch, capsys) -> None:
    """The Layer-0/1 path: a resources server with a declarative spec and no dataset."""
    _run_debug(monkeypatch, RESOURCES_SERVER_CONFIG, dry_run=True, command="python -V")
    out = capsys.readouterr().out
    assert "docker.io/library/python:3.12-slim" in out
    assert "booting the server's configured sandbox" in out


########################################
# debug: execution
########################################


def test_runs_command_and_writes_a_trace(monkeypatch, tmp_path) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    out_dir = tmp_path / "out"
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="echo hi",
        output_dirpath=str(out_dir),
    )

    provider = RecordingProvider.instances[-1]
    assert provider.exec_calls[0]["command"] == "echo hi"
    # Exec defaults come from sandbox_environment_kwargs.
    assert provider.exec_calls[0]["cwd"] == "/testbed"
    assert provider.exec_calls[0]["user"] == "root"
    # The sandbox is torn down even on the happy path.
    assert provider.closed == ["sbx-1"]

    traces = [json.loads(line) for line in (out_dir / "traces.jsonl").read_text().splitlines()]
    assert len(traces) == 1
    trace = traces[0]
    assert trace["ok"] is True and trace["reason"] == "pass" and trace["exit_code"] == 0
    assert trace["image"] == "img:a" and trace["sandbox_id"] == "sbx-1"
    assert trace["stdout"] == "hello"
    assert set(trace["timing"]) >= {"boot", "setup", "exec", "total"}
    # The resolved config is re-runnable via --config.
    assert (out_dir / "config.yaml").exists()


def test_nonzero_exit_is_classified_and_propagates(monkeypatch, tmp_path) -> None:
    """A failing command must fail the process — this is what makes it CI-usable."""
    register_provider("recording", lambda **k: RecordingProvider(exit_code=3), override=True)
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    with pytest.raises(SystemExit) as excinfo:
        _run_debug(
            monkeypatch,
            AGENT_CONFIG,
            input_jsonl_fpath=rows,
            command="false",
            output_dirpath=str(tmp_path / "out"),
        )
    assert excinfo.value.code == 1
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert trace["reason"] == "nonzero_exit" and trace["exit_code"] == 3 and trace["ok"] is False


def test_exit_zero_suppresses_failure_exit(monkeypatch, tmp_path) -> None:
    register_provider("recording", lambda **k: RecordingProvider(exit_code=3), override=True)
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="false",
        output_dirpath=str(tmp_path / "out"),
        exit_zero=True,
    )


def test_create_failure_is_recorded_not_raised(monkeypatch, tmp_path) -> None:
    """A boot failure should read as a result, not a traceback."""
    register_provider("recording", lambda **k: RecordingProvider(fail_create=True), override=True)
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    with pytest.raises(SystemExit):
        _run_debug(
            monkeypatch,
            AGENT_CONFIG,
            input_jsonl_fpath=rows,
            command="echo hi",
            output_dirpath=str(tmp_path / "out"),
        )
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert trace["reason"] == "error" and "image pull failed" in trace["error"]
    assert trace["sandbox_id"] is None


def test_script_path_is_uploaded_outside_the_workdir(monkeypatch, tmp_path) -> None:
    """The script must not land in the state being inspected."""
    script = tmp_path / "poke.sh"
    script.write_text("#!/bin/sh\necho poked\n")
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        script_path=str(script),
        output_dirpath=str(tmp_path / "out"),
    )
    provider = RecordingProvider.instances[-1]
    (_, remote) = provider.uploads[0]
    assert remote.startswith("/tmp/") and remote.endswith("poke.sh")
    assert remote in provider.exec_calls[0]["command"]
    assert "chmod +x" in provider.exec_calls[0]["command"]


def test_image_flag_overrides_everything(monkeypatch, tmp_path) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="echo hi",
        image="override:1",
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].image == "override:1"


def test_missing_image_is_actionable(monkeypatch, tmp_path, capsys) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a"}])
    _expect_error(
        capsys,
        "Could not determine a container image",
        _run_debug,
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="echo hi",
    )


def test_env_flag_merges_into_spec(monkeypatch, tmp_path) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="echo hi",
        env=["FOO=bar"],
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].env["FOO"] == "bar"


def test_malformed_env_flag_is_rejected(monkeypatch, tmp_path, capsys) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _expect_error(
        capsys,
        "--env expects KEY=VALUE",
        _run_debug,
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="echo hi",
        env=["FOO"],
    )


########################################
# debug: row selection
########################################


def test_task_selection_filters_rows(monkeypatch, tmp_path, capsys) -> None:
    rows = _rows(
        tmp_path,
        [
            {"instance_id": "task-a", "image_name": "img:a"},
            {"instance_id": "task-b", "image_name": "img:b"},
        ],
    )
    _run_debug(monkeypatch, AGENT_CONFIG, input_jsonl_fpath=rows, task_ids=["task-b"], dry_run=True)
    out = capsys.readouterr().out
    assert "task-b" in out and "task-a" not in out


def test_unknown_task_lists_what_is_available(monkeypatch, tmp_path, capsys) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _expect_error(
        capsys,
        "Available ids: task-a",
        _run_debug,
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        task_ids=["nope"],
        dry_run=True,
    )


def test_limit_caps_the_number_of_tasks(monkeypatch, tmp_path) -> None:
    rows = _rows(tmp_path, [{"instance_id": f"t{i}", "image_name": f"img:{i}"} for i in range(5)])
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="echo hi",
        limit=2,
        output_dirpath=str(tmp_path / "out"),
    )
    traces = (tmp_path / "out" / "traces.jsonl").read_text().splitlines()
    assert len(traces) == 2


def test_missing_input_file_is_actionable(monkeypatch, tmp_path, capsys) -> None:
    _expect_error(
        capsys,
        "Input file not found",
        _run_debug,
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=str(tmp_path / "nope.jsonl"),
        dry_run=True,
    )


def test_malformed_jsonl_names_the_line(monkeypatch, tmp_path, capsys) -> None:
    fpath = tmp_path / "bad.jsonl"
    fpath.write_text('{"instance_id": "a"}\nnot json\n')
    _expect_error(
        capsys,
        "at line 2",
        _run_debug,
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=str(fpath),
        dry_run=True,
    )


########################################
# debug: --keep
########################################


def test_keep_leaves_the_sandbox_running(monkeypatch, tmp_path) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        input_jsonl_fpath=rows,
        command="echo hi",
        keep=True,
        output_dirpath=str(tmp_path / "out"),
    )
    provider = RecordingProvider.instances[-1]
    assert provider.closed == []
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert trace["kept"] is True and trace["sandbox_id"] == "sbx-1"


def test_keep_is_refused_when_the_sandbox_could_not_be_reached_again(monkeypatch, tmp_path, capsys) -> None:
    """Keeping a sandbox you cannot reattach to just leaks capacity."""
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {"unconnectable": {}}
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _expect_error(
        capsys,
        "--keep needs a provider that can reattach",
        _run_debug,
        monkeypatch,
        config,
        input_jsonl_fpath=rows,
        command="echo hi",
        keep=True,
    )


########################################
# exec / rm
########################################


def test_exec_reattaches_and_runs(monkeypatch, capsys) -> None:
    merged = _config(AGENT_CONFIG, sandbox_id="sbx-9", command="git log -1")
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    cli_sandbox.exec_command()
    provider = RecordingProvider.instances[-1]
    assert provider.exec_calls[0]["command"] == "git log -1"
    assert "hello" in capsys.readouterr().out


def test_exec_requires_a_sandbox_id(monkeypatch, capsys) -> None:
    merged = _config(AGENT_CONFIG, command="ls")
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    _expect_error(capsys, "--sandbox-id is required", cli_sandbox.exec_command)


def test_exec_explains_providers_that_cannot_reattach(monkeypatch, capsys) -> None:
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {"unconnectable": {}}
    merged = _config(config, sandbox_id="sbx-9", command="ls")
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    _expect_error(capsys, "cannot reattach", cli_sandbox.exec_command)


def test_rm_deletes_the_sandbox(monkeypatch, capsys) -> None:
    merged = _config(AGENT_CONFIG, sandbox_id="sbx-9")
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    cli_sandbox.rm()
    assert RecordingProvider.instances[-1].closed == ["sbx-9"]
    assert "deleted sandbox sbx-9" in capsys.readouterr().out


def test_rm_requires_a_sandbox_id(monkeypatch, capsys) -> None:
    merged = _config(AGENT_CONFIG)
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    _expect_error(capsys, "--sandbox-id is required", cli_sandbox.rm)


########################################
# Config validation
########################################


def test_requires_an_action() -> None:
    with pytest.raises(ValueError, match=r"pass --command, --script-path, --dry-run, or --list-tasks"):
        cli_sandbox.SandboxDebugConfig.model_validate({})


def test_rejects_two_actions() -> None:
    with pytest.raises(ValueError, match="at most one of --command or --script-path"):
        cli_sandbox.SandboxDebugConfig.model_validate({"command": "ls", "script_path": "x"})


def test_dry_run_allows_a_command() -> None:
    """--dry-run is a modifier: showing how a command would be wrapped is the point."""
    config = cli_sandbox.SandboxDebugConfig.model_validate({"dry_run": True, "command": "ls"})
    assert config.command == "ls"


def test_rejects_missing_script(tmp_path) -> None:
    with pytest.raises(ValueError, match="--script-path is not a file"):
        cli_sandbox.SandboxDebugConfig.model_validate({"script_path": str(tmp_path / "nope.sh")})


def test_rejects_bad_concurrency() -> None:
    with pytest.raises(ValueError, match="--concurrency must be >= 1"):
        cli_sandbox.SandboxDebugConfig.model_validate({"command": "ls", "concurrency": 0})


########################################
# Timeout classification
########################################


class TimingOutProvider(RecordingProvider):
    """Provider that enforces its own timeout, reporting it as a negative code.

    This is what OpenSandbox actually does: `exec` returns rather than raising,
    with a code no real process could produce.
    """

    name = "timing_out"

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        self.exec_calls.append({"command": command, "cwd": cwd, "user": user, "timeout_s": timeout_s})
        if timeout_s:
            await asyncio.sleep(timeout_s)
        return SandboxExecResult(stdout=None, stderr="CommandExecError: -1", return_code=-1)


def _timing_out_config() -> dict:
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {"timing_out": {}}
    return config


def test_provider_enforced_timeout_is_not_reported_as_a_plain_failure(monkeypatch, tmp_path) -> None:
    """A negative code after the budget elapsed is a timeout, not a non-zero exit.

    Providers enforce the timeout themselves and return instead of raising, so
    without this the run reads as "your command failed" when it never finished.
    """
    register_provider("timing_out", TimingOutProvider, override=True)
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    with pytest.raises(SystemExit):
        _run_debug(
            monkeypatch,
            _timing_out_config(),
            input_jsonl_fpath=rows,
            command="sleep 60",
            timeout_total=0.05,
            output_dirpath=str(tmp_path / "out"),
        )
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert trace["reason"] == "timeout"
    assert "timed out after 0.05s" in trace["error"]


class AbnormalExitProvider(RecordingProvider):
    """Returns a negative code immediately — killed, but not by our timeout."""

    name = "abnormal"

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        self.exec_calls.append({"command": command})
        return SandboxExecResult(stdout=None, stderr=None, return_code=-9)


def test_negative_code_well_inside_the_budget_is_an_error_not_a_timeout(monkeypatch, tmp_path) -> None:
    register_provider("abnormal", AbnormalExitProvider, override=True)
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {"abnormal": {}}
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    with pytest.raises(SystemExit):
        _run_debug(
            monkeypatch,
            config,
            input_jsonl_fpath=rows,
            command="boom",
            timeout_total=600.0,
            output_dirpath=str(tmp_path / "out"),
        )
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert trace["reason"] == "error"
    assert "terminated abnormally" in trace["error"]


class SlowBootProvider(RecordingProvider):
    """Provider whose create never finishes in time — e.g. an image that won't pull."""

    name = "slow_boot"

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        await asyncio.sleep(10)
        raise AssertionError("should have timed out")


def test_boot_timeout_names_the_stage_and_its_own_budget(monkeypatch, tmp_path) -> None:
    """A failed image pull must not be reported against the command's budget.

    Getting this wrong sends you looking at your command when the sandbox never
    started.
    """
    register_provider("slow_boot", SlowBootProvider, override=True)
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {"slow_boot": {}}
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    with pytest.raises(SystemExit):
        _run_debug(
            monkeypatch,
            config,
            input_jsonl_fpath=rows,
            command="true",
            timeout_setup=0.05,
            timeout_total=600.0,
            output_dirpath=str(tmp_path / "out"),
        )
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert trace["reason"] == "timeout"
    assert trace["error"] == "boot timed out after 0.05s"
    assert trace["sandbox_id"] is None
    # A cancelled create can leave a sandbox the provider never handed back a
    # handle for. We cannot clean that up, so the trace has to admit it.
    assert trace["may_have_orphaned_sandbox"] is True


########################################
# Persisted config
########################################


def test_written_config_redacts_secrets(monkeypatch, tmp_path) -> None:
    """The run dir gets shared and attached to bug reports; keys must not be in it."""
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {
        "recording": {"connection": {"api_key": "super-secret-value"}}
    }
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        config,
        input_jsonl_fpath=rows,
        command="echo hi",
        output_dirpath=str(tmp_path / "out"),
    )
    written = (tmp_path / "out" / "config.yaml").read_text()
    assert "super-secret-value" not in written
    assert "****" in written


########################################
# Sandbox lifetime
########################################

# A server that leaves ttl_s unset — providers read that as "never auto-terminate".
NO_TTL_CONFIG = {
    "no_ttl": {
        "responses_api_agents": {
            "no_ttl": {
                "entrypoint": "app.py",
                "host": "127.0.0.1",
                "port": 10003,
                "sandbox_provider": {"recording": {}},
                "sandbox_spec": {"image": "img:x"},
            }
        }
    }
}


def test_sandbox_never_gets_an_unbounded_lifetime(monkeypatch, tmp_path) -> None:
    """A server with no ttl_s must not yield a sandbox that never expires.

    Providers treat a null lifetime as "delete me explicitly" — a promise this
    tool cannot keep if it is killed or the user forgets a --keep.
    """
    _run_debug(
        monkeypatch,
        NO_TTL_CONFIG,
        command="echo hi",
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].ttl_s == cli_sandbox.DEFAULT_TTL_S


def test_rollout_sized_server_ttl_is_capped_for_debugging(monkeypatch, tmp_path) -> None:
    """A server's ttl_s is sized for a whole rollout; a debug poke shouldn't inherit it.

    mini-swe-agent asks for five hours. Holding a debugging sandbox that long is
    how a cluster fills up with boxes nobody is using.
    """
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_spec"]["ttl_s"] = 18000
    _run_debug(
        monkeypatch,
        config,
        command="echo hi",
        image="img:a",
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].ttl_s == cli_sandbox.DEFAULT_TTL_S


def test_server_ttl_shorter_than_the_cap_is_kept(monkeypatch, tmp_path) -> None:
    """Capping is a ceiling, not a floor — a server asking for less still gets less."""
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_spec"]["ttl_s"] = 120
    _run_debug(
        monkeypatch,
        config,
        command="echo hi",
        image="img:a",
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].ttl_s == 120


def test_explicit_ttl_is_not_capped(monkeypatch, tmp_path) -> None:
    """The cap protects the default; someone who asks for longer means it."""
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        command="echo hi",
        image="img:a",
        ttl_s=7200,
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].ttl_s == 7200


def test_long_ttl_warns_about_wasted_capacity(monkeypatch, tmp_path, capsys) -> None:
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        command="echo hi",
        image="img:a",
        ttl_s=7200,
        output_dirpath=str(tmp_path / "out"),
    )
    printed = " ".join(capsys.readouterr().out.split())
    assert "--ttl 2h keeps a sandbox alive well past a normal debugging session" in printed


def test_short_ttl_does_not_warn(monkeypatch, tmp_path, capsys) -> None:
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        command="echo hi",
        image="img:a",
        ttl_s=600,
        output_dirpath=str(tmp_path / "out"),
    )
    assert "keeps a sandbox alive" not in capsys.readouterr().out


def test_ttl_flag_overrides_the_server(monkeypatch, tmp_path) -> None:
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        command="echo hi",
        image="img:a",
        ttl_s=120,
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].ttl_s == 120


def test_ttl_flag_overrides_even_an_unset_server_ttl(monkeypatch, tmp_path) -> None:
    _run_debug(
        monkeypatch,
        NO_TTL_CONFIG,
        command="echo hi",
        ttl_s=60,
        output_dirpath=str(tmp_path / "out"),
    )
    assert RecordingProvider.instances[-1].created_specs[0].ttl_s == 60


def test_kept_sandbox_reports_its_expiry(monkeypatch, tmp_path, capsys) -> None:
    """A kept sandbox holds capacity until its TTL, so the deadline is the point."""
    _run_debug(
        monkeypatch,
        AGENT_CONFIG,
        command="echo hi",
        image="img:a",
        keep=True,
        ttl_s=5400,
        output_dirpath=str(tmp_path / "out"),
    )
    printed = " ".join(capsys.readouterr().out.split())
    assert "expires in 1h30m" in printed
    assert "gym sandbox rm --sandbox-id sbx-1" in printed
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert trace["ttl_s"] == 5400


@pytest.mark.parametrize(
    "seconds, expected",
    [(45, "45s"), (600, "10m"), (3600, "1h"), (5400, "1h30m"), (18000, "5h")],
)
def test_duration_formatting(seconds, expected) -> None:
    assert cli_sandbox._format_duration(seconds) == expected


def test_rejects_nonpositive_ttl() -> None:
    with pytest.raises(ValueError, match="--ttl must be > 0"):
        cli_sandbox.SandboxDebugConfig.model_validate({"command": "ls", "ttl_s": 0})


########################################
# Execution timing
########################################


class TimedProvider(RecordingProvider):
    """Provider that spends time on transport and reports only the command's share.

    Sleeps longer than the duration it reports, mimicking a real round-trip: the
    caller observes ~0.15s while the sandbox says the command itself took 0.05s.
    """

    name = "timed"
    reported_ms = 50.0

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        self.exec_calls.append({"command": command})
        await asyncio.sleep(0.15)
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0, duration_ms=self.reported_ms)


def _timed_config(provider: str = "timed") -> dict:
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {provider: {}}
    return config


def test_timing_separates_command_from_transport(monkeypatch, tmp_path, capsys) -> None:
    """ "Why did this take ten minutes?" needs command time apart from round-trip time."""
    register_provider("timed", TimedProvider, override=True)
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        _timed_config(),
        input_jsonl_fpath=rows,
        command="pip install torch",
        output_dirpath=str(tmp_path / "out"),
    )

    timing = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])["timing"]
    assert timing["command"] == 0.05
    # exec covers command + transport, so overhead is what exec added on top.
    assert timing["overhead"] == pytest.approx(timing["exec"] - 0.05, abs=0.01)
    assert timing["overhead"] > 0
    assert "command 0.05s" in " ".join(capsys.readouterr().out.split())


class OverreportingProvider(TimedProvider):
    """Claims more command time than the caller observed — clock skew, or a lying provider."""

    name = "overreporting"
    reported_ms = 60_000.0


def test_overhead_never_goes_negative(monkeypatch, tmp_path) -> None:
    """A provider reporting more time than we observed must not yield negative overhead."""
    register_provider("overreporting", OverreportingProvider, override=True)
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch,
        _timed_config("overreporting"),
        input_jsonl_fpath=rows,
        command="true",
        output_dirpath=str(tmp_path / "out"),
    )
    timing = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])["timing"]
    assert timing["overhead"] == 0.0


def test_timing_breakdown_omitted_when_the_provider_cannot_measure(monkeypatch, tmp_path) -> None:
    """Most providers report no duration; don't invent one."""
    rows = _rows(tmp_path, [{"instance_id": "task-a", "image_name": "img:a"}])
    _run_debug(
        monkeypatch, AGENT_CONFIG, input_jsonl_fpath=rows, command="echo hi", output_dirpath=str(tmp_path / "out")
    )
    trace = json.loads((tmp_path / "out" / "traces.jsonl").read_text().splitlines()[0])
    assert "command" not in trace["timing"]
    assert "overhead" not in trace["timing"]


########################################
# Renewal on reattach
########################################


def test_exec_extends_the_expiry_before_running(monkeypatch, capsys) -> None:
    """Each command bumps the clock, so an in-use sandbox stays alive on a short TTL."""
    renewals: list[float] = []

    class RenewingProvider(RecordingProvider):
        name = "renewing"

        async def renew(self, handle, ttl_s: float) -> None:
            renewals.append(ttl_s)

    register_provider("renewing", RenewingProvider, override=True)
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {"renewing": {}}
    merged = _config(config, sandbox_id="sbx-9", command="ls", ttl_s=900, timeout_total=60)
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    cli_sandbox.exec_command()

    assert renewals == [900]
    assert "expires in 15m" in " ".join(capsys.readouterr().out.split())


def test_exec_renewal_covers_a_command_longer_than_the_ttl(monkeypatch) -> None:
    """A 30-minute command on a 15-minute TTL must not have the sandbox die under it."""
    renewals: list[float] = []

    class RenewingProvider(RecordingProvider):
        name = "renewing"

        async def renew(self, handle, ttl_s: float) -> None:
            renewals.append(ttl_s)

    register_provider("renewing", RenewingProvider, override=True)
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = {"renewing": {}}
    merged = _config(config, sandbox_id="sbx-9", command="sleep 1800", ttl_s=900, timeout_total=1800)
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    cli_sandbox.exec_command()

    assert renewals == [1860]  # command budget plus a minute of slack


def test_exec_still_works_when_the_provider_cannot_renew(monkeypatch, capsys) -> None:
    """Renewal is best effort; a provider without it keeps its original lifetime."""
    merged = _config(AGENT_CONFIG, sandbox_id="sbx-9", command="ls")
    monkeypatch.setattr(cli_sandbox, "get_global_config_dict", lambda **k: merged)
    cli_sandbox.exec_command()
    printed = capsys.readouterr().out
    assert "hello" in printed
    assert "expires in" not in printed


########################################
# Task discovery
########################################


def test_list_tasks_shows_ids_and_context(monkeypatch, tmp_path, capsys) -> None:
    """Guessing an id and reading it out of the error is a poor way to start."""
    rows = _rows(
        tmp_path,
        [
            {"instance_id": "task-a", "image_name": "img:a", "repo": "org/one"},
            {"instance_id": "task-b", "image_name": "img:b", "repo": "org/two"},
        ],
    )
    _run_debug(monkeypatch, AGENT_CONFIG, input_jsonl_fpath=rows, list_tasks=True)
    printed = " ".join(capsys.readouterr().out.split())
    assert "task-a" in printed and "task-b" in printed
    assert "repo=org/one" in printed
    assert "2 task(s)" in printed


def test_list_tasks_needs_no_provider(monkeypatch, tmp_path, capsys) -> None:
    """Listing reads a file; it should not need provider credentials or a cluster.

    The provider block here is deliberately unresolvable — listing must still work.
    """
    config = json.loads(json.dumps(AGENT_CONFIG))
    config["swe_like"]["responses_api_agents"]["swe_like"]["sandbox_provider"] = "not_a_real_reference"
    rows = _rows(tmp_path, [{"instance_id": "task-a"}])
    _run_debug(monkeypatch, config, input_jsonl_fpath=rows, list_tasks=True)
    assert "task-a" in capsys.readouterr().out


def test_list_tasks_json(monkeypatch, tmp_path, capsys) -> None:
    rows = _rows(tmp_path, [{"instance_id": "task-a", "repo": "org/one"}])
    _run_debug(monkeypatch, AGENT_CONFIG, input_jsonl_fpath=rows, list_tasks=True, json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload == [{"index": 0, "id": "task-a", "repo": "org/one"}]


def test_list_tasks_is_a_standalone_action() -> None:
    """It needs no --command; asking what exists precedes running anything."""
    config = cli_sandbox.SandboxDebugConfig.model_validate({"list_tasks": True})
    assert config.list_tasks is True
