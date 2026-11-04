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
"""`gym env validate --server-types`: compose the run args, check each server block, print one finding per line."""

import json
import sys
from pathlib import Path

import pytest
import yaml

import nemo_gym.config.global_config as global_config
from nemo_gym.cli.main import main


_RESOURCES_APP = """
from pydantic import BaseModel


class DemoConfig(BaseModel):
    name: str
    entrypoint: str
    domain: str


class DemoServer(BaseModel):
    config: DemoConfig


if __name__ == "__main__":
    DemoServer.run_webserver()
"""

# Rejects a removed option the way the Stirrup agent does: one error located at the option's own key.
_AGENT_APP = """
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError


class AgentConfig(BaseModel):
    name: str
    entrypoint: str
    max_turns: int = 10
    capture_dir: int = 0

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_options(cls, data: Any) -> Any:
        if "judge_only" in (data or {}):
            error = PydanticCustomError("removed_option", "removed option `judge_only` is no longer supported")
            details = [InitErrorDetails(type=error, loc=("judge_only",), input=data["judge_only"])]
            raise ValidationError.from_exception_data(cls.__name__, details)
        return data


class AgentServer(BaseModel):
    config: AgentConfig


if __name__ == "__main__":
    AgentServer.run_webserver()
"""

_SERVER_TYPES = ["--server-types", "resources_servers,responses_api_agents", "--json"]


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for server_dir, source in (
        ("resources_servers/demo_check", _RESOURCES_APP),
        ("responses_api_agents/demo_agent", _AGENT_APP),
    ):
        (tmp_path / server_dir).mkdir(parents=True)
        (tmp_path / server_dir / "requirements.txt").write_text("")
        (tmp_path / server_dir / "app.py").write_text(source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEMO_DOMAIN", raising=False)
    # Each run composes afresh, and the command's process-wide settings end with the test.
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    return tmp_path


def _benchmark(checkout: Path, *, domain: object = "math", **agent: object) -> str:
    config = {
        "env": {"resources_servers": {"demo_check": {"entrypoint": "app.py", "domain": domain}}},
        "agent": {"responses_api_agents": {"demo_agent": {"entrypoint": "app.py", **agent}}},
    }
    (checkout / "benchmark.yaml").write_text(yaml.safe_dump(config))
    return "+config_paths=[benchmark.yaml]"


def _validate(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, *args: str) -> tuple[int, list[dict]]:
    monkeypatch.setattr(sys, "argv", ["gym", "env", "validate", *args])
    with pytest.raises(SystemExit) as exit_info:
        main()
    lines = capsys.readouterr().out.splitlines()
    return exit_info.value.code, [json.loads(line) for line in lines if line.startswith("{")]


def test_a_valid_config_exits_0_with_no_findings_and_writes_nothing(checkout, monkeypatch, capsys) -> None:
    run_args = _benchmark(checkout, max_turns=3)
    assert _validate(monkeypatch, capsys, run_args, *_SERVER_TYPES) == (0, [])
    assert not list(checkout.rglob("__pycache__"))


def test_a_removed_option_is_invalid_at_its_full_path(checkout, monkeypatch, capsys) -> None:
    run_args = _benchmark(checkout, judge_only=False)
    assert _validate(monkeypatch, capsys, run_args, *_SERVER_TYPES) == (
        1,
        [
            {
                "kind": "invalid",
                "path": "agent.responses_api_agents.demo_agent.judge_only",
                "class": "AgentConfig",
                "message": "removed option `judge_only` is no longer supported",
                "input": "False",
            }
        ],
    )


def test_an_unknown_server_type_and_a_missing_server_are_unchecked(checkout, monkeypatch, capsys) -> None:
    run_args = _benchmark(checkout)
    missing_server = "++other.responses_api_agents.no_such_agent.entrypoint=app.py"
    code, findings = _validate(
        monkeypatch,
        capsys,
        run_args,
        missing_server,
        "--server-types",
        "custom_servers,responses_api_agents",
        "--json",
    )
    assert code == 0
    assert [(finding["kind"], finding["path"]) for finding in findings] == [
        ("unchecked", "custom_servers"),
        ("unchecked", "other.responses_api_agents.no_such_agent"),
    ]
    assert findings[0]["why"].startswith("Gym defines no server type 'custom_servers'")


def test_a_crash_exits_3_never_as_a_finding(checkout, monkeypatch, capsys) -> None:
    # Server discovery asserts that entrypoints are relative.
    absolute = "++other.responses_api_agents.demo_agent.entrypoint=/app.py"
    assert _validate(monkeypatch, capsys, _benchmark(checkout), absolute, *_SERVER_TYPES) == (3, [])


def test_job_time_values_need_the_flag(checkout, monkeypatch, capsys) -> None:
    # An unset ${oc.env:X} without a default, and a value only the job supplies (++model_call_capture_dir=...).
    run_args = _benchmark(checkout, domain="${oc.env:DEMO_DOMAIN}", capture_dir="${model_call_capture_dir}")
    (checkout / "job.yaml").write_text("model_call_capture_dir: ???\n")
    run_args = "+config_paths=[benchmark.yaml,job.yaml]"
    code, findings = _validate(monkeypatch, capsys, run_args, *_SERVER_TYPES)
    assert (code, [finding["kind"] for finding in findings]) == (2, ["uncomposed"])

    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    assert _validate(monkeypatch, capsys, run_args, *_SERVER_TYPES, "--allow-job-time-values") == (0, [])


def test_an_input_holding_the_callers_marker_is_not_reported(checkout, monkeypatch, capsys) -> None:
    run_args = [_benchmark(checkout), "++agent.responses_api_agents.demo_agent.max_turns=PREFLIGHT-PLACEHOLDER"]
    code, findings = _validate(monkeypatch, capsys, *run_args, *_SERVER_TYPES)
    assert (code, [finding["path"] for finding in findings]) == (
        1,
        ["agent.responses_api_agents.demo_agent.max_turns"],
    )

    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    marker = ["--ignore-input-containing", "PREFLIGHT-PLACEHOLDER"]
    assert _validate(monkeypatch, capsys, *run_args, *_SERVER_TYPES, *marker) == (0, [])


def test_the_help_lists_the_flags_a_caller_probes_for(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["gym", "env", "validate", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    out = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert all(flag in out for flag in ("--server-types", "--allow-job-time-values", "--ignore-input-containing"))


def test_a_value_that_does_not_resolve_is_reported_at_its_full_path(checkout, monkeypatch, capsys) -> None:
    """An int reaching oc.decode names its key and the resolver's message, not OmegaConf's raw report."""
    _benchmark(checkout, max_turns="${oc.decode:${turns}}")
    (checkout / "turns.yaml").write_text("turns: 3\n")
    run_args = "+config_paths=[benchmark.yaml,turns.yaml]"
    assert _validate(monkeypatch, capsys, run_args, *_SERVER_TYPES) == (
        2,
        [
            {
                "kind": "uncomposed",
                "path": "agent.responses_api_agents.demo_agent.max_turns",
                "why": "TypeError raised while resolving interpolation: `oc.decode` can only take strings or None "
                "as input, but `3` is of type int",
                "supplied": [],
            }
        ],
    )


_OSWORLD_AGENT_CONFIG = Path(__file__).parents[3] / "responses_api_agents/osworld_agent/configs/osworld_agent.yaml"


def _osworld_transport_version_line() -> str:
    config = yaml.safe_load(_OSWORLD_AGENT_CONFIG.read_text())
    [agent] = config["osworld_simple_agent"]["responses_api_agents"].values()
    return agent["exact_trace_transport_version"]


@pytest.mark.parametrize(
    "resolver",
    ["built-in", "stand-in returning its default as is"],
)
@pytest.mark.parametrize(("variable", "expected"), [(None, 3), ("2", 2), ("3", 3)])
def test_the_osworld_transport_version_resolves_to_an_int_under_any_oc_env(
    monkeypatch: pytest.MonkeyPatch, resolver: str, variable: str | None, expected: int
) -> None:
    from omegaconf import OmegaConf
    from omegaconf.resolvers.oc import env as oc_env

    if variable is None:
        monkeypatch.delenv("OSWORLD_EXACT_TRACE_TRANSPORT_VERSION", raising=False)
    else:
        monkeypatch.setenv("OSWORLD_EXACT_TRACE_TRANSPORT_VERSION", variable)
    stand_in = lambda key, default="stand-in": __import__("os").environ.get(key, default)  # noqa: E731
    OmegaConf.register_new_resolver("oc.env", oc_env if resolver == "built-in" else stand_in, replace=True)
    try:
        value = OmegaConf.create({"version": _osworld_transport_version_line()}).version
    finally:
        OmegaConf.register_new_resolver("oc.env", oc_env, replace=True)

    assert value == expected and type(value) is int
