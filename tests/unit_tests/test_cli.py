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
import json
import shutil
import sys
import tomllib
from importlib import import_module
from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch, raises

import nemo_gym.cli.env
import nemo_gym.global_config
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, PARENT_DIR
from nemo_gym.cli.env import (
    _FORCE_KILL_REAP_TIMEOUT_SEC,
    _GRACEFUL_SHUTDOWN_TIMEOUT_SEC,
    RunConfig,
    RunHelper,
    TestConfig,
    _resolve_server_dir,
    _select_shard,
    dump_config,
    init_resources_server,
    list_environments,
    pip_list,
    run,
    status,
    validate,
)
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config_types import ConfigError, NoServerInstancesError, ResourcesServerInstanceConfig
from nemo_gym.registry import EnvironmentEntry


class TestSelectShard:
    def test_no_sharding_returns_all(self) -> None:
        paths = [Path(f"resources_servers/s{i}") for i in range(5)]
        assert _select_shard(paths, shard_index=0, num_shards=1) == paths

    def test_round_robin_partition_is_complete_and_disjoint(self) -> None:
        paths = [Path(f"resources_servers/s{i:02d}") for i in range(10)]
        num_shards = 4
        shards = [_select_shard(paths, i, num_shards) for i in range(num_shards)]
        # Every module appears in exactly one shard, and the union is the full sorted set.
        flattened = [p for shard in shards for p in shard]
        assert sorted(flattened, key=str) == sorted(paths, key=str)
        assert len(flattened) == len(set(flattened)) == len(paths)
        # Round-robin stride: shard 0 gets indices 0,4,8 of the sorted list.
        assert shards[0] == [
            Path("resources_servers/s00"),
            Path("resources_servers/s04"),
            Path("resources_servers/s08"),
        ]

    def test_balanced_sizes(self) -> None:
        paths = [Path(f"resources_servers/s{i:02d}") for i in range(10)]
        sizes = sorted(len(_select_shard(paths, i, 4)) for i in range(4))
        # 10 across 4 shards -> sizes differ by at most 1.
        assert sizes[-1] - sizes[0] <= 1

    def test_shard_index_out_of_range_raises(self) -> None:
        paths = [Path("resources_servers/s0")]
        with raises(AssertionError):
            _select_shard(paths, shard_index=4, num_shards=4)


# TODO: Eventually we want to add more tests to ensure that the CLI flows do not break
class TestCLI:
    def test_sanity(self) -> None:
        RunConfig(entrypoint="", name="")

    def test_pyproject_scripts_are_importable(self) -> None:
        """Every console-script entry point must resolve to an importable callable."""
        pyproject_path = PARENT_DIR / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject_data = tomllib.load(f)

        for script_name, import_path in pyproject_data["project"]["scripts"].items():
            module, fn = import_path.split(":")
            target = getattr(import_module(module), fn)
            assert callable(target), f"{script_name} -> {import_path} is not callable"

    def test_init_resources_server_includes_domain(self) -> None:
        """Test that init_resources_server creates a config with the required domain field."""

        server_name = "test_cli_server"
        entrypoint = f"resources_servers/{server_name}"
        server_path = Path(entrypoint).resolve()

        # Clean up any existing test server directory
        if server_path.exists():
            shutil.rmtree(server_path)

        try:
            with MonkeyPatch.context() as mp:
                # Set up the global config to point to our test entrypoint
                mp.setattr(
                    nemo_gym.global_config,
                    "_GLOBAL_CONFIG_DICT",
                    OmegaConf.create({"entrypoint": entrypoint}),
                )

                # Run init_resources_server
                init_resources_server()

                # Verify the generated config file exists
                config_file = server_path / "configs" / f"{server_name}.yaml"
                assert config_file.exists(), f"Config file not created at {config_file}"

                # Load and verify the config
                config_dict = OmegaConf.load(config_file)

                # Check that the domain field is present in the resources server config
                resources_server_key = f"{server_name}_resources_server"
                assert resources_server_key in config_dict, f"Resources server key '{resources_server_key}' not found"

                resources_config = config_dict[resources_server_key]
                assert "resources_servers" in resources_config
                assert server_name in resources_config["resources_servers"]

                server_config = resources_config["resources_servers"][server_name]
                assert "domain" in server_config, "Domain field missing from resources server config"
                assert server_config["domain"] == "other", f"Expected domain 'other', got '{server_config['domain']}'"

                # Generated config ships `verified: false` so the add-verified-flag pre-commit hook
                # is a no-op and does not rewrite (and strip the comments from) the file on commit.
                assert server_config["verified"] is False

                # The generated config is documented with inline comments (friction #7).
                config_text = config_file.read_text()
                assert "# Resources server:" in config_text
                assert config_text.count("#") >= 10, "expected inline field documentation comments"

                # The add-verified-flag hook must NOT modify the generated config (would strip comments).
                from scripts.add_verified_flag import ensure_verified_flag

                assert ensure_verified_flag(config_file) is False
                assert config_file.read_text() == config_text, "verified-flag hook altered the generated config"

                # Verify that the config can be validated (this would have failed before the fix)
                full_config_dict = OmegaConf.create(
                    {
                        "name": resources_server_key,
                        "server_type_config_dict": config_dict[resources_server_key],
                        **OmegaConf.to_container(config_dict[resources_server_key]),
                    }
                )

                # This should not raise an assertion error about missing domain
                instance_config = ResourcesServerInstanceConfig.model_validate(full_config_dict)
                assert instance_config is not None

                # The generated config points users at the unified `source:` identifier, not the
                # deprecated gitlab_identifier/huggingface_identifier.
                assert "source:" in config_text
                assert "gitlab_identifier" not in config_text
                assert "huggingface_identifier" not in config_text
        finally:
            # Clean up the test server directory
            if server_path.exists():
                shutil.rmtree(server_path)

    def test_run_helper_prefers_cwd_server_over_install(self, tmp_path: Path) -> None:
        """ng_run should use a local CWD server dir instead of the installed one."""
        # Create a fake local server dir in tmp_path (simulates user's own resources_servers/)
        local_server = tmp_path / "resources_servers" / "my_server"
        local_server.mkdir(parents=True)
        (local_server / "requirements.txt").write_text("nemo-gym\n")

        with patch.object(Path, "cwd", return_value=tmp_path):
            _cwd_path = Path.cwd() / Path("resources_servers", "my_server")
            dir_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / Path("resources_servers", "my_server")

        assert dir_path == local_server

    def test_run_helper_falls_back_to_install_when_not_in_cwd(self, tmp_path: Path) -> None:
        """ng_run should fall back to PARENT_DIR when the server doesn't exist in CWD."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            _cwd_path = Path.cwd() / Path("resources_servers", "arc_agi")
            dir_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / Path("resources_servers", "arc_agi")

        assert dir_path == PARENT_DIR / "resources_servers" / "arc_agi"


class TestResolveServerDir:
    """`_resolve_server_dir` resolves a relative server dir against cwd first, then the install root."""

    def test_prefers_local_server_in_cwd(self, tmp_path: Path, monkeypatch) -> None:
        local = tmp_path / "resources_servers" / "my_server"
        local.mkdir(parents=True)
        (local / "requirements.txt").write_text("nemo-gym\n")
        monkeypatch.chdir(tmp_path)
        assert _resolve_server_dir(Path("resources_servers/my_server")) == local

    def test_falls_back_to_install_root(self, tmp_path: Path, monkeypatch) -> None:
        # Empty cwd (no local server) -> the built-in resolves under the install root.
        monkeypatch.chdir(tmp_path)
        rel = Path("resources_servers/arc_agi")
        assert _resolve_server_dir(rel) == PARENT_DIR / rel

    def test_test_config_resolved_dir_path_uses_install_root(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = TestConfig(entrypoint="resources_servers/arc_agi")
        assert cfg.resolved_dir_path == PARENT_DIR / "resources_servers" / "arc_agi"


class TestRunHelperShutdownReap:
    """RunHelper.shutdown must reap every server subprocess on every exit path."""

    def _make_runner_with_processes(self, processes: dict) -> RunHelper:
        runner = RunHelper()
        runner._processes = processes
        runner._head_server = MagicMock()
        runner._head_server_thread = MagicMock()
        return runner

    def test_kill_is_followed_by_reap_wait(self) -> None:
        good = MagicMock()
        good.wait.return_value = 0
        bad = MagicMock()
        bad.wait.side_effect = [TimeoutExpired(cmd="bad", timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC), 0]

        runner = self._make_runner_with_processes({"good_server": good, "bad_server": bad})
        runner.shutdown()

        good.send_signal.assert_called_once()
        bad.send_signal.assert_called_once()
        good.kill.assert_not_called()
        bad.kill.assert_called_once()
        assert good.wait.call_count == 1
        assert bad.wait.call_count == 2
        assert runner._processes == {}

    def test_unreaped_server_after_sigkill_is_warned(self, capsys) -> None:
        zombie = MagicMock()
        zombie.wait.side_effect = TimeoutExpired(cmd="zombie", timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC)

        runner = self._make_runner_with_processes({"zombie_server": zombie})
        runner.shutdown()

        zombie.kill.assert_called_once()
        assert zombie.wait.call_count == 2
        out: str = capsys.readouterr().out
        assert "zombie_server" in out
        assert f"{_GRACEFUL_SHUTDOWN_TIMEOUT_SEC}s timeout" in out
        assert f"{_FORCE_KILL_REAP_TIMEOUT_SEC}s after SIGKILL" in out

    def test_shutdown_message_matches_actual_timeout(self, capsys) -> None:
        bad = MagicMock()
        bad.wait.side_effect = [TimeoutExpired(cmd="bad", timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC), 0]
        runner = self._make_runner_with_processes({"bad": bad})
        runner.shutdown()

        out: str = capsys.readouterr().out
        assert f"{_GRACEFUL_SHUTDOWN_TIMEOUT_SEC}s timeout" in out

    def test_graceful_termination_does_not_kill(self) -> None:
        a = MagicMock()
        a.wait.return_value = 0
        b = MagicMock()
        b.wait.return_value = 0
        runner = self._make_runner_with_processes({"a": a, "b": b})
        runner.shutdown()

        a.kill.assert_not_called()
        b.kill.assert_not_called()
        assert a.wait.call_count == 1
        assert b.wait.call_count == 1
        assert runner._processes == {}


class TestExitCleanlyOnConfigError:
    """The CLI decorator turns ConfigError into a clean message + non-zero exit, not a traceback."""

    # Every CLI entrypoint that must carry the clean-error contract (each calls get_global_config_dict
    # and wears @exit_cleanly_on_config_error). Keep this list in sync when decorating new commands —
    # it's the single place that asserts each one actually exits cleanly on a config error.
    DECORATED_COMMANDS = [run, validate, dump_config, status, pip_list]

    def test_config_error_becomes_clean_exit(self) -> None:
        @exit_cleanly_on_config_error
        def boom():
            raise NoServerInstancesError("nothing to run")

        with raises(SystemExit) as exc_info:
            boom()
        assert exc_info.value.code == 1

    def test_non_config_error_propagates(self) -> None:
        # The decorator must catch ONLY ConfigError. A non-ConfigError propagates unchanged — same
        # type and message, as a normal traceback — and is NOT converted to SystemExit (contrast
        # with test_config_error_becomes_clean_exit); requiring RuntimeError here, not SystemExit,
        # is what asserts the error type is left untouched.
        @exit_cleanly_on_config_error
        def boom():
            raise RuntimeError("unexpected")

        with raises(RuntimeError, match="unexpected"):
            boom()

    def test_success_passes_through(self) -> None:
        @exit_cleanly_on_config_error
        def ok():
            return 42

        assert ok() == 42

    def test_config_error_base_catches_subclasses(self) -> None:
        assert issubclass(NoServerInstancesError, ConfigError)

    @pytest.mark.parametrize("command", DECORATED_COMMANDS)
    def test_command_exits_cleanly_on_config_error(self, monkeypatch: MonkeyPatch, command) -> None:
        # A ConfigError surfacing from config parsing becomes exit(1), not a traceback. Without the
        # decorator the ConfigError would propagate as itself, so asserting SystemExit is exactly what
        # proves each command is decorated.
        def _raise(*args, **kwargs):
            raise NoServerInstancesError("nothing configured to run")

        monkeypatch.setattr(nemo_gym.cli.env, "get_global_config_dict", _raise)

        with raises(SystemExit) as exc_info:
            command()
        assert exc_info.value.code == 1

    @pytest.mark.parametrize("command", DECORATED_COMMANDS)
    def test_command_propagates_non_config_error(self, monkeypatch: MonkeyPatch, command) -> None:
        # The decorator must only swallow ConfigError; an unexpected error must surface unchanged.
        def _raise(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(nemo_gym.cli.env, "get_global_config_dict", _raise)

        with raises(RuntimeError, match="unexpected"):
            command()


class TestValidate:
    def _validate_config(self, monkeypatch: MonkeyPatch, tmp_path: Path, config_yaml: str) -> None:
        """Run the real `validate()` against a config file (no mocking of the parse path)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)
        # chdir to a clean dir so a repo-local env.yaml isn't picked up; clear the parse cache.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["gym", f"+config_paths=[{config_file}]"])
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        validate()

    def test_valid_config_passes(self, monkeypatch: MonkeyPatch, tmp_path, capsys) -> None:
        # A well-formed config (real parse, real checks) -> "valid".
        self._validate_config(
            monkeypatch,
            tmp_path,
            "my_server:\n  resources_servers:\n    my_server:\n      entrypoint: app.py\n      domain: other\n",
        )
        assert "valid" in capsys.readouterr().out.lower()

    def test_unknown_cross_reference_exits_nonzero(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        # An agent referencing a resources server that isn't defined -> ServerRefNotFoundError ->
        # clean exit(1) (real cross-reference validation, not a mock).
        bad = (
            "my_agent:\n"
            "  responses_api_agents:\n"
            "    a:\n"
            "      entrypoint: app.py\n"
            "      resources_server:\n        type: resources_servers\n        name: does_not_exist\n"
            "      model_server:\n        type: responses_api_models\n        name: policy_model\n"
        )
        with raises(SystemExit) as exc_info:
            self._validate_config(monkeypatch, tmp_path, bad)
        assert exc_info.value.code == 1


class TestListEnvironments:
    _ALPHA = EnvironmentEntry(
        name="alpha",
        config_path=Path("environments/alpha/config.yaml"),
        path=Path("environments/alpha"),
        description="Alpha env",
        domain="agent",
    )

    def test_lists_discovered_environments(self, monkeypatch: MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(nemo_gym.cli.env, "get_global_config_dict", lambda **k: OmegaConf.create({}))
        monkeypatch.setattr(nemo_gym.cli.env, "discover_environments", lambda *a, **k: {"alpha": self._ALPHA})

        list_environments()

        out = capsys.readouterr().out
        assert "alpha" in out
        assert "agent" in out

    def test_no_environments(self, monkeypatch: MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(nemo_gym.cli.env, "get_global_config_dict", lambda **k: OmegaConf.create({}))
        monkeypatch.setattr(nemo_gym.cli.env, "discover_environments", lambda *a, **k: {})

        list_environments()

        assert "No environments found" in capsys.readouterr().out

    def test_json_output(self, monkeypatch: MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(nemo_gym.cli.env, "get_global_config_dict", lambda **k: OmegaConf.create({"json": True}))
        monkeypatch.setattr(nemo_gym.cli.env, "discover_environments", lambda *a, **k: {"alpha": self._ALPHA})

        list_environments()

        assert json.loads(capsys.readouterr().out) == [
            {"name": "alpha", "domain": "agent", "description": "Alpha env"}
        ]

    def test_query_filters_environments(self, monkeypatch: MonkeyPatch, capsys) -> None:
        # `gym search environments <query>` reuses this command via the `query` config key
        # (matches name + domain + description).
        beta = EnvironmentEntry(
            name="beta",
            config_path=Path("environments/beta/config.yaml"),
            path=Path("environments/beta"),
            description="Beta env",
            domain="math",
        )
        monkeypatch.setattr(
            nemo_gym.cli.env, "get_global_config_dict", lambda **k: OmegaConf.create({"query": "alpha"})
        )
        monkeypatch.setattr(
            nemo_gym.cli.env, "discover_environments", lambda *a, **k: {"alpha": self._ALPHA, "beta": beta}
        )

        list_environments()

        out = capsys.readouterr().out
        assert "Environments matching 'alpha'" in out
        assert "agent" in out  # alpha's domain -> its row was rendered
        assert "beta" not in out and "math" not in out  # beta and its domain filtered out

    def test_query_matches_description(self, monkeypatch: MonkeyPatch, capsys) -> None:
        # "Robotics" only appears in gamma's description, not its name or domain.
        gamma = EnvironmentEntry(
            name="gamma",
            config_path=Path("environments/gamma/config.yaml"),
            path=Path("environments/gamma"),
            description="Robotics manipulation tasks",
            domain="control",
        )
        monkeypatch.setattr(
            nemo_gym.cli.env, "get_global_config_dict", lambda **k: OmegaConf.create({"query": "Robotics"})
        )
        monkeypatch.setattr(
            nemo_gym.cli.env, "discover_environments", lambda *a, **k: {"alpha": self._ALPHA, "gamma": gamma}
        )

        monkeypatch.setattr(
            nemo_gym.cli.env,
            "read_environment_details",
            lambda cfg: {
                "domain": "agent",
                "description": "Alpha env",
                "value": "Some value",
                "resources_servers": ["alpha_rs"],
                "agent": "simple_agent",
                "datasets": ["train", "example"],
            },
        )

        list_environments()

        out = capsys.readouterr().out

        assert "gamma" in out
        assert "alpha" not in out

    def test_inspect_environment_by_name(self, monkeypatch: MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(
            nemo_gym.cli.env, "get_global_config_dict", lambda **k: OmegaConf.create({"component_name": "alpha"})
        )
        monkeypatch.setattr(nemo_gym.cli.env, "discover_environments", lambda *a, **k: {"alpha": self._ALPHA})
        monkeypatch.setattr(
            nemo_gym.cli.env,
            "read_environment_details",
            lambda cfg: {
                "domain": "agent",
                "description": "Alpha env",
                "value": "Some value",
                "resources_servers": ["alpha_rs"],
                "agent": "simple_agent",
                "datasets": ["train", "example"],
            },
        )

        list_environments()

        out = capsys.readouterr().out

        assert "The alpha environment (domain: agent)" in out
        assert "Value: Some value" in out
        assert "resources servers: alpha_rs" in out and "agent: simple_agent" in out
        assert "datasets: train, example" in out
        assert "gym env start --environment alpha --model-type vllm_model" in out

    def _mock_inspect_alpha(self, monkeypatch: MonkeyPatch, config: dict) -> None:
        monkeypatch.setattr(
            nemo_gym.cli.env,
            "get_global_config_dict",
            lambda **k: OmegaConf.create({"component_name": "alpha", **config}),
        )
        monkeypatch.setattr(nemo_gym.cli.env, "discover_environments", lambda *a, **k: {"alpha": self._ALPHA})
        monkeypatch.setattr(
            nemo_gym.cli.env,
            "read_environment_details",
            lambda cfg: {
                "domain": "agent",
                "description": "Alpha env",
                "value": "Some value",
                "resources_servers": [],
                "agent": None,
                "datasets": [],
            },
        )

    def test_inspect_folds_value_into_description(self, monkeypatch: MonkeyPatch, capsys) -> None:
        # `value` is not a separate field: it is appended to the description, not surfaced on its own.
        self._mock_inspect_alpha(monkeypatch, {"json": True})

        list_environments()

        payload = json.loads(capsys.readouterr().out)
        assert payload["description"] == "Alpha env\nValue: Some value"
        assert "value" not in payload and "value" not in payload["details"]

    def test_inspect_json_output(self, monkeypatch: MonkeyPatch, capsys) -> None:
        self._mock_inspect_alpha(monkeypatch, {"json": True})

        list_environments()

        assert json.loads(capsys.readouterr().out) == {
            "name": "alpha",
            "type": "environment",
            "domain": "agent",
            "description": "Alpha env\nValue: Some value",
            "details": {"config": str(self._ALPHA.config_path.resolve())},
            "usage_example": "gym env start --environment alpha --model-type vllm_model",
        }

    def test_inspect_unknown_environment_exits(self, monkeypatch: MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(
            nemo_gym.cli.env, "get_global_config_dict", lambda **k: OmegaConf.create({"component_name": "alfa"})
        )
        monkeypatch.setattr(nemo_gym.cli.env, "discover_environments", lambda *a, **k: {"alpha": self._ALPHA})

        with raises(SystemExit):
            list_environments()

        out = capsys.readouterr().out
        assert "Unknown environment 'alfa'" in out and "alpha" in out

    def test_inspect_shows_absolute_config_path(self, monkeypatch: MonkeyPatch, capsys, tmp_path: Path) -> None:
        # Real discovery (via an extra root): the config line must be the config's absolute path.
        cfg = tmp_path / "environments" / "my_env" / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("my_env:\n  resources_servers:\n    my_env:\n      domain: agent\n      description: D\n")
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
        monkeypatch.setattr(
            nemo_gym.cli.env,
            "get_global_config_dict",
            lambda **k: OmegaConf.create({"component_name": "my_env"}),
        )

        list_environments()

        assert f"config: {cfg.resolve()}" in capsys.readouterr().out
