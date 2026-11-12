# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`gym eval prepare/run --benchmark` for a benchmark config that names a Harbor target."""

import argparse
import json
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.cli.eval as cli_eval
import nemo_gym.cli.main as cli_main
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.config.global_config import GlobalConfigDictParser
from nemo_gym.data.tasks.harbor.prepare import PreparedTaskset, build_run, harbor_benchmark
from nemo_gym.data.tasks.harbor.task import load_task
from nemo_gym.environment.assets import asset_config_path
from tests.unit_tests.data.tasks.harbor.harbor_build_helpers import PULL_DOCKERFILE, write_task


MINISWE_CONFIG = "responses_api_agents/mini_swe_agent/configs/mini_swe_agent_sandboxed.yaml"


@pytest.fixture(autouse=True)
def _isolate_extra_roots_env(monkeypatch: MonkeyPatch):
    monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)


def _harbor_config(**extra) -> dict:
    return {
        "harbor_target": "terminal_bench_4",
        "config_paths": [MINISWE_CONFIG],
        "mini_swe_agent": {"responses_api_agents": {"mini_swe_agent": {"step_limit": 500}}},
        **extra,
    }


def _benchmark(root: Path, name: str, config: dict) -> Path:
    path = root / "benchmarks" / name / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _main(monkeypatch: MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["gym", *argv])
    cli_main.main()


class TestHarborBenchmarkConfig:
    def test_names_the_target_and_the_benchmarks_own_harness(self, tmp_path: Path) -> None:
        path = _benchmark(tmp_path, "tb", _harbor_config())
        benchmark = harbor_benchmark(path)
        assert benchmark.target == "terminal_bench_4"
        assert benchmark.config_path == path.resolve()
        assert (benchmark.agent.config_path, benchmark.agent.instance_name, benchmark.agent.impl_name) == (
            path.resolve(),
            "mini_swe_agent",
            "mini_swe_agent",
        )
        assert benchmark.agent.runs_in_sandbox

    def test_a_config_without_a_target_composes_its_own_servers(self, tmp_path: Path) -> None:
        config = _harbor_config()
        config.pop("harbor_target")
        assert harbor_benchmark(_benchmark(tmp_path, "tb", config)) is None

    def test_the_benchmarks_own_settings_ride_along(self, tmp_path: Path) -> None:
        overlay = {"environment_servers": {"harbor": {"default_episode_timeout_seconds": 54000}}}
        config = _harbor_config(harbor_environment_server=overlay, builds={"enabled": False})
        benchmark = harbor_benchmark(_benchmark(tmp_path, "tb", config))
        assert benchmark.agent.benchmark_settings == {
            "harbor_environment_server": overlay,
            "builds": {"enabled": False},
        }

    @pytest.mark.parametrize(
        "extra,message",
        [
            ({"harbor_target": " "}, "must name a Harbor task folder"),
            (
                {"second": {"responses_api_agents": {"simple_agent": {}}}},
                "exactly one responses_api_agents instance",
            ),
            ({"verifier": {"resources_servers": {"demo": {}}}}, "cannot define verifier"),
            ({"relay": {"environment_servers": {"legacy_agent": {}}}}, "cannot define relay"),
            ({"harbor_environment_server": "54000"}, "must be a mapping of the server's settings"),
            # A named instance inheriting the shipped one leaves two agent instances once the include is loaded.
            (
                {"tb_agent": {"_inherit_from": "mini_swe_agent", "responses_api_agents": {"mini_swe_agent": {}}}},
                "exactly one responses_api_agents instance",
            ),
        ],
    )
    def test_refuses_what_the_run_could_not_honor(self, tmp_path: Path, extra: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            harbor_benchmark(_benchmark(tmp_path, "tb", _harbor_config(**extra)))

    def test_refuses_an_inherited_agent_instance(self, tmp_path: Path) -> None:
        """The run's agent inherits the benchmark's instance; a second, unresolved inheritance would not bind."""
        config = {
            "harbor_target": "terminal_bench_4",
            "tb_agent": {"_inherit_from": "elsewhere", "responses_api_agents": {"mini_swe_agent": {"step_limit": 1}}},
        }
        with pytest.raises(ValueError, match="sets `_inherit_from`; define it directly"):
            harbor_benchmark(_benchmark(tmp_path, "tb", config))

    def test_refuses_a_server_an_included_config_defines(self, tmp_path: Path) -> None:
        extra = tmp_path / "extra_verifier.yaml"
        extra.write_text(
            yaml.safe_dump(
                {"extra_verifier": {"resources_servers": {"example_single_tool_call": {"entrypoint": "app.py"}}}}
            )
        )
        config = _harbor_config()
        config["config_paths"] = [MINISWE_CONFIG, str(extra)]
        with pytest.raises(ValueError, match="configs it includes cannot define extra_verifier"):
            harbor_benchmark(_benchmark(tmp_path, "tb", config))

    def test_included_settings_ride_along_but_the_harness_config_stays_with_its_harness(self, tmp_path: Path) -> None:
        overlay = {"environment_servers": {"harbor": {"default_episode_timeout_seconds": "${episode_bound}"}}}
        settings = tmp_path / "settings.yaml"
        settings.write_text(yaml.safe_dump({"harbor_environment_server": overlay, "episode_bound": 54000}))
        # A config that defines the agent instance is the harness's: its other keys are not the benchmark's.
        harness = tmp_path / "harness.yaml"
        harness.write_text(
            yaml.safe_dump({"config_paths": [MINISWE_CONFIG], "mini_swe_agent": {}, "harness_only": {"a": 1}})
        )
        config = _harbor_config(builds={"enabled": False})
        config["config_paths"] = [str(harness), str(settings)]
        benchmark = harbor_benchmark(_benchmark(tmp_path, "tb", config))
        assert benchmark.agent.benchmark_settings == {
            "harbor_environment_server": overlay,
            "episode_bound": 54000,
            "builds": {"enabled": False},
        }

    def test_a_target_only_an_include_sets_is_refused(self, tmp_path: Path) -> None:
        """A variant that includes a Harbor-backed preset would otherwise fall through to the composed-server path."""
        base = _benchmark(tmp_path, "tb", _harbor_config())
        variant = _benchmark(tmp_path, "tb_variant", {"config_paths": [str(base)], "mini_swe_agent": {}})
        with pytest.raises(ValueError, match="set `harbor_target` in the benchmark file itself"):
            harbor_benchmark(variant)

    def test_the_agent_may_come_from_an_include_alone(self, tmp_path: Path) -> None:
        config = _harbor_config()
        config.pop("mini_swe_agent")
        benchmark = harbor_benchmark(_benchmark(tmp_path, "tb", config))
        assert (benchmark.agent.instance_name, benchmark.agent.impl_name) == ("mini_swe_agent", "mini_swe_agent")


class TestEvalRunDelegation:
    def test_run_hands_the_target_and_the_harness_to_run_target(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        path = _benchmark(tmp_path, "tb", _harbor_config())
        captured = []
        monkeypatch.setattr(
            cli_eval, "run_target", lambda args, overrides, **kwargs: captured.append((overrides, kwargs))
        )
        _main(monkeypatch, ["eval", "run", "--search-dir", str(tmp_path), "--benchmark", "tb", "+limit=3"])
        ((overrides, kwargs),) = captured
        assert kwargs["benchmark"].config_path == path.resolve()
        assert kwargs["benchmark"].target == "terminal_bench_4"
        assert "+limit=3" in overrides

    def test_task_selection_reaches_a_harbor_backed_benchmark(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        _benchmark(tmp_path, "tb", _harbor_config())
        captured = []
        monkeypatch.setattr(cli_eval, "run_target", lambda args, overrides, **kwargs: captured.append(args.only_tasks))
        _main(monkeypatch, ["eval", "run", "--search-dir", str(tmp_path), "--benchmark", "tb", "--only-tasks", "a,b"])
        assert captured == ["a,b"]

    def test_a_benchmark_that_composes_its_own_servers_is_unchanged(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        config = _harbor_config()
        config.pop("harbor_target")
        path = _benchmark(tmp_path, "plain", config)
        monkeypatch.setattr(cli_eval, "run_target", lambda *args, **kwargs: pytest.fail("not a Harbor benchmark"))
        dispatched = []
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: dispatched.append((target, overrides)))
        _main(monkeypatch, ["eval", "run", "--search-dir", str(tmp_path), "--benchmark", "plain"])
        ((target, overrides),) = dispatched
        assert target == "nemo_gym.cli.eval:e2e_rollout_collection"
        assert overrides == [f"+config_paths=[{path.resolve()}]"]

    def test_a_harbor_backed_benchmark_runs_alone(self, monkeypatch: MonkeyPatch, tmp_path, capsys) -> None:
        _benchmark(tmp_path, "tb", _harbor_config())
        plain = _harbor_config()
        plain.pop("harbor_target")
        _benchmark(tmp_path, "plain", plain)
        monkeypatch.setattr(cli_eval, "run_target", lambda *args, **kwargs: pytest.fail("must be refused"))
        with pytest.raises(SystemExit):
            _main(
                monkeypatch,
                ["eval", "run", "--search-dir", str(tmp_path), "--benchmark", "plain", "--benchmark", "tb"],
            )
        assert "select it on its own" in capsys.readouterr().err

    @pytest.mark.parametrize("command", ["run", "prepare"])
    def test_an_included_extra_server_is_refused_before_preparation(
        self, monkeypatch: MonkeyPatch, tmp_path, capsys, command
    ) -> None:
        extra = tmp_path / "extra_verifier.yaml"
        extra.write_text(yaml.safe_dump({"extra_verifier": {"resources_servers": {"demo": {"entrypoint": "app.py"}}}}))
        config = _harbor_config()
        config["config_paths"] = [MINISWE_CONFIG, str(extra)]
        _benchmark(tmp_path, "tb", config)
        monkeypatch.setattr(cli_eval, "run_target", lambda *args, **kwargs: pytest.fail("must be refused"))
        monkeypatch.setattr(cli_eval, "prepare_harbor_benchmark", lambda *args: pytest.fail("must be refused"))
        argv = ["eval", command, "--search-dir", str(tmp_path), "--benchmark", "tb"]
        if command == "run":
            argv += ["--agent-type", "hermes_agent"]
        with pytest.raises(SystemExit):
            _main(monkeypatch, argv)
        assert "cannot define extra_verifier" in capsys.readouterr().err

    @pytest.mark.parametrize("command", ["run", "prepare"])
    @pytest.mark.parametrize("where", ["benchmark", "include"])
    def test_malformed_benchmark_yaml_is_a_usage_error(
        self, monkeypatch: MonkeyPatch, tmp_path, capsys, command, where
    ):
        path = tmp_path / "benchmarks" / "bad" / "config.yaml"
        path.parent.mkdir(parents=True)
        if where == "benchmark":
            path.write_text("config_paths: [\n")
        else:
            include = tmp_path / "broken.yaml"
            include.write_text("a: [\n")
            path.write_text(yaml.safe_dump(_harbor_config(config_paths=[MINISWE_CONFIG, str(include)])))
        with pytest.raises(SystemExit) as raised:
            _main(monkeypatch, ["eval", command, "--search-dir", str(tmp_path), "--benchmark", "bad"])
        assert raised.value.code != 0
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "expected" in err or "while parsing" in err or "broken.yaml" in err

    @pytest.mark.parametrize("command", ["run", "prepare"])
    def test_a_variant_that_only_includes_the_target_is_a_usage_error(
        self, monkeypatch: MonkeyPatch, tmp_path, capsys, command
    ) -> None:
        base = _benchmark(tmp_path, "tb", _harbor_config())
        _benchmark(tmp_path, "tb_variant", {"config_paths": [str(base)], "mini_swe_agent": {}})
        monkeypatch.setattr(cli_main, "dispatch", lambda *args: pytest.fail("must be refused"))
        with pytest.raises(SystemExit):
            _main(monkeypatch, ["eval", command, "--search-dir", str(tmp_path), "--benchmark", "tb_variant"])
        assert "set `harbor_target` in the benchmark file itself" in capsys.readouterr().err

    @pytest.mark.parametrize("no_serve", [False, True])
    def test_a_harbor_backed_config_is_refused_by_plain_collection(
        self, monkeypatch: MonkeyPatch, tmp_path, capsys, no_serve
    ) -> None:
        """`gym eval run --config <harbor-backed preset>` would start its bare agent with no Harbor server or rows."""
        path = _benchmark(tmp_path, "tb", _harbor_config())
        monkeypatch.setattr(cli_main, "dispatch", lambda *args: pytest.fail("must be refused"))
        with pytest.raises(SystemExit):
            _main(monkeypatch, ["eval", "run", "--config", str(path), *(["--no-serve"] if no_serve else [])])
        err = capsys.readouterr().err
        assert "sets `harbor_target` ('terminal_bench_4')" in err and "--benchmark <name>" in err


class _Stop(Exception):
    pass


class TestRunTargetWithABenchmark:
    """`run_target` with a Harbor-backed benchmark, stopped at preparation."""

    def _run(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, config: dict | None = None, extra: list[str] = (), **flags
    ) -> tuple[dict, argparse.Namespace]:
        path = _benchmark(tmp_path, "tb", config or _harbor_config())
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
        seen: dict = {}

        def fake_prepare(target, **kwargs):
            seen.update(target=target, **kwargs)
            raise _Stop

        monkeypatch.setattr(cli_eval, "prepare_target", fake_prepare)
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            **{"_parser": parser, "target": None, "sandbox": None, "agent": None, "agent_type": None, **flags}
        )
        loaded = [f"+config_paths=[{path.resolve()}]"]
        if args.agent_type:
            loaded.append(f"+config_paths=[{asset_config_path('agent-type', args.agent_type)}]")
        with pytest.raises(_Stop):
            cli_eval.run_target(args, [*loaded, *extra, "+limit=3"], benchmark=harbor_benchmark(path))
        return seen, args

    def test_the_benchmarks_harness_runs_by_default(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        seen, args = self._run(monkeypatch, tmp_path)
        assert seen["target"] == args.target == "terminal_bench_4"
        assert seen["agent"] == "mini_swe_agent"
        assert seen["rows_name"] == "tasks_mini_swe_agent.jsonl"

    def test_agent_type_swaps_the_harness(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        seen, _ = self._run(monkeypatch, tmp_path, agent_type="hermes_agent")
        assert seen["agent"] == "hermes_agent"
        assert seen["rows_name"] == "tasks_hermes_agent.jsonl"

    @pytest.mark.parametrize("agent_type", [None, "hermes_agent", "harbor_oracle_agent"])
    def test_preparation_reads_the_benchmarks_settings(self, monkeypatch: MonkeyPatch, tmp_path, agent_type) -> None:
        """Preparation runs before the run config exists: it reads the preset's `builds:` block itself, also for the
        oracle, which validates through `validate_target`."""
        seen, _ = self._run(
            monkeypatch, tmp_path, config=_harbor_config(builds={"enabled": False}), agent_type=agent_type
        )
        assert seen["build"].enabled is False

    @pytest.mark.parametrize("agent_type", [None, "hermes_agent"])
    def test_an_explicit_overlay_still_wins_at_preparation(
        self, monkeypatch: MonkeyPatch, tmp_path, agent_type
    ) -> None:
        overlay = tmp_path / "deployment.yaml"
        overlay.write_text(yaml.safe_dump({"builds": {"enabled": True}}))
        seen, _ = self._run(
            monkeypatch,
            tmp_path,
            config=_harbor_config(builds={"enabled": False}),
            extra=[f"+config_paths=[{overlay}]"],
            agent_type=agent_type,
        )
        assert seen["build"].enabled is True
        seen, _ = self._run(
            monkeypatch,
            tmp_path / "cli",
            config=_harbor_config(builds={"enabled": False}),
            extra=["++builds.enabled=true"],
            agent_type=agent_type,
        )
        assert seen["build"].enabled is True

    def test_agent_is_refused_rather_than_ignored(self, monkeypatch: MonkeyPatch, tmp_path, capsys) -> None:
        path = _benchmark(tmp_path, "tb", _harbor_config())
        monkeypatch.setattr(cli_eval, "prepare_target", lambda *args, **kwargs: pytest.fail("must be refused"))
        args = argparse.Namespace(
            _parser=argparse.ArgumentParser(), target=None, sandbox=None, agent="hermes_agent", agent_type=None
        )
        with pytest.raises(SystemExit):
            cli_eval.run_target(args, [], benchmark=harbor_benchmark(path))
        assert "pass `--agent-type <harness>`" in capsys.readouterr().err


@pytest.mark.parametrize("agent_type", [None, "hermes_agent"])
def test_the_configs_chosen_for_the_benchmark_leave_the_overrides(tmp_path: Path, monkeypatch, agent_type) -> None:
    """`build_run` loads the selected agent's config; the CLI's own copies would add a second agent instance."""
    path = _benchmark(tmp_path, "tb", _harbor_config())
    other = tmp_path / "overlay.yaml"
    loaded = [f"+config_paths=[{path},{other}]"]
    if agent_type:
        loaded.append(f"+config_paths=[{asset_config_path('agent-type', agent_type)}]")
        loaded.append("+_selected_langgraph_config=x")
    args = argparse.Namespace(target=None, agent=None, agent_type=agent_type)
    agent, overrides, preparation = cli_eval._benchmark_agent(args, [*loaded, "+limit=3"], harbor_benchmark(path))
    assert agent.impl_name == (agent_type or "mini_swe_agent")
    assert overrides == [f"+config_paths=[{other}]", "+limit=3"]
    # Preparation loads the benchmark below the overlay, so the overlay wins there too.
    assert preparation == [f"+config_paths=[{json.dumps(str(path.resolve()))},{other}]", "+limit=3"]


def _merged_run_config(tmp_path: Path, config: dict, agent_type: str | None, overrides: list[str] = ()):
    """The run's config files for ``config``, merged as Gym merges them, and the generated server's name."""
    path = _benchmark(tmp_path, "tb", config)
    args = argparse.Namespace(target=None, agent=None, agent_type=agent_type)
    agent, _, _ = cli_eval._benchmark_agent(args, [], harbor_benchmark(path))
    folder = tmp_path / "ds"
    tasks = [load_task(write_task(folder / "a", dockerfile=PULL_DOCKERFILE))]
    prepared = PreparedTaskset("ds", folder, tasks, tmp_path / "out" / "tasks.jsonl", tmp_path / "out")
    prepared.output_dir.mkdir(parents=True)

    run_config, tokens = build_run(prepared, agent, sandbox=None, overrides=list(overrides))

    paths = [
        json.loads(entry) if entry.startswith('"') else entry
        for token in tokens
        if token.startswith("+config_paths=[")
        for entry in token[len("+config_paths=[") : -1].split(",")
    ]
    _, configs = GlobalConfigDictParser().load_extra_config_paths(paths)
    merged = OmegaConf.merge(*configs)
    written = yaml.safe_load(run_config.read_text())
    (server,) = [name for name, value in written.items() if isinstance(value, dict) and "environment_servers" in value]
    GlobalConfigDictParser()._recursively_swap_keys(merged)
    return merged, server


@pytest.mark.parametrize("agent_type", [None, "hermes_agent"])
def test_the_benchmarks_server_settings_win_in_the_runs_merged_config(tmp_path: Path, agent_type) -> None:
    """The run's config files, merged as Gym merges them, carry the benchmark's episode bound over the server's own
    default, whether the benchmark's harness runs or `--agent-type` swaps in another."""
    overlay = {"environment_servers": {"harbor": {"default_episode_timeout_seconds": 54000}}}
    merged, server = _merged_run_config(tmp_path, _harbor_config(harbor_environment_server=overlay), agent_type)
    assert merged[server].environment_servers.harbor.default_episode_timeout_seconds == 54000


@pytest.mark.parametrize("agent_type", [None, "hermes_agent"])
def test_an_interpolated_server_setting_keeps_its_input(tmp_path: Path, agent_type) -> None:
    """A swapped-in harness drops the benchmark's agent, not the top-level keys its server settings interpolate."""
    overlay = {"environment_servers": {"harbor": {"default_episode_timeout_seconds": "${episode_bound}"}}}
    config = _harbor_config(episode_bound=54000, harbor_environment_server=overlay)
    merged, server = _merged_run_config(tmp_path, config, agent_type)
    assert merged[server].environment_servers.harbor.default_episode_timeout_seconds == 54000
    later = tmp_path / "later.yaml"
    later.write_text(yaml.safe_dump({"episode_bound": 60000}))
    merged, server = _merged_run_config(tmp_path / "overlaid", config, agent_type, [f"+config_paths=[{later}]"])
    assert merged[server].environment_servers.harbor.default_episode_timeout_seconds == 60000


@pytest.mark.parametrize("agent_type", [None, "hermes_agent"])
def test_an_included_server_setting_wins_in_the_runs_merged_config(tmp_path: Path, agent_type) -> None:
    """The benchmark's server settings win over the server's default when an included config sets them too, and
    the keys they interpolate travel with a swapped-in harness."""
    overlay = {"environment_servers": {"harbor": {"default_episode_timeout_seconds": "${episode_bound}"}}}
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({"harbor_environment_server": overlay, "episode_bound": 54000}))
    config = _harbor_config(config_paths=[MINISWE_CONFIG, str(settings)])
    merged, server = _merged_run_config(tmp_path, config, agent_type)
    assert merged[server].environment_servers.harbor.default_episode_timeout_seconds == 54000


class TestEvalPrepareDelegation:
    def test_prepare_writes_the_targets_rows(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        _benchmark(tmp_path, "tb", _harbor_config())
        captured = []
        monkeypatch.setattr(
            cli_eval,
            "prepare_harbor_benchmark",
            lambda args, overrides, benchmark: captured.append(benchmark.target),
        )
        monkeypatch.setattr(cli_main, "dispatch", lambda *args: pytest.fail("not the dataset preparation"))
        _main(monkeypatch, ["eval", "prepare", "--search-dir", str(tmp_path), "--benchmark", "tb"])
        assert captured == ["terminal_bench_4"]

    def test_other_benchmarks_prepare_their_datasets_as_before(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        config = _harbor_config()
        config.pop("harbor_target")
        path = _benchmark(tmp_path, "plain", config)
        dispatched = []
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: dispatched.append((target, overrides)))
        _main(monkeypatch, ["eval", "prepare", "--search-dir", str(tmp_path), "--benchmark", "plain"])
        assert dispatched == [("nemo_gym.cli.eval:prepare_benchmark", [f"+config_paths=[{path.resolve()}]"])]

    @pytest.mark.parametrize("agent_type", [None, "hermes_agent"])
    def test_prepare_reads_the_benchmarks_settings(self, monkeypatch: MonkeyPatch, tmp_path, agent_type) -> None:
        path = _benchmark(tmp_path, "tb", _harbor_config(builds={"enabled": False}))
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
        seen: dict = {}

        def fake_prepare(target, **kwargs):
            seen.update(target=target, **kwargs)
            raise _Stop

        monkeypatch.setattr(cli_eval, "prepare_target", fake_prepare)
        args = argparse.Namespace(_parser=argparse.ArgumentParser(), target=None, agent=None, agent_type=agent_type)
        with pytest.raises(_Stop):
            cli_eval.prepare_harbor_benchmark(args, [f"+config_paths=[{path.resolve()}]"], harbor_benchmark(path))
        assert seen["build"].enabled is False
        assert seen["agent"] == (agent_type or "mini_swe_agent")


@pytest.mark.parametrize("command", ["prepare_benchmark", "plan_benchmark"])
def test_dataset_commands_point_a_harbor_config_to_its_target(monkeypatch: MonkeyPatch, capsys, command) -> None:
    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda **kwargs: {"harbor_target": "terminal_bench_4"})
    with pytest.raises(SystemExit):
        getattr(cli_eval, command)()
    captured = capsys.readouterr()
    assert "terminal_bench_4" in captured.out + captured.err
