# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.config.config_types import AlmostServerError, ConfigInterpolationError
from nemo_gym.environment.manifest import EnvironmentManifest, dump_manifest, load_manifest
from nemo_gym.environment.validation import (
    EnvironmentValidationError,
    ResolvedComponent,
    ResolvedComposition,
    _base_name,
    _infer_profile,
    _only_delegates_to_super,
    _only_raises_not_implemented,
    _resolve_manifest_composition,
    _with_component_root,
    resolve_config_servers,
    validate_environment,
)


@pytest.mark.parametrize(
    "name",
    [
        "terminal_bench_4/miniswe",
        "terminal_bench_4/miniswe_legacy_agent",
        "terminal_bench_science/miniswe",
        "terminal_bench_science/miniswe_legacy_agent",
        "terminal_bench_science/codex",
        "deepsearchqa",
        "widesearch",
        "fukuyamabench",
        "moldetox",
        "scicodepile",
        "partner_if",
    ],
)
def test_current_benchmark_manifests_match_runtime_composition(name: str) -> None:
    from nemo_gym.environment.manifest import resolve_manifest_config_path
    from nemo_gym.environment.validation import _mirror_differences

    path = Path(__file__).parents[3] / "benchmarks" / name / "manifest.yaml"
    manifest = load_manifest(path)
    composition = _resolve_manifest_composition(
        resolve_manifest_config_path(path, manifest), dataset_owner=manifest.dataset_owner
    )
    assert _mirror_differences(manifest, composition) == {}


def test_manifest_resolves_dataset_interpolation_against_its_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = _asset(tmp_path)
    config = path.with_name("config.yaml")
    text = config.read_text().replace(
        "jsonl_fpath: environments/demo/data/example.jsonl", "jsonl_fpath: ${fixture_data_path}"
    )
    config.write_text("fixture_data_path: environments/demo/data/example.jsonl\n" + text)
    report = validate_environment(path)
    assert report.datasets[0].rows == 1


@pytest.mark.parametrize("source", ["prepared", "agent"])
def test_benchmark_prompt_ownership_preserves_prepared_rows(tmp_path: Path, source: str) -> None:
    path = _asset(tmp_path, kind="benchmark")
    raw = yaml.safe_load(path.read_text())
    raw["prompt_source"] = source
    raw.pop("standard_prompt_config")
    path.write_text(yaml.safe_dump(raw))
    data = path.parent / "data/example.jsonl"
    original = json.dumps({"responses_create_params": {"input": [{"role": "user", "content": "Already rendered"}]}})
    data.write_text(original + "\n")

    report = validate_environment(path)

    assert report.datasets[0].rows == 1
    assert report.datasets[0].prompt_config is None
    assert data.read_text() == original + "\n"


def test_agent_prompt_is_checked_but_not_applied_to_dataset_rows(tmp_path: Path) -> None:
    path = _asset(tmp_path, kind="benchmark")
    raw = yaml.safe_load(path.read_text())
    raw["prompt_source"] = "agent"
    path.write_text(yaml.safe_dump(raw))
    prompt = path.parent / "prompts/default.yaml"
    prompt.write_text("user: '{runtime_step}'\n")
    (path.parent / "data/example.jsonl").write_text('{"responses_create_params": {"input": []}}\n')
    assert validate_environment(path).datasets[0].rows == 1
    prompt.unlink()
    with pytest.raises(EnvironmentValidationError, match="prompt"):
        validate_environment(path)


def test_prepared_prompt_source_rejects_rows_without_input(tmp_path: Path) -> None:
    path = _asset(tmp_path, kind="benchmark")
    raw = yaml.safe_load(path.read_text())
    raw.update(prompt_source="prepared")
    raw.pop("standard_prompt_config")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(EnvironmentValidationError, match="prepared.*input"):
        validate_environment(path)


def _benchmark_rows(
    tmp_path: Path,
    rows: list[dict],
    *,
    prompt_source: str = "prepared",
    environment_server: str | None = "single_agent_turn",
) -> Path:
    """A benchmark whose dataset holds ``rows``; taskset ``demo`` routes to an ``environment_server`` server."""
    path = _asset(tmp_path, kind="benchmark")
    if prompt_source != "template":
        raw = yaml.safe_load(path.read_text())
        raw["prompt_source"] = prompt_source
        raw.pop("standard_prompt_config")
        path.write_text(yaml.safe_dump(raw))
    if environment_server is not None:
        config_path = path.with_name("config.yaml")
        config = yaml.safe_load(config_path.read_text())
        config["demo_task_environment_server"] = {
            "environment_servers": {
                environment_server: {
                    "entrypoint": "app.py",
                    "agent_server": {"type": "responses_api_agents", "name": "demo_agent"},
                    "resources_server": {"type": "resources_servers", "name": "demo_resources"},
                }
            }
        }
        config["environment_server_routes"] = {"demo": "demo_task_environment_server"}
        config_path.write_text(yaml.safe_dump(config))
    (path.parent / "data/example.jsonl").write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    return path


def _task_row(task_id: str = "1", **task_input: object) -> dict:
    task_input.setdefault("responses_create_params", {"input": [{"role": "user", "content": "What is 1 + 1?"}]})
    task_input.setdefault("task_data", {"expected_answer": "2"})
    return {"task_id": {"taskset": "demo", "task_id": task_id}, "task_input": task_input}


@pytest.mark.parametrize("environment_server", ["single_agent_turn", "single_agent_turn_legacy"])
def test_prepared_materialized_task_rows_are_valid_rollout_inputs(tmp_path: Path, environment_server: str) -> None:
    path = _benchmark_rows(tmp_path, [_task_row("1"), _task_row("2")], environment_server=environment_server)

    report = validate_environment(path)

    assert report.datasets[0].rows == 2


@pytest.mark.parametrize("prompt_source", ["prepared", "agent"])
@pytest.mark.parametrize("environment_server", ["swe_together", None])
def test_tasks_of_another_environment_server_are_checked_as_task_envelopes(
    tmp_path: Path, prompt_source: str, environment_server: str | None
) -> None:
    # As swe_together's prepare writes them: the server builds the request, so the task input holds task_data only.
    row = {"task_id": {"taskset": "demo", "task_id": "1"}, "task_input": {"task_data": {"instance_id": "1"}}}
    path = _benchmark_rows(tmp_path, [row], prompt_source=prompt_source, environment_server=environment_server)

    report = validate_environment(path)

    assert report.datasets[0].rows == 1


@pytest.mark.parametrize("environment_server", ["single_agent_turn", "swe_together", None])
@pytest.mark.parametrize(
    ("row", "location"),
    [
        ({"task_id": {"taskset": "demo"}, "task_input": {"task_data": {}}}, "task_id.task_id"),
        ({"task_id": {"taskset": "demo", "task_id": "2"}, "task_input": "not an object"}, "task_input"),
    ],
)
def test_a_broken_task_envelope_fails_whatever_server_its_taskset_routes_to(
    tmp_path: Path, environment_server: str | None, row: dict, location: str
) -> None:
    path = _benchmark_rows(tmp_path, [_task_row(), row], prompt_source="agent", environment_server=environment_server)

    with pytest.raises(
        EnvironmentValidationError, match=rf"^Dataset 'example' row 2 is not a valid task at {location}:"
    ):
        validate_environment(path)


def test_prepared_materialized_task_row_without_input_names_the_dataset_and_row(tmp_path: Path) -> None:
    path = _benchmark_rows(tmp_path, [_task_row("1"), _task_row("2", responses_create_params={"input": []})])

    with pytest.raises(EnvironmentValidationError) as raised:
        validate_environment(path)

    assert str(raised.value) == (
        "Benchmark dataset 'example' row 2 declares prepared prompts but has no task_input.responses_create_params.input."
    )


@pytest.mark.parametrize(
    ("row", "location"),
    [
        ({"task_id": {"taskset": "demo"}, "task_input": _task_row()["task_input"]}, "task_id.task_id"),
        (_task_row(task_data="not an object"), "task_input.task_data"),
        (
            _task_row(responses_create_params={"input": "Hi", "temperature": "warm"}),
            "task_input.responses_create_params",
        ),
    ],
)
def test_materialized_task_rows_are_checked_as_the_environment_server_reads_them(
    tmp_path: Path, row: dict, location: str
) -> None:
    path = _benchmark_rows(tmp_path, [_task_row(), row], prompt_source="agent")

    with pytest.raises(
        EnvironmentValidationError, match=rf"^Dataset 'example' row 2 is not a valid task at {location}"
    ):
        validate_environment(path)


def test_prompt_templates_are_refused_for_materialized_task_rows(tmp_path: Path) -> None:
    path = _benchmark_rows(tmp_path, [_task_row()], prompt_source="template")

    with pytest.raises(EnvironmentValidationError) as raised:
        validate_environment(path)

    assert str(raised.value) == (
        "Benchmark dataset 'example' row 1 is a materialized task, and prompt templates are not supported for "
        "materialized task rows."
    )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"responses_create_params": {"input": []}},
            "Benchmark dataset 'example' row 1 declares prepared prompts but has no responses_create_params.input.",
        ),
        (
            {"responses_create_params": {"input": "Hi", "temperature": "warm"}},
            "Dataset 'example' row 1 is not a valid rollout input at responses_create_params.temperature: "
            "Input should be a valid number, unable to parse string as a number",
        ),
    ],
)
def test_flat_rows_keep_their_checks_and_messages(tmp_path: Path, row: dict, message: str) -> None:
    path = _benchmark_rows(tmp_path, [row])

    with pytest.raises(EnvironmentValidationError) as raised:
        validate_environment(path)

    assert str(raised.value) == message


def test_manifest_selects_named_config_and_dataset_owner(tmp_path: Path) -> None:
    path = _asset(tmp_path, kind="benchmark")
    config = path.with_name("config.yaml")
    config.rename(path.with_name("variant.yaml"))
    selected_config = path.with_name("variant.yaml")
    raw_config = yaml.safe_load(selected_config.read_text())
    raw_config["other_agent"] = raw_config["demo_agent"]
    raw_config["other_environment_server"] = {
        "environment_servers": {
            "legacy_agent": {
                "entrypoint": "app.py",
                "agent_server": {"type": "responses_api_agents", "name": "other_agent"},
            }
        }
    }
    selected_config.write_text(yaml.safe_dump(raw_config))
    raw = yaml.safe_load(path.read_text())
    raw.update(config_path="variant.yaml", dataset_owner="demo_agent")
    path.write_text(yaml.safe_dump(raw))

    report = validate_environment(path)

    assert report.config_path == str(selected_config)
    assert next(c.name for c in report.components if c.role == "agent_server") == "demo_agent"
    with pytest.raises(EnvironmentValidationError, match="exactly one"):
        _resolve_manifest_composition(selected_config)
    with pytest.raises(EnvironmentValidationError, match="dataset_owner"):
        _resolve_manifest_composition(selected_config, dataset_owner="missing")


def test_resolve_config_servers_reads_the_wiring_of_a_config(tmp_path: Path) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")

    assert resolve_config_servers(config_path) == ("demo", "simple_agent")


@pytest.mark.parametrize("route", ["agent", "taskset", "legacy"])
@pytest.mark.parametrize("agent_resources", [None, "other_resources"])
def test_resolve_config_servers_uses_selected_environment_resources(
    tmp_path: Path, route: str, agent_resources: str | None
) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    agent = config["demo_agent"]["responses_api_agents"]["simple_agent"]
    agent.pop("resources_server")
    if agent_resources:
        agent["resources_server"] = {"type": "resources_servers", "name": agent_resources}
    config["other_resources"] = {"resources_servers": {"other": {"entrypoint": "app.py", "domain": "math"}}}
    environment = {
        "entrypoint": "app.py",
        "agent_server": {"type": "responses_api_agents", "name": "demo_agent"},
        "resources_server": {"type": "resources_servers", "name": "demo_resources"},
    }
    implementation = "single_agent_turn" if route == "taskset" else "single_agent_turn_legacy"
    config["demo_environment_server"] = {"environment_servers": {implementation: environment}}
    if route != "agent":
        # A second environment fronts the same agent, so only the explicit route identifies the binding.
        config["a_decoy_environment_server"] = {
            "environment_servers": {
                implementation: environment
                | {"resources_server": {"type": "resources_servers", "name": "other_resources"}}
            }
        }
    if route == "taskset":
        agent["datasets"][0]["taskset"] = "demo:test"
        config["environment_server_routes"] = {"demo:test": "demo_environment_server"}
        # A materialized taskset takes precedence even over the flat-row route.
        config.update(environment_routing_mode="legacy", environment_server_name="a_decoy_environment_server")
    elif route == "legacy":
        config.update(environment_routing_mode="legacy", environment_server_name="demo_environment_server")
    config_path.write_text(yaml.safe_dump(config))

    assert resolve_config_servers(config_path) == ("demo", "simple_agent")
    # Catalog correction must not silently change the manifest's existing composition contract.
    assert _resolve_manifest_composition(config_path).resources_server == ("other" if agent_resources else None)


@pytest.mark.parametrize("second_resources", ["demo_resources", "other_resources"])
def test_resolve_config_servers_checks_every_taskset_binding(tmp_path: Path, second_resources: str) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    agent = config["demo_agent"]["responses_api_agents"]["simple_agent"]
    dataset = agent["datasets"][0]
    agent["datasets"] = [dict(dataset, name=taskset, taskset=taskset) for taskset in ("demo:first", "demo:second")]
    config["other_resources"] = {"resources_servers": {"other": {"entrypoint": "app.py", "domain": "math"}}}
    for name, resources in (("demo_environment_server", "demo_resources"), ("second_environment", second_resources)):
        config[name] = {
            "environment_servers": {
                "single_agent_turn": {
                    "entrypoint": "app.py",
                    "agent_server": {"type": "responses_api_agents", "name": "demo_agent"},
                    "resources_server": {"type": "resources_servers", "name": resources},
                }
            }
        }
    config["environment_server_routes"] = {
        "demo:first": "demo_environment_server",
        "demo:second": "second_environment",
    }
    config_path.write_text(yaml.safe_dump(config))

    expected = "demo" if second_resources == "demo_resources" else None
    assert resolve_config_servers(config_path) == (expected, "simple_agent")


@pytest.mark.parametrize("route", [None, "other_environment", "missing", "demo_resources"])
def test_resolve_config_servers_rejects_routes_that_do_not_front_the_selected_agent(
    tmp_path: Path, route: str | None
) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    other_agent = deepcopy(config["demo_agent"])
    other_agent["responses_api_agents"]["simple_agent"].pop("datasets")
    config["other_agent"] = other_agent
    config["other_resources"] = {"resources_servers": {"other": {"entrypoint": "app.py", "domain": "math"}}}
    config["other_environment"] = {
        "environment_servers": {
            "single_agent_turn_legacy": {
                "entrypoint": "app.py",
                "agent_server": {"type": "responses_api_agents", "name": "other_agent"},
                "resources_server": {"type": "resources_servers", "name": "other_resources"},
            }
        }
    }
    if route is not None:
        config.update(environment_routing_mode="legacy", environment_server_name=route)
    config_path.write_text(yaml.safe_dump(config))

    # An unrelated environment does not make the agent's implicit route ambiguous, but cannot be its explicit route.
    expected = "demo" if route is None else None
    assert resolve_config_servers(config_path) == (expected, "simple_agent")


@pytest.mark.parametrize("include_taskset", [False, True])
def test_resolve_config_servers_keeps_resources_unknown_for_unroutable_flat_datasets(
    tmp_path: Path, include_taskset: bool
) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["environment_routing_mode"] = "taskset"
    if include_taskset:
        datasets = config["demo_agent"]["responses_api_agents"]["simple_agent"]["datasets"]
        datasets.insert(0, dict(datasets[0], name="routed", taskset="demo:test"))
        config["environment_server_routes"] = {"demo:test": "demo_environment_server"}
    config_path.write_text(yaml.safe_dump(config))

    # A successful first route must not hide the next dataset's missing route or permit the raw-name fallback.
    assert resolve_config_servers(config_path) == (None, "simple_agent")


def test_resolve_config_servers_does_not_guess_between_environment_routes(tmp_path: Path) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["second_environment"] = config["demo_environment_server"]
    config_path.write_text(yaml.safe_dump(config))

    # Both relays front the same agent and even share its resource binding, but flat-row routing is ambiguous.
    assert resolve_config_servers(config_path) == (None, "simple_agent")


def test_resolve_config_servers_keeps_self_contained_agent_without_resources(tmp_path: Path) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    del config["demo_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]
    config_path.write_text(yaml.safe_dump(config))

    # The unrelated resources block in the file must not become the self-contained agent's binding.
    assert resolve_config_servers(config_path) == (None, "simple_agent")


def test_resolve_config_servers_is_unknown_when_the_config_needs_run_time_servers(tmp_path: Path) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    # A judge model server that only a run defines, as in environments/abstention.
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "      grading_mode: exact\n",
            "      grading_mode: exact\n"
            "      judge_model_server:\n"
            "        type: responses_api_models\n"
            "        name: judge_model\n",
        ),
        encoding="utf-8",
    )

    assert resolve_config_servers(config_path) is None


def test_resolve_config_servers_does_not_apply_manifest_dataset_rules(tmp_path: Path) -> None:
    config_path = _asset(tmp_path, kind="benchmark").with_name("config.yaml")
    # A benchmark whose data already exists may omit prepare_script; only a manifest requires it.
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("        prepare_script: benchmarks/demo/prepare.py\n", ""),
        encoding="utf-8",
    )

    assert resolve_config_servers(config_path) == ("demo", "simple_agent")
    with pytest.raises(ValidationError):
        _resolve_manifest_composition(config_path)


@pytest.mark.parametrize(
    "text",
    [b"demo: ${unclosed\n", b"- a list\n- not a mapping\n", b"\xff\xfe not utf-8\n"],
    ids=["interpolation-syntax", "top-level-list", "not-utf-8"],
)
def test_resolve_config_servers_is_unknown_for_a_malformed_config(tmp_path: Path, text: bytes) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(text)

    assert resolve_config_servers(config_path) is None


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("config_paths",), 1),
        (("demo_agent", "responses_api_agents", "simple_agent", "_inherit_from"), 1),
        (("demo_resources", "resources_servers", "demo"), "not a mapping"),
    ],
    ids=["config-paths-not-a-list", "inherit-from-not-a-path", "server-block-not-a-mapping"],
)
def test_resolve_config_servers_is_unknown_for_a_field_of_the_wrong_type(
    tmp_path: Path, field_path: tuple[str, ...], value: object
) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent = config
    for key in field_path[:-1]:
        parent = parent[key]
    parent[field_path[-1]] = value
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert resolve_config_servers(config_path) is None


@pytest.mark.parametrize(
    ("case", "error_type"),
    [("policy_model_port", StopIteration), ("unknown_home", RuntimeError), ("recursive_alias", RecursionError)],
)
def test_resolve_config_servers_logs_unresolvable_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: str,
    error_type: type[Exception],
) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    if case == "policy_model_port":
        config["policy_model"] = {"port": 12345}
    elif case == "unknown_home":
        config["results_dir"] = "~nemo_gym_missing_user/out"
        original_expanduser = Path.expanduser

        def expanduser(path: Path) -> Path:
            # Exercise the parser's home expansion without depending on the machine's user directory service.
            if str(path) == config["results_dir"]:
                raise RuntimeError("Could not determine home directory.")
            return original_expanduser(path)

        monkeypatch.setattr(Path, "expanduser", expanduser)
    else:
        config["recursive"] = config
    config_path.write_text(yaml.safe_dump(config))

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.environment.validation"):
        assert resolve_config_servers(config_path) is None
    [record] = [record for record in caplog.records if record.name == "nemo_gym.environment.validation"]
    assert record.levelno == logging.DEBUG
    assert str(config_path) in record.getMessage()
    assert record.exc_info is not None and record.exc_info[0] is error_type
    # Only best-effort catalog lookup catches these failures; manifest composition stays strict.
    with pytest.raises(error_type):
        _resolve_manifest_composition(config_path)


@pytest.mark.parametrize("error_on_almost_servers", [True, False])
def test_resolve_config_servers_rejects_unreferenced_almost_servers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], error_on_almost_servers: bool
) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["unreferenced_resources"] = {
        "resources_servers": {"other": {"entrypoint": "app.py", "domain": "not_a_domain"}}
    }
    config["error_on_almost_servers"] = error_on_almost_servers
    config_path.write_text(yaml.safe_dump(config))

    assert resolve_config_servers(config_path) is None
    assert capsys.readouterr() == ("", "")


def test_resolve_config_servers_prints_nothing_for_a_server_block_that_fails_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _asset(tmp_path).with_name("config.yaml")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("domain: math", "domain: not_a_domain"), encoding="utf-8"
    )

    assert resolve_config_servers(config_path) is None
    assert capsys.readouterr() == ("", "")
    # Validating the config still reports the block.
    with pytest.raises(AlmostServerError):
        _resolve_manifest_composition(config_path)
    assert "Almost-Server Detected: 'demo_resources'" in capsys.readouterr().err


def test_manifest_accepts_external_agent_without_resources(tmp_path: Path) -> None:
    path = _asset(tmp_path, kind="benchmark", profile="external-agent-loop")
    raw = yaml.safe_load(path.read_text())
    raw.update(resources_server=None, grading_mode=None, model_server="policy_model")
    path.write_text(yaml.safe_dump(raw))
    config = yaml.safe_load(path.with_name("config.yaml").read_text())
    del config["demo_resources"]
    del config["demo_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]
    path.with_name("config.yaml").write_text(yaml.safe_dump(config))
    report = validate_environment(path)
    assert all(c.role != "resources_server" for c in report.components)


@pytest.mark.parametrize("tasksets", [["demo:test"], ["demo:test", "demo:other"], ["demo:test", None]])
def test_manifest_resolves_taskset_datasets_through_environment(tmp_path: Path, tasksets: list[str | None]) -> None:
    path = _asset(tmp_path)
    config_path = path.with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    agent = config["demo_agent"]["responses_api_agents"]["simple_agent"]
    (dataset,) = agent.pop("datasets")
    config["demo_resources"]["resources_servers"]["demo"]["datasets"] = [
        dict(dataset, name=f"dataset-{index}", taskset=taskset) for index, taskset in enumerate(tasksets)
    ]
    if None not in tasksets:
        del agent["resources_server"]
    config["demo_environment_server"] = {
        "environment_servers": {
            "single_agent_turn": {
                "entrypoint": "app.py",
                "agent_server": {"type": "responses_api_agents", "name": "demo_agent"},
                "resources_server": {"type": "resources_servers", "name": "demo_resources"},
            }
        }
    }
    config["environment_server_routes"] = {taskset: "demo_environment_server" for taskset in tasksets if taskset}
    config_path.write_text(yaml.safe_dump(config))

    report = validate_environment(path, sync=True)
    assert next(c.name for c in report.components if c.role == "agent_server") == "demo_agent"
    assert len(report.datasets) == len(tasksets)
    assert resolve_config_servers(config_path) == ("demo", "simple_agent")

    # Every declared taskset must be checked, even if another route already resolved.
    del config["environment_server_routes"]["demo:test"]
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(EnvironmentValidationError, match="No Environment Server route for taskset 'demo:test'"):
        validate_environment(path)


def _manifest(*, kind: str = "environment", profile: str = "custom-gym-verifier") -> dict:
    root = f"{'benchmarks' if kind == 'benchmark' else 'environments'}/demo"
    dataset = {
        "name": "example",
        "type": "benchmark" if kind == "benchmark" else "example",
        "jsonl_fpath": f"{root}/data/example.jsonl",
        "num_repeats": 1,
    }
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "kind": kind,
        "integration_profile": profile,
        "domain": "math",
        "description": "A small exact-match evaluation.",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["contributor"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "demo",
        "agent_server": "simple_agent",
        "datasets": [dataset],
        "grading_mode": "exact",
    }
    if profile in {"custom-gym-verifier", "custom-gym-agent-loop"}:
        manifest["model_server"] = "policy_model"
    if kind == "benchmark":
        dataset["prepare_script"] = f"{root}/prepare.py"
        manifest.update(canonical_split="test", standard_prompt_config=f"{root}/prompts/default.yaml")
    if profile == "external-rollout-driver":
        manifest["rollout_driver"] = "environments.demo.rollout_driver:run_rollout_collection"
    return manifest


def test_ast_helpers_and_config_outside_catalog(tmp_path: Path) -> None:
    tree = ast.parse(
        "class Agent(pkg.SimpleAgent):\n"
        "    async def responses(self):\n"
        "        'unavailable'\n"
        "        raise NotImplementedError()\n"
        "    async def run(self):\n"
        "        'fallback'\n"
        "        return await super().run()\n"
    )
    server = tree.body[0]
    assert isinstance(server, ast.ClassDef)
    responses, run = server.body

    assert _base_name(server.bases[0]) == "SimpleAgent"
    assert _only_raises_not_implemented(responses)
    assert _only_delegates_to_super(run)
    with _with_component_root(tmp_path / "config.yaml"):
        pass


def _config(*, kind: str = "environment", profile: str = "custom-gym-verifier") -> str:
    root = f"{'benchmarks' if kind == 'benchmark' else 'environments'}/demo"
    dataset = f"""      - name: example
        type: {"benchmark" if kind == "benchmark" else "example"}
        jsonl_fpath: {root}/data/example.jsonl
"""
    if kind == "benchmark":
        dataset += f"        prepare_script: {root}/prepare.py\n"
    model_server = (
        ""
        if profile == "external-rollout-driver"
        else """      model_server:
        type: responses_api_models
        name: policy_model
"""
    )
    rollout_driver = (
        "rollout_collection_driver: environments.demo.rollout_driver:run_rollout_collection\n"
        if profile == "external-rollout-driver"
        else ""
    )
    return f"""demo_resources:
  resources_servers:
    demo:
      entrypoint: app.py
      domain: math
      grading_mode: exact
demo_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: demo_resources
{model_server.rstrip()}
      datasets:
{dataset.rstrip()}
demo_environment_server:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: demo_agent
{rollout_driver}"""


def _asset(tmp_path: Path, *, kind: str = "environment", profile: str = "custom-gym-verifier") -> Path:
    directory = tmp_path / ("benchmarks" if kind == "benchmark" else "environments") / "demo"
    (directory / "data").mkdir(parents=True)
    manifest_path = directory / "manifest.yaml"
    manifest_path.write_text(dump_manifest(_manifest(kind=kind, profile=profile)), encoding="utf-8")
    (directory / "config.yaml").write_text(_config(kind=kind, profile=profile), encoding="utf-8")

    if kind == "benchmark":
        row = {"question": "What is 1 + 1?", "expected_answer": "2"}
        (directory / "prompts").mkdir()
        (directory / "prompts/default.yaml").write_text("user: '{question}'\n", encoding="utf-8")
        (directory / "prepare.py").write_text(
            "from pathlib import Path\n\ndef prepare(output: Path) -> Path:\n    return output\n",
            encoding="utf-8",
        )
    else:
        row = {"responses_create_params": {"input": "What is 1 + 1?"}, "expected_answer": "2"}
    (directory / "data/example.jsonl").write_text(f"{json.dumps(row)}\n", encoding="utf-8")

    if profile == "external-rollout-driver":
        (directory / "rollout_driver.py").write_text(
            "def run_rollout_collection():\n    pass\n",
            encoding="utf-8",
        )
    return manifest_path


def _replace_manifest(path: Path, **changes: object) -> EnvironmentManifest:
    manifest = load_manifest(path)
    updated = manifest.model_copy(update=changes)
    path.write_text(dump_manifest(updated), encoding="utf-8")
    return manifest


def _custom_agent_asset(tmp_path: Path, *, profile: str, source: str) -> Path:
    manifest_path = _asset(tmp_path, profile=profile)
    changes: dict[str, object] = {"agent_server": "custom_agent"}
    if profile == "external-agent-loop":
        changes["model_server"] = "policy_model"
    _replace_manifest(manifest_path, **changes)

    config_path = manifest_path.with_name("config.yaml")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("    simple_agent:\n", "    custom_agent:\n"),
        encoding="utf-8",
    )
    agent_dir = tmp_path / "responses_api_agents/custom_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "app.py").write_text(source, encoding="utf-8")
    return manifest_path


def test_reports_resolved_composition_and_declared_profile(tmp_path: Path) -> None:
    report = validate_environment(_asset(tmp_path))

    assert report.name == "demo"
    assert report.declared_profile == "custom-gym-verifier"
    assert report.inferred_profile == "custom-gym-verifier"
    assert report.profile_evidence == "selected agent implementation is simple_agent"
    assert report.warnings == ()
    assert "integration_profile" not in report.to_dict()
    assert report.datasets[0].rows == 1
    assert report.grading_mode == "exact"
    assert report.rollout_driver is None
    assert [(item.role, item.implementation) for item in report.components] == [
        ("resources_server", "demo"),
        ("agent_server", "simple_agent"),
        ("model_server", "runtime-selected"),
    ]
    assert report.components[0].entrypoint == "app.py"


@pytest.mark.parametrize(
    ("profile", "responses_body", "evidence"),
    [
        (
            "custom-gym-agent-loop",
            "return None",
            "agent overrides responses() with measured behavior",
        ),
        (
            "external-agent-loop",
            "raise NotImplementedError",
            "agent responses() raises NotImplementedError",
        ),
    ],
)
def test_infers_custom_agent_profile_from_entrypoint_ast(
    tmp_path: Path,
    profile: str,
    responses_body: str,
    evidence: str,
) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile=profile,
        source=(
            "raise RuntimeError('must not import')\n\n"
            "class CustomAgent:\n"
            "    async def responses(self):\n"
            f"        {responses_body}\n\n"
            "    async def run(self):\n"
            "        pass\n\n"
            "CustomAgent.run_webserver()\n"
        ),
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == profile
    assert report.profile_evidence == evidence
    assert report.warnings == ()


@pytest.mark.parametrize(
    ("implementation", "profile"),
    [
        ("browsecomp_agent", "custom-gym-agent-loop"),
        ("harbor_agent", "external-agent-loop"),
        ("labbench2_vlm_agent", "custom-gym-verifier"),
    ],
)
def test_infers_representative_builtin_agent_profiles(implementation: str, profile: str) -> None:
    component = ResolvedComponent(
        role="agent_server",
        name=implementation,
        implementation=implementation,
        boundary="responses_api_agents",
        entrypoint=str(Path(__file__).parents[3] / "responses_api_agents" / implementation / "app.py"),
    )
    composition = ResolvedComposition(
        resources_server=None,
        agent_server=implementation,
        model_server=None,
        datasets=(),
        rollout_driver=None,
        grading_mode=None,
        components=(component,),
    )

    assert _infer_profile(composition)[0] == profile


def test_profile_mismatch_is_reported_as_a_warning(tmp_path: Path) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile="custom-gym-verifier",
        source=("class CustomAgent:\n    async def responses(self):\n        pass\n\nCustomAgent.run_webserver()\n"),
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == "custom-gym-agent-loop"
    assert report.warnings == (
        "Declared integration_profile 'custom-gym-verifier' does not match inferred profile 'custom-gym-agent-loop' "
        "(agent overrides responses() with measured behavior).",
    )


def test_inconclusive_profile_is_unknown_and_warns(tmp_path: Path) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile="custom-gym-agent-loop",
        source="class CustomAgent:\n    pass\n\nCustomAgent.run_webserver()\n",
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == "unknown"
    assert report.profile_evidence == "agent server does not declare an inspectable responses() behavior"
    assert report.warnings == (
        "Could not infer integration_profile (agent server does not declare an inspectable responses() behavior); "
        "declared profile is 'custom-gym-agent-loop'.",
    )


def test_conditionally_external_agent_profile_is_unknown(tmp_path: Path) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile="custom-gym-agent-loop",
        source=(
            "class CustomAgent:\n"
            "    async def responses(self):\n"
            "        if self.external:\n"
            "            raise NotImplementedError\n"
            "        return None\n\n"
            "CustomAgent.run_webserver()\n"
        ),
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == "unknown"
    assert report.profile_evidence == "agent responses() behavior depends on runtime configuration"


def test_multiple_dataset_agents_are_actionable(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    config_path = manifest_path.with_name("config.yaml")
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """
second_demo_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: demo_resources
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: duplicate
        type: example
        jsonl_fpath: environments/demo/data/example.jsonl
second_demo_environment_server:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: second_demo_agent
""",
        encoding="utf-8",
    )

    with pytest.raises(
        EnvironmentValidationError,
        match=r"exactly one dataset-bearing agent instance.*demo_agent \(simple_agent\).*second_demo_agent",
    ):
        validate_environment(manifest_path)


def test_benchmark_uses_root_prompt_without_executing_prepare(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    prepare_path = manifest_path.parent / "prepare.py"
    prepare_path.write_text(
        "from pathlib import Path\n\ndef prepare(source: Path) -> Path:\n    raise RuntimeError('must not execute')\n",
        encoding="utf-8",
    )

    report = validate_environment(manifest_path)

    assert report.datasets[0].type == "benchmark"
    assert report.datasets[0].prompt_config.endswith("prompts/default.yaml")


def test_unresolved_interpolation_is_reported_cleanly_not_as_a_traceback(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    config_path = manifest_path.with_name("config.yaml")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "jsonl_fpath: environments/demo/data/example.jsonl",
            "jsonl_fpath: environments/demo/data/example.jsonl\n      num_repeats: ${undefined_key}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigInterpolationError, match="undefined_key"):
        validate_environment(manifest_path)


def test_malformed_benchmark_prompt_is_an_actionable_validation_error(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prompts/default.yaml").write_text("user: [broken\n", encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match="Could not materialize benchmark dataset"):
        validate_environment(manifest_path)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("async def prepare() -> Path:\n    pass\n", "synchronous prepare"),
        ("def prepare(\n", "Could not parse dataset prepare script"),
    ],
)
def test_prepare_contract_is_checked_statically(source: str, message: str, tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prepare.py").write_text(source, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_prepare_annotations_are_not_a_runtime_requirement(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prepare.py").write_text("def prepare():\n    return 'output'\n", encoding="utf-8")

    validate_environment(manifest_path)


def test_manifest_kind_matches_its_catalog_tree(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    raw = load_manifest(manifest_path).model_dump(mode="json", exclude_none=True)
    raw.update(kind="benchmark", canonical_split="test", standard_prompt_config="prompt.yaml")
    raw["datasets"][0].update(type="benchmark", prepare_script="prepare.py")
    manifest_path.write_text(dump_manifest(raw), encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match="kind"):
        validate_environment(manifest_path)


def test_manifest_name_is_an_id_that_need_not_match_its_folder(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    _replace_manifest(manifest_path, name="renamed")

    assert validate_environment(manifest_path).name == "renamed"


def test_a_manifest_at_the_root_of_a_catalog_tree_is_refused(tmp_path: Path) -> None:
    # Catalog discovery refuses it, so validation must too, although its name no longer has to match a folder.
    manifest_path = _asset(tmp_path)
    root_manifest = manifest_path.rename(tmp_path / "environments/manifest.yaml")
    _replace_manifest(root_manifest, config_path="demo/config.yaml")

    with pytest.raises(EnvironmentValidationError, match="named catalog directory"):
        validate_environment(root_manifest)


def test_two_manifests_in_one_tree_cannot_declare_one_name(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    copy_path = tmp_path / "environments/copy/manifest.yaml"
    copy_path.parent.mkdir()
    copy = load_manifest(manifest_path).model_copy(update={"config_path": "../demo/config.yaml"})
    copy_path.write_text(dump_manifest(copy), encoding="utf-8")

    for path in (manifest_path, copy_path):
        with pytest.raises(EnvironmentValidationError, match="also declared by"):
            validate_environment(path)


def test_a_renamed_manifest_cannot_take_another_configs_path(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    _replace_manifest(manifest_path, name="other")
    other_config = tmp_path / "environments/other/config.yaml"
    other_config.parent.mkdir()
    other_config.write_text(manifest_path.with_name("config.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match="path of another config"):
        validate_environment(manifest_path)


def test_slash_qualified_name_matches_nested_catalog_path(tmp_path: Path) -> None:
    config_manifest_path = _asset(tmp_path)
    nested_manifest_path = tmp_path / "environments/acme/demo/manifest.yaml"
    nested_manifest_path.parent.mkdir(parents=True)
    manifest = load_manifest(config_manifest_path).model_copy(update={"name": "acme/demo"})
    nested_manifest_path.write_text(dump_manifest(manifest), encoding="utf-8")

    report = validate_environment(nested_manifest_path, config_manifest_path.with_name("config.yaml"))

    assert report.name == "acme/demo"


def test_config_defaults_to_sibling_and_must_exist(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    config_path = manifest_path.with_name("config.yaml")
    validate_environment(manifest_path)
    config_path.unlink()

    with pytest.raises(EnvironmentValidationError, match="config.yaml"):
        validate_environment(manifest_path)


def test_stale_mirrors_are_reported_and_sync_changes_only_them(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    original = load_manifest(manifest_path)
    stale_dataset = original.datasets[0].model_copy(update={"jsonl_fpath": "wrong.jsonl"})
    _replace_manifest(
        manifest_path,
        resources_server="wrong",
        agent_server="wrong",
        datasets=[stale_dataset],
    )

    with pytest.raises(EnvironmentValidationError, match="resources_server") as caught:
        validate_environment(manifest_path)
    assert '"jsonl_fpath": "wrong.jsonl"' in str(caught.value)

    report = validate_environment(manifest_path, sync=True)
    synchronized = load_manifest(manifest_path)
    assert report.synchronized_fields == ("resources_server", "agent_server", "datasets")
    assert synchronized.resources_server == "demo"
    assert synchronized.agent_server == "simple_agent"
    assert synchronized.description == original.description


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "is empty"),
        ("[broken\n", "Malformed JSON"),
        ("[]\n", "must contain a JSON object"),
        ('{"responses_create_params": {"input": 42}}\n', "responses_create_params.input"),
    ],
)
def test_dataset_row_errors_are_actionable(tmp_path: Path, content: str, message: str) -> None:
    manifest_path = _asset(tmp_path)
    manifest_path.parent.joinpath("data/example.jsonl").write_text(content, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_invalid_dataset_encoding_is_an_actionable_error(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    manifest_path.parent.joinpath("data/example.jsonl").write_bytes(b"\xff\n")

    with pytest.raises(EnvironmentValidationError, match="Could not read dataset"):
        validate_environment(manifest_path)


def test_custom_driver_is_checked_without_scaffolding(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, profile="external-rollout-driver")
    report = validate_environment(manifest_path)
    assert report.rollout_driver == "environments.demo.rollout_driver:run_rollout_collection"
    assert report.inferred_profile == "external-rollout-driver"
    assert report.profile_evidence == "rollout_collection_driver is configured"
    assert report.warnings == ()

    driver_path = manifest_path.parent / "rollout_driver.py"
    driver_path.write_text("def other_function():\n    pass\n", encoding="utf-8")
    with pytest.raises(EnvironmentValidationError, match="Rollout driver.*was not found"):
        validate_environment(manifest_path)

    driver_path.unlink()
    with pytest.raises(EnvironmentValidationError, match="Rollout driver module was not found"):
        validate_environment(manifest_path)


HARBOR_HARNESSES = {
    "mini_swe_agent": (
        "responses_api_agents/mini_swe_agent/configs/mini_swe_agent_sandboxed.yaml",
        {"step_limit": 500},
    ),
    # Its shipped template leaves `resources_server.name` for benchmark composition; the Harbor run clears it.
    "terminus_2_sandboxed_agent": (
        "responses_api_agents/terminus_2_sandboxed_agent/configs/terminus_2_sandboxed_agent.yaml",
        {},
    ),
}


def _harbor_asset(
    tmp_path: Path, *, config_target: str, manifest_target: str | None = None, harness: str = "mini_swe_agent"
) -> Path:
    directory = tmp_path / "benchmarks" / "harbor_demo"
    directory.mkdir(parents=True)
    shipped, settings = HARBOR_HARNESSES[harness]
    config = {
        "harbor_target": config_target,
        "config_paths": [shipped],
        harness: {"responses_api_agents": {harness: settings}},
    }
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = _manifest(kind="benchmark", profile="custom-gym-agent-loop")
    manifest.update(
        name="harbor_demo",
        harbor_target=manifest_target or config_target,
        resources_server=None,
        agent_server=harness,
        datasets=[],
        prompt_source="agent",
        grading_mode=None,
    )
    manifest.pop("standard_prompt_config")
    path = directory / "manifest.yaml"
    path.write_text(dump_manifest(manifest), encoding="utf-8")
    return path


@pytest.mark.parametrize("harness", sorted(HARBOR_HARNESSES))
def test_harbor_backed_manifest_validates_its_harness_and_target(tmp_path: Path, harness: str) -> None:
    path = _harbor_asset(tmp_path, config_target="terminal_bench_4", harness=harness)
    report = validate_environment(path)
    assert report.datasets == ()
    assert [(c.role, c.name) for c in report.components] == [
        ("agent_server", harness),
        ("model_server", "policy_model"),
    ]
    assert report.components[0].implementation == harness
    assert resolve_config_servers(path.parent / "config.yaml") == (None, harness)


def test_dataset_owner_does_not_apply_to_a_harbor_target(tmp_path: Path) -> None:
    path = _harbor_asset(tmp_path, config_target="terminal_bench_4")
    manifest = yaml.safe_load(path.read_text())
    manifest["dataset_owner"] = "mini_swe_agent"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(EnvironmentValidationError, match="dataset_owner does not apply to harbor_target"):
        validate_environment(path)


def test_a_harbor_target_only_an_include_sets_is_refused(tmp_path: Path) -> None:
    path = _harbor_asset(tmp_path, config_target="terminal_bench_4")
    config = path.parent / "config.yaml"
    base = path.parent / "base.yaml"
    base.write_text(config.read_text(), encoding="utf-8")
    config.write_text(yaml.safe_dump({"config_paths": [str(base)]}), encoding="utf-8")
    with pytest.raises(EnvironmentValidationError, match="set `harbor_target` in the benchmark file itself"):
        validate_environment(path)


def test_harbor_target_is_mirrored_from_the_config(tmp_path: Path) -> None:
    path = _harbor_asset(tmp_path, config_target="terminal_bench_4", manifest_target="terminal_bench_2_1")
    with pytest.raises(EnvironmentValidationError, match="harbor_target"):
        validate_environment(path)


def test_harbor_target_must_resolve_without_a_fetch(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentValidationError, match="not an entry under environments/harbor/"):
        validate_environment(_harbor_asset(tmp_path, config_target="no_such_harbor_entry"))
    folder = tmp_path / "tasks"
    folder.mkdir()
    assert validate_environment(_harbor_asset(tmp_path / "local", config_target=str(folder))).datasets == ()


@pytest.mark.parametrize("target", ["harbor:", "harbor:org/name/extra"])
def test_a_malformed_harbor_reference_is_refused(tmp_path: Path, target: str) -> None:
    with pytest.raises(EnvironmentValidationError):
        validate_environment(_harbor_asset(tmp_path, config_target=target))


def test_a_well_formed_harbor_reference_is_accepted_without_a_fetch(tmp_path: Path) -> None:
    path = _harbor_asset(tmp_path, config_target="harbor:terminal-bench/terminal-bench@4.0.0")
    assert validate_environment(path).datasets == ()


def test_a_file_is_not_a_task_folder_target(tmp_path: Path) -> None:
    task_file = tmp_path / "task.toml"
    task_file.write_text("")
    with pytest.raises(EnvironmentValidationError, match="not an entry under environments/harbor/"):
        validate_environment(_harbor_asset(tmp_path / "file", config_target=str(task_file)))


def test_component_verifier_validation_does_not_require_full_dataset(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    (manifest_path.parent / "data/example.jsonl").unlink()
    with pytest.raises(EnvironmentValidationError, match="Dataset file was not found"):
        validate_environment(manifest_path)
    report = validate_environment(manifest_path, validate_datasets=False)
    assert report.datasets == ()
    assert report.components
    # Dataset-free validation must still catch a stale manifest composition.
    _replace_manifest(manifest_path, resources_server="wrong_server")
    with pytest.raises(EnvironmentValidationError, match="composition is stale"):
        validate_environment(manifest_path, validate_datasets=False)


@pytest.mark.parametrize("selected_input", [None, "benchmarks/demo/data/selected.jsonl"])
def test_manifest_composition_resolves_dynamic_dataset_defaults(tmp_path: Path, selected_input: str | None) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    config_path = manifest_path.with_name("config.yaml")
    config_path.write_text(
        config_path.read_text().replace(
            "jsonl_fpath: benchmarks/demo/data/example.jsonl",
            "jsonl_fpath: ${oc.select:prepare_script_args.input,benchmarks/demo/data/example.jsonl}",
        )
    )
    expected_path = "benchmarks/demo/data/example.jsonl"
    expected_rows = 1
    if selected_input is not None:
        config_path.write_text(config_path.read_text() + f"\nprepare_script_args:\n  input: {selected_input}\n")
        manifest = load_manifest(manifest_path)
        _replace_manifest(
            manifest_path,
            datasets=[manifest.datasets[0].model_copy(update={"jsonl_fpath": selected_input})],
        )
        original_data = (manifest_path.parent / "data/example.jsonl").read_text()
        (manifest_path.parent / "data/selected.jsonl").write_text(original_data * 2)
        expected_path = selected_input
        expected_rows = 2

    report = validate_environment(manifest_path)

    assert Path(report.datasets[0].path) == tmp_path / expected_path
    assert report.datasets[0].rows == expected_rows


@pytest.mark.parametrize("value", ["${undefined_key}", "${oc.env:LAB_TEST_UNSET_VARIABLE}"])
def test_dataset_path_resolution_uses_validation_error(
    tmp_path: Path, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAB_TEST_UNSET_VARIABLE", raising=False)
    manifest = _asset(tmp_path, kind="benchmark")
    config = manifest.with_name("config.yaml")
    config.write_text(
        config.read_text().replace("jsonl_fpath: benchmarks/demo/data/example.jsonl", f"jsonl_fpath: {value}")
    )
    with pytest.raises(ConfigInterpolationError):
        validate_environment(manifest)


def test_dataset_validation_ignores_undeclared_runtime_fields(tmp_path: Path) -> None:
    manifest = _asset(tmp_path, kind="benchmark")
    config = manifest.with_name("config.yaml")
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config)
    raw.demo_agent.responses_api_agents.simple_agent.datasets[0].ignored = "${runtime_only_value}"
    OmegaConf.save(raw, config)

    assert validate_environment(manifest).datasets[0].rows == 1


@pytest.mark.parametrize("mutation", [None, "missing_params", "empty_id", "extra_envelope"])
def test_catalog_validates_materialized_single_agent_turn_rows(tmp_path, mutation):
    from nemo_gym.environment.manifest import ManifestDataset
    from nemo_gym.environment.validation import _validate_dataset

    row = {
        "task_id": {"taskset": "environment:codex", "task_id": "task-1"},
        "task_input": {
            "responses_create_params": {"input": [{"role": "user", "content": "Prepared prompt"}]},
            "task_data": {"private_reference": "verifier-owned"},
        },
    }
    if mutation == "missing_params":
        row["task_input"].pop("responses_create_params")
    elif mutation == "empty_id":
        row["task_id"]["task_id"] = ""
    elif mutation == "extra_envelope":
        row["responses_create_params"] = {"input": "must not fall back to flat"}
    path = tmp_path / "tasks.jsonl"
    original = json.dumps(row) + "\n"
    path.write_text(original)
    dataset = ManifestDataset(name="example", type="example", jsonl_fpath=str(path))
    routing = {"environment:codex": "single_agent_turn"}
    if mutation is None:
        report = _validate_dataset(
            dataset, standard_prompt_config=None, prompt_source="prepared", taskset_environment_servers=routing
        )
        assert report.rows == 1 and path.read_text() == original
    else:
        with pytest.raises(EnvironmentValidationError, match="valid task"):
            _validate_dataset(
                dataset, standard_prompt_config=None, prompt_source="prepared", taskset_environment_servers=routing
            )


@pytest.mark.parametrize("task_input", ["bad", [1], 42, True, None])
def test_prepared_benchmark_rejects_non_mapping_task_input_with_row_context(tmp_path, task_input):
    from nemo_gym.environment.manifest import ManifestDataset
    from nemo_gym.environment.validation import _validate_dataset

    prepare = tmp_path / "prepare.py"
    prepare.write_text("from pathlib import Path\ndef prepare(output: Path) -> Path:\n    return output\n")
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"task_id": {"taskset": "demo", "task_id": "1"}, "task_input": task_input}) + "\n")
    dataset = ManifestDataset(
        name="bad-nested-row", type="benchmark", jsonl_fpath=str(data), prepare_script=str(prepare)
    )
    with pytest.raises(EnvironmentValidationError, match="bad-nested-row.*row 1.*task_input"):
        _validate_dataset(dataset, standard_prompt_config=None, prompt_source="prepared")


def _listing(
    tmp_path: Path,
    *,
    listed_as: str = "setups",
    name: str = "demo/variant",
    config_name: str = "variant.yaml",
    kind: str = "benchmark",
    config_edit: tuple[str, str] | None = None,
    **fields: object,
) -> Path:
    """The demo manifest with one more config listed, a copy of the demo config beside it, edited by ``config_edit``."""
    main_path = tmp_path / ("benchmarks" if kind == "benchmark" else "environments") / "demo/manifest.yaml"
    if not main_path.is_file():
        main_path = _asset(tmp_path, kind=kind)
    config = main_path.with_name("config.yaml").read_text(encoding="utf-8")
    if config_edit is not None:
        assert config_edit[0] in config
        config = config.replace(*config_edit)
    (main_path.parent / config_name).write_text(config, encoding="utf-8")
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    listed = {"name": name, "config_path": config_name, **fields}
    if listed_as == "protocols":
        listed.setdefault("description", "The demo benchmark under another prompt.")
    raw.setdefault(listed_as, []).append(listed)
    main_path.write_text(dump_manifest(raw), encoding="utf-8")
    return main_path


@pytest.mark.parametrize("listed_as", ["setups", "protocols"])
def test_a_listed_config_validates_by_name_and_the_main_manifest_still_validates(
    listed_as: str, tmp_path: Path
) -> None:
    main_path = _listing(tmp_path, listed_as=listed_as)

    report = validate_environment(main_path, name="demo/variant")

    assert (report.name, Path(report.config_path).name, report.manifest_path) == (
        "demo/variant",
        "variant.yaml",
        str(main_path.resolve()),
    )
    assert validate_environment(main_path).name == "demo"


def test_a_name_the_manifest_does_not_list_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentValidationError, match="lists no setup or protocol named 'demo/missing'"):
        validate_environment(_listing(tmp_path), name="demo/missing")


def test_a_listed_name_is_an_id_that_need_not_match_its_config_path(tmp_path: Path) -> None:
    # Like a main manifest's name, a setup's name stays the same when its config moves or is renamed.
    main_path = _listing(tmp_path, name="variant_id")

    assert validate_environment(main_path).name == "demo"
    report = validate_environment(main_path, name="variant_id")
    assert (report.name, Path(report.config_path).name) == ("variant_id", "variant.yaml")


def _other_manifest(main_path: Path, *, name: str, config_path: str) -> Path:
    """A second main manifest in the demo benchmark's tree, at ``demo/other_folder``."""
    other = main_path.parent / "other_folder/manifest.yaml"
    other.parent.mkdir()
    manifest = load_manifest(main_path).model_copy(update={"name": name, "config_path": config_path, "setups": []})
    other.write_text(dump_manifest(manifest), encoding="utf-8")
    return other


def test_a_listed_name_cannot_be_another_manifests_name(tmp_path: Path) -> None:
    main_path = _listing(tmp_path)
    other = _other_manifest(main_path, name="demo/variant", config_path="config.yaml")
    shutil.copy(main_path.with_name("config.yaml"), other.with_name("config.yaml"))

    for path, name in ((main_path, None), (main_path, "demo/variant"), (other, None)):
        with pytest.raises(EnvironmentValidationError, match="Name 'demo/variant' is also declared by"):
            validate_environment(path, name=name)


def test_a_listed_config_has_one_id(tmp_path: Path) -> None:
    # A manifest of its own for the config a main manifest lists would list that config a second time.
    main_path = _listing(tmp_path)
    _other_manifest(main_path, name="variant_id", config_path="../variant.yaml")

    for name in (None, "demo/variant"):
        with pytest.raises(EnvironmentValidationError, match="is also the config of 'variant_id'.*one ID"):
            validate_environment(main_path, name=name)


def test_a_config_has_one_id_within_its_manifest(tmp_path: Path) -> None:
    # `../demo/config.yaml` differs from `config.yaml` as text, but it is the main manifest's config.
    main_path = _listing(tmp_path)
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    raw["setups"][0]["config_path"] = "../demo/config.yaml"
    main_path.write_text(dump_manifest(raw), encoding="utf-8")

    for name, other_name in ((None, "demo/variant"), ("demo/variant", "demo")):
        with pytest.raises(EnvironmentValidationError, match=f"is also the config of '{other_name}' in this manifest"):
            validate_environment(main_path, name=name)


def test_a_listed_name_cannot_take_another_configs_path(tmp_path: Path) -> None:
    main_path = _listing(tmp_path, name="demo/other")
    shutil.copy(main_path.with_name("config.yaml"), main_path.with_name("other.yaml"))

    for name in (None, "demo/other"):
        with pytest.raises(EnvironmentValidationError, match="Name 'demo/other' is the path of another config"):
            validate_environment(main_path, name=name)


def test_a_listed_config_must_exist(tmp_path: Path) -> None:
    main_path = _listing(tmp_path)
    (main_path.parent / "variant.yaml").unlink()

    with pytest.raises(EnvironmentValidationError, match="config_path 'variant.yaml' was not found"):
        validate_environment(main_path)


def test_sync_writes_a_listed_configs_composition_into_its_entry_and_drops_it_once_it_matches(
    tmp_path: Path,
) -> None:
    # A protocol may score under another grading mode, so this difference is one validation keeps.
    main_path = _listing(tmp_path, listed_as="protocols", config_edit=("grading_mode: exact", "grading_mode: fuzzy"))

    with pytest.raises(EnvironmentValidationError, match="composition of demo/variant is stale(.|\n)*grading_mode"):
        validate_environment(main_path, name="demo/variant")
    report = validate_environment(main_path, name="demo/variant", sync=True)

    assert report.synchronized_fields == ("grading_mode",)
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    assert raw["grading_mode"] == "exact"
    assert raw["protocols"][0]["grading_mode"] == "fuzzy"

    variant = main_path.with_name("variant.yaml")
    variant.write_text(variant.read_text(encoding="utf-8").replace("fuzzy", "exact"), encoding="utf-8")
    validate_environment(main_path, name="demo/variant", sync=True)

    assert "grading_mode" not in yaml.safe_load(main_path.read_text(encoding="utf-8"))["protocols"][0]


@pytest.mark.parametrize(
    ("config_edit", "fields", "message"),
    [
        (("grading_mode: exact", "grading_mode: fuzzy"), {"grading_mode": "fuzzy"}, "another resources server or"),
        (
            ("data/example.jsonl", "data/other.jsonl"),
            {
                "datasets": [
                    {
                        "name": "example",
                        "type": "benchmark",
                        "jsonl_fpath": "benchmarks/demo/data/other.jsonl",
                        "prepare_script": "benchmarks/demo/prepare.py",
                    }
                ]
            },
            "other task datasets",
        ),
    ],
)
def test_a_setup_runs_the_main_manifests_tasks_and_scoring_and_a_protocol_need_not(
    config_edit: tuple[str, str], fields: dict, message: str, tmp_path: Path
) -> None:
    setup = _listing(tmp_path, config_edit=config_edit, **fields)
    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(setup, name="demo/variant")

    protocol = _listing(
        tmp_path, listed_as="protocols", name="demo/other", config_name="other.yaml", config_edit=config_edit, **fields
    )
    (tmp_path / "benchmarks/demo/data/other.jsonl").write_text(
        (tmp_path / "benchmarks/demo/data/example.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert validate_environment(protocol, name="demo/other").name == "demo/other"


def test_a_setup_keeps_the_benchmarks_prompt_template(tmp_path: Path) -> None:
    # The same tasks, scorer and harness with a system prompt change the benchmark's rules: a protocol, not a setup.
    with_system = "benchmarks/demo/prompts/with_system.yaml"
    edit = (
        "prepare_script: benchmarks/demo/prepare.py",
        f"prepare_script: benchmarks/demo/prepare.py\n        prompt_config: {with_system}",
    )
    dataset = {
        "name": "example",
        "type": "benchmark",
        "jsonl_fpath": "benchmarks/demo/data/example.jsonl",
        "prepare_script": "benchmarks/demo/prepare.py",
        "prompt_config": with_system,
    }
    main_path = _listing(
        tmp_path, listed_as="protocols", config_edit=edit, standard_prompt_config=with_system, datasets=[dataset]
    )
    (tmp_path / with_system).write_text("system: Answer briefly.\nuser: '{question}'\n", encoding="utf-8")
    assert validate_environment(main_path, name="demo/variant").name == "demo/variant"

    _listing(tmp_path, name="demo/other", config_name="other.yaml", config_edit=edit, datasets=[dataset])
    with pytest.raises(EnvironmentValidationError, match="another prompt template"):
        validate_environment(main_path, name="demo/other")


def _setup_with_its_own_template(
    tmp_path: Path, template: str, template_text: str, *, agent_server: str | None = None, **fields: object
) -> Path:
    """The demo benchmark with a setup that names its own prompt template, run by ``agent_server`` when it is set."""
    edit = (
        "prepare_script: benchmarks/demo/prepare.py",
        f"prepare_script: benchmarks/demo/prepare.py\n        prompt_config: {template}",
    )
    dataset = {
        "name": "example",
        "type": "benchmark",
        "jsonl_fpath": "benchmarks/demo/data/example.jsonl",
        "prepare_script": "benchmarks/demo/prepare.py",
        "prompt_config": template,
    }
    if agent_server is not None:
        fields["agent_server"] = agent_server
    main_path = _listing(tmp_path, config_edit=edit, standard_prompt_config=template, datasets=[dataset], **fields)
    if agent_server is not None:
        variant = main_path.with_name("variant.yaml")
        variant.write_text(
            variant.read_text(encoding="utf-8").replace("    simple_agent:\n", f"    {agent_server}:\n"),
            encoding="utf-8",
        )
    (tmp_path / template).write_text(template_text, encoding="utf-8")
    return main_path


def test_a_setup_may_adapt_the_benchmarks_prompt_for_its_harness(tmp_path: Path) -> None:
    # The harness owns the system prompt, so the setup's template carries the benchmark's prompt in the user message.
    description = "The demo benchmark with its prompt in the user message, because the harness owns the system prompt."
    main_path = _setup_with_its_own_template(
        tmp_path,
        "benchmarks/demo/prompts/in_user_message.yaml",
        "user: '{question}'\n",
        agent_server="custom_agent",
        description=description,
    )
    assert validate_environment(main_path, name="demo/variant").name == "demo/variant"

    # Review checks the adaptation, so the setup has to say what it is.
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    del raw["setups"][0]["description"]
    main_path.write_text(dump_manifest(raw), encoding="utf-8")
    with pytest.raises(EnvironmentValidationError, match="names its own prompt template; its description must say"):
        validate_environment(main_path, name="demo/variant")


def test_a_setup_with_its_parents_agent_server_keeps_the_benchmarks_prompt_template(tmp_path: Path) -> None:
    # A system-prompt variant run by the same agent, such as arxivmath_0626/config_with_system, prompts under other
    # rules. Listed as a setup with its own template and a description, it is rejected: the same harness needs no
    # other form of the prompt, so the template comparison runs as usual.
    main_path = _setup_with_its_own_template(
        tmp_path,
        "benchmarks/demo/prompts/with_system.yaml",
        "system: Answer briefly.\nuser: '{question}'\n",
        description="The demo benchmark with a system prompt.",
    )
    with pytest.raises(EnvironmentValidationError, match="demo/variant uses another prompt template than demo"):
        validate_environment(main_path, name="demo/variant")

    # The same entry, moved unchanged to protocols, validates.
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    raw["protocols"] = raw.pop("setups")
    main_path.write_text(dump_manifest(raw), encoding="utf-8")
    assert validate_environment(main_path, name="demo/variant").name == "demo/variant"


def _protocol_with_a_setup(tmp_path: Path, *, nested_edit: tuple[str, str] | None = None) -> Path:
    """The demo benchmark with a system-prompt protocol that lists a setup, whose config ``nested_edit`` changes."""
    with_system = "benchmarks/demo/prompts/with_system.yaml"
    prompt_line = f"prompt_config: {with_system}"
    edit = (
        "prepare_script: benchmarks/demo/prepare.py",
        f"prepare_script: benchmarks/demo/prepare.py\n        {prompt_line}",
    )
    dataset = {
        "name": "example",
        "type": "benchmark",
        "jsonl_fpath": "benchmarks/demo/data/example.jsonl",
        "prepare_script": "benchmarks/demo/prepare.py",
        "prompt_config": with_system,
    }
    main_path = _listing(
        tmp_path, listed_as="protocols", config_edit=edit, standard_prompt_config=with_system, datasets=[dataset]
    )
    (tmp_path / with_system).write_text("system: Answer briefly.\nuser: '{question}'\n", encoding="utf-8")
    nested = main_path.with_name("variant.yaml").read_text(encoding="utf-8")
    if nested_edit is not None:
        assert nested_edit[0] in nested
        nested = nested.replace(*nested_edit)
    main_path.with_name("nested.yaml").write_text(nested, encoding="utf-8")
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    raw["protocols"][0]["setups"] = [{"name": "demo/nested", "config_path": "nested.yaml", "description": "Nested."}]
    main_path.write_text(dump_manifest(raw), encoding="utf-8")
    return main_path


def test_a_setup_a_protocol_lists_runs_the_protocols_prompt_not_the_main_manifests(tmp_path: Path) -> None:
    main_path = _protocol_with_a_setup(tmp_path)
    report = validate_environment(main_path, name="demo/nested")
    assert (report.name, Path(report.config_path).name) == ("demo/nested", "nested.yaml")
    assert validate_environment(main_path).name == "demo"

    # Another config with the main manifest's prompt, listed under the protocol, runs another template than it.
    main_path.with_name("other.yaml").write_text(
        main_path.with_name("config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    main_dataset = {key: value for key, value in raw["datasets"][0].items()}
    raw["protocols"][0]["setups"].append(
        {"name": "demo/other", "config_path": "other.yaml", "datasets": [main_dataset]}
    )
    main_path.write_text(dump_manifest(raw), encoding="utf-8")
    with pytest.raises(EnvironmentValidationError, match="demo/other uses another prompt template than demo/variant"):
        validate_environment(main_path, name="demo/other")


def test_sync_writes_a_nested_setups_composition_into_it_and_drops_what_matches_its_protocol(tmp_path: Path) -> None:
    with_system = "benchmarks/demo/prompts/with_system.yaml"
    main_path = _protocol_with_a_setup(
        tmp_path,
        nested_edit=(f"prompt_config: {with_system}", f"prompt_config: {with_system}\n        num_repeats: 2"),
    )

    with pytest.raises(EnvironmentValidationError, match="composition of demo/nested is stale"):
        validate_environment(main_path, name="demo/nested")
    assert validate_environment(main_path, name="demo/nested", sync=True).synchronized_fields == ("datasets",)

    protocol = yaml.safe_load(main_path.read_text(encoding="utf-8"))["protocols"][0]
    assert protocol["datasets"][0].get("num_repeats", 1) == 1
    assert protocol["setups"][0]["datasets"][0]["num_repeats"] == 2

    nested = main_path.with_name("nested.yaml")
    nested.write_text(nested.read_text(encoding="utf-8").replace("num_repeats: 2", "num_repeats: 1"), encoding="utf-8")
    validate_environment(main_path, name="demo/nested", sync=True)
    assert "datasets" not in yaml.safe_load(main_path.read_text(encoding="utf-8"))["protocols"][0]["setups"][0]


def test_an_environment_setup_shares_its_example_datasets(tmp_path: Path) -> None:
    # An environment declares only example datasets, so those are what a setup must share.
    main_path = _listing(tmp_path, kind="environment")
    assert validate_environment(main_path, name="demo/variant").name == "demo/variant"

    _listing(
        tmp_path,
        kind="environment",
        name="demo/other",
        config_name="other.yaml",
        config_edit=("data/example.jsonl", "data/other.jsonl"),
        datasets=[{"name": "example", "type": "example", "jsonl_fpath": "environments/demo/data/other.jsonl"}],
    )
    with pytest.raises(EnvironmentValidationError, match="other task datasets"):
        validate_environment(main_path, name="demo/other")


@pytest.mark.parametrize(("setup_target", "valid"), [("terminal_bench_4", True), ("terminal_bench_2_1", False)])
def test_a_setup_of_a_harbor_backed_benchmark_runs_the_same_target(
    setup_target: str, valid: bool, tmp_path: Path
) -> None:
    # Neither declares datasets, so the Harbor target is what says which tasks they run.
    main_path = _harbor_asset(tmp_path, config_target="terminal_bench_4")
    config = yaml.safe_load(main_path.with_name("config.yaml").read_text(encoding="utf-8"))
    config["harbor_target"] = setup_target
    main_path.with_name("other.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    raw = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    raw["setups"] = [{"name": "harbor_demo/other", "config_path": "other.yaml", "harbor_target": setup_target}]
    main_path.write_text(dump_manifest(raw), encoding="utf-8")

    if valid:
        assert validate_environment(main_path, name="harbor_demo/other").name == "harbor_demo/other"
    else:
        with pytest.raises(EnvironmentValidationError, match="another Harbor target"):
            validate_environment(main_path, name="harbor_demo/other")
