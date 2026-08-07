# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from nemo_gym.environment_manifest import EnvironmentKind, dump_manifest, load_manifest
from nemo_gym.environment_scaffold import scaffold_environment
from nemo_gym.environment_validation import EnvironmentValidationError, validate_environment


@dataclass(frozen=True)
class _Entry:
    name: str
    kind: EnvironmentKind
    path: Path
    config_path: Path | None
    manifest_path: Path | None


def _manifest(*, kind: str = "environment") -> dict:
    root = "benchmarks/demo" if kind == "benchmark" else "environments/demo"
    dataset = {
        "name": "example",
        "type": "example",
        "jsonl_fpath": f"{root}/data/example.jsonl",
        "num_repeats": 1,
    }
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "kind": kind,
        "integration_profile": "stock-loop",
        "domain": "math",
        "description": "A small exact-match evaluation.",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["contributor"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "demo",
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [dataset],
        "grading_mode": "exact",
    }
    if kind == "benchmark":
        dataset.update(
            {
                "type": "benchmark",
                "prepare_script": f"{root}/prepare.py",
                "prompt_config": f"{root}/prompts/default.yaml",
            }
        )
        manifest.update(canonical_split="test", standard_prompt_config=f"{root}/prompts/default.yaml")
    return manifest


def _config(*, kind: str = "environment") -> str:
    root = "benchmarks/demo" if kind == "benchmark" else "environments/demo"
    dataset = f"""      - name: example
        type: example
        jsonl_fpath: {root}/data/example.jsonl
"""
    if kind == "benchmark":
        dataset = f"""      - name: example
        type: benchmark
        jsonl_fpath: {root}/data/example.jsonl
        prepare_script: {root}/prepare.py
        prompt_config: {root}/prompts/default.yaml
"""
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
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
{dataset}"""


def _asset(tmp_path: Path, *, kind: str = "environment") -> _Entry:
    directory = tmp_path / ("benchmarks" if kind == "benchmark" else "environments") / "demo"
    (directory / "data").mkdir(parents=True)
    (directory / "config.yaml").write_text(_config(kind=kind), encoding="utf-8")
    (directory / "manifest.yaml").write_text(dump_manifest(_manifest(kind=kind)), encoding="utf-8")
    if kind == "environment":
        rows = [{"responses_create_params": {"input": "What is 1 + 1?"}, "expected_answer": "2"}]
    else:
        rows = [{"question": "What is 1 + 1?", "expected_answer": "2"}]
        (directory / "prompts").mkdir()
        (directory / "prompts/default.yaml").write_text("user: '{question}'\n", encoding="utf-8")
        (directory / "prepare.py").write_text(
            "from pathlib import Path\n\ndef prepare(output: Path) -> Path:\n    return output\n",
            encoding="utf-8",
        )
    (directory / "data/example.jsonl").write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    return _Entry(
        name="demo",
        kind=EnvironmentKind(kind),
        path=directory,
        config_path=directory / "config.yaml",
        manifest_path=directory / "manifest.yaml",
    )


def test_validates_resolved_composition_and_final_environment_rows(tmp_path: Path) -> None:
    report = validate_environment(_asset(tmp_path))

    assert report.name == "demo"
    assert report.datasets[0].rows == 1
    assert [(item.role, item.implementation) for item in report.components] == [
        ("resources_server", "demo"),
        ("agent_server", "simple_agent"),
        ("model_server", "runtime-selected"),
    ]


def test_validates_benchmark_prompt_without_executing_prepare(tmp_path: Path) -> None:
    entry = _asset(tmp_path, kind="benchmark")
    report = validate_environment(entry)

    assert report.datasets[0].type == "benchmark"
    assert report.datasets[0].prompt_config.endswith("prompts/default.yaml")

    entry.path.joinpath("prepare.py").write_text(
        "def prepare(source: Path) -> Path:\n    raise RuntimeError('must not execute')\n",
        encoding="utf-8",
    )
    assert validate_environment(entry).datasets[0].rows == 1


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("async def prepare() -> Path:\n    pass\n", "synchronous prepare"),
        ("def prepare() -> str:\n    return 'output'\n", "return pathlib.Path"),
        ("def prepare(\n", "Could not parse dataset prepare script"),
    ],
)
def test_prepare_contract_matches_runtime(source: str, message: str, tmp_path: Path) -> None:
    entry = _asset(tmp_path, kind="benchmark")
    entry.path.joinpath("prepare.py").write_text(source, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(entry)


def test_benchmark_standard_prompt_matches_dataset_prompt(tmp_path: Path) -> None:
    entry = _asset(tmp_path, kind="benchmark")
    manifest = load_manifest(entry.manifest_path)
    entry.manifest_path.write_text(
        dump_manifest(manifest.model_copy(update={"standard_prompt_config": "benchmarks/demo/prompts/other.yaml"})),
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentValidationError, match="standard_prompt_config"):
        validate_environment(entry)


def test_stale_mirror_is_reported_and_sync_changes_only_mirror_fields(tmp_path: Path) -> None:
    entry = _asset(tmp_path)
    manifest = load_manifest(entry.manifest_path)
    stale_dataset = manifest.datasets[0].model_copy(update={"jsonl_fpath": "wrong.jsonl"})
    stale = manifest.model_copy(update={"resources_server": "wrong", "datasets": [stale_dataset]})
    entry.manifest_path.write_text(dump_manifest(stale), encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match="resources_server") as caught:
        validate_environment(entry)
    assert "ManifestDataset" not in str(caught.value)
    assert '"jsonl_fpath": "wrong.jsonl"' in str(caught.value)

    report = validate_environment(entry, sync=True)
    synchronized = load_manifest(entry.manifest_path)
    assert report.synchronized_fields == ("resources_server", "datasets")
    assert synchronized.resources_server == "demo"
    assert synchronized.description == manifest.description


def test_rejects_legacy_entry_and_missing_config(tmp_path: Path) -> None:
    entry = _asset(tmp_path)
    legacy = _Entry(entry.name, entry.kind, entry.path, entry.config_path, None)
    with pytest.raises(EnvironmentValidationError, match="legacy environment"):
        validate_environment(legacy)

    no_config = _Entry(entry.name, entry.kind, entry.path, None, entry.manifest_path)
    with pytest.raises(EnvironmentValidationError, match="no config.yaml"):
        validate_environment(no_config)

    entry.path.joinpath("data/example.jsonl").unlink()
    with pytest.raises(EnvironmentValidationError, match="Dataset file was not found"):
        validate_environment(entry)


def test_rejects_invalid_materialized_rollout_input(tmp_path: Path) -> None:
    entry = _asset(tmp_path)
    entry.path.joinpath("data/example.jsonl").write_text(
        '{"responses_create_params": {"input": 42}}\n',
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentValidationError, match="responses_create_params.input"):
        validate_environment(entry)


def test_benchmark_prompt_errors_are_reported_as_validation_errors(tmp_path: Path) -> None:
    entry = _asset(tmp_path, kind="benchmark")
    entry.path.joinpath("data/example.jsonl").write_text(
        '{"question": "bad", "responses_create_params": "not-a-mapping"}\n',
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentValidationError, match="Could not materialize benchmark dataset"):
        validate_environment(entry)


@pytest.mark.parametrize(
    ("content", "message"),
    [("", "is empty"), ("[broken\n", "Malformed JSON"), ("[]\n", "must contain a JSON object")],
)
def test_rejects_empty_or_malformed_jsonl(tmp_path: Path, content: str, message: str) -> None:
    entry = _asset(tmp_path)
    entry.path.joinpath("data/example.jsonl").write_text(content, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(entry)


def test_requires_exactly_one_dataset_agent(tmp_path: Path) -> None:
    entry = _asset(tmp_path)
    entry.config_path.write_text(
        entry.config_path.read_text(encoding="utf-8")
        + """\
other_agent:
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
      - name: other
        type: example
        jsonl_fpath: environments/demo/data/example.jsonl
""",
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentValidationError, match="exactly one agent instance with datasets"):
        validate_environment(entry)


def test_custom_driver_module_must_exist(tmp_path: Path) -> None:
    result = scaffold_environment(root=tmp_path, kind="environment", name="driver", profile="custom-driver")
    entry = _Entry(
        name="driver",
        kind=EnvironmentKind.ENVIRONMENT,
        path=result.asset_dir,
        config_path=result.asset_dir / "config.yaml",
        manifest_path=result.asset_dir / "manifest.yaml",
    )

    validate_environment(entry)
    driver = result.asset_dir / "rollout_driver.py"
    driver.write_text(
        driver.read_text(encoding="utf-8").replace("run_rollout_collection", "other_function"),
        encoding="utf-8",
    )
    with pytest.raises(EnvironmentValidationError, match="Rollout driver.*was not found"):
        validate_environment(entry)

    result.asset_dir.joinpath("rollout_driver.py").unlink()
    with pytest.raises(EnvironmentValidationError, match="Rollout driver module was not found"):
        validate_environment(entry)
