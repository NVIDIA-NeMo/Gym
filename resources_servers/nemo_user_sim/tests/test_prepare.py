# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environments.nemo_user_sim import prepare as prepare_module
from nemo_gym.config_types import DatasetConfig
from nemo_gym.task_materialization import materialize_task
from resources_servers.nemo_user_sim.app import UserSimResourcesServerConfig
from resources_servers.nemo_user_sim.episode_contracts import UserSimEpisodeRequest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REGISTERED_PROBES = (
    "financial_services",
    "general_educational",
    "general_open_ended",
    "health_decision_support_disclosure",
    "health_general_disclosure",
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "identity_disclosure",
    "safety_agentic",
    "safety_chat_pressure",
    "sov_ai_dynamic",
    "sov_ai_facts",
    "sov_ai_multilingual_parity",
    "tool_calling",
)


def test_environment_config_declares_generated_validation_split() -> None:
    config = OmegaConf.load(prepare_module.ENVIRONMENT_DIR / "config.yaml")
    datasets = config.nemo_user_sim_resources.resources_servers.nemo_user_sim.datasets
    [example] = [dataset for dataset in datasets if dataset.type == "example"]
    assert example.name == "example"
    assert example.taskset == "nemo_user_sim:example"
    [raw_validation] = [dataset for dataset in datasets if dataset.type == "validation"]
    validation = DatasetConfig.model_validate(raw_validation)
    assert validation.name == "nemo_user_sim"
    assert validation.jsonl_fpath == "environments/nemo_user_sim/data/nemo_user_sim.jsonl"
    assert validation.prepare_script == Path("environments/nemo_user_sim/prepare.py")
    assert validation.taskset == "nemo_user_sim:validation"


def test_usersim_revision_pins_are_aligned() -> None:
    requirement_paths = (
        prepare_module.PREPARE_REQUIREMENTS_FPATH,
        REPOSITORY_ROOT / "environment_servers/nemo_user_sim/requirements.txt",
        REPOSITORY_ROOT / "resources_servers/nemo_user_sim/requirements.txt",
    )
    requirement_revisions = []
    for path in requirement_paths:
        match = re.search(r"UserSim\.git@([0-9a-f]{40})", path.read_text())
        assert match is not None, f"Missing UserSim revision in {path}"
        requirement_revisions.append(match.group(1))

    resources_config = OmegaConf.load(REPOSITORY_ROOT / "resources_servers/nemo_user_sim/configs/nemo_user_sim.yaml")
    configured_revision = resources_config.nemo_user_sim_resources.resources_servers.nemo_user_sim.usersim_revision
    server_default = UserSimResourcesServerConfig.model_fields["usersim_revision"].default
    assert set(requirement_revisions + [prepare_module.USERSIM_REVISION, configured_revision, server_default]) == {
        prepare_module.USERSIM_REVISION
    }
    configured_personas_version = (
        resources_config.nemo_user_sim_resources.resources_servers.nemo_user_sim.nemotron_personas_version
    )
    personas_version_default = UserSimResourcesServerConfig.model_fields["nemotron_personas_version"].default
    assert configured_personas_version == personas_version_default == prepare_module.NEMOTRON_PERSONAS_VERSION


def test_probe_seed_and_task_id_are_stable() -> None:
    assert prepare_module._probe_seed(1042, "financial_services") == 1199748067
    assert prepare_module._probe_seed(1042, "general_educational") == 2052074905
    assert prepare_module._task_id({"trajectory_id": "usersim-financial_services"}) == "usersim-financial_services"
    with pytest.raises(ValueError, match="non-empty string trajectory_id"):
        prepare_module._task_id({"trajectory_id": ""})


def test_validate_persona_asset_requires_pinned_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    managed_assets_path = tmp_path / "managed-assets"
    datasets_path = managed_assets_path / "datasets"
    datasets_path.mkdir(parents=True)
    monkeypatch.setenv("DATA_DESIGNER_MANAGED_ASSETS_PATH", str(managed_assets_path))
    with pytest.raises(
        RuntimeError,
        match=re.escape(prepare_module.NEMOTRON_PERSONAS_DOWNLOAD_COMMAND),
    ):
        prepare_module._validate_persona_asset("en_US")

    asset_path = datasets_path / "en_US.parquet"
    asset_path.write_bytes(b"wrong asset")
    with pytest.raises(RuntimeError, match="expected"):
        prepare_module._validate_persona_asset("en_US")

    expected_asset = b"pinned asset"
    asset_path.write_bytes(expected_asset)
    monkeypatch.setattr(
        prepare_module,
        "NEMOTRON_PERSONAS_SHA256",
        prepare_module.hashlib.sha256(expected_asset).hexdigest(),
    )
    assert prepare_module._validate_persona_asset("en_US") == managed_assets_path


def test_download_persona_asset_installs_verified_ngc_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed_assets_path = tmp_path / "managed-assets"
    expected_asset = b"pinned asset"
    monkeypatch.setattr(
        prepare_module,
        "NEMOTRON_PERSONAS_SHA256",
        prepare_module.hashlib.sha256(expected_asset).hexdigest(),
    )
    calls: list[list[str]] = []

    def fake_download(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        download_dir = Path(command[command.index("--dest") + 1])
        resource_dir = download_dir / "nemotron-personas-dataset-en_us_v0.0.2"
        resource_dir.mkdir()
        (resource_dir / "en_US.parquet").write_bytes(expected_asset)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_download)

    result = prepare_module._download_persona_asset(
        ngc_executable=Path("/opt/ngc"),
        locale="en_US",
        managed_assets_path=managed_assets_path,
    )

    assert calls[0][:5] == [
        "/opt/ngc",
        "registry",
        "resource",
        "download-version",
        prepare_module.NEMOTRON_PERSONAS_RESOURCE,
    ]
    assert result == managed_assets_path / "datasets" / "en_US.parquet"
    assert result.read_bytes() == expected_asset
    assert list(result.parent.glob(".*.tmp")) == []


def test_materialization_bootstraps_missing_persona_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    managed_assets_path = tmp_path / "managed-assets"
    models_path = tmp_path / "models.toml"
    destination = tmp_path / "resolved.jsonl"
    models_path.write_text("")
    download_calls: list[tuple[Path, str, Path]] = []

    def fake_download(*, ngc_executable: Path, locale: str, managed_assets_path: Path) -> Path:
        download_calls.append((ngc_executable, locale, managed_assets_path))
        asset_path = managed_assets_path / "datasets" / f"{locale}.parquet"
        asset_path.parent.mkdir(parents=True)
        asset_path.write_bytes(b"pinned asset")
        return asset_path

    def fake_materialize_episode_inputs(**kwargs: object) -> list[dict[str, object]]:
        assert (managed_assets_path / "datasets" / "en_US.parquet").is_file()
        return [{"trajectory_id": f"usersim-{kwargs['random_seed']}"}]

    monkeypatch.setattr(prepare_module, "_download_persona_asset", fake_download)
    package_names = ("usersim", "usersim.cli", "usersim.engine", "usersim.engine.core")
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    ngc_module = types.ModuleType("usersim.cli._ngc")
    ngc_module.ensure_ngc_cli = lambda: Path("/opt/ngc")
    ngc_module.ensure_ngc_org = lambda: "test-org"
    ngc_module.has_ngc_key = lambda: True
    monkeypatch.setitem(sys.modules, "usersim.cli._ngc", ngc_module)
    probes_module = types.ModuleType("usersim.engine.core.probes")
    probes_module.known_probes = lambda: ("general_open_ended",)
    monkeypatch.setitem(sys.modules, "usersim.engine.core.probes", probes_module)
    external_module = types.ModuleType("usersim.engine.external")
    external_module.materialize_episode_inputs = fake_materialize_episode_inputs
    monkeypatch.setitem(sys.modules, "usersim.engine.external", external_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "-c",
            str(REPOSITORY_ROOT),
            "en_US",
            "1042",
            str(managed_assets_path),
            str(models_path),
            str(destination),
        ],
    )

    exec(compile(prepare_module._MATERIALIZE_SCRIPT, "<materialize-script>", "exec"), {})

    assert download_calls == [(Path("/opt/ngc"), "en_US", managed_assets_path)]
    expected_seed = prepare_module._probe_seed(1042, "general_open_ended")
    assert json.loads(destination.read_text())["trajectory_id"] == f"usersim-{expected_seed}"


def test_managed_assets_path_matches_data_designer_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DESIGNER_HOME", str(tmp_path / "home"))
    assert prepare_module._managed_assets_path() == tmp_path / "home" / "managed-assets"

    monkeypatch.setenv("DATA_DESIGNER_MANAGED_ASSETS_PATH", str(tmp_path / "explicit"))
    assert prepare_module._managed_assets_path() == tmp_path / "explicit"


def test_write_tasks_removes_temporary_file_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_path = tmp_path / "nemo_user_sim.jsonl"
    monkeypatch.setattr(prepare_module, "TASKS_FPATH", tasks_path)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"Cannot replace {destination} with {source}")

    monkeypatch.setattr(prepare_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="Cannot replace"):
        prepare_module._write_tasks([{"task_id": "example", "resolved_row": {}}])

    assert list(tmp_path.iterdir()) == []


def test_prepare_materializes_every_registered_probe_with_usersim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_path = tmp_path / "nemo_user_sim.jsonl"
    managed_assets_path = tmp_path / "verified-managed-assets"
    monkeypatch.setattr(prepare_module, "TASKS_FPATH", tasks_path)
    monkeypatch.setattr(prepare_module, "_validate_persona_asset", lambda locale: managed_assets_path)
    monkeypatch.setattr(prepare_module.shutil, "which", lambda executable: f"/bin/{executable}")
    calls: list[tuple[list[str], dict[str, object]]] = []
    model_configs: list[str] = []

    def fake_materialize(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        model_configs.append(Path(command[-2]).read_text())
        output = Path(command[-1])
        output.write_text(
            "".join(
                json.dumps(
                    {
                        "probe_type": probe,
                        "probe_family": f"family-{probe}",
                        "probe_variant": "usersim-resolved",
                        "toolset_name": float("nan"),
                        "persona": {"source": "usersim"},
                        "theme": {"source": "usersim"},
                        "trajectory_id": f"usersim-{probe}",
                        "usersim_provenance": {
                            "code_sha": prepare_module.USERSIM_REVISION,
                            "nemotron_personas_version": prepare_module.NEMOTRON_PERSONAS_VERSION,
                        },
                        "usersim_config": {
                            "assets_dir": f"/tmp/usersim-assets-{index}",
                            "random_seed": 1042 + index,
                        },
                    }
                )
                + "\n"
                for index, probe in enumerate(REGISTERED_PROBES)
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_materialize)

    result = prepare_module.prepare(random_seed=1042)

    assert result == tasks_path.absolute()
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    assert len(rows) == 14
    assert [row["task_id"] for row in rows] == [f"usersim-{probe}" for probe in REGISTERED_PROBES]
    assert {row["resolved_row"]["probe_type"] for row in rows} == set(REGISTERED_PROBES)
    assert all(row["resolved_row"]["persona"] == {"source": "usersim"} for row in rows)
    assert all(row["resolved_row"]["theme"] == {"source": "usersim"} for row in rows)
    assert all(row["resolved_row"]["toolset_name"] is None for row in rows)
    assert "NaN" not in tasks_path.read_text()
    assert all(
        row["resolved_row"]["usersim_provenance"]["code_sha"] == prepare_module.USERSIM_REVISION for row in rows
    )
    assert all(
        row["resolved_row"]["usersim_provenance"]["nemotron_personas_version"]
        == prepare_module.NEMOTRON_PERSONAS_VERSION
        for row in rows
    )
    assert all("assets_dir" not in row["resolved_row"]["usersim_config"] for row in rows)
    for index, row in enumerate(rows):
        task = materialize_task(row, taskset="nemo_user_sim:validation", task_index=index)
        request = UserSimEpisodeRequest.model_validate(
            {"episode_id": {"rollout_id": f"rollout-{index}", "attempt": 0}, "task": task}
        )
        assert request.task.task_id.taskset == "nemo_user_sim:validation"
        assert request.task.task_id.task_id == row["task_id"]
        assert request.task.task_input.resolved_row == row["resolved_row"]
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:9] == [
        "/bin/uv",
        "run",
        "--no-config",
        "--no-project",
        "--isolated",
        "--with-requirements",
        str(prepare_module.PREPARE_REQUIREMENTS_FPATH),
        "python",
        "-c",
    ]
    assert command[10] == str(prepare_module.ENVIRONMENT_DIR.parents[1])
    assert 'alias = "assistant_model", model = "assistant_model"' in model_configs[0]
    assert 'alias = "judge_model", model = "judge_model"' in model_configs[0]
    assert kwargs["env"]["USERSIM_CODE_SHA"] == prepare_module.USERSIM_REVISION
    assert kwargs["env"]["USERSIM_NEMOTRON_PERSONAS_VERSION"] == prepare_module.NEMOTRON_PERSONAS_VERSION
    assert kwargs["env"]["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(managed_assets_path)
    assert Path(str(kwargs["cwd"])).name.startswith("usersim-materialize-")
