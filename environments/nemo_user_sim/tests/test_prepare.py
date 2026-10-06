# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environments.nemo_user_sim import prepare as prepare_module
from nemo_gym.config_types import DatasetConfig
from nemo_gym.task_materialization import materialize_task
from resources_servers.nemo_user_sim.episode_contracts import UserSimEpisodeRequest


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
    config = OmegaConf.load("environments/nemo_user_sim/config.yaml")
    datasets = config.nemo_user_sim_resources.resources_servers.nemo_user_sim.datasets
    [example] = [dataset for dataset in datasets if dataset.type == "example"]
    assert example.name == "example"
    assert example.taskset == "nemo_user_sim:example"
    [raw_validation] = [dataset for dataset in datasets if dataset.type == "validation"]
    validation = DatasetConfig.model_validate(raw_validation)
    assert validation.name == "nemo_user_sim"
    assert validation.jsonl_fpath == "environments/nemo_user_sim/data/nemo_user_sim.jsonl"
    assert validation.taskset == "nemo_user_sim:validation"


def test_probe_seed_and_task_id_are_stable() -> None:
    assert prepare_module._probe_seed(1042, "financial_services") == 1199748067
    assert prepare_module._probe_seed(1042, "general_educational") == 2052074905
    assert prepare_module._task_id({"trajectory_id": "usersim-financial_services"}) == "usersim-financial_services"
    with pytest.raises(ValueError, match="non-empty string trajectory_id"):
        prepare_module._task_id({"trajectory_id": ""})


def test_prepare_materializes_every_registered_probe_with_usersim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_path = tmp_path / "usersim.jsonl"
    monkeypatch.setattr(prepare_module, "TASKS_FPATH", tasks_path)
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
                        "persona": {"source": "usersim"},
                        "theme": {"source": "usersim"},
                        "trajectory_id": f"usersim-{probe}",
                        "usersim_provenance": {"code_sha": prepare_module.USERSIM_REVISION},
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
    assert all(
        row["resolved_row"]["usersim_provenance"]["code_sha"] == prepare_module.USERSIM_REVISION for row in rows
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
    assert 'alias = "assistant_model", model = "policy_model"' in model_configs[0]
    assert 'alias = "judge_model", model = "support_model"' in model_configs[0]
    assert kwargs["env"]["USERSIM_CODE_SHA"] == prepare_module.USERSIM_REVISION
    assert Path(str(kwargs["cwd"])).name.startswith("usersim-materialize-")
