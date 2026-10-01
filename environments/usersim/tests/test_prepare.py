# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environments.usersim import prepare as prepare_module


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


def test_environment_config_resolves_one_resources_owned_example_dataset() -> None:
    config_path = Path("environments/usersim/config.yaml")
    config = OmegaConf.merge(
        OmegaConf.load(config_path),
        {
            "support_model_base_url": "http://support.invalid/v1",
            "support_model_api_key": "test-key",
            "support_model_name": "test-support-model",
        },
    )

    [dataset] = config.usersim_resources.resources_servers.usersim.datasets

    assert dataset.name == "example"
    assert dataset.type == "example"
    assert dataset.agent == "usersim_assistant"
    assert dataset.prepare_script == "environments/usersim/prepare.py"


def test_participant_agents_use_independent_model_servers() -> None:
    config = OmegaConf.load("resources_servers/usersim/configs/usersim.yaml")
    raw_config = OmegaConf.to_container(config, resolve=False)
    assert isinstance(raw_config, dict)

    assert config.usersim_user.responses_api_agents.simple_agent.model_server.name == "user_model"
    assert config.usersim_assistant.responses_api_agents.simple_agent.model_server.name == "policy_model"
    assert raw_config["user_model"]["responses_api_models"]["vllm_model"]["base_url"] == "${user_model_base_url}"
    assert raw_config["policy_model"]["responses_api_models"]["vllm_model"]["base_url"] == "${policy_base_url}"
    assert raw_config["support_model"]["responses_api_models"]["vllm_model"]["base_url"] == "${support_model_base_url}"


def test_prepare_materializes_every_registered_probe_with_usersim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_path = tmp_path / "example.jsonl"
    monkeypatch.setattr(prepare_module, "TASKS_FPATH", tasks_path)
    monkeypatch.setattr(prepare_module.shutil, "which", lambda executable: f"/bin/{executable}")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_materialize(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        output = Path(command[-1])
        output.write_text(
            "".join(
                json.dumps(
                    {
                        "probe_type": probe,
                        "probe_family": f"native-{probe}",
                        "probe_variant": "usersim-resolved",
                        "persona": {"source": "usersim"},
                        "theme": {"source": "usersim"},
                        "trajectory_id": f"usersim-{probe}",
                        "usersim_provenance": {"code_sha": prepare_module.USERSIM_REVISION},
                        "usersim_config": {"random_seed": 1042 + index},
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
    assert all(row["task_id"]["taskset"] == "usersim:example" for row in rows)
    assert {row["task_input"]["resolved_row"]["probe_type"] for row in rows} == set(REGISTERED_PROBES)
    assert all(row["task_input"]["resolved_row"]["persona"] == {"source": "usersim"} for row in rows)
    assert all(row["task_input"]["resolved_row"]["theme"] == {"source": "usersim"} for row in rows)
    assert all(
        row["task_input"]["resolved_row"]["usersim_provenance"]["code_sha"] == prepare_module.USERSIM_REVISION
        for row in rows
    )
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
    assert "materialize_episode_inputs" in command[9]
    assert "known_probes" in command[9]
    assert kwargs["env"]["USERSIM_CODE_SHA"] == prepare_module.USERSIM_REVISION
    assert Path(str(kwargs["cwd"])).name.startswith("usersim-materialize-")


def test_gym_does_not_author_sampling_content() -> None:
    source = Path(prepare_module.__file__).read_text()

    assert "_stable_index" not in source
    assert "example_source.jsonl" not in source
    assert "personas_panel" not in source
