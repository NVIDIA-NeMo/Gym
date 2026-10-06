# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.stirrup_agent.app import get_task_strategy
from responses_api_agents.stirrup_agent.tasks.aa_briefcase_lite import AABriefcaseLiteTask, _dataset_path, _render


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    for directory in ("prompts", "tasks", "summary_docs/week_1", "source_files/shared", "source_files/week"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompts/eval_system.txt").write_text(
        "turns={max_turns}; finish={finish_tool_name}; abandon={abandon_task_finish}", encoding="utf-8"
    )
    (tmp_path / "prompts/eval_submission.txt").write_text(
        "{scenario_overview}\n{week_overview}\n{task}\n{expected_output_filenames}\n"
        "{finish_tool_name}\n{abandon_task_finish}",
        encoding="utf-8",
    )
    (tmp_path / "summary_docs/scenario_overview.md").write_text("scenario", encoding="utf-8")
    (tmp_path / "summary_docs/week_1/week_1_overview.md").write_text("week", encoding="utf-8")
    (tmp_path / "tasks/w1_t1.md").write_text("task", encoding="utf-8")
    return tmp_path


def metadata(dataset: Path) -> dict:
    return {
        "task_id": "w1_t1",
        "week": 1,
        "dataset_dir": str(dataset),
        "dataset_revision": "test-revision",
        "task_md_path": "tasks/w1_t1.md",
        "deliverable_filenames": ["market_overview.tex", "market_overview.pdf"],
        "scenario_overview_path": "summary_docs/scenario_overview.md",
        "week_overview_path": "summary_docs/week_1/week_1_overview.md",
        "shared_files": [],
        "week_files": [],
    }


def test_registry_returns_briefcase_strategy() -> None:
    assert isinstance(get_task_strategy("aa_briefcase_lite"), AABriefcaseLiteTask)


def test_official_prompts_are_rendered(dataset: Path) -> None:
    strategy = AABriefcaseLiteTask()
    task_info = strategy.extract_task_info(metadata(dataset))
    config = SimpleNamespace(agent_max_turns=500)

    assert strategy.build_system_prompt(task_info, config) == "turns=500; finish=finish; abandon=abandon_task_finish"
    user_prompt = strategy.build_user_prompt(task_info, config)
    assert "scenario\nweek\ntask" in user_prompt
    assert "- `market_overview.tex`\n- `market_overview.pdf`" in user_prompt


def test_runtime_options_match_offline_protocol() -> None:
    assert AABriefcaseLiteTask().runtime_options() == {
        "allow_web_tools": False,
        "require_exec_provider": True,
        "use_abandon_finish_tool": True,
        "skip_input_file_listing": True,
        "tool_response_as_user": False,
    }


def test_dataset_path_rejects_escape(dataset: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _dataset_path(dataset, "../outside")


def test_render_rejects_unknown_placeholder(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("known={known}; unknown={unknown}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\{unknown\}"):
        _render(prompt, {"known": "value"})


def test_exec_provider_has_read_only_mounts(dataset: Path, tmp_path: Path) -> None:
    image = tmp_path / "briefcase.sif"
    image.write_bytes(b"sif")
    strategy = AABriefcaseLiteTask()
    task_info = strategy.extract_task_info(metadata(dataset))
    provider = strategy.get_exec_provider(task_info, SimpleNamespace(aa_briefcase_container_path=str(image)))
    kwargs = provider._serializable_kwargs()

    assert kwargs["working_dir"] == "/home/user"
    assert kwargs["home_dir"] == "/home/user"
    assert kwargs["isolated_home"] is True
    assert kwargs["disable_network"] is True
    assert kwargs["env_passthrough"] == []
    assert kwargs["capture_git_diff"] is False
    assert kwargs["extra_mounts"] == [
        f"--mount type=bind,src={dataset / 'source_files/shared'},dst=/home/user/shared,ro",
        f"--mount type=bind,src={dataset / 'source_files/week'},dst=/home/user/week,ro",
    ]
