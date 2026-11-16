# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A server on a tiny dataset for tests that run /verify end to end."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
    AABriefcaseLiteVerifyRequest,
    AABriefcaseReference,
)
from resources_servers.gdpval.app import GDPValResourcesServer
from resources_servers.gdpval.judge_panel import ResolvedJudge


NO_SESSION = SimpleNamespace(session={})
TASK_ID = "w1_t1"


def check(check_id: str, scoring_type: str, files: str = "report.md") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "task_id": TASK_ID,
        "scoring_type": scoring_type,
        "check_type": "A" if scoring_type == "binary" else "AQ",
        "check_description": check_id,
        "score_1_criteria": "pass",
        "score_0_criteria": "fail",
        "taskdoer_output_file": files,
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "tasks").mkdir(parents=True)
    (root / "tasks" / f"{TASK_ID}.md").write_text("Write the report.")
    checks = [check("a1", "binary"), check("a2", "binary"), check("q1", "pairwise")]
    (root / "checks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in checks))
    return root


@pytest.fixture
def references(tmp_path: Path) -> dict[str, AABriefcaseReference]:
    """Two pairwise opponents that each submitted report.md."""
    output = {}
    for name in ("ref_a", "ref_b"):
        directory = tmp_path / name / f"task_{TASK_ID}" / "repeat_0"
        directory.mkdir(parents=True)
        (directory / "report.md").write_text(f"{name} report")
        output[name] = AABriefcaseReference(deliverables_dir=str(tmp_path / name))
    return output


@pytest.fixture
def make_server(
    tmp_path: Path, dataset: Path, references, monkeypatch
) -> Callable[..., AABriefcaseLiteResourcesServer]:
    def make(**overrides: Any) -> AABriefcaseLiteResourcesServer:
        monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
        config = AABriefcaseLiteResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            dataset_dir=str(dataset),
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            preconvert_office_to_pdf=False,
            sandbox_provider="sandbox",
            sandbox_config={},
            persist_deliverables_dir=str(tmp_path / "persist"),
            reward_mode=overrides.pop("reward_mode", "all"),
            pairwise_references=overrides.pop("pairwise_references", references),
            pairwise_num_trials=2,
            **overrides,
        )
        server = AABriefcaseLiteResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        # The skipped model_post_init also skips GDPVal's session bookkeeping and the dataset loading.
        GDPValResourcesServer.model_post_init(server, None)
        server._aa_dataset_root = dataset
        server._aa_checks = [json.loads(line) for line in (dataset / "checks.jsonl").read_text().splitlines()]
        server._aa_binary_system = "system"
        server._aa_binary_user = (
            "{task_markdown}{check_description}{score_1_criteria}{score_0_criteria}<<<SUBMISSION CONTENT MESSAGES>>>"
        )
        judge = ResolvedJudge(name="judge", model="m", base_url="http://upstream.invalid/v1", api_key="dummy")
        monkeypatch.setattr(server, "_resolve_judges", lambda: [judge])
        return server

    return make


def deliverables(tmp_path: Path, files: dict[str, str]) -> str:
    directory = tmp_path / "submitted"
    directory.mkdir(exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text)
    return str(directory)


def verify_body(deliverables_dir: str) -> AABriefcaseLiteVerifyRequest:
    return AABriefcaseLiteVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        task_id=TASK_ID,
        deliverables_dir=deliverables_dir,
    )


def pairwise_result(wins: int, losses: int, ties: int = 0, invalid: int = 0) -> dict[str, Any]:
    return {"win_count_a": wins, "win_count_b": losses, "tie_count": ties, "invalid_count": invalid}
