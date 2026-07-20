# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from resources_servers.agent_skills.checks.create_environment_sql import (
    check_config,
    check_example_data,
    check_public_tests,
    check_required_files,
    check_source_conventions,
    run_public_workflows,
    run_submission_contract,
)


GOOD_APP = r"""
import re
import sqlite3

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class SQLGenerationResourcesServerConfig(BaseResourcesServerConfig):
    database_path: str


class SQLGenerationVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict


class SQLGenerationVerifyResponse(BaseVerifyResponse):
    status: str


class SQLGenerationResourcesServer(SimpleResourcesServer):
    config: SQLGenerationResourcesServerConfig

    async def verify(self, body: SQLGenerationVerifyRequest) -> SQLGenerationVerifyResponse:
        text = body.response.output_text or ""
        match = re.search(r"(?:```sql\s*)?(SELECT|WITH)\b[\s\S]*?(?:```|$)", text, re.IGNORECASE)
        if not match:
            return SQLGenerationVerifyResponse(**body.model_dump(), reward=0.0, status="malformed")
        sql = match.group(0).replace("```sql", "").replace("```", "").strip()
        if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
            return SQLGenerationVerifyResponse(**body.model_dump(), reward=0.0, status="unsafe")
        try:
            connection = sqlite3.connect(f"file:{self.config.database_path}?mode=ro", uri=True)
            rows = [list(row) for row in connection.execute(sql).fetchall()]
            connection.close()
        except sqlite3.Error:
            return SQLGenerationVerifyResponse(**body.model_dump(), reward=0.0, status="execution_error")
        expected = body.verifier_metadata.get("expected_rows")
        reward = float(rows == expected)
        return SQLGenerationVerifyResponse(**body.model_dump(), reward=reward, status="pass" if reward else "wrong")
"""


def _write_fixture(workspace: Path) -> Path:
    server_dir = workspace / "resources_servers/sql_generation"
    (server_dir / "configs").mkdir(parents=True)
    (server_dir / "tests").mkdir()
    (server_dir / "data").mkdir()
    (server_dir / "app.py").write_text(GOOD_APP)
    (server_dir / "tests/test_app.py").write_text(
        """
async def test_correct_query(server, correct_request):
    result = await server.verify(correct_request)
    assert result.reward == 1.0

async def test_wrong_query(server, wrong_request):
    result = await server.verify(wrong_request)
    assert result.reward == 0.0

async def test_empty_query(server, empty_request):
    result = await server.verify(empty_request)
    assert result.reward == 0.0

async def test_unsafe_query(server, unsafe_request):
    result = await server.verify(unsafe_request)
    assert result.reward == 0.0
"""
    )
    (server_dir / "requirements.txt").write_text("")
    (server_dir / "README.md").write_text("# SQL Generation\n")
    (server_dir / "configs/sql_generation.yaml").write_text(
        """
sql_generation:
  resources_servers:
    sql_generation:
      entrypoint: app.py
      domain: coding
      database_path: /task_inputs/shop.db
sql_generation_simple_agent:
  responses_api_agents:
    simple_agent:
      resources_server:
        type: resources_servers
        name: sql_generation
      datasets:
        - name: example
          type: example
          jsonl_fpath: resources_servers/sql_generation/data/example.jsonl
"""
    )
    rows = [
        {
            "responses_create_params": {"input": [{"role": "user", "content": f"question {index}"}]},
            "verifier_metadata": {"expected_rows": [[index]]},
        }
        for index in range(5)
    ]
    (server_dir / "data/example.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return server_dir


def test_compliant_sql_environment_passes_hidden_checks(tmp_path: Path) -> None:
    server_dir = _write_fixture(tmp_path)

    check_required_files(server_dir)
    check_example_data(server_dir)
    check_config(server_dir)
    check_source_conventions(server_dir)
    check_public_tests(server_dir)
    run_submission_contract(tmp_path)


def test_source_conventions_reject_httpx(tmp_path: Path) -> None:
    server_dir = _write_fixture(tmp_path)
    (server_dir / "app.py").write_text(GOOD_APP + "\nimport httpx\n")

    with pytest.raises(AssertionError, match="must not use httpx"):
        check_source_conventions(server_dir)


def test_public_tests_reject_dummy_assertions(tmp_path: Path) -> None:
    server_dir = _write_fixture(tmp_path)
    (server_dir / "tests/test_app.py").write_text(
        "def test_placeholder():\n"
        "    verify = 'not called'\n"
        "    reward = 1.0\n"
        "    assert verify\n"
        "    assert reward\n"
        "    assert reward == 1.0\n"
        "    assert True\n"
    )

    with pytest.raises(AssertionError, match="at least four test cases"):
        check_public_tests(server_dir)


def test_example_data_requires_five_rows(tmp_path: Path) -> None:
    server_dir = _write_fixture(tmp_path)
    rows = (server_dir / "data/example.jsonl").read_text().splitlines()
    (server_dir / "data/example.jsonl").write_text("\n".join(rows[:4]) + "\n")

    with pytest.raises(AssertionError, match="at least 5 tasks"):
        check_example_data(server_dir)


def test_example_data_preserves_question_answer_pairing(tmp_path: Path) -> None:
    server_dir = _write_fixture(tmp_path)
    source_path = tmp_path / "source.jsonl"
    source_rows = [{"question": f"question {index}", "expected_rows": [[index]]} for index in range(5)]
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    converted = [json.loads(line) for line in (server_dir / "data/example.jsonl").read_text().splitlines()]
    converted[0]["verifier_metadata"]["expected_rows"] = [[1]]
    (server_dir / "data/example.jsonl").write_text("".join(json.dumps(row) + "\n" for row in converted))

    with pytest.raises(AssertionError, match="did not preserve expected rows"):
        check_example_data(server_dir, source_path)


def test_public_workflows_run_tests_and_collation(tmp_path: Path) -> None:
    with patch(
        "resources_servers.agent_skills.checks.create_environment_sql.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ) as run:
        run_public_workflows(tmp_path)

    commands = [call.args[0] for call in run.call_args_list]
    assert commands[0][1:4] == ["-m", "pytest", "-q"]
    assert commands[1][1:4] == ["dataset", "collate", "--config"]
