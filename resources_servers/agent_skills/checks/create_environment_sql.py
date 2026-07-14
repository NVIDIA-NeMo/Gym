# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hidden checks for the create-environment SQL resources-server task."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


SERVER_NAME = "sql_generation"
REQUIRED_PATHS = (
    "app.py",
    "configs/sql_generation.yaml",
    "tests/test_app.py",
    "data/example.jsonl",
    "requirements.txt",
    "README.md",
)


def check_required_files(server_dir: Path) -> None:
    missing = [path for path in REQUIRED_PATHS if not (server_dir / path).is_file()]
    if missing:
        raise AssertionError(f"Missing required resources-server files: {', '.join(missing)}")


def check_example_data(server_dir: Path, source_tasks_path: Path | None = None) -> None:
    rows = []
    for line_number, line in enumerate((server_dir / "data/example.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"example.jsonl line {line_number} is invalid JSON: {exc}") from exc
        responses_params = row.get("responses_create_params") or {}
        if not responses_params.get("input"):
            raise AssertionError(f"example.jsonl line {line_number} has no responses_create_params.input")
        metadata = row.get("verifier_metadata")
        if not isinstance(metadata, dict) or "expected_rows" not in metadata:
            raise AssertionError(f"example.jsonl line {line_number} has no verifier_metadata.expected_rows")
        rows.append(row)
    if len(rows) < 5:
        raise AssertionError(f"example.jsonl must contain at least 5 tasks, found {len(rows)}")
    if source_tasks_path is not None:
        source_rows = [json.loads(line) for line in source_tasks_path.read_text().splitlines() if line.strip()]
        converted_inputs = [json.dumps(row["responses_create_params"]["input"]) for row in rows]
        for source_row in source_rows:
            matching_rows = [
                row for row, model_input in zip(rows, converted_inputs) if source_row["question"] in model_input
            ]
            if not matching_rows:
                raise AssertionError(f"example.jsonl did not convert source question: {source_row['question']}")
            if not any(
                row["verifier_metadata"]["expected_rows"] == source_row["expected_rows"] for row in matching_rows
            ):
                raise AssertionError(f"example.jsonl did not preserve expected rows for: {source_row['question']}")


def check_config(server_dir: Path) -> None:
    config = yaml.safe_load((server_dir / "configs/sql_generation.yaml").read_text())
    resources = (config.get("sql_generation") or {}).get("resources_servers") or {}
    server_config = resources.get(SERVER_NAME)
    if not isinstance(server_config, dict):
        raise AssertionError("Config must define sql_generation.resources_servers.sql_generation")
    if server_config.get("entrypoint") != "app.py":
        raise AssertionError("Resources server entrypoint must be app.py")
    if not server_config.get("domain"):
        raise AssertionError("Resources server config must define domain")
    if server_config.get("database_path") != "/task_inputs/shop.db":
        raise AssertionError("Resources server config must set database_path to /task_inputs/shop.db")

    agent = (config.get("sql_generation_simple_agent") or {}).get("responses_api_agents") or {}
    simple_agent = agent.get("simple_agent")
    if not isinstance(simple_agent, dict):
        raise AssertionError("Config must define sql_generation_simple_agent.responses_api_agents.simple_agent")
    if ((simple_agent.get("resources_server") or {}).get("name")) != SERVER_NAME:
        raise AssertionError("Simple agent must reference the sql_generation resources server")
    datasets = simple_agent.get("datasets") or []
    if not any(dataset.get("jsonl_fpath", "").endswith("data/example.jsonl") for dataset in datasets):
        raise AssertionError("Simple agent config must include the example dataset")


def check_source_conventions(server_dir: Path) -> None:
    source = (server_dir / "app.py").read_text()
    if "httpx" in source:
        raise AssertionError("Async Gym resources servers must not use httpx")
    if "ray.get(" in source:
        raise AssertionError("Async Gym resources servers must not call ray.get()")
    if "async def verify" not in source:
        raise AssertionError("verify() must be async")
    for required_symbol in (
        "SQLGenerationResourcesServerConfig",
        "SQLGenerationVerifyRequest",
        "SQLGenerationVerifyResponse",
        "SQLGenerationResourcesServer",
    ):
        if required_symbol not in source:
            raise AssertionError(f"app.py must define {required_symbol}")


def check_public_tests(server_dir: Path) -> None:
    source = (server_dir / "tests/test_app.py").read_text()
    tree = ast.parse(source)
    tests = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]
    if len(tests) < 4:
        raise AssertionError("Public tests must define at least four test cases")
    verify_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "verify"
    ]
    if not verify_calls:
        raise AssertionError("Public tests must call the resources server verify method")
    reward_references = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "reward"]
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    if len(reward_references) < 4 or len(assertions) < 4:
        raise AssertionError("Public tests must assert verifier rewards across behavioral cases")


def run_submission_contract(workspace: Path) -> None:
    contract = r'''
import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient

server_dir = Path(sys.argv[1])
sys.path.insert(0, str(server_dir))
from app import (
    SQLGenerationResourcesServer,
    SQLGenerationResourcesServerConfig,
    SQLGenerationVerifyRequest,
)

def response(text):
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                role="assistant",
                type="message",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

with tempfile.TemporaryDirectory() as temp_dir:
    database_path = Path(temp_dir) / "shop.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');
        """
    )
    connection.close()

    config = SQLGenerationResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="sql_generation",
        database_path=str(database_path),
    )
    server = SQLGenerationResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )

    async def verify(text, expected_rows):
        request = SQLGenerationVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="query"),
            response=response(text),
            verifier_metadata={"expected_rows": expected_rows},
        )
        return await server.verify(request)

    correct = asyncio.run(verify("```sql\nSELECT name FROM users WHERE id = 1;\n```", [["Alice"]]))
    assert correct.reward == 1.0, correct

    wrong = asyncio.run(verify("SELECT name FROM users WHERE id = 2;", [["Alice"]]))
    assert wrong.reward == 0.0, wrong

    malformed = asyncio.run(verify("This is not SQL.", [["Alice"]]))
    assert malformed.reward == 0.0, malformed

    empty = asyncio.run(verify("", [["Alice"]]))
    assert empty.reward == 0.0, empty

    execution_error = asyncio.run(verify("SELECT * FROM table_that_does_not_exist;", []))
    assert execution_error.reward == 0.0, execution_error

    for destructive_sql in (
        "DELETE FROM users;",
        "UPDATE users SET name = 'Mallory' WHERE id = 1;",
        "DROP TABLE users;",
        "INSERT INTO users (id, name) VALUES (3, 'Mallory');",
        "ALTER TABLE users ADD COLUMN email TEXT;",
        "ATTACH DATABASE '/tmp/other.db' AS other;",
        "PRAGMA writable_schema = ON;",
    ):
        mutation = asyncio.run(verify(destructive_sql, []))
        assert mutation.reward == 0.0, mutation
        connection = sqlite3.connect(database_path)
        assert connection.execute("SELECT name FROM users ORDER BY id").fetchall() == [('Alice',), ('Bob',)]
        connection.close()

print("SQL_GENERATION_CONTRACT_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", contract, str(workspace / "resources_servers" / SERVER_NAME)],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0 or "SQL_GENERATION_CONTRACT_OK" not in result.stdout:
        raise AssertionError(
            "SQL resources-server contract failed:\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )


def run_public_workflows(workspace: Path) -> None:
    python = workspace / ".venv/bin/python"
    gym = workspace / ".venv/bin/gym"
    commands = [
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "resources_servers/sql_generation/tests/test_app.py",
        ],
    ]
    with tempfile.TemporaryDirectory(prefix="nemo_gym_sql_collate_") as output_dir:
        commands.append(
            [
                str(gym),
                "dataset",
                "collate",
                "--config",
                "resources_servers/sql_generation/configs/sql_generation.yaml",
                "--output-dir",
                output_dir,
                "--mode",
                "example_validation",
            ]
        )
        for command in commands:
            result = subprocess.run(
                command,
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=180,
            )
            if result.returncode != 0:
                raise AssertionError(
                    f"Public validation command failed: {' '.join(command)}\n"
                    f"stdout:\n{result.stdout[-4000:]}\n"
                    f"stderr:\n{result.stderr[-4000:]}"
                )


def main() -> int:
    workspace_value = os.environ.get("NEMO_GYM_WORKSPACE")
    if not workspace_value:
        print("NEMO_GYM_WORKSPACE is not set", file=sys.stderr)
        return 2
    workspace = Path(workspace_value)
    server_dir = workspace / "resources_servers" / SERVER_NAME
    source_tasks_path = Path("/task_inputs/sql_tasks.jsonl")

    checks = (
        ("required_files", lambda: check_required_files(server_dir)),
        ("example_data", lambda: check_example_data(server_dir, source_tasks_path)),
        ("config", lambda: check_config(server_dir)),
        ("source_conventions", lambda: check_source_conventions(server_dir)),
        ("public_tests", lambda: check_public_tests(server_dir)),
        ("submission_contract", lambda: run_submission_contract(workspace)),
        ("public_workflows", lambda: run_public_workflows(workspace)),
    )
    outcomes: dict[str, bool] = {}
    for name, check in checks:
        try:
            check()
        except Exception as exc:
            outcomes[name] = False
            print(json.dumps({"status": "fail", "check": name, "reason": str(exc)}))
        else:
            outcomes[name] = True
            print(json.dumps({"status": "pass", "check": name}))

    scores = {
        "correctness": float(outcomes.get("submission_contract", False)),
        "completeness": float(
            all(
                outcomes.get(name, False)
                for name in (
                    "required_files",
                    "example_data",
                    "config",
                    "public_tests",
                    "public_workflows",
                )
            )
        ),
        "convention_compliance": float(outcomes.get("source_conventions", False)),
    }
    scores["task_success"] = float(all(outcomes.values()))
    print(json.dumps({"status": "summary", "scores": scores}))
    return 0 if scores["task_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
