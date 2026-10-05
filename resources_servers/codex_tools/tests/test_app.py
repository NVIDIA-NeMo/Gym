# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.codex_tools.app import CodexToolsResourcesServer, CodexToolsResourcesServerConfig
from resources_servers.codex_tools.make_tasks import make_row


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")

_RESPONSE = {
    "id": "resp",
    "created_at": 0.0,
    "model": "m",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / ".gitignore").write_text("*.log\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return repo


def _client(repo: Path, **config: object) -> Iterator[TestClient]:
    server = CodexToolsResourcesServer(
        config=CodexToolsResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="codex_tools",
            repo_path=str(repo),
            login_shell=False,
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    with TestClient(server.setup_webserver()) as client:
        yield client


@pytest.fixture
def client(repo: Path) -> Iterator[TestClient]:
    yield from _client(repo)


def _row(check_command: str | None = None) -> dict:
    return make_row("Fix add().", check_command=check_command)


def _verify(client: TestClient, row: dict) -> dict:
    response = client.post("/verify", json={**row, "response": _RESPONSE})
    assert response.status_code == 200, response.text
    return response.json()


FIX_PATCH = "*** Begin Patch\n*** Update File: calc.py\n@@\n def add(a, b):\n-    return a - b\n+    return a + b\n*** End Patch"
CHECK = "python3 -c 'from calc import add; assert add(2, 3) == 5'"


def test_episode_in_worktree(client: TestClient, repo: Path) -> None:
    row = _row(CHECK)
    assert client.post("/seed_session", json=row).status_code == 200

    pwd = client.post("/exec_command", json={"cmd": "pwd"}).text
    workspace = Path(pwd.split("Output:\n", 1)[1].strip())
    assert workspace != repo and _git(workspace, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")

    patched = client.post("/apply_patch", json={"input": FIX_PATCH}).text
    assert patched.startswith("Exit code: 0\n") and patched.endswith(
        "Success. Updated the following files:\nM calc.py\n"
    )
    # The shell has an `apply_patch` command too, as Codex models sometimes expect.
    add_file = "apply_patch '*** Begin Patch\n*** Add File: notes.txt\n+hello\n*** End Patch'"
    assert "A notes.txt" in client.post("/exec_command", json={"cmd": add_file}).text
    client.post("/exec_command", json={"cmd": "echo ignored > debug.log"})
    bytecode = client.post(
        "/exec_command", json={"cmd": "python3 -c 'import calc' && touch stray.pyc && ls __pycache__"}
    )
    assert "calc." in bytecode.text  # bytecode really was written
    plan = {"plan": [{"step": "fix add", "status": "completed"}]}
    assert client.post("/update_plan", json=plan).text == "Plan updated"

    result = _verify(client, row)

    assert result["reward"] == 1.0 and result["resolved"] is True and result["check_exit_code"] == 0
    assert "-    return a - b\n+    return a + b" in result["diff"]
    assert "+++ b/notes.txt" in result["diff"] and "debug.log" not in result["diff"]
    # Bytecode from running code is excluded even though this repo does not ignore it.
    assert "__pycache__" not in result["diff"] and ".pyc" not in result["diff"]
    assert result["plan"] == {"explanation": None, **plan}
    # The real working tree is untouched and the worktree is removed.
    assert (repo / "calc.py").read_text().endswith("return a - b\n") and _git(repo, "status", "--porcelain") == ""
    assert not workspace.exists() and _git(repo, "worktree", "list").count("\n") == 1


def test_failed_check_and_no_changes(client: TestClient) -> None:
    row = _row(CHECK)
    client.post("/seed_session", json=row)
    result = _verify(client, row)
    assert result["reward"] == 0.0 and result["resolved"] is False and result["diff"] == ""
    assert "AssertionError" in result["check_output"]


def test_without_check_reward_is_zero_and_unresolved(client: TestClient) -> None:
    client.post("/seed_session", json=_row())
    client.post("/apply_patch", json={"input": FIX_PATCH})
    result = _verify(client, _row())
    assert result["reward"] == 0.0 and result["resolved"] is None and "+    return a + b" in result["diff"]


def test_in_place_edits_the_repository(repo: Path) -> None:
    for client in _client(repo, isolation="in_place"):
        client.post("/seed_session", json=_row(CHECK))
        assert client.post("/exec_command", json={"cmd": "pwd"}).text.endswith(f"{repo}\n")
        client.post("/apply_patch", json={"input": FIX_PATCH})
        # A second concurrent in-place session on the same repository is refused.
        other = TestClient(client.app)
        assert other.post("/seed_session", json=_row()).status_code == 409
        result = _verify(client, _row(CHECK))
    assert result["reward"] == 1.0 and result["workspace"] == str(repo)
    assert (repo / "calc.py").read_text().endswith("return a + b\n")
    # The diff was staged in a throwaway index, not the repository's own.
    assert _git(repo, "diff", "--cached") == ""


def test_commands_do_not_see_secrets(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-secret")  # pragma: allowlist secret
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "{}")
    monkeypatch.setenv("VISIBLE_SETTING", "ok")
    client.post("/seed_session", json=_row())
    output = client.post("/exec_command", json={"cmd": "env"}).text
    assert "VISIBLE_SETTING=ok" in output
    assert "sk-secret" not in output and "NEMO_GYM_CONFIG_DICT" not in output


def test_tool_errors_are_model_visible_text(client: TestClient) -> None:
    client.post("/seed_session", json=_row())
    assert client.post("/apply_patch", json={}).text == "failed to parse function arguments: missing field `input`"
    assert client.post("/write_stdin", json={"session_id": 5}).text == "write_stdin failed: Unknown process id 5"
    assert client.post("/update_plan", json={"plan": [{"step": 1}]}).text.startswith(
        "failed to parse function arguments"
    )
    missing = client.post("/apply_patch", json={"input": FIX_PATCH.replace("calc.py", "nope.py")}).text
    assert missing.startswith("apply_patch verification failed: Failed to read file to update ")


def test_verify_without_session_is_masked(client: TestClient) -> None:
    result = _verify(client, _row())
    assert result["mask_sample"] is True and result["error"] == "no workspace for this session"


def test_directory_outside_git_is_edited_in_place_without_diff(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    for client in _client(plain):  # the default worktree isolation does not apply without git
        assert client.post("/seed_session", json=_row(CHECK)).status_code == 200
        assert client.post("/exec_command", json={"cmd": "pwd"}).text.endswith(f"{plain}\n")
        assert client.post("/apply_patch", json={"input": FIX_PATCH}).text.startswith("Exit code: 0\n")
        # One in-place session per directory, as with isolation: in_place.
        assert TestClient(client.app).post("/seed_session", json=_row()).status_code == 409
        result = _verify(client, _row(CHECK))
    assert result["reward"] == 1.0 and result["resolved"] is True
    assert result["diff"] == "" and result.get("error") is None and result["workspace"] == str(plain)
    assert (plain / "calc.py").read_text().endswith("return a + b\n") and not (plain / ".git").exists()


def test_missing_repo_path_is_a_clear_error(tmp_path: Path) -> None:
    for client in _client(tmp_path / "missing"):
        response = client.post("/seed_session", json=_row())
    assert response.status_code == 400 and "is not a directory" in response.text


def test_shutdown_releases_unverified_workspaces(repo: Path) -> None:
    for client in _client(repo):
        client.post("/seed_session", json=_row())
        client.post("/exec_command", json={"cmd": "sleep 60", "yield_time_ms": 250})
        workspace = Path(client.post("/exec_command", json={"cmd": "pwd"}).text.split("Output:\n", 1)[1].strip())
        assert workspace.exists()
    assert not workspace.exists() and _git(repo, "worktree", "list").count("\n") == 1


def test_example_rows_are_valid_tasks() -> None:
    rows = [
        json.loads(line) for line in (Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()
    ]
    assert rows and all(row["verifier_metadata"]["check_command"] for row in rows)
    assert all(
        [tool["name"] for tool in row["responses_create_params"]["tools"]]
        == ["exec_command", "write_stdin", "apply_patch", "update_plan"]
        for row in rows
    )
