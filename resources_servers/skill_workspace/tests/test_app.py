# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.server_utils import ServerClient
from resources_servers.skill_workspace.app import SkillWorkspaceResourcesServer
from resources_servers.skill_workspace.schemas import (
    CloseRequest,
    ReadFileRequest,
    RunBashRequest,
    SkillWorkspaceResourcesServerConfig,
    SkillWorkspaceSeedSessionRequest,
)


def _make_skill(root: Path, name: str = "demo-skill") -> Path:
    skill = root / name
    skill.mkdir()
    (skill / "SKILL.md").write_text("# SKILL\nbody\n")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "helper.py").write_text("print('hi')\n")
    (skill / "references").mkdir()
    (skill / "references" / "note.md").write_text("ref\n")
    fixtures = skill / "evals" / "files"
    fixtures.mkdir(parents=True)
    (fixtures / "input.txt").write_text("fixture content\n")
    (fixtures / "secret.txt").write_text("should not copy\n")
    return skill


def _make_server(tmp_path: Path, **overrides) -> SkillWorkspaceResourcesServer:
    config = SkillWorkspaceResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="skill_workspace",
        workspace_root=str(tmp_path / "workspaces"),
        **overrides,
    )
    (tmp_path / "workspaces").mkdir()
    return SkillWorkspaceResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestApp:
    def test_sanity(self, tmp_path: Path) -> None:
        _make_server(tmp_path)

    def test_setup_webserver_registers_routes(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        app = server.setup_webserver()
        paths = {r.path for r in app.routes}
        assert {"/seed_session", "/run_bash", "/read_file", "/close", "/verify"} <= paths


class TestSeedSession:
    @pytest.mark.asyncio
    async def test_copies_skill_structure_and_listed_fixtures_only(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)

        resp = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(
                skill_path=str(skill),
                scenario_id=1,
                files=["evals/files/input.txt"],
            )
        )

        ws = server.env_id_to_workspace[resp.env_id]
        # SKILL.md is intentionally NOT copied into the workspace — see the
        # sibling test and the comment in seed_session() for the rationale.
        assert not (ws / "SKILL.md").exists()
        assert (ws / "scripts" / "helper.py").is_file()
        assert (ws / "references" / "note.md").is_file()
        assert (ws / "evals" / "files" / "input.txt").read_text() == "fixture content\n"
        assert not (ws / "evals" / "files" / "secret.txt").exists()

    @pytest.mark.asyncio
    async def test_skill_md_is_not_copied_into_workspace(self, tmp_path: Path) -> None:
        """Prevents the without_skill arm from `cat SKILL.md`'ing the skill body
        off disk. The skill specification belongs in the system prompt, not on
        the filesystem the rollout can read."""
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        resp = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )
        ws = server.env_id_to_workspace[resp.env_id]
        assert not (ws / "SKILL.md").exists()
        # But supplementary artifacts the scenario may reference still ride along.
        assert (ws / "scripts" / "helper.py").is_file()
        assert (ws / "references" / "note.md").is_file()

    @pytest.mark.asyncio
    async def test_with_references_false_omits_references_dir(self, tmp_path: Path) -> None:
        """The 'no-docs' arm of the 2×2 must not leak skill references onto disk.
        We observed 100% of without_skill rollouts reading references/*.md when
        it was unconditionally seeded; this flag is how the control arm stays
        clean."""
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        resp = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[], with_references=False)
        )
        ws = server.env_id_to_workspace[resp.env_id]
        assert not (ws / "references").exists()
        # scripts still seeded by default, independent flag
        assert (ws / "scripts").is_dir()

    @pytest.mark.asyncio
    async def test_with_scripts_false_omits_scripts_dir(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        resp = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[], with_scripts=False)
        )
        ws = server.env_id_to_workspace[resp.env_id]
        assert not (ws / "scripts").exists()
        assert (ws / "references").is_dir()

    @pytest.mark.asyncio
    async def test_blind_arm_has_only_fixtures_and_sandbox_bin(self, tmp_path: Path) -> None:
        """The 'blind' cell of the 2×2 (no skill, no docs) should contain only
        scenario fixtures and the workspace-local python symlink — nothing the
        skill author authored that could hint at the answer."""
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        resp = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(
                skill_path=str(skill),
                scenario_id=1,
                files=["evals/files/input.txt"],
                with_references=False,
                with_scripts=False,
            )
        )
        ws = server.env_id_to_workspace[resp.env_id]
        assert not (ws / "SKILL.md").exists()
        assert not (ws / "references").exists()
        assert not (ws / "scripts").exists()
        assert (ws / "evals" / "files" / "input.txt").is_file()
        assert (ws / ".sandbox_bin" / "python").exists()

    @pytest.mark.asyncio
    async def test_parallel_sessions_are_isolated(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)

        req = SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        a, b = await asyncio.gather(server.seed_session(req), server.seed_session(req))
        ws_a = server.env_id_to_workspace[a.env_id]
        ws_b = server.env_id_to_workspace[b.env_id]

        assert ws_a != ws_b
        (ws_a / "output.txt").write_text("from A")
        (ws_b / "output.txt").write_text("from B")
        assert (ws_a / "output.txt").read_text() == "from A"
        assert (ws_b / "output.txt").read_text() == "from B"

    @pytest.mark.asyncio
    async def test_rejects_missing_skill_path(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await server.seed_session(
                SkillWorkspaceSeedSessionRequest(skill_path=str(tmp_path / "nope"), scenario_id=1, files=[])
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_escaping_fixture_path(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await server.seed_session(
                SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=["../../etc/passwd"])
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_absolute_fixture_path(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await server.seed_session(
                SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=["/etc/passwd"])
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_missing_fixture(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await server.seed_session(
                SkillWorkspaceSeedSessionRequest(
                    skill_path=str(skill), scenario_id=1, files=["evals/files/missing.txt"]
                )
            )
        assert exc.value.status_code == 400


class TestRunBash:
    @pytest.mark.asyncio
    async def test_executes_in_workspace_cwd(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )

        result = await server.run_bash(RunBashRequest(env_id=seed.env_id, cmd="ls"))
        assert result.exit_code == 0
        assert "scripts" in result.stdout
        assert "references" in result.stdout
        assert "SKILL.md" not in result.stdout  # see test_skill_md_is_not_copied_into_workspace

    @pytest.mark.asyncio
    async def test_truncates_output_above_cap(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, output_cap_bytes=1024)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )

        result = await server.run_bash(RunBashRequest(env_id=seed.env_id, cmd="yes x | head -c 5000"))
        assert result.truncated is True
        assert len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8")) <= 1024

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, bash_timeout_hard_cap_seconds=2)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )

        result = await server.run_bash(RunBashRequest(env_id=seed.env_id, cmd="sleep 10", timeout_seconds=1))
        assert result.timed_out is True
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_unknown_env_id_returns_404(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await server.run_bash(RunBashRequest(env_id="does-not-exist", cmd="ls"))
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_sandbox_env_strips_host_venv_from_path(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )

        # PATH should only contain system dirs — no project .venv, no conda, no pyenv.
        result = await server.run_bash(RunBashRequest(env_id=seed.env_id, cmd="echo $PATH"))
        assert result.exit_code == 0
        path_value = result.stdout.strip()
        assert ".venv" not in path_value
        assert "conda" not in path_value
        assert "pyenv" not in path_value

        # `ng_status` and friends must not resolve — any `ng_*` command in the host
        # venv should be invisible to the rollout.
        result = await server.run_bash(RunBashRequest(env_id=seed.env_id, cmd="command -v ng_status"))
        assert result.exit_code != 0, "ng_status leaked into sandbox PATH"

    @pytest.mark.asyncio
    async def test_sandbox_provides_python_alias_for_python3(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )

        result = await server.run_bash(RunBashRequest(env_id=seed.env_id, cmd="python --version"))
        assert result.exit_code == 0, f"`python` should resolve inside sandbox: {result.stderr}"
        assert "Python 3" in (result.stdout + result.stderr)


class TestReadFile:
    @pytest.mark.asyncio
    async def test_reads_relative_path(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=["evals/files/input.txt"])
        )

        result = await server.read_file(ReadFileRequest(env_id=seed.env_id, path="evals/files/input.txt"))
        assert result.content == "fixture content\n"
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_rejects_absolute_path(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )
        with pytest.raises(HTTPException) as exc:
            await server.read_file(ReadFileRequest(env_id=seed.env_id, path="/etc/passwd"))
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_parent_escape(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )
        with pytest.raises(HTTPException) as exc:
            await server.read_file(ReadFileRequest(env_id=seed.env_id, path="../../../etc/passwd"))
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_file_returns_404(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )
        with pytest.raises(HTTPException) as exc:
            await server.read_file(ReadFileRequest(env_id=seed.env_id, path="missing.txt"))
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_env_id_returns_404(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        with pytest.raises(HTTPException) as exc:
            await server.read_file(ReadFileRequest(env_id="nope", path="x"))
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_truncates_large_file(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, output_cap_bytes=32)
        skill = _make_skill(tmp_path)
        big = skill / "big.txt"
        big.write_text("x" * 200)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=["big.txt"])
        )
        result = await server.read_file(ReadFileRequest(env_id=seed.env_id, path="big.txt"))
        assert result.truncated is True
        assert len(result.content.encode("utf-8")) == 32


class TestClose:
    @pytest.mark.asyncio
    async def test_reclaims_tmpdir(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )
        ws = server.env_id_to_workspace[seed.env_id]
        assert ws.exists()

        resp = await server.close(CloseRequest(env_id=seed.env_id))
        assert resp.success is True
        assert not ws.exists()
        assert seed.env_id not in server.env_id_to_workspace

    @pytest.mark.asyncio
    async def test_unknown_env_id_returns_success_false(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        resp = await server.close(CloseRequest(env_id="unknown"))
        assert resp.success is False

    @pytest.mark.asyncio
    async def test_rmtree_failure_returns_success_false(self, tmp_path: Path, monkeypatch) -> None:
        import shutil as _shutil

        server = _make_server(tmp_path)
        skill = _make_skill(tmp_path)
        seed = await server.seed_session(
            SkillWorkspaceSeedSessionRequest(skill_path=str(skill), scenario_id=1, files=[])
        )

        def boom(*args, **kwargs):
            raise OSError("disk on fire")

        monkeypatch.setattr(_shutil, "rmtree", boom)
        resp = await server.close(CloseRequest(env_id=seed.env_id))
        assert resp.success is False
        assert "disk on fire" in resp.message


class TestVerify:
    @pytest.mark.asyncio
    async def test_returns_zero_reward(self, tmp_path: Path) -> None:
        from nemo_gym.base_resources_server import BaseVerifyRequest
        from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

        server = _make_server(tmp_path)
        body = BaseVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
        )
        resp = await server.verify(body)
        assert resp.reward == 0.0
