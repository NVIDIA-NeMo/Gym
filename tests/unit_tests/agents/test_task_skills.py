# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A task's own skills, copied inside the sandbox to where a harness's CLI discovers them.

The helper runs its commands in a real shell on a local sandbox, so the tests observe the folders it leaves.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.agents.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.agents.utils import task_skills
from nemo_gym.agents.utils.task_skills import stage_task_skills
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.providers.local import LocalProvider


@pytest.fixture
async def local_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local sandbox whose ``task.json`` is under ``tmp_path``, as the Environment Server leaves it in a task."""
    context = tmp_path / "task.json"
    monkeypatch.setattr(task_skills, "TASK_CONTEXT_FILE", str(context))
    sandbox = AsyncSandbox(LocalProvider())
    await sandbox.start(SandboxSpec(workdir=str(tmp_path)))
    yield SimpleNamespace(sandbox=sandbox, context=context, root=tmp_path)
    await sandbox.stop()


def _skills_folder(root: Path, name: str = "skills") -> Path:
    folder = root / name
    (folder / "demo").mkdir(parents=True)
    (folder / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill.\n---\nThe magic word is pineapple.\n"
    )
    (folder / "demo" / "scripts").mkdir()
    (folder / "demo" / "scripts" / "helper.sh").write_text("echo helper\n")
    (folder / "demo" / ".hidden").write_text("dotfile\n")
    return folder


async def test_the_declared_folder_is_copied_whole_to_the_destination(local_sandbox) -> None:
    source = _skills_folder(local_sandbox.root)
    local_sandbox.context.write_text(json.dumps({"mcp_servers": [], "skills_dir": str(source)}))
    destination = local_sandbox.root / "config" / "skills"

    staged = await stage_task_skills(local_sandbox.sandbox, str(destination))

    assert staged == str(source)
    assert (destination / "demo" / "SKILL.md").read_text() == (source / "demo" / "SKILL.md").read_text()
    # Supporting files and dotfiles come along, which a `cp dir/*` would miss.
    assert (destination / "demo" / "scripts" / "helper.sh").read_text() == "echo helper\n"
    assert (destination / "demo" / ".hidden").read_text() == "dotfile\n"
    # A copy: the task's own folder stays as it was.
    assert (source / "demo" / "SKILL.md").is_file()


async def test_symlinks_to_files_outside_the_folder_are_staged_as_their_readable_contents(local_sandbox) -> None:
    """An image can ship a skill whose files link outside ``skills_dir``. The copy lives elsewhere, so a relative link
    copied as a link would no longer resolve: the staged files hold the targets' contents and keep their modes."""
    source = local_sandbox.root / "app" / "skills"
    (source / "demo" / "scripts").mkdir(parents=True)
    shared = local_sandbox.root / "app" / "shared.md"
    shared.write_text("---\nname: demo\ndescription: Demo skill.\n---\nThe magic word is pineapple.\n")
    helper = local_sandbox.root / "app" / "helper.sh"
    helper.write_text("echo helper\n")
    helper.chmod(0o755)
    (source / "demo" / "SKILL.md").symlink_to("../../shared.md")
    (source / "demo" / "scripts" / "helper.sh").symlink_to("../../../helper.sh")
    local_sandbox.context.write_text(json.dumps({"skills_dir": str(source)}))
    destination = local_sandbox.root / "config" / "skills"

    await stage_task_skills(local_sandbox.sandbox, str(destination))

    staged_skill = destination / "demo" / "SKILL.md"
    staged_helper = destination / "demo" / "scripts" / "helper.sh"
    assert not staged_skill.is_symlink() and staged_skill.read_text() == shared.read_text()
    assert not staged_helper.is_symlink() and staged_helper.read_text() == "echo helper\n"
    assert staged_helper.stat().st_mode & 0o111


async def test_a_symlink_whose_target_is_missing_is_refused_instead_of_staged_broken(local_sandbox) -> None:
    source = _skills_folder(local_sandbox.root)
    (source / "demo" / "reference.md").symlink_to("../../missing.md")
    local_sandbox.context.write_text(json.dumps({"skills_dir": str(source)}))

    with pytest.raises(AgentSeedRejectedError, match="Could not stage the task's skills_dir"):
        await stage_task_skills(local_sandbox.sandbox, str(local_sandbox.root / "config" / "skills"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files whatever their mode")
async def test_a_symlink_whose_target_cannot_be_read_is_refused(local_sandbox) -> None:
    source = _skills_folder(local_sandbox.root)
    secret = local_sandbox.root / "secret.md"
    secret.write_text("unreadable\n")
    secret.chmod(0)
    (source / "demo" / "reference.md").symlink_to(secret)
    local_sandbox.context.write_text(json.dumps({"skills_dir": str(source)}))

    try:
        with pytest.raises(AgentSeedRejectedError, match="Could not stage the task's skills_dir"):
            await stage_task_skills(local_sandbox.sandbox, str(local_sandbox.root / "config" / "skills"))
    finally:
        secret.chmod(0o644)


async def test_paths_with_spaces_and_shell_characters_are_copied_as_written(local_sandbox) -> None:
    source = _skills_folder(local_sandbox.root, "my skills; $(touch pwned)")
    local_sandbox.context.write_text(json.dumps({"skills_dir": str(source)}))
    destination = local_sandbox.root / "config dir" / "skills"

    await stage_task_skills(local_sandbox.sandbox, str(destination))

    assert (destination / "demo" / "SKILL.md").is_file()
    assert not (local_sandbox.root / "pwned").exists()


async def test_a_declared_folder_that_holds_no_skills_is_copied_all_the_same(local_sandbox) -> None:
    """The baseline arm of a skills comparison declares ``skills_dir`` over a folder with no skills in it: the task
    still counts as one that declares skills, so the harness sets up the same discovery folder for both arms."""
    source = local_sandbox.root / "skills"
    source.mkdir()
    (source / "README.md").write_text("Reference notes, not a skill.\n")
    local_sandbox.context.write_text(json.dumps({"mcp_servers": [], "skills_dir": str(source)}))
    destination = local_sandbox.root / "config" / "skills"

    assert await stage_task_skills(local_sandbox.sandbox, str(destination)) == str(source)
    assert [entry.name for entry in destination.iterdir()] == ["README.md"]

    (source / "README.md").unlink()
    emptied = local_sandbox.root / "config-2" / "skills"

    assert await stage_task_skills(local_sandbox.sandbox, str(emptied)) == str(source)
    assert emptied.is_dir() and list(emptied.iterdir()) == []


@pytest.mark.parametrize("context", [None, json.dumps({"mcp_servers": []})], ids=["task_json_lost", "no_skills_dir"])
async def test_a_folder_the_caller_names_is_copied_without_reading_task_json(local_sandbox, context) -> None:
    """A session re-bound after a checkpoint restore names the folder its seed staged: the fork may have lost
    ``task.json`` with the rest of ``/tmp``."""
    source = _skills_folder(local_sandbox.root)
    if context is not None:
        local_sandbox.context.write_text(context)
    destination = local_sandbox.root / "config" / "skills"

    assert await stage_task_skills(local_sandbox.sandbox, str(destination), skills_dir=str(source)) == str(source)
    assert (destination / "demo" / "SKILL.md").read_text() == (source / "demo" / "SKILL.md").read_text()


async def test_a_folder_the_caller_names_must_be_an_absolute_path(local_sandbox) -> None:
    destination = local_sandbox.root / "config" / "skills"

    with pytest.raises(AgentSeedRejectedError, match="is not an absolute path"):
        await stage_task_skills(local_sandbox.sandbox, str(destination), skills_dir="skills")

    assert not destination.exists()


@pytest.mark.parametrize(
    "context",
    [
        None,
        "",
        json.dumps({"mcp_servers": [{"name": "search"}]}),
        json.dumps({"skills_dir": None}),
        json.dumps({"skills_dir": ""}),
    ],
)
async def test_a_task_that_declares_no_skills_leaves_the_destination_alone(local_sandbox, context) -> None:
    if context is not None:
        local_sandbox.context.write_text(context)
    destination = local_sandbox.root / "config" / "skills"

    assert await stage_task_skills(local_sandbox.sandbox, str(destination)) is None
    assert not destination.exists()


@pytest.mark.parametrize(
    ("context", "message"),
    [
        ("not json", "is not JSON"),
        ("[1, 2]", "is not a JSON object"),
        ('{"skills_dir": 7}', "is not an absolute path"),
        # A path only a shell resolves is passed through as written, and cannot be copied from safely.
        ('{"skills_dir": "~/skills"}', "is not an absolute path"),
        ('{"skills_dir": "$HOME/skills"}', "is not an absolute path"),
        ('{"skills_dir": "skills"}', "is not an absolute path"),
    ],
)
async def test_an_unusable_task_context_is_refused_at_seed(local_sandbox, context, message) -> None:
    local_sandbox.context.write_text(context)
    destination = local_sandbox.root / "config" / "skills"

    with pytest.raises(AgentSeedRejectedError, match=message):
        await stage_task_skills(local_sandbox.sandbox, str(destination))

    assert not destination.exists()


async def test_a_skills_dir_that_is_not_a_folder_in_the_sandbox_is_refused_by_name(local_sandbox) -> None:
    missing = local_sandbox.root / "never-uploaded"
    local_sandbox.context.write_text(json.dumps({"skills_dir": str(missing)}))
    destination = local_sandbox.root / "config" / "skills"

    with pytest.raises(AgentSeedRejectedError) as refusal:
        await stage_task_skills(local_sandbox.sandbox, str(destination))

    assert str(missing) in refusal.value.detail
    assert "not a folder in the sandbox" in refusal.value.detail
    assert not destination.exists()


class _ScriptedSandbox:
    """Answers each exec in turn, for results a local shell does not produce."""

    def __init__(self, *results: SandboxExecResult) -> None:
        self.results = list(results)

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float | None = None):
        return self.results.pop(0)


_DECLARES_SKILLS = SandboxExecResult(stdout='{"skills_dir": "/workspace/skills"}', stderr="", return_code=0)


@pytest.mark.parametrize(
    "read",
    [
        SandboxExecResult(stdout="", stderr="", return_code=None, error_type="timeout"),
        SandboxExecResult(stdout="", stderr="permission denied", return_code=1),
    ],
)
async def test_a_read_that_reported_no_clean_exit_fails_the_seed_instead_of_passing_for_no_skills(read) -> None:
    sandbox = _ScriptedSandbox(read)

    with pytest.raises(RuntimeError, match="could not read"):
        await stage_task_skills(sandbox, "/session/config/skills")


async def test_a_copy_that_reported_no_exit_code_is_an_infrastructure_failure_not_a_refusal() -> None:
    sandbox = _ScriptedSandbox(
        _DECLARES_SKILLS, SandboxExecResult(stdout="", stderr="", return_code=None, error_type="timeout")
    )

    with pytest.raises(RuntimeError, match="timeout") as failure:
        await stage_task_skills(sandbox, "/session/config/skills")

    assert not isinstance(failure.value, AgentSeedRejectedError)


async def test_a_failed_copy_is_refused_with_its_exit_code_and_message() -> None:
    sandbox = _ScriptedSandbox(
        _DECLARES_SKILLS, SandboxExecResult(stdout="", stderr="cp: cannot create directory", return_code=1)
    )

    with pytest.raises(AgentSeedRejectedError) as refusal:
        await stage_task_skills(sandbox, "/session/config/skills")

    assert "/workspace/skills" in refusal.value.detail
    assert "exit 1" in refusal.value.detail and "cannot create directory" in refusal.value.detail
