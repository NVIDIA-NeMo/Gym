import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_agents.claude_code_agent.app import ClaudeCodeAgent
from responses_api_agents.nemorl_env import author_worker


@pytest.fixture
def source_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_task_environment) -> Path:
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    revisions = {}
    for name, package, files in (
        ("gym", "nemo_gym", {"reward.py": "def score():\n    return 1\n"}),
        (
            "nrl",
            "nemo_rl",
            {
                "core.py": "def loss(values):\n    return sum(abs(value) for value in values)\n",
                "obsolete.py": "old = True\n",
            },
        ),
    ):
        repo = tmp_path / name
        (repo / package).mkdir(parents=True)
        (repo / package / "__init__.py").write_text("")
        for filename, content in files.items():
            (repo / package / filename).write_text(content)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        if name == "nrl":
            subprocess.run(
                ["git", "submodule", "add", "-q", str(tmp_path / "gym"), "3rdparty/Gym-workspace/Gym"],
                cwd=repo,
                check=True,
            )
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
        revisions[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    monkeypatch.setattr(author_worker, "NEMORL_REVISION", revisions["nrl"])
    monkeypatch.setattr(author_worker, "GYM_REVISION", revisions["gym"])
    (tmp_path / "nrl" / "nemo_rl" / "core.py").write_text("raise RuntimeError('dirty source must not be copied')\n")
    (tmp_path / "gym" / "untracked_answers.json").write_text('{"answer": 42}\n')
    return tmp_path / "nrl"


@pytest.mark.parametrize("research_seconds", [120, 3600])
@pytest.mark.parametrize("model", ["nvidia/qwen/qwen3.8-27b", "Qwen/Qwen3.8-27B"])
async def test_source_patch_replays_new_loss_and_gym_edits(
    tmp_path: Path, source_repository: Path, research_seconds: int, model: str
) -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    job = {
        "model": {"model": model},
        "source_repository": str(source_repository),
        "train_seconds": 3600,
        "research_seconds": research_seconds,
        "max_turns": 200,
        "body": {"input": [], "metadata": {"problem_statement": "Implement a new loss"}},
    }
    clean = tmp_path / "clean_verifier"
    expected = object()

    async def create_response(author: ClaudeCodeAgent, body, *, rollout_id):
        assert author.config.model == model
        command = author._build_command(author.config.model, "test")
        assert command[command.index("--model") + 1] == model
        assert author.config.max_turns == 200 and author.config.timeout == research_seconds
        assert author.config.token_id_capture is False and rollout_id is None
        assert "60-minute execution budget" in author.config.system_prompt
        assert f"You have {research_seconds // 60} minutes and 200 turns" in author.config.system_prompt
        work = Path(author.config.cwd)
        source = work / "NeMo-RL"
        gym = source / "3rdparty/Gym-workspace/Gym"
        assert not (source / ".git").exists()
        assert not (gym / ".git").exists()
        assert not (gym / "untracked_answers.json").exists()
        assert "dirty source" not in (source / "nemo_rl/core.py").read_text()
        assert len((work / "train_math.jsonl").read_text().splitlines()) == 480
        assert not (work / "run.sh").exists()
        assert not (work / "launch_inner.py").exists()
        shutil.copytree(work, clean, ignore=shutil.ignore_patterns(".git"))

        (source / "nemo_rl/new_loss.py").write_text(
            "def new_loss(values):\n    return sum(value * value for value in values)\n"
        )
        (source / "nemo_rl/core.py").write_text("from .new_loss import new_loss\n\nloss = new_loss\n")
        (source / "nemo_rl/obsolete.py").unlink()
        (source / "nemo_rl/fixture.bin").write_bytes(b"\x00new\xffloss\x00")
        (gym / "nemo_gym/reward.py").write_text("def score():\n    return 2\n")
        for filename in ("run.sh", "launch_inner.py", "evaluation_data.json", "train_math.jsonl"):
            (work / filename).write_text("outside the accepted patch scope\n")
        return expected

    with (
        patch("responses_api_agents.claude_code_agent.app.ensure_claude_code"),
        patch("responses_api_agents.claude_code_agent.app.subprocess.run"),
        patch.object(ClaudeCodeAgent, "_create_response", create_response),
    ):
        response, diff = await author_worker.author(job, client)

    assert response is expected
    assert "new_loss.py" in diff
    assert "GIT binary patch" in diff
    assert "outside the accepted patch scope" not in diff

    shutil.rmtree(clean / "NeMo-RL")
    subprocess.run(
        ["git", "clone", "-q", "--recurse-submodules", str(source_repository), str(clean / "NeMo-RL")],
        check=True,
    )
    subprocess.run(["git", "apply", "--check"], input=diff, text=True, cwd=clean, check=True)
    subprocess.run(["git", "apply"], input=diff, text=True, cwd=clean, check=True)
    source = clean / "NeMo-RL"
    assert not (source / "nemo_rl/obsolete.py").exists()
    assert (source / "nemo_rl/fixture.bin").read_bytes() == b"\x00new\xffloss\x00"
    assert len((clean / "train_math.jsonl").read_text().splitlines()) == 480
    for filename in ("run.sh", "launch_inner.py", "evaluation_data.json"):
        assert not (clean / filename).exists()
    for repo, expression in (
        (source, "from nemo_rl.core import loss; assert loss([3, -4]) == 25"),
        (source / "3rdparty/Gym-workspace/Gym", "from nemo_gym.reward import score; assert score() == 2"),
    ):
        subprocess.run([sys.executable, "-c", expression], cwd=repo, check=True)
