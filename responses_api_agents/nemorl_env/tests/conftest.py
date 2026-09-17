import json
import shutil
from pathlib import Path

import pytest

from responses_api_agents.nemorl_env import app, author_worker


@pytest.fixture
def prepared_task_environment(tmp_path, monkeypatch):
    source = Path(app.__file__).parent
    root = tmp_path / "bundle/Gym"
    task = root / "responses_api_agents/nemorl_env"
    task.mkdir(parents=True)
    for name in ("pyproject.toml", "README.md", "LICENSE", "nemo_gym", "responses_api_models"):
        (root / name).symlink_to(app.PARENT_DIR / name)
    (task.parent / "claude_code_agent").symlink_to(source.parent / "claude_code_agent")
    for path in source.glob("*.py"):
        shutil.copy2(path, task / path.name)
    shutil.copytree(
        source / "task_environment", task / "task_environment", ignore=shutil.ignore_patterns("*.jsonl", "__pycache__")
    )
    (task / "data").mkdir()
    (task / "task_environment/train_math.jsonl").write_text(
        "".join(json.dumps({"input": f"Compute {i} + 1", "output": str(i + 1)}) + "\n" for i in range(512))
    )
    monkeypatch.setattr(app, "PARENT_DIR", root)
    monkeypatch.setattr(app, "__file__", str(task / "app.py"))
    monkeypatch.setattr(author_worker, "__file__", str(task / "author_worker.py"))
