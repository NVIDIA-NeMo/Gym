# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from resources_servers.workspace_bench import app


def test_judge_uses_upstream_evaluator(monkeypatch, tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "upstream" / "evaluation"
    (evaluation_dir / "src").mkdir(parents=True)
    (evaluation_dir / "src" / "agent_as_a_judge.py").touch()
    monkeypatch.setattr(app, "ensure_upstream", lambda: tmp_path / "upstream")
    server = app.WorkspaceBenchResourcesServer.model_construct(
        config=app.WorkspaceBenchConfig.model_construct(
            judge_base_url="https://judge.example",
            judge_api_key="test",
            judge_model="judge-model",
        )
    )
    server._upstream_dir = tmp_path / "upstream"

    def run(command, **kwargs):
        assert command[1] == str(evaluation_dir / "src" / "agent_as_a_judge.py")
        assert kwargs["cwd"] == evaluation_dir
        (tmp_path / "rubrics_judge--gym-judge.json").write_text(
            json.dumps({"rubrics": [{"index": 0, "passed": True}]}), encoding="utf-8"
        )
        (tmp_path / "dependency_graph--gym-judge.json").write_text(
            json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
        )

    monkeypatch.setattr(app.subprocess, "run", run)

    rubrics, graph = server._judge(tmp_path)

    assert rubrics == [{"index": 0, "passed": True}]
    assert graph == {"nodes": [], "edges": []}
