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
import json
import sys
from pathlib import Path

import pytest

from resources_servers.vibench import prepare


def make_checkout(tmp_path: Path, app: str = "notes", with_assets: bool = False) -> Path:
    """A minimal ViBench-shaped checkout."""
    app_dir = tmp_path / "prds" / app
    (app_dir / "prd").mkdir(parents=True)
    (app_dir / "prd" / "mvp.txt").write_text("build a notes app")
    (app_dir / "prd" / "feature1.txt").write_text("add sharing")
    (app_dir / "tests" / "mvp").mkdir(parents=True)
    (app_dir / "tests" / "mvp" / "test1.txt").write_text("<step>a<skippable>n</skippable></step>")
    (app_dir / "tests" / "mvp" / "test2.txt").write_text("<step>b</step>")
    (app_dir / "tests" / "feature1").mkdir(parents=True)
    (app_dir / "tests" / "feature1" / "test1.txt").write_text("<step>c</step>")
    if with_assets:
        (app_dir / "assets").mkdir()
        (app_dir / "assets" / "data.csv").write_text("a,b")
        (app_dir / "test_assets").mkdir()
        (app_dir / "test_assets" / "fixture.png").write_bytes(b"png")
    return tmp_path


class TestArtifactResolution:
    def test_feature_on_mvp_reuses_the_base_feature_test_dir(self, tmp_path):
        """featureN-on_mvp has no test folder of its own; plans live under featureN."""
        root = make_checkout(tmp_path)
        app_dir = root / "prds" / "notes"

        assert prepare.artifact_test_dir(app_dir, "feature1-on_mvp").name == "feature1"
        assert prepare.artifact_test_dir(app_dir, "feature1").name == "feature1"
        assert prepare.artifact_test_dir(app_dir, "mvp").name == "mvp"

    def test_feature_prd_chain_prepends_the_mvp(self, tmp_path):
        """A feature is built on top of the MVP, so the agent needs both briefs in order."""
        root = make_checkout(tmp_path)
        app_dir = root / "prds" / "notes"

        assert [p.name for p in prepare.prd_chain(app_dir, "mvp")] == ["mvp.txt"]
        assert [p.name for p in prepare.prd_chain(app_dir, "feature1")] == ["mvp.txt", "feature1.txt"]
        assert [p.name for p in prepare.prd_chain(app_dir, "feature1-on_mvp")] == ["mvp.txt", "feature1.txt"]

    def test_discover_artifacts_lists_mvp_first(self, tmp_path):
        root = make_checkout(tmp_path)

        assert prepare.discover_artifacts(root / "prds" / "notes")[0] == "mvp"

    def test_discover_artifacts_on_missing_app(self, tmp_path):
        assert prepare.discover_artifacts(tmp_path / "nope") == []


class TestRowConstruction:
    def test_row_carries_paths_relative_to_the_checkout(self, tmp_path, monkeypatch):
        """Absolute paths would tie the dataset to one machine."""
        root = make_checkout(tmp_path, with_assets=True)
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "RENDERED BRIEF")

        row = prepare.build_row(root, "notes", "mvp", None, 300)

        assert row["prd_files"] == ["prds/notes/prd/mvp.txt"]
        assert row["test_plans"] == ["prds/notes/tests/mvp/test1.txt", "prds/notes/tests/mvp/test2.txt"]
        assert not any(Path(p).is_absolute() for p in row["prd_files"] + row["test_plans"])

    def test_builder_assets_and_grader_assets_are_kept_apart(self, tmp_path, monkeypatch):
        """test_assets/ belongs to the evaluation agent and must never reach the builder."""
        root = make_checkout(tmp_path, with_assets=True)
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")

        row = prepare.build_row(root, "notes", "mvp", None, 300)

        assert row["asset_dirs"] == ["prds/notes/assets"]
        assert row["test_assets_dir"] == "prds/notes/test_assets"
        assert "test_assets" not in row["asset_dirs"][0]

    def test_absent_asset_dirs_are_omitted(self, tmp_path, monkeypatch):
        root = make_checkout(tmp_path, with_assets=False)
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")

        row = prepare.build_row(root, "notes", "mvp", None, 300)

        assert row["asset_dirs"] == []
        assert row["test_assets_dir"] is None

    def test_system_prompt_is_prepended_when_given(self, tmp_path, monkeypatch):
        root = make_checkout(tmp_path)
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")

        row = prepare.build_row(root, "notes", "mvp", "SYS", 300)

        assert [m["role"] for m in row["responses_create_params"]["input"]] == ["system", "user"]

    def test_row_is_dropped_when_the_prompt_cannot_render(self, tmp_path, monkeypatch):
        """A row without ViBench's brief would be graded against a contract it never saw."""
        root = make_checkout(tmp_path)
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: None)

        assert prepare.build_row(root, "notes", "mvp", None, 300) is None

    def test_row_is_dropped_when_the_artifact_has_no_test_plans(self, tmp_path, monkeypatch):
        root = make_checkout(tmp_path)
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")

        assert prepare.build_row(root, "notes", "feature2", None, 300) is None

    def test_row_is_dropped_when_a_prd_is_missing(self, tmp_path, monkeypatch):
        root = make_checkout(tmp_path)
        (root / "prds" / "notes" / "prd" / "mvp.txt").unlink()
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")

        assert prepare.build_row(root, "notes", "mvp", None, 300) is None


class TestPromptRendering:
    def test_returns_none_when_the_template_is_absent(self, tmp_path):
        assert prepare.render_task_prompt(tmp_path, "prd text", 300, "mvp") is None

    def test_prefers_vibench_own_interpreter(self, tmp_path):
        """ViBench's venv has jinja2; the caller's may not."""
        venv_python = tmp_path / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("")

        assert prepare.vibench_python(tmp_path) == str(venv_python)

    def test_falls_back_to_the_current_interpreter(self, tmp_path):
        assert prepare.vibench_python(tmp_path) == sys.executable

    def test_renders_the_real_template(self, tmp_path):
        """The brief is a contract; a paraphrase would change what is measured."""
        pytest.importorskip("jinja2")
        prompts = tmp_path / "_harness" / "runner" / "agent" / "prompts"
        prompts.mkdir(parents=True)
        (prompts / "coding_prompt.j2").write_text("goal={{ goal }} iters={{ max_iterations }}\n{{ prd }}")

        out = prepare.render_task_prompt(tmp_path, "MY PRD", 42, "mvp")

        assert "goal=zero-to-one" in out
        assert "iters=42" in out
        assert "MY PRD" in out

    def test_feature_artifacts_get_the_feature_goal(self, tmp_path):
        """zero-to-one tells the model to build from scratch; a feature extends a codebase."""
        pytest.importorskip("jinja2")
        prompts = tmp_path / "_harness" / "runner" / "agent" / "prompts"
        prompts.mkdir(parents=True)
        (prompts / "coding_prompt.j2").write_text("goal={{ goal }}")

        assert "goal=zero-to-one" in prepare.render_task_prompt(tmp_path, "PRD", 1, "mvp")
        assert "goal=feature-building" in prepare.render_task_prompt(tmp_path, "PRD", 1, "feature1")
        assert "goal=feature-building" in prepare.render_task_prompt(tmp_path, "PRD", 1, "feature1-on_mvp")

    def test_render_failure_is_reported_not_raised(self, tmp_path):
        pytest.importorskip("jinja2")
        prompts = tmp_path / "_harness" / "runner" / "agent" / "prompts"
        prompts.mkdir(parents=True)
        (prompts / "coding_prompt.j2").write_text("{{ undefined_variable }}")

        assert prepare.render_task_prompt(tmp_path, "PRD", 1, "mvp") is None


class TestMain:
    def test_writes_one_row_per_app(self, tmp_path, monkeypatch, capsys):
        root = make_checkout(tmp_path)
        make_checkout(tmp_path, app="quiz")
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")
        out = tmp_path / "out.jsonl"
        monkeypatch.setattr(sys, "argv", ["prepare.py", "--vibench-root", str(root), "--output", str(out)])

        prepare.main()

        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert sorted(r["app"] for r in rows) == ["notes", "quiz"]

    def test_limit_truncates(self, tmp_path, monkeypatch):
        root = make_checkout(tmp_path)
        make_checkout(tmp_path, app="quiz")
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")
        out = tmp_path / "out.jsonl"
        monkeypatch.setattr(
            sys, "argv", ["prepare.py", "--vibench-root", str(root), "--output", str(out), "--limit", "1"]
        )

        prepare.main()

        assert len(out.read_text().splitlines()) == 1

    def test_unknown_artifact_is_skipped(self, tmp_path, monkeypatch):
        root = make_checkout(tmp_path)
        monkeypatch.setattr(prepare, "render_task_prompt", lambda *a, **k: "BRIEF")
        out = tmp_path / "out.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            ["prepare.py", "--vibench-root", str(root), "--output", str(out), "--artifacts", "nonexistent"],
        )

        prepare.main()

        assert out.read_text() == ""

    def test_missing_prds_dir_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["prepare.py", "--vibench-root", str(tmp_path), "--output", str(tmp_path / "o.jsonl")]
        )

        with pytest.raises(SystemExit, match="No prds/"):
            prepare.main()
