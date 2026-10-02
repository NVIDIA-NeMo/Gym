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
import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.visual_agent.app as app_module
import resources_servers.visual_agent.build_dataset as build_dataset_module
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.visual_agent.app import (
    JudgeOutcome,
    VisualAgentResourcesServer,
    VisualAgentResourcesServerConfig,
    VisualAgentVerifyRequest,
    VisualTask,
)
from resources_servers.visual_agent.build_dataset import build_row, load_specs, validate
from resources_servers.visual_agent.figma_render import count_nodes, design_to_html
from resources_servers.visual_agent.grading import RubricItem, parse_verdict
from resources_servers.visual_agent.opencode_runner import JudgeOpenCodeConfig, opencode_install_command
from resources_servers.visual_agent.prompts import (
    GRADER_VERDICT,
    group_judge_prompt,
    judge_prompt,
    policy_prompt,
    request_text_from_input,
)


SERVER_DIR = Path(app_module.__file__).parent
RUBRIC = [{"id": "R1", "type": "instruction", "criterion": "has a hero"}]
TASK: Dict[str, Any] = {
    "task_id": "web-test",
    "category": "website",
    "mode": "open_ended",
    "artifact": {"kind": "html", "entry": "index.html"},
    "rubric": RUBRIC,
    "checks": {"mobile": True},
}
MEASUREMENTS = {
    "kind": "html",
    "artifact_found": True,
    "load_ok": True,
    "blank": False,
    "render_stats": {"desktop": {}},
    "runtime": {"page_errors": [], "console_errors": [], "failed_requests": [], "external_requests": []},
    "layout": {"horizontal_overflow_px": 0},
    "mobile_layout": {"horizontal_overflow_px": 0},
    "renders": {"desktop": "/grader/renders/desktop.png"},
}


def make_server(tmp_path: Path, **overrides: Any) -> VisualAgentResourcesServer:
    config = VisualAgentResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="visual_agent",
        sandbox_provider="sandbox",
        sandbox_config={},
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_base_url="http://judge:1/v1",
        data_dir=str(SERVER_DIR / "data"),
        grading_output_dir=str(tmp_path / "results"),
        **overrides,
    )
    return VisualAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def verify_body(task_index: int = 0, rollout_index: int = 0, **task: Any) -> VisualAgentVerifyRequest:
    return VisualAgentVerifyRequest.model_validate(
        {**TASK, **task}
        | {
            "responses_create_params": {"input": [{"role": "user", "content": "Build a landing page"}]},
            "response": {
                "id": "r",
                "created_at": 0,
                "model": "m",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            "_ng_task_index": task_index,
            "_ng_rollout_index": rollout_index,
        }
    )


def request(session: str) -> Any:
    return SimpleNamespace(session={SESSION_ID_KEY: session}, cookies={})


def judge_ok(passed: bool = True) -> JudgeOutcome:
    verdict = parse_verdict(
        json.dumps({"rubric": [{"id": "R1", "pass": passed}], "summary": "fine"}), verify_body().rubric
    )
    return JudgeOutcome(status="ok", attempts=1, verdict=verdict)


class TestSeed:
    async def test_seed_uploads_task_files_and_vendor_but_no_grader_tooling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = make_server(tmp_path)
        sandbox = MagicMock(_handle=SimpleNamespace(sandbox_id="sbx-1"))
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="", stderr=""))
        monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
        uploaded: Dict[str, Any] = {}

        async def fake_upload_tree(sb: Any, files: Dict[str, Path], remote: str) -> None:
            uploaded["remote"] = remote
            uploaded["files"] = dict(files)

        monkeypatch.setattr(app_module, "upload_tree", fake_upload_tree)
        vendor_dir = SERVER_DIR / "data" / "vendor" / "three"
        task = {**TASK, "assets": ["design.json"], "task_id": "figma-rep-01"}
        if vendor_dir.is_dir():
            task["vendor"] = ["three"]
        response = await server.seed_session(request("s1"), VisualTask.model_validate(task))

        assert response.sandbox_handle == "sbx-1"
        assert server._session_id_to_sandbox["s1"] is sandbox
        assert uploaded["remote"] == "/workspace"
        assert uploaded["files"]["task/design.json"].name == "design.json"
        # A real OpenCode setup has no grader renderer or task spec: nothing under tools/, no task.json.
        assert not any(name.startswith("tools/") or name.endswith("task.json") for name in uploaded["files"])
        if vendor_dir.is_dir():
            assert "output/vendor/three/three.module.js" in uploaded["files"]

    async def test_seed_missing_asset_stops_sandbox(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = make_server(tmp_path)
        sandbox = MagicMock(stop=AsyncMock())
        monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
        with pytest.raises(FileNotFoundError, match="missing"):
            await server.seed_session(request("s1"), VisualTask.model_validate({**TASK, "assets": ["nope.json"]}))
        sandbox.stop.assert_awaited_once()
        assert server._session_id_to_sandbox == {}


class TestVerify:
    def _patch_grading(
        self, server: VisualAgentResourcesServer, monkeypatch: pytest.MonkeyPatch, judge: JudgeOutcome
    ) -> MagicMock:
        policy = MagicMock(stop=AsyncMock())

        async def collect(sb: Any, dest: Path) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"tar")
            return dest

        async def grade(body: Any, artifact: Path, grading_dir: Path) -> Any:
            (grading_dir / "renders").mkdir(parents=True, exist_ok=True)
            (grading_dir / "renders" / "desktop.png").write_bytes(b"png")
            (grading_dir / "measurements.json").write_text(json.dumps(MEASUREMENTS))
            return dict(MEASUREMENTS), judge

        monkeypatch.setattr(server, "_collect_artifact", collect)
        monkeypatch.setattr(server, "_grade", grade)
        return policy

    async def test_verify_scores_and_writes_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = make_server(tmp_path)
        policy = self._patch_grading(server, monkeypatch, judge_ok(passed=False))
        server._session_id_to_sandbox["s1"] = policy
        response = await server.verify(request("s1"), verify_body())

        # AUTO-runtime + AUTO-layout pass, R1 fails -> 2/3.
        assert response.reward == pytest.approx(2 / 3, abs=1e-6)
        assert response.pointwise_reward == response.reward
        assert response.judge_status == "ok" and response.groupwise_status == "disabled"
        assert not response.mask_sample
        policy.stop.assert_awaited_once()
        result = json.loads((Path(response.grading_dir) / "result.json").read_text())
        assert result["reward"] == response.reward and "response" not in result

    async def test_verify_without_session_is_masked(self, tmp_path: Path) -> None:
        server = make_server(tmp_path)
        response = await server.verify(request("unknown"), verify_body())
        assert response.reward == 0.0 and response.mask_sample and response.failure_kind == "session_lost"

    async def test_judge_failure_masks_sample(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = make_server(tmp_path)
        server._session_id_to_sandbox["s1"] = self._patch_grading(
            server, monkeypatch, JudgeOutcome(status="failed", attempts=2)
        )
        response = await server.verify(request("s1"), verify_body())
        assert response.mask_sample and response.failure_kind == "judge_failed"

    async def test_groupwise_cohort(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = make_server(
            tmp_path, groupwise={"enabled": True, "group_size": 3, "collection_timeout_s": 30, "bonus": 0.2}
        )
        judges = {0: judge_ok(True), 1: judge_ok(False), 2: judge_ok(True)}
        seen: List[Dict[str, Any]] = []

        async def run_group_judge(body: Any, candidates: Dict[str, Any]) -> Optional[Dict[str, int]]:
            seen.append(candidates)
            # Candidate labels are shuffled; rank the member with the lowest pointwise score as weaker.
            worst = min(candidates, key=lambda label: candidates[label].pointwise)
            return {label: (-1 if label == worst else 0) for label in candidates}

        monkeypatch.setattr(server, "_run_group_judge", run_group_judge)

        for index in range(3):
            server._session_id_to_sandbox[f"s{index}"] = MagicMock(stop=AsyncMock())

        async def grade(body: Any, artifact: Path, grading_dir: Path) -> Any:
            (grading_dir / "renders").mkdir(parents=True, exist_ok=True)
            (grading_dir / "measurements.json").write_text("{}")
            return dict(MEASUREMENTS), judges[body.rollout_index]

        async def collect(sb: Any, dest: Path) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"tar")
            return dest

        monkeypatch.setattr(server, "_grade", grade)
        monkeypatch.setattr(server, "_collect_artifact", collect)
        results = await asyncio.gather(
            *(server.verify(request(f"s{i}"), verify_body(task_index=5, rollout_index=i)) for i in range(3))
        )
        assert len(seen) == 1 and len(seen[0]) == 3
        by_index = dict(enumerate(results))
        assert [r.groupwise_status for r in results] == ["applied"] * 3
        assert by_index[1].pointwise_reward == pytest.approx(2 / 3, abs=1e-6)
        assert by_index[1].reward == pytest.approx(2 / 3 - 0.2, abs=1e-6)
        assert by_index[1].groupwise_tier == -1
        assert by_index[0].reward == by_index[0].pointwise_reward == 1.0

    async def test_failed_member_does_not_stall_its_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = make_server(
            tmp_path, groupwise={"enabled": True, "group_size": 3, "collection_timeout_s": 30, "bonus": 0.2}
        )
        group_judge = AsyncMock(return_value={"A": 1, "B": -1})
        monkeypatch.setattr(server, "_run_group_judge", group_judge)

        async def collect(sb: Any, dest: Path) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"tar")
            return dest

        async def grade(body: Any, artifact: Path, grading_dir: Path) -> Any:
            if body.rollout_index == 2:
                raise RuntimeError("grader sandbox OOM-killed")
            (grading_dir / "renders").mkdir(parents=True, exist_ok=True)
            return dict(MEASUREMENTS), judge_ok(True)

        monkeypatch.setattr(server, "_collect_artifact", collect)
        monkeypatch.setattr(server, "_grade", grade)
        for index in range(3):
            server._session_id_to_sandbox[f"s{index}"] = MagicMock(stop=AsyncMock())
        results = await asyncio.wait_for(
            asyncio.gather(
                *(server.verify(request(f"s{i}"), verify_body(task_index=6, rollout_index=i)) for i in range(3))
            ),
            timeout=10,
        )
        assert [r.groupwise_status for r in results] == ["applied", "applied", "excluded"]
        assert results[2].mask_sample and results[2].reward == 0.0
        group_judge.assert_awaited_once()
        assert len(group_judge.await_args.args[1]) == 2  # only the two graded members are compared

    async def test_groupwise_timeout_keeps_pointwise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = make_server(
            tmp_path, groupwise={"enabled": True, "group_size": 2, "collection_timeout_s": 0.2, "bonus": 0.2}
        )
        server._session_id_to_sandbox["s0"] = self._patch_grading(server, monkeypatch, judge_ok(True))
        response = await server.verify(request("s0"), verify_body(task_index=9, rollout_index=0))
        assert response.groupwise_status == "timeout" and response.reward == response.pointwise_reward


class TestJudgeRestart:
    async def test_dead_grader_is_replaced_before_retry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = make_server(tmp_path, judge_max_attempts=2)
        dead = MagicMock(stop=AsyncMock(), exec=AsyncMock(), download=AsyncMock(side_effect=RuntimeError("410 Gone")))
        fresh = MagicMock(stop=AsyncMock(), exec=AsyncMock())

        async def download(remote: str, local: Path) -> None:
            Path(local).write_text(json.dumps({"rubric": [{"id": "R1", "pass": True}], "summary": "ok"}))

        fresh.download = download
        runs = iter(
            [
                app_module.OpenCodeRunResult(finished=False, error_type="SandboxApiException"),
                app_module.OpenCodeRunResult(finished=True, return_code=0),
            ]
        )
        monkeypatch.setattr(app_module, "run_opencode_judge", AsyncMock(side_effect=lambda *a, **k: next(runs)))
        restart = AsyncMock(return_value=fresh)
        graders = [dead]

        outcome = await server._run_judge(graders, restart, verify_body(), MEASUREMENTS, tmp_path)

        assert outcome.status == "ok" and outcome.attempts == 2
        restart.assert_awaited_once()
        dead.stop.assert_awaited_once()
        assert graders == [dead, fresh]


class TestMetrics:
    def test_compute_metrics(self, tmp_path: Path) -> None:
        server = make_server(tmp_path)
        rows = [
            {
                "category": "svg",
                "mode": "replication",
                "pointwise_reward": 0.8,
                "reward": 0.8,
                "similarity": 0.6,
                "gate_passed": True,
                "judge_status": "ok",
            },
            {
                "category": "svg",
                "mode": "open_ended",
                "pointwise_reward": 0.4,
                "reward": 0.55,
                "gate_passed": True,
                "judge_status": "ok",
                "groupwise_status": "applied",
            },
            {
                "category": "game",
                "mode": "open_ended",
                "pointwise_reward": 0.0,
                "reward": 0.0,
                "gate_passed": False,
                "judge_status": "skipped",
            },
            {
                "category": "game",
                "mode": "open_ended",
                "pointwise_reward": 0.0,
                "reward": 0.0,
                "gate_passed": True,
                "judge_status": "failed",
                "mask_sample": True,
            },
        ]
        metrics = server.compute_metrics([rows[:2], rows[2:]])
        assert metrics["pointwise_reward/category/svg"] == pytest.approx(0.6)
        assert metrics["pointwise_reward/category/game"] == 0.0  # the masked row is excluded
        assert metrics["rate/gate_failed"] == 0.25 and metrics["rate/judge_failed"] == 0.25
        assert metrics["mean_similarity/replication"] == 0.6
        assert metrics["groupwise/mean_abs_shift"] == pytest.approx(0.15)


class TestPrompts:
    def test_policy_prompt_replication(self) -> None:
        task = {
            **TASK,
            "prompt": "Rebuild the attached page.",
            "mode": "replication",
            "reference_images": ["reference.png"],
            "vendor": ["three"],
        }
        text = policy_prompt(task)
        assert text.startswith("Rebuild the attached page.")
        assert "/workspace/output/" in text and "`vendor/three/`" in text
        assert "/workspace/task/reference.png" in text and "do not embed" in text
        # Only the task and its I/O contract: no grader tooling, no hints on how to check the work.
        for hint in ("vtools", "Playwright", "preview", "pixel", "screenshot", "fc-list"):
            assert hint not in text

    def test_policy_prompt_responsive_replication_pairs_targets_with_viewports(self) -> None:
        task = {
            **TASK,
            "prompt": "Rebuild it.",
            "mode": "replication",
            "reference_images": ["desktop.png", "mobile.png"],
            "reference_viewports": [{"width": 1280, "height": 800}, {"width": 390, "height": 844}],
        }
        text = policy_prompt(task)
        assert "`desktop.png` at a 1280x800 viewport, `mobile.png` at a 390x844 viewport" in text

    def test_judge_prompt_lists_rubric_renders_and_verdict_path(self) -> None:
        body = verify_body()
        text = judge_prompt(body.model_dump(), "Build a landing page", body.rubric, MEASUREMENTS)
        assert "- R1 [instruction, weight 1]: has a hero" in text
        assert "/grader/renders/desktop.png" in text and GRADER_VERDICT in text
        assert "interact --artifact-dir /grader/artifact --entry index.html" in text
        assert "exactly as delivered" in text  # the judge must not patch the artifact to grade it

    def test_slide_decks_get_the_interaction_helper(self) -> None:
        deck = {
            **TASK,
            "category": "slides",
            "prompt": "Build a deck.",
            "artifact": {"kind": "slides", "entry": "slides.html"},
        }
        deck["interaction_hints"] = "press ArrowRight twice"
        text = judge_prompt(deck, "Build a deck", [RubricItem(**item) for item in RUBRIC], MEASUREMENTS)
        assert "interact --artifact-dir /grader/artifact --entry slides.html" in text
        assert "Hints for this task: press ArrowRight twice" in text
        assert "vtools" not in policy_prompt(deck)
        svg = {**TASK, "category": "svg", "artifact": {"kind": "svg", "entry": "icon.svg"}}
        assert "interact" not in judge_prompt(svg, "Draw", [RubricItem(**item) for item in RUBRIC], MEASUREMENTS)

    def test_group_prompt_and_request_text(self) -> None:
        text = group_judge_prompt(TASK, "req", [{"label": "A", "rubric_score": 0.5, "render_names": ["desktop.png"]}])
        assert "- A: rubric score 0.50; renders: desktop.png" in text and "/group/ranking.json" in text
        items = [{"role": "user", "content": [{"type": "input_text", "text": "hi"}, {"type": "input_image"}]}]
        assert request_text_from_input(items) == "hi\n\n[attached image]"


class TestOpenCodeInstall:
    def test_downloads_the_pinned_version_without_cached_binaries(self) -> None:
        command = opencode_install_command(JudgeOpenCodeConfig())
        assert "https://opencode.ai/install" in command and "VERSION=1.17.11" in command

    def test_uses_cached_binaries_when_configured(self) -> None:
        config = JudgeOpenCodeConfig(
            install_script_path="/opt/oc/install.sh", binary_path="/opt/oc/oc", musl_binary_path="/opt/oc/oc-musl"
        )
        assert opencode_install_command(config) == (
            "bash /opt/oc/install.sh --glibc-binary /opt/oc/oc --musl-binary /opt/oc/oc-musl"
        )
        glibc_only = config.model_copy(update={"musl_binary_path": None})
        assert opencode_install_command(glibc_only) == "bash /opt/oc/install.sh --binary /opt/oc/oc"


class TestFigmaRender:
    def test_exemplar_design_renders_all_text(self) -> None:
        design = json.loads((SERVER_DIR / "data/assets/figma-rep-01/design.json").read_text())
        html = design_to_html(design)
        for node_text in (
            "Ledgerline",
            "Nora Alvarez",
            "Export CSV",
            "Create, send and track invoices for Halden Studio.",
        ):
            assert node_text in html
        assert count_nodes(design)["TEXT"] == html.count("<span>")
        assert "border-radius:50%" in html and "box-shadow:" in html
        assert "inset 0 0 0 1px" in html and "border:" not in html  # strokes do not shift children

    def test_unsupported_node(self) -> None:
        design = {
            "document": {
                "children": [
                    {
                        "children": [
                            {
                                "type": "FRAME",
                                "absoluteBoundingBox": {"x": 0, "y": 0, "width": 10, "height": 10},
                                "children": [
                                    {
                                        "type": "VECTOR",
                                        "absoluteBoundingBox": {"x": 0, "y": 0, "width": 1, "height": 1},
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="VECTOR"):
            design_to_html(design)


class TestDataset:
    def test_task_specs_are_valid(self) -> None:
        specs = load_specs()
        assert specs, "no task specs found"
        assert validate(specs) == []

    def test_replication_row_attaches_reference_images(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Rendered references are not committed (build_dataset.py references); supply one.
        png = b"\x89PNG\r\n\x1a\nreference"
        (tmp_path / "svg-rep-01").mkdir()
        (tmp_path / "svg-rep-01" / "reference.png").write_bytes(png)
        monkeypatch.setattr(build_dataset_module, "ASSETS_DIR", tmp_path)
        spec = next(s for s in load_specs() if s["task_id"] == "svg-rep-01")
        row = build_row(spec)
        content = row["responses_create_params"]["input"][0]["content"]
        assert content[0]["type"] == "input_text" and "Recreate the attached national park badge" in content[0]["text"]
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == "data:image/png;base64," + base64.b64encode(png).decode()
        assert "golden" not in row and "_source" not in row
        VisualAgentVerifyRequest.model_validate(row | {"response": verify_body().response.model_dump()})
