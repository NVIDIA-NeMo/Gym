# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import threading
import time
from asyncio import Semaphore
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from resources_servers.legal_agent_bench.harbor_bridge import REPO_ROOT, LegalAgentBenchHarborBridge
from resources_servers.legal_agent_bench.legal_harbor_agent import (
    LegalAgentBenchHarborAgent,
    OpenAICompatibleAdapter,
    _chat_with_timeout,
)
from resources_servers.legal_agent_bench.prepare import (
    EXPECTED_TASK_COUNT,
    REQUIRED_SKILLS,
    SMOKE_TASK_IDS,
    _marker,
    flatten_task_id,
)
from resources_servers.legal_agent_bench.vendor.harvey_labs.lab_harbor import scoring as lab_scoring
from resources_servers.legal_agent_bench.vendor.harvey_labs.lab_harbor.judge import (
    _extract_judge_message_text,
)
from resources_servers.legal_agent_bench.verifier import (
    _build_reward,
    _validate_reward_mode,
    _write_report,
    score_rubric,
)
from responses_api_agents.harbor_agent.app import HarborAgentConfig


BENCH_DIR = Path(__file__).resolve().parents[1]


def _write_skills(skills_dir: Path) -> None:
    for name in REQUIRED_SKILLS:
        (skills_dir / name).mkdir(parents=True)
        (skills_dir / name / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    marker = _marker("skills")
    (skills_dir / ".nemo_gym_asset.json").write_text(json.dumps(marker), encoding="utf-8")


def _bridge() -> LegalAgentBenchHarborBridge:
    config = HarborAgentConfig(
        name="legal_agent_bench_harbor_agent",
        host="127.0.0.1",
        port=0,
        entrypoint="../../resources_servers/legal_agent_bench/harbor_bridge.py",
        concurrency=1,
        model_server={"type": "responses_api_models", "name": "policy_model"},
        harbor_datasets={"legal_agent_bench": {"local_dataset_path": "/tmp/tasks"}},
        harbor_environment_type="docker",
        harbor_jobs_dir="results/legal_agent_bench/harbor_jobs",
    )
    return LegalAgentBenchHarborBridge.model_construct(
        config=config,
        server_client=MagicMock(),
        sem=Semaphore(1),
    )


def test_folder_config_is_public_docker_only() -> None:
    config = OmegaConf.to_container(OmegaConf.load(BENCH_DIR / "configs" / "legal_agent_bench.yaml"), resolve=False)
    resource = config["legal_agent_bench"]["resources_servers"]["legal_agent_bench"]
    agent = config["legal_agent_bench_harbor_agent"]["responses_api_agents"]["harbor_agent"]

    assert resource["reward_mode"] == "full_task"
    assert resource["auto_prepare_assets"] is True
    assert agent["harbor_environment_type"] == "docker"
    assert agent["harbor_environment_import_path"] is None
    assert "docker_image" not in json.dumps(agent)


def test_pinned_snapshot_has_1749_tasks_and_five_committed_examples() -> None:
    examples = [json.loads(line) for line in (BENCH_DIR / "data" / "example.jsonl").read_text().splitlines()]
    expected_ids = {f"legal_agent_bench::{flatten_task_id(source_id)}" for source_id in SMOKE_TASK_IDS}

    assert EXPECTED_TASK_COUNT == 1749
    assert len(examples) == 5
    assert {row["instance_id"] for row in examples} == expected_ids


def test_committed_example_rollouts_are_complete_and_match_examples() -> None:
    examples = [json.loads(line) for line in (BENCH_DIR / "data" / "example.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (BENCH_DIR / "data" / "example_rollouts.jsonl").read_text().splitlines()]

    assert len(rollouts) == 5
    assert {row["instance_id"] for row in rollouts} == {row["instance_id"] for row in examples}
    for row in rollouts:
        assert row["response"]["status"] == "completed"
        assert row["response"]["error"] is None
        assert row["response"]["output"]
        assert row["context_length_exceeded_error"] == 0
        assert row["memory_limit_exceeded_error"] == 0
        assert row["agent_timeout_error"] == 0
        assert set(row["metadata"]) == {"verifier_result"}
        rewards = row["metadata"]["verifier_result"]["rewards"]
        assert rewards["judge_error_count"] == 0
        assert 0.0 < rewards["criteria_pass_rate"] <= 1.0


def test_lab_bridge_restores_standard_routes_and_absolute_paths(monkeypatch, tmp_path) -> None:
    bridge = _bridge()
    routes = {route.path for route in bridge.setup_webserver().routes}
    timestamp = datetime(2026, 7, 8, tzinfo=timezone.utc)
    jobs_before = bridge._get_jobs_output_dir("org/model", "legal_agent_bench", timestamp)
    monkeypatch.chdir(tmp_path)
    jobs_after = bridge._get_jobs_output_dir("org/model", "legal_agent_bench", timestamp)

    assert {"/v1/responses", "/run", "/aggregate_metrics"} <= routes
    assert jobs_before == jobs_after
    assert jobs_before.is_absolute()
    assert jobs_before.is_relative_to(REPO_ROOT)


def test_agent_discovers_exactly_three_skills_lazily(tmp_path) -> None:
    skills = tmp_path / "skills"
    agent = LegalAgentBenchHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="policy-model",
        api_base="http://policy/v1",
        skills_dir=skills,
    )
    with pytest.raises(FileNotFoundError, match="skills are incomplete"):
        agent._skill_names()

    _write_skills(skills)
    assert agent._skill_names() == list(REQUIRED_SKILLS)


def test_agent_rejects_unknown_requested_skill(tmp_path) -> None:
    skills = tmp_path / "skills"
    _write_skills(skills)
    agent = LegalAgentBenchHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="policy-model",
        api_base="http://policy/v1",
        skills_dir=skills,
        skills=["docx", "pdf"],
    )

    with pytest.raises(FileNotFoundError, match="pdf"):
        agent._skill_names()


@pytest.mark.asyncio
async def test_container_hydration_uploads_configured_skills_and_documents(tmp_path) -> None:
    skills = tmp_path / "skills"
    documents = tmp_path / "documents"
    documents.mkdir()
    _write_skills(skills)
    agent = LegalAgentBenchHarborAgent(
        logs_dir=tmp_path / "logs",
        model_name="policy-model",
        api_base="http://policy/v1",
        skills_dir=skills,
    )

    class Environment:
        def __init__(self):
            self.uploads = []

        async def exec(self, *_args, **_kwargs):
            return MagicMock(stdout="", stderr="", return_code=0)

        async def upload_dir(self, source, destination):
            self.uploads.append((Path(source), destination))

    environment = Environment()
    await agent._hydrate_environment(environment, documents)

    assert environment.uploads == [(documents, "/workspace/vdr"), (skills, "/workspace/skills")]


@pytest.mark.asyncio
async def test_policy_adapter_uses_gym_model_server(monkeypatch) -> None:
    observed = {}

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "done",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": '{"path":"input.docx"}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }

    class Session:
        def __init__(self, *, timeout, headers):
            observed["timeout"] = timeout.total
            observed["headers"] = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, *, json):
            observed["url"] = url
            observed["payload"] = json
            return Response()

    monkeypatch.setattr("harness.adapters.openai_compatible.ClientSession", Session)
    adapter = OpenAICompatibleAdapter(
        model="policy-model",
        base_url="http://policy/v1",
        reasoning_effort="high",
        timeout_seconds=30,
    )

    response = await adapter.chat([{"role": "user", "content": "work"}], [])

    assert observed["url"] == "http://policy/v1/chat/completions"
    assert observed["timeout"] == 30
    assert observed["headers"] == {"Authorization": "Bearer EMPTY"}
    assert observed["payload"]["reasoning_effort"] == "high"
    assert response.text == "done"
    assert response.input_tokens == 10
    assert response.output_tokens == 2
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "read"
    assert response.tool_calls[0].arguments == '{"path":"input.docx"}'


@pytest.mark.asyncio
async def test_policy_adapter_timeout(monkeypatch) -> None:
    adapter = OpenAICompatibleAdapter(
        model="policy-model",
        base_url="http://policy/v1",
        timeout_seconds=0.001,
    )

    async def slow_chat(_messages, _tools):
        await asyncio.sleep(1)

    monkeypatch.setattr(adapter, "chat", slow_chat)

    with pytest.raises(TimeoutError, match="agent model request exceeded timeout"):
        await _chat_with_timeout(adapter, [], [])


@pytest.mark.parametrize(
    "message, expected",
    [
        ({"content": '{"verdict":"pass"}'}, ('{"verdict":"pass"}', "content")),
        ({"content": "", "reasoning_content": '{"verdict":"pass"}'}, ('{"verdict":"pass"}', "reasoning_content")),
        ({"content": [{"text": "structured"}]}, ("structured", "content")),
    ],
)
def test_judge_response_text_sources(message, expected) -> None:
    assert _extract_judge_message_text(message) == expected


def test_empty_judge_message_fails_clearly() -> None:
    with pytest.raises(ValueError, match="contained no text"):
        _extract_judge_message_text({"content": "", "reasoning_content": None})


def test_reward_mode_validation() -> None:
    assert _validate_reward_mode("full_task") == "full_task"
    assert _validate_reward_mode("criteria_pass_rate") == "criteria_pass_rate"
    with pytest.raises(ValueError, match="Invalid LEGAL_AGENT_BENCH_REWARD_MODE"):
        _validate_reward_mode("partial")


def test_harbor_reward_payload_is_numeric_only() -> None:
    scores = {
        "score": 0.0,
        "n_passed": 22,
        "n_criteria": 23,
        "judge_error_count": 0,
        "all_pass": False,
    }

    full_task = _build_reward(scores, "full_task")
    diagnostic = _build_reward(scores, "criteria_pass_rate")

    assert full_task["reward"] == 0.0
    assert diagnostic["reward"] == pytest.approx(22 / 23)
    assert all(isinstance(value, (int, float)) for value in full_task.values())
    assert "reported_reward_mode" not in full_task


def test_scoring_records_judge_errors_separately(tmp_path) -> None:
    class FailingJudge:
        def evaluate_prompt(self, *_args, **_kwargs):
            raise RuntimeError("judge unavailable")

    scores = score_rubric(
        criteria=[{"id": "C-1", "title": "Criterion", "match_criteria": "Pass."}],
        run_dir=tmp_path / "run",
        judge=FailingJudge(),
        task_desc="Task",
    )

    assert scores["score"] == 0.0
    assert scores["judge_error_count"] == 1
    assert scores["criteria_results"][0]["judge_error"] is True
    assert "judge unavailable" in scores["criteria_results"][0]["reasoning"]


def test_scoring_includes_docx_redlines_when_requested(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "run" / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "response.docx").write_bytes(b"placeholder")
    observed = []

    def read_file(_path, *, track_changes=lab_scoring.DocxTrackChanges.ACCEPT):
        observed.append(track_changes)
        return "document content"

    class Judge:
        last_raw_response = json.dumps({"verdict": "pass"})
        last_structured = True

        def evaluate_prompt(self, *_args, **_kwargs):
            return {"verdict": "pass", "reasoning": "redlines reviewed"}

    monkeypatch.setattr(lab_scoring, "_read_file_as_text", read_file)
    scores = score_rubric(
        criteria=[
            {
                "id": "C-1",
                "title": "Review redline",
                "match_criteria": "Pass.",
                "deliverables": ["response.docx"],
                "evaluation_options": {"include_docx_redlines": True},
            }
        ],
        run_dir=tmp_path / "run",
        judge=Judge(),
        task_desc="Task",
    )

    assert observed == [lab_scoring.DocxTrackChanges.ALL]
    assert scores["score"] == 1.0


def test_docx_conversion_replaces_invalid_utf8(monkeypatch, tmp_path) -> None:
    path = tmp_path / "response.docx"
    path.write_bytes(b"placeholder")

    def run(*_args, **kwargs):
        output = b"valid text\xff".decode(kwargs["encoding"], errors=kwargs["errors"])
        return lab_scoring.subprocess.CompletedProcess([], 0, stdout=output, stderr="")

    monkeypatch.setattr(lab_scoring.subprocess, "run", run)

    assert lab_scoring._read_file_as_text(path) == "valid text\ufffd"


def test_deliverable_matching_ignores_thread_export() -> None:
    assert lab_scoring._match_deliverables(
        {"contract": "revised-contract.docx"},
        ["output.docx", "final-contract.docx"],
    ) == {"contract": "final-contract.docx"}


def test_parallel_judging_uses_isolated_judges_and_preserves_order(tmp_path) -> None:
    output_dir = tmp_path / "run" / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "response.md").write_text("response", encoding="utf-8")
    criteria = [{"id": f"C-{index}", "title": f"Criterion {index}", "match_criteria": "Pass."} for index in range(3)]
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    created = []

    class Judge:
        last_raw_response = '{"verdict":"pass"}'
        last_structured = True

        def __init__(self, identifier):
            self.identifier = identifier
            self.context = {}

        def set_trace_context(self, context):
            self.context = context

        def evaluate_prompt(self, *_args, **_kwargs):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.02)
            with lock:
                state["active"] -= 1
            return {"verdict": "pass", "reasoning": self.context["criterion_id"]}

    def judge_factory():
        with lock:
            judge = Judge(len(created))
            created.append(judge)
        return judge

    transcript_path = tmp_path / "transcript.jsonl"
    scores = score_rubric(
        criteria=criteria,
        run_dir=tmp_path / "run",
        judge=Judge("unused"),
        judge_factory=judge_factory,
        parallelism=3,
        task_desc="Task",
        transcript_path=transcript_path,
    )

    assert len({judge.identifier for judge in created}) == 3
    assert state["max_active"] > 1
    assert [result["id"] for result in scores["criteria_results"]] == ["C-0", "C-1", "C-2"]
    assert scores["score"] == 1.0
    assert all(json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines())


def test_html_report_escapes_task_and_judge_content(tmp_path) -> None:
    scores = {
        "run_id": "run<script>",
        "summary": "<script>alert(1)</script>",
        "score": 0.0,
        "judge_model": "judge&model",
        "criteria_results": [
            {
                "id": "C<1>",
                "verdict": "fail",
                "title": "Title <unsafe>",
                "reasoning": '<img src="x">',
            }
        ],
    }

    _write_report(scores, tmp_path)

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<title>LAB Score - run&lt;script&gt;</title>" in report
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "Title &lt;unsafe&gt;" in report
    assert "&lt;img src=&quot;x&quot;&gt;" in report
