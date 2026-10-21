# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytest.importorskip("harbor")

from fastapi.testclient import TestClient
from harbor.agents.factory import AgentFactory
from harbor.models.task.id import LocalTaskId
from harbor.models.trajectories import Trajectory
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from omegaconf import OmegaConf
from pydantic import ConfigDict, ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.judge import JudgeError
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.server_utils import (
    IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME,
    IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME,
    BaseServerConfig,
    ServerClient,
    SimpleServer,
)
from resources_servers.partner_if.evidence import MAX_EVIDENCE_CHARS
from resources_servers.partner_if.grading_contract import GradingContract
from resources_servers.partner_if.package import TaskBundle
from resources_servers.partner_if.tests.test_app import response, server
from responses_api_agents.harbor_agent_general import app as harbor
from responses_api_agents.harbor_agent_general.app import HarborRunRequest
from responses_api_agents.partner_if_harbor_agent.app import (
    MAX_RECEIPT_BYTES,
    PartnerIFHarborAgent,
    PartnerIFHarborConfig,
    artifact_receipt,
    validate_openhands_completion,
)


MODEL_SERVER_URL = "http://192.0.2.10:9000"
INSTRUCTION = "Write a greeting to /root/output/answer.md."
RECEIPT = {"exists": True, "text": "HELLO"}
FINISHED = {"agent_state": "finished"}
ITERATION_CAP = {
    "agent_state": "error",
    "reason": "RuntimeError: Agent reached maximum iteration. Current iteration: 30, max iteration: 30",
}
JUDGE = "resources_servers.partner_if.grading_service.call_judge"


@pytest.fixture(autouse=True)
def model_server(monkeypatch):
    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", lambda name: MODEL_SERVER_URL)
    monkeypatch.setattr(harbor, "get_global_config_dict", lambda: {})


def harbor_config(tmp_path, **overrides):
    return PartnerIFHarborConfig.model_validate(
        {
            "name": "partner_if_markdown",
            "host": "127.0.0.1",
            "port": 8000,
            "entrypoint": "app.py",
            "harbor_jobs_dir": str(tmp_path / "jobs"),
            "harbor_resolve_paths_from_gym_root": True,
            "harbor_reward_key": "artifact_capture",
            "task_bundles_dir": str(tmp_path),
            "harbor_dataset": {"path": str(tmp_path / "tasks")},
            "harbor_agent": {
                "name": "openhands",
                "model_name": "openai/policy",
                "env": {"LLM_API_KEY": "gym-proxy-no-provider-secret"},
                "kwargs": {"version": "0.62.0", "max_iterations": 30},
            },
            "model_server": {"type": "responses_api_models", "name": "partner_if_harbor_model"},
            "resources_server": {"type": "resources_servers", "name": "partner_if"},
            **overrides,
        }
    )


class _Reply:
    def __init__(self, reply):
        self.ok, self.status, self.content = reply.is_success, reply.status_code, reply.content

    async def read(self):
        return self.content


class _ResourcesClient(ServerClient):
    """Sends the agent's calls to a real partner_if resources server."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resources: TestClient
    calls: list[tuple[str, str, dict]]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Reply:
        payload = kwargs["json"]
        payload = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        self.calls.append((server_name, url_path, payload))
        return _Reply(self.resources.post(url_path, json=payload))


def make_agent(tmp_path, *, observability=False, **overrides):
    client = _ResourcesClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({"observability_enabled": observability}),
        resources=TestClient(server().setup_webserver()),
        calls=[],
    )
    return PartnerIFHarborAgent(config=harbor_config(tmp_path, **overrides), server_client=client), client


def live_contract(*rubrics):
    return GradingContract(
        task_id="native-1",
        origin="synthetic",
        source="authored",
        rubric_version="native-v1",
        audit_signoff=True,
        data_kind="synthetic_smoke",
        evidence_mode="live_agent",
        rubrics=rubrics
        or [
            {
                "id": "said",
                "constraint": "The answer says HELLO.",
                "verifier": {"id": "text_contains_v1", "params": {"source": "answer", "literal": "HELLO"}},
            },
            {"id": "greeting", "constraint": "The file is a greeting.", "verifier": {"id": "llm_judge_v1"}},
        ],
    )


def run_body(tmp_path, contract):
    (tmp_path / "native.json").write_text(
        TaskBundle(contract=contract, documents={"skill": "Greet briefly."}).model_dump_json()
    )
    return HarborRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": INSTRUCTION}]},
            "verifier_metadata": {"partner_if": contract.model_dump()},
            "task_name": "native",
            "_ng_task_index": 3,
            "_ng_rollout_index": 1,
        }
    )


def write_trial(tmp_path, *, state=FINISHED, receipt=RECEIPT) -> Path:
    """A finished Harbor trial as the job worker leaves it: result, ATIF trajectory, OpenHands events and receipt."""
    task_dir = tmp_path / "tasks/native"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(INSTRUCTION)
    trial_dir = tmp_path / "jobs/t3-r1/native__trial"
    events = trial_dir / "agent/sessions/session/events"
    events.mkdir(parents=True)
    for index, extras in enumerate([{"agent_state": "running"}, state], 1):
        (events / f"{index}.json").write_text(json.dumps({"observation": "agent_state_changed", "extras": extras}))
    trajectory = {
        "agent": {"name": "openhands", "version": "0.62.0"},
        "steps": [{"step_id": 1, "source": "agent", "message": "HELLO"}],
    }
    (trial_dir / "agent/trajectory.json").write_text(Trajectory.model_validate(trajectory).model_dump_json())
    if receipt is not None:
        (trial_dir / "verifier").mkdir()
        (trial_dir / "verifier/artifact.json").write_text(json.dumps(receipt))
    result = TrialResult(
        task_name="native",
        trial_name=trial_dir.name,
        trial_uri=trial_dir.as_uri(),
        task_id=LocalTaskId(path=task_dir),
        task_checksum="checksum",
        config=TrialConfig(task=TaskConfig(path=task_dir)),
        agent_info=AgentInfo(name="openhands", version="0.62.0"),
        # Harbor's own verifier only reports whether the artifact was captured; partner_if supplies the reward.
        verifier_result=VerifierResult(rewards={"artifact_capture": 0.0}),
    )
    (trial_dir / "result.json").write_text(result.model_dump_json())
    return trial_dir


class FakeJobWorker:
    """Stands in for the Ray task: records each job and returns the trial it left behind."""

    def __init__(self, trial_dir):
        self.trial_dir, self.jobs = trial_dir, []

    def options(self, **_):
        return self

    def remote(self, job_config, task_name, worker_env):
        self.jobs.append(job_config)

        async def trial():
            return str(self.trial_dir)

        return trial()


GREETING = {
    "id": "greeting",
    "passed": True,
    "reason": "A greeting.",
    "evidence": "HELLO",
    "references": [{"source": "artifact:submission", "start_line": 1, "end_line": 1}],
}


def judge_verdict(*items):
    return response(json.dumps({"items": list(items)}))


@pytest.mark.parametrize("state", [FINISHED, ITERATION_CAP])
async def test_finished_or_iteration_capped_trial_is_graded_by_the_resources_server(tmp_path, monkeypatch, state):
    contract = live_contract()
    body = run_body(tmp_path, contract)
    worker = FakeJobWorker(write_trial(tmp_path, state=state))
    monkeypatch.setattr(harbor, "harbor_job_worker", worker)
    agent, client = make_agent(tmp_path, observability=True)
    with patch(JUDGE, AsyncMock(return_value=judge_verdict(GREETING))):
        row = (await agent.run(body)).model_dump()
    [(name, path, sent)] = client.calls
    assert (name, path) == ("partner_if", "/verify")
    assert sent["verification_input"] == {
        "input": [{"role": "user", "content": INSTRUCTION, "type": "message"}],
        "documents": {"skill": "Greet briefly."},
        "artifacts": {"submission": RECEIPT},
    }
    # The resources server's scoring fields replace Harbor's capture reward.
    assert (row["reward"], row["mask_sample"], row["failure_kind"]) == (1.0, False, None)
    assert row["if_result"]["status"] == "scored" and row["if_result"]["passed"] == 2
    assert NG_FAILURE_CLASS_KEY not in row and "verification_input" not in row
    assert row["partner_if_artifact"] == RECEIPT and row["atif_conversion"]["lossless"]
    # OpenHands calls the model server under this run's rollout prefix.
    [openhands] = worker.jobs[0]["agents"]
    assert (
        openhands["kwargs"]["api_base"] == openhands["env"]["LLM_BASE_URL"] == f"{MODEL_SERVER_URL}/ng-rollout/3-1/v1"
    )


@pytest.mark.parametrize(
    "flags",
    [{"observability_enabled": True}, {"checkpoint": {"enabled": True, "control_auth_token": "token"}}],
    ids=["observability", "checkpoint"],
)
async def test_verify_is_posted_under_the_runs_rollout_prefix(tmp_path, monkeypatch, flags):
    # A real ServerClient takes the prefix from the posted body; checkpointing refuses a /verify without it.
    body = run_body(tmp_path, live_contract())
    monkeypatch.setattr(harbor, "harbor_job_worker", FakeJobWorker(write_trial(tmp_path)))
    resources, urls = TestClient(server().setup_webserver()), []

    async def request(*, method, url, json, **_):
        urls.append(url)
        return _Reply(resources.post(url.removeprefix("http://192.0.2.20:9001"), json=json))

    monkeypatch.setattr("nemo_gym.server_utils.request", request)
    config = {"partner_if": {"resources_servers": {"partner_if": {"host": "192.0.2.20", "port": 9001}}}} | flags
    client = ServerClient(
        head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=OmegaConf.create(config)
    )
    agent = PartnerIFHarborAgent(config=harbor_config(tmp_path), server_client=client)
    with patch(JUDGE, AsyncMock(return_value=judge_verdict(GREETING))):
        row = (await agent.run(body)).model_dump()
    assert urls == ["http://192.0.2.20:9001/ng-rollout/3-1/verify"]
    assert (row["reward"], row["mask_sample"]) == (1.0, False)


@pytest.mark.parametrize("dimension,masked", [("completion", False), ("if", True)])
async def test_missing_receipt_leaves_only_the_items_that_need_the_artifact_unscorable(
    tmp_path, monkeypatch, dimension, masked
):
    contract = live_contract(
        {
            "id": "said",
            "constraint": "The answer says HELLO.",
            "verifier": {"id": "text_contains_v1", "params": {"source": "answer", "literal": "HELLO"}},
        },
        {
            "id": "file",
            "constraint": "The file says HELLO.",
            "dimension": dimension,
            "verifier": {"id": "text_contains_v1", "params": {"source": "artifact:submission", "literal": "HELLO"}},
        },
    )
    body = run_body(tmp_path, contract)
    monkeypatch.setattr(harbor, "harbor_job_worker", FakeJobWorker(write_trial(tmp_path, receipt=None)))
    agent, client = make_agent(tmp_path)
    row = (await agent.run(body)).model_dump()
    assert client.calls[0][2]["verification_input"]["artifacts"]["submission"]["error"].startswith(
        "Cannot read the artifact capture receipt"
    )
    items = {item["id"]: item for item in row["grading_result"]["items"]}
    assert items["said"]["passed"] is True and items["file"]["passed"] is None
    # An unscorable row is masked in the main output, not sent to the failures file.
    assert NG_FAILURE_CLASS_KEY not in row and row["mask_sample"] is masked
    assert (row["reward"], row["failure_kind"]) == ((0.0, "partner_if:unscorable") if masked else (1.0, None))


async def test_failed_judge_call_returns_a_judge_failed_row_through_verify(tmp_path, monkeypatch):
    body = run_body(tmp_path, live_contract())
    monkeypatch.setattr(harbor, "harbor_job_worker", FakeJobWorker(write_trial(tmp_path)))
    agent, _ = make_agent(tmp_path)
    with patch(JUDGE, AsyncMock(side_effect=JudgeError("judge unavailable"))):
        row = (await agent.run(body)).model_dump()
    assert row[NG_FAILURE_CLASS_KEY] == row["failure_kind"] == "judge_failed"
    assert row["mask_sample"] is True and row["reward"] == 0.0
    assert row["atif_conversion"]["lossless"]


async def test_controller_error_is_a_harbor_failure_and_is_never_graded(tmp_path, monkeypatch):
    body = run_body(tmp_path, live_contract())
    state = {"agent_state": "error", "reason": "RuntimeError: LLM provider rejected the request"}
    monkeypatch.setattr(harbor, "harbor_job_worker", FakeJobWorker(write_trial(tmp_path, state=state)))
    agent, client = make_agent(tmp_path)
    row = (await agent.run(body)).model_dump()
    assert client.calls == []
    assert (row[NG_FAILURE_CLASS_KEY], row["failure_kind"], row["mask_sample"]) == (
        "harbor_failed",
        "agent_run_error",
        True,
    )
    assert "LLM provider rejected the request" in row["failure_reason"]


def test_openhands_zero_exit_is_not_enough_without_finished_event(tmp_path):
    events = tmp_path / "agent/sessions/run/events"
    events.mkdir(parents=True)
    (events / "1.json").write_text(
        json.dumps({"observation": "agent_state_changed", "extras": {"agent_state": "running"}})
    )
    with pytest.raises(RuntimeError, match="completion state"):
        validate_openhands_completion(tmp_path)
    (events / "2.json").write_text(json.dumps({"observation": "agent_state_changed", "extras": ITERATION_CAP}))
    validate_openhands_completion(tmp_path)
    budget = {"agent_state": "error", "reason": "RuntimeError: Agent reached maximum budget for conversation."}
    (events / "3.json").write_text(json.dumps({"observation": "agent_state_changed", "extras": budget}))
    with pytest.raises(RuntimeError, match="maximum budget"):
        validate_openhands_completion(tmp_path)


@pytest.mark.parametrize(
    "content,error",
    [
        (None, "Cannot read"),
        ("{not json", "Cannot read"),
        ("[]", "not a JSON object"),
        (json.dumps({"exists": True, "text": "x" * MAX_RECEIPT_BYTES}), "exceeds size limit"),
    ],
    ids=["missing", "invalid", "not_an_object", "oversized"],
)
def test_unreadable_receipt_is_a_capture_error(tmp_path, content, error):
    if content is not None:
        (tmp_path / "verifier").mkdir()
        (tmp_path / "verifier/artifact.json").write_text(content)
    assert error in artifact_receipt(tmp_path)["error"]


async def test_artifact_text_over_the_evidence_limit_is_too_large_however_it_is_escaped(tmp_path, monkeypatch):
    # json.dumps writes each "é" as 6 bytes, so the receipt is over 6 MB; its character count decides.
    receipt = {"exists": True, "text": "é" * (MAX_EVIDENCE_CHARS + 1)}
    body = run_body(tmp_path, live_contract())
    monkeypatch.setattr(harbor, "harbor_job_worker", FakeJobWorker(write_trial(tmp_path, receipt=receipt)))
    agent, _ = make_agent(tmp_path)
    with patch(JUDGE, AsyncMock()) as judge:
        row = (await agent.run(body)).model_dump()
    judge.assert_not_called()
    assert (row["mask_sample"], row["failure_kind"]) == (True, "partner_if:evidence_too_large")


async def test_aggregate_metrics_are_partner_if_metrics(tmp_path):
    agent, client = make_agent(tmp_path)
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "if_result": {"passed": 2, "total": 2}}]
    metrics = await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert [(name, path) for name, path, _ in client.calls] == [("partner_if", "/aggregate_metrics")]
    assert "accuracy" not in json.dumps(metrics.model_dump())


def test_llm_base_url_points_openhands_at_the_gym_model_server(tmp_path, monkeypatch):
    agent, _ = make_agent(tmp_path)
    body = run_body(tmp_path, live_contract())
    job = agent.build_job_config(body)
    [openhands] = job.agents
    assert openhands.kwargs == {"version": "0.62.0", "max_iterations": 30, "api_base": f"{MODEL_SERVER_URL}/v1"}
    assert openhands.env == {"LLM_API_KEY": "gym-proxy-no-provider-secret", "LLM_BASE_URL": f"{MODEL_SERVER_URL}/v1"}
    # The inherited job keeps its task, and the configured agent is not mutated.
    assert job.job_name == "t3-r1" and job.datasets[0].task_names == ["native"]
    assert agent.config.harbor_agent.env == {"LLM_API_KEY": "gym-proxy-no-provider-secret"}
    # Harbor's OpenHands prefers LLM_BASE_URL over api_base, the agent's env over the worker's.
    monkeypatch.setenv("LLM_BASE_URL", "http://other-provider.example/v1")
    created = AgentFactory.create_agent_from_config(AgentConfig.model_validate(openhands), logs_dir=tmp_path)
    assert created.model_connection.base_url == f"{MODEL_SERVER_URL}/v1"


def test_a_configured_llm_base_url_is_rejected(tmp_path):
    agent, _ = make_agent(tmp_path)
    agent.config.harbor_agent = agent.config.harbor_agent.model_copy(
        update={"env": {"LLM_BASE_URL": "http://other-provider.example/v1"}}
    )
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        agent.build_job_config(run_body(tmp_path, live_contract()))


@pytest.mark.parametrize("url", ["http://127.0.0.1:9000", "http://localhost:9000", "http://0.0.0.0:9000"])
def test_startup_rejects_a_model_server_the_sandbox_cannot_reach(tmp_path, monkeypatch, url):
    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", lambda name: url)
    with pytest.raises(ValueError, match="sandbox-reachable"):
        make_agent(tmp_path)


def test_startup_requires_the_opensandbox_key_for_the_artifact_environment(tmp_path, monkeypatch):
    environment = {"import_path": "responses_api_agents.partner_if_harbor_agent.environment:ArtifactEnvironment"}
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENSANDBOX_API_KEY"):
        make_agent(tmp_path, harbor_environment=environment)
    make_agent(tmp_path, harbor_environment=environment, harbor_worker_env={"OPENSANDBOX_API_KEY": "key"})
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "key")
    make_agent(tmp_path, harbor_environment=environment)


def test_model_server_is_required_and_relative_package_paths_resolve_under_the_gym_roots(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="model_server"):
        harbor_config(tmp_path, model_server=None)
    monkeypatch.chdir(tmp_path)
    relative = {"task_bundles_dir": "resources_servers/partner_if/data", "harbor_dataset": {"path": "benchmarks"}}
    built_in = harbor_config(tmp_path, **relative)
    assert built_in.task_bundles_dir == PARENT_DIR / "resources_servers/partner_if/data"
    assert built_in.harbor_dataset.path == (PARENT_DIR / "benchmarks").resolve()
    # A benchmark plugin root (--search-dir) is searched first.
    plugin = tmp_path / "plugin"
    (plugin / "benchmarks").mkdir(parents=True)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin))
    assert harbor_config(tmp_path, **relative).harbor_dataset.path == (plugin / "benchmarks").resolve()
    assert harbor_config(tmp_path).harbor_dataset.path == (tmp_path / "tasks").resolve()


def test_example_config_validates_as_the_agent_starts(monkeypatch):
    from nemo_gym.global_config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
        get_first_server_config_dict,
    )

    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {
                    "config_paths": [
                        "responses_api_agents/partner_if_harbor_agent/configs/partner_if_harbor_agent.yaml"
                    ]
                },
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    block = OmegaConf.to_container(get_first_server_config_dict(config, "partner_if_harbor_agent"), resolve=True)
    agent = PartnerIFHarborConfig.model_validate(block | {"name": "partner_if_harbor_agent"})
    assert (agent.model_server.name, agent.resources_server.name) == ("partner_if_harbor_model", "partner_if")
    assert config.partner_if_harbor_model.responses_api_models.vllm_model.is_responses_native is False
    assert agent.harbor_dataset.path == PARENT_DIR / "data/partner_if_harbor/tasks"
    assert agent.harbor_jobs_dir == PARENT_DIR / "results/partner_if_harbor_agent/harbor_jobs"
    assert agent.harbor_environment.kwargs["sandbox_provider"]["opensandbox"]["connection"]["api_key"] == (
        "supplied-by-worker-environment"
    )
    assert agent.harbor_agent.env == {"LLM_API_KEY": "gym-proxy-no-provider-secret"}


def test_uvicorn_workers_import_a_module_level_app(monkeypatch):
    # With num_workers > 1, each uvicorn worker imports `responses_api_agents.partner_if_harbor_agent.app:app`.
    path, name = str(Path(__file__).parents[1] / "app.py"), "responses_api_agents.partner_if_harbor_agent.app"
    started = MagicMock(return_value=object())
    monkeypatch.setattr(SimpleServer, "run_webserver", started)
    for key in (IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME):
        monkeypatch.delenv(key, raising=False)
    assert "app" not in runpy.run_path(path, run_name=name)
    started.assert_not_called()
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME, "responses_api_agents/partner_if_harbor_agent/app.py")
    assert runpy.run_path(path, run_name=name)["app"] is started.return_value
    started.assert_called_once_with()
