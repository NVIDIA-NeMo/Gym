# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import tarfile
from pathlib import Path
from time import monotonic
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nemorl_env.app import NeMoRLEnvAgent, NeMoRLEnvConfig, NeMoRLEnvRunRequest


@pytest.fixture
def agent(prepared_task_environment):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {
            "token_id_capture": {"enabled": True, "all_agents": False},
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "base_url": "https://inference-api.nvidia.com/v1",
                        "api_key": "inference-test-only",
                        "model": "nvidia/qwen/qwen3.8-27b",
                        "uses_reasoning_parser": True,
                    }
                }
            },
        }
    )
    return NeMoRLEnvAgent(
        config=NeMoRLEnvConfig(
            host="127.0.0.1",
            port=0,
            name="nemorl_env",
            entrypoint="app.py",
            image="pytorch/pytorch@sha256:test",
            author_image="author:test",
            aime25_path="aime25.jsonl",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            sandbox_provider={"opensandbox": {"connection": {"domain": "cell4", "api_key": "gpu-control-only"}}},
            author_sandbox_provider={
                "opensandbox": {"connection": {"domain": "cell3", "api_key": "cpu-control-only"}}
            },
            author_image_auth={"username": "test", "password": "registry-only"},
        ),
        server_client=client,
    )


@pytest.fixture
def response():
    return NeMoGymResponse(
        id="author-response",
        created_at=0,
        model="policy_model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


@pytest.mark.parametrize("url_list", [False, True])
async def test_run_isolates_author_and_sets_256k(agent, response, monkeypatch, url_list):
    monkeypatch.setenv("WANDB_API_KEY", "host-wandb-only")
    model = agent.server_client.global_config_dict.policy_model.responses_api_models.vllm_model
    if url_list:
        model.base_url = [model.base_url]
    sandbox = AsyncMock()
    sandbox.__aenter__.return_value = sandbox
    sandbox.__aexit__.return_value = False
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload(local, remote):
        assert remote == "/rollout/author.tar.gz"
        with tarfile.open(local) as archive:
            names = archive.getnames()
            assert "Gym/responses_api_agents/nemorl_env/__init__.py" in names
            assert "Gym/responses_api_agents/nemorl_env/author_worker.py" in names
            assert "Gym/responses_api_agents/nemorl_env/task_environment/train_math.jsonl" in names
            assert not any(
                name.endswith(("math_eval.jsonl", "evaluate.py", "aime25_benchmark.jsonl")) for name in names
            )
            assert not any(part.startswith(".") or part == "tests" for name in names for part in Path(name).parts)

    async def download(remote, local):
        Path(local).write_text(response.model_dump_json() if remote.endswith("response.json") else "source patch")

    sandbox.upload.side_effect = upload
    sandbox.download.side_effect = download
    body = NeMoRLEnvRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Improve loss"))
    with (
        patch("responses_api_agents.nemorl_env.app.AsyncSandbox", return_value=sandbox) as create,
        patch.object(
            NeMoRLEnvAgent, "_evaluate", new=AsyncMock(return_value={"reward": 0.5, "completed": 1})
        ) as evaluate,
        patch("asyncio.create_subprocess_exec", side_effect=AssertionError("author must not run on host")),
    ):
        result = await agent.run(body)
    provider, spec = create.call_args.args
    assert provider["opensandbox"]["connection"]["domain"] == "cell3"
    assert spec.resources.gpu is None
    assert spec.ttl_s == 5400
    assert sandbox.exec.await_args.kwargs["timeout_s"] == 4200
    assert spec.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert spec.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"
    assert spec.env["DISABLE_COMPACT"] == "1"
    assert spec.env["NEMO_GYM_EXTRA_ROOTS"] == "/rollout/Gym"
    job = json.loads(spec.files["/rollout/job.json"])
    assert job["max_turns"] == 200 and job["research_seconds"] == 3600 and job["train_seconds"] == 3600
    assert job["model"]["model"] == "nvidia/qwen/qwen3.8-27b"
    assert isinstance(job["model"]["base_url"], list if url_list else str)
    for secret in ("cpu-control-only", "gpu-control-only", "registry-only", "host-wandb-only"):
        assert secret not in json.dumps({"env": spec.env, "files": spec.files})
    assert spec.provider_options["image_auth"]["password"] == "registry-only"
    evaluate.assert_awaited_once_with("source patch")
    assert result.reward == 0.5 and result.response.id == response.id
    sandbox.__aexit__.assert_awaited_once()


def test_training_token_capture_fails_closed(agent):
    cfg = agent.config.model_copy(update={"token_id_capture": True})
    with pytest.raises(ValueError, match="rollout evaluation only"):
        NeMoRLEnvAgent(config=cfg, server_client=agent.server_client)


def test_shipped_config_resolves_separate_cells_and_one_hour(monkeypatch):
    for name in (
        "OPENSANDBOX_CELL3_API_KEY",
        "OPENSANDBOX_CELL4_API_KEY",
        "NEMORL_ENV_REGISTRY_USER",
        "NEMORL_ENV_REGISTRY_PASSWORD",
        "NEMORL_ENV_AUTHOR_IMAGE",
    ):
        monkeypatch.setenv(name, "test-only")
    for name in ("OPENSANDBOX_CELL3_DOMAIN", "OPENSANDBOX_CELL4_DOMAIN"):
        monkeypatch.setenv(name, name.lower())
    merged = OmegaConf.load(Path(__file__).parents[1] / "config.yaml")
    cfg = OmegaConf.to_container(merged.nemorl_env.responses_api_agents.nemorl_env, resolve=True)
    env = NeMoRLEnvAgent(
        config=NeMoRLEnvConfig(**cfg, name="nemorl_env", host="127.0.0.1", port=0),
        server_client=ServerClient.model_construct(global_config_dict=merged),
    )
    assert env.config.train_seconds == 3600
    assert env.config.research_seconds == 3600 and env.config.max_turns == 200
    assert env._author_provider["opensandbox"]["connection"]["domain"] == "opensandbox_cell3_domain"
    assert env._provider["opensandbox"]["connection"]["domain"] == "opensandbox_cell4_domain"
    assert not env._token_id_capture_enabled()


async def test_source_training_is_separate_from_trusted_evaluation(agent, tmp_path, monkeypatch):
    (tmp_path / "aime25.jsonl").write_text('{"question":"held out", "expected_answer":"42"}\n')
    monkeypatch.setenv("WANDB_API_KEY", "test-only-secret")
    sandboxes = [AsyncMock(), AsyncMock()]
    events = []
    for stage, sandbox in zip(("train", "eval"), sandboxes, strict=True):
        sandbox.__aenter__.return_value = sandbox
        sandbox.__aexit__.return_value = False
        sandbox.start.side_effect = lambda stage=stage: events.append(f"start:{stage}")
        sandbox.__aexit__.side_effect = lambda *args, stage=stage: events.append(f"close:{stage}")
    sandboxes[0].exec.return_value = SandboxExecResult(
        stderr=None,
        stdout='NEMORL_ENV_RESULT={"reward":999}\nNEMORL_ENV_TRAIN_RESULT={"inner_steps":26,"train_reward":0.5}\n',
        return_code=0,
    )
    sandboxes[0].download.side_effect = lambda remote, local: Path(local).write_bytes(b"tensor-fixture")
    uploaded = []

    async def evaluate(command, **kwargs):
        assert uploaded and not uploaded[0].exists()
        return SandboxExecResult(
            stderr=None,
            stdout='NEMORL_ENV_RESULT={"reward":0.25,"completed":1}\n',
            return_code=0,
        )

    sandboxes[1].exec.side_effect = evaluate

    async def upload(local, remote):
        assert Path(local).read_bytes() == b"tensor-fixture"
        assert remote == "/testbed/model.safetensors"
        uploaded.append(Path(local))

    sandboxes[1].upload.side_effect = upload
    with (
        patch("responses_api_agents.nemorl_env.app.PARENT_DIR", tmp_path),
        patch("responses_api_agents.nemorl_env.app.AsyncSandbox", side_effect=sandboxes) as create,
    ):
        score = await agent._evaluate("source and recipe patch")
    assert events == ["start:train", "close:train", "start:eval", "close:eval"]
    assert score == {"reward": 0.25, "completed": 1, "inner_steps": 26, "train_reward": 0.5}
    train, evaluation = [call.args[1] for call in create.call_args_list]
    assert all(call.args[0]["opensandbox"]["connection"]["domain"] == "cell4" for call in create.call_args_list)
    assert train.resources.gpu == evaluation.resources.gpu == 1
    assert train.files["/root/change.diff"] == "source and recipe patch"
    assert "WANDB_API_KEY" not in train.env
    assert evaluation.env["WANDB_API_KEY"] == "test-only-secret"
    for path in ("/root/aime25.jsonl", "/root/math_eval.jsonl", "/testbed/evaluate.py"):
        assert path not in train.files and path in evaluation.files
    for path in ("/root/change.diff", "/testbed/recipe.yaml", "/testbed/launch_inner.py"):
        assert path in train.files and path not in evaluation.files
    train_rows = [json.loads(line) for line in train.files["/testbed/train_math.jsonl"].splitlines()]
    eval_rows = [json.loads(line) for line in evaluation.files["/root/math_eval.jsonl"].splitlines()]
    assert len(train_rows) == 480 and len(eval_rows) == 32
    assert not {row["input"] for row in train_rows} & {row["input"] for row in eval_rows}
    sandboxes[0].download.assert_awaited_once()
    sandboxes[1].upload.assert_awaited_once()
    for stage, sandbox in zip(("train", "eval"), sandboxes, strict=True):
        setup, execute = sandbox.exec.call_args_list
        assert setup.args[0].endswith(f"bash run.sh {stage} setup")
        assert setup.kwargs["timeout_s"] == 7200
        assert execute.args[0].endswith(f"bash run.sh {stage} execute")
        assert "INNER_TRAIN_SECONDS=3480" in execute.args[0]
        assert execute.kwargs["timeout_s"] == (3600 if stage == "train" else 7200)


@pytest.mark.parametrize("failure", [None, "download", "upload"])
async def test_checkpoint_transfers_are_serialized_and_release_after_errors(agent, tmp_path, failure):
    (tmp_path / "aime25.jsonl").write_text("{}\n")
    training = 0
    training_together = asyncio.Event()
    active_transfers = peak_transfers = 0
    failed = False

    async def transfer(direction, local):
        nonlocal active_transfers, peak_transfers, failed
        active_transfers += 1
        peak_transfers = max(peak_transfers, active_transfers)
        try:
            await asyncio.sleep(0.01)
            if direction == failure and not failed:
                failed = True
                raise RuntimeError("transfer failed")
            if direction == "download":
                Path(local).write_bytes(b"weights")
            else:
                assert Path(local).read_bytes() == b"weights"
        finally:
            active_transfers -= 1

    def create(provider, spec):
        evaluation = "/root/aime25.jsonl" in spec.files
        sandbox = AsyncMock()
        sandbox.__aenter__.return_value = sandbox
        sandbox.__aexit__.return_value = False

        async def execute(command, **kwargs):
            nonlocal training
            if not evaluation and command.endswith(" execute"):
                training += 1
                if training == 3:
                    training_together.set()
                await training_together.wait()
            record = (
                'NEMORL_ENV_RESULT={"reward":0.25,"completed":1}'
                if evaluation
                else 'NEMORL_ENV_TRAIN_RESULT={"inner_steps":2,"train_reward":0.1}'
            )
            return SandboxExecResult(stdout=record, stderr=None, return_code=0)

        async def download(remote, local):
            await transfer("download", local)

        async def upload(local, remote):
            await transfer("upload", local)

        sandbox.exec.side_effect = execute
        sandbox.download.side_effect = download
        sandbox.upload.side_effect = upload
        return sandbox

    with (
        patch("responses_api_agents.nemorl_env.app.PARENT_DIR", tmp_path),
        patch("responses_api_agents.nemorl_env.app.AsyncSandbox", side_effect=create),
    ):
        async with asyncio.timeout(2):
            results = await asyncio.gather(*(agent._evaluate("patch") for _ in range(3)), return_exceptions=True)
    assert training == 3 and peak_transfers == 1 and active_transfers == 0
    assert sum(isinstance(result, RuntimeError) for result in results) == (1 if failure else 0)
    assert sum(isinstance(result, dict) and result["completed"] == 1 for result in results) == (2 if failure else 3)


@pytest.mark.parametrize("failure", ["create", "setup", "training", "missing_record", "cleanup", "evaluation"])
async def test_verifier_errors_propagate_without_scoring(agent, tmp_path, failure):
    (tmp_path / "aime25.jsonl").write_text("{}\n")
    train, evaluation = AsyncMock(), AsyncMock()
    ok = SandboxExecResult(stderr=None, stdout="", return_code=0)
    failed = SandboxExecResult(stderr=None, stdout="stage failed", return_code=1)
    trained = SandboxExecResult(
        stderr=None,
        stdout='NEMORL_ENV_TRAIN_RESULT={"inner_steps":1,"train_reward":0.1}\n',
        return_code=0,
    )
    for sandbox in (train, evaluation):
        sandbox.__aenter__.return_value = sandbox
        sandbox.__aexit__.return_value = False
    train.exec.side_effect = [ok, trained]
    train.download.side_effect = lambda remote, local: Path(local).write_bytes(b"weights")
    evaluation.exec.side_effect = [ok, failed]
    if failure == "create":
        train.start.side_effect = TimeoutError("sandbox unavailable")
    elif failure == "setup":
        train.exec.side_effect = [failed]
    elif failure == "training":
        train.exec.side_effect = [ok, failed]
    elif failure == "missing_record":
        train.exec.side_effect = [ok, ok]
    elif failure == "cleanup":
        train.__aexit__.side_effect = TimeoutError("kill timed out")
    with (
        patch("responses_api_agents.nemorl_env.app.PARENT_DIR", tmp_path),
        patch("responses_api_agents.nemorl_env.app.AsyncSandbox", side_effect=[train, evaluation]) as create,
        pytest.raises((TimeoutError, RuntimeError)),
    ):
        await agent._evaluate("source patch")
    assert create.call_count == (2 if failure == "evaluation" else 1)


async def test_external_deadline_destroys_sandbox_even_when_training_ignores_it(agent, tmp_path):
    (tmp_path / "aime25.jsonl").write_text("{}\n")
    agent.config.train_seconds = 0.03
    sandbox = AsyncMock()
    sandbox.__aenter__.return_value = sandbox
    sandbox.__aexit__.return_value = False
    cancelled = asyncio.Event()

    async def execute(command, **kwargs):
        if command.endswith(" setup"):
            return SandboxExecResult(stdout="", stderr=None, return_code=0)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    sandbox.exec.side_effect = execute
    start = monotonic()
    with (
        patch("responses_api_agents.nemorl_env.app.PARENT_DIR", tmp_path),
        patch("responses_api_agents.nemorl_env.app.AsyncSandbox", return_value=sandbox) as create,
        pytest.raises(TimeoutError),
    ):
        await agent._evaluate("patched timer never stops")
    assert monotonic() - start < 2
    assert cancelled.is_set()
    sandbox.__aexit__.assert_awaited_once()
    sandbox.download.assert_not_awaited()
    assert create.call_count == 1


def test_failed_run_returns_http_error_not_zero_reward(agent, response):
    api = FastAPI()
    api.add_api_route("/run", agent.run, methods=["POST"])
    with (
        patch.object(NeMoRLEnvAgent, "_author", new=AsyncMock(return_value=(response, "patch"))),
        patch.object(NeMoRLEnvAgent, "_evaluate", new=AsyncMock(side_effect=TimeoutError("OSB unavailable"))),
        TestClient(api, raise_server_exceptions=False) as client,
    ):
        result = client.post("/run", json={"responses_create_params": {"input": "Improve loss"}})
    assert result.status_code == 500
    assert "reward" not in result.text
