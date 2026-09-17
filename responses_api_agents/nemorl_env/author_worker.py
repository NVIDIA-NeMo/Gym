# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native author entrypoint, executed only inside the CPU sandbox."""

import asyncio
import io
import json
import tarfile
import tempfile
from pathlib import Path

import uvicorn
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, ServerClient, set_global_aiohttp_client
from responses_api_agents.claude_code_agent.app import ClaudeCodeAgent, ClaudeCodeAgentConfig
from responses_api_agents.nemorl_env.app import CLAUDE_CODE_MODEL_ALIAS, GYM_REVISION, INSTRUCTIONS, NEMORL_REVISION
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


async def git(cwd: Path, *args: str) -> bytes:
    proc = await asyncio.create_subprocess_exec("git", *args, cwd=cwd, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed")
    return stdout


async def author(job: dict, server_client: ServerClient):
    with tempfile.TemporaryDirectory(prefix="nemorl_env_author_") as tmp:
        root = Path(tmp)
        work = root / "work"
        work.mkdir()
        task = Path(__file__).with_name("task_environment")
        for name in ("recipe.yaml", "train_math.jsonl"):
            (work / name).write_text((task / name).read_text())
        source = root / "source"
        await git(root, "clone", "--no-checkout", job["source_repository"], str(source))
        await git(source, "checkout", "--detach", NEMORL_REVISION)
        gym_path = "3rdparty/Gym-workspace/Gym"
        await git(source, "submodule", "update", "--init", "--depth", "1", "--", gym_path)

        for repo, revision, target in (
            (source, NEMORL_REVISION, work / "NeMo-RL"),
            (source / gym_path, GYM_REVISION, work / "NeMo-RL" / gym_path),
        ):
            archive = await git(repo, "archive", revision)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(target, filter="data")
        await git(work, "init", "-q")
        await git(work, "config", "user.email", "gym@nvidia.com")
        await git(work, "config", "user.name", "NeMo Gym")
        await git(work, "add", ".")
        await git(work, "commit", "-qm", "baseline")
        baseline = (await git(work, "rev-parse", "HEAD")).decode().strip()
        agent = ClaudeCodeAgent(
            config=ClaudeCodeAgentConfig(
                host="127.0.0.1",
                port=0,
                name="nemorl_env_author",
                entrypoint="app.py",
                resources_server=ResourcesServerRef(name="nemorl_env", type="resources_servers"),
                model_server=ModelServerRef(name="policy_model", type="responses_api_models"),
                model=CLAUDE_CODE_MODEL_ALIAS,
                token_id_capture=False,
                anthropic_api_key="local",
                max_turns=job["max_turns"],
                timeout=job["research_seconds"],
                bare=True,
                system_prompt=INSTRUCTIONS.format(
                    train_minutes=job["train_seconds"] // 60, research_minutes=job["research_seconds"] // 60
                ),
                cwd=str(work),
            ),
            server_client=server_client,
        )
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(job["body"])
        problem = str((body.metadata or {}).get("problem_statement") or "Improve post-training in NeMo RL and Gym.")
        body = body.model_copy(update={"input": [NeMoGymEasyInputMessage(role="user", content=problem)]})
        response = await agent._create_response(body, rollout_id=None)
        await git(work, "add", "-A", "--", "recipe.yaml", "NeMo-RL")
        diff = await git(work, "diff", "--cached", "--binary", baseline, "--", "recipe.yaml", "NeMo-RL")
        return response, diff.decode()


async def main():
    job_path = Path("/rollout/job.json")
    job = json.loads(job_path.read_text())
    job_path.unlink()
    client = ServerClient.model_construct(
        global_config_dict=OmegaConf.create(
            {
                "policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8000}}},
                "token_id_capture": {"enabled": False},
            }
        )
    )
    adapter = VLLMModel(
        config=VLLMModelConfig(
            name="policy_model",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            return_token_id_information=False,
            **job.pop("model"),
        ),
        server_client=client,
    )
    http = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    server = uvicorn.Server(uvicorn.Config(adapter.setup_webserver(), host="127.0.0.1", port=8000))
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if serving.done():
                await serving
                raise RuntimeError("Author model adapter failed to start")
            await asyncio.sleep(0.1)
        response, patch = await author(job, client)
        Path("/rollout/response.json").write_text(response.model_dump_json())
        Path("/rollout/change.diff").write_text(patch)
    finally:
        server.should_exit = True
        await serving
        await http.close()


if __name__ == "__main__":
    asyncio.run(main())
