# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Pi/Hermes file-task smoke with a real model, NOT a scored GDP evaluation.

python -m resources_servers.gdpval.tests.live_sandbox_smoke --harness hermes \
    --model-url http://MODEL:8000/v1 --model n35-super-ga \
    --host 10.111.115.167 --image EXISTING_LINUX_IMAGE --output /absolute/new/run

Without --input, use a synthetic CSV fixture. With --input and --task-index,
download the real task's references and check generation/export, not correctness.
Only the judge (and CSV fixture download) is replaced. All sessions, model calls,
sandbox tools, artifact export and cleanup are real. This starts HTTP services
directly; it does not test the standard Gym CLI or an audited GDP runtime.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import uvicorn
from aiohttp import ClientTimeout
from fastapi import Request
from omegaconf import OmegaConf

import nemo_gym.global_config as global_config
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from nemo_gym.server_utils import request as http_request
from resources_servers.gdpval.app import (
    GDPValResourcesServer,
    GDPValResourcesServerConfig,
    GDPValVerifyRequest,
    GDPValVerifyResponse,
)
from resources_servers.gdpval.task_data import GDPFileTask, prepare_task
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


class GenerationOnlyResources(GDPValResourcesServer):
    """Test-only no-judge boundary; production input staging and export are unchanged."""

    async def verify(self, body: GDPValVerifyRequest, *, request: Request = None) -> GDPValVerifyResponse:
        directory = await self.export_deliverables(request.session[SESSION_ID_KEY])
        return GDPValVerifyResponse(
            **(body.model_dump() | {"deliverables_dir": str(directory)}),
            reward=0.0,
            mask_sample=True,
            failure_reason="Generation smoke only: judge deliberately not run; reward placeholder is not a score.",
        )


class ExportOnlyResources(GenerationOnlyResources):
    """Replace the synthetic fixture download only."""

    async def _stage_references(self, sandbox: AsyncSandbox, task: GDPFileTask) -> None:
        source = self.config.deliverables_root / "reference.csv"
        await sandbox.upload(source, "/workspace/input/reference.csv")


def make_agent(
    args: argparse.Namespace, config: dict[str, str | int], client: ServerClient
) -> SimpleResponsesAPIAgent:
    """Use the existing adapter; import optional dependencies only when selected."""
    common = dict(
        **config,
        model=args.model,
        model_server={"type": "responses_api_models", "name": "policy_model"},
    )
    if args.harness == "hermes":
        from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig

        return HermesAgent(
            config=HermesAgentConfig(
                **common,
                resources_server={"type": "resources_servers", "name": "resources"},
                enabled_toolsets=["terminal", "file"],
                max_turns=60,
                max_tokens=args.max_output_tokens,
                sandbox_runner_timeout_seconds=900,
            ),
            server_client=client,
        )
    from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig

    return PiAgent(
        config=PiAgentConfig(
            **common,
            pi_version="0.80.2",
            timeout=900 if args.input else 300,
            context_window=262144,
            max_output_tokens=args.max_output_tokens,
            thinking="low",
        ),
        server_client=client,
    )


async def run(args: argparse.Namespace) -> None:
    started = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.input:
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        source = rows[args.task_index]
    else:
        (args.output / "reference.csv").write_text("item,amount\nalpha,1\nalpha,5\nbeta,9\n")
        source = {
            "task_id": "synthetic-csv-contract",
            "prompt": "Use Python 3's standard-library csv module to read reference.csv and sum amount by item. "
            "Submit summary.csv with a header row exactly item,total. The reference first line is a header. "
            "Include one row for each item, sorted alphabetically. Read the output back before finishing. "
            "Do not copy the reference as output or install packages.",
            "reference_files": ["reference.csv"],
            "reference_file_urls": ["https://example.invalid/local-test-fixture"],
        }
    task = prepare_task(source, taskset="gdpval")
    agent_name = f"{args.harness}_agent"
    sockets = []
    for _ in range(4):
        sock = socket.socket()
        sock.bind(("0.0.0.0", 0))
        sockets.append(sock)
    names = ["resources", agent_name, "policy_model", "environment"]
    common = {
        name: {"host": args.host, "port": sock.getsockname()[1], "name": name, "entrypoint": "app.py"}
        for name, sock in zip(names, sockets, strict=True)
    }
    cfg = OmegaConf.create(
        {
            "resources": {"resources_servers": {"gdpval": common["resources"]}},
            agent_name: {"responses_api_agents": {agent_name: common[agent_name]}},
            "policy_model": {"responses_api_models": {"openai_model": common["policy_model"]}},
            "environment": {"environment_servers": {"single_agent_turn": common["environment"]}},
            "sandbox": {"docker": {"create": {"use_init": True}}},
        }
    )
    global_config._GLOBAL_CONFIG_DICT = cfg
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=cfg)
    resources_type = GenerationOnlyResources if args.input else ExportOnlyResources
    resources = resources_type(
        config=GDPValResourcesServerConfig(
            **common["resources"],
            sandbox_provider="sandbox",
            num_workers=1,
            image=args.image,
            deliverables_root=args.output,
            preconvert_office_to_pdf=False,
            judge_model_server={"type": "responses_api_models", "name": "policy_model"},
        ),
        server_client=client,
    )
    agent = make_agent(args, common[agent_name], client)
    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            **common["policy_model"],
            openai_base_url=args.model_url,
            openai_model=args.model,
            openai_api_key=os.environ.get("GDP_SMOKE_MODEL_KEY", os.environ.get("PI_SMOKE_MODEL_KEY", "dummy")),
            max_http_attempts=1,
        ),
        server_client=client,
    )
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            **common["environment"],
            resources_server={"type": "resources_servers", "name": "resources"},
            agent_server={"type": "responses_api_agents", "name": agent_name},
            default_episode_timeout_seconds=2100,
            cleanup_timeout_seconds=120,
        ),
        server_client=client,
    )
    servers = [
        uvicorn.Server(uvicorn.Config(component.setup_webserver(), log_level="info"))
        for component in (resources, agent, model, environment)
    ]
    tasks = [asyncio.create_task(server.serve(sockets=[sock])) for server, sock in zip(servers, sockets, strict=True)]
    try:
        async with asyncio.timeout(30):
            while not all(server.started for server in servers):
                if any(task.done() for task in tasks):
                    raise RuntimeError("A smoke server failed at startup")
                await asyncio.sleep(0.1)
        body = {
            "episode_id": {"rollout_id": f"{args.harness}-gdp-smoke-{uuid4().hex}"},
            "task": task.model_dump(mode="json"),
        }
        (args.output / "request.json").write_text(json.dumps(body, indent=2))
        url = f"http://127.0.0.1:{common['environment']['port']}/run"
        reply = await http_request("POST", url, json=body, timeout=ClientTimeout(total=2400))
        try:
            reply.raise_for_status()
            result = await reply.json()
        finally:
            reply.release()
        (args.output / "episode.json").write_text(json.dumps(result, indent=2))
        if result.get("failure"):
            raise RuntimeError(json.dumps(result["failure"]))
        verdict = result["result"]
        assert verdict["mask_sample"] is True
        directory = Path(verdict["deliverables_dir"])
        assert (directory / "finish_params.json").is_file()
        files = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "finish_params.json"
        ]
        values = None
        if not args.input:
            with (directory / "summary.csv").open() as stream:
                records = list(csv.DictReader(stream))
            values = {record["item"]: float(record["total"]) for record in records}
            assert len(records) == 2 and values == {"alpha": 6, "beta": 9}, records
        status = verdict["response"].get("status")
        closed = not resources._sessions
        passed = bool(files) and all(file["bytes"] > 0 for file in files) and closed and status == "completed"
        summary = {
            "generation_smoke" if args.input else "contract_smoke": "passed" if passed else "failed",
            "harness": args.harness,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": task.task_id.task_id,
            "task_index": args.task_index if args.input else None,
            "reference_download": "production HTTPS path" if args.input else "local CSV fixture",
            "image": args.image,
            "model": args.model,
            "max_output_tokens_per_call": args.max_output_tokens,
            "benchmark": False,
            "official_runtime": False,
            "judge_run": False,
            "reward": None,
            "accuracy_validated": False,
            "response_status": status,
            "response_metadata": verdict["response"].get("metadata"),
            "deliverables": str(directory),
            "files": files,
            "csv_values": values,
            "usage": verdict["response"].get("usage"),
            "resources_closed": closed,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        assert passed, "Generation/export smoke failed; inspect summary.json and episode.json"
        if args.harness == "hermes":
            assert summary["response_metadata"]["harness_execution"] == "sandbox"
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()


def main() -> None:
    """Run one explicitly requested live smoke with a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=["pi", "hermes"], default="pi")
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True, help="Host IP reachable from task containers")
    parser.add_argument("--image", required=True)
    parser.add_argument("--input", type=Path, help="Optional real GDP JSONL; otherwise use the CSV fixture")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--output", required=True, type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
