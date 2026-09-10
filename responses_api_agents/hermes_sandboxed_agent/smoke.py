# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small real HTTP rollout: three Gym servers, no Ray scheduler or GPU allocation.

This driver does not implement agent tools, model replies or grading. Those run
in Hermes, the configured model proxy and the SWE-bench Pro resources server.
"""

import argparse
import asyncio
import contextlib
import json
import multiprocessing
import os
import socket
import time
from pathlib import Path
from uuid import uuid4

from responses_api_agents.hermes_sandboxed_agent.scoring import is_scored, summarize


def serve(config, name, log_path):
    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(config)
    with open(log_path, "w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import uvicorn
        from omegaconf import OmegaConf

        from nemo_gym.server_utils import ServerClient
        from resources_servers.swebench_pro.app import SWEBenchProResourcesServer
        from responses_api_agents.hermes_sandboxed_agent.app import HermesSandboxedAgent
        from responses_api_models.openai_model.app import SimpleModelServer

        cls = {
            "policy_model": SimpleModelServer,
            "swebench_pro_resources_server": SWEBenchProResourcesServer,
            "hermes_sandboxed_agent": HermesSandboxedAgent,
        }[name]
        block = next(iter(next(iter(config[name].values())).values()))
        client = ServerClient(head_server_config=config["head_server"], global_config_dict=OmegaConf.create(config))
        server = cls(config=cls.model_fields["config"].annotation(name=name, **block), server_client=client)
        app = server.setup_webserver()
        server.setup_liveness(app)
        server.setup_exception_middleware(app)
        server.setup_cancellation_middleware(app)
        uvicorn.run(app, host="0.0.0.0", port=block["port"], access_log=True)


def free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


async def collect(config, rows, output, processes):
    from omegaconf import OmegaConf

    from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status

    client = ServerClient(head_server_config=config["head_server"], global_config_dict=OmegaConf.create(config))
    deadline = time.monotonic() + 120
    for name in processes:
        while True:
            if not processes[name].is_alive():
                raise RuntimeError(f"{name} exited during startup; inspect its log")
            try:
                response = await client.get(server_name=name, url_path="/", _max_connection_retries=0)
                await raise_for_status(response)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(0.5)
    summaries = []
    for row in rows:
        row = row | {"_ng_rollout_id": uuid4().hex}
        print(f"RUN {row['instance_id']}", flush=True)
        try:
            response = await client.post(server_name="hermes_sandboxed_agent", url_path="/run", json=row)
            await raise_for_status(response)
            result = await get_response_json(response)
        except Exception as exc:
            result = {
                "instance_id": row["instance_id"],
                "score_valid": False,
                "failure_kind": "request_error",
                "failure_reason": str(exc),
                "_ng_failure_class": "agent_request_failed",
            }
        filename = "rollouts.jsonl" if is_scored(result) else "failures.jsonl"
        with (output / filename).open("a") as file:
            file.write(json.dumps(result) + "\n")
        summary = {
            key: result.get(key)
            for key in (
                "instance_id",
                "reward",
                "verifier_reward",
                "resolved",
                "evaluation_completed",
                "hermes_finished",
                "hermes_return_code",
                "hermes_error_type",
                "turns_used",
                "hermes_result_path",
                "score_valid",
                "failure_kind",
                "failure_reason",
                "agent_image_provenance",
                "image_provenance",
            )
        }
        summary["patch_bytes"] = len((result.get("model_patch") or "").encode())
        print(json.dumps(summary), flush=True)
        summaries.append(summary)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    metrics = summarize(summaries)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sif-dir", type=Path, help="Optional cache containing {dockerhub_tag}.sif images")
    parser.add_argument("--runtime-python", default="/opt/hermes/bin/python3")
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--sandbox-timeout", type=int, default=1800)
    parser.add_argument("--api-timeout", type=int, default=600)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "responses_api_models/openai_model/configs/openai_model.yaml"),
        OmegaConf.load(args.provider_config),
        OmegaConf.load(root / "responses_api_agents/hermes_sandboxed_agent/configs/hermes_sandboxed_agent.yaml"),
        OmegaConf.load(root / "resources_servers/swebench_pro/configs/swebench_pro.yaml"),
    )
    for name in list(config):
        if name.startswith("swebench_pro_example_"):
            del config[name]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    host = socket.gethostbyname(socket.gethostname())
    names = ["policy_model", "swebench_pro_resources_server", "hermes_sandboxed_agent"]
    for name in names:
        block = next(iter(next(iter(config[name].values())).values()))
        block.host, block.port = host, free_port()
        block.num_workers = 1
    config.head_server = {"host": host, "port": free_port()}
    config.policy_base_url = args.model_url
    config.policy_api_key = os.environ.get("SMOKE_MODEL_API_KEY", "gym")
    config.policy_model_name = args.model
    config.observability_enabled = True
    config.model_call_capture_dir = str(output / "model-calls")
    agent = config.hermes_sandboxed_agent.responses_api_agents.hermes_sandboxed_agent
    agent.resources_server.name = "swebench_pro_resources_server"
    agent.model = args.model
    agent.runtime_python = args.runtime_python
    agent.results_dir = str(output / "tasks")
    agent.max_turns, agent.max_tokens = args.max_turns, args.max_tokens
    agent.sandbox_timeout, agent.api_timeout = args.sandbox_timeout, args.api_timeout
    resources = config.swebench_pro_resources_server.resources_servers.swebench_pro
    if args.sif_dir:
        resources.image_template = str(args.sif_dir.resolve() / "{image_digest_hex}.sif")
    plain = OmegaConf.to_container(config, resolve=True)
    safe_config = json.loads(json.dumps(plain))
    safe_config["policy_api_key"] = "REDACTED"
    safe_config["policy_model"]["responses_api_models"]["openai_model"]["openai_api_key"] = "REDACTED"
    (output / "config.json").write_text(json.dumps(safe_config, indent=2))
    rows = {row["instance_id"]: row for row in (json.loads(line) for line in args.dataset.read_text().splitlines())}
    selected = [rows[instance_id] for instance_id in args.instance_id]
    (output / "inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(plain)
    processes = {}
    try:
        for name in names:
            process = multiprocessing.get_context("spawn").Process(
                target=serve, args=(plain, name, output / f"{name}.log")
            )
            process.start()
            processes[name] = process
        metrics = asyncio.run(collect(plain, selected, output, processes))
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join()
    if metrics["excluded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
