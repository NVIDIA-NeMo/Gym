# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute one Apex harness rollout inside the task sandbox.

This file is uploaded by the Gym agent. It deliberately receives no verifier
metadata, rubric, gold answer, host credential, or judge configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path("/app/apex-gym")
OUTPUT = ROOT / "output"
MCP_ROOT = Path("/app/mcp_servers")
FOUNDRY_APPS_ROOT = ROOT / "foundry-apps"


def _prepare_foundry_apps() -> None:
    """Adapt Archipelago's public MCP layout to the names expected by the harness."""
    FOUNDRY_APPS_ROOT.mkdir(parents=True, exist_ok=True)
    for source_name, harness_name in (("fmp", "mercor-fmp"), ("edgar_sec", "mercor-edgarsec")):
        link = FOUNDRY_APPS_ROOT / harness_name
        if not link.exists():
            link.symlink_to(MCP_ROOT / source_name, target_is_directory=True)


def _jsonable_trajectory(agent: Any, request_usages: list[dict[str, int]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    cumulative_input = 0
    cumulative_output = 0
    cumulative_reasoning = 0
    for index, turn in enumerate(agent.get_trajectory_snapshot()):
        exported = dict(turn)
        if index < len(request_usages):
            usage = request_usages[index]
            cumulative_input += usage["input_tokens"]
            cumulative_output += usage["output_tokens"]
            cumulative_reasoning += usage["reasoning_tokens"]
            exported["n_input_tokens"] = cumulative_input
            exported["n_output_tokens"] = cumulative_output
            exported["n_thinking_tokens"] = cumulative_reasoning
        exported_input = dict(exported.get("input") or {})
        exported_input.pop("context_turns", None)
        exported["input"] = exported_input
        turns.append(exported)
    return turns


def _install_aiohttp_transport(client: Any) -> Any:
    """Use a pooled aiohttp session and retain server-reported token usage."""
    import aiohttp

    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
    client._request_usages = []
    client._n_input_tokens = 0
    client._n_output_tokens = 0

    def _usage_value(usage: dict[str, Any], *names: str) -> int:
        for name in names:
            value = usage.get(name)
            if value is not None:
                return int(value)
        return 0

    async def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{client._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {client._api_key}",
            "Content-Type": "application/json",
        }
        rate_limit_attempt = 0
        error_attempts = 0
        while True:
            try:
                async with session.post(url, json=body, headers=headers) as response:
                    text = await response.text()
                    if response.status == 429:
                        retry_after = response.headers.get("retry-after")
                        wait = (
                            float(retry_after)
                            if retry_after
                            else min(2**rate_limit_attempt + random.uniform(0, 1), 60)
                        )
                        rate_limit_attempt += 1
                        await asyncio.sleep(wait)
                        continue
                    if response.status >= 500 and error_attempts < 4:
                        error_attempts += 1
                        await asyncio.sleep(min(2**error_attempts + random.uniform(0, 1), 60))
                        continue
                    if response.status >= 400:
                        raise RuntimeError(f"model server returned {response.status}: {text[:500]}")
                    data = json.loads(text)
                    usage = dict(data.get("usage") or {})
                    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
                    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
                    output_details = dict(
                        usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
                    )
                    client._request_usages.append(
                        {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "reasoning_tokens": _usage_value(output_details, "reasoning_tokens"),
                        }
                    )
                    client._n_input_tokens += input_tokens
                    client._n_output_tokens += output_tokens
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                error_attempts += 1
                if error_attempts >= 5:
                    raise RuntimeError("model server failed after five transport errors") from exc
                await asyncio.sleep(min(2**error_attempts + random.uniform(0, 1), 60))

    client._post = _post
    return session


async def main() -> None:
    config = json.loads((ROOT / "runner_config.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    _prepare_foundry_apps()

    os.environ["APEX_INTERNAL_API_KEY"] = "unused"
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    if config.get("edgar_user_agent"):
        os.environ["EDGAR_USER_AGENT"] = config["edgar_user_agent"]

    from apex_harness.apex_agent.agent import ApexAgent
    from apex_harness.apex_agent.snapshots import create_initial_snapshot, write_final_snapshot
    from apex_harness.environments.archipelago.environment import ArchipelagoMCPEnvironment
    from apex_harness.providers import inference as inference
    from apex_harness.providers.inference import ComputeProviderSpec, create_client
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    inference.PROVIDERS["nemo_gym"] = ComputeProviderSpec(
        type="sampling_only",
        base_url=config["model_base_url"],
        key_env="APEX_INTERNAL_API_KEY",
        api="chat",
        openai_tools=True,
    )

    environment = ArchipelagoMCPEnvironment(
        environment_dir=ROOT / "environment",
        environment_name="apex-agents",
        session_id=uuid.uuid4().hex,
        trial_paths=TrialPaths(trial_dir=ROOT / "trial"),
        task_env_config=TaskEnvironmentConfig(),
        archipelago_mcp_root=str(MCP_ROOT),
        foundry_apps_root=str(FOUNDRY_APPS_ROOT),
        foundry_services=config.get("foundry_services") or [],
        world_id=config["world_id"],
        world_zip_path=str(ROOT / "world.zip"),
        task_file_dir="",
        fs_root_path="/filesystem",
        apps_data_root_path="/.apps_data",
    )
    agent = ApexAgent(
        logs_dir=ROOT / "logs",
        model_name=config["policy_model"],
        reward_spec={},
        max_turns=int(config["max_turns"]),
        # Structured messages go directly to the policy server. It owns the
        # tokenizer and context-window enforcement.
        context_window_size=0,
        max_tool_calls_per_turn=int(config["max_tool_calls_per_turn"]),
        max_tool_output_tokens=0,
    )
    client = create_client(SimpleNamespace(compute_provider="nemo_gym", model=config["policy_model"]))
    http_session = _install_aiohttp_transport(client)
    try:
        # set_client() is shared with the harness's token-ID/Tinker loop and
        # eagerly loads a Hugging Face tokenizer. The OpenAI-tools loop only
        # needs these fields and explicitly supports tokenizer=None.
        agent._client = client
        agent._tokenizer = None
        agent._sampling_params = {
            "model": config["policy_model"],
            "max_tokens": int(config["max_output_tokens"]),
            "temperature": float(config["temperature"]),
            "top_p": float(config["top_p"]),
        }
        agent._mid_tokens = []
        agent._end_tokens = []

        context = AgentContext(metadata={})
        await environment.start()
        await environment.reset_filesystem(
            world_zip_path=str(ROOT / "world.zip"),
            task_file_dir="",
            world_id=config["world_id"],
            foundry_services=config.get("foundry_services") or [],
        )
        snapshot = create_initial_snapshot(environment, {}, OUTPUT / "snapshots")
        shutil.copy2(snapshot.initial_path, OUTPUT / "initial.zip")
        await agent.setup(environment)
        await agent.run(config["instruction"], environment, context)
        context.n_input_tokens = client._n_input_tokens
        context.n_output_tokens = client._n_output_tokens
        final_snapshot, snapshot_metadata = write_final_snapshot(environment, snapshot)
        final_target = OUTPUT / "final.zip"
        final_snapshot.replace(final_target)

        with zipfile.ZipFile(final_target) as archive:
            artifact_manifest = sorted(name for name in archive.namelist() if name and not name.endswith("/"))
        metadata = context.metadata or {}
        result = {
            "task_id": config["task_id"],
            "world_id": config["world_id"],
            "final_answer": metadata.get("response", ""),
            "agent_mode": metadata.get("agent_mode"),
            "n_input_tokens": context.n_input_tokens,
            "n_output_tokens": context.n_output_tokens,
            "trajectory": _jsonable_trajectory(agent, client._request_usages),
            "artifact_manifest": artifact_manifest,
            "snapshot": snapshot_metadata,
        }
        (OUTPUT / "result.json").write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
    finally:
        try:
            await environment.stop()
        finally:
            await http_session.close()


if __name__ == "__main__":
    asyncio.run(main())
