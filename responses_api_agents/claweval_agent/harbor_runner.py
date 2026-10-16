# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one staged ClawEval task with OpenCode, media MCP, and native grading."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import sys
from pathlib import Path

from dotenv import dotenv_values

from responses_api_agents.agentic_vbench_agent.harbor_runner import (
    OPENCODE_OUTPUT_CAP,
    TRAINING_MODES,
    job_config,
    run_harbor,
)


ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _bool(value: str | None) -> bool | None:
    return None if value is None else value == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--backend", choices=("podman", "remote"), default="podman")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--verifier-timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=None)
    parser.add_argument("--model-timeout-ms", type=int, default=None)
    parser.add_argument("--tool-output-max-lines", type=int, default=None)
    parser.add_argument("--tool-output-max-bytes", type=int, default=None)
    parser.add_argument("--compaction-auto", choices=("true", "false"), default=None)
    parser.add_argument("--compaction-prune", choices=("true", "false"), default=None)
    parser.add_argument("--disable-title", action="store_true")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default=None)
    parser.add_argument("--subagents", action="store_true")
    parser.add_argument("--subagent-max-depth", type=int, default=1)
    parser.add_argument("--subagent-max-active", type=int, default=None)
    parser.add_argument("--delegation-prompt", default=None)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-api-key-env", required=True)
    args = parser.parse_args()

    if importlib.metadata.version("harbor") != "0.6.6":
        raise RuntimeError("ClawEval requires upstream harbor==0.6.6")
    if not ENV_NAME.fullmatch(args.judge_api_key_env):
        raise ValueError("judge-api-key-env must be an environment-variable name")
    from harbor.models.job.config import JobConfig

    task_path = args.task_path.resolve(strict=True)
    media_server = task_path / "steps" / "solve" / "workdir" / ".claw_tools" / "mcp_server.py"
    if not media_server.is_file():
        raise FileNotFoundError(media_server)

    credentials = dotenv_values(args.credentials_file) if args.credentials_file else {}
    judge_key = os.environ.get(args.judge_api_key_env) or credentials.get(args.judge_api_key_env)
    if not judge_key:
        raise ValueError(
            f"Native ClawEval grading requires {args.judge_api_key_env} in the environment or credentials file"
        )
    env = dict(os.environ)
    env.update(
        {
            "CLAW_EVAL_JUDGE_API_KEY": str(judge_key),
            "CLAW_EVAL_JUDGE_BASE_URL": args.judge_base_url,
            "CLAW_EVAL_JUDGE_MODEL": args.judge_model,
            OPENCODE_OUTPUT_CAP: str(args.output_tokens),
        }
    )

    raw_config = job_config(
        task_path=task_path,
        output=args.output.resolve(strict=True),
        runtime_root=args.runtime_root.resolve(),
        model=args.model,
        endpoint=args.endpoint,
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        backend=args.backend,
        max_turns=args.max_turns,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
        agent_timeout_multiplier=args.agent_timeout_multiplier,
        model_timeout_ms=args.model_timeout_ms,
        tool_output_max_lines=args.tool_output_max_lines,
        tool_output_max_bytes=args.tool_output_max_bytes,
        compaction_auto=_bool(args.compaction_auto),
        compaction_prune=_bool(args.compaction_prune),
        disable_title=args.disable_title,
        training_mode=args.training_mode,
        subagents_enabled=args.subagents,
        subagent_max_depth=args.subagent_max_depth,
        subagent_max_active=args.subagent_max_active,
        delegation_prompt=args.delegation_prompt,
    )
    opencode_config = raw_config["agents"][0]["kwargs"]["opencode_config"]
    opencode_config["mcp"] = {
        "claw": {
            "type": "local",
            "command": ["python3", "/workspace/.claw_tools/mcp_server.py"],
            "enabled": True,
        }
    }
    opencode_config.setdefault("permission", {})["claw_*"] = "allow"
    for agent in opencode_config.get("agent", {}).values():
        if isinstance(agent, dict) and isinstance(agent.get("permission"), dict):
            agent["permission"]["claw_*"] = "allow"

    config = JobConfig.model_validate(raw_config)
    config_path = args.output / "harbor_job.json"
    with config_path.open("x") as handle:
        handle.write(config.model_dump_json(indent=2) + "\n")
    command = [str(Path(sys.executable).with_name("harbor")), "run", "--config", str(config_path), "--yes"]
    if args.credentials_file:
        command.extend(["--env-file", str(args.credentials_file.resolve(strict=True))])
    gym_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [gym_root, env.get("PYTHONPATH")]))
    run_harbor(command, env=env, jobs_dir=args.output.resolve() / "jobs")


if __name__ == "__main__":
    main()
