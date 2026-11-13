# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Native instruction prefix: eval/run_batch.py (MIT), https://github.com/internlm/WildClawBench
# Revision: 0c6551f798f8028bdfdd065f10d8d6b048a7fd06. See benchmarks/wildclawbench/third-party-notices.md.
"""Trusted host bridge; Enroot shares the host PID and network namespaces."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from responses_api_agents.wildclawbench_agent.data import (
    task_spec,
    verify_source,
    workspace_digest,
)


def grader_failures(value: object, prefix: str = "") -> list[str]:
    """Retain rubric zeros but mask explicit judge/transport failures, including nested ones."""
    failures = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, str) and item in {"llm_judge_failed", "skipped: no api key"}:
                failures.append(name)
            if key.lower().endswith("judge_method") and item in (
                "failed",
                "rule_fallback",
                "regex_fallback",
                "keyword_fallback",
            ):
                failures.append(name)
            error_field = key.lower() in {"error", "exception"} or key.lower().endswith(("_error", "_exception"))
            if item and error_field:
                # Exact native policy-failure diagnostics; missing ground truth is never admitted.
                policy_failure = key == "error" and (
                    str(item)
                    in {
                        "results directory not found",
                        "results.md not found",
                        "results.md is empty",
                        "screenshot.png not found",
                        "screenshot.png too small, likely invalid",
                    }
                    or str(item).startswith(
                        (
                            "workspace not found: /tmp_workspace/results",
                            "output dir not found: /tmp_workspace/results",
                            "missing summary file: /tmp_workspace/results/",
                            "invalid json on line ",
                        )
                    )
                )
                if not policy_failure:
                    failures.append(name)
            failures.extend(grader_failures(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(grader_failures(item, f"{prefix}[{index}]"))
    return failures


def policy_client_refusal(message: dict) -> bool:
    """Final invalid requests (including context overflow) are policy outcomes, not outages."""
    return message.get("stopReason") == "error" and bool(
        re.match(r"^(?:HTTP )?(?:400|413|422)(?::|\s|$)", str(message.get("errorMessage", "")))
    )


def failure_fields(errors: list[str]) -> dict:
    if not errors:
        return {}
    kind = errors[0] if re.fullmatch(r"[a-z][a-z0-9_]*", errors[0]) else "grader_error"
    return dict(
        failure_kind=f"wildclawbench:{kind}",
        failure_reason="WildClawBench episode is unscorable; inspect private artifacts",
    )


def read_events(path: Path) -> list[dict]:
    """Only LF delimits JSONL; U+2028 in a tool result is ordinary JSON content."""
    events = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("Missing or malformed OpenClaw transcript")
    return events


def execute(config: dict) -> dict:
    root = Path(config["source_root"])
    output = Path(config["output"])
    verify_source(root)
    expected = task_spec(root, config["metadata"]["task_id"])
    prompt = expected.pop("prompt")
    expected["workspace_sha256"] = workspace_digest(root, expected["workspace"])
    if expected != config["metadata"] or config["input_prompt"] != prompt:
        raise ValueError("Prepared task, assets or actor prompt changed; prepare the dataset again")
    sys.path.insert(0, str(root))
    from src.agents.base import AgentTaskSpec
    from src.agents.openclaw import OpenClawAgent
    from src.utils.docker_utils import collect_output_from_container, remove_container, stop_background_process
    from src.utils.grading import run_grading
    from src.utils.openclaw_runtime import available_gateway_port, context_defaults
    from src.utils.task_parser import parse_task_md
    from src.utils.trace_artifacts import write_trace_artifact

    task = parse_task_md(root / expected["task_path"])
    required = {"BRAVE_API_KEY"} | {
        line.strip() for line in task.get("env", "").splitlines() if line.strip() and not line.strip().startswith("#")
    }
    missing = sorted(name for name in required if not os.environ.get(name, "").strip())
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    backend = OpenClawAgent(
        gateway_port=available_gateway_port(),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    model = config["model_name"]
    provider = dict(
        baseUrl=config["policy_base_url"],
        apiKey="gym-policy",
        api="openai-completions",
        models=[
            dict(
                id=model,
                name=model,
                reasoning=True,
                input=["text", "image"],
                contextWindow=config["context_window"],
                maxTokens=config["max_tokens"],
            )
        ],
    )
    timeout = task["timeout_seconds"]
    # Match the native runner's actor instruction prefix without exposing task grading sections.
    prefix = (
        f"You are an expert in a restricted, non-interactive environment. Solve the task efficiently before the timeout ({timeout}s). "
        "Run all processes in the foreground without user input or background services. "
        "Provide a complete, functional solution in a single pass with no placeholders. \n"
    )
    execution = None
    errors = []
    try:
        execution = backend.run_task(
            AgentTaskSpec(
                task_id=output.name,
                task=task,
                workspace_path=task["workspace_path"],
                prompt=prefix + prompt,
                timeout_seconds=timeout,
                output_dir=output,
                model=f"gym-policy/{model}",
                thinking=config["thinking"],
                models_config=dict(providers={"gym-policy": provider}),
                openclaw_defaults=context_defaults(config["context_window"], config["max_tokens"]),
            )
        )
        # Stop owned gateway/agent groups before verification and the next task.
        for process in (execution.agent_proc, execution.gateway_proc):
            if process is not None:
                stop_background_process(process)
        if execution.error:
            errors.append("agent_execution")
        returncode = execution.agent_proc.returncode if execution.agent_proc is not None else None
        deadline_reached = execution.elapsed_time >= timeout and returncode in (
            -signal.SIGTERM,
            -signal.SIGKILL,
            143,
            137,
        )
        usage = backend.collect_usage(output.name, output, execution.elapsed_time)
        (output / "usage.json").write_text(json.dumps(usage, indent=2))
        write_trace_artifact(output, output.name, usage)
        events = read_events(output / "chat.jsonl")
        assistants = [event["message"] for event in events if event.get("message", {}).get("role") == "assistant"]
        client_refusal = bool(assistants) and policy_client_refusal(assistants[-1])
        if returncode not in (None, 0) and not deadline_reached and not client_refusal:
            errors.append("agent_process_exit")
        if not assistants or (assistants[-1].get("stopReason") == "error" and not client_refusal):
            errors.append("policy_terminal_error_or_missing_assistant")
        if "falling back to embedded" in (output / "agent.log").read_text(errors="replace"):
            errors.append("gateway_fallback")
        for message in assistants:
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "thinking" and block.get("thinking"):
                    if block.get("thinkingSignature") not in ("reasoning", "reasoning_content", "reasoning_text"):
                        errors.append("reasoning_replay_signature_missing")
        gt = Path(task["workspace_path"]) / "gt"
        if gt.is_dir():
            subprocess.run(
                ["docker", "cp", str(gt), f"{output.name}:/tmp_workspace/gt"], check=True, capture_output=True
            )
        scores = run_grading(
            task_id=output.name,
            automated_checks=task["automated_checks"],
            output_dir=output,
            extra_env=task.get("env", ""),
            transcript_container_path=backend.transcript_container_path,
            write_error_score=True,
        )
        errors.extend(grader_failures(scores))
        native_reward = scores.get("overall_score")
        if isinstance(native_reward, bool) or not isinstance(native_reward, (int, float)):
            errors.append("missing_grader_reward")
            native_reward = None
        elif not math.isfinite(native_reward) or not 0 <= native_reward <= 1:
            errors.append("invalid_grader_reward")
            native_reward = None
        collect_output_from_container(output.name, output)
        return dict(
            **expected,
            evaluation_protocol=config["evaluation_protocol"],
            native_reward=native_reward,
            reward=0.0 if errors else native_reward,
            mask_sample=bool(errors),
            errors=errors,
            **failure_fields(errors),
            artifact_dir=str(output),
            events=events,
            elapsed_time=execution.elapsed_time,
            agent_returncode=returncode,
            task_deadline_reached=deadline_reached,
            policy_request_errors=[
                message.get("errorMessage") for message in assistants if message.get("stopReason") == "error"
            ],
        )
    finally:
        try:
            if execution is not None:
                for process in (execution.agent_proc, execution.gateway_proc):
                    if process is not None:
                        stop_background_process(process)
        finally:
            remove_container(output.name)


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    output = Path(config["output"])

    def terminate(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    if "--cleanup" in sys.argv:
        cleanup = subprocess.run(["docker", "rm", "-f", output.name], capture_output=True, text=True, errors="replace")
        if cleanup.returncode and f"No such container: {output.name}" not in cleanup.stderr:
            raise RuntimeError("Sandbox cleanup failed; inspect the private worker log")
        return
    try:
        result = execute(config)
    except Exception as exc:
        # Error details stay in restricted local logs, not public rollout rows.
        import traceback

        traceback.print_exc()
        result = dict(
            **config["metadata"],
            evaluation_protocol=config["evaluation_protocol"],
            reward=0.0,
            native_reward=None,
            mask_sample=True,
            errors=[type(exc).__name__],
            **failure_fields([type(exc).__name__]),
            artifact_dir=str(output),
            events=[],
        )
    temporary = output / "result.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(output / "result.json")


if __name__ == "__main__":
    main()
