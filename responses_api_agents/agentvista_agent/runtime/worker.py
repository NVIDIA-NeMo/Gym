# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://gitlab-master.nvidia.com/pritamb/agentvista
# Revision: 06f6a8f1532766d876415757107a11731b6148ae (Apache-2.0 adapter files).
# Original file: agentvista_harness/worker.py
"""One isolated OpenCode episode; inference always targets the Gym model proxy."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from PIL import Image
from utils.context_utils import pil_to_base64_data_uri

from responses_api_agents.agentvista_agent.runtime import HARNESS_VERSION, OPENCODE_VERSION
from responses_api_agents.agentvista_agent.runtime.data import inventory
from responses_api_agents.agentvista_agent.runtime.scoring import score


def configuration(config: dict, workspace: Path, mcp_config: Path) -> dict:
    endpoint = urlsplit(config["endpoint"])
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("Missing Gym model endpoint")
    if not 0 < config["output_tokens"] < config["context_tokens"] or config["max_steps"] < 1:
        raise ValueError("Invalid generation limits")
    root = Path(os.environ["AGENTVISTA_ROOT"])
    prompt = yaml.safe_load((root / "prompts/inference_prompts.yaml").read_text())["system_prompts"][
        "multi_tool_agent_search"
    ]
    prompt += "\nTools are exposed with the agentvista_ prefix. Input images retain the original_image names."
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": "nemo/" + config["model"],
        "share": "disabled",
        "autoupdate": False,
        "snapshot": False,
        "provider": {
            "nemo": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": config["endpoint"].rstrip("/"),
                    "apiKey": "EMPTY",
                    "timeout": 600000,
                },
                "models": {
                    config["model"]: {
                        "name": config["model"],
                        "limit": {
                            "context": config["context_tokens"],
                            "input": config["context_tokens"],
                            "output": config["output_tokens"],
                        },
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                    }
                },
            }
        },
        "compaction": {"reserved": config["output_tokens"]},
        "agent": {
            "build": {"steps": config["max_steps"], "prompt": prompt},
            "general": {"disable": True},
            "explore": {"disable": True},
        },
        "permission": {"*": "deny", "agentvista_*": "allow", "doom_loop": "allow"},
        "mcp": {
            "agentvista": {
                "type": "local",
                "enabled": True,
                "command": [
                    sys.executable,
                    "-m",
                    "responses_api_agents.agentvista_agent.runtime.mcp",
                    str(mcp_config),
                ],
                "timeout": config.get("mcp_timeout_ms", 180000),
            }
        },
    }


def actor_environment(workspace: Path) -> dict[str, str]:
    # No reference answers or judge credentials in the CLI/MCP environment.
    allowed = {
        "PATH",
        "AGENTVISTA_ROOT",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SERPAPI_KEY",
        "JINA_API_KEY",
        "IMGBB_API_KEY",
        "AGENTVISTA_VISIT_API_KEY",
        "AGENTVISTA_VISIT_API_ENDPOINT",
        "AGENTVISTA_VISIT_MODEL_NAME",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(
        PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[3]), os.environ["AGENTVISTA_ROOT"]]),
        PYTHONDONTWRITEBYTECODE="1",
        HOME=str(workspace / "home"),
        XDG_CONFIG_HOME=str(workspace / "config"),
        XDG_DATA_HOME=str(workspace / "data"),
        XDG_CACHE_HOME=str(workspace / "cache"),
        OPENCODE_CONFIG=str(workspace / "opencode.json"),
    )
    for key in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run(config: dict) -> dict:
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "artifacts": str(output),
        "harness_version": HARNESS_VERSION,
        "mask_sample": True,
        "reward": 0.0,
        "trajectory": [],
        "native_trajectory": [],
        "verify": {"status": "ERROR"},
        "task_success": None,
    }
    process = None

    def stop(signum=None, frame=None):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if signum is not None:
            raise InterruptedError(f"Worker interrupted by {signum}")

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        tasks = inventory(config["dataset"], config["image_folder"], config.get("task_selector", "all"))
        task = tasks[config["task_id"]]
        if task["source_sha256"] != config["source_sha256"]:
            raise ValueError("Prepared task/image hashes changed")
        command = config["command"]
        version = subprocess.check_output([command, "--version"], text=True).strip()
        if version != OPENCODE_VERSION:
            raise ValueError(f"Expected OpenCode {OPENCODE_VERSION}, got {version}")
        # The original visit configuration enables summarization. Require its fixed
        # auxiliary model explicitly instead of silently changing the tool contract.
        for key in (
            "VERIFIER_API_KEY",
            "VERIFIER_END_POINT",
            "VERIFIER_MODEL_NAME",
            "SERPAPI_KEY",
            "AGENTVISTA_VISIT_API_KEY",
            "AGENTVISTA_VISIT_API_ENDPOINT",
            "AGENTVISTA_VISIT_MODEL_NAME",
        ):
            if not os.environ.get(key):
                raise ValueError(f"Missing tool environment variable: {key}")
        workspace = output / "workspace"
        workspace.mkdir(exist_ok=True)
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
        inputs = workspace / "inputs"
        inputs.mkdir(exist_ok=True)
        attachments = []
        for index, path in enumerate(task["images"]):
            name = "original_image" + (f"_{index}" if index else "")
            uri = pil_to_base64_data_uri(Image.open(path))
            import base64

            target = inputs / (name + ".jpg")
            target.write_bytes(base64.b64decode(uri.split(",", 1)[1]))
            attachments.extend(["--file", str(target)])
        mcp = output / "mcp.json"
        mcp.write_text(
            json.dumps(
                {
                    "output": str(output / "tools"),
                    "images": task["images"],
                    "web_search_max_calls": config.get("web_search_max_calls", 100),
                    "image_search_max_calls": config.get("image_search_max_calls", 100),
                }
            )
        )
        (workspace / "opencode.json").write_text(json.dumps(configuration(config, workspace, mcp)))
        instruction = task["question"]
        (output / "prompt.txt").write_text(instruction)
        with (
            (output / "events.jsonl").open("w") as stdout,
            (output / "opencode.stderr").open("w") as stderr,
        ):
            process = subprocess.Popen(
                [
                    command,
                    "run",
                    "--format",
                    "json",
                    "--title",
                    "AgentVista episode",
                    "--model",
                    "nemo/" + config["model"],
                    *attachments,
                    "--",
                    instruction,
                ],
                cwd=workspace,
                env=actor_environment(workspace),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=config.get("agent_timeout", 5400))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                stop()
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines() if line.strip()]
        texts = [e["part"]["text"] for e in events if e.get("type") == "text"]
        result.update(
            trajectory=events,
            native_trajectory=events,
            agent_output=texts[-1] if texts else "",
            returncode=process.returncode,
            timed_out=timed_out,
        )
        if process.returncode or timed_out or any(e.get("type") == "error" for e in events) or not texts:
            raise RuntimeError("OpenCode episode did not complete cleanly; inspect native artifacts")
        verdict = score(task["question"], task["record"].get("solution"), texts)
        result.update(
            verify=verdict,
            reward=verdict["reward"],
            mask_sample=False,
            task_success=verdict["reward"] == 1,
            reward_kind="llm_judged",
            deterministically_verified=False,
        )
    except Exception as exc:
        result.update(failure_kind="agentvista:episode", error=f"{type(exc).__name__}: {exc}")
    finally:
        stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(output / "result.json")
    return result


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
