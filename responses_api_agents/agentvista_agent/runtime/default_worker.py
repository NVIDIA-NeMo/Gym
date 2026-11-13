# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the upstream AgentVista greedy harness with policy traffic through Gym."""

import argparse
import json
import os
import re
import signal
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from responses_api_agents.agentvista_agent.runtime import HARNESS_VERSION
from responses_api_agents.agentvista_agent.runtime.code_sandbox import install_code_sandbox
from responses_api_agents.agentvista_agent.runtime.data import inventory
from responses_api_agents.agentvista_agent.runtime.paths import output_path
from responses_api_agents.agentvista_agent.runtime.scoring import score
from responses_api_agents.agentvista_agent.runtime.source import verify_source


def _gym_messages(messages: list[dict]) -> list[dict]:
    """Copy history for Gym, omitting provider-only tool-call stream indices."""
    copied = deepcopy(messages)
    for message in copied:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    call.pop("index", None)
                    if isinstance(call.get("function"), dict):
                        call["function"].pop("index", None)
    return copied


def _native_reply(reply: object) -> object:
    """Remove Gym's inline reasoning envelope before the upstream answer/tool parser."""
    if isinstance(reply, dict):
        reply = {key: value for key, value in reply.items() if key != "function_call" or value is not None}
    text = reply if isinstance(reply, str) else reply.get("content") if isinstance(reply, dict) else None
    if not isinstance(text, str):
        return reply
    match = re.match(r"\s*<think>(.*?)</think>", text, re.DOTALL)
    if not match:
        # An interrupted reasoning block is never a final answer, even if it contains answer tags.
        if re.match(r"\s*<think>", text):
            return "" if isinstance(reply, str) else {**reply, "content": ""}
        return reply
    content = text[match.end() :]
    return content if isinstance(reply, str) else {**reply, "content": content, "reasoning_content": match.group(1)}


def run(config: dict) -> dict:
    """Execute one original-harness episode and preserve its original trajectory."""
    output = output_path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "artifacts": str(output),
        "harness_version": HARNESS_VERSION,
        "harness": "default",
        "mask_sample": True,
        "reward": 0.0,
        "verify": {"status": "ERROR"},
        "native_trajectory": [],
    }

    def interrupt(signum: int, frame: object) -> None:
        raise InterruptedError(f"Original harness interrupted by signal {signum}")

    sandbox = None
    close_sandbox = lambda: None
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        result["failure_kind"] = "agentvista:setup"
        for key in ("VERIFIER_API_KEY", "VERIFIER_END_POINT", "VERIFIER_MODEL_NAME"):
            if not os.environ.get(key):
                raise ValueError(f"Missing judge environment variable: {key}")
        source = Path(config["source_root"]).resolve(strict=True)
        result["source_provenance"] = verify_source(source)
        sys.path.insert(0, str(source))
        task = inventory(config["dataset"], config["image_folder"], config.get("task_selector", "all"))[
            config["task_id"]
        ]
        if task["source_sha256"] != config["source_sha256"]:
            raise ValueError("Prepared task/image hashes changed")
        endpoint = config["endpoint"].rstrip("/")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Missing Gym model endpoint")
        os.environ.update(
            REASONING_API_KEY="EMPTY",
            REASONING_END_POINT=endpoint + "/chat/completions",
            REASONING_MODEL_NAME=config["model"],
            ENABLE_FUNCTION_CALLING="true",
            ENABLED_TOOLS="web_search,image_search,visit,code_interpreter",
        )
        for key in ("REASONING_API_KEY_2", "REASONING_END_POINT_2"):
            os.environ.pop(key, None)
        # Load the original prompt initializer and exact single-sample algorithm.
        sandbox, close_sandbox = install_code_sandbox(config, output)
        import infer
        from engine.api import api_model_caller, api_processors

        original_policy = api_model_caller.call_vision_api

        def gym_policy(model_name, messages, *args, **kwargs):
            return _native_reply(original_policy(model_name, _gym_messages(messages), *args, **kwargs))

        api_model_caller.call_vision_api = gym_policy

        tool_config = yaml.safe_load((source / "configs/tool_configs.yaml").read_text())
        for key in ("api_key", "api_endpoint", "model_name"):
            tool_config["visit"][key] = os.environ["AGENTVISTA_VISIT_" + key.upper()]
        tool_config["code_interpreter"]["work_dir"] = str(output / "kernel")
        args = argparse.Namespace(
            model_name=config["model"],
            output_dir=str(output / "original"),
            image_folder=config["image_folder"],
            inference_prompts_path=str(source / "prompts/inference_prompts.yaml"),
            system_prompt_key="multi_tool_agent_search",
            max_turns=config.get("default_max_turns", 30),
            max_images=config.get("default_max_images", 100),
            max_total_tokens=config.get("context_tokens", 65536),
            max_pixels=2000000,
            min_pixels=40000,
            image_search_max_calls=config.get("image_search_max_calls", 100),
            web_search_max_calls=config.get("web_search_max_calls", 100),
            bbox_format="norm999",
            tool_configs=tool_config,
        )
        prompts = infer.load_inference_prompts(args)
        system, tool, _, feedback, tts_tool = infer.setup_global_prompts(prompts)
        api_processors.set_global_prompts(system, tts_tool, feedback)
        api_model_caller.set_global_prompts(feedback, tts_tool)
        verdicts, judge_errors = [], []

        def strict_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict) -> tuple:
            # The original judge swallows provider errors as zero; keep those unscored.
            try:
                verdict = score(prompt, ground_truth, predict_str_list)
                verdicts.append(verdict)
                return verdict["reward"], verdict["analysis"]
            except Exception as exc:
                judge_errors.append(str(exc))
                raise

        api_processors.compute_score = strict_score
        sampling = {
            "temperature": config.get("temperature", 0.0),
            "top_p": config.get("top_p", 1.0),
            "max_tokens": config.get("output_tokens", 8192),
        }
        result["failure_kind"] = "agentvista:execution"
        original = api_processors.process_single_sample(task["record"], args, sampling)
        if sandbox.failed:
            result["failure_kind"] = "agentvista:sandbox"
            raise RuntimeError("Code sandbox failed; episode is unscorable")
        if original is None:
            raise RuntimeError("Original AgentVista harness returned no result")
        trace = Path(args.output_dir) / str(original["question_id"]) / "traj.jsonl"
        native = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        result.update(
            native_trajectory=native,
            trajectory=native,
            agent_output=original.get("final_answer", ""),
            original_result=original,
            response_items=message_items(original["conversation_history"]),
        )
        # Upstream returns an ordinary result for exhausted budgets and API failures.
        if any(row.get("error") for row in native):
            raise RuntimeError("Original harness recorded a model/capture error; inspect trajectory")
        if judge_errors or len(verdicts) != 1:
            result["failure_kind"] = "agentvista:verifier"
            raise RuntimeError("Original answer judge did not complete: " + "; ".join(judge_errors))
        verdict = verdicts[0]
        result.pop("failure_kind", None)
        result.update(
            verify=verdict,
            reward=verdict["reward"],
            mask_sample=False,
            task_success=verdict["reward"] == 1,
            reward_kind="llm_judged",
            deterministically_verified=False,
        )
    except Exception as exc:
        result.setdefault("failure_kind", "agentvista:episode")
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        close_sandbox()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(output / "result.json")
        interpreter = sys.modules.get("tools.code_interpreter")
        if interpreter is not None:
            interpreter._kill_kernels_and_subprocesses()
    return result


def message_items(history: list[dict]) -> list[dict]:
    """Expose assistant text from the original transcript without inventing CLI events."""
    return [
        {
            "id": f"msg_{index}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": str(row["content"]), "annotations": []}],
        }
        for index, row in enumerate(history)
        if row.get("role") == "assistant" and row.get("content")
    ]


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
