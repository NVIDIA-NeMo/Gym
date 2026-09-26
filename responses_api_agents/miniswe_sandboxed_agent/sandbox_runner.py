# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run pinned mini-SWE inside the task sandbox against Gym's model API."""

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
from minisweagent import __version__
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from openai.types.chat import ChatCompletionMessageToolCall


MINI_CONFIG = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text())


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


class RunnerCancelled(BaseException):
    pass


def responses_input(messages: list[dict]) -> list[dict]:
    items = []
    for message in messages:
        if message["role"] == "tool":
            items.append(
                {"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]}
            )
        elif "response_output" in message.get("extra", {}):
            items.extend(message["extra"]["response_output"])
        else:
            items.append({"role": message["role"], "content": message.get("content", "")})
    return items


class GymModel:
    def __init__(self, payload: dict, artifact: dict, checkpoint) -> None:
        self.payload = payload
        self.artifact = artifact
        self.checkpoint = checkpoint
        self.length_limited: list[bool] = []

    def query(self, messages: list[dict]) -> dict:
        params = self.payload["model_params"].copy()
        params["input"] = responses_input(messages)
        params["tools"] = [{"type": "function", **BASH_TOOL["function"], "strict": False}]
        request = Request(
            self.payload["model_base_url"].rstrip("/") + "/responses",
            data=json.dumps(params).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy_key",
                "x-session-id": self.payload["session_id"],
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.payload["model_timeout_sec"]) as result:
                response = json.load(result)
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            overflow = error.code == 400 and (
                "context_length_exceeded" in detail
                or "context length" in detail.lower()
                or "maximum model length" in detail.lower()
                or ("max_tokens" in detail and "too large" in detail.lower())
            )
            if overflow:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": detail,
                        "extra": {"exit_status": "ContextWindowExceeded", "submission": ""},
                    }
                ) from error
            raise RuntimeError(f"Model HTTP {error.code}: {detail}") from error
        self.artifact["model_history"].append({"request": params, "response": response, "timestamp": time.time()})
        self.checkpoint()
        length_limited = (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
        self.length_limited.append(length_limited)
        output = response["output"]
        calls = [
            ChatCompletionMessageToolCall(
                id=item["call_id"], type="function", function={"name": item["name"], "arguments": item["arguments"]}
            )
            for item in output
            if item["type"] == "function_call"
        ]
        actions = parse_toolcall_actions(
            calls,
            format_error_template=MINI_CONFIG["model"]["format_error_template"],
            template_kwargs={"finish_reason": "length" if length_limited else "stop"},
        )
        return {
            "role": "assistant",
            "content": "\n".join(
                part["text"]
                for item in output
                if item["type"] == "message"
                for part in item["content"]
                if part["type"] == "output_text"
            ),
            "tool_calls": [call.model_dump() for call in calls],
            "extra": {"actions": actions, "response_output": output},
        }

    def format_message(self, **kwargs: Any) -> dict:
        return kwargs

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        messages = format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=[{"exception_info": None, **output} for output in outputs],
            observation_template=MINI_CONFIG["model"]["observation_template"],
            template_vars=template_vars,
        )
        for observation, output in zip(messages, outputs):
            if output.get("images"):
                observation["content"] = [{"type": "input_text", "text": observation["content"]}] + [
                    {"type": "input_image", "image_url": uri} for uri in output["images"]
                ]
        return messages

    def get_template_vars(self) -> dict:
        return {}

    def serialize(self) -> dict:
        return {"info": {"model_transport": "nemo_gym_responses"}}


class SandboxEnvironment:
    """Commands are local to this process, which runs inside the task sandbox."""

    def __init__(self, model: GymModel, payload: dict, artifact: dict, checkpoint) -> None:
        self.model, self.payload = model, payload
        self.artifact, self.checkpoint = artifact, checkpoint

    def execute(self, action: dict) -> dict:
        tool = {
            "tool_call_id": action["tool_call_id"],
            "model_index": len(self.artifact["model_history"]),
            "started_at": time.time(),
            "status": "incomplete",
        }
        self.artifact["tool_history"].append(tool)
        self.checkpoint()
        # DefaultAgent saves after a step. Keep the assistant decision when a
        # running shell command is interrupted before that step finishes.
        self.agent.save(self.agent.config.output_path)
        started = time.monotonic()
        timeout = self.payload["step_timeout_sec"]
        process = subprocess.Popen(
            ["bash", "-c", action["command"]],
            cwd=self.payload["workdir"],
            env=os.environ | MINI_CONFIG["environment"]["env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        for path in (Path(self.payload["artifact_directory"]) / "processes", Path(self.payload["process_registry"])):
            with path.open("a") as stream:
                stream.write(f"{process.pid}\n")
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        except RunnerCancelled:
            tool.update(
                completed_at=time.time(),
                duration_ms=(time.monotonic() - started) * 1000,
                status="cancelled",
                error_type="cancelled",
            )
            self.checkpoint()
            raise
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        images = []
        try:
            tool_result = json.loads(output)
            for part in tool_result.get("content", []):
                if part.get("type") == "image":
                    images.append(f"data:{part['mimeType']};base64,{part.pop('data')}")
            if images:
                output = json.dumps(tool_result)
        except (ValueError, AttributeError, KeyError, TypeError):
            pass
        outcome = {
            "output": output,
            "returncode": -1 if timed_out else process.returncode,
            "images": images,
            "exception_info": f"Command timed out after {timeout} seconds." if timed_out else None,
        }
        messages = self.model.format_observation_messages({"extra": {"actions": [action]}}, [outcome])
        tool.update(
            message=messages[0],
            completed_at=time.time(),
            duration_ms=(time.monotonic() - started) * 1000,
            status="timeout" if timed_out else "failed" if process.returncode else "completed",
            error_type="timeout" if timed_out else None,
        )
        self.checkpoint()
        LocalEnvironment._check_finished(self, outcome)
        return outcome

    def get_template_vars(self) -> dict:
        return platform.uname()._asdict()

    def serialize(self) -> dict:
        return {"info": {"environment_type": "gym_sandbox"}}


def run(directory: Path) -> None:
    payload = json.loads((directory / "input.json").read_text())
    payload["artifact_directory"] = str(directory)
    artifact = {"model_history": [], "tool_history": []}

    def checkpoint():
        write_json(directory / "result.json", artifact)

    model = GymModel(payload, artifact, checkpoint)
    environment = SandboxEnvironment(model, payload, artifact, checkpoint)
    agent = DefaultAgent(
        model,
        environment,
        system_template=MINI_CONFIG["agent"]["system_template"],
        instance_template=MINI_CONFIG["agent"]["instance_template"],
        step_limit=payload["step_limit"],
        cost_limit=0,
        output_path=directory / "trajectory.json",
    )
    environment.agent = agent

    def cancel(signum, frame):
        raise RunnerCancelled()

    signal.signal(signal.SIGTERM, cancel)
    termination = {"reason": "completed"}
    try:
        info = agent.run(payload["instruction"])
        if info.get("exit_status") == "RepeatedFormatError" and all(
            model.length_limited[-agent.n_consecutive_format_errors :]
        ):
            info["exit_status"] = "OutputTokenLimitExceeded"
            agent.messages[-1]["content"] = "OutputTokenLimitExceeded"
        if info.get("exit_status") != "Submitted":
            termination = {"reason": "nonzero_exit", "detail": info.get("exit_status")}
    except RunnerCancelled:
        termination = {"reason": "cancelled"}
    except Exception as error:
        termination = {"reason": "infrastructure_error", "detail": f"{type(error).__name__}: {error}"}
    finally:
        agent.save(agent.config.output_path)
        artifact.update(
            termination=termination,
            harness_version=__version__,
            runtime={"hostname": platform.node(), "pid": os.getpid(), "python": sys.executable},
        )
        checkpoint()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
