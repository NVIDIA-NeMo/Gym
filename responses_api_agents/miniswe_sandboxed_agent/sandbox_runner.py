# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run pinned mini-SWE inside the task sandbox; relay inference to Gym."""

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from minisweagent import __version__
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages, parse_toolcall_actions
from openai.types.chat import ChatCompletionMessageToolCall


MINI_CONFIG = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text())


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


class RunnerCancelled(BaseException):
    pass


class FileRelay:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index = 0

    def call(self, kind: str, **payload: Any) -> dict:
        index = self.index
        self.index += 1
        request = self.directory / f"request-{index}.json"
        response = self.directory / f"response-{index}.json"
        write_json(request, {"kind": kind, **payload})
        while not response.exists():
            time.sleep(0.025)
        result = json.loads(response.read_text())
        response.unlink()
        request.unlink(missing_ok=True)
        if error := result.get("error"):
            if error["type"] == "ContextWindowExceeded":
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": error["detail"],
                        "extra": {"exit_status": "ContextWindowExceeded", "submission": ""},
                    }
                )
            # Preserve the model transport's error type in the native trajectory.
            raise type(error["type"], (RuntimeError,), {})(error["detail"])
        return result


class GymModel:
    def __init__(self, relay: FileRelay) -> None:
        self.relay = relay
        self.length_limited: list[bool] = []

    def query(self, messages: list[dict]) -> dict:
        response = self.relay.call("query", messages=messages)["response"]
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

    def __init__(self, relay: FileRelay, model: GymModel, payload: dict) -> None:
        self.relay, self.model, self.payload = relay, model, payload

    def execute(self, action: dict) -> dict:
        self.relay.call("tool_started", action=action, started_at=time.time())
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
        for path in (self.relay.directory / "processes", Path(self.payload["process_registry"])):
            with path.open("a") as stream:
                stream.write(f"{process.pid}\n")
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
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
        self.relay.call(
            "tool_finished",
            action=action,
            message=messages[0],
            completed_at=time.time(),
            duration_ms=(time.monotonic() - started) * 1000,
            status="timeout" if timed_out else "failed" if process.returncode else "completed",
            error_type="timeout" if timed_out else None,
        )
        LocalEnvironment._check_finished(self, outcome)
        return outcome

    def get_template_vars(self) -> dict:
        return platform.uname()._asdict()

    def serialize(self) -> dict:
        return {"info": {"environment_type": "gym_sandbox"}}


def run(directory: Path) -> None:
    payload = json.loads((directory / "input.json").read_text())
    relay = FileRelay(directory)
    model = GymModel(relay)
    agent = DefaultAgent(
        model,
        SandboxEnvironment(relay, model, payload),
        system_template=MINI_CONFIG["agent"]["system_template"],
        instance_template=MINI_CONFIG["agent"]["instance_template"],
        step_limit=payload["step_limit"],
        cost_limit=0,
        output_path=directory / "trajectory.json",
    )

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
        trajectory = agent.save(agent.config.output_path)
        write_json(
            directory / "output.json",
            {
                "termination": termination,
                "mini_swe_trajectory": trajectory,
                "harness_version": __version__,
                "runtime": {"hostname": platform.node(), "pid": os.getpid(), "python": sys.executable},
            },
        )


if __name__ == "__main__":
    run(Path(sys.argv[1]))
