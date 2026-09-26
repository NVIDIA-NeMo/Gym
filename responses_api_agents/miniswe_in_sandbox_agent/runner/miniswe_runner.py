# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""mini-SWE loop that runs INSIDE the task sandbox and calls the Gym model server directly.

Uploaded per session by ``miniswe_in_sandbox_agent``; runs as the task's agent identity with only the Python
standard library (3.8+) plus a vendored pure-Python Jinja2 (``vendor.zip``) so the pinned mini-swe-agent 2.4.6
``mini.yaml`` templates render byte-identically. It re-implements ``DefaultAgent``/``LocalEnvironment``/
``actions_toolcall`` semantics and the wire behaviour of Gym's server-side ``MiniSWEHarness`` (native Responses
tool calls against ``/v1/responses``), and leaves the same records the agent server turns into a Gym response:
``trajectory.json`` (mini-swe-agent-1.1), ``output_items.json``, ``usages.json``, ``result.json``.
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request


BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
        "required": ["command"],
    },
    "strict": False,
}
CONTEXT_OVERFLOW_MARKERS = ("context_length_exceeded", "context length", "maximum model length")
RETRY_STATUSES = (408, 425, 429, 500, 502, 503, 504)
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class InterruptAgentFlow(Exception):
    def __init__(self, *messages):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    pass


class LimitsExceeded(InterruptAgentFlow):
    pass


class FormatError(InterruptAgentFlow):
    pass


class ModelServerError(RuntimeError):
    pass


def now():
    return time.time()


def load_jinja(vendor_zip):
    if vendor_zip and vendor_zip not in sys.path:
        sys.path.insert(0, vendor_zip)
    from jinja2 import StrictUndefined, Template  # noqa: E402

    return Template, StrictUndefined


def responses_input(messages):
    """Replay native Responses items and associate observations with their calls (Gym harness parity)."""
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


def is_context_overflow(status, detail):
    lowered = detail.lower()
    return status == 400 and (
        any(marker in lowered for marker in CONTEXT_OVERFLOW_MARKERS)
        or ("max_tokens" in detail and "too large" in lowered)
    )


class ContextOverflow(Exception):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail)


class Runner:
    def __init__(self, config):
        self.cfg = config
        self.Template, self.StrictUndefined = load_jinja(config.get("vendor_zip"))
        self.messages = []
        self.output_items = []
        self.responses = []
        self.usages = []
        self.n_calls = 0
        self.steps = 0
        self.n_consecutive_format_errors = 0
        self.started = now()
        self.output_dir = config["output_dir"]
        self.templates = config["templates"]
        self.env = dict(os.environ)
        self.env.update({k: str(v) for k, v in (config.get("env") or {}).items()})
        self.workdir = config.get("workdir") or os.getcwd()
        self.uname = platform.uname()._asdict()
        self.http_errors = []
        os.makedirs(self.output_dir, exist_ok=True)

    # --- templates -------------------------------------------------------------------------------------------------

    def template_vars(self, **extra):
        values = {
            "system_template": self.templates["system_template"],
            "instance_template": self.templates["instance_template"],
            "step_limit": self.cfg.get("step_limit", 0),
            "cost_limit": 0,
            "wall_time_limit_seconds": int(self.cfg.get("budget_sec") or 0),
            "max_consecutive_format_errors": self.cfg.get("max_consecutive_format_errors", 3),
            "output_path": os.path.join(self.output_dir, "trajectory.json"),
        }
        values.update(self.uname)
        values.update(
            {
                "n_model_calls": self.n_calls,
                "model_cost": 0.0,
                "elapsed_seconds": int(now() - self.started),
                "task": self.cfg["task"],
            }
        )
        values.update(extra)
        return values

    def render(self, template, **variables):
        return self.Template(template, undefined=self.StrictUndefined).render(**variables)

    # --- model ---------------------------------------------------------------------------------------------------------

    def remaining(self):
        budget = self.cfg.get("budget_sec")
        if not budget:
            return None
        return max(0.0, budget - (now() - self.started))

    def post_responses(self, body):
        url = self.cfg["model_url"].rstrip("/") + "/responses"
        headers = {"Content-Type": "application/json"}
        headers.update(self.cfg.get("headers") or {})
        data = json.dumps(body).encode("utf-8")
        retries = int(self.cfg.get("http_retries", 3))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        attempt = 0
        while True:
            remaining = self.remaining()
            timeout = float(self.cfg.get("http_timeout_sec") or 3600)
            if remaining is not None:
                timeout = max(5.0, min(timeout, remaining))
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with opener.open(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if is_context_overflow(exc.code, detail):
                    raise ContextOverflow(detail)
                self.http_errors.append({"status": exc.code, "detail": detail[:500], "attempt": attempt})
                if exc.code in RETRY_STATUSES and attempt < retries:
                    time.sleep(min(60, 2**attempt * 5))
                    attempt += 1
                    continue
                raise ModelServerError("model server HTTP %s: %s" % (exc.code, detail[:2000]))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self.http_errors.append({"error": "%s: %s" % (type(exc).__name__, exc), "attempt": attempt})
                if attempt < retries and (self.remaining() is None or self.remaining() > 10):
                    time.sleep(min(60, 2**attempt * 5))
                    attempt += 1
                    continue
                raise ModelServerError("model server unreachable: %s: %s" % (type(exc).__name__, exc))

    def parse_actions(self, calls, finish_reason):
        template = self.templates["format_error_template"]
        if not calls:
            raise FormatError(
                {
                    "role": "user",
                    "content": self.render(
                        template,
                        error="No tool calls found in the response. Every response MUST include at least one tool call.",
                        actions=[],
                        has_tool_calls=False,
                        finish_reason=finish_reason,
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions = []
        for call in calls:
            error = ""
            args = {}
            try:
                args = json.loads(call.get("arguments") or "")
            except Exception as exc:  # noqa: BLE001 - mirrors mini-swe-agent
                error = "Error parsing tool call arguments: %s." % exc
            if call.get("name") != "bash":
                error += "Unknown tool '%s'." % call.get("name")
            if not isinstance(args, dict) or "command" not in args:
                error += "Missing 'command' argument in bash tool call."
            if error:
                raise FormatError(
                    {
                        "role": "user",
                        "content": self.render(
                            template, actions=[], error=error.strip(), has_tool_calls=True, finish_reason=finish_reason
                        ),
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )
            actions.append({"command": args["command"], "tool_call_id": call["call_id"]})
        return actions

    def query(self):
        step_limit = int(self.cfg.get("step_limit") or 0)
        if 0 < step_limit <= self.n_calls:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise LimitsExceeded(
                {"role": "exit", "content": "TimeExceeded", "extra": {"exit_status": "TimeExceeded", "submission": ""}}
            )
        self.n_calls += 1
        body = dict(self.cfg.get("request_params") or {})
        body["input"] = responses_input(self.messages)
        body["tools"] = [BASH_TOOL]
        try:
            response = self.post_responses(body)
        except ContextOverflow as exc:
            # An overfull transcript cannot be repaired by format-error retries; end normally so the caller grades.
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": exc.detail,
                    "extra": {"exit_status": "ContextWindowExceeded", "submission": ""},
                }
            )
        self.responses.append(response)
        output = response.get("output") or []
        self.output_items.extend(output)
        self.usages.append(response.get("usage"))
        content = "\n".join(
            part.get("text", "")
            for item in output
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        )
        calls = [item for item in output if item.get("type") == "function_call"]
        incomplete = response.get("incomplete_details") or {}
        finish_reason = "length" if incomplete.get("reason") == "max_output_tokens" else "stop"
        actions = self.parse_actions(calls, finish_reason)
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call["call_id"],
                    "type": "function",
                    "function": {"name": call.get("name"), "arguments": call.get("arguments")},
                }
                for call in calls
            ],
            "extra": {"actions": actions, "response_output": output},
        }

    # --- environment ---------------------------------------------------------------------------------------------------

    def register_pgid(self, pid):
        pids = self.cfg.get("pids_file")
        if not pids:
            return
        try:
            with open(pids, "a") as handle:
                handle.write("%d\n" % pid)
        except OSError:
            pass

    def execute(self, action):
        step_timeout = int(self.cfg.get("step_timeout_sec") or 30)
        remaining = self.remaining()
        timeout = step_timeout if remaining is None else max(1, int(min(step_timeout, remaining)))
        process = subprocess.Popen(
            ["bash", "-c", action["command"]],
            cwd=self.workdir,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.register_pgid(process.pid)
        exception_info = None
        try:
            stdout, _ = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            stdout, _ = process.communicate()
            returncode = -1
            exception_info = "Command timed out after %d seconds." % timeout
        self.steps += 1
        output = (stdout or b"").decode("utf-8", errors="replace")
        return {"output": output, "returncode": returncode, "exception_info": exception_info}

    def observation_message(self, action, outcome):
        content = self.render(self.templates["observation_template"], output=outcome, **self.template_vars())
        message = {
            "role": "tool",
            "tool_call_id": action["tool_call_id"],
            "content": content,
            "extra": {
                "raw_output": outcome.get("output", ""),
                "returncode": outcome.get("returncode"),
                "timestamp": now(),
                "exception_info": outcome.get("exception_info"),
            },
        }
        # The Gym harness records every executed action's output item, even when a later action submits.
        self.output_items.append(
            {"type": "function_call_output", "call_id": action["tool_call_id"], "output": content}
        )
        return message

    @staticmethod
    def check_finished(outcome):
        lines = outcome.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_MARKER and outcome["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def execute_actions(self, message):
        observations = []
        for action in message.get("extra", {}).get("actions", []):
            outcome = self.execute(action)
            observations.append(self.observation_message(action, outcome))
            self.check_finished(outcome)
        self.messages.extend(observations)
        return observations

    # --- loop ------------------------------------------------------------------------------------------------------------

    def run(self):
        variables = self.template_vars()
        self.messages = [
            {"role": "system", "content": self.render(self.templates["system_template"], **variables)},
            {"role": "user", "content": self.render(self.templates["instance_template"], **variables)},
        ]
        max_format_errors = int(self.cfg.get("max_consecutive_format_errors", 3))
        while True:
            try:
                message = self.query()
                self.messages.append(message)
                self.execute_actions(message)
                self.n_consecutive_format_errors = 0
            except FormatError as exc:
                self.n_consecutive_format_errors += 1
                if 0 < max_format_errors <= self.n_consecutive_format_errors:
                    status = "RepeatedFormatError"
                    recent = self.responses[-self.n_consecutive_format_errors :]
                    if recent and all(
                        (r.get("incomplete_details") or {}).get("reason") == "max_output_tokens" for r in recent
                    ):
                        status = "OutputTokenLimitExceeded"
                    self.messages.extend(exc.messages)
                    self.messages.append(
                        {"role": "exit", "content": status, "extra": {"exit_status": status, "submission": ""}}
                    )
                else:
                    self.messages.extend(exc.messages)
            except InterruptAgentFlow as exc:
                self.messages.extend(exc.messages)
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for a non-zero exit
                self.messages.append(
                    {
                        "role": "exit",
                        "content": str(exc),
                        "extra": {
                            "exit_status": type(exc).__name__,
                            "submission": "",
                            "exception_str": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    }
                )
                self.save()
                raise
            finally:
                self.save()
            if self.messages and self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    # --- records -----------------------------------------------------------------------------------------------------

    def serialize(self):
        last = self.messages[-1] if self.messages else {}
        extra = last.get("extra", {})
        return {
            "info": {
                "model_stats": {"instance_cost": 0.0, "api_calls": self.n_calls},
                "config": {
                    "agent": {
                        "system_template": self.templates["system_template"],
                        "instance_template": self.templates["instance_template"],
                        "step_limit": self.cfg.get("step_limit", 0),
                        "cost_limit": 0,
                        "wall_time_limit_seconds": int(self.cfg.get("budget_sec") or 0),
                        "max_consecutive_format_errors": self.cfg.get("max_consecutive_format_errors", 3),
                        "output_path": os.path.join(self.output_dir, "trajectory.json"),
                    },
                    "agent_type": "nemo_gym.miniswe_in_sandbox_agent.Runner",
                    "environment": {
                        "cwd": self.workdir,
                        "env": self.cfg.get("env") or {},
                        "timeout": int(self.cfg.get("step_timeout_sec") or 30),
                    },
                    "environment_type": "nemo_gym.miniswe_in_sandbox_agent.InSandboxBash",
                },
                "mini_version": self.cfg.get("mini_version", "2.4.6"),
                "model_transport": "nemo_gym_responses",
                "environment_type": "gym_sandbox_in_process",
                "exit_status": extra.get("exit_status", ""),
                "submission": extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }

    def result(self):
        last = self.messages[-1].get("extra", {}) if self.messages else {}
        return {
            "session_id": self.cfg.get("session_id"),
            "exit_status": last.get("exit_status", ""),
            "submission": last.get("submission", ""),
            "n_calls": self.n_calls,
            "steps": self.steps,
            "elapsed_sec": now() - self.started,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "cwd": self.workdir,
            "python": sys.version.split()[0],
            "http_errors": self.http_errors[-20:],
            "finished": bool(self.messages) and self.messages[-1].get("role") == "exit",
        }

    def write_json(self, name, value):
        path = os.path.join(self.output_dir, name)
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(value, handle, indent=2)
        os.replace(temporary, path)

    def save(self):
        self.write_json("trajectory.json", self.serialize())
        self.write_json("output_items.json", self.output_items)
        self.write_json("usages.json", self.usages)
        self.write_json("result.json", self.result())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    runner = Runner(config)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
