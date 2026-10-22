# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenHands SDK harness for resources-server-seeded sandboxes (e.g. nl2repobench).

Structurally a sibling of opencode_sandboxed_agent: the resources server seeds a sandbox, this
agent installs the OpenHands SDK into it and runs `runner.py` there against the policy model
server, then hands the response to the resources server's /verify. Nothing here modifies the
opencode agent; it is a separate agent type.
"""

import asyncio
import json
import sys
from pathlib import Path
from shlex import quote
from time import monotonic, time
from traceback import format_exc
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)


RUNNER_PATH = Path(__file__).parent / "runner.py"
RUNNER_V2_PATH = Path(__file__).parent / "runner_v2.py"


class OpenHandsSDKSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # Sandbox (the resources server seeds it; we only reconnect to it by handle).
    sandbox_provider: str = "sandbox"
    sandbox_timeout: float = 21600

    # SDK install. litellm is pinned because an unpinned litellm resolves to a release that is
    # incompatible with the mcp version openhands-sdk 1.50 pulls in (ImportError: CacheableResult).
    install_packages: List[str] = Field(
        default_factory=lambda: ["openhands-sdk==1.50.0", "openhands-tools==1.50.0", "litellm==1.93.0"]
    )
    # Optional pre-staged wheelhouse (e.g. under /mnt/s3-data) to install from without PyPI.
    install_find_links: Optional[str] = None
    venv_dir: str = "/tmp/nemo-gym-openhands-venv"
    install_timeout_s: float = 900
    # For sandboxes with no internet and no python>=3.12/tmux (e.g. agent_network_mode=
    # no_internet_preinstalled, python:3.11-slim task images): a pre-built tarball (standalone
    # python 3.12 + SDK site-packages + static tmux; see openhands_bundle/) uploaded and extracted
    # in the sandbox instead of pip-installing. When set, install_packages/install_find_links are unused.
    bundle_path: Optional[str] = None
    # Run the SDK as a detached background process and poll for its exit code, instead of one long
    # sandbox.exec(). A single multi-hour exec was observed to fail ("Get command logs failed"), which
    # then stopped the sandbox mid-run. With this on, the runner (runner_v2.py) also writes partial
    # results periodically, so a killed run still leaves a trajectory.
    background_exec: bool = True  # detached via `setsid -f` (see _detached_launch_command); the earlier nohup-based launch was killed with the exec
    poll_interval_s: float = 30
    max_consecutive_poll_errors: int = 40
    bundle_python_relpath: str = "py/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12"

    # Agent run.
    max_iterations: int = 500
    model_call_timeout_s: Optional[int] = 14400
    num_retries: int = 5
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    litellm_extra_body: Dict[str, Any] = Field(default_factory=dict)
    # The user prompt the SFT data was collected with. Set to null to use the benchmark's own.
    user_prompt: Optional[str] = (
        "According to the start.md in the workspace, implement the entire project as per the "
        "requirements specified in the document, ensuring that the final product can be directly "
        "run in the current directory. The running requirements should comply with the "
        "<API Usage Guide> section of the document. Please complete this task step by step."
    )

    debug: bool = False


class OpenHandsSDKSandboxedAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenHandsSDKSandboxedAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class OpenHandsSDKSandboxedAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    openhands_results_fpath: str
    openhands_run_stdout: str
    openhands_run_stderr: str
    openhands_finished: bool
    openhands_result_found: bool


def _result_to_output_items(messages: List[Dict[str, Any]]) -> List[NeMoGymResponseOutputItem]:
    """Convert the runner's chat-format messages (system + user stripped by the caller)."""
    items: List[NeMoGymResponseOutputItem] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if reasoning:
                items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}", summary=[NeMoGymSummary(type="summary_text", text=reasoning)]
                    )
                )
            if content:
                items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        role="assistant",
                        status="completed",
                        type="message",
                        content=[NeMoGymResponseOutputText(type="output_text", text=content, annotations=[])],
                    )
                )
            for call in message.get("tool_calls") or []:
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=call["function"]["arguments"],
                        call_id=call["id"],
                        name=call["function"]["name"],
                    )
                )
        elif role == "tool":
            items.append(NeMoGymFunctionCallOutput(call_id=message["tool_call_id"], output=content))
        # role == "user" mid-run messages (e.g. SDK nudges) have no Responses output equivalent.
    return items


class OpenHandsSDKSandboxedAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: OpenHandsSDKSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _connect_sandbox(self, sandbox_id: str, workdir: Optional[str]) -> AsyncSandbox:
        from nemo_gym.global_config import get_global_config_dict

        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        return await AsyncSandbox.connect({"sandbox_id": sandbox_id, "workdir": workdir}, provider=provider)

    @staticmethod
    def _detached_launch_command(run_cmd: str, run_log: str, exit_file: str, pid_file: str) -> str:
        """Start `run_cmd` so it survives after this exec returns (OpenSandbox kills the exec's process tree).

        Verified in a real sandbox: `setsid -f` (and a python double fork) survive; `nohup setsid bash -c '...' &`
        and plain `&`/disown are killed when the launching exec returns. The inner shell records its own pid
        (`$$`) because `setsid -f` does not report one, and writes the exit code last.
        """
        inner = f"echo $$ > {pid_file}; {run_cmd} > {run_log} 2>&1; echo $? > {exit_file}"
        return f"setsid -f bash -c {quote(inner)} >/dev/null 2>&1 < /dev/null; echo launched"

    def _install_command(self) -> str:
        venv = quote(self.config.venv_dir)
        links = f"--find-links {quote(self.config.install_find_links)} --no-index " if self.config.install_find_links else ""
        packages = " ".join(quote(p) for p in self.config.install_packages)
        # Needs python>=3.12 and tmux (the SDK's terminal tool runs commands through tmux).
        return (
            "set -e; "
            'PY=""; for c in python3.13 python3.12 python3; do '
            'if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(sys.version_info < (3,12))"; '
            'then PY="$c"; break; fi; done; '
            '[ -n "$PY" ] || { echo "no python>=3.12 in sandbox" >&2; exit 41; }; '
            "command -v tmux >/dev/null 2>&1 || { echo 'tmux missing in sandbox' >&2; exit 42; }; "
            # block_target_v2 wraps python* (real binary kept at <path>.nl2repobench-real); the harness's own venv
            # creation must bypass that wrapper, which denies `-m venv`.
            'P=$(command -v "$PY"); R="$(readlink -f "$(dirname "$P")")/$(basename "$P").nl2repobench-real"; '
            '[ -e "$R" ] && PY="$R"; '
            f'"$PY" -m venv {venv}; '
            f"{venv}/bin/pip install --quiet --disable-pip-version-check {links}{packages}; "
            f"{venv}/bin/python -c 'import openhands.sdk, openhands.tools'; "
            "echo OPENHANDS_INSTALLED"
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_key = request.cookies["sandbox_id"]
        sandbox = self._sandbox_id_to_sandbox[session_key]
        cfg = self.config

        query = None
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]
        assert query, body.input
        prompt = cfg.user_prompt or query

        run_id = uuid4().hex
        results_dir: Path = Path(__file__).parent / "results" / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        local_request = results_dir / "request.json"
        local_result = results_dir / "result.json"
        local_result.unlink(missing_ok=True)

        runner_params = {
            "model": body.model or "dummy_model",
            "base_url": self.resolve_model_base_url(cfg.model_server.name, getattr(request.state, "_ng_rollout_id", None)),
            "workdir": "/workspace",
            "prompt": prompt,
            "max_iterations": cfg.max_iterations,
            "model_call_timeout_s": cfg.model_call_timeout_s,
            "num_retries": cfg.num_retries,
            "temperature": body.temperature if body.temperature is not None else cfg.temperature,
            "top_p": body.top_p if body.top_p is not None else cfg.top_p,
            "max_output_tokens": body.max_output_tokens if body.max_output_tokens is not None else cfg.max_output_tokens,
            "reasoning_effort": cfg.reasoning_effort,
            "litellm_extra_body": cfg.litellm_extra_body,
        }
        local_request.write_text(json.dumps(runner_params))

        remote_dir = f"/tmp/nemo-gym-openhands-run-{run_id}"
        remote_request, remote_result, remote_runner = (f"{remote_dir}/{n}" for n in ("request.json", "result.json", "runner.py"))

        stdout = stderr = ""
        run_error_type: Optional[str] = None
        install_s = run_s = 0.0
        finished_marker = False
        poll_error: Optional[str] = None
        try:
            started = monotonic()
            await sandbox.exec(f"mkdir -p {remote_dir}", cwd="/", timeout_s=60)
            if cfg.bundle_path:
                remote_bundle = f"{remote_dir}/bundle.tar.gz"
                await sandbox.upload(Path(cfg.bundle_path), remote_bundle)
                install = await sandbox.exec(
                    f"set -e; mkdir -p {quote(cfg.venv_dir)}; tar -xzf {remote_bundle} -C {quote(cfg.venv_dir)}; "
                    f"rm -f {remote_bundle}; {quote(cfg.venv_dir)}/{cfg.bundle_python_relpath} "
                    "-c 'import openhands.sdk, openhands.tools'; echo OPENHANDS_INSTALLED",
                    cwd="/",
                    timeout_s=cfg.install_timeout_s,
                )
            else:
                install = await sandbox.exec(self._install_command(), cwd="/", timeout_s=cfg.install_timeout_s)
            install_s = monotonic() - started
            stdout += f"[install rc={install.return_code}]\n{install.stdout or ''}"
            stderr += install.stderr or ""
            if install.return_code != 0 or "OPENHANDS_INSTALLED" not in (install.stdout or ""):
                raise RuntimeError(f"OpenHands SDK install failed (rc={install.return_code}): {(install.stderr or '')[-1500:]}")

            await sandbox.upload(local_request, remote_request)
            await sandbox.upload(RUNNER_V2_PATH if cfg.background_exec else RUNNER_PATH, remote_runner)
            started = monotonic()
            # cwd=/workspace so the SDK's terminal starts where start.md was written.
            python_bin = f"{cfg.venv_dir}/{cfg.bundle_python_relpath}" if cfg.bundle_path else f"{cfg.venv_dir}/bin/python"
            path_prefix = f"PATH={quote(cfg.venv_dir)}/bin:$PATH " if cfg.bundle_path else ""
            run_cmd = f"{path_prefix}OPENHANDS_SUPPRESS_BANNER=1 {quote(python_bin)} -I {remote_runner} {remote_request} {remote_result}"
            if not cfg.background_exec:
                result = await sandbox.exec(run_cmd, cwd="/workspace", timeout_s=cfg.sandbox_timeout)
                stdout += result.stdout or ""
                stderr += result.stderr or ""
                finished_marker = "OpenHands run finished" in (result.stdout or "")
            else:
                run_log, exit_file, pid_file = f"{remote_dir}/run.log", f"{remote_dir}/exit_code", f"{remote_dir}/pid"
                launch = await sandbox.exec(
                    self._detached_launch_command(run_cmd, run_log, exit_file, pid_file),
                    cwd="/workspace",
                    timeout_s=120,
                )
                if "launched" not in (launch.stdout or ""):
                    raise RuntimeError(f"failed to launch OpenHands runner: {launch}")
                deadline = monotonic() + cfg.sandbox_timeout
                consecutive_errors = 0
                while True:
                    try:
                        poll = await sandbox.exec(
                            f"if [ -f {exit_file} ]; then cat {exit_file}; "
                            f"elif [ -d /proc/$(cat {pid_file}) ]; then echo RUNNING; else echo DEAD; fi",
                            cwd="/",
                            timeout_s=120,
                        )
                        consecutive_errors = 0
                        if "DEAD" in (poll.stdout or ""):
                            poll_error = "runner process died without writing an exit code"
                            break
                        if "RUNNING" not in (poll.stdout or ""):
                            break
                    except Exception as exc:
                        consecutive_errors += 1
                        text = f"{type(exc).__name__}: {exc}"
                        # A stopped/OOM-killed sandbox will never answer again - don't wait it out.
                        if "SANDBOX::ENDED" in text or "OOMKilled" in text or consecutive_errors >= cfg.max_consecutive_poll_errors:
                            poll_error = text
                            break
                    if monotonic() > deadline:
                        poll_error = f"sandbox_timeout {cfg.sandbox_timeout}s exceeded"
                        break
                    await asyncio.sleep(cfg.poll_interval_s)
                try:
                    local_log = results_dir / "run.log"
                    await sandbox.download(run_log, local_log)
                    log_text = local_log.read_text(errors="replace")
                    stdout += log_text
                    finished_marker = "OpenHands run finished" in log_text
                except Exception:
                    print("Failed to download OpenHands run.log", format_exc(), file=sys.stderr)
            run_s = monotonic() - started
        except Exception as exc:
            run_error_type = type(exc).__name__
            stderr += f"\n[agent exec error] {run_error_type}: {exc}\n"
            print("OpenHands exec hit error.", format_exc(), file=sys.stderr)

        try:
            await sandbox.download(remote_result, local_result)
        except Exception:
            print("Failed to download OpenHands result", format_exc(), file=sys.stderr)

        sdk_result: Dict[str, Any] = {}
        if local_result.exists():
            sdk_result = json.loads(local_result.read_text().strip() or "{}")

        output: List[NeMoGymResponseOutputItem] = []
        usage = None
        system_prompt = None
        if sdk_result:
            messages = sdk_result.get("messages") or []
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0].get("content")
                if isinstance(system_prompt, list):
                    system_prompt = "\n".join(p.get("text", "") for p in system_prompt)
                messages = messages[1:]
            if messages and messages[0].get("role") == "user":
                messages = messages[1:]
            output = _result_to_output_items(messages)
            metrics = sdk_result.get("metrics") or {}
            usage = NeMoGymResponseUsage(
                input_tokens=metrics.get("prompt_tokens", 0),
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=metrics.get("cache_read_tokens", 0)),
                output_tokens=metrics.get("completion_tokens", 0),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=metrics.get("reasoning_tokens", 0)),
                total_tokens=metrics.get("prompt_tokens", 0) + metrics.get("completion_tokens", 0),
            )
            stdout += f"\n[openhands] execution_status={sdk_result.get('execution_status')} error={sdk_result.get('error')} " \
                f"llm_calls={metrics.get('n_llm_calls')} install_s={install_s:.0f} run_s={run_s:.0f}\n"
            if sdk_result.get("traceback"):
                stderr += sdk_result["traceback"]

        self._sandbox_id_to_run_result[session_key] = {
            "openhands_results_fpath": str(local_result) if sdk_result else "",
            "openhands_run_stdout": stdout,
            "openhands_run_stderr": stderr,
            "openhands_result_found": bool(sdk_result),
            "openhands_finished": bool(sdk_result.get("finished")) or finished_marker,
            "openhands_execution_status": sdk_result.get("execution_status"),
            "openhands_error_codes": [e.get("code") for e in sdk_result.get("errors") or []],
            "openhands_partial": bool(sdk_result.get("partial")) if sdk_result else None,
            "openhands_poll_error": poll_error,
            "_system_prompt": system_prompt,
        }

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

    async def run(
        self, request: Request, body: OpenHandsSDKSandboxedAgentRunRequest
    ) -> OpenHandsSDKSandboxedAgentVerifyResponse:
        cookies = request.cookies
        session_key = request.session[SESSION_ID_KEY]
        request.state._ng_rollout_id = self.rollout_id_from_run(body)

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies

        seed_session_result = await seed_session_response.json()
        sandbox = await self._connect_sandbox(seed_session_result["sandbox_handle"], seed_session_result.get("workdir"))
        self._sandbox_id_to_sandbox[session_key] = sandbox

        cookies["sandbox_id"] = session_key
        request._cookies = cookies
        response = await self.responses(request, body.responses_create_params)

        verify_request = OpenHandsSDKSandboxedAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": response}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)

        try:
            await sandbox.stop()
        except Exception:
            print("Failed to stop sandbox", format_exc(), file=sys.stderr)
        self._sandbox_id_to_sandbox.pop(session_key, None)

        response_dict = await get_response_json(verify_response)
        run_result = self._sandbox_id_to_run_result.pop(session_key)
        system_prompt = run_result.pop("_system_prompt", None)
        response_dict |= run_result
        response_dict.pop("verifier_sandbox_observation", None)
        if system_prompt:
            response_dict["responses_create_params"]["input"].insert(0, {"content": system_prompt, "role": "system"})
        # Record the prompt the agent actually saw, not the benchmark's shorter one.
        if self.config.user_prompt:
            for item in response_dict["responses_create_params"]["input"]:
                if item.get("role") == "user":
                    item["content"] = self.config.user_prompt
        return OpenHandsSDKSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    OpenHandsSDKSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenHandsSDKSandboxedAgent.run_webserver()  # noqa: F401
