# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prime SDK child sessions borrowing a Resources Server's task sandbox."""

import asyncio
import hashlib
import json
import logging
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import ClassVar
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import Field

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.utils import read_text, upload_text
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.server_utils import request as http_request
from responses_api_agents.prime_agent.app import (
    PrimeAgent,
    PrimeAgentConfig,
    _extract_instruction,
    parse_prime_agent_events,
)


LOG = logging.getLogger(__name__)
HERE = Path(__file__).parent
# Release assets for https://github.com/PrimeIntellect-ai/prime-agent/releases/tag/v0.7.0.
ARCHIVES = {
    "prime-agent": "88b6578518c72cd51a825bc80f28e0fef9a64c67de4a7d6fd7afd7ca1b34da0b",
    "prime-agent-ai": "7cdbb3e835f48dd103325f7a351ce540b27af4d161aeb9c7b9bdcc12fe7909af",
    "prime-agent-core": "0313373089831d9a2ce06e874fab8b9c05762c0094ff9fc202908cf7db7f99cd",
    "prime-agent-tui": "3225f7f92e87db80fe2c9005d1f7770735ae625c32935ef2283688fc9bd33951",
}


class PrimeChildConfig(PrimeAgentConfig):
    """Evaluation-only child mode; task preparation belongs to the Resources Server."""

    resources_server: None = None
    num_workers: int = 1
    prime_agent_version: str = "0.7.0"
    max_turns: int = Field(default=48, ge=1)
    temperature: float = 0
    context_window: int = 131072
    max_output_tokens: int = 65536
    session_close_timeout_seconds: float = 60
    sandbox_install_timeout_seconds: float = 1800
    kernel_packages: list[str] = Field(default_factory=list)
    capture_directory: str | None = None
    artifact_directory: str = "outputs/prime_agent/child_sessions"


@dataclass
class ChildSession(AgentSessionState):
    session: SandboxSession[dict]
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation: NeMoGymResponseCreateParamsNonStreaming | None = None


class PrimeChildAgent(PrimeAgent):
    """Use Prime's native IPython and agent-message skill, with a recording parent."""

    config: PrimeChildConfig
    calls_model_from_sandbox: ClassVar[bool] = True

    def model_post_init(self, context: object) -> None:
        # The sandbox owns the runtime; do not install an unused host CLI.
        self.sem = asyncio.Semaphore(self.config.concurrency)
        if self.config.prime_agent_version != "0.7.0":
            raise ValueError("Prime child sessions require version 0.7.0")
        if self.config.num_workers != 1:
            raise ValueError("Prime child sessions require num_workers=1")
        if self.config.capture_directory and not self.config.capture_directory.startswith("/"):
            raise ValueError("capture_directory must be absolute")

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ChildSession:
        if body.sandbox_access is None:
            raise AgentSeedRejectedError("Prime child sessions require a task sandbox")
        if any(grant.required for grant in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError(
                "Prime child sessions use only native IPython; required tool grants unsupported"
            )
        if body.sandbox_access.user is not None or body.sandbox_access.env:
            raise AgentSeedRejectedError("Prime child sessions do not support sandbox user or environment overrides")
        if self.config.model_server is None:
            raise AgentSeedRejectedError("Prime child sessions require a Gym Model Server")
        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        directory = f"/tmp/nemo-gym-prime/{uuid4().hex}"
        state = ChildSession(
            request=body,
            runtime=f"{directory}/runtime",
            session=SandboxSession(
                sandbox=sandbox, session_dir=directory, workdir=body.sandbox_access.workdir, harness="Prime child"
            ),
        )
        try:
            result = await sandbox.exec(f"mkdir -p {shlex.quote(state.runtime)}", timeout_s=30)
            if result.return_code != 0 or result.error_type:
                raise RuntimeError("Cannot create Prime runtime directory")
            with tempfile.TemporaryDirectory(prefix="gym-prime-") as temporary:
                for name, expected in ARCHIVES.items():
                    filename = f"{name}-0.7.0.tgz"
                    response = await http_request(
                        "GET", f"https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v0.7.0/{filename}"
                    )
                    try:
                        response.raise_for_status()
                        content = await response.read()
                    finally:
                        response.release()
                    if hashlib.sha256(content).hexdigest() != expected:
                        raise RuntimeError(f"Prime release checksum mismatch: {filename}")
                    local = Path(temporary) / filename
                    local.write_bytes(content)
                    await sandbox.upload(local, f"{state.runtime}/{filename}")
            await sandbox.upload(HERE / "install_child.py", f"{directory}/install_child.py")
            await sandbox.upload(HERE / "child_runtime.mjs", f"{directory}/child_runtime.mjs")
            result = await sandbox.exec(
                f"python3 {shlex.quote(directory + '/install_child.py')} {shlex.quote(state.runtime)} "
                f"{shlex.quote(json.dumps(self.config.kernel_packages))}",
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if result.return_code != 0 or result.error_type:
                raise RuntimeError(f"Prime runtime setup failed: {(result.stderr or result.stdout or '')[-4000:]}")
        except BaseException as error:
            try:
                await state.session.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, ChildSession)
        await state.session.close(timeout=self.config.session_close_timeout_seconds)
        if state.task is not None and not state.task.done():
            state.task.cancel()
            await asyncio.wait_for(
                asyncio.gather(state.task, return_exceptions=True), timeout=self.config.session_close_timeout_seconds
            )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(
                source="prime",
                gaps=[
                    ObservationGap(
                        code="harness_turn_observations_unavailable",
                        detail="Evaluation-only child adapter; raw SDK events retained",
                    ),
                    ObservationGap(
                        code="model_call_ownership_unavailable", detail="Use rollout-prefixed Model Server captures"
                    ),
                ],
            ),
        )

    def validate_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> None:
        """Reject unsupported controls and supplied history instead of silently dropping them."""
        supported = {"input", "instructions", "model"}
        for name, field in type(body).model_fields.items():
            if name in supported:
                continue
            value = getattr(body, name)
            if value != field.get_default(call_default_factory=True):
                raise HTTPException(422, f"Prime child mode does not support request field {name}")
        if body.model not in (None, self.config.model):
            raise HTTPException(422, "Prime child model must match the configured model")
        if isinstance(body.input, str):
            return
        if [getattr(item, "role", None) for item in body.input] not in (["user"], ["system", "user"]):
            raise HTTPException(422, "Prime child mode accepts one user prompt and an optional system prompt")
        for item in body.input:
            if not isinstance(item.content, str) and any(
                (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) != "input_text"
                for part in item.content
            ):
                raise HTTPException(422, "Prime child mode accepts text input only")

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Prime child mode requires a seeded agent session")
        self.validate_input(body)
        state = self._require_agent_session(session_id)
        assert isinstance(state, ChildSession)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Prime activation rollout differs from the seeded episode")
        if state.session.closing:
            raise HTTPException(409, "Prime child session is closing")
        if state.task is None:
            state.activation = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._execute_child(state, body))
        elif body != state.activation:
            raise HTTPException(409, "Prime child sessions support only one activation")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def _execute_child(
        self, state: ChildSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        if isinstance(body.input, str):
            prompt, system = body.input, None
        else:
            prompt, system = _extract_instruction(body.input)
        directory = state.session.session_dir
        config = {
            "cwd": state.session.workdir,
            "runtime": state.runtime,
            "state": f"{directory}/state",
            "models": self._build_models_config(state.request.episode_id.capture_key),
            "model": self.config.model,
            "prompt": prompt,
            "system": "\n\n".join(filter(None, [self.config.system_prompt, body.instructions, system])),
            "maxTurns": self.config.max_turns,
            "maxTokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "captureDirectory": self.config.capture_directory,
        }

        async def stage() -> SandboxCommand:
            await upload_text(state.session.sandbox, path=f"{directory}/input.json", text=json.dumps(config))
            environment = {
                "PRIME_AGENT_CODING_AGENT_DIR": config["state"],
                "PRIME_AGENT_KERNEL_PYTHON": f"{state.runtime}/kernel/bin/python",
                "PYTHONPATH": state.session.workdir,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            return SandboxCommand(
                argv=[
                    "env",
                    *(f"{key}={value}" for key, value in environment.items()),
                    "node",
                    f"{directory}/child_runtime.mjs",
                    f"{directory}/input.json",
                ]
            )

        async def collect() -> dict:
            value = json.loads(await read_text(state.session.sandbox, path=f"{directory}/state/capture.json"))
            if value.get("version") != 1 or value.get("termination") not in {"completed", "turn_limit"}:
                raise RuntimeError("Prime child did not produce a complete SDK capture")
            destination = Path(self.config.artifact_directory).expanduser() / uuid4().hex
            destination.mkdir(parents=True, mode=0o700)
            (destination / "capture.json").write_text(json.dumps(value))
            (destination / "cleanup.json").write_text(json.dumps(state.session.cleanup))
            for filename in ("effective-session.json", "model-requests.jsonl"):
                text = await read_text(state.session.sandbox, path=f"{directory}/state/{filename}")
                (destination / filename).write_text(text)
            return value

        captured = await state.session.execute(
            stage_activation=stage,
            collect=collect,
            timeout=self.config.timeout,
            close_timeout=self.config.session_close_timeout_seconds,
        )
        events = captured["events"]
        assistants = [
            e["message"]
            for e in events
            if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant"
        ]
        failures = [m for m in assistants if m.get("stopReason") == "error" or m.get("errorMessage")]
        # A budget abort is a control event; preserve the preceding model/tool trajectory.
        visible = [
            e
            for e in events
            if not (e.get("type") == "message_end" and e.get("message", {}).get("stopReason") in {"aborted", "error"})
        ]
        output, totals = parse_prime_agent_events("\n".join(json.dumps(e) for e in visible))
        usage = None
        model_messages = [message for message in assistants if message.get("stopReason") != "aborted"]
        if (
            model_messages
            and not failures
            and all(
                isinstance(m.get("usage"), dict) and {"input", "output"} <= m["usage"].keys() for m in model_messages
            )
        ):
            usage = NeMoGymResponseUsage(
                input_tokens=totals["input_tokens"],
                output_tokens=totals["output_tokens"],
                total_tokens=totals["input_tokens"] + totals["output_tokens"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=totals["cached_tokens"]),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=None),
            )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output,
            status="failed" if failures else "completed" if captured["termination"] == "completed" else "incomplete",
            error={"code": "server_error", "message": str(failures[-1].get("errorMessage", "Model request failed"))}
            if failures
            else None,
            usage=usage,
            tools=body.tools,
            tool_choice=body.tool_choice,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                "prime_agent_version": "0.7.0",
                "prime_child_turns": str(captured["assistantTurns"]),
                "prime_child_termination": captured["termination"],
            },
        )
        return self.report_outcome(response, failures[-1]) if failures else response

    async def run(self, request: Request, body: object) -> None:
        raise HTTPException(422, "Prime child mode runs through an Environment Server")


if __name__ == "__main__":
    PrimeChildAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PrimeChildAgent.run_webserver()
