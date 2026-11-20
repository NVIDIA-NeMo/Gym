# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prime SDK child sessions borrowing a Resources Server's task sandbox."""

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, ClassVar
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import Field, model_validator

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    SandboxCommand,
    resolve_agent_timeout,
)
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import download_bytes, read_text, upload_text
from nemo_gym.trajectory import ObservationGap, SandboxObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.prime_agent.app import (
    PrimeAgent,
    PrimeAgentConfig,
    parse_prime_agent_events,
)
from responses_api_agents.prime_agent.child_observations import child_observations
from responses_api_agents.prime_agent.sandbox import PrimeSession


HERE = Path(__file__).parent


def _journal_records(data: bytes) -> list[dict]:
    """Recover complete journal records after the supervisor has stopped the writer."""
    end = data.rfind(b"\n")
    if end == -1:
        return []
    records = [json.loads(line) for line in data[:end].split(b"\n")]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Prime journal records must be JSON objects")
    return records


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
    capture_directory: str | None = None
    artifact_directory: str = "outputs/prime_agent/child_sessions"

    @model_validator(mode="before")
    @classmethod
    def no_custom_kernel_installer(cls, values):
        if isinstance(values, dict) and "kernel_packages" in values:
            raise ValueError("Configure pinned task_python_packages on PartnerIF Resources instead of kernel_packages")
        return values


class PrimeChildAgent(PrimeAgent):
    """Use Prime's native IPython and agent-message skill, with a recording parent."""

    config: PrimeChildConfig
    calls_model_from_sandbox: ClassVar[bool] = True

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Leave SDK failures to the pairing's verifier, which checks retained parent-delivery evidence."""
        return AgentOutcome(failure_kind="unknown")

    def model_post_init(self, context: object) -> None:
        # The sandbox owns the runtime; do not install an unused host CLI.
        super().model_post_init(context)
        if self.config.prime_agent_version != "0.7.0":
            raise ValueError("Prime child sessions require version 0.7.0")
        if self.config.num_workers != 1:
            raise ValueError("Prime child sessions require num_workers=1")
        if self.config.capture_directory and not self.config.capture_directory.startswith("/"):
            raise ValueError("capture_directory must be absolute")

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> PrimeSession:
        # Resources owns the repository sandbox; the existing Prime lifecycle connects, installs and releases it.
        if body.sandbox_access is None:
            raise AgentSeedRejectedError("Prime child sessions require a task sandbox")
        return await super()._seed_agent_session_state(body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Close through Prime's shared lifecycle and retain its exported observations."""
        return await super()._close_agent_session_state(state)

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise AgentSessionConflictError("Prime child mode requires a seeded agent session")
        return await self._session_response(request, session_id, body)

    async def _read_capture(self, state: PrimeSession) -> dict:
        session = state.session
        cleanup = session.cleanup
        if not (cleanup and cleanup["cleanup_confirmed"] and cleanup["timed_out"]):
            return json.loads(await read_text(session.sandbox, path=f"{state.agent_dir}/capture.json"))

        # The supervisor may stop the process before the SDK writes its final capture.
        events = _journal_records(await download_bytes(session.sandbox, path=f"{state.agent_dir}/events.jsonl"))
        deliveries = _journal_records(
            await download_bytes(session.sandbox, path=f"{state.agent_dir}/parent-messages.jsonl")
        )
        value = {
            "version": 1,
            "termination": "stopped" if session.expired else "timeout",
            "assistantTurns": sum(
                event.get("type") == "message_end" and (event.get("message") or {}).get("role") == "assistant"
                for event in events
            ),
            "events": events,
            "deliveries": deliveries,
        }
        if self.config.capture_directory:
            await upload_text(
                session.sandbox, path=f"{self.config.capture_directory}/capture.json", text=json.dumps(value)
            )
        return value

    async def _activate_session(
        self, state: PrimeSession, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        prompt, system = self._session_conversation(body)
        directory = state.session.session_dir
        # Only the Gym provider: the model-controlled sandbox must not receive host providers' keys.
        nemo = self._build_models_config(state.request.episode_id.capture_key)["providers"]["nemo"]
        config = {
            "cwd": state.session.workdir,
            "runtime": state.runtime,
            "state": state.agent_dir,
            "models": {"providers": {"nemo": nemo}},
            "model": self.config.model,
            "prompt": prompt,
            "system": system or "",
            "maxTurns": self.config.max_turns,
            "maxTokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "captureDirectory": self.config.capture_directory,
        }

        runtime_validation = {}

        async def stage() -> SandboxCommand:
            nonlocal runtime_validation
            await state.session.sandbox.upload(HERE / "child_runtime.mjs", f"{directory}/child_runtime.mjs")
            await upload_text(state.session.sandbox, path=f"{directory}/input.json", text=json.dumps(config))
            access = state.request.sandbox_access
            environment = {
                key: value
                for key, value in (access.env if access else {}).items()
                if not key.startswith(("PRIME_AGENT_", "PI_", "OPENAI_"))
            }
            package = f"{state.runtime}/lib/node_modules/prime-agent"
            environment |= {
                "PRIME_AGENT_CODING_AGENT_DIR": state.agent_dir,
                "PRIME_AGENT_KERNEL_PYTHON": f"{state.runtime}/kernel/bin/python3",
                "PI_SKIP_VERSION_CHECK": "1",
                # The verified release contains this dependency-free skill; no separate pip build is needed.
                "PYTHONPATH": ":".join(
                    filter(
                        None,
                        [
                            f"{package}/dist/skills/agent-message/src",
                            environment.get("PYTHONPATH"),
                            state.session.workdir,
                        ],
                    )
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            checked = await state.session.sandbox.exec(
                shlex.join(
                    [f"{state.runtime}/bin/node", f"{directory}/child_runtime.mjs", "--check-runtime", state.runtime]
                ),
                cwd="/",
                env=environment,
                timeout_s=min(60, self.config.sandbox_install_timeout_seconds),
            )
            if checked.return_code != 0 or checked.error_type:
                raise RuntimeError(
                    f"Prime SDK capability check failed: {(checked.stderr or checked.stdout or '')[-4000:]}"
                )
            runtime_validation = json.loads((checked.stdout or "").strip().splitlines()[-1])
            if not isinstance(runtime_validation, dict):
                raise RuntimeError("Prime SDK capability check returned invalid evidence")
            return await state.stage(
                command=[f"{state.runtime}/bin/node", f"{directory}/child_runtime.mjs", f"{directory}/input.json"],
                env=environment,
                models=json.dumps(config["models"]),
            )

        async def collect() -> dict:
            if state.session.expired_before_launch:
                state.observations = self._session_observations(
                    state, status="incomplete", error_type="stopped", output=[]
                )
                return {"version": 1, "termination": "stopped", "events": [], "deliveries": [], "assistantTurns": 0}
            value = await self._read_capture(state)
            if (
                not isinstance(value, dict)
                or value.get("version") != 1
                or value.get("termination") not in {"completed", "turn_limit", "error", "timeout", "stopped"}
                or not isinstance(value.get("events"), list)
                or not isinstance(value.get("deliveries"), list)
                or not all(isinstance(item, dict) for item in value["events"] + value["deliveries"])
                or type(value.get("assistantTurns")) is not int
                or value["assistantTurns"] < 0
            ):
                raise RuntimeError("Prime child did not produce a valid SDK capture")
            destination = Path(self.config.artifact_directory).expanduser() / uuid4().hex
            destination.mkdir(parents=True, mode=0o700)
            (destination / "capture.json").write_text(json.dumps(value))
            (destination / "cleanup.json").write_text(json.dumps(state.session.cleanup))
            (destination / "runtime-check.json").write_text(json.dumps(runtime_validation))
            effective, gaps = {}, []
            filenames = ["effective-session.json", "model-requests.jsonl"]
            if value["termination"] == "error":
                filenames.append("errors.jsonl")
            for filename in filenames:
                try:
                    text = await read_text(state.session.sandbox, path=f"{state.agent_dir}/{filename}")
                    (destination / filename).write_text(text)
                    if filename == "effective-session.json":
                        effective = json.loads(text)
                        if not isinstance(effective, dict) or not isinstance(effective.get("systemPrompt"), str):
                            raise ValueError("Effective system prompt unavailable")
                except Exception:
                    gaps.append(
                        ObservationGap(
                            code=f"{filename.replace('.', '_')}_unavailable",
                            invocation_id=state.request.agent_session_id,
                        )
                    )
                    if filename == "effective-session.json":
                        effective = {}
            state.observations = child_observations(
                value,
                system_prompt=effective.get("systemPrompt", ""),
                model_ref=self.config.model_server,
                invocation_id=state.request.agent_session_id,
            )
            state.observations.gaps.extend(gaps)
            cleanup = state.session.cleanup
            if isinstance(cleanup, dict) and isinstance(state.sandbox_id, str):
                outcome = (
                    "timeout"
                    if cleanup["timed_out"]
                    else "failed"
                    if cleanup["error"] or cleanup["return_code"] not in (0, None)
                    else "cancelled"
                    if cleanup["return_code"] is None
                    else "completed"
                )
                state.observations.records.append(
                    SandboxObservation(
                        role="agent",
                        provider=state.provider_name,
                        sandbox_id=state.sandbox_id,
                        outcome=outcome,
                        exit_code=cleanup["return_code"],
                        error_type=cleanup["error"] or (None if outcome == "completed" else outcome),
                    )
                )
            return value

        timeout = resolve_agent_timeout(
            body.metadata, ceiling=min(self.config.timeout, state.request.agent_timeout_sec or self.config.timeout)
        )
        try:
            async with self.sem:
                captured = await state.session.execute(
                    stage_activation=stage,
                    collect=collect,
                    timeout=timeout,
                    close_timeout=self.config.session_close_timeout_seconds,
                )
        except BaseException as error:
            state.observations = self._session_observations(
                state, status="failed", error_type=type(error).__name__, output=[]
            )
            raise
        events = captured["events"]
        assistants = [
            e["message"]
            for e in events
            if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant"
        ]
        failures = [m for m in assistants if m.get("stopReason") == "error" or m.get("errorMessage")]
        if captured["termination"] == "error" and not failures:
            failures.append(
                {
                    "errorMessage": "Prime SDK execution or disposal failed; retained capture contains available evidence"
                }
            )
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
        return (
            self.report_outcome(response, failures[-1])
            if failures
            else self.report_rollout_stop(response, state.session)
        )

    async def run(self, request: Request, body: object) -> None:
        raise HTTPException(422, "Prime child mode runs through an Environment Server")


if __name__ == "__main__":
    PrimeChildAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PrimeChildAgent.run_webserver()
