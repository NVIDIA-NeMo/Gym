# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sqlite3
import sys
from pathlib import Path
from shlex import quote
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from openai.types.responses import ResponseInputTextParam
from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
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
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
)
from responses_api_agents.sandboxed_harness_agent.app import (
    HarnessTranscript,
    SandboxedHarnessAgent,
    SandboxedHarnessAgentConfig,
    SandboxedHarnessAgentRunRequest,
    SandboxedHarnessAgentVerifyRequest,
    SandboxedHarnessAgentVerifyResponse,
)


def _load_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _milliseconds(value: Any) -> Optional[float]:
    """Convert OpenCode's Date.now()-based epoch milliseconds to seconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return None
    return float(value) / 1000


def parse_opencode_observations(db_path: Path, fallback_invocation_id: str) -> AgentObservationBundle:
    """Read OpenCode's persisted session tree before its workspace is removed."""
    if not db_path.is_file():
        return AgentObservationBundle(
            source="opencode",
            records=[AgentInvocation(invocation_id=fallback_invocation_id)],
            gaps=[
                ObservationGap(code="agent_artifact_unavailable"),
                ObservationGap(code="agent_transcript_unavailable"),
                ObservationGap(code="model_call_ownership_unavailable"),
            ],
        )

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        session_rows = con.execute(
            "select id, parent_id, time_created from session order by time_created, id"
        ).fetchall()
        message_rows = con.execute(
            "select id, session_id, data, time_created from message order by time_created, id"
        ).fetchall()
        part_rows = con.execute(
            "select id, message_id, session_id, data, time_created from part order by time_created, id"
        ).fetchall()
    finally:
        con.close()

    messages = {row["id"]: _load_json(row["data"]) for row in message_rows}
    message_sessions = {row["id"]: row["session_id"] for row in message_rows}
    conversations: dict[str, list[Any]] = {row["id"]: [] for row in session_rows}
    invocation_status: dict[str, str] = {row["id"]: "unknown" for row in session_rows}
    tools: list[ToolCallObservation] = []
    child_tools: dict[str, set[str]] = {}
    child_status: dict[str, str] = {}
    compaction_parts: list[tuple[str, str, float | None, dict[str, Any]]] = []
    gaps: list[ObservationGap] = []
    summary_text: dict[str, list[str]] = {}
    summaries_by_parent: dict[str, list[str]] = {}
    first_item_id_by_message: dict[tuple[str, str], str] = {}

    for row in message_rows:
        message = messages[row["id"]]
        session_id = row["session_id"]
        if not isinstance(session_id, str) or session_id not in invocation_status:
            gaps.append(ObservationGap(code="agent_artifact_record_unowned", detail=row["id"]))
        elif message.get("role") == "assistant":
            if isinstance(message.get("error"), dict):
                invocation_status[session_id] = "failed"
            message_time = message.get("time") if isinstance(message.get("time"), dict) else {}
            if invocation_status[session_id] != "failed" and _milliseconds(message_time.get("completed")) is not None:
                invocation_status[session_id] = "completed"
        if message.get("summary") is True:
            summary_text[row["id"]] = []
            parent_id = message.get("parentID")
            if isinstance(parent_id, str):
                summaries_by_parent.setdefault(parent_id, []).append(row["id"])

    for row in part_rows:
        part = _load_json(row["data"])
        if not part:
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable"))
            continue
        ptype = part.get("type")
        message_id = row["message_id"]
        message = messages.get(message_id, {})
        session_id = row["session_id"] or message_sessions.get(message_id)
        if not isinstance(session_id, str):
            gaps.append(ObservationGap(code="agent_artifact_record_unowned"))
            continue
        conversation = conversations.setdefault(session_id, [])
        role = message.get("role")

        if ptype == "step-finish":
            continue

        text = part.get("text")
        if ptype == "text" and isinstance(text, str) and text.strip():
            if message_id in summary_text:
                summary_text[message_id].append(text)
            if role == "user" and part.get("ignored") is not True:
                conversation.append(NeMoGymEasyInputMessage(role="user", content=text))
            elif role == "assistant":
                item = NeMoGymResponseOutputMessage(
                    id=row["id"],
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
                conversation.append(item)
                first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "reasoning" and role == "assistant" and isinstance(text, str) and text.strip():
            conversation.append(
                NeMoGymResponseReasoningItem(
                    id=row["id"],
                    summary=[NeMoGymSummary(type="summary_text", text=text)],
                )
            )
            first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "tool" and role == "assistant":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            native_call_id = part.get("callID")
            observed_call_id = native_call_id if isinstance(native_call_id, str) and native_call_id else None
            call_id = observed_call_id or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            native_status = state.get("status")
            response_status = "completed" if native_status == "completed" else "incomplete"
            call = NeMoGymResponseFunctionToolCall(
                arguments=arguments,
                call_id=call_id,
                name=part.get("tool", ""),
                type="function_call",
                id=call_id,
                status=response_status,
            )
            conversation.append(call)
            first_item_id_by_message.setdefault((session_id, message_id), call_id)
            native_time = state.get("time") if isinstance(state.get("time"), dict) else {}
            # OpenCode retains raw output in SQLite but substitutes this literal in later model inputs after pruning.
            if native_status == "completed" and native_time.get("compacted") is not None:
                observed_tool_output = "[Old tool result content cleared]"
            else:
                observed_tool_output = state.get("output") if state.get("output") is not None else state.get("error")
            if observed_tool_output is not None:
                result = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=str(observed_tool_output),
                    status=response_status,
                )
                conversation.append(result)

            native_start = native_time.get("start")
            native_end = native_time.get("end")
            valid_interval = (
                isinstance(native_start, (int, float))
                and not isinstance(native_start, bool)
                and isinstance(native_end, (int, float))
                and not isinstance(native_end, bool)
                and native_end >= native_start
            )
            started_at = _milliseconds(native_start) if valid_interval else None
            completed_at = _milliseconds(native_end) if valid_interval else None
            duration_ms = float(native_end - native_start) if valid_interval else None
            status = {
                "completed": "completed",
                "error": "failed",
                "running": "incomplete",
                "pending": "incomplete",
            }.get(native_status, "unknown")
            if observed_call_id is not None:
                tools.append(
                    ToolCallObservation(
                        invocation_id=session_id,
                        tool_call_id=observed_call_id,
                        tool_name=part.get("tool") if isinstance(part.get("tool"), str) else None,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        timing_source="artifact" if started_at is not None else None,
                        status=status,
                        error_type="tool_error" if native_status == "error" else None,
                    )
                )
            else:
                gaps.append(
                    ObservationGap(
                        code="tool_call_identity_unavailable",
                        invocation_id=session_id,
                        detail=row["id"],
                    )
                )
            if observed_call_id is not None and (
                started_at is None or (native_status in {"completed", "error"} and completed_at is None)
            ):
                gaps.append(
                    ObservationGap(
                        code="tool_timing_unavailable",
                        invocation_id=session_id,
                        detail=observed_call_id,
                    )
                )
            metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
            if not metadata and isinstance(part.get("metadata"), dict):
                metadata = part["metadata"]
            child_id = metadata.get("sessionId")
            if isinstance(child_id, str) and observed_call_id is not None:
                child_tools.setdefault(child_id, set()).add(observed_call_id)
                child_status[child_id] = status
            continue
        if ptype == "compaction":
            compaction_parts.append((session_id, message_id, _milliseconds(row["time_created"]), part))
            continue

    compactions: list[ContextCompactionObservation] = []
    for session_id, message_id, observed_at, part in compaction_parts:
        summary_ids = summaries_by_parent.get(message_id, [])
        summary = "\n".join(summary_text.get(summary_ids[0], [])) if len(summary_ids) == 1 else None
        if len(summary_ids) > 1:
            gaps.append(
                ObservationGap(
                    code="compaction_summary_ambiguous",
                    invocation_id=session_id,
                )
            )
        trigger = "overflow" if part.get("overflow") is True else "automatic" if part.get("auto") is True else "manual"
        tail_start_id = part.get("tail_start_id") if isinstance(part.get("tail_start_id"), str) else None
        first_kept_item_id = (
            first_item_id_by_message.get((session_id, tail_start_id)) if tail_start_id is not None else None
        )
        compactions.append(
            ContextCompactionObservation(
                invocation_id=session_id,
                observed_at=observed_at,
                trigger=trigger,
                outcome="completed" if summary else "unknown",
                summary=summary,
                first_kept_item_id=first_kept_item_id,
            )
        )
        if tail_start_id is not None and first_kept_item_id is None:
            gaps.append(
                ObservationGap(
                    code="compaction_first_kept_item_unavailable",
                    invocation_id=session_id,
                    detail=tail_start_id,
                )
            )
        if not summary:
            gaps.append(ObservationGap(code="compaction_summary_unavailable", invocation_id=session_id))
        gaps.append(ObservationGap(code="compaction_token_counts_unavailable", invocation_id=session_id))
        gaps.append(
            ObservationGap(
                code="compaction_model_call_boundary_unavailable",
                invocation_id=session_id,
            )
        )

    session_ids = {row["id"] for row in session_rows}
    invocations = []
    for row in session_rows:
        invocation_id = row["id"]
        parent_id = row["parent_id"]
        spawn_candidates = child_tools.get(invocation_id, set())
        if parent_id is not None and parent_id not in session_ids:
            gaps.append(
                ObservationGap(
                    code="subagent_parent_unavailable",
                    invocation_id=invocation_id,
                    detail=parent_id,
                )
            )
        if len(spawn_candidates) > 1:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_ambiguous",
                    invocation_id=invocation_id,
                )
            )
        elif parent_id is not None and not spawn_candidates:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_tool_unavailable",
                    invocation_id=invocation_id,
                )
            )
        invocations.append(
            AgentInvocation(
                invocation_id=invocation_id,
                parent_invocation_id=parent_id,
                spawned_by_tool_call_id=next(iter(spawn_candidates)) if len(spawn_candidates) == 1 else None,
                status=(
                    invocation_status.get(invocation_id, "unknown")
                    if invocation_status.get(invocation_id, "unknown") != "unknown"
                    else child_status.get(invocation_id, "unknown")
                ),
                conversation=conversations.get(invocation_id, []),
            )
        )
    if not invocations:
        invocations = [AgentInvocation(invocation_id=fallback_invocation_id)]
        gaps.append(ObservationGap(code="agent_transcript_unavailable"))

    return AgentObservationBundle(
        source="opencode",
        records=[*invocations, *tools, *compactions],
        gaps=gaps,
    )


class OpenCodeSandboxedAgentConfig(SandboxedHarnessAgentConfig):
    opencode_version: str
    remote_opencode_install_script_path: Optional[str] = None
    remote_opencode_binary_path: Optional[str] = None
    remote_opencode_musl_binary_path: Optional[str] = None
    opencode_config: Dict[str, Any] = Field(default_factory=dict)
    opencode_max_context_window: int


# The request classes now live in the generic agent; these names stay for code that imports them.
OpenCodeSandboxedAgentRunRequest = SandboxedHarnessAgentRunRequest
OpenCodeSandboxedAgentVerifyRequest = SandboxedHarnessAgentVerifyRequest


def _build_remote_opencode_install_command(
    install_script_path: str,
    binary_path: str,
    musl_binary_path: str,
) -> str:
    """Build the invocation for the network-free, libc-aware cached installer."""
    return (
        f"bash {quote(install_script_path)} "
        f"--glibc-binary {quote(binary_path)} "
        f"--musl-binary {quote(musl_binary_path)}"
    )


def _extract_opencode_session_id(session_list_stdout: str) -> str:
    """Return the newest OpenCode session ID from ``session list`` JSON output."""
    sessions = json.loads(session_list_stdout)
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("OpenCode did not return any sessions")

    session_id = sessions[0].get("id") if isinstance(sessions[0], dict) else None
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("The newest OpenCode session does not have a valid ID")
    return session_id


class OpenCodeSandboxedAgentVerifyResponse(SandboxedHarnessAgentVerifyResponse):
    opencode_results_fpath: str
    opencode_run_stdout: str
    opencode_run_stderr: str
    opencode_finished: bool
    opencode_export_found: bool


class OpenCodeSandboxedAgent(SandboxedHarnessAgent):
    config: OpenCodeSandboxedAgentConfig

    harness_name: ClassVar[str] = "OpenCode"
    harness_id: ClassVar[str] = "opencode"
    finished_marker: ClassVar[str] = "OpenCode run finished"
    verify_response_class: ClassVar[type[SandboxedHarnessAgentVerifyResponse]] = OpenCodeSandboxedAgentVerifyResponse

    async def _create_opencode_config(self, request: Request) -> Dict[str, Any]:
        base_url = (
            self.base_url_for_run(
                base_url=get_server_url(self.config.model_server.name),
                body=await request.json(),
            )
            + "/v1"
        )
        return {
            "model": "nemo_gym/dummy_model",
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "nemo_gym": {
                    # TODO @bxyu-nvidia: We should use @ai-sdk/openai here but there is some /v1/responses streaming error.
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "dummy_key",  # pragma: allowlist secret
                        "timeout": False,
                        "chunkTimeout": 600000,  # in milliseconds, 10 min
                    },
                    "models": {
                        "dummy_model": {
                            "limit": {
                                "context": self.config.opencode_max_context_window,
                                "input": self.config.opencode_max_context_window,
                                # See the OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX flag below for more information.
                                "output": self.config.opencode_max_context_window,
                            },
                        },
                    },
                }
            },
            **self.config.opencode_config,
        }

    def _opencode_export_to_usages(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseUsage]:
        usages: List[NeMoGymResponseUsage] = []
        for message in opencode_export["messages"]:
            if message["info"]["role"] != "assistant":
                continue

            token_info = message["info"].get("tokens")
            if not token_info:
                continue

            usage = NeMoGymResponseUsage(
                input_tokens=token_info["input"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=token_info["cache"]["read"]),
                output_tokens=token_info["output"],
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=token_info["reasoning"]),
                total_tokens=token_info.get("total", 0),  # Somehow total may be missing
            )
            usages.append(usage)

        return usages

    def _opencode_export_to_output_items(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        messages = []
        for message in opencode_export["messages"]:
            if message["info"]["role"] == "user":
                message_parts = []
                for part in message["parts"]:
                    if part["type"] != "text":
                        continue

                    message_parts.append(ResponseInputTextParam(text=part["text"], type="input_text"))

                messages.append(NeMoGymEasyInputMessage(content=message_parts, role="user"))
            elif message["info"]["role"] == "assistant":
                converter = ResponsesConverter(return_token_id_information=True)
                for part in message["parts"]:
                    if part["type"] == "text":
                        output_items = converter.postprocess_assistant_message_dict(
                            message_dict={
                                "content": part["text"],
                                "role": "assistant",
                            }
                        )
                        messages.extend(output_items)
                    elif part["type"] == "reasoning":
                        output_items = converter.postprocess_assistant_message_dict(
                            message_dict={
                                "content": converter._wrap_reasoning_in_think_tags([part["text"]]),
                                "role": "assistant",
                            }
                        )
                        messages.extend(output_items)
                    elif part["type"] == "tool":
                        messages.append(
                            NeMoGymResponseFunctionToolCall(
                                arguments=json.dumps(part["state"]["input"]),
                                call_id=part["callID"],
                                name=part["tool"],
                            )
                        )
                        messages.append(
                            NeMoGymFunctionCallOutput(
                                call_id=part["callID"],
                                # @bxyu-nvidia: Somehow the output here may be missing...
                                output=part["state"].get("output", ""),
                            )
                        )
                    elif part["type"] in ("step-finish", "step-start", "patch"):
                        pass
                    else:
                        # @bxyu-nvidia: Defensive raise in case we're missing something.
                        raise NotImplementedError(part)
            else:
                # @bxyu-nvidia: Defensive raise in case we're missing something.
                raise NotImplementedError(message)

        return messages

    async def _harness_command(
        self, request: Request, query: str, collect_observations: bool
    ) -> tuple[str, Optional[str]]:
        opencode_debug_str = ""
        if self.config.debug:
            opencode_debug_str = "--print-logs --log-level DEBUG"

        opencode_thinking_str = "--thinking"

        if self.config.remote_opencode_binary_path and self.config.remote_opencode_install_script_path:
            if self.config.remote_opencode_musl_binary_path:
                install_str = _build_remote_opencode_install_command(
                    install_script_path=self.config.remote_opencode_install_script_path,
                    binary_path=self.config.remote_opencode_binary_path,
                    musl_binary_path=self.config.remote_opencode_musl_binary_path,
                )
            else:
                install_str = (
                    f"bash {quote(self.config.remote_opencode_install_script_path)} "
                    f"--binary {quote(self.config.remote_opencode_binary_path)}"
                )
        else:
            print(
                "Downloading and installing OpenCode in the sandbox. Please consider mounting or uploading the appropriate OpenCode binary instead!",
                file=sys.stderr,
            )
            install_str = f"""installer=$(mktemp) && curl -fL -o "$installer" https://opencode.ai/install \
        && echo "Downloaded OpenCode installer to $installer" \
        && VERSION={self.config.opencode_version} bash "$installer\""""

        opencode_config_content = json.dumps(await self._create_opencode_config(request))
        xdg_home_str = ""
        remote_data_home = None
        if collect_observations:
            remote_data_home = f"/tmp/nemo-gym-opencode-{uuid4().hex}"
            xdg_home_str = f"XDG_DATA_HOME={remote_data_home}"

        # @bxyu-nvidia: Regarding `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000` below:
        # OpenCode defaults to 32k here https://github.com/anomalyco/opencode/blob/58a99916bb96edf5cf605dc03e1be1e4bacf9ff7/packages/opencode/src/provider/transform.ts#L21
        # and there is no way to set it to null.
        # Here, we set an exorbitantly high number that cannot ever be reached.
        # In future versions of OpenCode, this can be directly passed via maxOutputTokens in the limit config above https://github.com/anomalyco/opencode/blob/1b18a50418f730aca32630ccfcde850f2b5fc360/packages/opencode/src/provider/transform.ts#L1418
        command = f"""
        echo "Shell: $SHELL" \
        && {install_str} \
        && export PATH=$HOME/.opencode/bin:$PATH \
        && echo "Installed OpenCode" \
        && OPENCODE_CONFIG_CONTENT={quote(opencode_config_content)} OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000 {xdg_home_str} \
            opencode run --title "NG dummy title" {opencode_debug_str} {opencode_thinking_str} -- {quote(query)} \
        && echo "{self.finished_marker}"
        """

        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\n", file=sys.stderr)
            print(f"OpenCode config JSON str: {opencode_config_content}", file=sys.stderr)

        return command, remote_data_home

    async def _harness_collect(
        self,
        request: Request,
        sandbox: AsyncSandbox,
        remote_data_home: Optional[str],
        collect_observations: bool,
        observation_invocation_id: Optional[str],
    ) -> HarnessTranscript:
        export_fname = "export.json"
        # Kept outside the sandbox workdir on purpose: SWE-bench-style environments set the workdir
        # to the git repo, and resources servers extract the model patch with `git add -N . && git
        # diff`, which would sweep this transcript into the patch.
        export_remote_fpath = f"/tmp/opencode_{export_fname}"
        try:
            session_env = {"XDG_DATA_HOME": remote_data_home} if remote_data_home is not None else None
            session_list_result = await sandbox.exec(
                command="export PATH=$HOME/.opencode/bin:$PATH && opencode session list --format json",
                env=session_env,
            )
            if session_list_result.return_code != 0:
                raise RuntimeError(f"Failed to list OpenCode sessions: {session_list_result}")
            session_id = _extract_opencode_session_id(session_list_result.stdout or "")
            export_result = await sandbox.exec(
                command=(
                    "export PATH=$HOME/.opencode/bin:$PATH "
                    f"&& opencode export {quote(session_id)} > {quote(export_remote_fpath)}"
                ),
                env=session_env,
            )
        except Exception:
            export_result = None
            print("Failed to export results", format_exc(), file=sys.stderr)
        if self.config.debug and export_result:
            print("Export stdout:\n", export_result.stdout, file=sys.stderr)
            print("Export stderr:\n", export_result.stderr, file=sys.stderr)

        results_dir: Path = Path(__file__).parent / "results" / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        results_local_fpath = results_dir / export_fname
        if export_result is not None and export_result.return_code == 0:
            if self.config.debug:
                print(f"Downloading results from {export_remote_fpath} to {results_local_fpath}", file=sys.stderr)
            try:
                await sandbox.download(export_remote_fpath, results_local_fpath)
            except:
                print(f"Failed to download export results to {results_local_fpath}", format_exc(), file=sys.stderr)
                print("Export stdout:\n", export_result.stdout, file=sys.stderr)
                print("Export stderr:\n", export_result.stderr, file=sys.stderr)

        observations = None
        if collect_observations:
            assert remote_data_home is not None
            observations_remote_fpath = f"{remote_data_home}/opencode/opencode.db"
            snapshot_remote_fpath = f"{remote_data_home}/opencode/nemo-gym-observations.db"
            observations_local_fpath = results_dir / "opencode.db"
            observations_local_fpath.unlink(missing_ok=True)
            try:
                snapshot_script = (
                    "import sqlite3,sys;"
                    "source=sqlite3.connect(f'file:{sys.argv[1]}?mode=ro',uri=True);"
                    "destination=sqlite3.connect(sys.argv[2]);"
                    "source.backup(destination);destination.close();source.close()"
                )
                snapshot_result = await sandbox.exec(
                    command=(
                        f"python3 -c {quote(snapshot_script)} "
                        f"{quote(observations_remote_fpath)} {quote(snapshot_remote_fpath)}"
                    )
                )
                if snapshot_result.return_code != 0 or snapshot_result.error_type is not None:
                    raise RuntimeError("OpenCode database snapshot failed")
                await sandbox.download(snapshot_remote_fpath, observations_local_fpath)
                observations = parse_opencode_observations(observations_local_fpath, observation_invocation_id)
            except Exception:
                print("Failed to capture OpenCode observations", format_exc(), file=sys.stderr)
                observations = AgentObservationBundle(
                    source="opencode",
                    records=[AgentInvocation(invocation_id=observation_invocation_id)],
                    gaps=[
                        ObservationGap(code="agent_artifact_unavailable"),
                        ObservationGap(code="agent_transcript_unavailable"),
                        ObservationGap(code="model_call_ownership_unavailable"),
                        ObservationGap(code="observation_capture_failed"),
                    ],
                )
            finally:
                observations_local_fpath.unlink(missing_ok=True)

        opencode_export = dict()
        if results_local_fpath.exists():
            opencode_export = json.loads(results_local_fpath.read_text().strip() or "{}")

        output = []
        usage = None
        opencode_export_found = False
        if opencode_export:
            opencode_export_found = True
            # Assume only one input message. May change with a system/developer message later on.
            output = self._opencode_export_to_output_items(opencode_export)[1:]
            usage = NeMoGymResponseUsage.sum_from_list(self._opencode_export_to_usages(opencode_export))

        return HarnessTranscript(
            output=output,
            usage=usage,
            results_fpath=results_local_fpath if opencode_export_found else None,
            observations=observations,
        )

    async def run(
        self, request: Request, body: OpenCodeSandboxedAgentRunRequest
    ) -> OpenCodeSandboxedAgentVerifyResponse:
        # FastAPI builds the /run request and response models from these annotations, so they stay OpenCode's.
        return await super().run(request, body)

    # @bxyu-nvidia: This is scraped from the raw create params. Later on we can dynamically set this if OpenCode exports this :rofl:
    system_prompt: ClassVar[Optional[str]] = (
        "You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.\n\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.\n\nIf the user asks for help or wants to give feedback inform them of the following:\n- /help: Get help with using opencode\n- To give feedback, users should report the issue at https://github.com/anomalyco/opencode/issues\n\nWhen the user directly asks about opencode (eg 'can opencode do...', 'does opencode have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from opencode docs at https://opencode.ai\n\n# Tone and style\nYou should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).\nRemember that your output will be displayed on a command line interface. Your responses can use GitHub-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.\nOutput text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.\nIf you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.\nOnly use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.\nIMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.\nIMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.\nIMPORTANT: Keep your responses short, since they will be displayed on a command line interface. You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail. Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as \"The answer is <answer>.\", \"Here is the content of the file...\" or \"Based on the information provided, the answer is...\" or \"Here is what I will do next...\". Here are some examples to demonstrate appropriate verbosity:\n<example>\nuser: what is 2+2?\nassistant: 4\n</example>\n\n<example>\nuser: is 11 a prime number?\nassistant: Yes\n</example>\n\n<example>\nuser: what command should I run to list files in the current directory?\nassistant: ls\n</example>\n\n<example>\nuser: what command should I run to watch files in the current directory?\nassistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]\nnpm run dev\n</example>\n\n<example>\nuser: what files are in the directory src/?\nassistant: [runs ls and sees foo.c, bar.c, baz.c]\nuser: which file contains the implementation of foo?\nassistant: src/foo.c\n</example>\n\n<example>\nuser: write tests for new feature\nassistant: [uses grep and glob search tools to find where similar tests are defined, uses concurrent read file tool use blocks in one tool call to read relevant files at the same time, uses edit file tool to write new tests]\n</example>\n\n# Proactiveness\nYou are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:\n1. Doing the right thing when asked, including taking actions and follow-up actions\n2. Not surprising the user with actions you take without asking\nFor example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.\n3. Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.\n\n# Following conventions\nWhen making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.\n- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).\n- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.\n- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.\n- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.\n\n# Code style\n- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked\n\n# Doing tasks\nThe user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:\n- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.\n- Implement the solution using all tools available to you\n- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.\n- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (e.g. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to AGENTS.md so that you will know to run it next time.\nNEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.\n\n- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.\n\n# Tool usage policy\n- When doing file search, prefer to use the Task tool in order to reduce context usage.\n- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run \"git status\" and \"git diff\", send a single message with two tool calls to run the calls in parallel.\n\nYou MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.\n\nIMPORTANT: Before you begin work, think about what the code you're editing is supposed to do based on the filenames directory structure.\n\n# Code References\n\nWhen referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.\n\n<example>\nuser: Where are errors from the client handled?\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\n</example>\n\nYou are powered by the model named dummy_model. The exact model ID is nemo_gym/dummy_model\nHere is some useful information about the environment you are running in:\n<env>\n  Working directory: /testbed\n  Workspace root folder: /testbed\n  Is directory a git repo: yes\n  Platform: linux\n  Today's date: Tue Aug 04 2026\n</env>\nSkills provide specialized instructions and workflows for specific tasks.\nUse the skill tool to load a skill when a task matches its description.\n<available_skills>\n  <skill>\n    <name>customize-opencode</name>\n    <description>Use ONLY when the user is editing or creating opencode's own configuration: opencode.json, opencode.jsonc, files under .opencode/, or files under ~/.config/opencode/. Also use when creating or fixing opencode agents, subagents, skills, plugins, MCP servers, or permission rules. Do not use for the user's own application code, or for any project that is not configuring opencode itself.</description>\n    <location>file:///testbed/%3Cbuilt-in%3E</location>\n  </skill>\n</available_skills>"
    )


if __name__ == "__main__":
    OpenCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeSandboxedAgent.run_webserver()  # noqa: F401
