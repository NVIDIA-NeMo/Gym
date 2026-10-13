# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VCQA agent.

A `responses_api_agent` for the Verified Code QA benchmark. Each task gives
the model a per-task working tree (a repo snapshot or a git bundle checked
out at a specific commit) plus a problem statement; the agent runs a tool
loop in a sandboxed subprocess, then grades the model's final answer
against a must-have rubric via an LLM judge.

Per-rollout pipeline (`/run`):

1. Validate the row's `verifier_metadata` (`dataset_kind`, repo info, rubric).
2. Allocate a per-rollout scratch dir and fetch+extract the task's working
   tree there (tarball to extract, or git bundle to clone+checkout).
3. Start the configured sandbox backend, exposing the working tree at
   `/codebase:ro` plus a writable `/tmp/scratch` for `write_todos`.
4. Run the model+tool loop (mirrors `simple_agent.responses` line-by-line so
   token-ID-bearing output items propagate verbatim across turns and
   `prompt_token_ids` / `generation_token_ids` / `generation_log_probs`
   never get dropped).
5. Pull the last assistant text and grade it against the rubric's must-have
   items via an LLM judge.
6. Stop the sandbox, clean scratch state, and return the verify response.

Any failure inside steps 2-5 is caught and surfaced as `reward=0.0` plus an
`error` field on the verify response; a single bad row never crashes the
agent server.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.vcqa_agent.judge import grade as judge_grade
from responses_api_agents.vcqa_agent.materialize import materialize_working_tree
from responses_api_agents.vcqa_agent.sandbox import (
    ApptainerDirectSandbox,
    ApptainerSandbox,
    LocalSandbox,
    Sandbox,
    make_instance_name,
)
from responses_api_agents.vcqa_agent.tools import build_tool_definitions, dispatch_tool


########################################
# Config
########################################


class VcqaAgentConfig(BaseResponsesAPIAgentConfig):
    """Static config: every knob comes from YAML, never from env vars."""

    model_server: ModelServerRef
    sandbox_backend: Literal["apptainer", "apptainer_exec", "local"] = Field(
        default="apptainer",
        description="Where to run tool commands. `apptainer` = Apptainer instance per rollout (production). "
        "`apptainer_exec` = direct Apptainer exec per tool call, for nested-container environments "
        "where instance exec cannot enter its namespace. "
        "`local` = `cwd=working_tree`, no container, intended for dev / macOS testing only. "
        "the host filesystem outside the working tree is reachable to anything that escapes "
        "path containment.",
    )
    container_image: str = Field(
        default="docker://debian:bookworm-slim",
        description="Apptainer image URI. Pulled+cached on first use. Ignored when sandbox_backend='local'.",
    )
    apptainer_setup_command: str = Field(
        default=(
            "apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
            "ripgrep git fd-find tree ca-certificates "
            "&& ln -sf $(command -v fdfind) /usr/local/bin/fd"
        ),
        description="One-shot bash command run inside the container before the tool loop.",
    )
    apptainer_exec_timeout_s: int = Field(
        default=30, description="Default timeout for `apptainer exec` calls (per tool)."
    )
    artifact_url_prefix: str = Field(
        description="Base URL prefix that, joined with `verifier_metadata.artifact_key`, "
        "gives a fetchable URL for the per-task tarball or git bundle."
    )
    artifact_fetch_timeout_s: int = Field(
        default=120,
        description="Hard ceiling for a single artifact fetch. Past this the rollout fails "
        "with reward=0 instead of letting nemo_gym's global retry loop run forever on a "
        "broken artifact URL.",
    )
    artifact_request_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Extra HTTP headers attached to every artifact fetch (tarball / git "
        "bundle). Use this to authenticate against private hosts. Example for a private "
        "Hugging Face dataset repo:\n"
        "  artifact_request_headers:\n"
        "    Authorization: Bearer ${oc.env:HF_TOKEN}\n"
        "Values are passed through verbatim, so OmegaConf interpolation (incl. env-var "
        "lookup) works as usual.",
    )
    judge_base_url: str = Field(description="OpenAI-compatible judge endpoint base URL.")
    judge_api_key: str = Field(
        default="dummy", description="Bearer token for the judge endpoint."
    )  # pragma: allowlist secret
    judge_model_name: str = Field(description="Model name to send to the judge endpoint.")
    judge_timeout_s: int = Field(default=60, description="Per-criterion judge call timeout.")
    judge_max_completion_tokens: int = Field(
        default=2048,
        description="Per-call output budget for the judge. For reasoning models (gpt-5 / o-series) "
        "this needs to be large enough to cover hidden reasoning + the final YES/NO token.",
    )
    judge_reasoning_effort: Optional[str] = Field(
        default=None,
        description="Forwarded as `reasoning_effort` to the judge endpoint. Set to `minimal` "
        "when judging with a gpt-5 / o-series reasoning model so the YES/NO actually fits "
        "inside `judge_max_completion_tokens`. Leave None for non-reasoning models.",
    )
    max_turns: int = Field(default=150, description="Hard cap on tool-loop turns.")
    inject_tools: bool = Field(
        default=True,
        description="If true, the agent injects the VCQA tool definitions into "
        "responses_create_params.tools before each model call. Set false to let "
        "the row's tools field passthrough unchanged.",
    )
    include_distractor_tools: bool = Field(
        default=True,
        description="If true, surface the four distractor tools (install_package, "
        "send_pr_review, websearch, ask_user) alongside the six real tools. They "
        "exercise the model's investigator persona by surfacing tools that are "
        "off-task for a read-only code-investigation question (mutation, web "
        "search, asking the user). The grader does not currently penalize "
        "distractor usage. Set false for an A/B against the leaner "
        "real-tools-only surface. Ignored when `inject_tools` is false.",
    )


########################################
# Request / response models
########################################


class VcqaVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = None
    dataset_kind: Literal["fileset", "githistory"]
    repo_full_name: Optional[str] = None
    pre_merge_sha: Optional[str] = None
    head_sha: Optional[str] = None
    artifact_key: str
    rubric: Optional[Dict[str, Any]] = None
    verifiers: Optional[Dict[str, Any]] = None
    max_turns: Optional[int] = None


class VcqaAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: VcqaVerifierMetadata


class VcqaAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    error: Optional[str] = None
    judge_per_item: Optional[List[Dict[str, Any]]] = None
    must_pass: Optional[int] = None
    must_total: Optional[int] = None
    final_answer: Optional[str] = None
    num_turns: Optional[int] = None


########################################
# Agent
########################################


class VcqaAgent(SimpleResponsesAPIAgent):
    config: VcqaAgentConfig

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Standalone /v1/responses is not supported because VCQA needs a per-row sandbox."""
        raise NotImplementedError("vcqa_agent does not support /v1/responses standalone. Post a row to /run instead.")

    async def run(
        self,
        request: Request,
        response: Response,
        body: VcqaAgentRunRequest = Body(),
    ) -> VcqaAgentVerifyResponse:
        meta = body.verifier_metadata
        rubric = self._select_rubric(meta)
        max_turns = meta.max_turns or self.config.max_turns

        scratch_dir = Path(tempfile.mkdtemp(prefix="vcqa_"))
        sandbox: Optional[Sandbox] = None
        model_response: Optional[NeMoGymResponse] = None
        final_answer = ""
        num_turns = 0
        run_error: Optional[str] = None
        judge_per_item: Optional[List[Dict[str, Any]]] = None
        must_pass: Optional[int] = None
        must_total: Optional[int] = None
        reward = 0.0

        try:
            artifact_url = self._build_artifact_url(meta.artifact_key)
            working_tree = await materialize_working_tree(
                dataset_kind=meta.dataset_kind,
                artifact_url=artifact_url,
                head_sha=meta.head_sha,
                scratch_dir=scratch_dir,
                fetch_timeout_s=self.config.artifact_fetch_timeout_s,
                fetch_headers=self.config.artifact_request_headers,
            )

            sandbox = self._build_sandbox(working_tree=working_tree, scratch_dir=scratch_dir / "sandbox_scratch")
            await sandbox.start()

            model_response, num_turns = await self._tool_loop(
                request=request,
                response=response,
                body=body.responses_create_params,
                sandbox=sandbox,
                max_turns=max_turns,
            )

            final_answer = _extract_final_assistant_text(model_response)
            problem_statement = _extract_problem_statement(body.responses_create_params)

            judge_result = await judge_grade(
                rubric=rubric,
                problem_statement=problem_statement,
                model_answer=final_answer,
                base_url=self.config.judge_base_url,
                api_key=self.config.judge_api_key,
                model_name=self.config.judge_model_name,
                timeout_s=self.config.judge_timeout_s,
                cookies=dict(request.cookies),
                max_completion_tokens=self.config.judge_max_completion_tokens,
                reasoning_effort=self.config.judge_reasoning_effort,
            )
            reward = judge_result.reward
            judge_per_item = judge_result.per_item
            must_pass = judge_result.must_pass
            must_total = judge_result.must_total
            run_error = judge_result.error

        except Exception as e:
            run_error = f"{type(e).__name__}: {e}"
            reward = 0.0
        finally:
            if sandbox is not None:
                try:
                    await sandbox.stop()
                except Exception:
                    pass
            shutil.rmtree(scratch_dir, ignore_errors=True)

        if model_response is None:
            model_response = _empty_response(body.responses_create_params)

        return VcqaAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=model_response,
            reward=reward,
            error=run_error,
            judge_per_item=judge_per_item,
            must_pass=must_pass,
            must_total=must_total,
            final_answer=final_answer,
            num_turns=num_turns,
        )

    ########################################
    # Internals
    ########################################

    def _build_artifact_url(self, artifact_key: str) -> str:
        prefix = self.config.artifact_url_prefix.rstrip("/")
        key = artifact_key.lstrip("/")
        return f"{prefix}/{key}"

    def _build_sandbox(self, *, working_tree: Path, scratch_dir: Path) -> Sandbox:
        if self.config.sandbox_backend == "local":
            return LocalSandbox(
                working_tree=working_tree,
                scratch_dir=scratch_dir,
                exec_timeout_s=self.config.apptainer_exec_timeout_s,
            )
        if self.config.sandbox_backend == "apptainer_exec":
            return ApptainerDirectSandbox(
                container_image=self.config.container_image,
                working_tree=working_tree,
                scratch_dir=scratch_dir,
                exec_timeout_s=self.config.apptainer_exec_timeout_s,
                extra_setup_command=self.config.apptainer_setup_command or None,
            )
        return ApptainerSandbox(
            instance_name=make_instance_name(),
            container_image=self.config.container_image,
            working_tree=working_tree,
            scratch_dir=scratch_dir,
            exec_timeout_s=self.config.apptainer_exec_timeout_s,
            extra_setup_command=self.config.apptainer_setup_command or None,
        )

    @staticmethod
    def _select_rubric(meta: VcqaVerifierMetadata) -> Optional[Dict[str, Any]]:
        """Fileset rows put the rubric under `rubric`; githistory rows under `verifiers`."""
        if meta.rubric:
            return meta.rubric
        return meta.verifiers

    async def _tool_loop(
        self,
        *,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming,
        sandbox: Sandbox,
        max_turns: int,
    ) -> tuple[NeMoGymResponse, int]:
        """Mirror of `simple_agent.responses`, with local sandbox-backed tool dispatch.

        Output items are appended verbatim to the running input list so any
        token-ID/log-prob fields the model server stamped on them propagate
        unchanged into subsequent turns (NeMo Gym guideline 2).
        """
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        if self.config.inject_tools:
            existing = list(body.tools) if body.tools else []
            body.tools = existing + build_tool_definitions(include_distractors=self.config.include_distractor_tools)

        new_outputs: List[Any] = []
        usage = None
        step = 0
        model_server_cookies = None
        latest_response: Optional[NeMoGymResponse] = None

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response_raw = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response_raw)
            model_server_cookies = model_response_raw.cookies
            model_response_json = await get_response_json(model_response_raw)
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(f"Invalid response from model server: {json.dumps(model_response_json)}") from e
            latest_response = model_response

            output = model_response.output
            new_outputs.extend(output)

            if not usage:
                usage = model_response.usage
                model_response.usage = None
            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details:
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in output if isinstance(o, NeMoGymResponseFunctionToolCall)
            ]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if isinstance(o, NeMoGymResponseOutputMessage)
            ]
            if not all_fn_calls and all_output_messages:
                break

            for fn_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(fn_call.arguments) if fn_call.arguments else {}
                except (json.JSONDecodeError, TypeError) as e:
                    new_outputs.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=fn_call.call_id,
                            output=json.dumps({"error": f"Invalid tool call arguments: {e!r}"}),
                        )
                    )
                    continue

                tool_result = await dispatch_tool(fn_call.name, parsed_arguments, sandbox)
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=fn_call.call_id,
                        output=json.dumps(tool_result),
                    )
                )

            if step >= max_turns:
                break

        for k, v in (model_server_cookies or {}).items():
            response.set_cookie(k, v)

        assert latest_response is not None
        latest_response.output = new_outputs
        latest_response.usage = usage
        return latest_response, step


########################################
# Helpers
########################################


def _extract_final_assistant_text(resp: NeMoGymResponse) -> str:
    """Last assistant message's concatenated `output_text` in the response."""
    for item in reversed(resp.output):
        if isinstance(item, NeMoGymResponseOutputMessage) and item.role == "assistant":
            text_parts: List[str] = []
            for content in item.content:
                text = getattr(content, "text", None)
                if text:
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts).strip()
    return ""


def _extract_problem_statement(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Best-effort: concatenate user-message contents from the input."""
    if isinstance(params.input, str):
        return params.input
    parts: List[str] = []
    for item in params.input:
        if isinstance(item, (NeMoGymEasyInputMessage,)):
            if item.role == "user":
                if isinstance(item.content, str):
                    parts.append(item.content)
                else:
                    for c in item.content:
                        text = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                        if text:
                            parts.append(text)
    return "\n".join(parts).strip()


def _empty_response(params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
    """Build a minimal NeMoGymResponse for the failure-before-model case."""
    return NeMoGymResponse(
        id="vcqa-empty",
        created_at=0,
        model=params.model or "unknown",
        object="response",
        output=[],
        parallel_tool_calls=params.parallel_tool_calls,
        tool_choice=params.tool_choice,
        tools=[],
    )


if __name__ == "__main__":
    VcqaAgent.run_webserver()
