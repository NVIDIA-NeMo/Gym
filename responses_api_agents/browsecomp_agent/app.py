# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_models.vllm_model.app import VLLMConverter


def _qid(text: str) -> str:
    """Short stable id for a question, for [browsecomp] debug logs."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:10]


class BrowsecompAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 400
    keep_rounds: int = 9999
    nudge_steps: bool = True
    max_context_tokens: int = 196608
    context_reset_pct: float = 0.3
    # Absolute token threshold for context reset. When > 0 it OVERRIDES
    # max_context_tokens * context_reset_pct. 50000 = the token-based reset
    # standard (matches the bc_frankie_bash_tool baselines).
    context_reset_tokens: int = 0
    context_reset_keep_rounds: int = 3
    max_run_retries: int = 1
    # Cap on the number of context resets per trajectory (None = unlimited).
    max_reset_count: Optional[int] = None
    # When set, save a JSONL snapshot of the full conversation at every context
    # reset and at the end of the trajectory, under
    # {snap_dir}/sample_{task_index}/attempt_{attempt}_{reset_<N>|final}.jsonl.
    # Off when None. (ported from gym-gitlab fe9845ee)
    snap_dir: Optional[str] = None
    # Estimate prompt tokens via the vLLM /tokenize endpoint BEFORE the model
    # call to decide context reset, instead of paying for a full generation and
    # discarding it. (ported from gym-gitlab b66e37c6)
    save_model_call_using_vllm_tokenize_endpoint: bool = False


class BrowsecompAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgent(SimpleResponsesAPIAgent):
    config: BrowsecompAgentConfig
    _policy_model_openai_client: Optional[NeMoGymAsyncOpenAI] = None

    def setup_webserver(self):
        app = super().setup_webserver()
        # For the /tokenize-based pre-call token estimation we need a direct
        # client to the policy model's vLLM /tokenize endpoint. Built only when
        # the feature is enabled. (ported from gym-gitlab b66e37c6)
        if self.config.save_model_call_using_vllm_tokenize_endpoint:
            global_config = get_global_config_dict()
            policy_model_config_dict = get_first_server_config_dict(global_config, self.config.model_server.name)
            base_urls = policy_model_config_dict["base_url"]
            base_url = base_urls if isinstance(base_urls, str) else base_urls[0]
            self._policy_model_openai_client = NeMoGymAsyncOpenAI(
                base_url=base_url, api_key=policy_model_config_dict["api_key"]
            )
        return app

    @staticmethod
    def _reset_threshold(config: "BrowsecompAgentConfig") -> int:
        """Token count past which the context is reset. Absolute
        context_reset_tokens (50k standard) takes precedence; otherwise fall back
        to max_context_tokens * context_reset_pct. 0 disables reset."""
        if config.context_reset_tokens:
            return int(config.context_reset_tokens)
        if config.max_context_tokens and config.context_reset_pct:
            return int(config.max_context_tokens * config.context_reset_pct)
        return 0

    @staticmethod
    def _last_message_text(response: NeMoGymResponse) -> str:
        """Text of the most-recent assistant message item that has non-empty content, walking
        back from the end of the trajectory. Mirrors bc_frankie (browsecomp_agent.py:1054-1059):
        the empty-answer retry keys on the LAST content-bearing assistant turn, NOT the
        concatenation of every assistant turn (NeMoGymResponse.output_text). So a final
        think-only turn triggers a retry even when an earlier turn emitted a real answer.
        Empty string if no assistant message produced content."""
        for item in reversed(response.output):
            if getattr(item, "type", None) == "message":
                text = "".join(c.text for c in item.content if getattr(c, "type", None) == "output_text")
                if text:
                    return text
        return ""

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        qid = _qid(json.dumps([m.model_dump() if hasattr(m, "model_dump") else m for m in body.input], default=str))

        new_outputs = []
        full_trajectory = []  # never-trimmed; mirrors new_outputs appends across resets
        usage = None
        step = 0
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        reset_threshold = self._reset_threshold(self.config)

        # --- snapshot keys + per-trajectory counters (ported from gym-gitlab fe9845ee) ---
        task_index, attempt = None, None
        if self.config.snap_dir and body.metadata:
            task_index = body.metadata.pop("task_index", None)
            attempt = body.metadata.pop("attempt", None)
        reset_count = 0
        reset_steps = []  # step numbers at which a context reset fired (for the trajectory header)
        num_tool_calls = 0
        max_reset_count = self.config.max_reset_count

        while True:
            step += 1

            if self.config.keep_rounds is not None and new_outputs:
                new_outputs = self._compact_old_tool_messages(new_outputs)

            new_body = body.model_copy(update={"input": body.input + new_outputs})

            # --- Pre-call context reset via the vLLM /tokenize endpoint ---
            # Estimate the prompt token count BEFORE generating; if over the
            # threshold, reset now and skip the (otherwise wasted) generation.
            # (ported from gym-gitlab b66e37c6)
            if self.config.save_model_call_using_vllm_tokenize_endpoint:
                pre_prompt_tokens = await self._count_prompt_tokens(new_body)
                if (
                    reset_threshold
                    and pre_prompt_tokens > reset_threshold
                    and (max_reset_count is None or reset_count < max_reset_count)
                ):
                    reset_count += 1
                    reset_steps.append(step)
                    if self.config.snap_dir:
                        self._save_snapshot(
                            messages=body.input + new_outputs,
                            task_index=task_index,
                            attempt=attempt,
                            reset_count=reset_count,
                            is_final=False,
                        )
                    # Adaptive shrink: keep the largest number of recent rounds
                    # whose resulting prompt still fits under the threshold.
                    chosen = None
                    for n in range(self.config.context_reset_keep_rounds, -1, -1):
                        cand = self._extract_last_rounds(new_outputs, n=n)
                        cand_body = body.model_copy(update={"input": body.input + cand})
                        if await self._count_prompt_tokens(cand_body) <= reset_threshold:
                            chosen = cand
                            break
                    new_outputs = chosen if chosen is not None else []
                    continue

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            # We raise for status here since we expect model calls to always work.
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            # --- Check context reset threshold (post-call fallback; used when
            # save_model_call_using_vllm_tokenize_endpoint is off) ---
            prompt_tokens = model_response.usage.input_tokens if model_response.usage else 0
            if (
                reset_threshold
                and prompt_tokens > reset_threshold
                and (max_reset_count is None or reset_count < max_reset_count)
            ):
                reset_count += 1
                reset_steps.append(step)
                if self.config.snap_dir:
                    self._save_snapshot(
                        messages=body.input + new_outputs,
                        task_index=task_index,
                        attempt=attempt,
                        reset_count=reset_count,
                        is_final=False,
                    )
                if self.config.context_reset_keep_rounds > 0:
                    new_outputs = self._extract_last_rounds(new_outputs)
                else:
                    new_outputs = []
                continue

            output = model_response.output
            new_outputs.extend(output)
            full_trajectory.extend(output)

            if not usage:
                usage = model_response.usage
                model_response.usage = None

            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens

                # TODO support more advanced token details
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            # --- If the model decided to answer (no tool calls), we are done ---
            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            # --- Execute tool calls ---
            for output_function_call in all_fn_calls:
                num_tool_calls += 1
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                tool_output = (await api_response.content.read()).decode()
                # bc_frankie parity: the resources server wraps every tool result in a one-field
                # JSON envelope ({"results_string": "..."}), which JSON-escapes newlines — the
                # model then sees a single escaped line instead of raw multi-line text. This was
                # the last remaining materialized-prompt difference vs the bc_frankie harness.
                # Unwrap it so the model sees the raw text. Error bodies (different JSON shape)
                # and non-JSON payloads are left untouched.
                try:
                    parsed_tool_output = json.loads(tool_output)
                    if (
                        isinstance(parsed_tool_output, dict)
                        and set(parsed_tool_output) == {"results_string"}
                        and isinstance(parsed_tool_output["results_string"], str)
                    ):
                        tool_output = parsed_tool_output["results_string"]
                except json.JSONDecodeError:
                    pass
                if api_response.status >= 400:
                    print(
                        f"[browsecomp][tool_fail][{qid}] step={step} tool={output_function_call.name} "
                        f"status={api_response.status} body={tool_output[:300]}",
                        flush=True,
                    )
                if self.config.nudge_steps:
                    turns_left = self.config.max_steps - step
                    tool_output += "\n\n[%d turns remaining out of %d]" % (turns_left, self.config.max_steps)

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(tool_response)
                full_trajectory.append(tool_response)

            # --- Nudge the model at milestone steps ---
            if self.config.nudge_steps and all_fn_calls:
                quarter = self.config.max_steps // 4
                half = self.config.max_steps // 2
                near_end = int(self.config.max_steps * 0.875)
                nudge_msg = None
                if step == quarter:
                    nudge_msg = (
                        "\n\n\n\n\n"
                        "[SYSTEM NOTE: You have used %d out of %d turns. "
                        "Please consider consolidating your findings and "
                        "delivering an answer soon.]" % (step, self.config.max_steps)
                    )
                elif step == half:
                    nudge_msg = (
                        "\n\n\n\n\n"
                        "[SYSTEM NOTE: You have used %d out of %d turns — "
                        "you are halfway through your budget. You should start "
                        "formulating your final answer based on the research "
                        "you have already done. Do not keep searching endlessly.]" % (step, self.config.max_steps)
                    )
                elif step == near_end:
                    nudge_msg = (
                        "\n\n\n\n\n"
                        "[SYSTEM NOTE: URGENT — You have used %d out of %d turns. "
                        "You are almost out of turns. YOU MUST deliver your final "
                        "answer NOW using the information you have already gathered. "
                        "Do NOT make any more tool calls. Provide your best answer "
                        "immediately in the required format with 'Exact Answer:' on "
                        "a line by itself.]" % (step, self.config.max_steps)
                    )

                if nudge_msg:
                    last_tool = new_outputs[-1]
                    new_output = last_tool.output + nudge_msg
                    new_outputs[-1] = last_tool.model_copy(update={"output": new_output})
                    if full_trajectory:
                        full_trajectory[-1] = new_outputs[-1]

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                print(f"[browsecomp][max_steps][{qid}] step={step} max_steps={self.config.max_steps}", flush=True)
                break

        # --- Final trajectory snapshot (ported from gym-gitlab fe9845ee) ---
        if self.config.snap_dir:
            self._save_snapshot(
                messages=body.input + new_outputs,
                task_index=task_index,
                attempt=attempt,
                reset_count=None,
                is_final=True,
            )
            # Full untrimmed conversation (bc_frankie parity: one trajectory.jsonl per sample).
            self._save_trajectory(
                input_messages=body.input,
                full_trajectory=full_trajectory,
                task_index=task_index,
                attempt=attempt,
                reset_steps=reset_steps,
                reset_count=reset_count,
                num_tool_calls=num_tool_calls,
            )

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = full_trajectory
        model_response.usage = usage
        # Surface counters for downstream analysis (ported from gym-gitlab fe9845ee).
        # NeMoGymResponse(Response) has extra="allow", so these round-trip to /verify.
        model_response.reset_count = reset_count
        model_response.num_tool_calls = num_tool_calls
        return model_response

    async def run(self, request: Request, body: BrowsecompAgentRunRequest) -> BrowsecompAgentVerifyResponse:
        cookies = request.cookies

        question_text = getattr(body, "question", None) or ""
        rcp_input = body.responses_create_params.input
        if isinstance(rcp_input, str):
            rcp_input = [NeMoGymEasyInputMessage(role="user", content=rcp_input)]
        qid = _qid(json.dumps([m.model_dump() if hasattr(m, "model_dump") else m for m in rcp_input], default=str))
        print(f"[browsecomp][start][{qid}] question={question_text[:200]!r}", flush=True)

        try:
            seed_session_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_session_response)
            cookies = seed_session_response.cookies

            last_verify_response = None
            for attempt in range(self.config.max_run_retries):
                # Seed snapshot keys so responses() can name per-reset/-final files.
                # (ported from gym-gitlab fe9845ee)
                if self.config.snap_dir:
                    body.responses_create_params.metadata = dict(body.responses_create_params.metadata or {})
                    body.responses_create_params.metadata["task_index"] = str(getattr(body, "_ng_task_index", qid))
                    body.responses_create_params.metadata["attempt"] = str(attempt)
                response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path="/v1/responses",
                    json=body.responses_create_params,
                    cookies=cookies,
                )
                await raise_for_status(response)
                cookies = response.cookies

                # Retry if the model's LAST content-bearing turn was empty after <think>-strip.
                # (Keyed on the last assistant message, matching bc_frankie, NOT the concatenated
                # output_text — a final think-only turn retries even if an earlier turn had text.)
                response_json = await get_response_json(response)
                raw_output_text = self._last_message_text(NeMoGymResponse.model_validate(response_json))
                cleaned_output_text = re.sub(r"<think>.*?</think>", "", raw_output_text, flags=re.DOTALL).strip()
                # Need to get last_verify_response if all attempts are exhausted
                if not cleaned_output_text and attempt != self.config.max_run_retries - 1:
                    print(
                        f"[browsecomp][retry][{qid}] attempt={attempt + 1}/{self.config.max_run_retries} "
                        f"reason=empty_output_after_think_strip",
                        flush=True,
                    )
                    continue

                verify_request = BrowsecompAgentVerifyRequest.model_validate(
                    body.model_dump() | {"response": response_json}
                )

                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=verify_request.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(verify_response)

                last_verify_response = BrowsecompAgentVerifyResponse.model_validate(
                    await get_response_json(verify_response)
                )
                break

            reward = getattr(last_verify_response, "reward", None) if last_verify_response is not None else None
            outcome = "success" if (reward is not None and reward > 0) else "failure"
            print(f"[browsecomp][end][{qid}] outcome={outcome} reward={reward} attempts={attempt + 1}", flush=True)

            return last_verify_response
        except Exception as e:
            print(f"[browsecomp][abort][{qid}] error_type={type(e).__name__} error={str(e)[:300]}", flush=True)
            raise

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    def _compact_old_tool_messages(self, messages):
        """
        Replace old tool-call results with a placeholder, keeping only the most
        recent *keep_rounds* tool messages.  This is the key context-management
        trick that enables long agent trajectories within a finite context window.
        """
        tool_indices = [i for i, m in enumerate(messages) if m.type == "function_call_output"]
        if len(tool_indices) <= self.config.keep_rounds:
            return messages

        for i in range(len(tool_indices) - self.config.keep_rounds):
            idx = tool_indices[i]
            messages[idx] = messages[idx].model_copy(
                update={"output": "[Previous tool result hidden for context management]"}
            )
        return messages

    def _extract_last_rounds(self, new_outputs, n=None):
        """
        Extract the last n complete tool-call rounds from new_outputs.
        A round = one or more function_call items + their corresponding
        function_call_output items. Returns a flat list preserving order.
        n defaults to context_reset_keep_rounds; the /tokenize adaptive-shrink
        path passes smaller n to find a window that fits under the threshold.
        """
        if n is None:
            n = self.config.context_reset_keep_rounds
        if n <= 0:
            return []

        rounds = []
        i = len(new_outputs) - 1
        while i >= 0 and len(rounds) < n:
            if new_outputs[i].type == "function_call_output":
                # Walk backwards to collect all tool messages for this round
                tool_outputs = []
                while i >= 0 and new_outputs[i].type == "function_call_output":
                    tool_outputs.insert(0, new_outputs[i])
                    i -= 1
                # The assistant message that triggered these tool calls
                fn_calls = []
                while i >= 0 and new_outputs[i].type == "function_call":
                    fn_calls.insert(0, new_outputs[i])
                    i -= 1
                # Add to rounds
                if fn_calls:
                    rounds.insert(0, (fn_calls, tool_outputs))
            else:
                i -= 1

        result = []
        for fn_calls, tool_outputs in rounds:
            result.extend(fn_calls)
            result.extend(tool_outputs)
        return result

    def _save_snapshot(self, messages, task_index, attempt, reset_count, is_final):
        """Save a JSONL snapshot of the full conversation at a context reset or
        at trajectory end, keyed by task_index/attempt. Lets us inspect the
        pre-reset context (otherwise lost when history is trimmed).
        (ported from gym-gitlab fe9845ee)"""
        sample_dir = Path(f"{self.config.snap_dir}/sample_{task_index}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        if is_final:
            sample_path = f"{sample_dir}/attempt_{attempt}_final.jsonl"
        else:
            sample_path = f"{sample_dir}/attempt_{attempt}_reset_{reset_count}.jsonl"
        with open(sample_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(msg.model_dump_json() + "\n")

    def _save_trajectory(
        self, input_messages, full_trajectory, task_index, attempt, reset_steps, reset_count, num_tool_calls
    ):
        """Save the FULL untrimmed conversation for one sample to
        {snap_dir}/sample_{task_index}/attempt_{attempt}_trajectory.jsonl.
        Line 1 = metadata header; remaining lines = input prefix + every model/tool item, in order
        (never trimmed at context resets). (bc_frankie parity: one trajectory.jsonl per sample.)"""
        sample_dir = Path(f"{self.config.snap_dir}/sample_{task_index}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = f"{sample_dir}/attempt_{attempt}_trajectory.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "task_index": task_index,
                        "attempt": attempt,
                        "reset_count": reset_count,
                        "num_tool_calls": num_tool_calls,
                        "reset_steps": reset_steps,
                    }
                )
                + "\n"
            )
            for msg in list(input_messages) + full_trajectory:
                f.write(msg.model_dump_json() + "\n")

    async def _count_prompt_tokens(self, body) -> int:
        """Hit the policy model's vLLM /tokenize endpoint and return the prompt
        token count (used to decide context reset without a full generation).
        (ported from gym-gitlab b66e37c6)"""
        converter = VLLMConverter(return_token_id_information=False)
        chat_completion_create_params = converter.responses_to_chat_completion_create_params(body)
        chat_completion_create_params = chat_completion_create_params.model_dump()
        # Same projection as vllm_model/app.py's tokenize path.
        tokenize_body_dict = {}
        for key in ("model", "messages", "tools", "chat_template_kwargs"):
            if key in chat_completion_create_params:
                tokenize_body_dict[key] = chat_completion_create_params[key]
        tokenize_response = await self._policy_model_openai_client.create_tokenize(**tokenize_body_dict)
        return len(tokenize_response["tokens"])


if __name__ == "__main__":
    BrowsecompAgent.run_webserver()
