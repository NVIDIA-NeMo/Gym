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
import os
import re
import time
import traceback
from datetime import datetime
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
from nemo_gym.global_config import (
    NEMO_GYM_LOG_DIR_KEY_NAME,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_models.vllm_model.app import VLLMConverter


class BrowsecompAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 400
    keep_rounds: int = 9999
    nudge_steps: bool = True
    max_context_tokens: int = 196608
    context_reset_pct: float = 0.3
    context_reset_keep_rounds: int = 3
    max_reset_count: Optional[int] = None
    max_run_retries: int = 1
    snap_dir: Optional[str] = None
    # When True, on /v1/responses entry, look for a per-sample checkpoint file
    # ({sample_id}.ckpt.json under the trajectories directory) and restore agent
    # state from it. The checkpoint is overwritten atomically at the end of every
    # completed step iteration; on successful completion of the trajectory it is
    # deleted. Lets a slurm restart resume from the last completed step instead
    # of redoing the whole sample.
    resume_from_trajectory: bool = True
    # vLLM-specific optimization: before each model call, hit the model server's
    # /tokenize endpoint to get the prompt's token count. If it exceeds the reset
    # threshold, skip the model call entirely and reset context — saving the cost
    # of generating an output that would just be discarded.
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
        res = super().setup_webserver()

        if self.config.save_model_call_using_vllm_tokenize_endpoint:
            global_config = get_global_config_dict()
            policy_model_config_dict = get_first_server_config_dict(global_config, self.config.model_server.name)

            # The logic below is fixed to only vllm_model! Other models like openai_model and local_vllm_model don't work.

            # Just call the first base_url endpoint with the tokenize request.
            base_urls = policy_model_config_dict["base_url"]
            base_url = base_urls if isinstance(base_urls, str) else base_urls[0]

            self._policy_model_openai_client = NeMoGymAsyncOpenAI(
                base_url=base_url, api_key=policy_model_config_dict["api_key"]
            )

        return res

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        task_index, attempt = None, None
        if self.config.snap_dir:
            task_index = body.metadata.pop("task_index")
            attempt = body.metadata.pop("attempt")
            body.metadata = body.metadata or {}

        new_outputs = []
        usage = None
        step = 0
        num_tool_calls = 0
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        reset_threshold = 0
        reset_count = 0
        max_reset_count = self.config.max_reset_count
        if self.config.max_context_tokens and self.config.context_reset_pct:
            reset_threshold = int(self.config.max_context_tokens * self.config.context_reset_pct)

        missing_end_think_count = 0

        # Per-step wall times since the last loop/start print; cleared after each loop log.
        step_times: List[float] = []
        # Step indices at which the context was reset, since the last loop/start print.
        context_reset_steps: List[int] = []

        user_query = [i for i in body.input if i.role == "user"][0].content
        user_query = user_query if isinstance(user_query, str) else user_query[0]["text"]

        time_taken = time.monotonic()
        time_taken_model_call = 0.0
        time_taken_tool_call = 0.0
        max_output_tokens = 0

        def print_log(label):
            q = body.input[0].content if body.input else ""
            last = new_outputs[-1] if new_outputs else None
            if step_times:
                stats = (
                    f"n={len(step_times)} "
                    f"min={min(step_times):.2f}s "
                    f"avg={sum(step_times) / len(step_times):.2f}s "
                    f"max={max(step_times):.2f}s"
                )
                series = "[" + ", ".join(f"{t:.2f}" for t in step_times) + "]"
            else:
                stats = "n=0"
                series = "[]"
            ctx = ", ".join(str(s) for s in context_reset_steps) if context_reset_steps else "none"
            print(
                f"[browsecomp {label} step={step}]\n"
                f"  step_times: {stats}\n"
                f"  step_times_s: {series}\n"
                f"  context_cleared_at: {ctx}\n"
                f"  missing_end_think: {missing_end_think_count}\n"
                f"  q:    {q}\n"
                f"  last: {getattr(last, 'type', 'none')} → {last}",
                flush=True,
            )

        # --- Per-rollout-attempt trajectory log: one JSONL file per (sample, attempt) ---
        user_msg_content = next((m.content for m in body.input if getattr(m, "role", None) == "user"), "")
        sample_id = hashlib.sha1(user_msg_content.encode("utf-8")).hexdigest()[:12] if user_msg_content else "anon"
        log_dir = Path(get_global_config_dict().get(NEMO_GYM_LOG_DIR_KEY_NAME) or "nemo_gym_logs")
        traj_dir = log_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        # Fresh filename per attempt: timestamp with microseconds disambiguates retries of the same sample.
        attempt_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        traj_path = traj_dir / f"{sample_id}__{attempt_stamp}.jsonl"
        traj_f = traj_path.open("w", encoding="utf-8")
        prev_event_ts = time.monotonic()

        def log_event(event, **extra):
            nonlocal prev_event_ts
            now = time.monotonic()
            rec = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "step": step,
                "event": event,
                "dt_s": round(now - prev_event_ts, 3),
                **extra,
            }
            traj_f.write(json.dumps(rec, default=str) + "\n")
            traj_f.flush()
            prev_event_ts = now

        log_event("start", sample_id=sample_id, question=user_msg_content)

        # --- Per-sample JSON checkpoint (one file per sample, overwritten each step) ---
        ckpt_path = traj_dir / f"{sample_id}.ckpt.json"

        def save_ckpt() -> None:
            """Atomically dump the minimum state needed to resume after a crash."""
            try:
                tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
                payload = {
                    "step": step,
                    "new_outputs": [o.model_dump(exclude_none=True) for o in new_outputs],
                    "missing_end_think_count": missing_end_think_count,
                    "num_tool_calls": num_tool_calls,
                    "reset_count": reset_count,
                    "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
                }
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, default=str)
                os.replace(tmp, ckpt_path)
            except Exception as e:
                log_event("checkpoint_save_failed", error_type=type(e).__name__, error_msg=str(e))

        if self.config.resume_from_trajectory and ckpt_path.exists():
            try:
                ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
                new_outputs = [self._parse_output_item(d) for d in ckpt.get("new_outputs") or []]
                step = int(ckpt.get("step") or 0)
                missing_end_think_count = int(ckpt.get("missing_end_think_count") or 0)
                num_tool_calls = int(ckpt.get("num_tool_calls") or 0)
                reset_count = int(ckpt.get("reset_count") or 0)
                saved_usage = ckpt.get("usage")
                if saved_usage is not None:
                    usage = NeMoGymResponseUsage.model_validate(saved_usage)
                log_event(
                    "resumed",
                    from_step=step,
                    n_outputs=len(new_outputs),
                    missing_end_think_count=missing_end_think_count,
                    num_tool_calls=num_tool_calls,
                    reset_count=reset_count,
                )
                print(
                    f"[browsecomp resumed sample={sample_id} from_step={step} "
                    f"n_outputs={len(new_outputs)} reset_count={reset_count}]",
                    flush=True,
                )
            except Exception as e:
                # Corrupted checkpoint — start fresh, leave the file alone for inspection.
                log_event("resume_failed", error_type=type(e).__name__, error_msg=str(e))

        print_log("start")
        step_start = time.monotonic()

        try:
            while True:
                step += 1
                step_iter_start = time.monotonic()

                if self.config.keep_rounds is not None and new_outputs:
                    new_outputs = self._compact_old_tool_messages(new_outputs)

                new_body = body.model_copy(update={"input": body.input + new_outputs})

                # --- Optional pre-call tokenize check (vLLM-specific): avoid wasted model call ---
                # If enabled, hit the model server's /tokenize endpoint to count prompt tokens
                # before generation. If over threshold, reset context immediately and skip the
                # model call. This violates Gym's abstractions (only works on vLLM-style models
                # exposing /tokenize) but saves the cost of generating an output we'd discard.
                if self.config.save_model_call_using_vllm_tokenize_endpoint:
                    pre_prompt_tokens = await self._count_prompt_tokens(new_body)
                    if (
                        reset_threshold
                        and pre_prompt_tokens > reset_threshold
                        and (max_reset_count is None or reset_count < max_reset_count)
                    ):
                        reset_count += 1
                        log_event(
                            "context_reset",
                            source="tokenize",
                            reset_count=reset_count,
                            prompt_tokens=pre_prompt_tokens,
                        )
                        print(
                            f"Step {step} hit {reset_count} reset(s) inside the tokenize flow at "
                            f"prompt_tokens={pre_prompt_tokens}",
                            flush=True,
                        )
                        if self.config.snap_dir:
                            self._save_snapshot(
                                messages=body.input + new_outputs,
                                task_index=task_index,
                                attempt=attempt,
                                reset_count=reset_count,
                                is_final=False,
                            )
                        # Adaptive shrink: try keep_rounds, keep_rounds-1, ... 0,
                        # accept the largest n whose resulting prompt fits under
                        # the threshold
                        chosen_outputs = None
                        post_prompt_tokens = pre_prompt_tokens
                        for n in range(self.config.context_reset_keep_rounds, -1, -1):
                            cand = self._extract_last_rounds(new_outputs, n=n)
                            cand_body = body.model_copy(update={"input": body.input + cand})
                            post_prompt_tokens = await self._count_prompt_tokens(cand_body)
                            if post_prompt_tokens <= reset_threshold:
                                chosen_outputs = cand
                                log_event("context_reset_shrink", kept_rounds=n, post_prompt_tokens=post_prompt_tokens)
                                break
                        if chosen_outputs is None:
                            # System prompt + user query alone exceed the reset
                            # threshold. There's no way to make this sample fit
                            log_event("context_reset_shrink_failed", post_prompt_tokens=post_prompt_tokens)
                            raise RuntimeError(
                                f"Context reset failed: prompt is {post_prompt_tokens} tokens "
                                f"(threshold {reset_threshold}) even with all tool history dropped. "
                                f"sample_id={sample_id} step={step}"
                            )
                        new_outputs = chosen_outputs
                        context_reset_steps.append(step)
                        save_ckpt()
                        continue

                # --- Model call (with per-call retry on empty <think>-only output) ---
                model_call_start = time.monotonic()
                returned_valid_response = False
                for retry_idx in range(self.config.max_run_retries):
                    log_event("model_call_begin", retry=retry_idx)
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

                    # Retry if the model only produced <think> content with no final answer.
                    raw_output_text = model_response.output_text
                    cleaned_output_text = re.sub(r"<think>.*?</think>", "", raw_output_text, flags=re.DOTALL).strip()
                    if (
                        not model_response.incomplete_details
                        or model_response.incomplete_details.reason == "content_filter"
                    ) and (cleaned_output_text or any(o for o in model_response.output if o.type == "function_call")):
                        returned_valid_response = True
                        break

                    missing_end_think_count += 1
                    print(
                        f"A model call is missing the end think ({missing_end_think_count} for this sample)",
                        flush=True,
                    )
                    log_event(
                        "model_call_retry",
                        reason="empty_after_think_strip",
                        retry=retry_idx,
                        raw_output_text_len=len(raw_output_text),
                    )
                model_call_dur = time.monotonic() - model_call_start
                time_taken_model_call += model_call_dur
                if model_response.usage and model_response.usage.output_tokens:
                    max_output_tokens = max(max_output_tokens, model_response.usage.output_tokens)

                # If we retried all the way through and still didn't get a valid response, exit.
                if not returned_valid_response:
                    log_event("model_response_invalid_break", retries=self.config.max_run_retries)
                    print(
                        f"Exited the agent loop due to no valid model response after "
                        f"{self.config.max_run_retries} retries (step={step})",
                        flush=True,
                    )
                    break

                prompt_tokens = model_response.usage.input_tokens if model_response.usage else 0
                output_tokens = model_response.usage.output_tokens if model_response.usage else None
                incomplete = model_response.incomplete_details.reason if model_response.incomplete_details else None
                log_event(
                    "model_call",
                    duration_s=round(model_call_dur, 3),
                    input_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    incomplete=incomplete,
                    output=[o.model_dump(exclude_none=True) for o in model_response.output],
                    input=(
                        [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m for m in new_body.input]
                        if incomplete
                        else None
                    ),
                )

                # --- Post-call context-reset check (fallback when tokenize pre-flight wasn't enabled) ---
                if (
                    reset_threshold
                    and prompt_tokens > reset_threshold
                    and (max_reset_count is None or reset_count < max_reset_count)
                ):
                    reset_count += 1
                    log_event(
                        "context_reset",
                        source="model_call",
                        reset_count=reset_count,
                        prompt_tokens=prompt_tokens,
                    )
                    print(
                        f"Step {step} hit {reset_count} reset(s) inside the model call flow at "
                        f"prompt_tokens={prompt_tokens}",
                        flush=True,
                    )
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
                    context_reset_steps.append(step)
                    save_ckpt()
                    continue

                output = model_response.output
                new_outputs.extend(output)

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

                if model_response.incomplete_details:
                    break

                # --- If the model decided to answer (no tool calls), we are done ---
                all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
                all_output_messages: List[NeMoGymResponseOutputMessage] = [
                    o for o in output if o.type == "message" and o.role == "assistant"
                ]
                if not all_fn_calls and all_output_messages:
                    break

                # --- Execute tool calls (sequentially; timed individually) ---
                tool_total_dur = 0.0
                for output_function_call in all_fn_calls:
                    num_tool_calls += 1
                    log_event("tool_call_begin", name=output_function_call.name, args=output_function_call.arguments)
                    tool_start = time.monotonic()
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{output_function_call.name}",
                        json=json.loads(output_function_call.arguments),
                        cookies=resources_server_cookies,
                    )
                    # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                    resources_server_cookies = api_response.cookies

                    tool_output = (await api_response.content.read()).decode()
                    tool_dur = time.monotonic() - tool_start
                    tool_total_dur += tool_dur
                    log_event(
                        "tool_call",
                        duration_s=round(tool_dur, 3),
                        name=output_function_call.name,
                        args=output_function_call.arguments,
                        output=tool_output,
                        output_len=len(tool_output),
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

                time_taken_tool_call += tool_total_dur

                now = time.monotonic()
                step_times.append(now - step_start)
                step_start = now

                step_total_dur = now - step_iter_start
                # Per-step summary (to trajectory JSONL and stdout) — makes bottleneck
                # analysis a one-grep operation.
                log_event(
                    "step_end",
                    duration_s=round(step_total_dur, 3),
                    model_s=round(model_call_dur, 3),
                    tool_s=round(tool_total_dur, 3),
                    n_tools=len(all_fn_calls),
                    input_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
                print(
                    f"[browsecomp step_end sample={sample_id} step={step}] "
                    f"dur={step_total_dur:.1f}s "
                    f"model={model_call_dur:.1f}s "
                    f"tools={tool_total_dur:.1f}s "
                    f"n_tools={len(all_fn_calls)} "
                    f"input_tokens={prompt_tokens} "
                    f"output_tokens={output_tokens} "
                    f"missing_end_think={missing_end_think_count}",
                    flush=True,
                )

                if step < 10:
                    print_log("loop")
                elif step % 10 == 0:
                    print_log("loop")
                    step_times.clear()
                    context_reset_steps.clear()

                # Check if max steps is not None and if we have exhausted it.
                if self.config.max_steps and step >= self.config.max_steps:
                    break

                # End of iteration — checkpoint reflects the post-step state.
                save_ckpt()

            print_log("final")
            log_event(
                "final",
                total_steps=step,
                missing_end_think_count=missing_end_think_count,
                num_tool_calls=num_tool_calls,
                reset_count=reset_count,
            )

            # record final context
            if self.config.snap_dir:
                self._save_snapshot(
                    messages=body.input + new_outputs,
                    task_index=task_index,
                    attempt=attempt,
                    reset_count=None,
                    is_final=True,
                )
        except Exception as e:
            err_input_source = locals().get("new_body")
            if err_input_source is not None and hasattr(err_input_source, "input"):
                err_input = err_input_source.input
            else:
                err_input = body.input
            log_event(
                "error",
                error_type=type(e).__name__,
                error_msg=str(e),
                traceback=traceback.format_exc()[:4000],
                input=[m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m for m in err_input],
            )
            raise
        finally:
            traj_f.close()

        # Trajectory finished cleanly — drop the checkpoint so a future re-run starts fresh.
        try:
            ckpt_path.unlink(missing_ok=True)
        except OSError:
            pass  # best-effort; a stale ckpt won't be re-read because the driver caches this sample

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        print(
            f"{user_query[:20]}... | FINISHED | Step {step} | Time: {time.monotonic() - time_taken:.2f}s (model {time_taken_model_call:.2f}s, tool {time_taken_tool_call:.2f}s) | Max output tokens: {max_output_tokens} | Missing end thinks: {missing_end_think_count}"
        )

        model_response.output = new_outputs
        model_response.usage = usage
        model_response.reset_count = reset_count
        model_response.num_tool_calls = num_tool_calls
        model_response.metadata = {"missing_end_think_count": str(missing_end_think_count)}
        return model_response

    async def run(self, request: Request, body: BrowsecompAgentRunRequest) -> BrowsecompAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        # prepare for recording
        if self.config.snap_dir:
            body.responses_create_params.metadata = dict(body.responses_create_params.metadata or {})
            body.responses_create_params.metadata["task_index"] = str(body._ng_task_index)
            body.responses_create_params.metadata["attempt"] = str(0)

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        response_json = await get_response_json(response)

        verify_request = BrowsecompAgentVerifyRequest.model_validate(body.model_dump() | {"response": response_json})

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)

        return BrowsecompAgentVerifyResponse.model_validate(
            await get_response_json(verify_response)
            | {"missing_end_think_count": response_json["metadata"]["missing_end_think_count"]}
        )

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

    @staticmethod
    def _parse_output_item(d: dict):
        """Turn a serialized output-item dict (from a checkpoint JSON) back into its Pydantic model."""
        t = d.get("type")
        if t == "reasoning":
            return NeMoGymResponseReasoningItem.model_validate(d)
        if t == "function_call":
            return NeMoGymResponseFunctionToolCall.model_validate(d)
        if t == "message":
            return NeMoGymResponseOutputMessage.model_validate(d)
        if t == "function_call_output":
            return NeMoGymFunctionCallOutput.model_validate(d)
        raise ValueError(f"Unknown output item type in checkpoint: {t!r}")

    async def _count_prompt_tokens(self, body) -> int:
        """Hit vLLM's /tokenize endpoint and return the prompt token count"""
        converter = VLLMConverter(return_token_id_information=False)
        chat_completion_create_params = converter.responses_to_chat_completion_create_params(body)
        chat_completion_create_params = chat_completion_create_params.model_dump()

        # Same projection as vllm_model/app.py:384.
        tokenize_body_dict = {}
        for key in ("model", "messages", "tools", "chat_template_kwargs"):
            if key in chat_completion_create_params:
                tokenize_body_dict[key] = chat_completion_create_params[key]

        tokenize_response = await self._policy_model_openai_client.create_tokenize(**tokenize_body_dict)
        return len(tokenize_response["tokens"])

    def _extract_last_rounds(self, new_outputs, n=None):
        """
        Extract the last n complete tool-call rounds from new_outputs.
        A round = one or more function_call items + their corresponding
        function_call_output items. Returns a flat list preserving order.
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
        sample_dir = Path(f"{self.config.snap_dir}/sample_{task_index}")
        if not sample_dir.exists():
            sample_dir.mkdir(parents=True)

        if is_final:
            sample_path = f"{sample_dir}/attempt_{attempt}_final.jsonl"
        else:
            sample_path = f"{sample_dir}/attempt_{attempt}_reset_{reset_count}.jsonl"

        with open(sample_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(msg.model_dump_json() + "\n")


if __name__ == "__main__":
    BrowsecompAgent.run_webserver()
