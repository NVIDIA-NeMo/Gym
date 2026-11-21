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
"""ARC code agent: a multi-turn, image-aware, code-executing harness for ARC-style grid puzzles.

The agent builds the whole prompt itself from the row fields (``examples``, ``test_input``): a system prompt, then one
user message with each grid as digits plus an image. The model writes Python inside ``<code>...</code>``; the agent
runs it in a sandbox and returns the output as a ``<tool_response>`` user turn, and repeats until the model gives the
final grid inside ``<answer>...</answer>``. Turns cut by the per-turn cap get a nudge; when the token budget is low,
a final answer is forced with thinking off. The resources server (``arc_code``) scores the final answer.

Row schema: ``{"task_id", "examples": [{"input", "output"}, ...], "test_input", "expected_output",
"responses_create_params": {"input": [...]}}`` with grids as ``list[list[int]]``. The row's own ``input`` is ignored.
"""

import hashlib
import json
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, Request, Response
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.arc_code.grid import Grid, is_grid, parse_answer, unreadable_reason
from responses_api_agents.arc_code_agent.arc_prompt import (
    CUT_NUDGE,
    FINAL_NUDGE,
    HYGIENE_IMPORTS,
    HYGIENE_RECURSION,
    NOCODE_NUDGE,
    PYTHON_TOOL,
    RESEND_NUDGE,
    build_system_prompt,
    sandbox_header,
    user_content,
)
from responses_api_agents.arc_code_agent.arc_sandbox import (
    SandboxPool,
    checked_function,
    extract_code,
    full_check_pass,
    tool_response_text,
)
from responses_api_agents.arc_code_agent.arc_sandbox_api import DEFAULT_SETUP_COMMAND, SandboxApiPool


class ArcCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # Token budget. ``cap`` bounds everything generated plus everything added to the conversation after the first
    # prompt; ``answer_tokens`` is reserved for the forced final answer. ``turn_cap`` (default ``cap``) bounds one turn.
    effort: str = "xhigh"
    cap: int = 73728
    answer_tokens: int = 4096
    turn_cap: Optional[int] = None
    margin: int = 64
    max_runs: int = 24
    max_cuts: int = 6  # turns cut by the per-turn cap before the attempt goes straight to the forced final answer
    chars_per_token: float = 3.0  # estimate for harness text (tool output, nudges) before the model server counts it

    # Sampling. ``seed_mode`` "per_row" hashes (seed_offset, task_id, rollout index, turn) into a request seed.
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    seed_mode: Literal["per_row", "none"] = "per_row"
    seed_offset: int = 141
    stop_strings: List[str] = ["</code>"]
    include_stop_str_in_output: bool = True
    # Chat-template kwargs per call: thinking turns default to {"enable_thinking": true, "reasoning_effort": effort};
    # the forced final answer defaults to thinking off. Extra vLLM request fields can go in ``extra_body``.
    thinking_template_kwargs: Optional[Dict[str, Any]] = None
    final_template_kwargs: Dict[str, Any] = {"enable_thinking": False}
    extra_body: Dict[str, Any] = {}
    keep_thinking: bool = True  # False drops the reasoning of past turns when the history is re-sent

    # Images of the first user message.
    image_px: int = 336
    image_scale: Optional[int] = 12
    image_style: Optional[Literal["pad32"]] = "pad32"
    record_images: bool = True  # False stores "<image omitted>" instead of the data URLs in the returned prompt

    # Harness behaviour.
    auto_answer: bool = (
        True  # the first run whose `check` passes every example pair ends the attempt with its prediction
    )
    answer_resend: int = 2  # how often an unreadable <answer> block is sent back to the model
    clean_final: bool = True  # drop a trailing cut nudge before the forced final answer
    runs_cap_final: bool = True  # reaching max_runs goes straight to the forced final answer
    nocode_nudge: int = 0  # turns with neither code nor answer that are nudged instead of ending the attempt
    parse_tool_calls: bool = True  # native <tool_call> blocks count as code
    # Native function calling: offer a `python` function tool in each request instead of the <code> text protocol (no
    # stop string); every function call of a turn is run in the sandbox and answered with a function_call_output item.
    native_tools: bool = False

    # Sandbox.
    timeout_s: float = 30.0
    out_chars: int = 4000
    mem_mb: int = 6144
    drop_timed_out: bool = True  # a timed-out cell is not replayed in later runs
    tool_fixes: bool = True
    helpers_v2: bool = True
    helpers_v3_new: bool = True
    sandbox_hygiene: bool = True  # more allowed modules, higher recursion limit, common names preloaded
    num_sandbox_workers: int = 8
    sandbox_python: Optional[str] = None  # interpreter with numpy, scipy and networkx (default: the agent's own)
    sandbox_command_prefix: List[str] = []  # for example ["unshare", "-rn"] to run the sandbox without network
    # Where code runs. "local": sandbox processes on the agent's host (above). "sandbox_api": remote sandboxes through
    # Gym's sandbox API (nemo_gym.sandbox, for example OpenSandbox), num_sandbox_workers of them, configured by
    # sandbox_provider (provider config, for example {"opensandbox": {"connection": {...}}}) and sandbox_spec
    # (image, ttl_s, resources, metadata, ...); sandbox_setup_command installs numpy, scipy and networkx in each one.
    sandbox_backend: Literal["local", "sandbox_api"] = "local"
    sandbox_provider: Optional[Dict[str, Any]] = None
    sandbox_spec: Dict[str, Any] = {}
    sandbox_setup_command: Optional[str] = DEFAULT_SETUP_COMMAND


class ArcCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = None
    examples: List[Dict[str, Any]] = []
    test_input: List[List[int]] = []
    expected_output: List[List[int]] = []


class ArcCodeAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ArcCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    turns: int = 0
    code_runs: int = 0
    cuts: int = 0
    resends: int = 0
    generated_tokens: int = 0
    used_tokens: int = 0
    prompt_tokens: int = 0
    wall_seconds: float = 0.0
    finished_with_answer: bool = False
    answer_source: str = "none"  # typed | auto | forced | none
    forced_final: bool = False


def _message_text(response: NeMoGymResponse) -> str:
    """Visible text of the assistant messages of a response (reasoning is a separate output item)."""
    texts: List[str] = []
    for item in response.output:
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            for part in getattr(item, "content", None) or []:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
    return "\n".join(texts)


def _assistant_message(text: str, tag: str) -> NeMoGymResponseOutputMessage:
    # Every `type`, `role` and `status` is passed explicitly, also on the text part: the request is serialized with
    # exclude_unset, which drops defaulted fields, and the model server then rejects the item (KeyError 'type', or a
    # 422 when only the text part lacks its type).
    return NeMoGymResponseOutputMessage(
        id=f"{tag}-{uuid.uuid4().hex[:12]}",
        type="message",
        role="assistant",
        status="completed",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _answer_block(grid: Grid) -> str:
    return "<answer>\n" + "\n".join(" ".join(str(v) for v in row) for row in grid) + "\n</answer>"


def _was_cut(response: NeMoGymResponse, max_tokens: int) -> bool:
    """True when the turn stopped at its token limit (the model server may not report a finish reason)."""
    if getattr(response, "status", None) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    if details is not None and getattr(details, "reason", None) == "max_output_tokens":
        return True
    return bool(response.usage is not None and response.usage.output_tokens >= max_tokens)


class _Ledger:
    """Tokens used so far: generated plus added after the first prompt.

    After a call the model server's prompt size gives the exact conversation length; text added since (tool output,
    nudges) is estimated from its length until the next call measures it.
    """

    def __init__(self, chars_per_token: float) -> None:
        self.chars_per_token = chars_per_token
        self.first_prompt = 0
        self.last_prompt = 0
        self.last_output = 0
        self.generated = 0
        self.pending = 0
        self.exhausted = False
        self.cap = 0

    def estimate(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + 1

    def add_text(self, text: str) -> None:
        self.pending += self.estimate(text)

    def record_call(self, input_tokens: int, output_tokens: int) -> None:
        if not self.first_prompt:
            self.first_prompt = input_tokens
        self.last_prompt, self.last_output = input_tokens, output_tokens
        self.generated += output_tokens
        self.pending = 0

    def used(self, cap: int) -> int:
        if self.exhausted:
            return cap
        if self.first_prompt:
            return max(self.last_prompt - self.first_prompt, 0) + self.last_output + self.pending
        return self.generated + self.pending


class ArcCodeAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: ArcCodeAgentConfig
    _pool: Optional[SandboxPool] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        cfg = self.config
        common = dict(
            num_workers=cfg.num_sandbox_workers,
            timeout_s=cfg.timeout_s,
            out_chars=cfg.out_chars,
            mem_mb=cfg.mem_mb,
            extra_imports=HYGIENE_IMPORTS if cfg.sandbox_hygiene else [],
            recursion_limit=HYGIENE_RECURSION if cfg.sandbox_hygiene else 0,
        )
        if cfg.sandbox_backend == "sandbox_api":
            self._pool = SandboxApiPool(
                provider=cfg.sandbox_provider,
                sandbox_spec=cfg.sandbox_spec,
                setup_command=cfg.sandbox_setup_command,
                **common,
            )
        else:
            self._pool = SandboxPool(python=cfg.sandbox_python, command_prefix=cfg.sandbox_command_prefix, **common)

    # -- model calls ---------------------------------------------------------------------------------------------

    def _request_seed(self, task_id: str, sample: int, turn: int) -> Optional[int]:
        if self.config.seed_mode == "none":
            return None
        key = f"{self.config.seed_offset}:{task_id}:{sample}:{turn}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)

    def _model_body(
        self,
        base: NeMoGymResponseCreateParamsNonStreaming,
        items: List[Any],
        max_tokens: int,
        seed: Optional[int],
        template_kwargs: Dict[str, Any],
        stop: bool,
        tool_choice: Optional[str] = None,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        cfg = self.config
        extra_body: Dict[str, Any] = {"top_k": cfg.top_k, **cfg.extra_body}
        if seed is not None:
            extra_body["seed"] = seed
        if stop and cfg.stop_strings and not cfg.native_tools:
            extra_body["stop"] = cfg.stop_strings
            if cfg.include_stop_str_in_output:
                extra_body["include_stop_str_in_output"] = True
        metadata = dict(base.metadata or {})
        metadata["extra_body"] = json.dumps(extra_body)
        metadata["chat_template_kwargs"] = json.dumps(template_kwargs)
        update: Dict[str, Any] = {
            "input": items,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_output_tokens": max_tokens,
            "metadata": metadata,
        }
        if cfg.native_tools:
            update["tools"] = [dict(PYTHON_TOOL)]
            update["tool_choice"] = tool_choice or "auto"
        return base.model_copy(update=update)

    async def _call_model(self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: Any, url_path: str):
        resp = await self.server_client.post(
            server_name=self.config.model_server.name, url_path=url_path, json=body, cookies=cookies
        )
        await raise_for_status(resp)
        return NeMoGymResponse.model_validate(await get_response_json(resp)), resp.cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError("ArcCodeAgent builds its own prompt from a puzzle row; use /run")

    # -- the attempt ---------------------------------------------------------------------------------------------

    async def _attempt(
        self,
        base: NeMoGymResponseCreateParamsNonStreaming,
        row: ArcCodeAgentRunRequest,
        sample: int,
        cookies: Any,
        url_path: str,
    ) -> Dict[str, Any]:
        """Run one attempt; returns the prompt items, the new items, the final assistant text and the counters."""
        cfg = self.config
        task_id = row.task_id or "task"
        system = build_system_prompt(
            cfg.cap,
            cfg.max_runs,
            cfg.out_chars,
            helpers_v2=cfg.helpers_v2,
            helpers_v3_new=cfg.helpers_v3_new,
            sandbox_hygiene=cfg.sandbox_hygiene,
            native_tools=cfg.native_tools,
        )
        content = user_content(row.examples, row.test_input, cfg.image_px, cfg.image_scale, cfg.image_style)
        prompt = [
            NeMoGymEasyInputMessage(role="system", content=system),
            NeMoGymEasyInputMessage(role="user", content=content),
        ]
        header = sandbox_header(
            row.examples,
            row.test_input,
            tool_fixes=cfg.tool_fixes,
            helpers_v2=cfg.helpers_v2,
            helpers_v3_new=cfg.helpers_v3_new,
            sandbox_hygiene=cfg.sandbox_hygiene,
        )
        thinking = cfg.thinking_template_kwargs or {"enable_thinking": True, "reasoning_effort": cfg.effort}
        turn_cap = cfg.turn_cap or cfg.cap

        history: List[Any] = []  # everything after the prompt, in order: model output items and harness user turns
        codes: List[str] = []
        ledger = _Ledger(cfg.chars_per_token)
        usage = None
        counters = {"turns": 0, "code_runs": 0, "cuts": 0, "resends": 0, "forced_final": False}
        resend_left, nocode_left, force_now = cfg.answer_resend, cfg.nocode_nudge, False
        final_text: Optional[str] = None
        answer_source = "none"
        last_response: Optional[NeMoGymResponse] = None

        def items() -> List[Any]:
            out = list(prompt) + history
            if not cfg.keep_thinking:
                out = [i for i in out if getattr(i, "type", None) != "reasoning"]
            return out

        def add_model_output(resp: NeMoGymResponse) -> None:
            history.extend(resp.output)
            # A cut turn may hold reasoning only; keep the assistant turn boundary so the history stays well formed.
            if not any(getattr(i, "type", None) in ("message", "function_call") for i in resp.output):
                history.append(_assistant_message("", "empty"))

        def add_user(text: str) -> None:
            history.append(NeMoGymEasyInputMessage(role="user", content=text))
            ledger.add_text(text)

        def add_tool_output(call_id: str, text: str) -> None:
            # `type` is passed explicitly: the request is serialized with exclude_unset, which would drop a defaulted type.
            history.append(NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=text))
            ledger.add_text(text)

        def record(resp: NeMoGymResponse) -> None:
            nonlocal usage, last_response
            usage = accumulate_response_usage(usage, resp.usage)
            last_response = resp
            if resp.usage is not None:
                ledger.record_call(resp.usage.input_tokens, resp.usage.output_tokens)
            else:
                ledger.record_call(0, 0)

        started = time.time()
        while True:
            remaining = 0 if force_now else cfg.cap - ledger.used(cfg.cap) - cfg.answer_tokens - cfg.margin
            if remaining < 512:
                # Not enough left for another turn: ask for the answer with thinking off.
                convo = items()
                if cfg.clean_final and history and _is_user_text(history[-1], CUT_NUDGE):
                    convo = convo[
                        :-1
                    ]  # the cut nudge asks for a code run; the final request must be the only instruction
                convo = convo + [NeMoGymEasyInputMessage(role="user", content=FINAL_NUDGE)]
                body = self._model_body(
                    base,
                    convo,
                    cfg.answer_tokens,
                    self._request_seed(task_id, sample, 0),
                    cfg.final_template_kwargs,
                    False,
                    "none",
                )
                resp, cookies = await self._call_model(body, cookies, url_path)
                record(resp)
                counters["forced_final"] = True
                history.append(NeMoGymEasyInputMessage(role="user", content=FINAL_NUDGE))
                add_model_output(resp)
                final_text = _message_text(resp)
                answer_source = "forced"
                break

            counters["turns"] += 1
            limit = min(turn_cap, remaining)
            body = self._model_body(
                base, items(), limit, self._request_seed(task_id, sample, counters["turns"]), thinking, True
            )
            resp, cookies = await self._call_model(body, cookies, url_path)
            record(resp)
            raw = _message_text(resp)
            add_model_output(resp)

            calls = [i for i in resp.output if getattr(i, "type", None) == "function_call"] if cfg.native_tools else []

            if _was_cut(resp, limit):  # cut: keep the turn, then tell the model to test or answer
                for call in calls:  # a call of a cut turn may be incomplete: answer it without running it
                    add_tool_output(call.call_id, "[harness] not run: the turn hit the per-turn token limit")
                add_user(CUT_NUDGE)
                counters["cuts"] += 1
                if counters["cuts"] > cfg.max_cuts:
                    ledger.exhausted = True
                continue

            if calls:
                auto_grid, refused = None, False
                for call in calls:
                    if counters["code_runs"] >= cfg.max_runs:
                        refused = True
                        add_tool_output(call.call_id, "[harness] not run: the limit of code runs is reached")
                        continue
                    counters["code_runs"] += 1
                    code, bad = _call_code(call)
                    if code is None:
                        add_tool_output(call.call_id, f"[error] {bad}")
                        continue
                    reply = await self._pool.run(header, codes, code)
                    if cfg.drop_timed_out and reply.get("timed_out"):
                        reply = dict(
                            reply,
                            error=reply.get("error", "")
                            + "; the cell was discarded, so the sandbox state is as it was before it",
                        )
                    else:
                        codes.append(code)
                    add_tool_output(call.call_id, _tool_output(reply))
                    if cfg.auto_answer and auto_grid is None and full_check_pass(reply.get("output", "")):
                        fn = checked_function(code)
                        auto_grid = await self._pool.predict_answer(header, codes, fn) if fn else None
                if auto_grid is not None:
                    final_text, answer_source = _answer_block(auto_grid), "auto"
                    history.append(_assistant_message(final_text, "auto-answer"))
                    break
                if cfg.runs_cap_final and refused:  # the model asked for a run beyond the cap
                    force_now = True
                continue

            code = None if cfg.native_tools else extract_code(raw, cfg.parse_tool_calls)
            if nocode_left > 0 and code is None and parse_answer(raw) is None:
                nocode_left -= 1
                add_user(NOCODE_NUDGE)
                continue
            if cfg.runs_cap_final and code is not None and counters["code_runs"] >= cfg.max_runs:
                force_now = True
                continue

            if code is None or counters["code_runs"] >= cfg.max_runs:
                # No code: this turn is the answer (or the attempt ends without one).
                final_text, answer_source = raw, "typed"
                why = unreadable_reason(raw) if parse_answer(raw) is None and resend_left > 0 else None
                if why:  # say what is wrong with the grid and ask again
                    resend_left -= 1
                    counters["resends"] += 1
                    add_user(RESEND_NUDGE.format(why=why))
                    final_text, answer_source = None, "none"
                    continue
                break

            reply = await self._pool.run(header, codes, code)
            counters["code_runs"] += 1
            if cfg.drop_timed_out and reply.get("timed_out"):
                reply = dict(
                    reply,
                    error=reply.get("error", "")
                    + "; the cell was discarded, so the sandbox state is as it was before it",
                )
            else:
                codes.append(code)
            tool_text = tool_response_text(reply)
            add_user(tool_text)

            if cfg.auto_answer and full_check_pass(reply.get("output", "")):
                fn = checked_function(code)
                grid = await self._pool.predict_answer(header, codes, fn) if fn else None
                if grid is not None:
                    # The checked function's prediction is the answer; recorded as a marked synthetic assistant turn.
                    final_text, answer_source = _answer_block(grid), "auto"
                    history.append(_assistant_message(final_text, "auto-answer"))
                    break

        return {
            "prompt": prompt,
            "history": history,
            "final_text": final_text,
            "usage": usage,
            "response": last_response,
            "cookies": cookies,
            "counters": counters,
            "answer_source": answer_source
            if final_text is not None and parse_answer(final_text) is not None
            else "none",
            "generated": ledger.generated,
            "used": ledger.used(cfg.cap),
            "prompt_tokens": ledger.first_prompt,
            "wall": time.time() - started,
        }

    # -- Gym entry point -----------------------------------------------------------------------------------------

    async def run(self, request: Request, body: ArcCodeAgentRunRequest) -> ArcCodeAgentVerifyResponse:
        if not is_grid(body.test_input) or not is_grid(body.expected_output) or not body.examples:
            raise ValueError("row needs `examples`, a `test_input` grid and an `expected_output` grid (digits 0-9)")
        for pair in body.examples:
            if not is_grid(pair.get("input")) or not is_grid(pair.get("output")):
                raise ValueError("every entry of `examples` needs `input` and `output` grids (digits 0-9)")
        extra = body.model_extra or {}
        sample = int(extra.get(ROLLOUT_INDEX_KEY_NAME, 0) or 0)
        url_path = self.url_path_for_run("/v1/responses", body)

        result = await self._attempt(
            body.responses_create_params.model_copy(deep=True), body, sample, request.cookies, url_path
        )
        last: Optional[NeMoGymResponse] = result["response"]
        if last is None:
            raise RuntimeError("ArcCodeAgent did not receive a model response")

        full = last.model_copy(update={"output": list(result["history"]), "usage": result["usage"]})
        # The verifier reads the last assistant message; give it just that one.
        final_msg = next(
            (
                i
                for i in reversed(result["history"])
                if getattr(i, "type", None) == "message" and getattr(i, "role", None) == "assistant"
            ),
            None,
        )
        verify_response_obj = last.model_copy(update={"output": [final_msg] if final_msg is not None else []})

        recorded_input = deepcopy([m.model_dump() for m in result["prompt"]])
        if not self.config.record_images:
            for message in recorded_input:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part.get("type") == "input_image":
                            part["image_url"] = "<image omitted>"
        verify_body = body.model_dump()
        verify_body["responses_create_params"] = {
            **body.responses_create_params.model_dump(exclude_unset=True),
            "input": recorded_input,
        }
        verify_body["response"] = verify_response_obj.model_dump()
        verify_request = ArcCodeAgentVerifyRequest.model_validate(verify_body)
        verify_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=result["cookies"] if result["cookies"] else request.cookies,
        )
        await raise_for_status(verify_resp)
        payload = await get_response_json(verify_resp)

        counters = result["counters"]
        payload["response"] = full.model_dump()
        payload["responses_create_params"] = verify_body["responses_create_params"]
        payload.update(
            turns=counters["turns"],
            code_runs=counters["code_runs"],
            cuts=counters["cuts"],
            resends=counters["resends"],
            forced_final=counters["forced_final"],
            generated_tokens=result["generated"],
            used_tokens=result["used"],
            prompt_tokens=result["prompt_tokens"],
            wall_seconds=round(result["wall"], 2),
            finished_with_answer=result["answer_source"] != "none",
            answer_source=result["answer_source"],
        )
        return ArcCodeAgentVerifyResponse.model_validate(payload)


def _call_code(call: Any) -> Any:
    """(code, None) for a call of the python tool, or (None, reason) when its name or arguments are unusable."""
    if call.name != PYTHON_TOOL["name"]:
        return None, f"unknown tool {call.name!r}; the only tool is `{PYTHON_TOOL['name']}`"
    try:
        args = json.loads(call.arguments)
    except (TypeError, ValueError):
        return None, 'the tool arguments are not valid JSON; pass an object like {"code": "..."}'
    code = args.get("code") if isinstance(args, dict) else None
    if not isinstance(code, str) or not code.strip():
        return None, "the tool arguments need a non-empty string field `code`"
    return code, None


def _tool_output(reply: Dict[str, Any]) -> str:
    """The sandbox output as a function result (the same text as the tool turn, without its wrapper)."""
    text = tool_response_text(reply)
    return text[len("<tool_response>\n") : -len("\n</tool_response>")]


def _is_user_text(item: Any, text: str) -> bool:
    return getattr(item, "role", None) == "user" and getattr(item, "content", None) == text


if __name__ == "__main__":
    ArcCodeAgent.run_webserver()
