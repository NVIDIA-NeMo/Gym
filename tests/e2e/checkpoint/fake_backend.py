# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic OpenAI-compatible inference backend for the checkpoint e2e suite.

It stands in for a vLLM worker and for the training framework's durable token store, both of which outlive a Gym crash.

Turn policy: step ``n`` (the number of tool results so far) emits the ``n``-th scripted tool call, then a final answer.
The default script is one ``get_weather`` call.

Control routes:

- ``POST /_ctl/hold {"after_calls": n}``: every call after the first ``n`` waits until released.
- ``POST /_ctl/release``: release held calls, and stop gating.
- ``POST /_ctl/gate``: from now on every call waits until a step releases it.
- ``POST /_ctl/step {"calls": n}``: release the ``n`` longest-waiting gated calls; returns how many were released.
- ``POST /_ctl/script {"tool_calls": [...]}``: set the scripted tool calls.
- ``GET /_ctl/calls``: every request received, with its message count and capture admission.
- ``GET /_ctl/cuts``: the model call IDs cut by each generation-cut round.
"""

import asyncio
import json
import sys
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request

from nemo_gym._checkpoint.generation_cut import GenerationCutInventory, GenerationCutPrefixAck, GenerationCutReceipt
from nemo_gym.token_id_capture.staging import CaptureAdmission, StagedCallRecord, StageResult
from nemo_gym.token_id_capture.staging.capture import ActiveCall, RolloutTokenCapture


DEFAULT_SCRIPT = [{"name": "get_weather", "arguments": {"city": "SF"}}]


class FakeTransferQueue:
    """Durable token rows and generation-cut prefix records."""

    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}
        self.cuts: dict[str, StagedCallRecord] = {}

    def stage(self, record: Any, *, attachments: Any = None) -> StageResult:
        key = f"{record.rollout_id}/{record.model_call_id}"
        self.rows[key] = record
        return StageResult(ok=True, staging_key=key)


TRANSFER_QUEUE = FakeTransferQueue()
CAPTURE = RolloutTokenCapture(sink=TRANSFER_QUEUE, weight_version_fn=lambda: 7)
# Calls that are generating, with their prompt and the tokens produced so far: a checkpoint may cut them.
ACTIVE: dict[str, tuple[ActiveCall, list[int], list[int]]] = {}
STATE: dict[str, Any] = {
    "calls": [],
    "cuts": [],
    "hold_after": None,
    "released": asyncio.Event(),
    "gated": False,
    # Gated calls waiting for a step, oldest first.
    "waiting": [],
}
SCRIPT: list[dict[str, Any]] = list(DEFAULT_SCRIPT)

app = FastAPI()


def _tokens(text: str) -> list[int]:
    return [(ord(ch) % 250) + 1 for ch in text][:64] or [1]


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}


@app.post("/_ctl/hold")
async def hold(body: dict) -> dict:
    STATE["hold_after"] = body["after_calls"]
    STATE["released"] = asyncio.Event()
    return {"ok": True}


@app.post("/_ctl/release")
async def release() -> dict:
    STATE["hold_after"] = None
    STATE["released"].set()
    STATE["gated"] = False
    waiting, STATE["waiting"] = STATE["waiting"], []
    for event in waiting:
        event.set()
    return {"ok": True}


@app.post("/_ctl/gate")
async def gate() -> dict:
    STATE["gated"] = True
    return {"ok": True}


@app.post("/_ctl/step")
async def step(body: dict) -> dict:
    released = STATE["waiting"][: body["calls"]]
    del STATE["waiting"][: body["calls"]]
    for event in released:
        event.set()
    return {"released": len(released), "waiting": len(STATE["waiting"])}


@app.post("/_ctl/script")
async def script(body: dict) -> dict:
    SCRIPT[:] = body["tool_calls"]
    return {"ok": True}


@app.get("/_ctl/calls")
async def calls() -> list:
    return STATE["calls"]


@app.get("/_ctl/cuts")
async def cuts() -> list:
    return STATE["cuts"]


@app.post("/ng-control/v1/generation-cut")
async def generation_cut(inventory: dict) -> dict:
    typed = GenerationCutInventory.model_validate(inventory)
    acks = []
    for prefix in typed.active_prefixes:
        active = ACTIVE.get(prefix.model_call_id)
        if active is None:
            acks.append(GenerationCutPrefixAck.failure(prefix))
            continue
        call, prompt, partial = active
        # Stage the prefix as a real record while the call keeps decoding, as a real worker does.
        record = CAPTURE.build_prefix_record(
            call, prompt_token_ids=prompt, generated_token_ids=partial, generated_logprobs=[-0.1] * len(partial)
        )
        key = f"__generation_cut__/{prefix.rollout_id}/{prefix.model_call_id}"
        TRANSFER_QUEUE.cuts[key] = record
        acks.append(
            GenerationCutPrefixAck(
                **prefix.model_dump(),
                disposition="durable_prefix",
                cut_kind="active_prefix",
                frozen_buffer_id=uuid.uuid4().hex,
                staging_keys=(key,),
                prefix_token_count=len(partial),
                prefix_digest=record.digest,
                effective_output_limit=1024,
            )
        )
    STATE["cuts"].append([ack.model_call_id for ack in acks if ack.disposition == "durable_prefix"])
    return GenerationCutReceipt(
        checkpoint_id=typed.checkpoint_id,
        cut_id=uuid.uuid4().hex,
        inventory_digest=typed.inventory_digest,
        inventory=typed,
        backend_snapshot_id="fake-transfer-queue",
        prefixes=tuple(acks),
    ).model_dump(mode="json")


def _next_message(messages: list[dict], index: int) -> tuple[dict, str]:
    step = sum(message["role"] == "tool" for message in messages)
    if step >= len(SCRIPT):
        return {"role": "assistant", "content": "It is cold in SF."}, "stop"
    tool_call = {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": SCRIPT[step]["name"], "arguments": json.dumps(SCRIPT[step]["arguments"])},
    }
    return {"role": "assistant", "content": None, "tool_calls": [tool_call]}, "tool_calls"


@app.post("/v1/chat/completions")
async def chat(request: Request) -> dict:
    body = await request.json()
    messages = body["messages"]
    index = len(STATE["calls"])
    admission_payload = body.get("ng_capture")
    call = {"index": index, "n_messages": len(messages), "admission": admission_payload, "t": time.time()}
    STATE["calls"].append(call)

    message, finish = _next_message(messages, index)
    capture = None
    if admission_payload is not None:
        admission = CaptureAdmission.model_validate(admission_payload)
        prefix = None
        if admission.mode == "token_in":
            prefix = [token for key in admission.staging_chain for token in TRANSFER_QUEUE.rows[key].token_ids_delta]
        full = _tokens(json.dumps(message))
        cut = admission.generation_cut
        if cut is None:
            capture_call = CAPTURE.begin_call(admission, prefix_token_ids=prefix)
            prompt = (prefix or []) + _tokens(json.dumps(messages[-1]))
            remaining = full
        else:
            # Continue a cut prefix. begin_call validates it against the admission Gym attached: staging keys,
            # source call, digest, lineage, and generated-token count.
            [snapshot] = [TRANSFER_QUEUE.cuts[key] for key in cut.staging_keys]
            capture_call = CAPTURE.begin_call(
                admission,
                prefix_token_ids=prefix,
                generation_cut=snapshot,
                generation_cut_staging_keys=cut.staging_keys,
            )
            prompt = capture_call.prefix_token_ids + list(snapshot.token_ids_delta)
            remaining = full[cut.prefix_token_count :]
            call["continued_from"] = cut.source_model_call_id
            call["reused_prefix_tokens"] = cut.prefix_token_count
        capture = (capture_call, prompt, remaining)
        # A held call has generated part of its answer; a checkpoint may cut this prefix.
        ACTIVE[admission.model_call_id] = (capture_call, prompt, remaining[:5])
    hold_after = STATE["hold_after"]
    if hold_after is not None and index >= hold_after:
        await STATE["released"].wait()
    if STATE["gated"]:
        event = asyncio.Event()
        STATE["waiting"].append(event)
        await event.wait()

    coords = None
    if capture is not None:
        capture_call, prompt, remaining = capture
        ACTIVE.pop(capture_call.model_call_id, None)
        coords = CAPTURE.complete_call(
            capture_call,
            prompt_token_ids=prompt,
            generated_token_ids=remaining,
            generated_logprobs=[-0.1] * len(remaining),
        ).model_dump(mode="json")

    response = {
        "id": f"chatcmpl-{index}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "fake-model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    if coords is not None:
        response["ng_commit_coords"] = coords
    call["returned"] = True
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
