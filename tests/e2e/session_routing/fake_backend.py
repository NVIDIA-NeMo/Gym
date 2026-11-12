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
"""A deterministic OpenAI-compatible backend for the multi-worker session e2e test.

Each completion emits the next scripted counter increment (by 1, then by 2), then a final answer.
"""

import json
import sys
import time

import uvicorn
from fastapi import FastAPI, Request


SCRIPT = [1, 2]

app = FastAPI()


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request) -> dict:
    body = await request.json()
    step = sum(message["role"] == "tool" for message in body["messages"])
    if step < len(SCRIPT):
        tool_call = {
            "id": f"call_{step}",
            "type": "function",
            "function": {"name": "increment_counter", "arguments": json.dumps({"count": SCRIPT[step]})},
        }
        message, finish = {"role": "assistant", "content": None, "tool_calls": [tool_call]}, "tool_calls"
    else:
        message, finish = {"role": "assistant", "content": "Done."}, "stop"
    return {
        "id": f"chatcmpl-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "fake-model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
