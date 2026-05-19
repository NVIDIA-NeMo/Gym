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

"""Lightweight mock model server for K8s integration tests.

Returns a canned NeMoGymResponse on POST /v1/responses.
No nemo_gym imports — avoids heavy init (Ray, OmegaConf) and keeps
the mock decoupled from internal schema evolution.
"""

import uvicorn
from fastapi import FastAPI, Request


app = FastAPI()

CANNED_RESPONSE = {
    "id": "resp_mock_001",
    "created_at": 1700000000.0,
    "model": "mock-model",
    "object": "response",
    "output": [
        {
            "id": "msg_mock_001",
            "content": [
                {
                    "annotations": [],
                    "text": "The weather is sunny.",
                    "type": "output_text",
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 15,
    },
}


@app.post("/v1/responses")
async def responses(request: Request):
    await request.body()
    return CANNED_RESPONSE


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
