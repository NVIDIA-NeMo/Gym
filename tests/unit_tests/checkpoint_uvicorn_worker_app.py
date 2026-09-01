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
"""Importable two-worker Uvicorn app for checkpoint coordinator tests."""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from nemo_gym.checkpoint import AdmissionLimiter, AdmissionMiddleware, WorkerAdmissionAgent


limiter = AdmissionLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = WorkerAdmissionAgent(
        Path(os.environ["NG_CHECKPOINT_COORDINATOR_SOCKET"]),
        worker_id=str(os.getpid()),
        limiter=limiter,
        pid=os.getpid(),
    )
    await agent.start()
    try:
        yield
    finally:
        await agent.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/pid")
async def pid() -> dict[str, int]:
    return {"pid": os.getpid()}


@app.post("/hold")
async def hold() -> dict[str, int]:
    release_path = Path(os.environ["NG_CHECKPOINT_RELEASE_PATH"])
    while not release_path.exists():
        await asyncio.sleep(0.01)
    return {"pid": os.getpid()}


app.add_middleware(AdmissionMiddleware, limiter=limiter, gated_suffixes=("/hold",))
