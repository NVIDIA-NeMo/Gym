# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The model server's read route for captured training tokens.

The route is registered on a model server only when training-token capture is
enabled, so a default run exposes nothing. It lets a non-co-located trainer pull
a rollout's ``TokenEntry`` records over HTTP (see ``HttpTokenReader``) instead of
requiring shared-filesystem access to the store.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Response

from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.store import TokenCaptureStore


def make_token_store(global_config_dict: Any) -> Optional[TokenCaptureStore]:
    """Build the training-token store, or ``None`` when capture is disabled."""
    config = TokenIdCaptureConfig.model_validate(global_config_dict)
    if not config.token_id_capture_enabled:
        return None
    return TokenCaptureStore(config.resolved_dir())


def install_token_capture_routes(app: Any, store: TokenCaptureStore) -> None:
    router = APIRouter()

    @router.get("/ng-capture/tokens/{rollout_id}")
    async def get_tokens(rollout_id: str) -> Response:
        try:
            entries = await asyncio.to_thread(store.read_entries, rollout_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        body = "\n".join(entry.model_dump_json() for entry in entries)
        return Response(content=body, media_type="application/x-ndjson")

    app.include_router(router)
