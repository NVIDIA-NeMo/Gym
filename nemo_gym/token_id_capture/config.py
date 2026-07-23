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

"""Run-wide switch for training-token capture.

This is a separate switch from evaluation capture (``observability_enabled``).
Evaluation capture records a compact request/response summary; training-token
capture records token ids and log probabilities for RL. A run can enable either,
both, or neither. When no dedicated directory is given, tokens are written
alongside the eval capture files in ``model_call_capture_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class TokenIdCaptureConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token_id_capture_enabled: bool = False
    token_id_capture_dir: Optional[Path] = None
    # Shared fallback directory (also used by evaluation capture).
    model_call_capture_dir: Optional[Path] = None

    @model_validator(mode="after")
    def _validate(self) -> "TokenIdCaptureConfig":
        if not self.token_id_capture_enabled:
            return self
        directory = self.resolved_dir()
        if directory is None:
            raise ValueError(
                "token_id_capture_dir (or model_call_capture_dir) is required when token_id_capture_enabled=true"
            )
        if not directory.is_absolute():
            raise ValueError("training-token capture directory must be an absolute path")
        return self

    def resolved_dir(self) -> Optional[Path]:
        return self.token_id_capture_dir or self.model_call_capture_dir
