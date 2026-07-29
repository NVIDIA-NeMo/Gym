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

Turning the file store off
--------------------------
A training framework that stages tokens through its own transport does not want
Gym writing files as well. There are two levels:

- ``token_id_capture_enabled: false`` disables capture entirely. No records are
  produced anywhere, and an external harness then yields rollouts with no token
  ids, which is not trainable. Use this only for evaluation runs.
- Leave capture enabled and install a different sink. ``install_token_sink``
  replaces the destination for the whole process, so records go to the
  framework's transport and Gym's store is never constructed. This is the
  configuration to use when the framework owns the inference worker: the model
  server still assembles the record, but the bulk token arrays never touch disk
  and never ride back through an HTTP response.

The per-agent ``token_id_capture`` flag is a third, narrower control: it scopes
which agents participate. Native agents leave it off because they already carry
token ids on their response items.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class TokenIdCaptureConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token_id_capture_enabled: bool = False
    token_id_capture_dir: Path | None = None
    # Shared fallback directory (also used by evaluation capture).
    model_call_capture_dir: Path | None = None

    # Bearer token required by the token read route. The route serves a rollout's raw training
    # tokens on the same app the harness calls to generate, which is fine inside a trusted cluster
    # and not fine once the harness runs in a sandbox whose only egress is this server -- it could
    # read its own training data, or another rollout's. Set this before the sandbox work; when unset
    # the route stays open and logs a warning once.
    token_id_capture_read_token: str | None = None

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

    def resolved_dir(self) -> Path | None:
        return self.token_id_capture_dir or self.model_call_capture_dir
