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

"""Define run-wide training-token capture settings.

```yaml
env:
  nemo_gym:
    token_id_capture:
      enabled: true
      dir: /tmp/ng_tokcap                  # The writer and consumer share this node-local directory.
      sink: my_pkg.sinks:MyDataPlaneSink   # This optional sink replaces the file store.
```

Evaluation capture uses ``/ng-rollout/<id>/...``.
Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
Training capture records token ids and log probabilities.
Evaluation capture records request and response summaries.
A run can enable either path independently.
Training capture applies through the static agent flag or run-level ``all_agents``.
Native agents normally leave the static flag disabled.
Their responses already carry token ids.
The top-level ``model_call_capture_dir`` is the fallback file-store directory.

Choosing where records go
-------------------------
``sink`` names a class implementing ``TokenSink``, as ``module.path:ClassName``.
Each server process constructs its sink at app startup.
A framework must make that class importable in the server process.
A configured sink replaces the file store.
Consumers construct and inject their ``TokenSource`` in their own process.
Consumers call ``TokenSource.freeze`` to obtain an atomic snapshot.
Consumers retire that exact snapshot with its ``snapshot_id`` and version.
There is no HTTP token reader.
Uvicorn workers use spawned processes.
They do not inherit a sink installed by a launcher.
Configure the sink here so each worker builds its own.
Programmatic installation must occur inside the serving process.

Choosing who reads them back
----------------------------
``rebuild_response`` controls whether Gym rebuilds a finished rollout.
Gym then swaps the rebuilt trajectory into ``response.output``.
Set it to false when a framework reads through its own ``TokenSource``.
Gym then stops after the write.
Read ownership is independent of write ownership.
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.token_id_capture.protocols import (
    TokenSink,
    installed_token_sink,
)


logger = logging.getLogger(__name__)

TOKEN_ID_CAPTURE_BLOCK = "token_id_capture"


class TokenIdCaptureSettings(BaseModel):
    """The ``token_id_capture`` block."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Capture model calls from every agent.
    # The default keeps capture scoped by each agent's ``token_id_capture`` setting.
    all_agents: bool = False
    # Where the default file store writes.
    # Falls back to ``model_call_capture_dir``.
    dir: Path | None = None
    # ``module.path:ClassName`` implementing TokenSink, constructed per server process.
    sink: str | None = None
    # Keyword arguments for that constructor.
    # A real transport needs explicit endpoint, client, or credential wiring.
    # Use ``${oc.env:VAR}`` for secrets instead of writing them here.
    sink_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Rebuild opaque-harness responses from captured records after the run.
    rebuild_response: bool = True


class TokenIdCaptureConfig(BaseModel):
    """The capture block plus the one top-level key it falls back to."""

    model_config = ConfigDict(extra="ignore")

    token_id_capture: TokenIdCaptureSettings = TokenIdCaptureSettings()
    # Shared with evaluation capture, which owns it, so it stays top-level.
    model_call_capture_dir: Path | None = None

    @model_validator(mode="after")
    def _validate(self) -> "TokenIdCaptureConfig":
        block = self.token_id_capture
        if not block.enabled:
            # Keep inactive settings for templated configurations.
            # A run may toggle only ``enabled``.
            return self
        if block.sink is not None:
            if block.dir is not None:
                # The custom sink replaces the configured directory.
                # Warn because no files will appear there.
                logger.warning(
                    "token_id_capture.dir is set alongside token_id_capture.sink. The sink replaces "
                    "the file store, so %s will not be written to.",
                    block.dir,
                )
            return self
        directory = self.resolved_dir()
        if directory is None:
            # A programmatic sink replaces the file store.
            # That process does not need a directory.
            if installed_token_sink() is not None:
                return self
            if not block.rebuild_response:
                return self
            raise ValueError("token_id_capture requires a directory or sink")
        if not directory.is_absolute():
            raise ValueError("training-token capture directory must be an absolute path")
        return self

    @property
    def enabled(self) -> bool:
        return self.token_id_capture.enabled

    def resolved_dir(self) -> Path | None:
        return self.token_id_capture.dir or self.model_call_capture_dir

    def build_sink(self) -> TokenSink | None:
        """Construct the configured sink.

        Return ``None`` when the file store is in use.
        Call this once in each server process.
        Launcher-installed sinks do not reach spawned workers.
        """
        target = self.token_id_capture.sink
        if not self.token_id_capture.enabled or target is None:
            return None
        return self._build_endpoint(target, self.token_id_capture.sink_kwargs, TokenSink, "sink")

    @staticmethod
    def _build_endpoint(target: str, kwargs: dict[str, Any], protocol: type, kind: str):
        if ":" not in target:
            raise ValueError(f"token_id_capture.{kind} must be 'module.path:ClassName' (got {target!r})")
        module_path, _, class_name = target.partition(":")
        try:
            factory = getattr(import_module(module_path), class_name)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"could not load token_id_capture.{kind} {target!r}: {error}") from error
        try:
            endpoint = factory(**kwargs)
        except TypeError as error:
            raise ValueError(
                f"could not construct token_id_capture.{kind} {target!r} with {kind}_kwargs={sorted(kwargs)}: {error}"
            ) from error
        # Validate the endpoint at startup.
        # A missing lifecycle method can make incomplete capture look complete.
        missing = [name for name in sorted(protocol.__protocol_attrs__) if not callable(getattr(endpoint, name, None))]
        if missing or not isinstance(endpoint, protocol):
            raise ValueError(
                f"token_id_capture.{kind} {target!r} does not satisfy {protocol.__name__}: "
                f"{', '.join(missing) or 'attribute check failed'}"
            )
        return endpoint


def token_id_capture_config(global_config_dict: Any) -> TokenIdCaptureConfig:
    """Read the capture settings out of a global config dict."""
    return TokenIdCaptureConfig.model_validate(global_config_dict or {})
