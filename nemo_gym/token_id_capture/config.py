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

"""Run-wide settings for training-token capture, in one block.

```yaml
env:
  nemo_gym:
    token_id_capture:
      enabled: true
      dir: /tmp/ng_tokcap                  # node-local; writer and reader share a node
      sink: my_pkg.sinks:MyDataPlaneSink   # optional; default is the file store at `dir`
```

This is a separate switch from evaluation capture (``observability_enabled``).
Evaluation capture records a compact request/response summary; training-token
capture records token ids and log probabilities for RL. A run can enable either,
both, or neither. When no ``dir`` is given, tokens are written alongside the eval
capture files in the top-level ``model_call_capture_dir``.

The per-agent ``token_id_capture`` flag is a narrower, separate control: it scopes
which agents participate. Native agents leave it off because they already carry
token ids on their response items.

Choosing where records go
-------------------------
``sink`` names a class implementing ``TokenSink``, as ``module.path:ClassName``.
It is constructed once per server process at app startup and replaces the file
store, so records go to a framework's own transport and never touch disk.

Construction has to happen inside the serving process. A model server configured
with ``num_workers > 1`` is launched by uvicorn with an app string and
``workers=N``, and uvicorn spawns those workers with the ``spawn`` start method,
which re-imports the app module rather than inheriting the parent's memory. A
sink installed by a launcher script therefore does not exist in any worker, and
capture silently falls back to the file store, or writes nothing at all when no
``dir`` is set. Naming the sink here avoids that: each worker builds its own.

``install_token_sink`` remains for programmatic use and is subject to the same
constraint, so call it at module import of the app, not from a parent process.
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.token_id_capture.protocols import (
    TokenSink,
    TokenSource,
    installed_token_sink,
    installed_token_source,
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
    # Where the default file store writes. Falls back to ``model_call_capture_dir``.
    dir: Path | None = None
    # ``module.path:ClassName`` implementing TokenSink, constructed per server process.
    sink: str | None = None
    # Keyword arguments for that constructor.
    # A real transport needs explicit endpoint, client, or credential wiring.
    # Use ``${oc.env:VAR}`` for secrets instead of writing them here.
    sink_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Optional paired reader for framework-owned transports.
    source: str | None = None
    source_kwargs: dict[str, Any] = Field(default_factory=dict)
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
            # The rest of the block is left alone rather than rejected. Configs are templated, and
            # setting a directory unconditionally while toggling `enabled` per run is ordinary.
            return self
        if block.sink is not None:
            if block.dir is not None:
                # Not an error: nothing is lost, the directory is simply never read. Worth saying
                # once, because someone expecting files on disk will not find any.
                logger.warning(
                    "token_id_capture.dir is set alongside token_id_capture.sink. The sink replaces "
                    "the file store, so %s will not be written to.",
                    block.dir,
                )
            if block.rebuild_response and block.source is None and installed_token_source() is None:
                raise ValueError(
                    "token_id_capture.source is required when a custom sink is used with rebuild_response=true"
                )
            return self
        directory = self.resolved_dir()
        if directory is None:
            # A process that installed a sink programmatically writes through that transport and
            # never constructs the file store, so it has no directory to give.
            if installed_token_sink() is not None and (
                not block.rebuild_response or installed_token_source() is not None
            ):
                return self
            if block.source is not None:
                return self
            if not block.rebuild_response:
                return self
            raise ValueError("token_id_capture requires a directory or paired source when rebuild_response=true")
        if not directory.is_absolute():
            raise ValueError("training-token capture directory must be an absolute path")
        return self

    @property
    def enabled(self) -> bool:
        return self.token_id_capture.enabled

    def resolved_dir(self) -> Path | None:
        return self.token_id_capture.dir or self.model_call_capture_dir

    def build_sink(self) -> TokenSink | None:
        """Construct the configured sink, or ``None`` when the file store is in use.

        Called once per server process at app startup, which is what makes this work under
        ``num_workers > 1`` where a sink installed by a launcher does not reach the workers.
        """
        target = self.token_id_capture.sink
        if not self.token_id_capture.enabled or target is None:
            return None
        return self._build_endpoint(target, self.token_id_capture.sink_kwargs, TokenSink, "sink")

    def build_source(self) -> TokenSource | None:
        """Construct a configured framework-owned source."""
        target = self.token_id_capture.source
        if not self.token_id_capture.enabled or target is None:
            return None
        return self._build_endpoint(target, self.token_id_capture.source_kwargs, TokenSource, "source")

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
        # Checked here rather than at first use: an endpoint missing a lifecycle method makes an
        # incomplete rollout look complete, and a startup error is better than that at step 400.
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
