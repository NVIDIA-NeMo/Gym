# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The declared shape of what a processor hands to both sides of an episode."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from nemo_gym.config_types import ResourcesServerRef


EPISODE_CONTEXT_KEY = "episode_context"


class EpisodeContext(BaseModel):
    """One producer, two consumers: the processor builds it, the harness and the verifier read it.

    Declared on the base request models so every resources server and every agent inherits the
    field instead of each declaring its own, which is what makes `extra="forbid"` meaningful:
    a subclass that never declares a field silently drops it under `extra="ignore"`.

    Replaces the untyped `sandbox_handle` key that crosses a server boundary today, and the
    `resources_server` reference an agent server used to carry in its own config.
    """

    model_config = ConfigDict(extra="forbid")

    # Optional because rollout correlation is only enabled with observability or token capture.
    rollout_id: Optional[str] = None

    # Where the harness calls `/<tool_name>`. Set by the processor, so the harness needs no
    # environment in its own config.
    env: Optional[ResourcesServerRef] = None

    # `AsyncSandbox.serialize()`; rebuilt with `AsyncSandbox.connect()`. Absent when the
    # benchmark declares no runtime.
    sandbox: Optional[dict[str, Any]] = None
