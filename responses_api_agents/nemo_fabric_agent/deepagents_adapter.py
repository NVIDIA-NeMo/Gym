# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate DeepAgents' filesystem root from its execution workspace."""

from copy import deepcopy
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.deepagents.adapter import DeepAgentsRuntime as FabricDeepAgentsRuntime


class DeepAgentsRuntime(FabricDeepAgentsRuntime):
    """Honor an explicit filesystem root without changing the SDK child's cwd."""

    async def start(self, payload: dict[str, Any]) -> None:
        harness = payload["config"].harness
        root = harness.settings.get("filesystem_root") if harness else None
        if root is not None:
            # The native adapter derives its backend only from this context field.
            payload = deepcopy(payload)
            payload["runtime_context"]["environment"]["workspace"] = root
        await super().start(payload)


if __name__ == "__main__":
    lifecycle.serve(DeepAgentsRuntime, config_loader=contract.AgentConfig.from_mapping)
