# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harnesses a row can name with `verifier_metadata.harness`, and how a row selects one. `GENERIC` (in `base`)
is the default; opencode.py holds opencode and its fork Kilo Code."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from resources_servers.agentic_turn_simulation.harnesses.base import GENERIC, Harness
from resources_servers.agentic_turn_simulation.harnesses.claude_code import CLAUDE_CODE
from resources_servers.agentic_turn_simulation.harnesses.cline import CLINE
from resources_servers.agentic_turn_simulation.harnesses.hermes import HERMES
from resources_servers.agentic_turn_simulation.harnesses.openclaw import OPENCLAW
from resources_servers.agentic_turn_simulation.harnesses.opencode import KILO_CODE, OPENCODE
from resources_servers.agentic_turn_simulation.harnesses.sandbox import SANDBOX


LOG = logging.getLogger(__name__)

HARNESSES: dict[str, Harness] = {
    h.name: h for h in (SANDBOX, GENERIC, CLAUDE_CODE, HERMES, OPENCLAW, OPENCODE, KILO_CODE, CLINE)
}
# Harness names that rows may give for variants of a harness listed here.
ALIASES = {"oh_my_opencode": "opencode"}

_reported_unknown: set[str] = set()


def select_harness(metadata: Mapping[str, Any] | None) -> Harness:
    """The row's harness: Cline's for a row whose `protocol` is Cline's, else the one its `harness` names; generic
    without one, or for a harness not listed here (logged once per name)."""
    metadata = metadata or {}
    if metadata.get("protocol") == CLINE.protocol:
        return CLINE
    name = metadata.get("harness")
    if name is None:
        return GENERIC
    harness = HARNESSES.get(ALIASES.get(name, name)) if isinstance(name, str) else None
    if harness is None:
        key = json.dumps(name, default=str)
        if key not in _reported_unknown:
            _reported_unknown.add(key)
            LOG.warning("unknown harness=%s; using %s", key, GENERIC.name)
        return GENERIC
    return harness
