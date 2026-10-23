# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored guard policies for built-in tools, shared by import and execution contracts."""

import re
from collections.abc import Collection
from typing import Literal

from pydantic import RootModel, model_validator


class ReadFirstGuard(RootModel[dict[Literal["edit", "write"], dict[Literal["unread", "changed"], str]]]):
    """Optional per-tool refusals, with the authored path placeholder kept literal."""

    @model_validator(mode="after")
    def validate_messages(self) -> "ReadFirstGuard":
        if not self.root or any(not cases for cases in self.root.values()):
            raise ValueError("read_first requires non-empty tool/case objects")
        for cases in self.root.values():
            for message in cases.values():
                if not message.strip():
                    raise ValueError("read_first messages must not be blank")
                if set(re.findall(r"\{\{(\w+)\}\}", message)) - {"path"}:
                    raise ValueError("read_first messages support only the {{path}} placeholder")
        return self

    def validate_tools(self, selected: Collection[str]) -> None:
        """Reject configurations whose execution path cannot enforce the guard."""
        missing = ({"read"} | self.root.keys()) - set(selected)
        if missing:
            raise ValueError(f"read_first requires exposed built-in tools: {sorted(missing)}")
