# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist mini-SWE decisions before a tool can terminate its sandbox."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from minisweagent.agents.default import DefaultAgent


class CheckpointAgent(DefaultAgent):
    def query(self) -> dict:
        """Save the actual model response before dispatching its actions."""
        message = super().query()
        self.save(self.config.output_path)
        return message

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Keep the previous complete checkpoint if a write is interrupted."""
        if path is None:
            return super().save(path, *extra_dicts)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            data = super().save(temporary, *extra_dicts)
            os.replace(temporary, path)
            return data
        finally:
            temporary.unlink(missing_ok=True)
