# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources-owned repository state, persisted outside candidate sandboxes."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from uuid import uuid4

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.utils import read_text, upload_text
from resources_servers.swe_together.diff_protocol import _strip_junk, truncate_diff_for_user_sim


@dataclass
class RepositorySnapshots:
    directory: Path
    python_executable: str = "python3"
    namespace: str = field(default_factory=lambda: "refs/nemo-gym/" + uuid4().hex)
    baseline: dict[str, str] = field(default_factory=dict)
    previous: dict[str, str] = field(default_factory=dict)
    final_patch: str = ""

    async def capture(self, sandbox: AsyncSandbox, turn: int | None = None) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        remote = f"/tmp/gym-swet-snapshot-{uuid4().hex}"
        await sandbox.upload(Path(__file__).with_name("snapshot_worker.py"), remote + ".py")
        await upload_text(
            sandbox,
            path=remote + ".json",
            text=json.dumps({"baseline": self.baseline, "previous": self.previous, "namespace": self.namespace}),
        )
        try:
            result = await sandbox.exec(
                f"{quote(self.python_executable)} {quote(remote + '.py')} {quote(remote + '.json')} {quote(remote + '.out')}",
                timeout_s=180,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Repository snapshot failed: {result.stderr}")
            raw = await read_text(sandbox, path=remote + ".out")
        finally:
            await sandbox.exec(
                "rm -f -- " + " ".join(quote(remote + suffix) for suffix in [".py", ".json", ".out"]), timeout_s=15
            )
        record = json.loads(raw)
        (self.directory / ("baseline.json" if turn is None else f"turn-{turn}.json")).write_text(raw)
        if not self.baseline:
            self.baseline = record["trees"]
        self.previous = record["trees"]
        projections = {
            kind: "\n".join(
                f"=== {repo} ({kind} vs {'harbor-base' if kind == 'cumulative' or not turn else f'harbor-turn-{turn - 1}'}) ===\n"
                + data[kind]
                for repo, data in record["repositories"].items()
            )
            for kind in ["cumulative", "incremental"]
        }
        self.final_patch = _strip_junk(projections["cumulative"])
        # Banners alone are not submissions or meaningful progress observations.
        if "diff --git " not in self.final_patch:
            self.final_patch = ""
        incremental = _strip_junk(projections["incremental"])
        if turn is not None:
            (self.directory / f"turn-{turn}.patch").write_text(self.final_patch + "\n")
            (self.directory / f"turn-{turn}.incremental.patch").write_text(incremental + "\n")
            (self.directory / "final.patch").write_text(self.final_patch + "\n")
        return truncate_diff_for_user_sim(incremental)
