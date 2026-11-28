# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local verifier entrypoint, run only in the separate verifier sandbox."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def verify(workspace: Path, task_data: dict[str, Any], response: dict[str, Any] | None) -> dict[str, float]:
    """Grade collected deliverables using trusted task data and the agent-server response."""
    sys.path.insert(0, str(Path(__file__).parent))
    import legal_agent_bench_verify as verifier

    with tempfile.TemporaryDirectory(prefix="lab-verification-") as temporary:
        run = Path(temporary)
        output = workspace / "output"
        if output.is_dir():
            shutil.copytree(output, run / "output")
        else:
            (run / "output").mkdir()
        response = response or {}
        metadata = response.get("metadata") or {}
        config = {"task": task_data["instance_id"], "model": response.get("model")}
        config.update(json.loads(metadata.get("lab_harness_provenance", "{}")))
        (run / "config.json").write_text(json.dumps(config))
        metrics = {
            **(response.get("usage") or {}),
            "usage": response.get("usage") or {},
            "finish_reason": metadata.get("lab_finish_reason"),
            "status": response.get("status"),
        }
        (run / "metrics.json").write_text(json.dumps(metrics))
        args = verifier.build_parser().parse_args(
            [
                "--task-json",
                str(Path(__file__).with_name("task.json")),
                "--run-dir",
                str(run),
                "--verifier-dir",
                "/logs/verifier",
                "--reward-json",
                "/logs/verifier/reward.json",
            ]
        )
        try:
            verifier.run_verifier(args)
        except Exception as error:
            verifier.write_failure_outputs(args, error)
        return json.loads(Path(args.reward_json).read_text())
