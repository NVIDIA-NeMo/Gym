# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from runpy import run_path

import pytest

from resources_servers.sav_tracks.app import (
    SavTracksResourcesServer,
    SavTracksResourcesServerConfig,
    SavTracksVerifyRequest,
)


assert_stateless_checkpoint_recovery = run_path(
    str(Path(__file__).resolve().parents[3] / "tests/helpers/stateless_checkpoint.py")
)["assert_stateless_checkpoint_recovery"]


@pytest.mark.parametrize("task", ["pt", "box", "ref"])
@pytest.mark.parametrize("valid", [True, False], ids=["correct", "malformed"])
async def test_checkpoint_replays_verification(tmp_path: Path, task: str, valid: bool) -> None:
    id_key = {"pt": "point", "box": "box", "ref": "id"}[task]
    location = {"bbox": [250, 250, 750, 750]} if task == "box" else {"xy": [500, 500]}
    target = {"bbox": [250, 250, 750, 750]} if task == "box" else {"mask": {"size": [2, 2], "counts": [0, 4]}}
    tracks = [{id_key: 7 if task == "ref" else 1, "frame": 1, "visible": True, **location}]
    answer = f"<answer>{json.dumps({'tracks': tracks})}</answer>" if valid else "invalid"
    request = SavTracksVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Track the object in frame 1."},
            "response": {
                "id": "response",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [
                    {
                        "id": "message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": answer, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "task": task,
            "objects": [{"id": 1, "targets": [{"frame": 1, "visible": True, **target}]}],
            "source_id": "checkpoint-fixture",
        }
    )
    await assert_stateless_checkpoint_recovery(
        SavTracksResourcesServer,
        SavTracksResourcesServerConfig(host="localhost", port=0, entrypoint="app.py", name="sav_tracks"),
        request,
        tmp_path,
        expected_reward=float(valid),
    )
