# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from runpy import run_path

import pytest

from resources_servers.gui_coordinate.app import (
    GuiCoordinateResourcesServer,
    GuiCoordinateResourcesServerConfig,
    GuiCoordinateVerifyRequest,
)


assert_stateless_checkpoint_recovery = run_path(
    str(Path(__file__).resolve().parents[3] / "tests/helpers/stateless_checkpoint.py")
)["assert_stateless_checkpoint_recovery"]


@pytest.mark.parametrize(
    ("answer", "reward"),
    [("<point>(500,250)</point>", 1.0), ("<point>(900,900)</point>", 0.0), ("invalid", 0.0)],
)
async def test_checkpoint_replays_verification(tmp_path: Path, answer: str, reward: float) -> None:
    request = GuiCoordinateVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Point to the target."},
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
            "expected_answer": "0.5,0.25",
            "metadata": {"target_color": "yellow"},
        }
    )
    await assert_stateless_checkpoint_recovery(
        GuiCoordinateResourcesServer,
        GuiCoordinateResourcesServerConfig(host="localhost", port=0, entrypoint="app.py", name="gui_coordinate"),
        request,
        tmp_path,
        expected_reward=reward,
    )
