# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from types import SimpleNamespace

from resources_servers.cube.server import CubeResourcesServer


def test_close_all_open_environments_closes_and_clears():
    class StubTask:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    a, b = StubTask(), StubTask()
    srv = SimpleNamespace(
        env_id_to_task={"e1": a, "e2": b},
        env_id_to_total_reward=defaultdict(float, {"e1": 0.5, "e2": 0.25}),
    )

    CubeResourcesServer.close_all_open_environments(srv)

    assert a.closed and b.closed
    assert srv.env_id_to_task == {}
    assert dict(srv.env_id_to_total_reward) == {}
