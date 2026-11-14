# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LVBench single-turn and agentic evaluation resources (not LongVideoBench)."""

from typing import ClassVar

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.video_mcqa.app import VideoMCQAResourcesServer
from resources_servers.video_mcqa.dataset import Benchmark


class LVBenchResourcesServer(VideoMCQAResourcesServer):
    """Overall and overlapping capability accuracy for THUDM/LVBench."""

    benchmark: ClassVar[Benchmark] = "lvbench"


if __name__ == "__main__":
    LVBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = LVBenchResourcesServer.run_webserver()
