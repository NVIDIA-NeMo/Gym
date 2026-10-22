# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VideoMME-v1 single-turn and agentic evaluation resources."""

from typing import ClassVar

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.video_mcqa.app import VideoMCQAResourcesServer
from resources_servers.video_mcqa.dataset import Benchmark


class VideoMMEV1ResourcesServer(VideoMCQAResourcesServer):
    """Overall, duration, domain, sub-category and task-type accuracy."""

    benchmark: ClassVar[Benchmark] = "videomme_v1"


if __name__ == "__main__":
    VideoMMEV1ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VideoMMEV1ResourcesServer.run_webserver()
