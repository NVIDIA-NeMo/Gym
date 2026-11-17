# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-exports of the shared video MCQA task contract (video_mcqa)."""

from resources_servers.video_mcqa.task import (
    LABELS as LABELS,
)
from resources_servers.video_mcqa.task import (
    InputText as InputText,
)
from resources_servers.video_mcqa.task import (
    InputVideo as InputVideo,
)
from resources_servers.video_mcqa.task import (
    PreparedTask as PreparedTask,
)
from resources_servers.video_mcqa.task import (
    ResponsesCreateParams as ResponsesCreateParams,
)
from resources_servers.video_mcqa.task import (
    Setting as Setting,
)
from resources_servers.video_mcqa.task import (
    UserMessage as UserMessage,
)
from resources_servers.video_mcqa.task import (
    VerifierMetadata as VerifierMetadata,
)
from resources_servers.video_mcqa.task import (
    VideoQuestion as VideoQuestion,
)
from resources_servers.video_mcqa.task import (
    prepare_agentic as prepare_agentic,
)
from resources_servers.video_mcqa.task import (
    prepare_single_turn as prepare_single_turn,
)
