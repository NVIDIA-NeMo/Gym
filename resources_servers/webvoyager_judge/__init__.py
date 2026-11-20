# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebVoyager multimodal judge resource server."""

from resources_servers.webvoyager_judge.app import WebVoyagerJudgeResourcesServer
from resources_servers.webvoyager_judge.config import WebVoyagerJudgeConfig


__all__ = ["WebVoyagerJudgeConfig", "WebVoyagerJudgeResourcesServer"]
