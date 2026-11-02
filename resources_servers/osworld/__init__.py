# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native Resources Server for stateful OSWorld sessions."""

from resources_servers.osworld.app import OSWorldResourcesServer, OSWorldResourcesServerConfig


__all__ = ["OSWorldResourcesServer", "OSWorldResourcesServerConfig"]
