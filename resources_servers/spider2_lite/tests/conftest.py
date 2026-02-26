# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite


def pytest_configure(config):
    ensure_spider2_lite()
