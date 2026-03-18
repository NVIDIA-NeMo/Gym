# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests requiring downloaded databases and Spider2 reference repo"
    )
    config.addinivalue_line("markers", "e2e_llm: end-to-end tests requiring a local vLLM server and GPU")
    ensure_spider2_lite()
