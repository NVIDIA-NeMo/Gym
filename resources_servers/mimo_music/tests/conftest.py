# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

from resources_servers.mimo_music.setup_abc2midi import ensure_abc2midi


def pytest_configure(config) -> None:
    try:
        ensure_abc2midi()
    except Exception as error:
        logging.getLogger(__name__).warning("Renderer unavailable; integration tests will skip: %s", error)
