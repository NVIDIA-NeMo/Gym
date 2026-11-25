# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier construction for the workload contract fixtures."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemo_gym.client import ServerClient


if TYPE_CHECKING:
    from resources_servers.audio_eval.app import AudioEvalResourcesServer


def create_server() -> "AudioEvalResourcesServer":
    """A server over a tiny ASR profile."""
    from resources_servers.audio_eval.app import AudioEvalConfig, AudioEvalResourcesServer

    return AudioEvalResourcesServer(
        config=AudioEvalConfig(
            name="audio_eval",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            profile_contracts=str(Path(__file__).parent / "tests" / "fixture_profiles.json"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
