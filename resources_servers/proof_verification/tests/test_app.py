# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import fcntl
import json
import threading
import time
from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from resources_servers.proof_verification.app import (
    ProofVerificationResourcesServer,
    ProofVerificationResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ProofVerificationResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        )
        ProofVerificationResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_log_appends_wait_for_another_workers_lock(self, tmp_path) -> None:
        """With several workers appending to one debug log, an append waits for a lock another process holds."""
        server = ProofVerificationResourcesServer(
            config=ProofVerificationResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        log = tmp_path / "log.jsonl"
        other_worker = open(log, "a")
        fcntl.flock(other_worker, fcntl.LOCK_EX)
        released_at: list[float] = []

        def release() -> None:
            time.sleep(0.3)
            released_at.append(time.monotonic())
            other_worker.close()

        releaser = threading.Thread(target=release)
        releaser.start()
        asyncio.run(
            server._append_log_jsonl(log_path=str(log), problem="p", generated_sequence="g", reward=1.0, details={})
        )
        appended_at = time.monotonic()
        releaser.join()

        assert appended_at >= released_at[0]
        assert json.loads(log.read_text())["reward"] == 1.0
