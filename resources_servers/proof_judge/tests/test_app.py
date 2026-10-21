# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import fcntl
import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient, refuse_unsupported_num_workers
from resources_servers.proof_judge.app import (
    ProofWithJudgeResourcesServer,
    ProofWithJudgeResourcesServerConfig,
)


def _config(**overrides) -> ProofWithJudgeResourcesServerConfig:
    return ProofWithJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="proof_judge",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        **overrides,
    )


class TestApp:
    def test_sanity(self) -> None:
        config = ProofWithJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        )
        ProofWithJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_log_appends_wait_for_another_workers_lock(self, tmp_path) -> None:
        """With several workers appending to one debug log, an append waits for a lock another process holds."""
        server = ProofWithJudgeResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))
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

    def test_several_workers_are_refused_only_for_group_rewards(self) -> None:
        grouped = ProofWithJudgeResourcesServer(
            config=_config(num_workers=2, zero_reward_incorrect_groups=True, expected_group_size=4),
            server_client=MagicMock(spec=ServerClient),
        )
        with pytest.raises(ValueError, match="needs num_workers=1: zero_reward_incorrect_groups waits"):
            refuse_unsupported_num_workers(grouped)

        stateless = ProofWithJudgeResourcesServer(
            config=_config(num_workers=2), server_client=MagicMock(spec=ServerClient)
        )
        refuse_unsupported_num_workers(stateless)
