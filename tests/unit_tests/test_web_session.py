# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from nemo_gym.web.browser_session import BrowserSessionHandle
from nemo_gym.web.models import WebBenchmark, WebObservation, WebTask
from nemo_gym.web.operation_runner import DirectWebOperationRunner
from nemo_gym.web.session import WebSessionState
from nemo_gym.web.site_pool import SiteLease


class _Backend:
    def reset(self, task):
        del task

    def observe(self):
        return WebObservation()

    def step(self, action):
        del action

    def evaluate(self, final_answer=None):
        del final_answer

    def close(self):
        return None


def test_common_session_state_owns_idempotency_and_verifier_slots():
    task = WebTask(benchmark=WebBenchmark.WEBARENA, task_id="7")
    state = WebSessionState(
        session_id="session-7",
        task=task,
        backend=_Backend(),
        browser_lease=BrowserSessionHandle(
            session_id="browser-session-7",
            endpoint=None,
            metadata={},
        ),
        site_lease=SiteLease(lease_id="test:7", isolated=True),
        observation=WebObservation(url="about:blank"),
        seed_info={},
        created_at=1.0,
        last_access_at=1.0,
        operation_runner=DirectWebOperationRunner(),
    )

    assert state.operations == OrderedDict()
    assert state.verifier_result is None
    assert state.status == "ready"
