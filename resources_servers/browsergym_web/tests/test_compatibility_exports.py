# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.web.api_models import WebSeedSessionRequest, WebStepRequest
from nemo_gym.web.session import (
    BenchmarkPreconditionError,
    CapacityUnavailableError,
    SessionConflictError,
    SessionNotFoundError,
)
from resources_servers.browsergym_web import session_manager
from resources_servers.browsergym_web.models import (
    WebSeedSessionRequest as BrowserGymSeedSessionRequest,
)
from resources_servers.browsergym_web.models import WebStepRequest as BrowserGymStepRequest


def test_browsergym_reexports_backend_neutral_wire_models() -> None:
    assert BrowserGymSeedSessionRequest is WebSeedSessionRequest
    assert BrowserGymStepRequest is WebStepRequest


def test_browsergym_reexports_common_session_errors() -> None:
    assert session_manager.SessionNotFoundError is SessionNotFoundError
    assert session_manager.SessionConflictError is SessionConflictError
    assert session_manager.CapacityUnavailableError is CapacityUnavailableError
    assert session_manager.BenchmarkPreconditionError is BenchmarkPreconditionError
