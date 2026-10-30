# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the GenRM tests."""

from typing import Optional

from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY


def request_with_session(session_id: Optional[str] = None) -> Request:
    """A request as the session middleware leaves it, for calling ``verify`` directly."""
    return Request({"type": "http", "session": {} if session_id is None else {SESSION_ID_KEY: session_id}})
