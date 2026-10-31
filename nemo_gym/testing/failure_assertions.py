# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from nemo_gym.core.failure_kinds import AGENT_ROUTE_FAILED
from nemo_gym.typed_exceptions import NgFailureEvent, route_reply_error


@contextmanager
def raises_agent_failure(
    expected_exception: type[BaseException], *, match: str | None = None
) -> Iterator[pytest.ExceptionInfo]:
    """Require an ``agent route failure`` from an agent route, then assert on its cause as ``pytest.raises`` does."""
    with pytest.raises(expected_exception, match=match) as original:
        try:
            yield original
        except NgFailureEvent as failure:
            if failure.failure_kind != AGENT_ROUTE_FAILED:
                pytest.fail(f"Expected an agent route failure, got {failure.failure_kind}")
            cause = failure.__cause__
            assert cause is not None
            assert route_reply_error(failure) is cause
            raise cause
        except Exception as error:
            pytest.fail(f"Expected agent route failure, got {type(error).__name__}")
        else:
            pytest.fail("Agent HTTP boundary did not raise agent route failure")
