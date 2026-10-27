# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from nemo_gym.typed_exceptions import AgentFailure, reply_cause


@contextmanager
def raises_agent_failure(
    expected_exception: type[BaseException], *, match: str | None = None
) -> Iterator[pytest.ExceptionInfo]:
    """Require the HTTP wrapper and preserve original cause type, message and identity assertions."""
    with pytest.raises(expected_exception, match=match) as original:
        try:
            yield original
        except AgentFailure as failure:
            cause = failure.__cause__
            assert cause is not None
            assert reply_cause(failure) is cause
            raise cause
        except Exception as error:
            pytest.fail(f"Expected AgentFailure, got {type(error).__name__}")
        else:
            pytest.fail("Agent HTTP boundary did not raise AgentFailure")
