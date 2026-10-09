# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable capture of a harness's model calls from inside its task sandbox.

A session capture is a component that runs next to the harness in the task sandbox, for example a proxy that
records every model call with its token ids. It supplies the endpoint the harness calls, and when the session
closes it is collected into a :class:`~nemo_gym.base_responses_api_agent.TokenCapture` that the agent returns
from its close response. Gym ships only this interface and its lifecycle in
:class:`~nemo_gym.agent_utils.sandbox_session.SandboxSession`; implementations are selected by configuration.
"""

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_responses_api_agent import ModelEndpoint, TokenCapture
from nemo_gym.sandbox.api import AsyncSandbox


class SessionCapture(ABC):
    """One capture component for one sandbox session.

    ``SandboxSession`` builds a new instance per session, calls ``start`` once before the harness command is
    built, and then exactly one of ``collect`` (after the harness has stopped and before the sandbox is
    released) or ``abort`` (when collection cannot happen or did not finish). The constructor receives the
    configured ``options`` as keyword arguments; it runs once at configuration load to validate them, so it
    must not do I/O.
    """

    @abstractmethod
    async def start(self, sandbox: AsyncSandbox) -> ModelEndpoint:
        """Start the component in ``sandbox`` and return the endpoint the harness must call.

        Raise when the component cannot start; the session then masks its capture and refuses to run the harness.
        If start is cancelled or raises, ``abort`` is called next, so it must be able to stop a partial start.
        """

    @abstractmethod
    async def collect(self, sandbox: AsyncSandbox) -> TokenCapture:
        """Stop the component and return what it captured.

        Report an unusable capture as ``TokenCapture(masked=True, mask_reason=...)`` instead of raising.
        """

    @abstractmethod
    async def abort(self, sandbox: AsyncSandbox) -> None:
        """Stop the component without collecting, best effort; never raise."""


def load_session_capture_class(implementation: str) -> type[SessionCapture]:
    """Import ``"package.module:ClassName"`` and check that it is a concrete :class:`SessionCapture`.

    Raises ``ValueError`` naming the configured path when it is malformed, cannot be imported, or does not
    name a concrete ``SessionCapture`` subclass.
    """
    module_name, separator, class_name = implementation.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(
            f"session capture implementation must look like 'package.module:ClassName', got {implementation!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"cannot import session capture implementation {implementation!r}: {error}") from error
    capture_class = getattr(module, class_name, None)
    if not (isinstance(capture_class, type) and issubclass(capture_class, SessionCapture)):
        raise ValueError(f"session capture implementation {implementation!r} is not a SessionCapture subclass")
    if inspect.isabstract(capture_class):
        raise ValueError(f"session capture implementation {implementation!r} does not implement every method")
    return capture_class


class SessionCaptureConfig(BaseModel):
    """Selects the session capture an agent runs in each task sandbox.

    An agent embeds this as an optional configuration field and passes it to ``SandboxSession``.
    The implementation and its options are validated when the configuration loads.
    """

    model_config = ConfigDict(extra="forbid")

    # Import path of a SessionCapture subclass, as "package.module:ClassName".
    implementation: str
    # Keyword arguments for the implementation's constructor.
    options: dict[str, Any] = Field(default_factory=dict)
    # Bound on collect, and separately on abort after a failed collect. Together with the agent's own close
    # timeouts it must fit inside the environment server's cleanup_timeout_seconds (180 seconds by default).
    collect_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)

    @field_validator("implementation")
    @classmethod
    def _load_implementation(cls, implementation: str) -> str:
        load_session_capture_class(implementation)
        return implementation

    @model_validator(mode="after")
    def _validate_options(self) -> "SessionCaptureConfig":
        self.build()
        return self

    def build(self) -> SessionCapture:
        """Construct a new capture component for one session."""
        capture_class = load_session_capture_class(self.implementation)
        try:
            return capture_class(**self.options)
        except TypeError as error:
            raise ValueError(f"invalid options for session capture {self.implementation!r}: {error}") from error
