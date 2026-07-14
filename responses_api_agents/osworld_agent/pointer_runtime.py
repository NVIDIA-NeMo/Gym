# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local Pointer hardening owned by the Gym resource server."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Mapping


QUOTA_RETRY_WINDOW_SECONDS = 900.0
QUOTA_RETRY_MAX_DELAY_SECONDS = 300.0
QUOTA_RETRY_BASE_DELAY_SECONDS = 60.0
QUOTA_EXHAUSTED_MESSAGE = "Provider quota retry window exhausted."
PROVIDER_FAILURE_MESSAGE = "Provider request failed after Pointer retries."
CLIENT_REFRESH_FAILURE_MESSAGE = "Provider client refresh failed."

_POINTER_ROLES = ("gate", "planner", "executor", "verifier")
_RUNTIME_STATE = "_nemo_gym_pointer_runtime"
_LOG = logging.getLogger(__name__)


class _QuotaExhausted(BaseException):
    """Bypass Pointer's broad ``except Exception`` retry blocks."""

    def __init__(self, role: str, cumulative_wait: float) -> None:
        self.role = role
        self.cumulative_wait = cumulative_wait


def is_budget_exceeded(error: Exception) -> bool:
    """Match only Anthropic's structured HTTP 429 budget response."""

    try:
        from anthropic import APIStatusError  # noqa: PLC0415
    except ImportError:
        return False
    if not isinstance(error, APIStatusError) or error.status_code != 429:
        return False
    body = error.body
    detail = body.get("error") if isinstance(body, dict) else None
    return isinstance(detail, dict) and detail.get("type") == "budget_exceeded"


def _is_content_length_error(error: Exception) -> bool:
    if getattr(error, "status_code", None) == 413:
        return True
    body = getattr(error, "body", None)
    detail = body.get("error") if isinstance(body, dict) else None
    error_type = detail.get("type", "") if isinstance(detail, dict) else ""
    text = f"{error_type} {error}".lower()
    return "content_length_limit" in text or "content length exceeded" in text


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


class _RetryController:
    def __init__(self, role: str, logger: Any, deadline: float | None) -> None:
        self.role = role
        self.logger = logger
        self.deadline = deadline
        self.waited = 0.0
        self.attempt = 0
        self.quota_deadline: float | None = None
        self.quota_started: float | None = None
        self.active = False

    def begin(self) -> None:
        if self.active:
            raise RuntimeError(f"Pointer {self.role} generation re-entered")
        self.active = True
        self._reset()

    def end(self) -> None:
        self._reset()
        self.active = False

    def call(self, create: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        while True:
            if self.quota_deadline is not None:
                now = time.monotonic()
                self._update_waited(now)
                if self._remaining(now) <= 0:
                    self._raise_exhausted()
            budget_error: Exception | None = None
            try:
                result = create(*args, **kwargs)
            except Exception as error:
                if is_budget_exceeded(error):
                    budget_error = error
                else:
                    if not self.active:
                        self._reset()
                    if _is_content_length_error(error):
                        raise RuntimeError("413 request too large") from None
                    raise
            if budget_error is not None:
                # Logging happens after the provider exception leaves the active
                # exception context so handler failures cannot expose its body.
                self._pause(budget_error)
                continue
            if not self.active:
                self._reset()
            return result

    def _pause(self, error: Exception) -> None:
        now = time.monotonic()
        if self.quota_deadline is None:
            self.quota_started = now
            self.quota_deadline = now + QUOTA_RETRY_WINDOW_SECONDS
        self._update_waited(now)
        remaining = self._remaining(now)
        if remaining <= 0:
            self._raise_exhausted()

        cap = min(QUOTA_RETRY_MAX_DELAY_SECONDS, QUOTA_RETRY_BASE_DELAY_SECONDS * (2**self.attempt))
        delay = cap / 2.0 + random.random() * cap / 2.0
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            retry_after = min(retry_after, QUOTA_RETRY_MAX_DELAY_SECONDS)
            if retry_after > remaining:
                self._raise_exhausted()
            delay = max(delay, retry_after)
        planned_delay = min(delay, remaining)

        try:
            self.logger.warning(
                "Provider quota pause; retrying in %.1fs (attempt %d, cumulative wait %.1fs)",
                planned_delay,
                self.attempt + 1,
                self.waited + planned_delay,
            )
        except Exception:  # noqa: BLE001 - observability must not break quota recovery.
            _LOG.warning("Pointer quota retry logging failed")

        now = time.monotonic()
        self._update_waited(now)
        remaining = self._remaining(now)
        if remaining <= 0:
            self._raise_exhausted()
        delay = min(planned_delay, remaining)
        try:
            time.sleep(delay)
        except BaseException as interruption:
            interruption.__suppress_context__ = True
            raise

        now = time.monotonic()
        self._update_waited(now)
        self.attempt += 1
        if self._remaining(now) <= 0:
            self._raise_exhausted()

    def _remaining(self, now: float) -> float:
        if self.quota_deadline is None:
            return math.inf
        remaining = self.quota_deadline - now
        if self.deadline is not None:
            remaining = min(remaining, self.deadline - now)
        return remaining

    def _update_waited(self, now: float) -> None:
        if self.quota_started is not None:
            self.waited = max(self.waited, now - self.quota_started)

    def _reset(self) -> None:
        self.waited = 0.0
        self.attempt = 0
        self.quota_deadline = None
        self.quota_started = None

    def _raise_exhausted(self) -> None:
        if self.active:
            raise _QuotaExhausted(self.role, self.waited) from None
        raise RuntimeError(f"[{self.role}] {QUOTA_EXHAUSTED_MESSAGE} Cumulative wait: {self.waited:.1f}s") from None


class _MessagesProxy:
    def __init__(self, messages: Any, controller: _RetryController, *, sanitize_errors: bool = False) -> None:
        self._messages = messages
        self._controller = controller
        self._sanitize_errors = sanitize_errors

    def create(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._controller.call(self._messages.create, *args, **kwargs)
        except Exception:
            if self._sanitize_errors:
                raise RuntimeError(f"[{self._controller.role}] {PROVIDER_FAILURE_MESSAGE}") from None
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _ClientProxy:
    def __init__(self, client: Any, controller: _RetryController) -> None:
        self._controller = controller
        self._client = client
        self.beta = _BetaProxy(client.beta, controller)
        self.messages = _MessagesProxy(client.messages, controller, sanitize_errors=True)

    def close(self) -> None:
        self._client.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _BetaProxy:
    def __init__(self, beta: Any, controller: _RetryController) -> None:
        self._beta = beta
        self.messages = _MessagesProxy(beta.messages, controller)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._beta, name)


@dataclass
class _RoleRuntime:
    client_label: str
    llm: Any
    original_generate: Callable[..., Any]
    original_refresh: Callable[..., Any]
    old_client: Any
    old_context_client: Any
    had_counting_client: bool
    old_counting_client: Any
    client_options: dict[str, Any]
    client_wrapper: Callable[[Any, str], Any] | None
    controller: _RetryController
    client: Any


@dataclass
class _AgentRuntime:
    closed: bool = False


def _new_client(
    options: Mapping[str, Any],
    controller: _RetryController,
    client_label: str,
    client_wrapper: Callable[[Any, str], Any] | None,
) -> Any:
    from anthropic import Anthropic  # noqa: PLC0415

    client = Anthropic(**dict(options))
    setup_error: BaseException | None = None
    try:
        instrumented_client = client_wrapper(client, client_label) if client_wrapper is not None else client
        return _ClientProxy(instrumented_client, controller)
    except BaseException as error:
        setup_error = error
    _close_client(client)
    assert setup_error is not None
    raise setup_error from None


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - cleanup is best effort.
        _LOG.warning("Pointer Anthropic client cleanup failed")


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "value", provider)).rsplit(".", 1)[-1].lower()


def _unique_clients(clients: list[Any]) -> list[Any]:
    unique = []
    for client in clients:
        if client is not None and all(client is not existing for existing in unique):
            unique.append(client)
    return unique


def _pointer_llms(pointer_agent: Any) -> list[tuple[str, Any]]:
    if not callable(getattr(pointer_agent, "predict", None)):
        raise RuntimeError("Pointer agent is incompatible with Gym runtime hardening: missing predict")
    result = []
    for role in _POINTER_ROLES:
        llm = getattr(getattr(pointer_agent, role, None), "llm", None)
        context = getattr(llm, "context_manager", None)
        client = getattr(llm, "client", None)
        beta_messages = getattr(getattr(client, "beta", None), "messages", None)
        stable_messages = getattr(client, "messages", None)
        valid = (
            _provider_name(getattr(llm, "provider", "")) == "anthropic"
            and callable(getattr(llm, "generate", None))
            and callable(getattr(llm, "_refresh_client", None))
            and getattr(llm, "logger", None) is not None
            and context is not None
            and callable(getattr(beta_messages, "create", None))
            and callable(getattr(stable_messages, "create", None))
        )
        if not valid:
            raise RuntimeError(f"Pointer {role} LLM is incompatible with Gym runtime hardening")
        result.append((role, llm))
    return result


def _generate(self: Any) -> Any:
    runtime: _RoleRuntime = self._nemo_gym_pointer_role_runtime
    runtime.controller.begin()
    try:
        return runtime.original_generate()
    except _QuotaExhausted as error:
        raise RuntimeError(
            f"[{error.role}] {QUOTA_EXHAUSTED_MESSAGE} Cumulative wait: {error.cumulative_wait:.1f}s"
        ) from None
    except RuntimeError as error:
        if "All API attempts exhausted" in str(error):
            raise RuntimeError(f"[{runtime.controller.role}] {PROVIDER_FAILURE_MESSAGE}") from None
        raise
    finally:
        runtime.controller.end()


def _refresh_client(self: Any) -> None:
    runtime: _RoleRuntime = self._nemo_gym_pointer_role_runtime
    try:
        replacement = _new_client(
            runtime.client_options,
            runtime.controller,
            runtime.client_label,
            runtime.client_wrapper,
        )
    except BaseException as error:
        if not isinstance(error, Exception):
            error.__suppress_context__ = True
            raise
        raise RuntimeError(f"[{runtime.controller.role}] {CLIENT_REFRESH_FAILURE_MESSAGE}") from None
    context = self.context_manager
    old_client = self.client
    old_context_client = context.client
    had_counting_client = hasattr(context, "_counting_client")
    old_counting_client = getattr(context, "_counting_client", None)
    old_runtime_client = runtime.client
    old_clients = _unique_clients(
        [
            old_client,
            old_context_client,
            old_counting_client,
        ]
    )
    swap_error: BaseException | None = None
    try:
        self.client = replacement
        context.client = replacement
        context._counting_client = replacement
        runtime.client = replacement
    except BaseException as error:
        swap_error = error

    if swap_error is not None:
        rollback_error: BaseException | None = None
        try:
            self.client = old_client
            context.client = old_context_client
            if had_counting_client:
                context._counting_client = old_counting_client
            else:
                vars(context).pop("_counting_client", None)
            runtime.client = old_runtime_client
        except BaseException as error:
            rollback_error = error
        _close_client(replacement)
        interruption = next(
            (
                error
                for error in (swap_error, rollback_error)
                if error is not None and not isinstance(error, Exception)
            ),
            None,
        )
        if interruption is not None:
            interruption.__suppress_context__ = True
            raise interruption from None
        raise RuntimeError(f"[{runtime.controller.role}] {CLIENT_REFRESH_FAILURE_MESSAGE}") from None
    for client in old_clients:
        if client is not replacement:
            _close_client(client)


def _restore_role(runtime: _RoleRuntime) -> None:
    context = runtime.llm.context_manager
    runtime.llm.generate = runtime.original_generate
    runtime.llm._refresh_client = runtime.original_refresh
    runtime.llm.client = runtime.old_client
    context.client = runtime.old_context_client
    if runtime.had_counting_client:
        context._counting_client = runtime.old_counting_client
    else:
        vars(context).pop("_counting_client", None)
    vars(runtime.llm).pop("_nemo_gym_pointer_role_runtime", None)


def harden_pointer_agent(
    pointer_agent: Any,
    anthropic_client_options: Mapping[str, Any],
    *,
    deadline_monotonic: float | None = None,
    client_wrapper: Callable[[Any, str], Any] | None = None,
) -> None:
    """Replace one initialized PointerAgent's clients without changing OSWorld."""

    existing = getattr(pointer_agent, _RUNTIME_STATE, None)
    if existing is not None:
        if existing.closed:
            raise RuntimeError("Pointer runtime is already closed")
        return
    options = dict(anthropic_client_options)
    if options.get("max_retries") != 0:
        raise ValueError("Pointer runtime hardening requires Anthropic max_retries=0")

    roles: list[_RoleRuntime] = []
    try:
        for role, llm in _pointer_llms(pointer_agent):
            context = llm.context_manager
            controller = _RetryController(role, llm.logger, deadline_monotonic)
            roles.append(
                _RoleRuntime(
                    client_label=str(getattr(llm, "name", role)),
                    llm=llm,
                    original_generate=llm.generate,
                    original_refresh=llm._refresh_client,
                    old_client=llm.client,
                    old_context_client=context.client,
                    had_counting_client=hasattr(context, "_counting_client"),
                    old_counting_client=getattr(context, "_counting_client", None),
                    client_options=options,
                    client_wrapper=client_wrapper,
                    controller=controller,
                    client=_new_client(
                        options,
                        controller,
                        str(getattr(llm, "name", role)),
                        client_wrapper,
                    ),
                )
            )
    except BaseException:
        for runtime in roles:
            _close_client(runtime.client)
        raise

    state = _AgentRuntime()
    try:
        for runtime in roles:
            runtime.llm._nemo_gym_pointer_role_runtime = runtime
            runtime.llm.generate = MethodType(_generate, runtime.llm)
            runtime.llm._refresh_client = MethodType(_refresh_client, runtime.llm)
            runtime.llm.client = runtime.client
            runtime.llm.context_manager.client = runtime.client
            runtime.llm.context_manager._counting_client = runtime.client
        setattr(pointer_agent, _RUNTIME_STATE, state)
    except BaseException:
        vars(pointer_agent).pop(_RUNTIME_STATE, None)
        for runtime in roles:
            _restore_role(runtime)
            _close_client(runtime.client)
        raise

    old_clients = [
        client
        for runtime in roles
        for client in (runtime.old_client, runtime.old_context_client, runtime.old_counting_client)
    ]
    for client in _unique_clients(old_clients):
        _close_client(client)


def close_pointer_agent(pointer_agent: Any) -> None:
    """Close Pointer role clients, including clients from failed setup."""

    state = getattr(pointer_agent, _RUNTIME_STATE, None)
    if state is not None and state.closed:
        return
    clients = []
    for role in _POINTER_ROLES:
        llm = getattr(getattr(pointer_agent, role, None), "llm", None)
        context = getattr(llm, "context_manager", None)
        clients.extend(
            (
                getattr(llm, "client", None),
                getattr(context, "client", None),
                getattr(context, "_counting_client", None),
            )
        )
    for client in _unique_clients(clients):
        _close_client(client)
    if state is not None:
        state.closed = True
