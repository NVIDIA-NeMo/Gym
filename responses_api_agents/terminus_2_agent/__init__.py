"""Standalone Terminus-2 agent server."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from responses_api_agents.terminus_2_agent.app import Terminus2Agent, Terminus2AgentConfig

_APP_EXPORTS = {"Terminus2Agent", "Terminus2AgentConfig"}


def __getattr__(name: str) -> object:
    # A model server imports this package to read the harness's assistant-message header
    # (`ServerClient.assistant_message_header`) in its own environment, which has no Harbor; load the app on first use.
    if name in _APP_EXPORTS:
        from responses_api_agents.terminus_2_agent import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Terminus2Agent", "Terminus2AgentConfig"]
