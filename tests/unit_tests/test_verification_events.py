# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym import base_resources_server
from nemo_gym.base_resources_server import notify_verification_started, verification_events_url
from nemo_gym.rollout_correlation import rollout_context


class _Response:
    def release(self) -> None:
        pass


class TestVerificationEventsUrl:
    def test_loopback_urls_are_accepted(self) -> None:
        assert (
            verification_events_url({"verification_events_url": "http://127.0.0.1:9/events"})
            == "http://127.0.0.1:9/events"
        )
        assert verification_events_url({"verification_events_url": "http://localhost:9/e"}) == "http://localhost:9/e"

    def test_unset_means_no_events(self) -> None:
        assert verification_events_url({}) is None
        assert verification_events_url(None) is None

    @pytest.mark.parametrize("url", ["http://10.0.0.1:9/events", "https://127.0.0.1:9/events", "file:///tmp/x"])
    def test_other_hosts_and_schemes_are_refused(self, url: str) -> None:
        with pytest.raises(ValueError, match="loopback"):
            verification_events_url({"verification_events_url": url})


class TestNotifyVerificationStarted:
    async def test_posts_the_rollout_id_before_verifying(self, monkeypatch: pytest.MonkeyPatch) -> None:
        order = []

        async def fake_request(method, url, json):
            order.append(("event", method, url, json))
            return _Response()

        async def verify(body):
            order.append(("verify", body))
            return "verified"

        monkeypatch.setattr(base_resources_server, "request", fake_request)
        wrapped = notify_verification_started(verify, "http://127.0.0.1:9/events", "grader")
        with rollout_context("trial-7"):
            assert await wrapped(body="b") == "verified"
        assert order == [
            (
                "event",
                "POST",
                "http://127.0.0.1:9/events",
                {"event": "verification_started", "rollout_id": "trial-7", "server": "grader"},
            ),
            ("verify", "b"),
        ]

    async def test_a_failed_notification_does_not_change_verification(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def failing_request(method, url, json):
            raise ConnectionError("nobody listening")

        async def verify(body):
            return "verified"

        monkeypatch.setattr(base_resources_server, "request", failing_request)
        assert await notify_verification_started(verify, "http://127.0.0.1:9/events", "grader")(body="b") == "verified"

    def test_without_a_url_the_handler_is_unchanged(self) -> None:
        async def verify(body):
            return "verified"

        assert notify_verification_started(verify, None, "grader") is verify


class TestAgentKeepsRolloutIdForEvents:
    def test_correlation_is_on_when_verification_events_are_configured(self) -> None:
        from types import SimpleNamespace

        from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

        def agent(global_config_dict):
            return SimpleNamespace(server_client=SimpleNamespace(global_config_dict=global_config_dict))

        enabled = SimpleResponsesAPIAgent._verification_events_enabled
        assert enabled(agent({"verification_events_url": "http://127.0.0.1:9/e"}))
        assert not enabled(agent({}))
