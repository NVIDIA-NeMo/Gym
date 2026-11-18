# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit internal redirect refusal does not change a retry cap's meaning."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import ClientResponseError, RequestInfo
from omegaconf import OmegaConf
from yarl import URL

from nemo_gym.client import http_client, openai_client, server_client
from nemo_gym.config import BaseServerConfig
from nemo_gym.telemetry import contrib, spans


class RedirectOptInCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_only_explicit_mutation_opt_in_refuses_actual_redirect_statuses(self) -> None:
        redirect_statuses = (301, 302, 303, 307, 308)
        cases = [
            ("server", True, "POST", True, status, traced, counted, False)
            for status in (*redirect_statuses, 300, 304)
            for traced, counted in ((False, False), (True, False), (True, True), (False, True))
        ]
        cases += [
            ("server", True, method, True, 307, False, False, False) for method in ("GET", "HEAD", "OPTIONS", "TRACE")
        ]
        cases += [("server", opt_in, "POST", True, 307, False, False, False) for opt_in in (False, "true")]
        cases += [
            (route, False, "POST", redirects, 307, False, False, control)
            for route in ("direct", "provider")
            for redirects in (True, False)
            for control in (False, True)
        ]
        url = URL("http://synthetic.invalid/v1/responses")
        for route, opt_in, method, redirects, status, traced, counted, control in cases:
            with self.subTest(case=(route, opt_in, method, redirects, status, traced, counted, control)):
                raw = SimpleNamespace(
                    status=status,
                    ok=True,
                    request_info=RequestInfo(url, method, {}, url),
                    history=(),
                    headers={"Location": "/synthetic-not-followed"},
                    release=MagicMock(),
                    content=SimpleNamespace(read=AsyncMock(side_effect=AssertionError("No body read"))),
                )
                send = AsyncMock(return_value=raw)
                counts = SimpleNamespace(outbound_in_flight=0) if counted else None
                config = OmegaConf.create(
                    {
                        "single_attempt_mutating_requests": opt_in,
                        "target": {"responses_api_models": {"example": {"host": "127.0.0.1", "port": 1}}},
                    }
                )
                client = server_client.ServerClient(
                    head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config
                )
                error = None
                result = None
                with (
                    patch.object(http_client, "get_global_aiohttp_client", return_value=SimpleNamespace(request=send)),
                    patch.object(
                        http_client, "get_global_aiohttp_control_client", return_value=SimpleNamespace(request=send)
                    ),
                    patch.object(http_client, "is_span_group_enabled", return_value=traced),
                    patch.object(http_client, "is_trace_exporter_active", return_value=traced),
                    patch.object(http_client, "request_counts", return_value=counts),
                    patch.object(server_client, "getenv", return_value=None),
                    patch.object(spans, "client_span", return_value=nullcontext(None)),
                    patch.object(contrib, "inject_trace_context"),
                    patch.object(http_client.asyncio, "sleep", side_effect=AssertionError("No retry")),
                ):
                    try:
                        if route == "server":
                            result = await client.request(
                                "target", "/v1/responses", method, _control=control, allow_redirects=redirects
                            )
                        elif route == "provider":
                            provider = openai_client.NeMoGymAsyncOpenAI(
                                base_url=str(url), api_key="synthetic-not-a-credential", max_num_tries=1
                            )
                            result = await provider._request(
                                method=method, url=str(url), _control=control, allow_redirects=redirects
                            )
                        else:
                            result = await http_client.request(
                                method, str(url), _max_num_tries=1, _control=control, allow_redirects=redirects
                            )
                    except ClientResponseError as caught:
                        error = caught
                opted_mutation = (
                    route == "server" and opt_in is True and method not in {"GET", "HEAD", "OPTIONS", "TRACE"}
                )
                refused = opted_mutation and status in redirect_statuses
                if refused:
                    self.assertIsInstance(error, ClientResponseError)
                    self.assertEqual(error.status, status)
                    raw.release.assert_called_once_with()
                else:
                    self.assertIsNone(error)
                    self.assertIs(result, raw)
                    raw.release.assert_not_called()
                self.assertEqual(send.await_count, 1)
                self.assertIs(send.await_args.kwargs["allow_redirects"], False if opted_mutation else redirects)
                self.assertNotIn("_refuse_redirects", send.await_args.kwargs)
                raw.content.read.assert_not_awaited()
                if counts is not None:
                    self.assertEqual(counts.outbound_in_flight, 0)
