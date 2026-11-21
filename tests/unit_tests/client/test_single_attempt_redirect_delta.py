# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit ServerClient opt-in refuses redirects without changing legacy requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import ClientResponseError, RequestInfo
from omegaconf import OmegaConf
from yarl import URL

import nemo_gym.client.http_client as server_utils
import nemo_gym.client.openai_client as openai_utils
import nemo_gym.client.server_client as server_client
from nemo_gym.config.config_types import BaseServerConfig


class SingleAttemptRedirectTest(unittest.IsolatedAsyncioTestCase):
    async def test_split_clients_preserve_opt_in_redirect_refusal(self) -> None:
        configuration = OmegaConf.create(
            {
                "single_attempt_mutating_requests": True,
                "target": {"responses_api_models": {"example": {"host": "127.0.0.1", "port": 12345}}},
            }
        )
        client = server_client.ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=12344), global_config_dict=configuration
        )
        for status in (307, 308):
            with self.subTest(status=status):
                url = URL("http://synthetic.invalid/v1/responses")
                raw = SimpleNamespace(
                    status=status,
                    request_info=RequestInfo(url, "POST", {}, url),
                    history=(),
                    headers={"Location": "/not-followed"},
                    release=MagicMock(),
                )
                send = AsyncMock(return_value=raw)
                with (
                    patch.object(
                        server_utils, "get_global_aiohttp_client", return_value=SimpleNamespace(request=send)
                    ),
                    patch.object(server_utils, "is_span_group_enabled", return_value=False),
                    patch.object(server_utils, "request_counts", return_value=None),
                    patch.object(server_client, "getenv", return_value=None),
                ):
                    with self.assertRaises(ClientResponseError) as error:
                        await client.post(server_name="target", url_path="/v1/responses", allow_redirects=True)
                self.assertEqual(error.exception.status, status)
                self.assertEqual(send.await_count, 1)
                self.assertIs(send.await_args.kwargs["allow_redirects"], False)
                raw.release.assert_called_once_with()

    async def test_mutations_reject_redirects_without_changing_read_or_default_requests(self) -> None:
        configuration = OmegaConf.create(
            {
                "single_attempt_mutating_requests": True,
                "target": {"responses_api_models": {"example": {"host": "127.0.0.1", "port": 12345}}},
            }
        )
        server = server_client.ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=12344),
            global_config_dict=configuration,
        )
        upstream = openai_utils.NeMoGymAsyncOpenAI(
            base_url="http://provider.invalid/v1", api_key="synthetic-not-a-key", max_num_tries=1
        )
        for path in ("server", "upstream", "direct"):
            for status in (307, 308):
                with self.subTest(path=path, status=status):
                    url = URL("http://provider.invalid/v1/responses")
                    raw = SimpleNamespace(
                        status=status,
                        request_info=RequestInfo(url, "POST", {}, url),
                        history=(),
                        headers={"Location": "http://other.invalid/do-not-follow"},
                        release=MagicMock(),
                        content=SimpleNamespace(
                            read=AsyncMock(side_effect=AssertionError("Do not read redirect body"))
                        ),
                    )
                    send = AsyncMock(return_value=raw)
                    sleep = AsyncMock()
                    raised = None
                    actual = None
                    with (
                        patch.object(
                            server_utils, "get_global_aiohttp_client", return_value=SimpleNamespace(request=send)
                        ),
                        patch.object(
                            server_utils,
                            "get_global_aiohttp_control_client",
                            return_value=SimpleNamespace(request=send),
                        ),
                        patch.object(server_utils, "is_span_group_enabled", return_value=False),
                        patch.object(server_utils, "request_counts", return_value=None),
                        patch.object(server_client, "getenv", return_value=None),
                        patch.object(server_utils.asyncio, "sleep", sleep),
                    ):
                        try:
                            if path == "server":
                                await server.request(
                                    "target", "/v1/responses", "POST", _control=True, allow_redirects=True
                                )
                            elif path == "upstream":
                                actual = await upstream._request(method="POST", url=str(url), allow_redirects=True)
                            else:
                                actual = await server_utils.request(
                                    "pAtCh", str(url), _max_num_tries=1, allow_redirects=True
                                )
                        except ClientResponseError as error:
                            raised = error
                    if path == "server":
                        self.assertIsInstance(raised, ClientResponseError)
                        self.assertEqual(raised.status, status)
                        raw.release.assert_called_once_with()
                    else:
                        self.assertIsNone(raised)
                        self.assertIs(actual, raw)
                        raw.release.assert_not_called()
                    self.assertEqual(send.await_count, 1)
                    self.assertIs(send.await_args.kwargs["allow_redirects"], path != "server")
                    raw.content.read.assert_not_awaited()
                    sleep.assert_not_awaited()

        for method, cap, redirects in (("GET", 1, True), ("HEAD", 1, True), ("POST", None, True), ("POST", 2, True)):
            with self.subTest(control=(method, cap)):
                raw = SimpleNamespace(status=307, release=MagicMock())
                send = AsyncMock(return_value=raw)
                with (
                    patch.object(
                        server_utils, "get_global_aiohttp_client", return_value=SimpleNamespace(request=send)
                    ),
                    patch.object(server_utils, "is_span_group_enabled", return_value=False),
                    patch.object(server_utils, "request_counts", return_value=None),
                ):
                    actual = await server_utils.request(
                        method, "http://provider.invalid/control", _max_num_tries=cap, allow_redirects=redirects
                    )
                self.assertIs(actual, raw)
                self.assertIs(send.await_args.kwargs["allow_redirects"], redirects)
                raw.release.assert_not_called()
