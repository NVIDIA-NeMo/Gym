# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import Mock

import pytest

from responses_api_models.local_vllm_model.local_vllm_model_actor import _vllm_asyncio_task


@pytest.mark.parametrize("fail", [False, True])
def test_worker_receives_reserved_socket_and_closes_on_exit(monkeypatch, fail):
    sock = Mock()
    args = Namespace()
    received = []

    async def run_server_worker(address, reserved, server_args):
        received.append((address, reserved, server_args))
        if fail:
            raise RuntimeError("server failed")

    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.openai"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    api = ModuleType("vllm.entrypoints.openai.api_server")
    api.run_server_worker = run_server_worker
    monkeypatch.setitem(sys.modules, api.__name__, api)
    if fail:
        with pytest.raises(RuntimeError, match="server failed"):
            _vllm_asyncio_task(args, "http://localhost:1234", sock)
    else:
        _vllm_asyncio_task(args, "http://localhost:1234", sock)
    assert received == [("http://localhost:1234", sock, args)]
    sock.close.assert_called_once_with()
