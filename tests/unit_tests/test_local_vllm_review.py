# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for partial Gym startup and shared benchmark telemetry."""

import os
import signal
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from benchmarks import inference_metrics
from nemo_gym import server_utils
from nemo_gym.cli import env


@pytest.mark.parametrize("error", [RuntimeError("readiness failed"), KeyboardInterrupt()])
def test_partial_start_reaps_owned_cpu_child(monkeypatch, error):
    config = OmegaConf.create(
        {
            "dry_run": False,
            "policy_model": {
                "responses_api_models": {"local_vllm_model": {"entrypoint": "app.py", "launcher": "subprocess"}}
            },
        }
    )
    monkeypatch.setattr(env, "get_global_config_dict", lambda **_: config)
    monkeypatch.setattr(env, "initialize_ray", lambda: None)
    monkeypatch.setattr(env, "init_telemetry", lambda **_: None)
    monkeypatch.setattr(env, "shutdown_telemetry", lambda: None)
    monkeypatch.setattr(env, "setup_env_command", lambda *_: "true")
    head, thread, instance = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(env.HeadServer, "run_webserver", lambda: (head, thread, instance))
    client = MagicMock()
    client.poll_for_status.side_effect = error
    monkeypatch.setattr(env, "ServerClient", MagicMock(return_value=client))
    children = []

    def spawn(*args, **kwargs):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"],
            start_new_session=kwargs.get("start_new_session", False),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        children.append(child)
        return child

    monkeypatch.setattr(env, "run_command", spawn)
    runner = env.RunHelper()
    try:
        with pytest.raises(type(error)):
            runner.start(None)
        assert len(children) == 1
        assert children[0].poll() is not None, "partial startup left the owned process alive"
        assert head.should_exit is True
        thread.join.assert_called_once()
        runner.shutdown()  # Safe if an outer caller also cleans up.
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


@pytest.mark.asyncio
async def test_baseline_mooncake_config_collects_storage_metrics(tmp_path, monkeypatch):
    from aiohttp import web

    from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, set_global_aiohttp_client

    payload = "# TYPE master_allocated_bytes gauge\nmaster_allocated_bytes 30\n# TYPE master_total_capacity_bytes gauge\nmaster_total_capacity_bytes 100\n"
    app = web.Application()
    app.router.add_get("/metrics", lambda _: web.Response(text=payload))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    published = []
    monkeypatch.setattr(inference_metrics, "export_metrics", published.append)
    client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    try:
        config = inference_metrics.InferenceMetricsConfig(
            enabled=True,
            endpoints={},
            router_endpoints={},
            mooncake_endpoint=f"http://127.0.0.1:{port}/metrics",
            output_path=tmp_path / "metrics.jsonl",
            require_wandb=False,
        )
        async with inference_metrics.collect_inference_metrics(config):
            pass
        assert any(m.get("mooncake/total/kv_cache_usage_perc") == 0.3 for m in published)
        assert all(m.get("inference/scrape_up/mooncake", 1) == 1 for m in published)
    finally:
        await client.close()
        server_utils._GLOBAL_AIOHTTP_CLIENT = None
        await runner.cleanup()
