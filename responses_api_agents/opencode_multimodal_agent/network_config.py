# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect control origins without passing Gym configuration into task sandboxes."""

from collections.abc import Mapping

from responses_api_agents.opencode_multimodal_agent.network_runner import origin


def server_origin(host: str, port: int) -> str:
    host = {"0.0.0.0": "127.0.0.1", "::": "::1", "[::]": "::1"}.get(host, host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    url = f"http://{host}:{port}"
    origin(url)
    return url


def control_service_urls(config: Mapping, *, model_server: str) -> list[str]:
    """Protect every registered Gym server except the selected policy model.

    A tool service cohosted with verification/control routes needs a separate
    tool-only origin; granting its entire Gym origin would expose those routes.
    """
    urls = set()
    head = config.get("head_server")
    if head is not None:
        urls.add(server_origin(head["host"], head["port"]))
    groups = {"responses_api_agents", "responses_api_models", "resources_servers", "environment_servers"}
    for name, instance in config.items():
        if not isinstance(instance, Mapping):
            continue
        for group in groups & instance.keys():
            if name == model_server and group == "responses_api_models":
                continue
            for server in instance[group].values():
                urls.add(server_origin(server["host"], server["port"]))
    return sorted(urls)
