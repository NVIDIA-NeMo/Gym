# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The head server: the registry every Gym server and client asks for the run's config and server addresses."""

from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Response
from omegaconf import OmegaConf
from pydantic import BaseModel, PrivateAttr

from nemo_gym.client.server_client import (
    ServerClient,
    ServerStatus,
)
from nemo_gym.config.config_types import (
    BaseServerConfig,
)
from nemo_gym.config.global_config import (
    get_global_config_dict,
)
from nemo_gym.server_runtime.server_utils import BaseServer, UvicornBacklogConfig, UvicornProxyHeadersConfig
from nemo_gym.telemetry.observer import observe_app_event_loop


DEFAULT_HEAD_SERVER_PORT = 11000


class HeadServer(BaseServer):
    config: BaseServerConfig
    _server_instances: List[dict] = []
    _ready: bool = PrivateAttr(default=False)
    # Serialized global config returned to clients.
    _cached_yaml: Optional[str] = None

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        @app.get("/livez", include_in_schema=False)
        @app.get("/", include_in_schema=False)
        async def _liveness():
            return {"status": "ok"}

        @app.get("/readyz", include_in_schema=False)
        @app.get("/healthz", include_in_schema=False)
        @app.get("/health", include_in_schema=False)
        async def _readiness(response: Response):
            if not self._ready:
                response.status_code = 503
                return {"status": "starting"}
            return {"status": "ok"}

        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)
        app.get("/server_instances")(self.get_server_instances)

        return app

    def mark_ready(self) -> None:
        self._ready = True

    def get_server_instances(self) -> List[dict]:
        return self._server_instances

    def set_server_instances(self, instances: List) -> None:
        self._server_instances = instances

    def invalidate_global_config_dict_yaml_cache(self) -> None:
        """Clear the serialized global config cache."""
        self._cached_yaml = None

    @classmethod
    def run_webserver(cls) -> Tuple[uvicorn.Server, Thread, "HeadServer"]:
        config = ServerClient.load_head_server_config()
        server = cls(config=config)
        global_config_dict = get_global_config_dict()
        uvicorn_proxy_cfg = UvicornProxyHeadersConfig.model_validate(global_config_dict)

        app = server.setup_webserver()
        # The head server runs on its own thread and event loop inside the supervisor process.
        observe_app_event_loop(app, "head_server")

        config = uvicorn.Config(
            app,
            host=server.config.host,
            port=server.config.port,
            proxy_headers=uvicorn_proxy_cfg.uvicorn_proxy_headers,
            forwarded_allow_ips=uvicorn_proxy_cfg.uvicorn_forwarded_allow_ips or [],
            backlog=UvicornBacklogConfig.model_validate(global_config_dict).uvicorn_backlog,
        )
        uvicorn_server = uvicorn.Server(config=config)

        thread = Thread(target=uvicorn_server.run, daemon=True)
        thread.start()

        return uvicorn_server, thread, server

    async def global_config_dict_yaml(self) -> str:
        if self._cached_yaml is None:
            self._cached_yaml = OmegaConf.to_yaml(get_global_config_dict())
        return self._cached_yaml


class ServerInstanceDisplayConfig(BaseModel):
    config_path: Optional[str] = None
    dir_path: Optional[Path] = None
    entrypoint: Optional[str] = None
    host: Optional[str] = None
    name: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    process_name: Optional[str] = None
    server_type: Optional[str] = None
    start_time: Optional[float] = None
    status: Optional[ServerStatus] = None
    uptime_seconds: Optional[float] = None
    url: Optional[str] = None
