"""Standalone HeadServer entrypoint for Kubernetes deployment."""

import uvicorn

from nemo_gym.server_utils import HeadServer, ServerClient


config = ServerClient.load_head_server_config()
server = HeadServer(config=config)
app = server.setup_webserver()

uvicorn.run(app, host=config.host, port=config.port)
