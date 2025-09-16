from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MiniSweResourceResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MiniSweResourceResourcesServer(SimpleResourcesServer):
    config: MiniSweResourceResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    MiniSweResourceResourcesServer.run_webserver()
