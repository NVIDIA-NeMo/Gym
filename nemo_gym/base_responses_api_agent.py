from abc import abstractmethod

from pydantic import BaseModel

from fastapi import FastAPI, Body

import uvicorn

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)


class BaseResponsesAPIAgentConfig(BaseModel):
    host: str
    port: int


class BaseResponsesAPIAgent(BaseModel):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        app.post("/v1/responses")(self.responses)

        return app

    def run_webserver(self) -> None:  # pragma: no cover
        app = self.setup_webserver()

        uvicorn.run(
            app,
            host=self.config.host,
            port=self.config.port,
            # TODO eventually we want to make this FastAPI server served across multiple processes or workers.
            # Right now this will always use one process.
            # workers=self.config.num_fastapi_workers,
            # We don't have any explicit lifespan logic, so instead of defaulting to "auto"
            # We just turn lifespan off
            lifespan="off",
        )

    # TODO: right now there is no validation on the TypedDict NeMoGymResponseCreateParamsNonStreaming
    # We should explicitly add validation at this server level or we should explicitly not validate so that there is flexibility in this API.
    @abstractmethod
    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        pass


class StandalonePythonResponsesAPIAgent(SimpleResponsesAPIAgent):
    pass
