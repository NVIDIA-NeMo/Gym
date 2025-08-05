from typing import List

import json

from pydantic import BaseModel

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)


class MultiNeedleResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetSynonymValueRequest(BaseModel):
    synonym: str


class GetSynonymValueResponse(BaseModel):
    synonym_value: int


class ExtractSynonymValuesRequest(BaseModel):
    synonym_values: List[int]


class ExtractSynonymValuesResponse(BaseModel):
    success: bool


class MultiNeedleRunRequest(BaseRunRequest):
    id: int
    expected_synonym_values: List[int]
    expected_synonyms: List[str]
    minefield_label: str
    minefield_label_value: int


class MultiNeedleVerifyRequest(MultiNeedleRunRequest, BaseVerifyRequest):
    pass


class MultiNeedleVerifyResponse(BaseVerifyResponse):
    parsed_synonym_values: List[int]
    accuracy: bool
    set_overlap: float
    original_term_minefield_hit: bool
    order_instruction_following_failure: bool


class MultiNeedleResourcesServer(SimpleResourcesServer):
    config: MultiNeedleResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/get_synonym_value")(self.get_synonym_value)
        app.post("/extract_synonym_values")(self.extract_synonym_values)

        return app

    async def get_synonym_value(
        self, body: GetSynonymValueRequest
    ) -> GetSynonymValueResponse:
        return GetSynonymValueResponse(synonym_value=sum(map(ord, body.synonym)))

    async def extract_synonym_values(
        self, body: ExtractSynonymValuesRequest
    ) -> ExtractSynonymValuesResponse:
        return ExtractSynonymValuesResponse(success=True)

    async def verify(self, body: MultiNeedleVerifyRequest) -> MultiNeedleVerifyResponse:
        expected = body.expected_synonym_values

        actual = []
        for output in reversed(body.response.output):
            if (
                output.type == "function_call"
                and output.name == "extract_synonym_values"
            ):
                actual = json.loads(output.arguments)["synonym_values"]
                break

        accuracy = expected == actual
        set_overlap = len(set(actual) & set(expected)) / len(expected)
        return MultiNeedleVerifyResponse(
            **body.model_dump(),
            reward=float(accuracy),
            parsed_synonym_values=actual,
            accuracy=accuracy,
            set_overlap=set_overlap,
            original_term_minefield_hit=body.minefield_label in actual
            or body.minefield_label_value in actual,
            order_instruction_following_failure=not accuracy and set_overlap == 1.0,
        )


if __name__ == "__main__":
    MultiNeedleResourcesServer.run_webserver()
