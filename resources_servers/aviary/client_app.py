import os

from pydantic import field_validator, model_validator

from aviary.core import TaskDatasetClient, TaskEnvironmentClient
from resources_servers.aviary.app import AviaryResourcesServer, AviaryResourcesServerConfig


class AviaryClientResourcesServerConfig(AviaryResourcesServerConfig):
    server_url: str | None = None
    request_timeout: float | None = 300.0
    api_key: str | None = None

    @field_validator("server_url")
    @classmethod
    def validate_server_url(cls, v: str | None) -> str | None:
        if v is None:
            return os.getenv("AVIARY_SERVER_URL")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        if v is None:
            # Note that if AVIARY_API_KEY is not set, we will return
            # None, which is also valid - assuming the server does
            # not have auth enabled.
            return os.getenv("AVIARY_SERVER_API_KEY")
        return v


class AviaryClientResourcesServer(AviaryResourcesServer[TaskEnvironmentClient, TaskDatasetClient]):
    config: AviaryClientResourcesServerConfig
    dataset: TaskDatasetClient

    @model_validator(mode="before")
    @classmethod
    def load_dataset(cls, data: dict) -> dict:
        if "dataset" not in data:
            config = data["config"] = AviaryClientResourcesServerConfig.model_validate(data.get("config", {}))
            data["dataset"] = TaskDatasetClient(
                server_url=config.server_url,
                request_timeout=config.request_timeout,
                api_key=config.api_key,
                catch_http_errors=True,
            )
        return data


if __name__ == "__main__":
    AviaryClientResourcesServer.run_webserver()
