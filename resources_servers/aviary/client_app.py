from pydantic import model_validator

from aviary.core import TaskDatasetClient, TaskEnvironmentClient
from resources_servers.aviary.app import AviaryResourcesServer, AviaryResourcesServerConfig


class AviaryClientAppConfig(AviaryResourcesServerConfig):
    server_url: str
    request_timeout: float | None = 300.0
    api_key: str | None = None


class AviaryClientApp(AviaryResourcesServer[TaskEnvironmentClient, TaskDatasetClient]):
    config: AviaryClientAppConfig
    dataset: TaskDatasetClient

    @model_validator(mode="before")
    @classmethod
    def load_dataset(cls, data: dict) -> dict:
        if "dataset" not in data:
            config = data["config"] = AviaryClientAppConfig.model_validate(data.get("config", {}))
            data["dataset"] = TaskDatasetClient(
                server_url=config.server_url,
                request_timeout=config.request_timeout,
                api_key=config.api_key,
            )
        return data


if __name__ == "__main__":
    AviaryClientApp.run_webserver()
