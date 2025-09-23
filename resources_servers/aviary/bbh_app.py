import logging

from kestrel.env import DataAnalysisEnv, DatasetConfig, HypothesisDataset
from pydantic import model_validator

from resources_servers.aviary.app import AviaryResourcesServer, AviaryResourcesServerConfig


logger = logging.getLogger(__name__)


class BBHConfig(AviaryResourcesServerConfig):
    dataset: DatasetConfig


class BBHResourcesServer(AviaryResourcesServer[DataAnalysisEnv, HypothesisDataset]):
    config: BBHConfig
    dataset: HypothesisDataset

    @model_validator(mode="before")
    @classmethod
    def init_dataset(cls, data: dict) -> dict:
        if "dataset" not in data:
            config = BBHConfig.model_validate(data.get("config", {}))
            data["dataset"] = HypothesisDataset(config.dataset)

        return data


if __name__ == "__main__":
    BBHResourcesServer.run_webserver()
