import argparse
import asyncio
import os
import random
import shutil
import socket
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Self, cast
from uuid import UUID

import yaml
from aviary.core import TaskDataset, TaskDatasetServer
from lmi import LiteLLMModel
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, field_validator, model_validator

from hypotest.dataset_server import HypotestDatasetConfig, HypotestDataset
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance
from hypotest.env.kernel_server import NBLanguage
from resources_servers.aviary.app import AviaryResourcesServer
from resources_servers.aviary.schemas import AviaryResourcesServerConfig

class HypotestServerConfig(AviaryResourcesServerConfig):
    # dataset config
    dataset: HypotestDatasetConfig

class HypotestResourcesServer(AviaryResourcesServer[InterpreterEnv, HypotestDataset]):
    config: HypotestServerConfig
    dataset: HypotestDataset

    @model_validator(mode="before")
    @classmethod
    def load_dataset(cls, data: dict) -> dict:
        if "dataset" not in data:
            config = data['config'] = HypotestServerConfig.model_validate(data.get("config", {}))
            data["dataset"] = HypotestDataset(config.dataset)
        return data

if __name__ == "__main__":
    HypotestResourcesServer.run_webserver()

