# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)


########################################
# Server references
#
# We enable servers to reference other servers. The way they do so is through these schemas below.
########################################


class ModelServerRef(BaseModel):
    type: Literal["responses_api_models"]
    name: str


class ResourcesServerRef(BaseModel):
    type: Literal["resources_servers"]
    name: str


class AgentServerRef(BaseModel):
    type: Literal["responses_api_agents"]
    name: str


ServerRef = Union[ModelServerRef, ResourcesServerRef, AgentServerRef]
ServerRefTypeAdapter = TypeAdapter(ServerRef)


def is_server_ref(config_dict: DictConfig) -> Optional[ServerRef]:
    try:
        return ServerRefTypeAdapter.validate_python(config_dict)
    except ValidationError:
        return None


########################################
# Dataset configs for handling and upload/download
########################################


class UploadJsonlDatasetGitlabConfig(BaseModel):
    dataset_name: str
    version: str  # Must be x.x.x
    input_jsonl_fpath: str


class JsonlDatasetGitlabIdentifer(BaseModel):
    dataset_name: str
    version: str
    artifact_fpath: str


class DownloadJsonlDatasetGitlabConfig(JsonlDatasetGitlabIdentifer):
    output_fpath: str


DatasetType = Union[Literal["train"], Literal["validation"], Literal["example"]]


class DatasetConfig(BaseModel):
    name: str
    type: DatasetType
    jsonl_fpath: str

    gitlab_identifier: Optional[JsonlDatasetGitlabIdentifer] = None
    license: Optional[Union[Literal["Apache 2.0"], Literal["TBD"]]] = None

    @model_validator(mode="after")
    def check_train_validation_sets(self) -> "DatasetConfig":
        if self.type in ["train", "validation"]:
            assert self.gitlab_identifier is not None, f"A Gitlab path is required for {self.name}"
            assert self.license is not None, f"A license is required for {self.name}"

        return self


########################################
# Base server config classes
########################################


class BaseServerConfig(BaseModel):
    host: str
    port: int


class BaseRunServerConfig(BaseServerConfig):
    entrypoint: str


class BaseRunServerInstanceConfig(BaseRunServerConfig):
    name: str  # This name is unique at runtime.


########################################
# Server type and server instance configs
########################################


class BaseRunServerTypeConfig(BaseRunServerConfig):
    model_config = ConfigDict(extra="allow")

    host: Optional[str] = None
    port: Optional[int] = None

    datasets: Optional[List[DatasetConfig]] = None


class BaseServerTypeConfig(BaseModel):
    SERVER_TYPE: ClassVar[
        Union[
            Literal["responses_api_models"],
            Literal["resources_servers"],
            Literal["responses_api_agents"],
        ]
    ]


class ResponsesAPIModelServerTypeConfig(BaseServerTypeConfig):
    SERVER_TYPE: ClassVar[Literal["responses_api_models"]] = "responses_api_models"

    model_config = ConfigDict(extra="allow")

    responses_api_models: Dict[str, BaseRunServerTypeConfig] = Field(min_length=1, max_length=1)


class ResourcesServerTypeConfig(BaseServerTypeConfig):
    SERVER_TYPE: ClassVar[Literal["resources_servers"]] = "resources_servers"

    model_config = ConfigDict(extra="allow")

    resources_servers: Dict[str, BaseRunServerTypeConfig] = Field(min_length=1, max_length=1)


class ResponsesAPIAgentServerTypeConfig(BaseServerTypeConfig):
    SERVER_TYPE: ClassVar[Literal["responses_api_agents"]] = "responses_api_agents"

    model_config = ConfigDict(extra="allow")

    responses_api_agents: Dict[str, BaseRunServerTypeConfig] = Field(min_length=1, max_length=1)


ServerTypeConfig = Union[
    ResponsesAPIModelServerTypeConfig,
    ResourcesServerTypeConfig,
    ResponsesAPIAgentServerTypeConfig,
]


class BaseServerInstanceConfig(BaseServerTypeConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    server_type_config_dict: DictConfig = Field(exclude=True)

    def get_server_ref(self) -> ServerRef:
        return is_server_ref({"type": self.SERVER_TYPE, "name": self.name})

    def get_inner_run_server_config_dict(self) -> DictConfig:
        server_type_name = list(getattr(self, self.SERVER_TYPE))[0]
        return self.server_type_config_dict[self.SERVER_TYPE][server_type_name]

    def get_inner_run_server_config(self) -> BaseRunServerTypeConfig:
        return list(getattr(self, self.SERVER_TYPE).values())[0]

    @property
    def datasets(self) -> Optional[List[DatasetConfig]]:
        return self.get_inner_run_server_config().datasets


class ResponsesAPIModelServerInstanceConfig(ResponsesAPIModelServerTypeConfig, BaseServerInstanceConfig):
    pass


class ResourcesServerInstanceConfig(ResourcesServerTypeConfig, BaseServerInstanceConfig):
    pass


class ResponsesAPIAgentServerInstanceConfig(ResponsesAPIAgentServerTypeConfig, BaseServerInstanceConfig):
    pass


ServerInstanceConfig = Union[
    ResponsesAPIModelServerInstanceConfig,
    ResourcesServerInstanceConfig,
    ResponsesAPIAgentServerInstanceConfig,
]
ServerInstanceConfigTypeAdapter = TypeAdapter(ServerInstanceConfig)


def maybe_get_server_instance_config(name: str, server_type_config_dict: Any) -> Optional[ServerInstanceConfig]:
    if not isinstance(server_type_config_dict, DictConfig):
        return None

    maybe_server_instance_config_dict = {
        "name": name,
        "server_type_config_dict": server_type_config_dict,
        **OmegaConf.to_container(server_type_config_dict),
    }
    try:
        return ServerInstanceConfigTypeAdapter.validate_python(maybe_server_instance_config_dict)
    except ValidationError:
        return None


########################################
# Train dataset configs
########################################

AGENT_REF_KEY = "agent_ref"
