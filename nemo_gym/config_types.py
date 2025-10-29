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
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import rich
from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from rich.text import Text


########################################
# Base CLI configs
########################################


class BaseNeMoGymCLIConfig(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data):
        if not (data.get("h") or data.get("help")):
            return data

        rich.print(f"""Displaying help for [bold]{cls.__name__}[/bold]
""")
        # We use __doc__ directly here since inspect.getdoc will inherit the doc from parent classes.
        class_doc = cls.__doc__
        if class_doc:
            rich.print(f"""[bold]Description[/bold]
-----------
{class_doc.strip()}
""")

        fields = cls.model_fields.items()
        if fields:
            rich.print("""[bold]Parameters[/bold]
----------""")

            prefixes: List[Text] = []
            suffixes: List[Text] = []
            for field_name, field in fields:
                description_str = field.description if field.description else ""

                # Not sure if there is a better way to get this annotation_str, e.g. using typing.get_args or typing.get_origin
                annotation_str = (
                    field.annotation.__name__ if isinstance(field.annotation, type) else str(field.annotation)
                )
                annotation_str = annotation_str.replace("typing.", "")

                prefixes.append(Text.from_markup(f"- [blue]{field_name}[/blue] [yellow]({annotation_str})[/yellow]"))
                suffixes.append(description_str)

            max_prefix_length = max(map(len, prefixes))
            ljust_length = max_prefix_length + 3
            for prefix, suffix in zip(prefixes, suffixes):
                prefix.align("left", ljust_length)
                rich.print(prefix + suffix)
        else:
            print("There are no arguments to this CLI command!")

        # Exit after help is printed.
        exit()


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


class UploadJsonlDatasetGitlabConfig(BaseNeMoGymCLIConfig):
    """
    Upload a local jsonl dataset artifact to Gitlab.
    """

    dataset_name: str = Field(description="The dataset name.")
    version: str = Field(description="The version of this dataset. Must be in the format `x.x.x`.")
    input_jsonl_fpath: str = Field(description="Path to the jsonl file to upload.")


class JsonlDatasetGitlabIdentifer(BaseModel):
    dataset_name: str
    version: str
    artifact_fpath: str


class DownloadJsonlDatasetGitlabConfig(JsonlDatasetGitlabIdentifer, BaseNeMoGymCLIConfig):
    dataset_name: str = Field(description="The dataset name.")
    version: str = Field(description="The version of this dataset. Must be in the format `x.x.x`.")
    artifact_fpath: str = Field(description="The filepath to the artifact to download.")
    output_fpath: str = Field(description="Where to save the downloaded dataset.")


DatasetType = Union[Literal["train"], Literal["validation"], Literal["example"]]


class DatasetConfig(BaseModel):
    name: str
    type: DatasetType
    jsonl_fpath: str

    num_repeats: int = Field(default=1, ge=1)
    gitlab_identifier: Optional[JsonlDatasetGitlabIdentifer] = None
    license: Optional[
        Union[
            Literal["Apache 2.0"],
            Literal["MIT"],
            Literal["Creative Commons Attribution 4.0 International"],
            Literal["Creative Commons Attribution-ShareAlike 4.0 International"],
            Literal["TBD"],
        ]
    ] = None

    @model_validator(mode="after")
    def check_train_validation_sets(self) -> "DatasetConfig":
        if self.type in ["train", "validation"]:
            assert self.gitlab_identifier is not None, f"A Gitlab path is required for {self.name}"
            assert self.license is not None, f"A license is required for {self.name}"

        return self


########################################
# Base server config classes
########################################


class Domain(str, Enum):
    MATH = "math"
    CODING = "coding"
    AGENT = "agent"
    KNOWLEDGE = "knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"
    LONG_CONTEXT = "long_context"
    SAFETY = "safety"
    GAMES = "games"
    E2E = "e2e"
    OTHER = "other"


class BaseServerConfig(BaseModel):
    host: str
    port: int


class BaseRunServerConfig(BaseServerConfig):
    entrypoint: str
    domain: Optional[Domain] = None  # Only required for resource servers

    @model_validator(mode="after")
    def validate_domain(self) -> "BaseRunServerTypeConfig":
        name = getattr(self, "name", None)
        if name and self.name.endswith("_resources_server"):
            assert self.domain is not None, "A domain is required for resource servers."
        else:
            if hasattr(self, "domain"):
                del self.domain

        return self


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
