# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Tuple, Union

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
from pydantic_core import PydanticUndefined
from rich.markdown import Markdown
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
            # Render docstring as Markdown
            md = Markdown(class_doc.strip())
            rich.print(md)

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

                # Add default value information if available
                if field.default is not PydanticUndefined and field.default is not None:
                    default_str = f" [default: {field.default}]"
                    description_str = description_str + default_str if description_str else default_str.strip()
                elif field.default_factory is not None:
                    default_str = " [default: <factory>]"
                    description_str = description_str + default_str if description_str else default_str.strip()
                elif field.default is PydanticUndefined and field.is_required():
                    default_str = " [required]"
                    description_str = description_str + default_str if description_str else default_str.strip()

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

    Examples:

    ```bash
    ng_upload_dataset_to_gitlab \
        +dataset_name=example_multi_step \
        +version=0.0.1 \
        +input_jsonl_fpath=data/train.jsonl
    ```
    """

    dataset_name: str = Field(description="The dataset name.")
    version: str = Field(description="The version of this dataset. Must be in the format `x.x.x`.")
    input_jsonl_fpath: str = Field(description="Path to the jsonl file to upload.")


class JsonlDatasetGitlabIdentifer(BaseModel):
    dataset_name: str
    version: str
    artifact_fpath: str


class DownloadJsonlDatasetGitlabConfig(JsonlDatasetGitlabIdentifer, BaseNeMoGymCLIConfig):
    """
    Download a JSONL dataset from GitLab Model Registry.

    Examples:

    ```bash
    ng_download_dataset_from_gitlab \
        +dataset_name=example_multi_step \
        +version=0.0.1 \
        +artifact_fpath=train.jsonl \
        +output_fpath=data/train.jsonl
    ```
    """

    dataset_name: str = Field(description="The dataset name.")
    version: str = Field(description="The version of this dataset. Must be in the format `x.x.x`.")
    artifact_fpath: str = Field(description="The filepath to the artifact to download.")
    output_fpath: str = Field(description="Where to save the downloaded dataset.")


class DeleteJsonlDatasetGitlabConfig(BaseNeMoGymCLIConfig):
    """
    Delete a dataset from GitLab Model Registry (prompts for confirmation).

    Examples:

    ```bash
    ng_delete_dataset_from_gitlab +dataset_name=old_dataset
    ```
    """

    dataset_name: str = Field(description="Name of the dataset to delete from GitLab.")


class JsonlDatasetHuggingFaceIdentifer(BaseModel):
    repo_id: str = Field(description="The repo id.")
    artifact_fpath: Optional[str] = Field(
        default=None,
        description="Path to specific file in HuggingFace repo (e.g., 'train.jsonl'). If omitted, load_dataset will be used with split.",
    )


class BaseUploadJsonlDatasetHuggingFaceConfig(BaseNeMoGymCLIConfig):
    """
    Upload a JSONL dataset to HuggingFace Hub with automatic naming based on domain and resource server.

    Examples:

    ```bash
    resource_config_path="resources_servers/example_multi_step/configs/example_multi_step.yaml"
    ng_upload_dataset_to_hf \
        +dataset_name=my_dataset \
        +input_jsonl_fpath=data/train.jsonl \
        +resource_config_path=${resource_config_path}
    ```
    """

    hf_token: str = Field(description="HuggingFace API token for authentication.")
    hf_organization: str = Field(description="HuggingFace organization name where dataset will be uploaded.")
    hf_collection_name: str = Field(description="HuggingFace collection name for organizing datasets.")
    hf_collection_slug: str = Field(description="Alphanumeric collection slug found at the end of collection URI.")
    dataset_name: Optional[str] = Field(
        default=None, description="Name of the dataset (will be combined with domain and resource server name)."
    )
    input_jsonl_fpath: str = Field(description="Path to the local jsonl file to upload.")
    resource_config_path: str = Field(
        description="Path to resource server config file (used to extract domain for naming convention)."
    )
    hf_dataset_prefix: str = Field(
        default="Nemotron-RL", description="Prefix prepended to dataset name (default: 'NeMo-Gym')."
    )
    split: Literal["train", "validation", "test"] = Field(
        default="train",
        description="Dataset split type (e.g., 'train', 'validation', 'test'). Format validation only applies to 'train' splits.",
    )
    create_pr: bool = Field(
        default=False,
        description="Create a pull request instead of pushing directly. Required for repos where you do not have write access.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Git revision (branch name) to upload to. Use the same revision for multiple files to upload to the same PR. If not provided with create_pr=True, a new branch/PR will be created automatically.",
    )
    commit_message: Optional[str] = Field(
        default=None, description="Custom commit message. If not provided, HuggingFace auto-generates one."
    )
    commit_description: Optional[str] = Field(
        default=None, description="Optional commit description with additional context."
    )


class UploadJsonlDatasetHuggingFaceConfig(BaseUploadJsonlDatasetHuggingFaceConfig):
    """
    Upload a JSONL dataset to HuggingFace Hub and automatically delete from GitLab after successful upload.

    This command always deletes the dataset from GitLab after uploading to HuggingFace.
    Use `ng_upload_dataset_to_hf` if you want optional deletion control.

    Examples:

    ```bash
    resource_config_path="resources_servers/example_multi_step/configs/example_multi_step.yaml"
    ng_gitlab_to_hf_dataset \
        +dataset_name=my_dataset \
        +input_jsonl_fpath=data/train.jsonl \
        +resource_config_path=${resource_config_path}
    ```
    """

    forbidden_fields: ClassVar[Set[str]] = {"delete_from_gitlab"}

    @model_validator(mode="before")
    def check_forbidden_fields(cls, data):
        if isinstance(data, dict) or hasattr(data, "keys"):
            forbidden = cls.forbidden_fields.intersection(set(data.keys()))
            if forbidden:
                raise ValueError(f"Forbidden fields present: {forbidden}")
        return data


class UploadJsonlDatasetHuggingFaceMaybeDeleteConfig(BaseUploadJsonlDatasetHuggingFaceConfig):
    """
    Upload a JSONL dataset to HuggingFace Hub with optional GitLab deletion after successful upload.

    Examples:

    ```bash
    resource_config_path="resources_servers/example_multi_step/configs/example_multi_step.yaml"
    ng_upload_dataset_to_hf \
        +dataset_name=my_dataset \
        +input_jsonl_fpath=data/train.jsonl \
        +resource_config_path=${resource_config_path} \
        +delete_from_gitlab=true
    ```
    """

    delete_from_gitlab: Optional[bool] = Field(
        default=False, description="Delete the dataset from GitLab after successful upload to HuggingFace."
    )


class DownloadJsonlDatasetHuggingFaceConfig(JsonlDatasetHuggingFaceIdentifer, BaseNeMoGymCLIConfig):
    """
    Download a JSONL dataset from HuggingFace Hub to local filesystem.

    Examples:

    ```bash
    ng_download_dataset_from_hf \
        +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
        +artifact_fpath=train.jsonl \
        +output_fpath=data/train.jsonl
    ```
    """

    output_dirpath: Optional[str] = Field(
        default=None,
        description="Directory to save the downloaded dataset. Files will be named {split}.jsonl. If split is omitted, all available splits are downloaded.",
    )
    output_fpath: Optional[str] = Field(
        default=None,
        description="Exact local file path where the downloaded dataset will be saved. Requires `artifact_fpath` or `split`. Overrides output_dirpath.",
    )
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token for authentication.")
    split: Optional[Literal["train", "validation", "test"]] = Field(
        default=None, description="Dataset split to download. Omit to download all available splits."
    )

    @model_validator(mode="after")
    def check_output_path(self) -> "DownloadJsonlDatasetHuggingFaceConfig":
        if not self.output_dirpath and not self.output_fpath:
            raise ValueError("Either output_dirpath or output_fpath must be provided")
        if self.output_dirpath and self.output_fpath:
            raise ValueError("Cannot specify both output_dirpath and output_fpath")
        if self.artifact_fpath and self.split:
            raise ValueError(
                "Cannot specify both artifact_fpath and split. Use artifact_fpath for targeting a raw file, or split for structured datasets."
            )
        # Prevent output_fpath without split when not using artifact_fpath
        if self.output_fpath and not self.split and not self.artifact_fpath:
            raise ValueError(
                "When using output_fpath without artifact_fpath, split must be specified. Use output_dirpath to download all splits."
            )
        return self


DatasetType = Union[Literal["train"], Literal["validation"], Literal["example"]]


class DatasetConfig(BaseModel):
    name: str
    type: DatasetType
    jsonl_fpath: str

    num_repeats: int = Field(default=1, ge=1)
    gitlab_identifier: Optional[JsonlDatasetGitlabIdentifer] = None
    huggingface_identifier: Optional[JsonlDatasetHuggingFaceIdentifer] = None
    license: Optional[
        Union[
            Literal["Apache 2.0"],
            Literal["MIT"],
            Literal["Creative Commons Attribution 4.0 International"],
            Literal["Creative Commons Attribution-ShareAlike 4.0 International"],
            Literal["NVIDIA Internal Use Only, Do Not Distribute"],
            Literal["TBD"],
            Literal["GNU General Public License v3.0"],
        ]
    ] = None

    @model_validator(mode="after")
    def check_train_validation_sets(self) -> "DatasetConfig":
        if self.type in ["train", "validation"]:
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
    TRANSLATION = "translation"
    E2E = "e2e"
    OTHER = "other"


class BaseServerConfig(BaseModel):
    host: str
    port: int
    num_workers: Optional[int] = None


class BaseRunServerConfig(BaseServerConfig):
    entrypoint: str
    domain: Optional[Domain] = None  # Only required for resource servers


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

    @model_validator(mode="after")
    def validate_domain_for_resource_server(self) -> "BaseServerInstanceConfig":
        config = self.get_inner_run_server_config()
        if self.SERVER_TYPE == "resources_servers":
            assert config.domain is not None, "A domain is required for resource servers."
        else:
            # Remove domain field from Model and Agent servers.
            if hasattr(config, "domain"):
                del config.domain
        return self

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


def maybe_get_server_instance_config(
    name: str, server_type_config_dict: Any
) -> Tuple[Optional[ServerInstanceConfig], Optional[ValidationError]]:
    """Returns ServerInstanceConfig if a valid server, otherwise None with error details"""
    if not isinstance(server_type_config_dict, DictConfig):
        return None, None

    maybe_server_instance_config_dict = {
        "name": name,
        "server_type_config_dict": server_type_config_dict,
        **OmegaConf.to_container(server_type_config_dict),
    }
    try:
        config = ServerInstanceConfigTypeAdapter.validate_python(maybe_server_instance_config_dict)
        return config, None
    except ValidationError as e:
        return None, e


def is_almost_server(server_type_config_dict: Any) -> bool:
    """Detects if a config looks like a server but might fail validation."""
    from nemo_gym.global_config import ENTRYPOINT_KEY_NAME

    if not isinstance(server_type_config_dict, DictConfig):
        return False

    # Check for server type.
    server_type_keys = ["responses_api_models", "resources_servers", "responses_api_agents"]
    has_server_type = any(key in server_type_config_dict for key in server_type_keys)

    if not has_server_type:
        return False

    # Check for entrypoint presence.
    for server_type_key in server_type_keys:
        if server_type_key in server_type_config_dict:
            inner_dict = server_type_config_dict[server_type_key]
            if isinstance(inner_dict, DictConfig):
                for config in inner_dict.values():
                    if isinstance(config, DictConfig) and ENTRYPOINT_KEY_NAME in config:
                        return True

    return False


########################################
# Train dataset configs
########################################

AGENT_REF_KEY = "agent_ref"
