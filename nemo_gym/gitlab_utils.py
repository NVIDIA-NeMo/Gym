from os import environ

from pathlib import Path

import requests

from pydantic import BaseModel

from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_TRACKING_TOKEN
from mlflow.artifacts import get_artifact_repository
from mlflow.exceptions import RestException

from nemo_gym.config_types import (
    UploadJsonlDatasetGitlabConfig,
    DownloadJsonlDatasetGitlabConfig,
)
from nemo_gym.server_utils import get_global_config_dict


class MLFlowConfig(BaseModel):
    mlflow_tracking_uri: str
    mlflow_tracking_token: str


def create_mlflow_client() -> MlflowClient:  # pragma: no cover
    global_config = get_global_config_dict()
    config = MLFlowConfig.model_validate(global_config)

    environ["MLFLOW_TRACKING_TOKEN"] = config.mlflow_tracking_token
    client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)

    return client


def upload_jsonl_dataset(
    config: UploadJsonlDatasetGitlabConfig,
) -> None:  # pragma: no cover
    client = create_mlflow_client()

    try:
        client.create_registered_model(config.dataset_name)
    except RestException:
        pass

    tags = {"gitlab.version": config.version}
    try:
        model_version = client.get_model_version(config.dataset_name, config.version)
    except RestException:
        model_version = client.create_model_version(
            config.dataset_name, config.version, tags=tags
        )

    run_id = model_version.run_id
    client.log_artifact(run_id, config.input_jsonl_fpath, artifact_path="")

    filename = Path(config.input_jsonl_fpath).name
    print(f"""Download this artifact:
ng_download_dataset_from_gitlab \\
    +run_id={run_id} \\
    +artifact_fpath={filename} \\
    +output_fpath=<your output fpath>
""")


def upload_jsonl_dataset_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetGitlabConfig.model_validate(global_config)
    upload_jsonl_dataset(config)


def download_jsonl_dataset(
    config: DownloadJsonlDatasetGitlabConfig,
) -> None:  # pragma: no cover
    # TODO: There is probably a much better way to do this, but it is not clear at the moment.
    client = create_mlflow_client()

    model_version = client.get_model_version(config.dataset_name, config.version)
    run_id = model_version.run_id
    repo = get_artifact_repository(
        artifact_uri=f"runs:/{run_id}", tracking_uri=client.tracking_uri
    )
    artifact_uri = repo.repo.artifact_uri
    download_link = f"{artifact_uri.rstrip('/')}/{config.artifact_fpath.lstrip('/')}"

    response = requests.get(
        download_link,
        headers={"Authorization": f"Bearer {MLFLOW_TRACKING_TOKEN.get()}"},
    )
    with open(config.output_fpath, "w") as f:
        f.write(response.content.decode())


def download_jsonl_dataset_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = DownloadJsonlDatasetGitlabConfig.model_validate(global_config)
    download_jsonl_dataset(config)
