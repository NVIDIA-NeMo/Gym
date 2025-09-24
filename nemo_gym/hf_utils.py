from os import environ
from pathlib import Path

from huggingface_hub import HfApi, RepositoryNotFoundError
from pydantic import BaseModel

from nemo_gym.config_types import UploadJsonlDatasetHuggingFaceConfig
from nemo_gym.server_utils import get_global_config_dict


class HuggingFaceConfig(BaseModel):
    huggingface_token: str


def create_huggingface_client() -> HfApi:  # pragma: no cover
    global_config = get_global_config_dict()
    config = HuggingFaceConfig.model_validate(global_config)

    environ["HF_TOKEN"] = config.huggingface_token
    client = HfApi()

    return client


def upload_jsonl_dataset_to_hf(
    config: UploadJsonlDatasetHuggingFaceConfig,
) -> None:  # pragma: no cover
    client = create_huggingface_client()

    try:
        client.create_repo(config.dataset_name, token=config.huggingface_token)
    except RepositoryNotFoundError:
        pass

    repo_id = f"{config.hf_username}/{config.dataset_name}"
    client.upload_file(
        path_or_fileobj=config.input_jsonl_fpath,
        path_in_repo=Path(config.input_jsonl_fpath).name,
        repo_id=repo_id,
        token=config.huggingface_token,
    )

    print(f"""Download this artifact:
hf_hub_download \\
    +repo_id={repo_id} \\
    +filename={Path(config.input_jsonl_fpath).name} \\
    +output_path={config.input_jsonl_fpath}
""")


def upload_jsonl_dataset_to_hf_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetHuggingFaceConfig.model_validate(global_config)
    upload_jsonl_dataset_to_hf(config)
