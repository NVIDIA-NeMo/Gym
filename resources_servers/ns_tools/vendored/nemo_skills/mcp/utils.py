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

import importlib
import importlib.util
import json
import logging
import os
import tempfile
from typing import Optional

from mcp import StdioServerParameters
from mcp.types import CallToolResult
from omegaconf import DictConfig, OmegaConf

from .clients import MCPStdioClient, MCPStreamableHttpClient

logger = logging.getLogger(__name__)


def locate(path):
    # If it's not a string, assume already an object and return
    if not isinstance(path, str):
        return path

    # Double-colon syntax for both modules and files: module::Class or path.py::Class
    if "::" in path:
        left, attr_name = path.split("::", 1)
        # If it's a file path (endswith .py or exists), load from file; otherwise import module
        if left.endswith(".py") or os.path.exists(left):
            module_name = os.path.splitext(os.path.basename(left))[0]
            spec = importlib.util.spec_from_file_location(module_name, left)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, attr_name)
        module = importlib.import_module(left)
        return getattr(module, attr_name)

    # Handle standard dotted module path
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def exa_auth_connector(client: MCPStreamableHttpClient):
    client.base_url = f"{client.base_url}?exaApiKey={os.getenv('EXA_API_KEY')}"


def exa_stdio_connector(client: MCPStdioClient):
    client.server_params = StdioServerParameters(
        command=client.server_params.command,
        args=list(client.server_params.args) + ["--exa-api-key", os.getenv("EXA_API_KEY", "")],
    )


def exa_output_formatter(result: CallToolResult):
    if getattr(result, "isError", False):
        logger.error(f"Exa error: {result}")
        return result.content[0].text
    return json.loads(result.content[0].text)["results"]


def hydra_config_connector_factory(config_obj):
    """Return a connector that writes the provided OmegaConf config to a temp YAML and injects Hydra args.

    Assumes `config_obj` is an OmegaConf DictConfig; we just OmegaConf.save and inject
    `--config-dir`/`--config-name`. Path and argument management only.
    """

    def connector(client: MCPStdioClient):
        temp_dir = tempfile.mkdtemp(prefix="mcp_cfg_")
        cfg_path = os.path.join(temp_dir, "config.yaml")
        OmegaConf.save(config_obj, cfg_path, resolve=True)

        client.server_params = StdioServerParameters(
            command=client.server_params.command,
            args=list(client.server_params.args) + ["--config-dir", temp_dir, "--config-name", "config"],
            env=client.server_params.env,
        )

    return connector


def load_mcp_config(
    config: Optional[str] = None,
    config_dir: Optional[str] = None,
    config_name: str = "config",
) -> DictConfig:
    """Load an OmegaConf config for MCP servers, supporting `--config` and Hydra-like flags. No fallback.

    Precedence: config > (config_dir + config_name). If none found, raises ValueError.

    Args:
        config: Path to YAML config file.
        config_dir: Directory containing the config file (Hydra-compatible).
        config_name: Config filename stem without extension.

    Returns:
        DictConfig loaded from the discovered source or created from default.
    """

    # Determine config path if not explicitly provided
    config_path: Optional[str] = config
    if not config_path and config_dir:
        candidate_yaml = os.path.join(config_dir, f"{config_name}.yaml")
        candidate_yml = os.path.join(config_dir, f"{config_name}.yml")
        if os.path.exists(candidate_yaml):
            config_path = candidate_yaml
        elif os.path.exists(candidate_yml):
            config_path = candidate_yml

    if config_path and os.path.exists(config_path):
        return OmegaConf.load(config_path)

    raise ValueError("No configuration file found. Provide --config or --config-dir/--config-name.")


def add_config_args(parser):
    """Attach standard config args to an ArgumentParser: --config, --config-dir, --config-name."""
    parser.add_argument("--config", dest="config", type=str, required=False, help="Path to OmegaConf YAML file")
    parser.add_argument(
        "--config-dir",
        dest="config_dir",
        type=str,
        required=False,
        help="Directory containing config file (Hydra-compatible)",
    )
    parser.add_argument(
        "--config-name",
        dest="config_name",
        type=str,
        default="config",
        help="Config file name without extension (Hydra-compatible)",
    )
    return parser
