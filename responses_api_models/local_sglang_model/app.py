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
import atexit
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    find_open_port,
    get_global_config_dict,
    get_hf_token,
)
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


class LocalSGLangModelConfig(SGLangModelConfig):
    # Set after the local SGLang endpoint starts.
    base_url: Union[str, List[str]] = Field(default_factory=list)
    # Not used on local deployments.
    api_key: str = "dummy"  # pragma: allowlist secret

    # Optional separate model path. If unset, `model` is used as both the
    # SGLang --model-path and the OpenAI served model name.
    model_path: Optional[str] = None
    hf_home: Optional[str] = None
    sglang_env_prefix: Optional[str] = None
    sglang_python: Optional[str] = None
    sglang_serve_kwargs: Dict[str, Any]
    sglang_serve_env_vars: Dict[str, str] = Field(default_factory=dict)
    startup_timeout_seconds: int = 1200
    log_file: Optional[str] = None
    debug: bool = False

    def model_post_init(self, context):
        if not self.hf_home:
            self.hf_home = str(Path.cwd() / ".cache" / "huggingface")
        return super().model_post_init(context)


class GetInnerSGLangConfigResponse(BaseModel):
    base_url: List[str]
    api_key: str
    model: str


class LocalSGLangModel(SGLangModel):
    config: LocalSGLangModelConfig

    _sglang_process: subprocess.Popen
    _sglang_log_handle: Optional[Any] = None

    def setup_webserver(self):
        print("Starting SGLang server. This will take a few minutes...")
        self.start_sglang_server()

        app = super().setup_webserver()
        app.get("/get_inner_sglang_config")(self.get_inner_sglang_config)

        return app

    async def get_inner_sglang_config(self) -> GetInnerSGLangConfigResponse:
        return GetInnerSGLangConfigResponse(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
        )

    def get_cache_dir(self) -> str:
        return str(Path(self.config.hf_home) / "hub")

    def _resolve_sglang_python(self) -> str:
        if self.config.sglang_python:
            return self.config.sglang_python
        if self.config.sglang_env_prefix:
            return str(Path(self.config.sglang_env_prefix) / "bin" / "python")
        return sys.executable

    def _configure_sglang_serve(self) -> Tuple[List[str], Dict[str, str], int]:
        port = find_open_port(disallowed_ports=get_global_config_dict()[DISALLOWED_PORTS_KEY_NAME])
        server_args = dict(self.config.sglang_serve_kwargs)
        server_args = LocalSGLangModel._normalize_server_args(server_args)

        server_args.setdefault("model_path", self.config.model_path or self.config.model)
        server_args.setdefault("served_model_name", self.config.model)
        server_args.setdefault("host", "0.0.0.0")
        server_args.setdefault("port", port)

        env_vars = {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(self.config.hf_home),
            "PYTHONFAULTHANDLER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        maybe_hf_token = get_hf_token()
        if maybe_hf_token:
            env_vars["HF_TOKEN"] = maybe_hf_token
        env_vars.update(LocalSGLangModel._env_prefix_vars(self))
        env_vars.update(LocalSGLangModel._default_local_cache_vars())
        env_vars.update(self.config.sglang_serve_env_vars)

        cmd = [LocalSGLangModel._resolve_sglang_python(self), "-m", "sglang.launch_server"]
        cmd.extend(LocalSGLangModel._kwargs_to_cli_args(server_args))

        if self.config.debug:
            env_vars_to_print = env_vars.copy()
            if "HF_TOKEN" in env_vars_to_print:
                env_vars_to_print["HF_TOKEN"] = "****"
            print(f"""Final SGLang serve command: {cmd}
Environment variables: {env_vars_to_print}""")

        return cmd, env_vars, int(server_args["port"])

    @staticmethod
    def _normalize_server_args(server_args: Dict[str, Any]) -> Dict[str, Any]:
        aliases = {
            "tp_size": "tensor_parallel_size",
            "pp_size": "pipeline_parallel_size",
            "dp_size": "data_parallel_size",
            "ep_size": "expert_parallel_size",
        }
        normalized = dict(server_args)
        for old_key, new_key in aliases.items():
            if old_key not in normalized:
                continue
            value = normalized.pop(old_key)
            normalized.setdefault(new_key, value)
        return normalized

    def _env_prefix_vars(self) -> Dict[str, str]:
        if not self.config.sglang_env_prefix:
            return {}

        prefix = Path(self.config.sglang_env_prefix)
        path = f"{prefix / 'bin'}:{os.environ.get('PATH', '')}"
        ld_library_paths = [str(prefix / "lib")]
        ld_library_paths.extend(str(path) for path in (prefix / "lib").glob("python*/site-packages/torch/lib"))
        if os.environ.get("LD_LIBRARY_PATH"):
            ld_library_paths.append(os.environ["LD_LIBRARY_PATH"])

        return {
            "PATH": path,
            "LD_LIBRARY_PATH": ":".join(ld_library_paths),
        }

    @staticmethod
    def _default_local_cache_vars() -> Dict[str, str]:
        run_id = os.environ.get("SLURM_JOB_ID") or "local"
        user = os.environ.get("USER") or "wangling"
        return {
            "XDG_CACHE_HOME": f"/tmp/sglang-xdg-{user}-{run_id}",
            "TRITON_CACHE_DIR": f"/tmp/sglang-triton-{user}-{run_id}",
            "TORCH_EXTENSIONS_DIR": f"/tmp/sglang-torch-ext-{user}-{run_id}",
            "SGLANG_DG_CACHE_DIR": f"/tmp/sglang-deep-gemm-{user}-{run_id}",
            "DG_JIT_CACHE_DIR": f"/tmp/sglang-deep-gemm-{user}-{run_id}",
            "FLASHINFER_WORKSPACE_DIR": f"/tmp/sglang-flashinfer-{user}-{run_id}",
        }

    @staticmethod
    def _kwargs_to_cli_args(kwargs: Dict[str, Any]) -> List[str]:
        cli_args: List[str] = []
        for key, value in kwargs.items():
            if value is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)
                continue
            cli_args.append(flag)
            if isinstance(value, (dict, list)):
                cli_args.append(json.dumps(value))
            else:
                cli_args.append(str(value))
        return cli_args

    def start_sglang_server(self) -> None:
        cmd, env_vars, port = self._configure_sglang_serve()

        env = os.environ.copy()
        env.update(env_vars)

        stdout = sys.stderr
        stderr = subprocess.STDOUT
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._sglang_log_handle = log_path.open("a")
            stdout = self._sglang_log_handle
            stderr = subprocess.STDOUT

        self._sglang_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        atexit.register(self._terminate_sglang_process)

        self.config.base_url = [f"http://127.0.0.1:{port}/v1"]
        self._post_init()
        self.await_server_ready()

    def await_server_ready(self) -> None:
        poll_count = 0
        max_polls = max(1, self.config.startup_timeout_seconds // 3)
        while poll_count < max_polls:
            if self._sglang_process.poll() is not None:
                raise RuntimeError(
                    f"{self.config.name} LocalSGLangModel server spinup failed "
                    f"with exit code {self._sglang_process.returncode}."
                )

            try:
                response = requests.get(url=f"{self.config.base_url[0]}/models", timeout=5)
                if response.ok:
                    return
            except RequestException:
                pass

            if poll_count % 10 == 0:
                print(f"Waiting for {self.config.name} LocalSGLangModel server to spinup...")

            poll_count += 1
            sleep(3)

        raise TimeoutError(
            f"{self.config.name} LocalSGLangModel server did not become ready within "
            f"{self.config.startup_timeout_seconds} seconds."
        )

    def _terminate_sglang_process(self) -> None:
        process = getattr(self, "_sglang_process", None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        log_handle = getattr(self, "_sglang_log_handle", None)
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    LocalSGLangModel.run_webserver()
