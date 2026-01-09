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
import base64
import json
import time
from asyncio import Semaphore, get_running_loop
from typing import Any, Optional

import ray
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.swerl_gen.eval.process_patch import (
    extract_pred_patch,
    extract_pred_patch_relaxed_formatting,
    extract_repro_test,
)
from resources_servers.swerl_gen.eval.singularity_utils import (
    compute_score,
)


class SWEGenResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 1
    sandbox_timeout: int = 600
    debug: bool = False
    relaxed_formatting: bool = False


class SWEGenRunRequest(BaseRunRequest):
    instance: dict[
        str, Any
    ]  ## dictionary keys: instance_id, repo, setup_script, test_script, regression_script, PASS_TO_PASS, FAIL_TO_PASS, patch
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    metadata: dict[str, Any] = {}  ## keys: relevant_file_contents, remove_repo_name, image
    partial_similarity: Optional[bool] = None
    mode: str = "eval"  ## eval or repro-gen


class SWEGenVerifyRequest(SWEGenRunRequest, BaseVerifyRequest):
    pass


class SWEGenVerifyResponse(BaseVerifyResponse):
    verification_result: Optional[dict[str, Any]] = None
    verification_time: Optional[float] = None
    model_patch: Optional[str] = None
    repro_test_info_base64: Optional[str] = None
    model_output: Optional[str] = None


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract the last assistant message's text from the NeMo Gym response."""
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


class SWEGenResourcesServer(SimpleResourcesServer):
    config: SWEGenResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

    async def verify(self, body: SWEGenVerifyRequest) -> SWEGenVerifyResponse:
        # Extract full model output text (including <think> and <solution> blocks).
        predict_str = _extract_last_assistant_text(body)
        if not predict_str or not predict_str.strip():
            return SWEGenVerifyResponse(
                **body.model_dump(),
                reward=0.0,
            )

        # Extract the predicted patch or reproduction test info from the model output.
        if body.mode == "repro-gen":
            try:
                extracted_data = extract_repro_test(predict_str, body.instance["instance_id"])
            except Exception:
                extracted_data = None
            if extracted_data is None:
                return SWEGenVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    model_output=predict_str,
                )
            patch_str = body.instance["patch"]
            repro_test_info_base64 = extracted_data["repro_test_info_base64"]
        elif body.mode == "eval":
            try:
                if self.config.relaxed_formatting:
                    extracted_data = extract_pred_patch_relaxed_formatting(
                        json.loads(body.metadata["relevant_file_contents"]),
                        predict_str,
                        body.metadata["remove_repo_name"],
                    )
                else:
                    extracted_data = extract_pred_patch(
                        json.loads(body.metadata["relevant_file_contents"]),
                        predict_str,
                        body.metadata["remove_repo_name"],
                    )
            except Exception:
                extracted_data = None
            if extracted_data is None:
                return SWEGenVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    model_output=predict_str,
                )
            patch_str = extracted_data["model_patch"]
            repro_test_info_base64 = None
        else:
            raise ValueError(f"Invalid mode: {body.mode}")

        extra_info = {
            "instance_info": body.instance,
            "image": body.metadata["image"],
        }
        extra_info_base64 = base64.b64encode(json.dumps(extra_info).encode()).decode()

        async with self._semaphore:
            start_time = time.time()
            task_args = (
                extra_info_base64,
                patch_str,
                repro_test_info_base64,
                body.mode,
                self.config.sandbox_timeout,
                self.config.debug,
            )
            future = compute_score.remote(*task_args)
            reward, verification_result = await future
            verification_time = time.time() - start_time

        return SWEGenVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            verification_result=verification_result,
            verification_time=verification_time,
            model_patch=patch_str,
            repro_test_info_base64=repro_test_info_base64,
            model_output=predict_str,
        )


if __name__ == "__main__":
    SWEGenResourcesServer.run_webserver()
