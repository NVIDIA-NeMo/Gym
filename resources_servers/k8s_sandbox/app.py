# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.k8s_runner import K8sJobRunner


class K8sSandboxConfig(BaseResourcesServerConfig):
    job_namespace: str = "default"
    job_image: str = "python:3.12-slim"
    execution_timeout: int = 30


class ExecuteCodeRequest(BaseModel):
    code: str


class ExecuteCodeResponse(BaseModel):
    exit_code: int
    stdout: str
    stderr: str


class K8sSandboxRunRequest(BaseRunRequest):
    expected_output: str


class K8sSandboxVerifyRequest(K8sSandboxRunRequest, BaseVerifyRequest):
    pass


class K8sSandboxVerifyResponse(BaseVerifyResponse):
    matched: bool = False
    actual_output: Optional[str] = None


class K8sSandboxResourcesServer(SimpleResourcesServer):
    config: K8sSandboxConfig

    _runner: Optional[K8sJobRunner] = PrivateAttr(default=None)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._runner = K8sJobRunner(namespace=self.config.job_namespace)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/execute_code")(self.execute_code)
        return app

    async def execute_code(self, body: ExecuteCodeRequest) -> ExecuteCodeResponse:
        job_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        exit_code, stdout, stderr = await self._runner.run_job(
            job_name=job_name,
            image=self.config.job_image,
            command=["python", "-c", "import os; exec(os.environ['__CODE'])"],
            timeout=self.config.execution_timeout,
            env={"__CODE": body.code},
        )
        return ExecuteCodeResponse(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def verify(self, body: K8sSandboxVerifyRequest) -> K8sSandboxVerifyResponse:
        expected = body.expected_output.strip()

        actual_output: Optional[str] = None
        for output in reversed(body.response.output):
            if output.type != "function_call_output":
                continue
            try:
                tool_resp = json.loads(output.output)
                stdout_text = tool_resp.get("stdout", "").strip()
            except (json.JSONDecodeError, AttributeError):
                stdout_text = str(output.output).strip()

            if stdout_text:
                actual_output = stdout_text
                break

        matched = actual_output is not None and expected in actual_output
        reward = 1.0 if matched else 0.0

        return K8sSandboxVerifyResponse(
            **body.model_dump(),
            reward=reward,
            matched=matched,
            actual_output=actual_output,
        )


if __name__ == "__main__":
    K8sSandboxResourcesServer.run_webserver()
