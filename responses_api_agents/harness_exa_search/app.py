# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, Request
from pydantic import ConfigDict, Field, SecretStr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_response_json, raise_for_status


class HarnessExaSearchConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    harness_module: str
    harness_class: str
    harness_config_class: str
    harness_kwargs: dict[str, Any] = Field(default_factory=dict)
    image: str
    python: str = "python3"
    runtime_archive: Path | None = None
    setup_command: str | None = None
    sandbox_provider: str | dict[str, Any] = "sandbox"
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    sandbox_model_base_url: str | None = None
    exa_api_key: SecretStr | None = None


class HarnessExaSearchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HarnessExaSearchVerifyRequest(HarnessExaSearchRunRequest, BaseVerifyRequest):
    pass


class HarnessExaSearchResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class HarnessExaSearchAgent(SimpleResponsesAPIAgent):
    config: HarnessExaSearchConfig

    def model_post_init(self, context: Any) -> None:
        model = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        self._model_url = self.server_client._build_server_base_url(model)
        self._provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        self._metadata = resolve_provider_metadata(self.config.sandbox_provider, self.server_client.global_config_dict)
        super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        root = f"/tmp/nemo-gym-harness-exa-search-{uuid.uuid4().hex}"
        input_path, output_path = f"{root}/input.json", f"{root}/response.json"
        runner_path, config_path = f"{root}/agent_runner.py", f"{root}/runner.json"
        pi_extension_path = f"{root}/exa_pi_extension.ts"
        values = dict(self.config.sandbox_spec)
        spec = SandboxSpec(
            image=self.config.image.removeprefix("docker://"),
            ttl_s=values.pop("ttl_s", None),
            ready_timeout_s=values.pop("ready_timeout_s", 1200),
            workdir=values.pop("workdir", root),
            env=values.pop("env", {}),
            metadata={**self._metadata, **values.pop("metadata", {}), "nemo_gym_agent": "harness_exa_search"},
            resources=SandboxResources.from_mapping(values.pop("resources", {})),
            entrypoint=values.pop("entrypoint", None),
            provider_options=values.pop("provider_options", {}),
        )
        if values:
            raise ValueError(f"unknown sandbox_spec keys: {sorted(values)}")
        runner_config = {
            "harness_module": self.config.harness_module,
            "harness_class": self.config.harness_class,
            "harness_config_class": self.config.harness_config_class,
            "harness_kwargs": self.config.harness_kwargs,
            "model_url": (self.config.sandbox_model_base_url or self._model_url).rstrip("/")
            + self.url_path_for_request("", request).rstrip("/"),
            "input_path": input_path,
            "output_path": output_path,
            "exa_api_key": self.config.exa_api_key.get_secret_value() if self.config.exa_api_key else None,
            "pi_extension_path": pi_extension_path,
        }
        sandbox = AsyncSandbox(self._provider, spec)
        try:
            await sandbox.start()
            with tempfile.TemporaryDirectory() as temporary:
                local = Path(temporary)
                (local / "input.json").write_text(body.model_dump_json(exclude_none=True))
                (local / "runner.json").write_text(json.dumps(runner_config))
                await sandbox.upload(Path(__file__).with_name("agent_runner.py"), runner_path)
                await sandbox.upload(Path(__file__).with_name("exa_pi_extension.ts"), pi_extension_path)
                await sandbox.upload(local / "input.json", input_path)
                await sandbox.upload(local / "runner.json", config_path)
                python = self.config.python
                if self.config.runtime_archive:
                    archive_path = f"{root}/runtime.tar.gz"
                    runtime_path = f"{root}/runtime"
                    await sandbox.upload(self.config.runtime_archive, archive_path)
                    unpack = await sandbox.exec(
                        f"mkdir -p {runtime_path} && tar -xzf {archive_path} -C {runtime_path} --strip-components=1",
                        timeout_s=None,
                    )
                    if unpack.return_code != 0:
                        raise RuntimeError(f"failed to unpack harness runtime: {(unpack.stderr or '')[-2000:]}")
                    python = f"{runtime_path}/bin/python"
                command = (
                    f"PATH={shlex.quote(str(Path(python).parent))}:$PATH "
                    f"{shlex.quote(python)} {runner_path} {config_path}"
                )
                if self.config.setup_command:
                    command = f"{self.config.setup_command} && {command}"
                result = await sandbox.exec(command, timeout_s=None)
                if result.return_code != 0:
                    diagnostics = "\n".join(part for part in (result.stdout, result.stderr) if part)
                    raise RuntimeError(f"sandboxed harness failed: {diagnostics[-4000:]}")
                await sandbox.download(output_path, local / "response.json")
                response = NeMoGymResponse.model_validate_json((local / "response.json").read_text())
                if response.usage and response.usage.total_tokens == 0 and result.stderr:
                    self.logger.warning("sandboxed harness diagnostics: %s", result.stderr[-2000:])
                return response
        finally:
            await sandbox.stop()

    async def run(self, body: HarnessExaSearchRunRequest = Body()) -> HarnessExaSearchResponse:
        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
        )
        await raise_for_status(response)
        verify_request = HarnessExaSearchVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )
        verified = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )
        await raise_for_status(verified)
        return HarnessExaSearchResponse.model_validate(await get_response_json(verified))


if __name__ == "__main__":
    HarnessExaSearchAgent.run_webserver()
