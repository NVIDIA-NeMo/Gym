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

from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

from fastapi import Request
from pydantic import BaseModel
from swebench.harness.constants import END_TEST_OUTPUT, MAP_REPO_TO_EXT, START_TEST_OUTPUT
from swebench.harness.run_evaluation import make_test_spec
from swebench.harness.test_spec.test_spec import LATEST, TestSpec

from docker.models.containers import ExecResult
from nemo_gym import ROOT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.swebench.swebench_patches import run_instance


class SwebenchResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False

    evaluation_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]


class SWEBenchInstanceRequest(BaseModel):
    # See https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
    # See swebench.harness.run_evaluation.TestSpec https://github.com/SWE-bench/SWE-bench/blob/f7bbbb2ccdf479001d6467c9e34af59e44a840f9/swebench/harness/test_spec/test_spec.py#L28
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    # These are JSON strings.
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str
    difficulty: str
    subset: str
    split: str


class SWEBenchVerifyRequest(SWEBenchInstanceRequest, BaseVerifyRequest):
    pass


class SWEBenchVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool

    # Misc metrics
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float

    model_patch: Optional[str]

    log_dir: str


# @bxyu-nvidia: This is a wrapper that can be passed directly to a very lightly modified version of `run_instance`
# The method is almost identical to the original, just with async awaits rather than sync.
# See resources_servers/swebench/swebench_patches.py
class DockerContainer(BaseModel):
    id: str
    instance_id: str

    _inner_container: AsyncSandbox

    async def exec_run(
        self,
        command: str,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
    ) -> ExecResult:
        res = await self._inner_container.exec(
            command=command,
            cwd=workdir,
            user=user,
        )

        return ExecResult(
            exit_code=res.return_code,
            # @bxyu-nvidia: This is not entirely 1:1, but it works for the purposes of this patch.
            # The sandbox API returns None for an empty stream (docker-py returned bytes).
            output=((res.stdout or "") + (res.stderr or "")).encode(),
        )

    async def exec_run_with_timeout(self, command: str, timeout: int) -> Tuple[str, bool, float]:
        # Returns: test_output: str, timed_out: bool, total_runtime: float
        start_time = time()
        try:
            res = await self._inner_container.exec(
                command=command,
                # AsyncSandbox.exec takes timeout_s, not docker-py's timeout.
                timeout_s=timeout,
            )
            timed_out = False

            stdout = res.stdout or ""
            stderr = res.stderr or ""

            # For RuboCop tests in SWE Multilingual specifically, there is an issue with the logs parsing if the stdout and stderr returned is not interleaved.
            # We interleave the stderr inside the start and end tags in the stdout here instead. See `get_logs_eval`
            if "rubocop" in self.instance_id and START_TEST_OUTPUT in stderr:
                start, middle_end = stderr.split(START_TEST_OUTPUT)
                middle, end = middle_end.split(END_TEST_OUTPUT)
                test_output = start + START_TEST_OUTPUT + stdout + middle + END_TEST_OUTPUT + end
            else:
                test_output = stdout + stderr
        except TimeoutError:
            # Gym Sandbox API will throw a timeout error on actual timeout.
            timed_out = True
            test_output = ""

        return (test_output, timed_out, time() - start_time)

    async def copy(self, src: Path, dest: Path) -> None:
        await self._inner_container.upload(local_path=src, remote_path=str(dest))

    async def cleanup(self) -> None:
        await self._inner_container.stop()


# TODO @bxyu-nvidia: Eventually once the sandbox server infra is ready, these seed_session types need to upgrade to pass a sandbox spec.
# They can possibly even omitted once this graduates to core infra.
class SWEBenchSeedSessionRequest(SWEBenchInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: Optional[Dict[str, Any]] = None


class SWEBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.


class SwebenchResourcesServer(SimpleResourcesServer):
    config: SwebenchResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = dict()

    async def _create_sandbox(self, test_spec: TestSpec) -> AsyncSandbox:
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        eval_sandbox_spec = SandboxSpec(
            image=test_spec.instance_image_key,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=dict(),
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": test_spec.instance_id[:63],
            },
            resources=SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {})),
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        eval_sandbox = AsyncSandbox(resolved_sandbox_provider)
        await eval_sandbox.start(eval_sandbox_spec)

        if MAP_REPO_TO_EXT.get(test_spec.repo) == "java":
            await self._apply_sandbox_patches(eval_sandbox)

        return eval_sandbox

    async def _apply_sandbox_patches(self, sandbox: AsyncSandbox) -> None:
        base_path = ROOT_DIR / "responses_api_agents/swe_agents/maven_mirror"
        settings_xml_path = base_path / "settings.xml"
        init_gradle_path = base_path / "init.gradle"

        await sandbox.exec("""mkdir -p \
        /root/.m2 \
        /root/.gradle/init.d \
        /home/gradle/.gradle/init.d \
        /home/user/.gradle/init.d""")
        await sandbox.upload(settings_xml_path, "/root/.m2/settings.xml")
        await sandbox.upload(init_gradle_path, "/root/.gradle/init.d/maven_central_mirror.gradle")
        await sandbox.upload(init_gradle_path, "/home/gradle/.gradle/init.d/maven_central_mirror.gradle")
        await sandbox.upload(init_gradle_path, "/home/user/.gradle/init.d/maven_central_mirror.gradle")

    def _make_test_spec(self, body: SWEBenchVerifyRequest) -> TestSpec:
        return make_test_spec(
            # This accepts a SWEbenchInstance which is identically our body.
            body.model_dump(),
            namespace="swebench",  # Dockerhub namespace
            instance_image_tag=LATEST,
            env_image_tag=LATEST,
        )

    async def seed_session(self, request: Request, body: SWEBenchSeedSessionRequest) -> SWEBenchSeedSessionResponse:
        test_spec = self._make_test_spec(body)
        eval_sandbox = await self._create_sandbox(test_spec)
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = eval_sandbox

        # @bxyu-nvidia: Activate the necessary conda environments for SWE Bench Verified Python instances
        # This may be overfit and needs to be config'd or detected.
        # TODO @bxyu-nvidia: This pattern is not yet supported because calls to sandbox.exec use separate processes
        # For now, the activation is put on the harness side.
        # await eval_sandbox.exec("source /opt/miniconda3/bin/activate && conda activate testbed")

        return SWEBenchSeedSessionResponse(sandbox_handle=eval_sandbox._handle.sandbox_id)

    async def verify(self, request: Request, body: SWEBenchVerifyRequest) -> SWEBenchVerifyResponse:
        """
        Key requirements:
        1. Extract the model_patch from the input container
            Proposal
                1. Spinup a fresh container (need this anyways for running eval)
                2. pwd in the fresh container (defaults to WORKDIR)
                3. cd into WORKDIR in the input container
                4. extract the patch via git
            Notes
                1. DeepSWE expects the model to commit. That will go in the DeepSWE resources server and not this one.
        2. Docker Client - Make a mock client class here that wraps our sandbox client.
        3. Harnesses like OpenCode open a new terminal rather than reusing the existing one. Grab the environment variables and workdir from the outer terminal first, and then export/cd as appropriate in the new terminal
            Notes
                1. This is a harness-specific thing that the harness will handle across benchmarks.
        4. Interleaved thinking - Verify how is the harness behaving i.e. it has interleaved thinking or not and to force interleaved thinking unconditionally.
            Proposal
                1. For seeing what the harness is doing, use model call capture
                2. For forcing, we can add it in the Responses API model proxy i.e. save all the past requests/responses and populate as necessary.
        5. Restrict number of turns - same as interleaved thinking, we could add in Responses API model proxy
        """

        test_spec = self._make_test_spec(body)

        start_time = time()
        eval_sandbox = await self._create_sandbox(test_spec)
        eval_sandbox_start_time_taken = time() - start_time

        model_patch = ""
        if self.config.is_verifying_golden_patch:
            model_patch = body.patch
        else:
            original_sandbox = self._session_id_to_sandbox[request.session[SESSION_ID_KEY]]
            original_workdir = (await eval_sandbox.exec("pwd")).stdout.strip()
            model_patch_result = await original_sandbox.exec(f"cd {original_workdir} && git --no-pager diff")
            await original_sandbox.stop()
            model_patch = model_patch_result.stdout

        run_id = request.session[SESSION_ID_KEY]
        mock_container = DockerContainer(id=run_id, instance_id=test_spec.instance_id)
        mock_container._inner_container = eval_sandbox

        # Res has 2 keys: completed (whether evaluation completed or not), resolved (whether the issue is resolved)
        start_time = time()
        res = await run_instance(
            test_spec=test_spec,
            pred={
                "instance_id": test_spec.instance_id,
                "model_patch": model_patch,
            },
            rm_image=False,
            force_rebuild=False,
            client=mock_container,
            run_id=run_id,
            timeout=self.config.evaluation_timeout,
            rewrite_reports=False,
        )
        patch_verification_time_taken = time() - start_time
        return SWEBenchVerifyResponse(
            **body.model_dump(),
            # run_instance returns "completed"; the response field is "evaluation_completed".
            evaluation_completed=res["completed"],
            resolved=res["resolved"],
            reward=int(res["resolved"]),
            eval_sandbox_start_time_taken=eval_sandbox_start_time_taken,
            patch_verification_time_taken=patch_verification_time_taken,
            model_patch=model_patch or None,
            log_dir=str(Path(__file__).parent / "logs/run_evaluation" / run_id),
        )


if __name__ == "__main__":
    SwebenchResourcesServer.run_webserver()
