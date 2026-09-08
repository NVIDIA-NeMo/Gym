# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import json
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
from asyncio import Semaphore
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import apply_rollout_prefix


def _format_container(container_formatter: str | list[str], task_name: str, docker_image: str) -> str:
    """Resolve the pullable/local image reference for a task from a formatter template."""

    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    fmt = fmt or "docker://{docker_image}"
    docker_image = docker_image[len("docker://") :] if docker_image.startswith("docker://") else docker_image
    if fmt.endswith(".sif") or fmt.startswith(("/", ".")):
        return fmt.format(task_name=task_name, docker_image=docker_image)
    if fmt.startswith("docker://"):
        fmt = fmt[len("docker://") :]
    return f"docker://{fmt.format(task_name=task_name, docker_image=docker_image)}"


def _read_task_meta(task_dir: Path) -> dict:
    """Read workdir and timeouts from task.toml + Dockerfile at runtime (fallback when not in JSONL)."""
    result = {}
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
        result["agent_timeout_sec"] = (cfg.get("agent") or {}).get("timeout_sec")
        result["verifier_timeout_sec"] = (cfg.get("verifier") or {}).get("timeout_sec")
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        for line in dockerfile.read_text().splitlines():
            if line.strip().upper().startswith("WORKDIR"):
                parts = line.strip().split(None, 1)
                if len(parts) > 1:
                    result["workdir"] = parts[1]
    return result


class KernelBenchMetrics(BaseModel):
    resolved: Optional[bool] = None
    compiled: Optional[bool] = None
    correctness: Optional[bool] = None
    runtime: Optional[float] = None
    ref_runtime: Optional[float] = None
    speedup: Optional[float] = None
    agent_timed_out: bool = False
    container_timed_out: bool = False
    sandbox_failed: bool = False
    mask_sample: bool = False

    ray_queue_time: Optional[float] = None
    agent_run_time: Optional[float] = None
    eval_run_time: Optional[float] = None
    total_run_time: Optional[float] = None


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.write_text(json.dumps(existing | update))


def _safe_config_json(params: "KernelGymInstanceConfig", indent: Optional[int] = None) -> str:
    """Serialize config without secrets."""

    def redact(value: Any, key: str = "") -> Any:
        normalized = key.lower()
        if (
            any(secret in normalized for secret in ("api_key", "apikey", "secret", "password"))
            or normalized == "token"
            or normalized.endswith("_token")
        ):
            return "***"
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                normalized_key = key.lower().replace("_", "").replace("-", "")
                is_secret = any(part in normalized_key for part in ("apikey", "password", "secret")) or (
                    normalized_key.endswith("token")
                )
                result[key] = "***" if is_secret else redact(item)
            return result
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    d = json.loads(params.model_dump_json())
    d.pop("agent_command_str", None)
    return json.dumps(redact(d), indent=indent)


class KernelGymConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str
    agent_server_class: str
    agent_config_class: str
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    container_formatter: str | list[str] = Field(
        default="docker://{docker_image}",
        description="Template for the task's image reference: use as a path if it ends with .sif or starts with / or ., else as a docker:// URI.",
    )
    sandbox_provider: Dict[str, Any] = Field(default_factory=lambda: {"docker": {}})
    sandbox_default_metadata: Dict[str, Any] = Field(default_factory=dict)
    sandbox_model_base_url: Optional[str] = None
    kb_agent_timeout: int = 1800
    global_agent_timeout: Optional[int] = Field(default=None, gt=0)
    kb_eval_timeout: int = 300
    kb_sandbox_ttl: int = 7200
    concurrency: int = 256
    results_dir: Optional[Path] = None


class KernelGymRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class KernelGymInstanceConfig(KernelGymConfig):
    model_server_url: str
    model_name: str = ""
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    verifier_dir: Path
    metrics_fpath: Path
    container: str
    ray_queue_timestamp: float
    agent_command_str: Optional[str] = None

    @property
    def task_name(self) -> str:
        return self.problem_info.get("task_name", self.problem_info.get("instance_id", "unknown"))

    @property
    def instance_id(self) -> str:
        return self.problem_info.get("instance_id", self.task_name)


class KernelGymVerifyResponse(KernelBenchMetrics, BaseVerifyResponse):
    instance_config: Dict[str, Any]


class KernelGymAgent(SimpleResponsesAPIAgent):
    """Single sandbox: agent runs, host stages tests, sandbox runs test.sh."""

    config: KernelGymConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _base_results_dir: Optional[Path] = None
    _model_server_url: str = ""
    _model_name: str = ""

    async def _run_kernel(self, cfg: KernelGymInstanceConfig) -> bool:
        cfg.verifier_dir.mkdir(parents=True, exist_ok=True)
        staging_tests = cfg.persistent_dir / "staging" / "tests"
        staging_tests.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        sandbox = AsyncSandbox(
            cfg.sandbox_provider,
            SandboxSpec(
                image=cfg.container.removeprefix("docker://"),
                ttl_s=cfg.kb_sandbox_ttl,
                workdir=cfg.problem_info.get("workdir"),
                metadata=cfg.sandbox_default_metadata,
                resources=SandboxResources(
                    cpu=float(cfg.problem_info["cpus"]) if cfg.problem_info.get("cpus") else None,
                    memory_mib=int(cfg.problem_info["memory_mb"]) if cfg.problem_info.get("memory_mb") else None,
                    disk_gib=max(1, round(int(cfg.problem_info["storage_mb"]) / 1024))
                    if cfg.problem_info.get("storage_mb")
                    else None,
                    gpu=int(cfg.problem_info.get("gpus") or 1),
                    gpu_type=cfg.problem_info.get("gpu_type"),
                ),
            ),
        )
        agent_timed_out = container_timed_out = False
        sandbox_failed = False
        agent_run_time = eval_run_time = None
        try:
            await sandbox.start()
            result = await sandbox.exec("mkdir -p /workspace /trajectories_mount /logs/verifier", user="root")
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "failed to create sandbox directories")

            task_dir = Path(cfg.problem_info["task_dir"])
            for local, remote in (
                (cfg.persistent_dir / "instruction.txt", "/trajectories_mount/instruction.txt"),
                (cfg.persistent_dir / "agent_runner.py", "/trajectories_mount/agent_runner.py"),
                (task_dir / "reference.py", "/workspace/reference.py"),
                (task_dir / "solution.py", "/workspace/solution.py"),
            ):
                await sandbox.upload(local, remote)

            env = {
                "KB_MODEL_URL": cfg.model_server_url,
                "KB_MODEL_NAME": cfg.model_name,
                "KB_AGENT_MODULE": cfg.agent_server_module,
                "KB_AGENT_CLASS": cfg.agent_server_class,
                "KB_AGENT_CONFIG_CLASS": cfg.agent_config_class,
                "KB_AGENT_KWARGS": json.dumps(cfg.agent_kwargs),
                "KB_BODY": cfg.body.model_dump_json(),
            }
            agent_started = time.time()
            result = await sandbox.exec(
                cfg.agent_command_str or "",
                timeout_s=cfg.kb_agent_timeout,
                user="root",
                env=env,
            )
            agent_run_time = time.time() - agent_started
            agent_timed_out = result.error_type in ("timeout", "sandbox")
            (cfg.persistent_dir / "agent_result.json").write_text(
                json.dumps(
                    {
                        "return_code": result.return_code,
                        "error_type": result.error_type,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
            )
            if result.return_code != 0:
                detail = result.stderr or result.stdout or ""
                print(f"[{cfg.task_name}] agent exit {result.return_code}: {detail[-2000:]}", flush=True)

            shutil.copytree(task_dir / "tests", staging_tests)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
                archive = Path(temporary.name)
            try:
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(staging_tests, arcname=".")
                await sandbox.upload(archive, "/tmp/kernel-gym-tests.tar.gz")
            finally:
                archive.unlink(missing_ok=True)
            result = await sandbox.exec(
                "mkdir -p /trajectories_mount/staging/tests && "
                "tar -xzf /tmp/kernel-gym-tests.tar.gz -C /trajectories_mount/staging/tests",
                timeout_s=300,
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "failed to stage verifier")

            eval_started = time.time()
            result = await sandbox.pty.exec(
                "rm -rf /tests && ln -s /trajectories_mount/staging/tests /tests && "
                "printf '[pytest]\\naddopts =\\n' > /pytest.ini && bash /tests/test.sh",
                timeout_s=cfg.kb_eval_timeout,
                user="root",
                pty=False,
                detach=True,
                poll_interval_s=15,
            )
            eval_run_time = time.time() - eval_started
            container_timed_out = result.error_type == "timeout"
            if result.return_code != 0:
                print(f"[{cfg.task_name}] eval exit {result.return_code}", flush=True)

            for remote, local in (
                ("/workspace/solution.py", cfg.persistent_dir / "solution.py"),
                ("/logs/verifier/reward.txt", cfg.verifier_dir / "reward.txt"),
                ("/logs/verifier/result.json", cfg.verifier_dir / "result.json"),
                ("/logs/verifier/test-stdout.txt", cfg.verifier_dir / "test-stdout.txt"),
            ):
                await sandbox.download(remote, local)
            try:
                await sandbox.download("/trajectories_mount/response.json", cfg.persistent_dir / "response.json")
            except Exception:
                pass
        except Exception as e:
            sandbox_failed = True
            print(f"[{cfg.task_name}] sandbox run failed: {e}", flush=True)
        finally:
            try:
                await sandbox.stop()
            except Exception as e:
                sandbox_failed = True
                print(f"[{cfg.task_name}] sandbox cleanup failed: {e}", flush=True)
            shutil.rmtree(cfg.persistent_dir / "staging", ignore_errors=True)

        total_run_time = time.time() - t0

        reward_path = cfg.verifier_dir / "reward.txt"
        resolved = False
        if reward_path.exists():
            try:
                resolved = float(reward_path.read_text().strip()) > 0
            except (ValueError, OSError):
                pass

        verifier = {}
        result_path = cfg.verifier_dir / "result.json"
        if result_path.exists():
            try:
                verifier = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        metrics = KernelBenchMetrics(
            ray_queue_time=time.time() - cfg.ray_queue_timestamp,
            resolved=resolved,
            compiled=verifier.get("compiled"),
            correctness=verifier.get("correctness"),
            runtime=verifier.get("runtime"),
            ref_runtime=verifier.get("ref_runtime"),
            speedup=verifier.get("speedup"),
            agent_timed_out=agent_timed_out,
            container_timed_out=container_timed_out,
            sandbox_failed=sandbox_failed,
            mask_sample=bool(container_timed_out or agent_timed_out or sandbox_failed),
            agent_run_time=agent_run_time,
            eval_run_time=eval_run_time,
            total_run_time=total_run_time,
        )
        update_metrics(cfg.metrics_fpath, metrics.model_dump())
        return resolved

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)
        self.config.sandbox_default_metadata = resolve_provider_metadata(
            self.config.sandbox_provider, self.server_client.global_config_dict
        )
        self.config.sandbox_provider = resolve_provider_config(
            self.config.sandbox_provider, self.server_client.global_config_dict
        )

        model_url = self.config.sandbox_model_base_url or ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            if not model_url:
                model_url = self.server_client._build_server_base_url(model_cfg)

        model_name = str(self.server_client.global_config_dict.get("policy_model_name") or "")

        workspace = Path(__file__).parent
        results_dir = workspace / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        base_results_dir = self.config.results_dir
        if base_results_dir is None:
            session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            base_results_dir = results_dir / f"kernel_gym_results_{session_id}"
        else:
            session_id = base_results_dir.name
        base_results_dir.mkdir(parents=True, exist_ok=True)

        self._base_results_dir = base_results_dir
        self._model_server_url = model_url
        self._model_name = model_name
        super().model_post_init(context)

    def _setup_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> KernelGymInstanceConfig:
        problem_info = dict(body.metadata or {})
        task_name = problem_info.get("task_name", problem_info.get("instance_id", "unknown"))

        task_dir = Path(problem_info["task_dir"])
        if not all(k in problem_info for k in ("workdir", "agent_timeout_sec", "verifier_timeout_sec")):
            problem_info.update({k: v for k, v in _read_task_meta(task_dir).items() if k not in problem_info})

        instance_dir = f"{task_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        persistent_dir = self._base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)
        verifier_dir = persistent_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)

        config_overrides = {}
        if self.config.global_agent_timeout is not None:
            config_overrides["kb_agent_timeout"] = self.config.global_agent_timeout
        elif problem_info.get("agent_timeout_sec"):
            config_overrides["kb_agent_timeout"] = int(float(problem_info["agent_timeout_sec"]))
        if problem_info.get("verifier_timeout_sec"):
            config_overrides["kb_eval_timeout"] = int(float(problem_info["verifier_timeout_sec"]))

        # Leave time for sandbox startup and output collection around both phases.
        effective_agent_timeout = config_overrides.get("kb_agent_timeout", self.config.kb_agent_timeout)
        effective_eval_timeout = config_overrides.get("kb_eval_timeout", self.config.kb_eval_timeout)
        required_ttl = effective_agent_timeout + effective_eval_timeout + 600
        if required_ttl > self.config.kb_sandbox_ttl:
            config_overrides["kb_sandbox_ttl"] = required_ttl

        model_server_url = self._model_server_url
        if not self.config.sandbox_model_base_url and rollout_id and model_server_url:
            model_server_url = apply_rollout_prefix(model_server_url, rollout_id)

        params = KernelGymInstanceConfig(
            **{**self.config.model_dump(), **config_overrides},
            model_server_url=model_server_url,
            model_name=self._model_name,
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            verifier_dir=verifier_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            container=_format_container(
                self.config.container_formatter, task_name, problem_info.get("docker_image", "ubuntu:22.04")
            ),
            ray_queue_timestamp=time.time(),
        )
        params.metrics_fpath.write_text("{}")

        (persistent_dir / "instruction.txt").write_text(problem_info["instruction"])
        shutil.copy2(Path(__file__).parent / "agent_runner.py", persistent_dir / "agent_runner.py")
        params.agent_command_str = "/opt/agent/bin/python /trajectories_mount/agent_runner.py"

        return params

    async def _run_response(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> NeMoGymResponse:
        params = self._setup_params(body, rollout_id)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            await self._run_kernel(params)
        except Exception:
            kb_path = params.persistent_dir / "traceback.err"
            kb_path.write_text(format_exc())
            print(f"[{params.task_name}] exception: see {kb_path}", file=sys.stderr)
            raise

        persisted = KernelBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
        mask_sample = bool(
            persisted.mask_sample
            or persisted.container_timed_out
            or persisted.agent_timed_out
            or persisted.sandbox_failed
        )
        update_metrics(params.metrics_fpath, {"mask_sample": mask_sample})
        if mask_sample:
            raise RuntimeError(f"{params.task_name} did not produce a scoreable rollout")

        response_path = params.persistent_dir / "response.json"
        if not response_path.exists():
            raise RuntimeError(f"{params.task_name} did not produce response.json")
        saved = NeMoGymResponse.model_validate_json(response_path.read_text())

        return NeMoGymResponse(
            id=f"kernel-gym-{params.instance_id}",
            created_at=int(time.time()),
            model=params.model_name,
            object="response",
            output=saved.output,
            status=saved.status,
            error=saved.error,
            incomplete_details=saved.incomplete_details,
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=saved.tools or [],
            usage=saved.usage,
            metadata={
                "input": json.dumps(params.body.model_dump(mode="json").get("input") or []),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": _safe_config_json(params),
            },
        )

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        return await self._run_response(body)

    async def run(self, body: KernelGymRunRequest) -> KernelGymVerifyResponse:
        async with self._sem:
            response = await self._run_response(body.responses_create_params, self.rollout_id_from_run(body))

            meta, response.metadata = response.metadata, None
            metrics = KernelBenchMetrics.model_validate_json(meta["metrics"])

            return KernelGymVerifyResponse(
                responses_create_params=body.responses_create_params.model_dump()
                | {
                    "input": json.loads(meta["input"]),
                    "tools": [t.model_dump() for t in (response.tools or [])],
                    "model": response.model,
                },
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                **metrics.model_dump(),
                instance_config=KernelGymInstanceConfig.model_validate_json(meta["instance_config"]).model_dump(),
            )


if __name__ == "__main__":
    KernelGymAgent.run_webserver()
