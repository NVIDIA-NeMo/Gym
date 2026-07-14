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
import asyncio
import base64
import copy
import hashlib
import json
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from asyncio import Semaphore
from pathlib import Path
from subprocess import Popen
from traceback import format_exc
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxCreateError, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata


class SWEBenchMetrics(BaseModel):
    resolved: Optional[bool] = None
    patch_exists: Optional[bool] = None
    model_patch: Optional[str] = None

    agent_timed_out: Optional[bool] = None
    agent_error_kind: Optional[str] = None
    eval_timed_out: Optional[bool] = None
    error_kind: Optional[str] = None
    mask_sample: Optional[bool] = None

    agent_run_time: Optional[float] = None


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.write_text(json.dumps(existing | update))


def _safe_config_json(params: "AnySweInstanceConfig", indent: Optional[int] = None) -> str:
    def redact(value: Any, key: str = "") -> Any:
        normalized = key.lower()
        if (
            any(s in normalized for s in ("api_key", "apikey", "secret", "password"))
            or normalized == "token"
            or normalized.endswith("_token")
        ):
            return "***"
        if isinstance(value, dict):
            return {k: redact(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    d = redact(json.loads(params.model_dump_json()))
    return json.dumps(d, indent=indent)


def _classify_agent_error(error: str) -> Optional[str]:
    text = error.lower()
    if "maximum iteration" in text or "max iterations" in text or "max_turns" in text:
        return "max_iteration"
    if "contextwindow" in text or "context window" in text:
        return "context_window"
    if "stuck in a loop" in text:
        return "stuck_in_loop"
    return "other" if text.strip() else None


def _should_mask_sample(
    resolved: bool,
    agent_error_kind: Optional[str],
    agent_timed_out: bool,
    error_kind: Optional[str],
) -> bool:
    return bool(
        (resolved and agent_error_kind in ("max_iteration", "context_window"))
        or agent_timed_out
        or error_kind in ("eval_timeout", "sandbox")
    )


def _dataset_family(dataset_name: str) -> str:
    if "R2E-Gym" in dataset_name:
        return "r2e"
    if "SWE-bench_Multilingual" in dataset_name:
        return "swebench_multilingual"
    return "swebench"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except json.JSONDecodeError:
            return [value] if value else []
    return list(value or [])


def _r2e_resolved(instance: Dict[str, Any], log: str) -> bool:
    statuses: Dict[str, str] = {}
    for line in log.splitlines():
        fields = line.strip().replace(" - ", " ").split()
        if len(fields) > 1 and fields[0] in ("PASSED", "FAILED", "ERROR", "XFAIL", "SKIPPED"):
            statuses[fields[1]] = fields[0]
    required = _as_list(instance.get("FAIL_TO_PASS") or instance.get("fail_to_pass"))
    required += _as_list(instance.get("PASS_TO_PASS") or instance.get("pass_to_pass"))
    return bool(required) and all(statuses.get(test) == "PASSED" for test in required)


_RUNNER_TEMPLATE = """\
#!/usr/bin/env python3
import asyncio, base64, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/nemo_gym_mount")
os.environ["PATH"] = "/agent_deps_mount/bin:" + os.environ.get("PATH", "")

def _json_env(name):
    encoded = os.environ.get(name + "_B64")
    if encoded:
        return json.loads(base64.b64decode(encoded).decode())
    return json.loads(os.environ.get(name, "{{}}"))

MODEL_URL   = os.environ.get("NGSWE_MODEL_URL", "")
MODEL_NAME  = os.environ["NGSWE_MODEL_NAME"]
INSTRUCTION = Path("/trajectories_mount/instruction.txt").read_text()
AGENT_KWARGS = _json_env("NGSWE_AGENT_KWARGS")
SAMPLING = _json_env("NGSWE_SAMPLING")

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymEasyInputMessage
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from {agent_module} import {agent_class}, {agent_cfg_class}


_mock_client = ServerClient.model_construct(global_config_dict={{}})
_mock_client._build_server_base_url = lambda cfg: MODEL_URL


_cfg_sampling = {{k: v for k, v in SAMPLING.items() if k in {agent_cfg_class}.model_fields}}

_model_server = ModelServerRef(name=MODEL_NAME, type="responses_api_models") if MODEL_URL else None
config = {agent_cfg_class}(
    host="0.0.0.0",
    port=0,
    name="{agent_class_lower}",
    entrypoint="app.py",
    model_server=_model_server,
    resources_server=ResourcesServerRef(name="anyswe", type="resources_servers"),
    **{{**AGENT_KWARGS, **_cfg_sampling}},
)
agent = {agent_class}(config=config, server_client=_mock_client)

if MODEL_URL:
    if hasattr(agent, "_resolve_model_base_url"):
        _v1 = MODEL_URL if MODEL_URL.endswith("/v1") else MODEL_URL + "/v1"
        agent._resolve_model_base_url = lambda: _v1
    if hasattr(agent, "_resolve_base_url"):
        agent._resolve_base_url = lambda: MODEL_URL

body = NeMoGymResponseCreateParamsNonStreaming(
    input=[NeMoGymEasyInputMessage(role="user", content=INSTRUCTION)],
    model=MODEL_NAME,
    **SAMPLING,
)
response = asyncio.run(agent.responses(request=None, body=body))
Path("/trajectories_mount/response.json").write_text(response.model_dump_json())
print(f"agent finished: {{len(response.output)}} output items", flush=True)

patch = ""
for candidate in ["/testbed", "/workspace/repo", "/app", "/root/repo"]:
    p = Path(candidate)
    if p.exists() and (p / ".git").exists():
        subprocess.run(["git", "add", "-A"], check=True, cwd=str(p))
        patch = subprocess.run(
            ["git", "diff", "--no-color", "--cached", "HEAD"],
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
            cwd=str(p),
        ).stdout
        print(f"patch: {{len(patch)}} chars from {{p}}", flush=True)
        break
Path("/trajectories_mount/patch.diff").write_text(patch)
"""


class AnySweAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str = Field(
        description="Import path to the agent module, e.g. responses_api_agents.hermes_agent.app"
    )
    agent_server_class: str = Field(description="Agent class name, e.g. HermesAgent")
    agent_config_class: str = Field(description="Agent config class name, e.g. HermesAgentConfig")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    container_formatter: str = Field(description="Baked task image containing the AnySWE agent runtime")
    sandbox_provider: str | Dict[str, Any] = "sandbox"
    sandbox_spec: Dict[str, Any] = Field(default_factory=dict)
    sandbox_model_base_url: Optional[str] = None
    upload_agent_runtime: bool = False
    swebench_tests_timeout: int = 1800
    swebench_agent_timeout: int = 2700
    concurrency: int = 256


class AnySweServerConfig(BaseModel):
    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    agent_deps_archive: Optional[Path]
    resolved_sandbox_provider: Dict[str, Any]
    sandbox_default_metadata: Dict[str, Any]


class AnySweInstanceConfig(AnySweAgentConfig, AnySweServerConfig):
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    metrics_fpath: Path
    container: str
    mask_sample: bool = False

    @property
    def instance_id(self) -> str:
        return self.problem_info["instance_id"]


class AnySweVerifyResponse(SWEBenchMetrics, BaseVerifyResponse):
    instance_config: Dict[str, Any]


class GymAgentHarnessProcessor(BaseModel):
    config: Any

    @property
    def _agent_key(self) -> str:
        return self.config.agent_server_module.split(".")[-2]

    def setup(self) -> Path:
        deps_dir = Path(__file__).parent / f"anyswe_{self._agent_key}_deps"
        sentinel = deps_dir / ".installed"
        scripts = Path(__file__).parent / "setup_scripts"
        script = scripts / f"{self._agent_key}_deps.sh"
        sources = (
            script,
            scripts / "_portable_python.sh",
            PARENT_DIR / "responses_api_agents" / self._agent_key / "requirements.txt",
        )
        recipe = hashlib.sha256(b"".join(path.read_bytes() for path in sources if path.exists())).hexdigest()
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            return deps_dir

        lock = deps_dir.parent / f".{deps_dir.name}.lockdir"
        while True:
            try:
                lock.mkdir(exist_ok=False)
                break
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > 3600:
                    shutil.rmtree(lock, ignore_errors=True)
                else:
                    time.sleep(5)

        try:
            if sentinel.exists() and sentinel.read_text().strip() == recipe:
                return deps_dir
            if not script.exists():
                raise ValueError(f"missing agent runtime setup script: {script}")
            deps_dir.mkdir(parents=True, exist_ok=True)
            proc = Popen(f"DEPS_DIR={deps_dir} NEMO_GYM_ROOT={PARENT_DIR} bash {script}", shell=True)
            if proc.wait() != 0:
                raise RuntimeError(f"agent runtime setup failed: {script}")
            sentinel.write_text(recipe)
            return deps_dir
        finally:
            shutil.rmtree(lock, ignore_errors=True)

    def write_runner(self) -> None:
        cfg: AnySweInstanceConfig = self.config

        (cfg.persistent_dir / "instruction.txt").write_text(cfg.problem_info.get("problem_statement", ""))

        runner = _RUNNER_TEMPLATE.format(
            agent_module=cfg.agent_server_module,
            agent_class=cfg.agent_server_class,
            agent_cfg_class=cfg.agent_config_class,
            agent_class_lower=cfg.agent_server_class.lower(),
        )
        (cfg.persistent_dir / "agent_runner.py").write_text(runner)


class AnySweAgent(SimpleResponsesAPIAgent):
    """Runs a Gym agent in each task container."""

    config: AnySweAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[AnySweServerConfig] = None

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)

        model_url = self.config.sandbox_model_base_url or ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            if not model_url:
                model_url = self.server_client._build_server_base_url(model_cfg)

        workspace = Path(__file__).parent
        agent_deps_archive = None
        if self.config.upload_agent_runtime:
            agent_deps_dir = GymAgentHarnessProcessor(config=self.config).setup()
            agent_deps_archive = workspace / f".{agent_deps_dir.name}.tar.gz"
            sentinel = agent_deps_dir / ".installed"
            if not agent_deps_archive.exists() or agent_deps_archive.stat().st_mtime < sentinel.stat().st_mtime:
                temporary = agent_deps_archive.with_suffix(".tmp")
                with tarfile.open(temporary, "w:gz") as archive:
                    archive.add(agent_deps_dir, arcname=".")
                temporary.replace(agent_deps_archive)
        session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        self._server = AnySweServerConfig(
            run_session_id=session_id,
            base_results_dir=workspace / f"anyswe_results_{session_id}",
            model_server_url=model_url,
            agent_deps_archive=agent_deps_archive,
            resolved_sandbox_provider=resolve_provider_config(
                self.config.sandbox_provider, self.server_client.global_config_dict
            ),
            sandbox_default_metadata=resolve_provider_metadata(
                self.config.sandbox_provider, self.server_client.global_config_dict
            ),
        )
        super().model_post_init(context)

    @staticmethod
    def _sandbox_image(problem_info: Dict[str, Any]) -> str:
        instance = problem_info.get("instance_dict", {})
        instance = json.loads(instance) if isinstance(instance, str) else instance
        explicit_image = problem_info.get("image") or instance.get("image") or instance.get("docker_image")
        if explicit_image:
            return str(explicit_image).removeprefix("docker://")
        formatter = problem_info["container_formatter"]
        if formatter.endswith(".sif"):
            raise ValueError("sandbox_provider requires a container image, not a .sif file")
        formatter = formatter.removeprefix("docker://")
        instance_id = problem_info["instance_id"].replace("__", "_1776_").lower()
        image = formatter.format(instance_id=instance_id)
        if ":" not in image.rsplit("/", 1)[-1]:
            image += ":latest"
        return image

    @staticmethod
    def _sandbox_spec(
        params: AnySweInstanceConfig,
        *,
        files: Optional[Dict[str, str]] = None,
    ) -> SandboxSpec:
        config = dict(params.sandbox_spec)
        provider_options = dict(config.pop("provider_options", {}))
        metadata = {
            **params.sandbox_default_metadata,
            **config.pop("metadata", {}),
            "nemo_gym_agent": "anyswe_agent",
            "instance_id": params.instance_id[:63],
        }
        return SandboxSpec(
            image=params.container,
            ttl_s=config.pop("ttl_s", params.swebench_agent_timeout + params.swebench_tests_timeout + 600),
            ready_timeout_s=config.pop("ready_timeout_s", 1200),
            workdir=config.pop("workdir", "/testbed"),
            env=config.pop("env", {}),
            files=files or {},
            metadata=metadata,
            resources=SandboxResources.from_mapping(config.pop("resources", {})),
            entrypoint=config.pop("entrypoint", None),
            provider_options=provider_options,
        )

    @staticmethod
    def _sandbox_agent_env(params: AnySweInstanceConfig) -> Dict[str, str]:
        sampling = {
            key: getattr(params.body, key)
            for key in ("temperature", "top_p", "max_output_tokens")
            if getattr(params.body, key, None) is not None
        }
        model_name = params.body.model or "model"
        agent_kwargs = copy.deepcopy(params.agent_kwargs)
        if params.model_server_url and params.agent_server_module.split(".")[-2] == "opencode_agent":
            model_url = params.model_server_url.rstrip("/")
            model_url = model_url if model_url.endswith("/v1") else f"{model_url}/v1"
            agent_kwargs["model"] = f"nemo/{model_name}"
            agent_kwargs["openai_base_url"] = model_url
            agent_kwargs["openai_api_key"] = "EMPTY"
            providers = agent_kwargs.setdefault("opencode_config", {}).setdefault("provider", {})
            nemo = providers.setdefault("nemo", {"npm": "@ai-sdk/openai-compatible"})
            nemo.setdefault("options", {}).update({"baseURL": model_url, "apiKey": "EMPTY"})
            model_config = nemo.setdefault("models", {}).get(model_name, {})
            nemo["models"] = {model_name: model_config}
        return {
            "NGSWE_MODEL_NAME": model_name,
            "NGSWE_MODEL_URL": params.model_server_url,
            "NGSWE_AGENT_KWARGS_B64": base64.b64encode(json.dumps(agent_kwargs).encode()).decode(),
            "NGSWE_SAMPLING_B64": base64.b64encode(json.dumps(sampling).encode()).decode(),
        }

    @staticmethod
    def _instance_dict(params: AnySweInstanceConfig) -> Dict[str, Any]:
        value = params.problem_info["instance_dict"]
        return json.loads(value) if isinstance(value, str) else dict(value)

    @staticmethod
    def _apply_patch_script() -> str:
        return (
            "cd /testbed && "
            "if git apply --verbose /root/patch.diff || "
            "git apply --verbose --reject /root/patch.diff || "
            "patch --batch --fuzz=5 -p1 -i /root/patch.diff; then "
            "echo '>>>>> Applied Patch'; else "
            "echo '>>>>> Patch Apply Failed'; exit 1; fi\n"
        )

    async def _grade_sandbox_patch(self, params: AnySweInstanceConfig, patch: str) -> tuple[bool, Optional[str]]:
        if _dataset_family(params.problem_info.get("dataset_name", "")) == "r2e":
            return await self._grade_r2e_patch(params, patch)

        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec

        instance = self._instance_dict(params)
        test_spec = make_test_spec(instance, namespace="swebench")
        eval_script = (
            'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"\n' + self._apply_patch_script() + test_spec.eval_script
        )
        spec = self._sandbox_spec(
            params,
            files={
                "/root/patch.diff": patch if patch.endswith("\n") else patch + "\n",
                "/tmp/anyswe_eval.sh": eval_script,
            },
        )
        sandbox = AsyncSandbox(params.resolved_sandbox_provider, spec)
        try:
            await sandbox.start()
            result = await sandbox.exec(
                "bash /tmp/anyswe_eval.sh",
                cwd="/testbed",
                timeout_s=params.swebench_tests_timeout,
                user="root",
            )
        finally:
            await sandbox.stop()

        if result.error_type in ("timeout", "sandbox"):
            return False, "eval_timeout" if result.error_type == "timeout" else "sandbox"

        log = "\n".join(part for part in (result.stdout, result.stderr) if part)
        with tempfile.NamedTemporaryFile("w", suffix=".log") as log_file:
            log_file.write(log)
            log_file.flush()
            prediction = {
                "instance_id": params.instance_id,
                "model_name_or_path": params.body.model or "anyswe",
                "model_patch": patch,
            }
            report = get_eval_report(test_spec, prediction, log_file.name, include_tests_status=True)[
                params.instance_id
            ]
        return bool(report["resolved"]), None

    async def _grade_r2e_patch(self, params: AnySweInstanceConfig, patch: str) -> tuple[bool, Optional[str]]:
        instance = self._instance_dict(params)
        eval_script = instance.get("eval_script") or params.problem_info.get("eval_script")
        if not eval_script:
            eval_script = (
                "if [ -f /run_tests.sh ]; then bash /run_tests.sh; "
                "elif [ -f /testbed/run_tests.sh ]; then bash /testbed/run_tests.sh; "
                "elif [ -f /root/run_tests.sh ]; then bash /root/run_tests.sh; "
                "else echo 'R2E eval script not found'; exit 127; fi"
            )
        script = 'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"\n' + self._apply_patch_script() + str(eval_script)
        spec = self._sandbox_spec(
            params,
            files={
                "/root/patch.diff": patch if patch.endswith("\n") else patch + "\n",
                "/tmp/anyswe_eval.sh": script,
            },
        )
        sandbox = AsyncSandbox(params.resolved_sandbox_provider, spec)
        try:
            await sandbox.start()
            result = await sandbox.exec(
                "bash /tmp/anyswe_eval.sh",
                cwd="/testbed",
                timeout_s=params.swebench_tests_timeout,
                user="root",
            )
        finally:
            await sandbox.stop()

        if result.error_type in ("timeout", "sandbox"):
            return False, "eval_timeout" if result.error_type == "timeout" else "sandbox"

        log = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return _r2e_resolved(instance, log), None

    async def _run_agent_in_sandbox(self, params: AnySweInstanceConfig) -> NeMoGymResponse:
        files = {
            "/trajectories_mount/instruction.txt": (params.persistent_dir / "instruction.txt").read_text(),
            "/trajectories_mount/agent_runner.py": (params.persistent_dir / "agent_runner.py").read_text(),
        }
        spec = self._sandbox_spec(
            params,
            files=files,
        )
        result = None
        sandbox_failed = False
        sandbox = AsyncSandbox(params.resolved_sandbox_provider, spec)
        started = time.time()
        try:
            await sandbox.start()
            await sandbox.exec("mkdir -p /agent_deps_mount /trajectories_mount", timeout_s=30, user="root")
            if params.upload_agent_runtime:
                if params.agent_deps_archive is None:
                    raise RuntimeError("agent runtime archive is missing")
                await sandbox.upload(params.agent_deps_archive, "/tmp/anyswe_agent_deps.tar.gz")
            else:
                runtime = await sandbox.exec("test -x /agent_deps_mount/bin/python", timeout_s=30, user="root")
                if runtime.return_code != 0:
                    raise RuntimeError("task image does not contain /agent_deps_mount/bin/python")
            if _dataset_family(params.problem_info.get("dataset_name", "")) == "r2e":
                await sandbox.exec(
                    "rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests; "
                    "for f in /run_tests.sh /root/run_tests.sh /testbed/run_tests.sh; do "
                    'if grep -qs r2e_tests "$f"; then rm -f "$f"; fi; done',
                    timeout_s=30,
                    user="root",
                )
            command = "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py"
            if params.upload_agent_runtime:
                command = "tar -xzf /tmp/anyswe_agent_deps.tar.gz -C /agent_deps_mount && " + command
            result = await sandbox.exec(
                command,
                cwd="/testbed",
                env=self._sandbox_agent_env(params),
                timeout_s=params.swebench_agent_timeout,
                user="root",
            )
            for remote, local in (
                ("/trajectories_mount/response.json", params.persistent_dir / "response.json"),
                ("/trajectories_mount/patch.diff", params.persistent_dir / "patch.diff"),
            ):
                exists = await sandbox.exec(f"test -f {shlex.quote(remote)}", timeout_s=30, user="root")
                if exists.return_code == 0:
                    await sandbox.download(remote, local)
        except Exception as exc:
            sandbox_failed = True
            update_metrics(
                params.metrics_fpath,
                {
                    "resolved": False,
                    "patch_exists": False,
                    "error_kind": "sandbox",
                    "mask_sample": True,
                    "agent_run_time": time.time() - started,
                },
            )
            (params.persistent_dir / "sandbox_error.txt").write_text(repr(exc))
        finally:
            await sandbox.stop()

        patch_path = params.persistent_dir / "patch.diff"
        patch = patch_path.read_text() if patch_path.exists() else ""
        response_path = params.persistent_dir / "response.json"
        saved = NeMoGymResponse.model_validate_json(response_path.read_text()) if response_path.exists() else None

        agent_error = ""
        agent_timed_out = False
        error_kind = "sandbox" if sandbox_failed else None
        if result is not None:
            (params.persistent_dir / "agent_result.json").write_text(
                json.dumps(
                    {
                        "return_code": result.return_code,
                        "error_type": result.error_type,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
            )
            agent_error = "\n".join(part for part in (result.stdout, result.stderr) if part)
            agent_timed_out = result.error_type == "timeout"
            if result.error_type == "sandbox":
                error_kind = "sandbox"
        if saved is not None:
            agent_error += saved.model_dump_json()
        agent_error_kind = _classify_agent_error(agent_error)

        resolved = False
        if patch.strip() and error_kind is None:
            try:
                resolved, error_kind = await self._grade_sandbox_patch(params, patch)
            except SandboxCreateError as exc:
                error_kind = "sandbox"
                (params.persistent_dir / "eval_error.txt").write_text(repr(exc))
            except (asyncio.TimeoutError, TimeoutError) as exc:
                error_kind = "eval_timeout"
                (params.persistent_dir / "eval_error.txt").write_text(repr(exc))
            except Exception as exc:
                (params.persistent_dir / "eval_error.txt").write_text(repr(exc))

        eval_timed_out = error_kind == "eval_timeout"
        mask_sample = _should_mask_sample(resolved, agent_error_kind, agent_timed_out, error_kind)
        update_metrics(
            params.metrics_fpath,
            {
                "resolved": resolved,
                "patch_exists": bool(patch.strip()),
                "model_patch": patch or None,
                "agent_timed_out": agent_timed_out,
                "agent_error_kind": agent_error_kind,
                "eval_timed_out": eval_timed_out,
                "error_kind": error_kind,
                "mask_sample": mask_sample,
                "agent_run_time": time.time() - started,
            },
        )

        return NeMoGymResponse(
            id=f"anyswe-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model,
            object="response",
            output=saved.output if saved is not None else [],
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=(saved.tools or []) if saved is not None else [],
            usage=saved.usage if saved is not None else None,
            metadata={
                "input": json.dumps([]),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": _safe_config_json(params),
            },
        )

    def _setup_params(self, body: NeMoGymResponseCreateParamsNonStreaming) -> AnySweInstanceConfig:
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")
        persistent_dir = self._server.base_results_dir / (
            f"{instance_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        )
        persistent_dir.mkdir(parents=True, exist_ok=True)
        params = AnySweInstanceConfig(
            **self.config.model_dump(),
            **self._server.model_dump(),
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            container=self._sandbox_image(problem_info),
        )
        params.metrics_fpath.write_text("{}")
        GymAgentHarnessProcessor(config=params).write_runner()
        return params

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params = self._setup_params(body)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            return await self._run_agent_in_sandbox(params)
        except Exception:
            traceback_path = params.persistent_dir / "traceback.err"
            traceback_path.write_text(format_exc())
            print(f"[{params.instance_id}] exception: see {traceback_path}", flush=True)
            raise

    async def run(self, body: BaseRunRequest) -> AnySweVerifyResponse:
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
            body.responses_create_params.tool_choice = "auto"
            response = await self.responses(body.responses_create_params)

            meta, response.metadata = response.metadata, None
            metrics = SWEBenchMetrics.model_validate_json(meta["metrics"])

            return AnySweVerifyResponse(
                responses_create_params=body.responses_create_params.model_dump()
                | {"input": json.loads(meta["input"]), "tools": [t.model_dump() for t in (response.tools or [])]},
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                **metrics.model_dump(),
                instance_config=AnySweInstanceConfig.model_validate_json(meta["instance_config"]).model_dump(),
            )


if __name__ == "__main__":
    AnySweAgent.run_webserver()
