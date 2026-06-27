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

"""anyswe: run any thin Gym agent inside a SWE task sandbox and grade its patch.

The agent harness is provider-neutral: it provisions one working sandbox via the
shared ``swe_env`` library (``provision_and_collect`` over a ``SandboxProvider``),
runs the configured inner agent inside it via a generic dynamic-loader runner,
extracts the ``git diff`` patch, and grades it inline through the ``swe_env``
verifier in a fresh sandbox. No resources server, no second eval container.
"""

import base64
import dataclasses
import hashlib
import json
import shutil
import sys
import time
import uuid
from asyncio import Semaphore
from pathlib import Path
from subprocess import Popen
from traceback import format_exc
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.apptainer import ApptainerProvider
from nemo_gym.sandbox.providers.docker import DockerSandboxProvider
from responses_api_agents.swe_env.harness import SweTask
from responses_api_agents.swe_env.self_drive import provision_and_collect
from responses_api_agents.swe_env.verify_task import verify_task


class SWEBenchMetrics(BaseModel):
    resolved: Optional[bool] = None
    patch_exists: Optional[bool] = None
    model_patch: Optional[str] = None

    agent_timed_out: Optional[bool] = None
    agent_truncated: Optional[bool] = None
    eval_timed_out: Optional[bool] = None
    error_kind: Optional[str] = None
    mask_sample: Optional[bool] = None

    ray_queue_time: Optional[float] = None
    openhands_run_time: Optional[float] = None  # kept name for metric-schema parity
    generation_apptainer_spinup_time: Optional[float] = None
    final_eval_apptainer_spinup_time: Optional[float] = None
    final_eval_time: Optional[float] = None


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.write_text(json.dumps(existing | update))


def _safe_config_json(params: "AnySweInstanceConfig", indent: Optional[int] = None) -> str:
    """Serialize config without secrets: redact secret-looking agent_kwargs values."""

    def redact(value: Any, key: str = "") -> Any:
        if any(s in key.lower() for s in ("api_key", "apikey", "secret", "password", "token")):
            return "***"
        if isinstance(value, dict):
            return {k: redact(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    d = json.loads(params.model_dump_json())
    d["agent_kwargs"] = redact(d.get("agent_kwargs") or {})
    return json.dumps(d, indent=indent)


# --- dataset-name -> swe_env harness key ----------------------------------------------------------

_BENCHMARK_KEYS: list[Tuple[str, str]] = [
    ("R2E-Gym", "r2e-gym"),
    ("SWE-bench_Multilingual", "swe-bench-multilingual"),
    ("SWE-bench", "swe-bench"),
]


def _benchmark_key(dataset_name: str) -> str:
    """Map a HuggingFace dataset name to a registered ``swe_env`` harness key.

    Args:
        dataset_name: The dataset name from the task row (e.g.
            ``"princeton-nlp/SWE-bench_Verified"``).

    Returns:
        The matching registry key (substring match), defaulting to ``"swe-bench"``.
    """
    for needle, key in _BENCHMARK_KEYS:
        if needle in dataset_name:
            return key
    return "swe-bench"


def _build_swetask(problem_info: Dict[str, Any], *, flat_eval: bool = True) -> SweTask:
    """Build a ``SweTask`` from an anyswe task row's instance dict.

    SWE-bench rows nest the gradable fields (base commit, patches, test directives) inside a
    JSON-encoded ``instance_dict``; this unpacks them into a typed ``SweTask`` and flags
    ``flat_eval`` so the (otherwise apptainer-only) nested families grade host-side on any
    provider (e.g. docker).

    Args:
        problem_info: The task row metadata, including ``instance_dict`` (JSON string),
            ``dataset_name``, ``split``, and ``container_formatter``.

    Returns:
        A ``SweTask`` describing the instance, image, working directory, and grading inputs.
    """
    inst = (
        json.loads(problem_info["instance_dict"])
        if isinstance(problem_info.get("instance_dict"), str)
        else dict(problem_info.get("instance_dict", {}))
    )
    benchmark = _benchmark_key(problem_info.get("dataset_name", ""))
    instance_id = problem_info["instance_id"]
    image = _instance_image(problem_info.get("container_formatter"), instance_id)

    def _as_list(v: Any) -> list[str]:
        if isinstance(v, str):
            try:
                return list(json.loads(v))
            except (json.JSONDecodeError, TypeError):
                return [v] if v else []
        return list(v or [])

    return SweTask(
        instance_id=instance_id,
        image=image,
        base_commit=inst.get("base_commit"),
        repo_workdir="/testbed",
        test_patch=inst.get("test_patch", ""),
        fail_to_pass=_as_list(inst.get("FAIL_TO_PASS") or inst.get("fail_to_pass")),
        pass_to_pass=_as_list(inst.get("PASS_TO_PASS") or inst.get("pass_to_pass")),
        benchmark=benchmark,
        split=problem_info.get("split", "test"),
        metadata={"instance_dict": inst, "flat_eval": flat_eval},
    )


def _instance_image(container_formatter: Any, instance_id: str) -> str:
    """Resolve a pullable docker image reference for an instance.

    SWE-bench publishes per-instance images as ``swebench/sweb.eval.x86_64.<tag>`` where the
    tag lowercases the id and replaces ``__`` with ``_1776_``. A ``docker://`` prefix (the
    legacy apptainer formatter) is stripped so the docker provider can ``docker run`` it.

    Args:
        container_formatter: The configured formatter string (or list); may contain
            ``{instance_id}`` and a ``docker://`` scheme.
        instance_id: The benchmark instance id (e.g. ``astropy__astropy-13453``).

    Returns:
        A docker image reference (e.g.
        ``swebench/sweb.eval.x86_64.astropy_1776_astropy-13453:latest``).
    """
    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    fmt = fmt or "swebench/sweb.eval.x86_64.{instance_id}"
    # A local apptainer image (a ``.sif`` path) is used verbatim: substitute the raw instance_id
    # (no ``_1776_`` docker-tag mangling) and don't append a ``:latest`` docker tag, so a formatter
    # like ``/sifs/sweb.eval.x86_64.{instance_id}.sif`` resolves to an on-disk file the apptainer
    # provider can ``instance start`` directly (no registry pull).
    if fmt.endswith(".sif") or fmt.startswith(("/", ".")):
        return fmt.format(instance_id=instance_id)
    if fmt.startswith("docker://"):
        fmt = fmt[len("docker://") :]
    tag = instance_id.replace("__", "_1776_").lower()
    image = fmt.format(instance_id=tag)
    if ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    return image


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

_model_server = ModelServerRef(name="policy_model", type="responses_api_models") if MODEL_URL else None
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
        patch = subprocess.run(
            ["bash", "-c", "git add -A && git diff --cached"],
            capture_output=True, text=True, errors="replace", cwd=str(p),
        ).stdout
        print(f"patch: {{len(patch)}} chars from {{p}}", flush=True)
        break
Path("/trajectories_mount/patch.diff").write_text(patch)
"""


### Configuration


class AnySweAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str = Field(
        description="Import path to the agent module, e.g. responses_api_agents.hermes_agent.app"
    )
    agent_server_class: str = Field(description="Agent class name, e.g. HermesAgent")
    agent_config_class: str = Field(description="Agent config class name, e.g. HermesAgentConfig")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    container_formatter: str | list[str] = "docker://swebench/sweb.eval.x86_64.{instance_id}"
    sandbox_provider: Dict[str, Any] = Field(default_factory=lambda: {"docker": {}})
    # None = auto: nested (official) grading on apptainer, flat host-side grading on
    # docker/opensandbox (which can't run the nested harness). Set True/False to force.
    grade_flat_eval: Optional[bool] = None
    # Docker network for the agent container. "host" lets the in-container agent reach a
    # model server on host loopback; None uses the docker default (e.g. for a remote server).
    docker_network: Optional[str] = "host"
    swebench_tests_timeout: int = 1800
    swebench_agent_timeout: int = 2700
    concurrency: int = 16


class AnySweServerConfig(BaseModel):
    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    nemo_gym_root: Path
    agent_deps_dir: Path


class AnySweInstanceConfig(AnySweAgentConfig, AnySweServerConfig):
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    agent_run_id: str
    instance_dataset_path: Path
    metrics_fpath: Path
    mask_sample: bool = False

    @property
    def instance_id(self) -> str:
        return self.problem_info["instance_id"]


class AnySweVerifyResponse(SWEBenchMetrics, BaseVerifyResponse):
    instance_config: Dict[str, Any]


### Agent harness


class GymAgentHarnessProcessor(BaseModel):
    config: Any  # AnySweAgentConfig at setup time; AnySweInstanceConfig at run time

    @property
    def _parent(self) -> Path:
        return Path(__file__).parent

    @property
    def _agent_key(self) -> str:
        # responses_api_agents.hermes_agent.app -> hermes_agent
        return self.config.agent_server_module.split(".")[-2]

    def setup(self) -> Path:
        """Install agent deps into a portable prefix mounted read-only at /agent_deps_mount."""
        deps_dir = self._parent / f"anyswe_{self._agent_key}_deps"
        sentinel = deps_dir / ".installed"
        script = self._parent / "setup_scripts" / f"{self._agent_key}_deps.sh"
        # Reinstall when setup inputs change.
        shared = self._parent / "setup_scripts" / "_portable_python.sh"
        reqs = PARENT_DIR / "responses_api_agents" / self._agent_key / "requirements.txt"
        recipe_src = b"".join(p.read_bytes() for p in (script, shared, reqs) if p.exists()) or b"no-script"
        recipe = hashlib.sha256(recipe_src).hexdigest()
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            print(f"Agent deps already at {deps_dir}", flush=True)
            return deps_dir

        lock_path = deps_dir.parent / f".{deps_dir.name}.lockdir"
        while True:
            try:
                lock_path.mkdir(exist_ok=False)
                break
            except FileExistsError:
                if time.time() - lock_path.stat().st_mtime > 3600:
                    shutil.rmtree(lock_path, ignore_errors=True)
                    continue
                time.sleep(5)

        try:
            if sentinel.exists() and sentinel.read_text().strip() == recipe:
                print(f"Agent deps already at {deps_dir}", flush=True)
                return deps_dir

            if not script.exists():
                print(f"No setup script for {self._agent_key}, skipping deps install", flush=True)
                deps_dir.mkdir(parents=True, exist_ok=True)
                sentinel.write_text(recipe)
                return deps_dir

            deps_dir.mkdir(parents=True, exist_ok=True)
            proc = Popen(f"DEPS_DIR={deps_dir} NEMO_GYM_ROOT={PARENT_DIR} bash {script}", shell=True)
            assert proc.wait() == 0, f"Agent deps setup failed ({script})"
            sentinel.write_text(recipe)
            return deps_dir
        finally:
            shutil.rmtree(lock_path, ignore_errors=True)

    def write_runner(self) -> str:
        """Write instruction.txt + agent_runner.py into the run dir; return the launch command.

        Returns:
            The shell command that runs the dynamic-loader runner inside the sandbox.
        """
        cfg: AnySweInstanceConfig = self.config
        (cfg.persistent_dir / "instruction.txt").write_text(cfg.problem_info.get("problem_statement", ""))
        runner = _RUNNER_TEMPLATE.format(
            agent_module=cfg.agent_server_module,
            agent_class=cfg.agent_server_class,
            agent_cfg_class=cfg.agent_config_class,
            agent_class_lower=cfg.agent_server_class.lower(),
        )
        (cfg.persistent_dir / "agent_runner.py").write_text(runner)
        return "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py"


### Agent server


class AnySweAgent(SimpleResponsesAPIAgent):
    """Runs a thin Gym agent inside each task's sandbox and grades its patch."""

    config: AnySweAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[AnySweServerConfig] = None

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)

        model_url = ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            model_url = self.server_client._build_server_base_url(model_cfg)

        agent_deps_dir = GymAgentHarnessProcessor(config=self.config).setup()

        session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        workspace = Path(__file__).parent

        self._server = AnySweServerConfig(
            run_session_id=session_id,
            base_results_dir=workspace / f"anyswe_results_{session_id}",
            model_server_url=model_url,
            nemo_gym_root=PARENT_DIR,
            agent_deps_dir=agent_deps_dir,
        )
        super().model_post_init(context)

    def _setup_params(self, body: NeMoGymResponseCreateParamsNonStreaming) -> Tuple[AnySweInstanceConfig, str]:
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")

        instance_dir = f"{instance_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        persistent_dir = self._server.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{instance_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        dataset_dir = persistent_dir / "instance_datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = dataset_dir / f"{agent_run_id}.jsonl"
        # Accept instance_dict as a JSON string or an already-parsed dict (mirrors _build_swetask),
        # so a dict-valued row doesn't raise TypeError and mask the whole run with an opaque error.
        raw_instance_dict = problem_info["instance_dict"]
        instance_dict = (
            json.loads(raw_instance_dict) if isinstance(raw_instance_dict, str) else dict(raw_instance_dict)
        )
        instance_dict.setdefault("repo_name", instance_dict.get("repo", ""))
        instance_dataset_path.write_text(json.dumps(instance_dict) + "\n")

        params = AnySweInstanceConfig(
            **self.config.model_dump(),
            **self._server.model_dump(),
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            agent_run_id=agent_run_id,
            instance_dataset_path=instance_dataset_path,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
        )
        params.metrics_fpath.write_text("{}")

        launch_command = GymAgentHarnessProcessor(config=params).write_runner()
        return params, launch_command

    def _provider(self, params: AnySweInstanceConfig):
        """Build a sandbox provider with the per-instance mounts the runner needs.

        The deps prefix, the NeMo Gym tree, and the run dir are bind-mounted at the fixed
        paths the runner expects. For docker, ``docker_network`` (default ``"host"``) lets the
        in-container agent reach a host-side model server over loopback. For apptainer, the
        mounts are applied as ``--bind`` (via ``exec.default_binds``) and ``--writable-tmpfs``
        is added so ``/testbed`` is editable.

        Args:
            params: The per-instance config carrying the run dir + host mount sources.

        Returns:
            An ``ApptainerProvider`` (apptainer) or ``DockerSandboxProvider`` (docker) configured
            with the mounts, or the configured ``sandbox_provider`` mapping for other backends.
        """
        name = next(iter(self.config.sandbox_provider), "docker")
        if name == "apptainer":
            # Symmetric with the docker run_args below: the runner expects the deps prefix, the
            # NeMo Gym tree, and the run dir at fixed paths. The apptainer provider applies these
            # as ``--bind`` at ``instance start`` (via exec.default_binds), so the in-container
            # agent finds /agent_deps_mount, /nemo_gym_mount, and /trajectories_mount.
            appt = {
                k: v
                for k, v in (self.config.sandbox_provider.get("apptainer") or {}).items()
                if k in ("exec", "create", "probe")
            }
            exec_cfg = dict(appt.get("exec") or {})
            exec_cfg["default_binds"] = list(exec_cfg.get("default_binds") or []) + [
                f"{params.persistent_dir}:/trajectories_mount",
                f"{params.nemo_gym_root}:/nemo_gym_mount:ro",
                f"{params.agent_deps_dir}:/agent_deps_mount:ro",
            ]
            appt["exec"] = exec_cfg
            # The base .sif is mounted read-only; without an overlay the agent's edits to /testbed
            # fail with "Read-only file system" and the captured ``git diff`` is empty. --writable-tmpfs
            # adds a tmpfs overlay so /testbed is editable (parity with the swe_agents apptainer path).
            create_cfg = dict(appt.get("create") or {})
            start_args = list(create_cfg.get("extra_start_args") or [])
            if "--writable-tmpfs" not in start_args:
                start_args.append("--writable-tmpfs")
            create_cfg["extra_start_args"] = start_args
            appt["create"] = create_cfg
            return ApptainerProvider(**appt)
        if name != "docker":
            return self.config.sandbox_provider
        return DockerSandboxProvider(
            network=self.config.docker_network,
            run_args=[
                "-v",
                f"{params.persistent_dir}:/trajectories_mount",
                "-v",
                f"{params.nemo_gym_root}:/nemo_gym_mount:ro",
                "-v",
                f"{params.agent_deps_dir}:/agent_deps_mount:ro",
            ],
        )

    def _grading_provider(self):
        """Provider config for the (fresh) grading sandbox.

        The official (nested) grade applies the patch and runs the test suite inside the sandbox,
        so /testbed must be writable. For apptainer the base .sif is read-only, so we add
        --writable-tmpfs (same reason as the agent sandbox in ``_provider``); grading needs no
        per-instance binds. Non-apptainer providers (docker) are returned unchanged.
        """
        name = next(iter(self.config.sandbox_provider), "docker")
        if name != "apptainer":
            return self.config.sandbox_provider
        appt = {
            k: v
            for k, v in (self.config.sandbox_provider.get("apptainer") or {}).items()
            if k in ("exec", "create", "probe")
        }
        create_cfg = dict(appt.get("create") or {})
        start_args = list(create_cfg.get("extra_start_args") or [])
        if "--writable-tmpfs" not in start_args:
            start_args.append("--writable-tmpfs")
        create_cfg["extra_start_args"] = start_args
        appt["create"] = create_cfg
        return {"apptainer": appt}

    def _agent_egress_env(self, params: AnySweInstanceConfig) -> Dict[str, str]:
        """Build the NGSWE_* env the in-container runner reads for model egress + sampling."""
        sampling = {
            k: v
            for k, v in (params.body.model_dump().items())
            if k in ("temperature", "top_p", "max_output_tokens") and v is not None
        }
        env: Dict[str, str] = {
            "NGSWE_MODEL_NAME": params.body.model or "",
            "NGSWE_AGENT_KWARGS_B64": base64.b64encode(json.dumps(params.agent_kwargs).encode()).decode(),
            "NGSWE_SAMPLING_B64": base64.b64encode(json.dumps(sampling).encode()).decode(),
        }
        if params.model_server_url:
            env["NGSWE_MODEL_URL"] = params.model_server_url
        return env

    # Request handlers

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params, launch_command = self._setup_params(body)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            return await self._inner_responses(params, launch_command)
        except Exception:
            tb_path = params.persistent_dir / "traceback.err"
            tb_path.write_text(format_exc())
            print(f"[{params.instance_id}] exception: see {tb_path}", file=sys.stderr)
            raise

    async def _inner_responses(self, params: AnySweInstanceConfig, launch_command: str) -> NeMoGymResponse:
        provider_name = next(iter(self.config.sandbox_provider), "docker")
        flat = self.config.grade_flat_eval
        if flat is None:
            flat = provider_name != "apptainer"  # nested (official) only on apptainer
        task = _build_swetask(params.problem_info, flat_eval=flat)
        provider = self._provider(params)

        t0 = time.time()
        result = await provision_and_collect(
            task,
            provider=provider,
            agent_launch_command=launch_command,
            extra_env=self._agent_egress_env(params),
            agent_timeout_s=params.swebench_agent_timeout,
        )
        agent_run_time = time.time() - t0
        agent_timed_out = result.get("error_type") == "timeout"

        # The runner writes patch.diff into /trajectories_mount (= persistent_dir); fall back to
        # the git-diff provision_and_collect captured.
        patch_file = params.persistent_dir / "patch.diff"
        patch = patch_file.read_text() if patch_file.exists() else (result.get("patch") or "")

        # Grade the patch in a fresh sandbox (hermetic). task.metadata['flat_eval']=True so the
        # nested families grade host-side on docker.
        report = await verify_task(
            self._grading_provider(),
            dataclasses.replace(task, model_patch=patch),
            eval_timeout_s=params.swebench_tests_timeout,
        )
        resolved = bool(report.resolved)

        # Read the inner agent's NeMoGymResponse (it self-writes response.json) to recover the
        # output trajectory and any truncation signal.
        response_path = params.persistent_dir / "response.json"
        saved = NeMoGymResponse.model_validate_json(response_path.read_text()) if response_path.exists() else None
        # The inner agent ran out of turns / blew its context window if it reports the Responses
        # API "incomplete" status. A patch that happens to pass in that case is an accidental
        # reward, so mask it (mirrors main's swe_agents, which masks resolved AND
        # max_iteration/context_window) to keep it out of the training gradient.
        agent_truncated = bool(saved is not None and saved.status == "incomplete")
        if report.error_kind is not None or agent_timed_out or (resolved and agent_truncated):
            params.mask_sample = True

        update_metrics(
            params.metrics_fpath,
            {
                "resolved": resolved,
                "patch_exists": bool(patch.strip()),
                "model_patch": patch or None,
                "agent_timed_out": agent_timed_out,
                "agent_truncated": agent_truncated,
                "error_kind": report.error_kind,
                "mask_sample": params.mask_sample,
                "openhands_run_time": agent_run_time,
            },
        )

        output_items = saved.output if saved is not None else []
        tools = (saved.tools or []) if saved is not None else []

        return NeMoGymResponse(
            id=f"anyswe-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model,
            object="response",
            output=output_items,
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=tools,
            metadata={
                "input": json.dumps([]),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": _safe_config_json(params),
            },
        )

    async def run(self, body: BaseRunRequest) -> AnySweVerifyResponse:
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
            body.responses_create_params.tool_choice = "auto"
            try:
                response = await self.responses(body.responses_create_params)
            except Exception:
                # Failure isolation: one bad instance must not abort the whole cell, and resume
                # must still see a row. Return a present, masked, reward-0 result instead of raising.
                print(f"[anyswe] run failed: {format_exc()}", file=sys.stderr)
                return AnySweVerifyResponse(
                    responses_create_params=body.responses_create_params.model_dump(),
                    response=NeMoGymResponse(
                        id="anyswe-error",
                        created_at=int(time.time()),
                        model=body.responses_create_params.model,
                        object="response",
                        output=[],
                        parallel_tool_calls=True,
                        tool_choice="auto",
                        tools=[],
                    ),
                    reward=0.0,
                    resolved=False,
                    patch_exists=False,
                    mask_sample=True,
                    error_kind="agent_error",
                    instance_config={},
                )

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
