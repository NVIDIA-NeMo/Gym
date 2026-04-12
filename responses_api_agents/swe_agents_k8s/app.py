# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Kubernetes variant of swe_agents. Replaces Apptainer + Ray with K8s Jobs.
Subclasses SWEBenchWrapper; only the execution layer changes.
"""
import glob
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import ConfigDict, Field

from nemo_gym.k8s_runner import K8sJobRunner
from nemo_gym.profiling import Profiler
from responses_api_agents.swe_agents.app import (
    BaseDatasetHarnessProcessor,
    ExecuteContainerCommandArgs,
    RunOpenHandsAgent,
    SWEBenchMetrics,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
    SWEBenchWrapperInstanceConfig,
    update_metrics,
)


class SWEBenchK8sConfig(SWEBenchWrapperConfig):
    k8s_namespace: str = Field(default="default")
    workspace_pvc_name: str = Field(default="swe-workspace")
    setup_pvc_name: str = Field(default="swe-setup")
    workspace_mount_path: str = Field(default="/nemogym-workspace")
    setup_mount_path: str = Field(default="/nemogym-setup")
    k8s_memory_limit: str = Field(default="32Gi")
    k8s_cpu_limit: str = Field(default="4")


class SWEBenchK8sWrapper(SWEBenchWrapper):
    """K8s variant of SWEBenchWrapper. Inherits all setup, result parsing,
    and trajectory handling. Overrides execution to use K8s Jobs."""

    config: SWEBenchK8sConfig

    _k8s_runner: Optional[K8sJobRunner] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Override base_results_dir to workspace PVC mount
        self._swe_bench_wrapper_server_config.base_results_dir = (
            Path(self.config.workspace_mount_path) / f"swebench_results_{self._swe_bench_wrapper_server_config.run_session_id}"
        )
        self._k8s_runner = K8sJobRunner(namespace=self.config.k8s_namespace)

    # Stub out apptainer command building so inherited _setup_params doesn't break.
    def _build_apptainer_command(self, params, command):
        return ""

    def _container_to_image(self, container: str) -> str:
        if container.startswith("docker://"):
            return container[len("docker://"):]
        raise ValueError(f"K8s requires docker:// format, got: {container!r}")

    # ------------------------------------------------------------------
    # K8s volume spec builder
    # ------------------------------------------------------------------

    def _build_k8s_job_spec(
        self, params: SWEBenchWrapperInstanceConfig, command: ExecuteContainerCommandArgs
    ) -> Tuple[list, list, dict]:
        """Returns (volumes, volume_mounts, env) for a K8s Job. Uses PVCs only, no hostPath."""
        data_point = params.problem_info
        ws = self.config.workspace_mount_path
        volumes: list[dict] = []
        volume_mounts: list[dict] = []
        env: Dict[str, str] = {}

        # Workspace PVC -- all instance data lives here
        volumes.append({"name": "workspace", "persistentVolumeClaim": {"claimName": self.config.workspace_pvc_name}})
        # Mount full workspace so all paths under it are accessible
        volume_mounts.append({"name": "workspace", "mountPath": ws})

        # persistent_dir is a subdir of workspace. Mount it at /trajectories_mount.
        persistent_subpath = str(params.persistent_dir.relative_to(ws))
        volume_mounts.append({"name": "workspace", "mountPath": "/trajectories_mount", "subPath": persistent_subpath})

        # Instance dataset at /root/dataset/data.jsonl
        dataset_subpath = str(params.instance_dataset_path.relative_to(ws))
        volume_mounts.append({"name": "workspace", "mountPath": "/root/dataset/data.jsonl", "subPath": dataset_subpath, "readOnly": True})

        # Setup PVC -- read-only tool installations
        volumes.append({"name": "setup", "persistentVolumeClaim": {"claimName": self.config.setup_pvc_name}})

        openhands_dir = f"{params.openhands_setup_dir}/OpenHands"

        # OpenHands (read-only base)
        volume_mounts.extend([
            {"name": "setup", "mountPath": "/openhands_setup/OpenHands", "subPath": "openhands/OpenHands", "readOnly": True},
            {"name": "setup", "mountPath": openhands_dir, "subPath": "openhands/OpenHands", "readOnly": True},
        ])

        # Writable OpenHands subdirs on workspace PVC.
        # Job writes here; coordinator reads from same PVC path after job completes.
        oh_writable_base = f"oh_writable/{params.agent_run_id}"
        for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
            writable_subpath = f"{oh_writable_base}/{subdir}"
            volume_mounts.extend([
                {"name": "workspace", "mountPath": f"/openhands_setup/OpenHands/{subdir}", "subPath": writable_subpath},
                {"name": "workspace", "mountPath": f"{openhands_dir}/{subdir}", "subPath": writable_subpath},
            ])

        # miniforge3
        volume_mounts.extend([
            {"name": "setup", "mountPath": "/openhands_setup/miniforge3", "subPath": "openhands/miniforge3", "readOnly": True},
            {"name": "setup", "mountPath": f"{params.openhands_setup_dir}/miniforge3", "subPath": "openhands/miniforge3", "readOnly": True},
        ])

        # Prompt templates (on workspace PVC, written by _setup_params)
        if params.resolved_user_prompt_template:
            tpl_subpath = str(Path(params.resolved_user_prompt_template).relative_to(ws))
            volume_mounts.append({"name": "workspace", "mountPath": "/openhands_setup/OpenHands/user_prompt.j2", "subPath": tpl_subpath, "readOnly": True})

        if params.resolved_system_prompt_template:
            tpl_subpath = str(Path(params.resolved_system_prompt_template).relative_to(ws))
            volume_mounts.append({"name": "workspace", "mountPath": "/openhands_setup/OpenHands/system_prompt.j2", "subPath": tpl_subpath, "readOnly": True})
            volume_mounts.append({"name": "workspace", "mountPath": "/openhands_setup/OpenHands/system_prompt_long_horizon.j2", "subPath": tpl_subpath, "readOnly": True})

        # Eval-mode mounts
        if command.mode == "eval" and data_point["dataset_name"] != "nv-internal-1":
            volume_mounts.extend([
                {"name": "setup", "mountPath": "/swebench_setup", "subPath": "swebench", "readOnly": True},
                {"name": "setup", "mountPath": str(params.swebench_setup_dir), "subPath": "swebench", "readOnly": True},
            ])

        if command.mode == "eval" and "SWE-bench_Multilingual" in data_point["dataset_name"]:
            volume_mounts.extend([
                {"name": "setup", "mountPath": "/swebench_multilingual_setup", "subPath": "swebench_multilingual", "readOnly": True},
                {"name": "setup", "mountPath": str(params.swebench_multilingual_setup_dir), "subPath": "swebench_multilingual", "readOnly": True},
            ])

        if command.mode == "eval" and data_point["dataset_name"] == "nv-internal-1":
            for fname, mpath in [("run_script.sh", "/root/run_script.sh"), ("parsing_script.py", "/root/parsing_script.py")]:
                fpath_subpath = str((params.persistent_dir / fname).relative_to(ws))
                volume_mounts.append({"name": "workspace", "mountPath": mpath, "subPath": fpath_subpath, "readOnly": True})
            patch_subpath = str(params.model_patch_path.relative_to(ws))
            volume_mounts.append({"name": "workspace", "mountPath": "/root/patch.diff", "subPath": patch_subpath, "readOnly": True})

        if command.mode == "eval" and "R2E-Gym" in data_point["dataset_name"]:
            volume_mounts.extend([
                {"name": "setup", "mountPath": "/r2egym_setup", "subPath": "r2e_gym", "readOnly": True},
                {"name": "setup", "mountPath": str(params.r2e_gym_setup_dir), "subPath": "r2e_gym", "readOnly": True},
            ])

        if command.mode == "eval" and "SWE-rebench" in data_point["dataset_name"]:
            volume_mounts.append({"name": "setup", "mountPath": "/swe_rebench_setup", "subPath": "swe_rebench", "readOnly": True})

            for fname, mpath in [("test_patch.diff", "/root/test_patch.diff")]:
                fpath_subpath = str((params.persistent_dir / fname).relative_to(ws))
                volume_mounts.append({"name": "workspace", "mountPath": mpath, "subPath": fpath_subpath, "readOnly": True})
            patch_subpath = str(params.model_patch_path.relative_to(ws))
            volume_mounts.append({"name": "workspace", "mountPath": "/root/patch.diff", "subPath": patch_subpath, "readOnly": True})

            eval_meta_dir = params.persistent_dir / "eval_meta"
            for fname in ["expected_passed.json", "fail_to_pass.json", "pass_to_pass.json"]:
                meta_subpath = str((eval_meta_dir / fname).relative_to(ws))
                volume_mounts.append({"name": "workspace", "mountPath": f"/eval_meta/{fname}", "subPath": meta_subpath, "readOnly": True})

        if "SWE-rebench" in data_point["dataset_name"]:
            env["_JAVA_OPTIONS"] = "-Djava.net.preferIPv6Addresses=false"

        return volumes, volume_mounts, env

    def _build_container_command(
        self, params: SWEBenchWrapperInstanceConfig, command: ExecuteContainerCommandArgs
    ) -> list[str]:
        data_point = params.problem_info
        parts = ["echo '127.0.0.1 localhost' >/etc/hosts"]

        if command.mode == "agent" and "R2E-Gym" in data_point["dataset_name"]:
            for root_dir in ["", "/root", "/testbed"]:
                parts.append(
                    f"rm -rf {root_dir}/r2e_tests && "
                    f"if grep -qs r2e_tests {root_dir}/run_tests.sh; then rm -rf {root_dir}/run_tests.sh; fi"
                )

        parts.append(command.command)
        return ["bash", "-c", " && ".join(parts)]

    # ------------------------------------------------------------------
    # K8s job execution
    # ------------------------------------------------------------------

    async def _run_k8s_job(
        self,
        params: SWEBenchWrapperInstanceConfig,
        command: ExecuteContainerCommandArgs,
        image: str,
        volumes: list,
        volume_mounts: list,
        env: dict,
        suffix: str,
    ) -> str:
        job_name = f"swe-{params.agent_run_id[:20]}-{suffix}"[:63].rstrip("-")

        exit_code, stdout, stderr = await self._k8s_runner.run_job(
            job_name=job_name,
            image=image,
            command=self._build_container_command(params, command),
            timeout=command.timeout,
            env=env,
            volume_mounts=volume_mounts,
            volumes=volumes,
            resource_limits={"memory": self.config.k8s_memory_limit, "cpu": self.config.k8s_cpu_limit},
            cleanup=True,
            poll_interval=5.0,
        )

        if exit_code != 0:
            raise ValueError(f"Job {job_name} failed (exit {exit_code}). stdout[-2000:]: {stdout[-2000:]}")

        pred_files = glob.glob(command.expected_file_pattern, recursive=True)
        if len(pred_files) == 1:
            return pred_files[0]
        elif len(pred_files) > 1:
            return max(pred_files, key=os.path.getmtime)
        raise ValueError(f"No file matching {command.expected_file_pattern}")

    def _openhands_dir_copy_from_pvc(
        self, params: SWEBenchWrapperInstanceConfig, output_file_path: Optional[str]
    ) -> Optional[str]:
        """K8s version of RunOpenHandsAgent._openhands_dir_copy_from_host.
        Reads from workspace PVC instead of the OpenHands install dir."""
        ws = self.config.workspace_mount_path
        eval_dir_on_pvc = Path(ws) / f"oh_writable/{params.agent_run_id}/evaluation/oh"

        trajectories_root = params.trajectories_root
        llm_completions_dir = trajectories_root / "llm_completions" / params.instance_id
        trajectories_root.mkdir(parents=True, exist_ok=True)
        llm_completions_dir.mkdir(parents=True, exist_ok=True)

        dest_output: Optional[str] = None
        if output_file_path:
            source_output = Path(output_file_path)
            if not source_output.is_absolute():
                source_output = eval_dir_on_pvc / source_output
            if not source_output.exists():
                output_candidates = sorted(eval_dir_on_pvc.glob("*/*/*/output.jsonl"), key=os.path.getmtime)
                if not output_candidates:
                    raise FileNotFoundError(f"No output.jsonl under {eval_dir_on_pvc} for {params.instance_id}")
                source_output = output_candidates[-1]

            shutil.copy2(source_output, params.prediction_path)
            dest_output = str(params.prediction_path)

        completion_candidates = glob.glob(str(eval_dir_on_pvc / "*/*/*/llm_completions/*/*.json"))
        if completion_candidates:
            latest_completion = max(completion_candidates, key=os.path.getmtime)
            shutil.copy2(latest_completion, llm_completions_dir / Path(latest_completion).name)

        shutil.rmtree(eval_dir_on_pvc, ignore_errors=True)
        return dest_output

    async def _process_single_datapoint(
        self,
        params: SWEBenchWrapperInstanceConfig,
        image: str,
        agent_spec: Tuple[list, list, dict],
        eval_spec: Tuple[list, list, dict],
    ) -> Optional[Path]:
        """Run agent then eval sequentially as K8s Jobs."""
        instance_id = params.instance_id
        if params.debug:
            profiler = Profiler(name=instance_id, base_profile_dir=params.profiling_mounted_dir)
            profiler.start()

        metrics = SWEBenchMetrics(ray_queue_time=time.time() - params.ray_queue_timestamp)
        metrics.openhands_run_time = -time.time()
        metrics.generation_apptainer_spinup_time = metrics.openhands_run_time
        metrics.final_eval_apptainer_spinup_time = metrics.openhands_run_time

        agent_vols, agent_mounts, agent_env = agent_spec
        eval_vols, eval_mounts, eval_env = eval_spec

        # Agent phase
        try:
            out_file_in_eval = await self._run_k8s_job(
                params, params.agent_command, image, agent_vols, agent_mounts, agent_env, "agent"
            )
            out_file = self._openhands_dir_copy_from_pvc(params, output_file_path=out_file_in_eval)
        except Exception as e:
            print(f"Agent failed for {instance_id}: {e}", flush=True)
            try:
                self._openhands_dir_copy_from_pvc(params, output_file_path=None)
            except Exception:
                pass
            metrics.openhands_run_time += time.time()
            metrics.patch_exists = False
            metrics.final_eval_apptainer_spinup_time = None
            update_metrics(params.metrics_fpath, metrics.model_dump())
            if params.debug:
                profiler.stop()
            return None

        generation_ts = float(params.generation_apptainer_spinup_timestamp_fpath.read_text())
        metrics.generation_apptainer_spinup_time += generation_ts
        metrics.openhands_run_time += time.time()

        with open(out_file, "r") as f:
            out_dict = json.loads(f.read().strip())

        patch = out_dict["test_result"]["git_patch"] or None
        patch = patch + "\n" if patch and not patch.endswith("\n") else patch
        metrics.model_patch = patch

        params.output_for_eval_path.parent.mkdir(parents=True, exist_ok=True)
        with params.output_for_eval_path.open("w") as f:
            f.write(json.dumps({
                "model_name_or_path": out_dict["metadata"]["llm_config"]["model"],
                "instance_id": out_dict["instance_id"],
                "model_patch": patch,
                "oh_time_metrics": out_dict["metrics"],
            }))

        if not patch:
            metrics.patch_exists = False
            metrics.final_eval_apptainer_spinup_time = None
            update_metrics(params.metrics_fpath, metrics.model_dump())
            return None

        with open(params.model_patch_path, "w") as f:
            f.write(patch)

        # Eval phase (sequential, after agent)
        metrics.final_eval_time = -time.time()
        try:
            report_file = await self._run_k8s_job(
                params, params.eval_command, image, eval_vols, eval_mounts, eval_env, "eval"
            )
        except Exception as e:
            print(f"Eval failed for {instance_id}: {e}", flush=True)
            metrics.final_eval_time += time.time()
            metrics.patch_exists = True
            update_metrics(params.metrics_fpath, metrics.model_dump())
            if params.debug:
                profiler.stop()
            return None

        final_eval_ts = float(params.final_eval_apptainer_spinup_timestamp_fpath.read_text())
        metrics.final_eval_apptainer_spinup_time += final_eval_ts
        metrics.final_eval_time += time.time()

        metrics.patch_exists = True
        update_metrics(params.metrics_fpath, metrics.model_dump())

        if params.debug:
            profiler.stop()

        return report_file

    # ------------------------------------------------------------------
    # Override _inner_responses: K8s execution instead of Ray remote
    # ------------------------------------------------------------------

    async def _inner_responses(self, params, dataset_processor):
        image = self._container_to_image(params.container)
        agent_spec = self._build_k8s_job_spec(params, params.agent_command)
        eval_spec = self._build_k8s_job_spec(params, params.eval_command)

        maybe_report_file = await self._process_single_datapoint(params, image, agent_spec, eval_spec)

        # Result processing (same logic as parent, can't call super due to Ray call)
        from openai.types.responses.function_tool import FunctionTool

        from nemo_gym.openai_utils import NeMoGymResponse
        from responses_api_models.vllm_model.app import split_responses_input_output_items

        metrics_to_update = {}

        if maybe_report_file:
            dataset_processor.postprocess_after_run(maybe_report_file)
            report = json.loads(Path(maybe_report_file).read_text())
            assert params.instance_id in report, f"Report missing key: {params.instance_id}"
            metrics_to_update["resolved"] = report[params.instance_id]["resolved"]
        else:
            metrics_to_update["resolved"] = False

        trajectories_dir = params.persistent_dir / "trajectories"
        chat_trajectory, chat_tools = self.get_openhands_trajectory_from_completions(trajectories_dir, params.instance_id)

        tools = [FunctionTool.model_validate(t["function"] | {"type": "function"}) for t in chat_tools]
        responses_items = self._vllm_converter.chat_completions_messages_to_responses_items(chat_trajectory)
        input_items, output_items = split_responses_input_output_items(responses_items)

        update_metrics(params.metrics_fpath, metrics_to_update)

        return NeMoGymResponse(
            id=f"swebench-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model,
            object="response",
            output=output_items,
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=tools,
            metadata={
                "input": json.dumps([i.model_dump() for i in input_items]),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": params.model_dump_json(),
            },
        )


if __name__ == "__main__":
    SWEBenchK8sWrapper.run_webserver()
