# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shlex
import shutil
import signal
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh"
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "Launcher tests require Bash")
class TestSuperVllmLauncher(unittest.TestCase):
    def setUp(self) -> None:
        workdir = TemporaryDirectory(prefix="gym-launcher-")
        self.addCleanup(workdir.cleanup)
        self.workdir = workdir.name
        # Never inherit cluster credentials, tuning overrides, or real sbatch commands.
        self.env = {
            "PATH": os.defpath,
            # Bypass the sleep() stubs even in background functions on macOS Bash.
            "TEST_SLEEP": shutil.which("sleep", path=os.defpath),
            "USER": "launcher-test",
            "MODEL": "/test/model",
            "CONTAINER": "/test/image.sqsh",
            "MOUNTS": "/test:/test",
            "VLLM_CONFIG": "/dev/null",
            "EXPERIMENT_NAME": "launcher-test",
            "NUM_PREFILL_NODES": "4",
            "NUM_DECODE_NODES": "4",
            "SLURM_PROCID": "0",
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_USER": "launcher-test",
            "ROUTER_NODE": "node0",
            "ALL_NODES": "node0 node1 node2 node3 node4 node5 node6 node7",
        }

    def run_shell(self, command: str, *args: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
        proc = subprocess.Popen(
            [BASH, "--noprofile", "--norc", "-c", command, "launcher-test", *args],
            env=self.env | (env or {}),
            cwd=self.workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # The failure-path tests must terminate instead of leaking a polling worker.
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            self.fail(f"Launcher did not terminate. stdout={stdout!r}, stderr={stderr!r}")
        return proc.returncode, stdout, stderr

    def require_batch_bash(self) -> None:
        # The batch script uses [[ -v ]] and wait -n -p; macOS ships Bash 3.2.
        status, _, _ = self.run_shell("(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) ))")
        if status:
            self.skipTest("Generated batch scripts require Bash 5.1 or newer")

    def capture_submission(self, *eval_args, env=None):
        # Capture generated commands and both sbatch calls without submitting jobs.
        stub = r"""
sbatch() {
    if [[ -n "${vllm_command:-}" ]]; then
        printf '%s\0' "$eval_command" "$vllm_command" "$batch_command" >&2
    fi
    printf '%s\0' "$@" >&2
    printf '\0' >&2
    printf '12345\n'
}
launcher_script=$1
shift
source "$launcher_script" "$@"
"""
        status, _, captured = self.run_shell(stub, str(SCRIPT), *eval_args, env=env)
        self.assertEqual(status, 0, captured)
        eval_command, pd_command, batch_command, submissions = captured.split("\0", 3)
        self.assertTrue(submissions.endswith("\0\0"))
        calls = [call.split("\0") for call in submissions.removesuffix("\0\0").split("\0\0")]
        return eval_command, pd_command, batch_command, calls

    def generate_commands(self, *overrides, env=None):
        eval_command, pd_command, _, _ = self.capture_submission("--config", "benchmark.yaml", *overrides, env=env)
        return eval_command, pd_command

    def eval_arguments(self, *overrides: str, env: dict[str, str] | None = None) -> list[str]:
        command, _ = self.generate_commands(*overrides, env=env)
        command = command.replace("source /opt/Gym_venv/bin/activate", ":").replace("cd /opt/Gym\n", ":\n")
        stubs = r"""
# Supply model parameters for the default /dev/null fixture; real configs replace these.
GYM_MODEL_PARAMS=(++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0)
gym() {
    if [[ "$2" == run ]]; then printf '%s\0' "$@"; fi
}
date() { printf '%s\n' "${TEST_DATE:-20260909_120000}"; }
getent() { printf '10.0.0.1 node0\n'; }
"""
        status, stdout, stderr = self.run_shell(stubs + command, env=env)
        self.assertEqual(status, 0, stderr)
        return stdout.rstrip("\0").split("\0")

    def settings(self, args, key):
        return [arg for arg in args if arg.lstrip("+").startswith(key + "=")]

    def test_mooncake_metrics_endpoint_uses_only_master_node(self) -> None:
        for mode in ("independent", "coupled"):
            for enabled in ("0", "1"):
                with self.subTest(mode=mode, enabled=enabled), TemporaryDirectory() as temporary_dir:
                    command, _ = self.generate_commands(
                        env={"VLLM_PD_DEPLOYMENT_MODE": mode, "ENABLE_MOONCAKE": enabled}
                    )
                    start = 'read -r -a nodes <<< "$ALL_NODES"'
                    setup = start + command.split(start, 1)[1].split("gym_config_args+=(--config", 1)[0]
                    config_path = Path(temporary_dir) / "metrics.yaml"
                    status, _, stderr = self.run_shell(
                        setup, env={"inference_metrics_config": str(config_path), "ROUTER_NODE": "separate-router"}
                    )
                    self.assertEqual(status, 0, stderr)
                    metrics = yaml.safe_load(config_path.read_text())["inference_metrics"]
                    self.assertEqual(len(metrics["endpoints"]), 8 if mode == "independent" else 2)
                    if enabled == "1":
                        self.assertEqual(metrics["mooncake_endpoint"], "http://node0:9003/metrics")
                    else:
                        self.assertNotIn("mooncake_endpoint", metrics)

    def model_arguments(self, *, config_path: Path, enable_mooncake: bool, rank: int = 0) -> list[list[str]]:
        _, command = self.generate_commands(
            env={"VLLM_CONFIG": str(config_path), "ENABLE_MOONCAKE": str(int(enable_mooncake))}
        )
        # Execute model setup without installing packages or starting services.
        setup = command.split("# Increase the number of file descriptors", 1)[0]
        inspect = r"""
printf '%s\0' "${VLLM_COMMON_ARGS[@]}" ''
printf '%s\0' "${VLLM_PREFILL_ARGS[@]}" ''
printf '%s\0' "${VLLM_DECODE_ARGS[@]}" ''
"""
        # Real jobs pass ROUTER_NODE only to evaluation workers, not serving workers.
        status, stdout, stderr = self.run_shell(setup + inspect, env={"SLURM_PROCID": str(rank), "ROUTER_NODE": ""})
        self.assertEqual(status, 0, stderr)
        return [args.split("\0") for args in stdout.removesuffix("\0\0").split("\0\0")]

    def test_mooncake_preserves_every_model_connector_and_other_arguments(self) -> None:
        """The store toggle starts services; recipes own their connector settings."""
        recipes = sorted((SCRIPT.parent / "vllm_configs").glob("*.sh"))
        self.assertTrue(recipes)
        for config_path in recipes:
            for rank in (0, 4):
                with self.subTest(model=config_path.name, rank=rank):
                    original = self.model_arguments(config_path=config_path, enable_mooncake=False, rank=rank)
                    with_store = self.model_arguments(config_path=config_path, enable_mooncake=True, rank=rank)
                    self.assertEqual(len(original), 3)
                    self.assertTrue(original[0])
                    for role_args in original[1:]:
                        self.assertIn("--kv-transfer-config", role_args)
                    self.assertEqual(with_store, original)

    def test_mooncake_handles_common_equals_form_and_existing_multiconnector(self) -> None:
        """Preserve an existing store's settings and support transfer configs in common args."""
        nixl = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        multi = {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "connectors": [
                    nixl,
                    {
                        "kv_connector": "MooncakeStoreConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {"load_async": False, "save_decode_cache": True},
                    },
                ],
            },
        }
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "model config.sh"
            config_path.write_text(
                f"VLLM_COMMON_ARGS=({shlex.quote('--kv-transfer-config=' + json.dumps(nixl))})\n"
                f"VLLM_PREFILL_ARGS=(--kv-transfer-config {shlex.quote(json.dumps(multi))})\n"
                "VLLM_DECODE_ARGS=(--unrelated 'value with spaces')\n"
            )
            for rank in (0, 4):
                with self.subTest(rank=rank):
                    common, prefill, decode = self.model_arguments(
                        config_path=config_path, enable_mooncake=True, rank=rank
                    )
                    self.assertEqual(common, ["--kv-transfer-config=" + json.dumps(nixl)])
                    self.assertEqual(prefill, ["--kv-transfer-config", json.dumps(multi)])
                    self.assertEqual(decode, ["--unrelated", "value with spaces"])

    def serving_arguments(
        self, command: str, *, rank: int, coupled_head: bool = False, env: dict[str, str] | None = None
    ) -> tuple[str, str, list[str], list[str]]:
        # Record argv separately for vLLM and the router; marker files synchronize startup.
        stubs = r"""
VLLM_COMMON_ARGS=(--common-test 'value with spaces')
VLLM_PREFILL_ARGS=(--prefill-test producer)
VLLM_DECODE_ARGS=(--decode-test consumer)
vllm() {
    printf '%s\0' "${VLLM_NIXL_SIDE_CHANNEL_HOST:-}" "${VLLM_NIXL_SIDE_CHANNEL_PORT:-}" "$@"
    touch "$TEST_STATE_DIR/service-ready"
    if [[ "$TEST_COUPLED_HEAD" == 1 ]]; then
        while true; do "$TEST_SLEEP" 0.01; done
    fi
    if (( SLURM_PROCID == 0 )); then
        while [[ ! -f "$TEST_STATE_DIR/router-ready" ]]; do "$TEST_SLEEP" 0.01; done
    fi
}
vllm-router() {
    printf '%s\0' "$@" >&2
    touch "$TEST_STATE_DIR/router-ready"
    if [[ "$TEST_COUPLED_HEAD" == 1 ]]; then
        while [[ ! -f "$TEST_STATE_DIR/service-ready" ]]; do "$TEST_SLEEP" 0.01; done
        kill -TERM "$$"
    else
        # Independent mode checks router liveness before starting vLLM.
        while true; do "$TEST_SLEEP" 0.01; done
    fi
}
sleep() { "$TEST_SLEEP" 0.01; }
hostname() { printf 'node%s\n' "$SLURM_PROCID"; }
"""
        with TemporaryDirectory(prefix="gym-serving-args-") as state_dir:
            status, stdout, stderr = self.run_shell(
                stubs + command,
                env=(env or {})
                | {
                    "SLURM_PROCID": str(rank),
                    "TEST_STATE_DIR": state_dir,
                    "TEST_COUPLED_HEAD": str(int(coupled_head)),
                },
            )
        self.assertEqual(status, 143 if coupled_head else 0, stderr)
        host, nixl_port, *vllm_args = stdout.rstrip("\0").split("\0")
        router_args = stderr.rstrip("\0").split("\0") if stderr else []
        return host, nixl_port, vllm_args, router_args

    def assert_router_arguments(self, args: list[str], prefill_urls: list[str], decode_urls: list[str]) -> None:
        expected = {
            "--prefill-policy": "cache_aware",
            "--decode-policy": "cache_aware",
            "--host": "node0",
            "--port": "8000",
            "--intra-node-data-parallel-size": "1",
            "--request-timeout-secs": "86400",
            "--log-level": "error",
            "--prometheus-host": "0.0.0.0",
            "--prometheus-port": "29000",
            "--worker-startup-timeout-secs": "1200",
        }
        self.assertEqual(args.count("--vllm-pd-disaggregation"), 1)
        args = [arg for arg in args if arg != "--vllm-pd-disaggregation"]
        # URL options may precede or follow the common options; retain their tier order.
        for flag, urls in (("--prefill", prefill_urls), ("--decode", decode_urls)):
            actual_urls = []
            remaining = []
            i = 0
            while i < len(args):
                if args[i] == flag:
                    actual_urls.append(args[i + 1])
                    i += 2
                else:
                    remaining.append(args[i])
                    i += 1
            self.assertEqual(actual_urls, urls)
            args = remaining
        self.assertEqual(len(args), 2 * len(expected))
        self.assertEqual(dict(zip(args[::2], args[1::2], strict=True)), expected)

    def test_independent_mode_preserves_per_node_engines(self) -> None:
        """Independent mode runs one engine per node and routes requests to each tier."""
        for mode in (None, "independent"):
            for prefill_count, decode_count in ((1, 1), (1, 4), (4, 4)):
                env = {"NUM_PREFILL_NODES": str(prefill_count), "NUM_DECODE_NODES": str(decode_count)}
                if mode is not None:
                    env["VLLM_PD_DEPLOYMENT_MODE"] = mode
                _, command = self.generate_commands(env=env)
                for rank in range(prefill_count + decode_count):
                    with self.subTest(mode=mode, prefill=prefill_count, decode=decode_count, rank=rank):
                        host, nixl_port, args, router = self.serving_arguments(command, rank=rank, env=env)
                        is_prefill = rank < prefill_count
                        self.assertEqual(host, f"node{rank}")
                        self.assertEqual(nixl_port, "5600" if is_prefill else "5700")
                        self.assertEqual(
                            args,
                            [
                                "serve",
                                "/test/model",
                                "--served-model-name",
                                "/test/model",
                                "--common-test",
                                "value with spaces",
                                "--prefill-test" if is_prefill else "--decode-test",
                                "producer" if is_prefill else "consumer",
                                "--host",
                                f"node{rank}",
                                "--port",
                                "8001",
                            ],
                        )
                        if rank == 0:
                            self.assert_router_arguments(
                                router,
                                [f"http://node{i}:8001" for i in range(prefill_count)],
                                [f"http://node{i}:8001" for i in range(prefill_count, prefill_count + decode_count)],
                            )
                        else:
                            self.assertEqual(router, [])

    def test_aggregated_submission_preserves_node_count(self) -> None:
        """Aggregated allocation uses NUM_NODES, including its one-node default."""
        for count in (None, 3):
            for eval_args in ((), ("--config", "benchmark.yaml")):
                with self.subTest(count=count, evaluation=bool(eval_args)):
                    env = {"VLLM_MODE": "aggregated", "NUM_PREFILL_NODES": "", "NUM_DECODE_NODES": ""}
                    if count is not None:
                        env["NUM_NODES"] = str(count)
                    _, _, batch, calls = self.capture_submission(*eval_args, env=env)
                    self.assertEqual(len(calls), 2 if eval_args else 1)
                    self.assertIn(f"--nodes={count or 1}", calls[0])
                    self.assertIn(f"--segment={count or 1}", calls[0])
                    self.assertIn(f"--ntasks={count or 1}", batch)

    def test_aggregated_coupled_combination_is_rejected(self) -> None:
        """Reject unsupported multi-node aggregated execution before submitting jobs."""
        status, stdout, stderr = self.run_shell(
            'sbatch() { printf "unexpected-submission\\n"; }; source "$@"',
            str(SCRIPT),
            env={"VLLM_MODE": "aggregated", "VLLM_PD_DEPLOYMENT_MODE": "coupled"},
        )
        self.assertEqual(status, 1)
        self.assertNotIn("unexpected-submission", stdout)
        self.assertIn("VLLM_MODE=aggregated does not support VLLM_PD_DEPLOYMENT_MODE=coupled", stderr)

    def run_tp1_services(
        self,
        *,
        mode: str,
        rank: int = 0,
        exit_role: str = "engine-GPU-b",
        exit_status: int = 7,
        visible_gpus: str = "GPU-d,GPU-b,GPU-a,GPU-c",
        shutdown_signal: str = "",
        profile: bool = False,
    ) -> tuple[int, str, str, dict[str, list[str]]]:
        """Execute generated commands with mock services, recording argv and GPU assignments."""
        with TemporaryDirectory(prefix="gym-tp1-") as directory:
            root = Path(directory)
            config = root / "model.sh"
            config.write_text(
                "VLLM_COMMON_ARGS=(--tensor-parallel-size=4 --pipeline-parallel-size 2 "
                "--data-parallel-size=8 --data-parallel-size-local 4 --api-server-count=2 "
                "--common-test 'value with spaces')\n"
                'VLLM_PREFILL_ARGS=(--prefill-test producer --kv-transfer-config \'{"kv_role":"kv_producer"}\')\n'
                'VLLM_DECODE_ARGS=(--decode-test consumer \'--kv-transfer-config={"kv_role":"kv_consumer"}\')\n'
            )
            env = {
                "VLLM_MODE": mode,
                "VLLM_ENGINES_PER_NODE": "4",
                "NUM_NODES": "1" if mode == "aggregated" else "2",
                "NUM_PREFILL_NODES": "1",
                "NUM_DECODE_NODES": "1",
                "ALL_NODES": "node0" if mode == "aggregated" else "node0 node1",
                "SLURM_PROCID": str(rank),
                "VLLM_CONFIG": str(config),
                "CUDA_VISIBLE_DEVICES": visible_gpus,
                "TEST_STATE_DIR": directory,
                "TEST_EXIT_ROLE": exit_role,
                "TEST_EXIT_STATUS": str(exit_status),
                "TEST_SHUTDOWN_SIGNAL": shutdown_signal,
                "PROFILE_PREFILL": str(int(profile)),
                "PREFILL_PROFILE_DELAY_SECONDS": "0",
                "PREFILL_PROFILE_DIR": str(root / "profiles"),
            }
            _, command, _, submissions = self.capture_submission(env=env)
            self.assertIn(f"--nodes={env['NUM_NODES']}", submissions[0])
            stubs = r"""
run_service() {
    local role=$1
    trap 'printf "%s-stopped\n" "$role"; exit 0' TERM
    touch "$TEST_STATE_DIR/$role-ready"
    if [[ "$role" == "$TEST_EXIT_ROLE" ]]; then
        for gpu in GPU-d GPU-b GPU-a GPU-c; do
            while [[ ! -f "$TEST_STATE_DIR/engine-$gpu-ready" ]]; do "$TEST_SLEEP" 0.01; done
        done
        if [[ -z "$TEST_SHUTDOWN_SIGNAL" ]]; then
            return "$TEST_EXIT_STATUS"
        fi
        kill -s "$TEST_SHUTDOWN_SIGNAL" "$$"
    fi
    while true; do "$TEST_SLEEP" 0.01; done
}
vllm() {
    printf '%s\0' "$CUDA_VISIBLE_DEVICES" "$VLLM_NIXL_SIDE_CHANNEL_HOST" \
        "$VLLM_NIXL_SIDE_CHANNEL_PORT" "$@" > "$TEST_STATE_DIR/engine-$CUDA_VISIBLE_DEVICES.args"
    run_service "engine-$CUDA_VISIBLE_DEVICES"
}
vllm-router() {
    printf '%s\0' "$@" > "$TEST_STATE_DIR/router.args"
    run_service router
}
nsys() {
    printf '%s\0' "$@" > "$TEST_STATE_DIR/profile-$CUDA_VISIBLE_DEVICES.args"
    while [[ ! -f "$TEST_STATE_DIR/profile-$CUDA_VISIBLE_DEVICES-started" ]]; do "$TEST_SLEEP" 0.01; done
    while [[ "$1" != vllm ]]; do shift; done
    "$@"
}
curl() {
    printf '%s\0' "${*: -1}" >> "$TEST_STATE_DIR/trigger-$CUDA_VISIBLE_DEVICES.args"
    if [[ "${*: -1}" == */start_profile ]]; then touch "$TEST_STATE_DIR/profile-$CUDA_VISIBLE_DEVICES-started"; fi
}
hostname() { printf 'node%s\n' "$SLURM_PROCID"; }
sleep() { "$TEST_SLEEP" 0.01; }
"""
            status, stdout, stderr = self.run_shell(stubs + command, env=env)
            recorded = {p.stem: p.read_text().rstrip("\0").split("\0") for p in root.glob("*.args")}
            return status, stdout, stderr, recorded

    def test_tp1_engines_have_distinct_gpus_ports_and_router_endpoints(self) -> None:
        for mode in ("aggregated", "pd"):
            for rank in (0,) if mode == "aggregated" else (0, 1):
                with self.subTest(mode=mode, rank=rank):
                    status, stdout, stderr, recorded = self.run_tp1_services(mode=mode, rank=rank)
                    self.assertEqual(status, 7, stderr)
                    for index, gpu in enumerate(("GPU-d", "GPU-b", "GPU-a", "GPU-c")):
                        device, host, nixl_port, *args = recorded[f"engine-{gpu}"]
                        self.assertEqual((device, host), (gpu, f"node{rank}"))
                        self.assertEqual(nixl_port, str((5700 if mode == "pd" and rank == 1 else 5600) + index))
                        self.assertEqual(args[args.index("--port") + 1], str(8001 + index))
                        for flag in (
                            "--tensor-parallel-size",
                            "--pipeline-parallel-size",
                            "--data-parallel-size",
                            "--data-parallel-size-local",
                            "--api-server-count",
                        ):
                            self.assertEqual(args.count(flag), 1)
                            self.assertEqual(args[args.index(flag) + 1], "1")
                            self.assertFalse(any(arg.startswith(flag + "=") for arg in args))
                        self.assertEqual(args[args.index("--common-test") + 1], "value with spaces")
                        if mode == "aggregated":
                            self.assertFalse(any(arg.startswith("--kv-transfer-config") for arg in args))
                        elif rank == 0:
                            self.assertIn("--prefill-test", args)
                            self.assertEqual(
                                json.loads(args[args.index("--kv-transfer-config") + 1])["kv_role"], "kv_producer"
                            )
                        else:
                            self.assertIn("--decode-test", args)
                            self.assertIn('--kv-transfer-config={"kv_role":"kv_consumer"}', args)
                        if gpu != "GPU-b":
                            self.assertIn(f"engine-{gpu}-stopped", stdout)
                    if rank == 0:
                        router = recorded["router"]
                        if mode == "pd":
                            self.assert_router_arguments(
                                router,
                                [f"http://node0:{port}" for port in range(8001, 8005)],
                                [f"http://node1:{port}" for port in range(8001, 8005)],
                            )
                        else:
                            self.assertEqual(
                                router[router.index("--worker-urls") + 1 :],
                                [f"http://node0:{port}" for port in range(8001, 8005)],
                            )
                        self.assertIn("router-stopped", stdout)
                    else:
                        self.assertNotIn("router", recorded)

    def test_tp1_engine_and_router_exits_stop_all_peers(self) -> None:
        for role, exit_status in (("engine-GPU-b", 0), ("router", 9), ("router", 0)):
            with self.subTest(role=role, exit_status=exit_status):
                status, stdout, stderr, recorded = self.run_tp1_services(
                    mode="pd", exit_role=role, exit_status=exit_status
                )
                self.assertEqual(status, exit_status or 1, stderr)
                for service in recorded:
                    if service != role:
                        self.assertIn(f"{service}-stopped", stdout)

    def test_tp1_shutdown_stops_all_services(self) -> None:
        status, stdout, stderr, recorded = self.run_tp1_services(mode="pd", shutdown_signal="TERM")
        self.assertEqual(status, 143, stderr)
        for service in recorded:
            self.assertIn(f"{service}-stopped", stdout)

    def test_tp1_requires_four_visible_gpus(self) -> None:
        for visible in ("0,1", ""):
            with self.subTest(visible=visible):
                status, _, stderr, recorded = self.run_tp1_services(mode="pd", rank=1, visible_gpus=visible)
                self.assertEqual(status, 1, stderr)
                self.assertIn("require at least four visible GPUs", stderr)
                self.assertEqual(recorded, {})

    def test_tp1_profiling_uses_each_prefill_port_and_cleans_up(self) -> None:
        for mode, rank in (("pd", 0), ("pd", 1), ("aggregated", 0)):
            with self.subTest(mode=mode, rank=rank):
                status, stdout, stderr, recorded = self.run_tp1_services(mode=mode, rank=rank, profile=True)
                self.assertEqual(status, 7, stderr)
                for index, gpu in enumerate(("GPU-d", "GPU-b", "GPU-a", "GPU-c")):
                    should_profile = mode == "pd" and rank == 0
                    self.assertEqual(f"profile-{gpu}" in recorded, should_profile)
                    self.assertEqual(f"trigger-{gpu}" in recorded, should_profile)
                    if should_profile:
                        port = 8001 + index
                        self.assertEqual(
                            recorded[f"trigger-{gpu}"],
                            [f"http://node0:{port}/health", f"http://node0:{port}/start_profile"],
                        )
                        args = recorded[f"profile-{gpu}"]
                        output = args[args.index("--output") + 1]
                        self.assertTrue(output.endswith(f"/prefill-12345-node0-port{port}-%p"))
                    if gpu != "GPU-b":
                        self.assertIn(f"engine-{gpu}-stopped", stdout)

    def test_tp1_profile_shutdown_stops_servers(self) -> None:
        status, stdout, stderr, _ = self.run_tp1_services(mode="pd", profile=True, shutdown_signal="TERM")
        self.assertEqual(status, 143, stderr)
        for gpu in ("GPU-d", "GPU-b", "GPU-a", "GPU-c"):
            self.assertIn(f"engine-{gpu}-stopped", stdout)
        self.assertIn("router-stopped", stdout)

    def test_tp1_metrics_include_every_engine(self) -> None:
        for mode in ("aggregated", "pd"):
            with self.subTest(mode=mode):
                env = {"VLLM_MODE": mode, "VLLM_ENGINES_PER_NODE": "4", "ALL_NODES": "node0 node1"}
                command, _ = self.generate_commands(env=env)
                start = 'read -r -a nodes <<< "$ALL_NODES"'
                setup = start + command.split(start, 1)[1].split("gym_config_args+=(--config", 1)[0]
                path = Path(self.workdir) / "metrics.yaml"
                status, _, stderr = self.run_shell(setup, env=env | {"inference_metrics_config": str(path)})
                self.assertEqual(status, 0, stderr)
                self.assertEqual(
                    yaml.safe_load(path.read_text())["inference_metrics"]["endpoints"],
                    {
                        f"node{node * 4 + engine}": f"http://node{node}:{8001 + engine}/metrics"
                        for node in range(2)
                        for engine in range(4)
                    },
                )

    def test_tp1_incompatible_controls_fail_before_submission(self) -> None:
        for extra in ({"VLLM_PD_DEPLOYMENT_MODE": "coupled"}, {"ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE": "4"}):
            with self.subTest(extra=extra):
                status, stdout, stderr = self.run_shell(
                    'sbatch() { printf "unexpected-submission\\n"; }; source "$@"',
                    str(SCRIPT),
                    env={"VLLM_ENGINES_PER_NODE": "4"} | extra,
                )
                self.assertEqual(status, 1, stderr)
                self.assertNotIn("unexpected-submission", stdout)

    def test_generated_scripts_have_valid_syntax(self) -> None:
        """Parse the generated scripts too: outer bash -n cannot validate heredoc contents."""
        for env in (
            {},
            {"VLLM_PD_DEPLOYMENT_MODE": "coupled"},
            {"VLLM_MODE": "aggregated"},
            {"VLLM_ENGINES_PER_NODE": "4"},
            {"VLLM_MODE": "aggregated", "VLLM_ENGINES_PER_NODE": "4"},
        ):
            with self.subTest(env=env):
                evaluation, serving, batch, _ = self.capture_submission("--config", "benchmark.yaml", env=env)
                for name, command in (("evaluation", evaluation), ("serving", serving), ("batch", batch)):
                    with self.subTest(script=name):
                        if name == "batch":
                            self.require_batch_bash()
                        result = subprocess.run([BASH, "-n"], input=command, text=True, capture_output=True, timeout=5)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ultra_runtime_settings_preserve_launcher_network_defaults(self) -> None:
        """Ultra sets its model runner and state layout while inheriting the network defaults."""
        expected = {
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "UCX_TLS": "rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc",
            "UCX_RNDV_SCHEME": "get_zcopy",
            "UCX_RNDV_THRESH": "0",
            "UCX_NET_DEVICES": "mlx5_0,mlx5_1,mlx5_3,mlx5_4",
            "UCX_IB_ADDR_TYPE": "unset",
            "NCCL_CUMEM_ENABLE": "1",
            "NCCL_MNNVL_ENABLE": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
        recipe = str(SCRIPT.parent / "vllm_configs/nemotron_3_ultra.sh")
        _, command = self.generate_commands(env={"VLLM_CONFIG": recipe})
        setup = command.split("this_node_hostname=", 1)[0]
        inspect = '\nfor name in "$@"; do printf "%s\\0" "${!name-unset}"; done\n'
        status, stdout, stderr = self.run_shell(setup + inspect, *expected)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(dict(zip(expected, stdout.removesuffix("\0").split("\0"), strict=True)), expected)

    def test_prefill_profile_wraps_only_prefill_workers(self) -> None:
        recipe = str(SCRIPT.parent / "vllm_configs/nemotron_3.5_super_mtp_mooncake.sh")
        # Record the actual Nsight invocation, then pass through to the serving stub.
        profiler_stub = r"""
nsys() {
    printf '%s\0' "$VLLM_WORKER_MULTIPROC_METHOD" "$@" > "$TEST_NSYS_ARGS"
    while [[ "$1" != vllm ]]; do shift; done
    "$@"
}
"""
        capture = Path(self.workdir) / "nsys_args"
        profile_dir = Path(self.workdir) / "profile with spaces"
        for mode in ("independent", "coupled", "aggregated"):
            for enabled in ("0", "1"):
                env = {
                    "VLLM_CONFIG": recipe,
                    "VLLM_MODE": "aggregated" if mode == "aggregated" else "pd",
                    "VLLM_PD_DEPLOYMENT_MODE": "coupled" if mode == "coupled" else "independent",
                    "PROFILE_PREFILL": enabled,
                    "PREFILL_PROFILE_STEPS": "7",
                    "PREFILL_PROFILE_DELAY_SECONDS": "manual",
                    "PREFILL_PROFILE_DIR": str(profile_dir),
                    "TEST_NSYS_ARGS": str(capture),
                }
                _, command = self.generate_commands(env=env)
                for rank in (0, 1, 4, 5):
                    with self.subTest(mode=mode, enabled=enabled, rank=rank):
                        capture.unlink(missing_ok=True)
                        _, _, args, _ = self.serving_arguments(
                            profiler_stub + command,
                            rank=rank,
                            coupled_head=mode == "coupled" and rank == 0,
                            env=env,
                        )
                        should_profile = enabled == "1" and rank < 4 and mode != "aggregated"
                        self.assertEqual(capture.exists(), should_profile)
                        self.assertEqual("--profiler-config" in args, should_profile)
                        if should_profile:
                            profiler_args = capture.read_text().removesuffix("\0").split("\0")
                            self.assertEqual(profiler_args[:2], ["spawn", "profile"])
                            self.assertIn("--capture-range=cudaProfilerApi", profiler_args)
                            self.assertIn("--capture-range-end=stop", profiler_args)
                            self.assertIn("--cuda-graph-trace=node", profiler_args)
                            for flag in (
                                "--trace=cuda,nvtx,osrt,nccl,cublas,cublas-verbose",
                                "--nccl-trace=all",
                                "--pytorch=functions-trace-shapes",
                                "--sample=process-tree",
                                "--cudabacktrace=all",
                                "--python-backtrace=cuda",
                                "--python-sampling=true",
                            ):
                                self.assertIn(flag, profiler_args)
                            output = profiler_args[profiler_args.index("--output") + 1]
                            self.assertEqual(output, str(profile_dir / f"prefill-12345-node{rank}-port8001-%p"))
                            self.assertTrue(profile_dir.is_dir())
                            config = json.loads(args[args.index("--profiler-config") + 1])
                            self.assertEqual(config, {"profiler": "cuda", "max_iterations": 7})

    def test_prefill_profile_automatically_waits_for_health_and_warmup(self) -> None:
        _, command = self.generate_commands()
        setup = command.split('\nmooncake_pid=""', 1)[0]
        stubs = r"""
hostname() { echo node1; }
curl() {
    if [[ "${*: -1}" == http://node1:8001/health ]]; then
        echo health
        if [[ ! -f ready ]]; then touch ready; return 22; fi
    else
        [[ "${*: -3}" == '-X POST http://node1:8001/start_profile' ]] || return 99
        echo post
        return "${TEST_POST_STATUS:-0}"
    fi >> events
}
sleep() { echo sleep >> events; SECONDS=$((SECONDS + 1)); }
"""
        for post_status in ("0", "22"):
            with self.subTest(post_status=post_status):
                for name in ("ready", "events"):
                    (Path(self.workdir) / name).unlink(missing_ok=True)
                status, _, stderr = self.run_shell(
                    stubs + setup + "\nprofile_delay=2\nprofile_port=8001\nsteps=20\ntrigger_prefill_profile\n",
                    env={"TEST_POST_STATUS": post_status},
                )
                self.assertEqual(status, 0 if post_status == "0" else 1, stderr)
                self.assertEqual(
                    (Path(self.workdir) / "events").read_text().splitlines(),
                    ["health", "sleep", "health", "sleep", "sleep", "post"],
                )
                self.assertIn("capture started" if post_status == "0" else "Could not start", stderr)

    def test_prefill_profile_autotrigger_only_runs_on_api_workers(self) -> None:
        _, command = self.generate_commands()
        setup = command.split('\nmooncake_pid=""', 1)[0]
        stubs = r"""
nsys() {
    if [[ " $* " == *' --headless '* || "$PREFILL_PROFILE_DELAY_SECONDS" == manual ]]; then return; fi
    while [[ ! -f posted ]]; do "$TEST_SLEEP" 0.01; done
}
curl() {
    echo "${*: -1}" >> urls
    if [[ "${*: -1}" == */start_profile ]]; then touch posted; fi
}
"""
        for delay, headless in (("0", False), ("0", True), ("manual", False)):
            with self.subTest(delay=delay, headless=headless):
                for name in ("posted", "urls"):
                    (Path(self.workdir) / name).unlink(missing_ok=True)
                status, _, stderr = self.run_shell(
                    stubs + setup + '\nserve_prefill 8001 /test/model "$@"\n',
                    *(["--headless"] if headless else []),
                    env={"PROFILE_PREFILL": "1", "PREFILL_PROFILE_DELAY_SECONDS": delay},
                )
                self.assertEqual(status, 0, stderr)
                urls = Path(self.workdir) / "urls"
                self.assertEqual(urls.exists(), delay == "0" and not headless)
                if urls.exists():
                    self.assertEqual(
                        [url.rsplit("/", 1)[1] for url in urls.read_text().splitlines()], ["health", "start_profile"]
                    )

    def test_prefill_profile_rejects_invalid_steps_or_missing_nsys(self) -> None:
        _, command = self.generate_commands()
        setup = command.split('\nmooncake_pid=""', 1)[0]
        # Hide nsys regardless of what is installed on the test host.
        stubs = r"""
command() { if [[ "$*" == '-v nsys' ]]; then return 1; else builtin command "$@"; fi; }
id() { echo 1000; }
vllm() { echo 'unexpected server launch'; }
"""
        for steps, expected in (
            ("0", "must be a positive integer"),
            ("-1", "must be a positive integer"),
            ("1.5", "must be a positive integer"),
            ("abc", "must be a positive integer"),
            ("20", "Installing Nsight Systems requires root"),
        ):
            with self.subTest(steps=steps):
                status, stdout, stderr = self.run_shell(
                    stubs + setup + "\nserve_prefill 8001 /test/model\n",
                    env={"PROFILE_PREFILL": "1", "PREFILL_PROFILE_STEPS": steps},
                )
                self.assertNotEqual(status, 0)
                self.assertNotIn("unexpected server launch", stdout)
                self.assertIn(expected, stderr)

    def test_prefill_profile_installs_nsys_for_container_architecture(self) -> None:
        _, command = self.generate_commands()
        setup = command.split('\nmooncake_pid=""', 1)[0]
        # Stub every system mutation: installation must never touch the test host.
        stubs = r"""
command() {
    if [[ "$*" == '-v nsys' ]]; then [[ -f installed ]]; else builtin command "$@"; fi
}
source() {
    if [[ "$1" == /etc/os-release ]]; then ID=ubuntu; VERSION_ID=24.04; else builtin source "$@"; fi
}
id() { echo 0; }
dpkg() { echo "$TEST_ARCH"; }
apt-get() {
    echo "$*" >> apt_calls
    if [[ "$TEST_FAILURE" == apt ]]; then return 42; fi
    if [[ " $* " == *' nsight-systems-cli '* && "$TEST_FAILURE" != missing ]]; then touch installed; fi
}
mkdir() { [[ "$*" == '-p /usr/share/keyrings' ]]; }
wget() { echo public-key; }
gpg() { cat > key; [[ "$TEST_FAILURE" != key ]]; }
tee() { cat > repository; }
nsys() { echo "$*" >> nsys_calls; }
"""
        for arch, failure in (
            ("arm64", "none"),
            ("amd64", "none"),
            ("arm64", "apt"),
            ("arm64", "key"),
            ("arm64", "missing"),
        ):
            with self.subTest(arch=arch, failure=failure):
                for name in ("installed", "apt_calls", "repository", "key", "nsys_calls"):
                    (Path(self.workdir) / name).unlink(missing_ok=True)
                status, _, stderr = self.run_shell(
                    stubs + setup + "\nensure_nsys\nensure_nsys\n",
                    env={"TEST_ARCH": arch, "TEST_FAILURE": failure},
                )
                apt_calls = (Path(self.workdir) / "apt_calls").read_text().splitlines()
                if failure == "none":
                    self.assertEqual(status, 0, stderr)
                    self.assertEqual(
                        apt_calls,
                        [
                            "update",
                            "install -y --no-install-recommends wget gnupg ca-certificates",
                            "update",
                            "install -y --no-install-recommends nsight-systems-cli curl",
                        ],
                    )
                    self.assertEqual(
                        (Path(self.workdir) / "repository").read_text(),
                        "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] "
                        f"https://developer.download.nvidia.com/devtools/repos/ubuntu2404/{arch}/ /\n",
                    )
                    self.assertEqual((Path(self.workdir) / "nsys_calls").read_text(), "--version\n")
                else:
                    self.assertNotEqual(status, 0)
                    self.assertFalse((Path(self.workdir) / "nsys_calls").exists())
                    self.assertIn("installation completed" if failure == "missing" else "installation failed", stderr)
                    self.assertEqual(len(apt_calls), {"apt": 1, "key": 2, "missing": 4}[failure])

    def test_prefill_profile_cancels_pending_trigger_when_server_exits(self) -> None:
        _, command = self.generate_commands()
        setup = command.split('\nmooncake_pid=""', 1)[0]
        stubs = r"""
nsys() {
    while [[ ! -f trigger_started ]]; do "$TEST_SLEEP" 0.01; done
    return 17
}
trigger_prefill_profile() {
    trap 'touch trigger_stopped; exit 0' TERM
    touch trigger_started
    while true; do "$TEST_SLEEP" 0.01; done
}
"""
        status, _, stderr = self.run_shell(
            setup + stubs + "\nserve_prefill 8001 /test/model\n",
            env={"PROFILE_PREFILL": "1"},
        )
        self.assertEqual(status, 17, stderr)
        self.assertTrue((Path(self.workdir) / "trigger_stopped").exists())

    def test_api_server_count_is_removed_only_from_headless_ranks(self):
        with TemporaryDirectory(prefix="gym-headless-args-") as directory:
            config = Path(directory) / "config with spaces.sh"
            config.write_text(
                "VLLM_COMMON_ARGS=(--api-server-count 3 --common-test 'value with spaces')\n"
                "VLLM_PREFILL_ARGS=(--prefill-test producer --api-server-count=2)\n"
                "VLLM_DECODE_ARGS=(--api-server-count 5 --decode-test consumer)\n"
            )
            for mode, ranks in (("coupled", (1, 3, 4, 5, 7)), ("independent", (1,))):
                env = {"VLLM_PD_DEPLOYMENT_MODE": mode, "VLLM_CONFIG": str(config)}
                _, command = self.generate_commands(env=env)
                for rank in ranks:
                    with self.subTest(mode=mode, rank=rank):
                        _, _, args, _ = self.serving_arguments(command, rank=rank, env=env)
                        self.assertIn("value with spaces", args)
                        self.assertIn("producer" if rank < 4 else "consumer", args)
                        if mode == "coupled" and rank != 4:
                            self.assertIn("--headless", args)
                            self.assertFalse(any(arg.startswith("--api-server-count") for arg in args))
                            self.assertEqual(
                                args[: args.index("--headless")],
                                [
                                    "serve",
                                    "/test/model",
                                    "--served-model-name",
                                    "/test/model",
                                    "--common-test",
                                    "value with spaces",
                                    "--prefill-test" if rank < 4 else "--decode-test",
                                    "producer" if rank < 4 else "consumer",
                                ],
                            )
                        else:
                            self.assertNotIn("--headless", args)
                            self.assertIn("--api-server-count", args)
                            self.assertIn("3", args)
                            if mode == "coupled":
                                self.assertEqual(args[-2:], ["--api-server-count", "1"])
                            else:
                                self.assertIn("--api-server-count=2", args)

    def test_coupled_nodes_use_correct_tier_roles_and_ranks(self) -> None:
        """Assign coupled tier roles, ranks, and ports while leaving router balancing thresholds at defaults."""
        for prefill_count, decode_count in ((1, 1), (1, 4), (2, 3), (4, 4)):
            env = {
                "VLLM_PD_DEPLOYMENT_MODE": "coupled",
                "NUM_PREFILL_NODES": str(prefill_count),
                "NUM_DECODE_NODES": str(decode_count),
                "MODEL_NAME": "served-model-alias",
            }
            _, command = self.generate_commands(env=env)
            for rank in range(prefill_count + decode_count):
                with self.subTest(prefill=prefill_count, decode=decode_count, rank=rank):
                    host, nixl_port, args, router = self.serving_arguments(
                        command, rank=rank, coupled_head=rank == 0, env=env
                    )
                    is_prefill = rank < prefill_count
                    local_rank = rank if is_prefill else rank - prefill_count
                    head = "node0" if is_prefill else f"node{prefill_count}"
                    self.assertEqual(host, f"node{rank}")
                    self.assertEqual(nixl_port, "5600" if is_prefill else "5700")
                    expected = [
                        "serve",
                        "/test/model",
                        "--served-model-name",
                        "served-model-alias",
                        "--common-test",
                        "value with spaces",
                        "--prefill-test" if is_prefill else "--decode-test",
                        "producer" if is_prefill else "consumer",
                    ]
                    if local_rank == 0:
                        expected += ["--host", host, "--port", "8001"]
                    else:
                        expected += ["--headless"]
                    expected += ["--data-parallel-size", str(prefill_count if is_prefill else decode_count)]
                    if local_rank != 0:
                        expected += ["--data-parallel-start-rank", str(local_rank)]
                    expected += [
                        "--data-parallel-address",
                        head,
                        "--data-parallel-rpc-port",
                        "13345" if is_prefill else "13346",
                    ]
                    if local_rank == 0:
                        expected += ["--api-server-count", "1"]
                    self.assertEqual(args, expected)
                    if rank == 0:
                        self.assert_router_arguments(
                            router, ["http://node0:8001"], [f"http://node{prefill_count}:8001"]
                        )
                    else:
                        self.assertEqual(router, [])

    def test_router_overrides_apply_to_both_deployment_modes(self):
        """Apply router policy and local data-parallel overrides in both deployment modes."""
        for mode in ("independent", "coupled"):
            with self.subTest(mode=mode):
                env = {
                    "VLLM_PD_DEPLOYMENT_MODE": mode,
                    "ROUTER_PREFILL_POLICY": "round_robin",
                    "ROUTER_DECODE_POLICY": "random",
                    "ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE": "2",
                }
                _, command = self.generate_commands(env=env)
                _, _, _, router = self.serving_arguments(command, rank=0, coupled_head=mode == "coupled", env=env)
                for flag, value in (
                    ("--prefill-policy", "round_robin"),
                    ("--decode-policy", "random"),
                    ("--intra-node-data-parallel-size", "2"),
                ):
                    self.assertEqual(router.count(flag), 1)
                    self.assertEqual(router[router.index(flag) + 1], value)

    def test_independent_router_startup_failure_stops_launch(self) -> None:
        """Stop independent startup when the router exits during main's five-second startup check."""
        stubs = r"""
VLLM_COMMON_ARGS=(--common-test common)
VLLM_PREFILL_ARGS=(--prefill-test producer)
VLLM_DECODE_ARGS=()
vllm-router() { return "$TEST_ROUTER_STATUS"; }
vllm() { printf 'unexpected-vllm-start\n'; }
hostname() { printf 'node0\n'; }
# Reap the failed router deterministically instead of waiting five real seconds.
sleep() { printf 'startup-delay=%s\n' "$1"; wait "$router_pid" || true; }
"""
        _, command = self.generate_commands()
        for router_status in (0, 7):
            with self.subTest(router_status=router_status):
                status, stdout, stderr = self.run_shell(
                    stubs + command, env={"TEST_ROUTER_STATUS": str(router_status)}
                )
                self.assertEqual(status, 1, stderr)
                self.assertEqual(stdout, "startup-delay=5\n")
                self.assertIn("vllm-router exited during startup", stderr)

    def test_existing_recipes_preserve_sampling_overrides(self):
        """Pass through existing models' sampling parameters without injecting Ultra evaluation defaults."""
        for name in ("inkling_small.sh", "qwen3.5-122b-a10b.sh", "nemotron_3.5_super.sh", "nemotron_3.5_lightning.sh"):
            with self.subTest(recipe=name):
                args = self.eval_arguments(env={"VLLM_CONFIG": str(SCRIPT.parent / "vllm_configs" / name)})
                expected = (
                    []
                    if name == "inkling_small.sh"
                    else [
                        "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0",
                        "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=0.95",
                    ]
                )
                self.assertEqual([arg for arg in args if ".sampling_overrides." in arg], expected)
                self.assertEqual(self.settings(args, "num_samples_in_parallel"), [])
                self.assertEqual(self.settings(args, "resume_from_cache"), [])

    def test_submission_defaults_preserve_time_and_full_allocation_segment(self):
        """Default submissions use a four-hour limit and one segment covering all allocated nodes."""
        for prefill_count, decode_count in ((1, 1), (1, 4), (4, 4)):
            with self.subTest(prefill=prefill_count, decode=decode_count):
                _, _, _, calls = self.capture_submission(
                    "--config",
                    "benchmark.yaml",
                    env={"NUM_PREFILL_NODES": str(prefill_count), "NUM_DECODE_NODES": str(decode_count)},
                )
                self.assertEqual(len(calls), 2)
                self.assertIn(f"--nodes={prefill_count + decode_count}", calls[0])
                self.assertEqual(
                    [arg for arg in calls[0] if arg.startswith("--segment=")],
                    [f"--segment={prefill_count + decode_count}"],
                )
                self.assertIn("--time=04:00:00", calls[0])
                self.assertIn("--ntasks-per-node=1", calls[0])
                self.assertIn("--exclusive", calls[0])
                self.assertIn("--dependency=afterany:12345", calls[1])

    def test_segment_controls_resolve_to_one_submission_argument(self) -> None:
        """Use SEGMENT or the calculated node count without duplicate submission arguments."""
        cases = (
            ({}, "8"),
            ({"NUM_NODES": "99"}, "8"),
            ({"SEGMENT": "2"}, "2"),
            ({"SEGMENT": "4"}, "4"),
            ({"SEGMENT": ""}, "8"),
        )
        for mode in ("independent", "coupled"):
            for eval_args in ((), ("--config", "benchmark.yaml")):
                for segment_env, expected in cases:
                    with self.subTest(mode=mode, evaluation=bool(eval_args), controls=segment_env):
                        _, _, _, calls = self.capture_submission(
                            *eval_args, env={"VLLM_PD_DEPLOYMENT_MODE": mode} | segment_env
                        )
                        self.assertEqual(len(calls), 2 if eval_args else 1)
                        self.assertEqual(
                            [arg for arg in calls[0] if arg.startswith("--segment=")], [f"--segment={expected}"]
                        )
                        if eval_args:
                            self.assertFalse(any(arg.startswith("--segment=") for arg in calls[1]))

    def test_submission_overrides_do_not_change_cleanup_allocation(self) -> None:
        """Apply custom walltime and segment size only to the main job, leaving cleanup CPU-only and short."""
        _, _, _, calls = self.capture_submission(
            "--config", "benchmark.yaml", env={"SBATCH_TIMELIMIT": "7-00:00:00", "SEGMENT": "4"}
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("--time=7-00:00:00", calls[0])
        self.assertEqual([arg for arg in calls[0] if arg.startswith("--segment=")], ["--segment=4"])
        self.assertIn("--time=00:30:00", calls[1])
        self.assertIn("--nodes=1", calls[1])
        self.assertIn("--partition=cpu", calls[1])
        self.assertIn("--qos=cpu-normal", calls[1])
        self.assertIn("--gres=none", calls[1])
        self.assertFalse(any(arg.startswith("--segment=") for arg in calls[1]))

    def test_walltime_override_and_cleanup_isolation(self) -> None:
        """SBATCH_TIMELIMIT overrides the main job's default; cleanup has its own walltime."""
        for overrides, expected in (
            ({}, "04:00:00"),
            ({"SBATCH_TIMELIMIT": "06:00:00"}, "06:00:00"),
            ({"SBATCH_TIMELIMIT": "7-00:00:00"}, "7-00:00:00"),
            ({"SBATCH_TIMELIMIT": ""}, "04:00:00"),
        ):
            for mode in ("pd", "aggregated"):
                with self.subTest(overrides=overrides, mode=mode):
                    _, _, _, calls = self.capture_submission(
                        "--config", "benchmark.yaml", env={"VLLM_MODE": mode} | overrides
                    )
                    self.assertEqual([arg for arg in calls[0] if arg.startswith("--time=")], [f"--time={expected}"])
                    self.assertIn("--time=00:30:00", calls[1])
                    self.assertFalse(any(arg.startswith("--segment=") for arg in calls[1]))

    def test_invalid_launcher_controls_are_rejected_before_submission(self) -> None:
        """Reject invalid serving and deployment modes without submitting any job."""
        stub = 'sbatch() { printf "unexpected-submission\\n"; }; source "$@"'
        cases = {
            "VLLM_PD_DEPLOYMENT_MODE": (("bad", "COUPLED"), 1),
            "VLLM_MODE": (("bad", "PD"), 1),
            "VLLM_ENGINES_PER_NODE": (("0", "2", "bad"), 1),
        }
        for name, (values, expected_status) in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    status, stdout, stderr = self.run_shell(stub, str(SCRIPT), env={name: value})
                    self.assertEqual(status, expected_status, stderr)
                    self.assertNotIn("unexpected-submission", stdout)
                    self.assertIn(name, stderr)

    def test_serving_only_skips_evaluation_and_cleanup_submission(self) -> None:
        """Run only the serving step without eval arguments and preserve its success or failure status."""
        self.require_batch_bash()
        stubs = r"""
scontrol() { printf '%s\n' node0 node1 node2 node3 node4 node5 node6 node7; }
srun() { printf '%s\0' "$@"; printf '\0'; return "$TEST_SERVER_STATUS"; }
"""
        for mode in ("independent", "coupled"):
            for server_status in (0, 7):
                with self.subTest(mode=mode, server_status=server_status):
                    _, _, batch_command, calls = self.capture_submission(
                        env={"VLLM_PD_DEPLOYMENT_MODE": mode, "EXPERIMENT_NAME": "", "EXPORT_TO_CSV": "1"}
                    )
                    self.assertEqual(len(calls), 1)
                    self.assertIn("--job-name=gym-vllm_only-launcher-test", calls[0])
                    status, stdout, stderr = self.run_shell(
                        stubs + batch_command,
                        env={
                            "SLURM_JOB_NODELIST": "test-nodes",
                            "SLURM_SUBMIT_DIR": "/test",
                            "SLURM_CPUS_ON_NODE": "64",
                            "vllm_command": "fake-serving-command",
                            "eval_command": "fake-eval-command",
                            "TEST_SERVER_STATUS": str(server_status),
                        },
                    )
                    self.assertEqual(status, server_status, stderr)
                    steps = stdout.removesuffix("\0\0").split("\0\0")
                    self.assertEqual(len(steps), 1)
                    args = steps[0].split("\0")
                    self.assertIn("--nodes=8", args)
                    self.assertIn("--ntasks=8", args)
                    self.assertIn("--kill-on-bad-exit=1", args)
                    self.assertIn("fake-serving-command", args)
                    self.assertNotIn("--overlap", args)

    def test_default_evaluation_settings_are_unchanged(self):
        """Leave concurrency and resume to Gym and preserve timestamped output naming by default."""
        args = self.eval_arguments()
        self.assertEqual(self.settings(args, "num_samples_in_parallel"), [])
        self.assertEqual(self.settings(args, "resume_from_cache"), [])
        self.assertIn("++output_jsonl_fpath=results/launcher-test/slurm_job_id_12345/date_20260909_120000.jsonl", args)

    def test_explicit_concurrency_arguments_are_preserved(self):
        """Pass explicit concurrency values through to Gym without replacing or duplicating them."""
        for prefix in ("", "+", "++"):
            for value in (16, 32, 512):
                with self.subTest(prefix=prefix, value=value):
                    override = f"{prefix}num_samples_in_parallel={value}"
                    args = self.eval_arguments(override)
                    self.assertEqual(self.settings(args, "num_samples_in_parallel"), [override])

    def test_explicit_resume_arguments_are_preserved(self):
        """Pass explicit resume values through to Gym without changing the supplied output path."""
        for prefix in ("", "+", "++"):
            for value in ("true", "false"):
                with self.subTest(prefix=prefix, value=value):
                    override = prefix + "resume_from_cache=" + value
                    args = self.eval_arguments(override, env={"ROLLOUTS_FPATH": "results/existing/resumable.jsonl"})
                    self.assertEqual(self.settings(args, "resume_from_cache"), [override])
                    self.assertIn("++output_jsonl_fpath=results/existing/resumable.jsonl", args)

    def test_resume_without_fixed_path_uses_restart_timestamp(self):
        """Show that enabling resume alone still selects a new output path when the same job restarts."""
        for timestamp in ("20260909_120000", "20260909_180000"):
            with self.subTest(timestamp=timestamp):
                args = self.eval_arguments("++resume_from_cache=true", env={"TEST_DATE": timestamp})
                self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
                self.assertIn(
                    f"++output_jsonl_fpath=results/launcher-test/slurm_job_id_12345/date_{timestamp}.jsonl", args
                )

    def test_explicit_resume_keeps_output_path_across_restarts(self):
        """Keep the supplied resume path across requeues and new job IDs, with timestamped logs and W&B names."""
        for job_id in ("12345", "12346"):
            for timestamp in ("20260909_120000", "20260909_180000"):
                with self.subTest(job_id=job_id, timestamp=timestamp):
                    args = self.eval_arguments(
                        "++resume_from_cache=true",
                        env={
                            "ROLLOUTS_FPATH": "results/existing/resumable.jsonl",
                            "SLURM_JOB_ID": job_id,
                            "TEST_DATE": timestamp,
                        },
                    )
                    self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
                    self.assertIn("++output_jsonl_fpath=results/existing/resumable.jsonl", args)
                    experiment_name = f"launcher-test/slurm_job_id_{job_id}/date_{timestamp}"
                    self.assertIn(f"+wandb_name={experiment_name}", args)
                    self.assertIn(f"+nemo_gym_log_dir=results/{experiment_name}/logs", args)

    def test_coupled_router_manages_worker_readiness(self) -> None:
        """Start the router with a worker startup deadline even before health probes would succeed."""
        _, command = self.generate_commands(env={"VLLM_PD_DEPLOYMENT_MODE": "coupled"})
        stubs = r"""
VLLM_COMMON_ARGS=(--common-test common)
VLLM_PREFILL_ARGS=(--prefill-test producer)
vllm() {
    touch "$TEST_STATE_DIR/prefill-started"
    while true; do "$TEST_SLEEP" 0.01; done
}
curl() { printf 'unexpected-health-probe\n'; return 28; }
sleep() { "$TEST_SLEEP" 0.01; }
hostname() { printf 'node0\n'; }
vllm-router() {
    while [[ ! -f "$TEST_STATE_DIR/prefill-started" ]]; do "$TEST_SLEEP" 0.01; done
    printf '%s\0' "$@"
    kill -TERM "$$"
}
"""
        status, stdout, stderr = self.run_shell(stubs + command, env={"TEST_STATE_DIR": self.workdir})
        self.assertEqual(status, 143, stderr)
        self.assertEqual(stderr, "")
        self.assert_router_arguments(stdout.rstrip("\0").split("\0"), ["http://node0:8001"], ["http://node4:8001"])

    def run_coupled_lifecycle(
        self, *, exit_role: str = "", exit_status: int = 0, shutdown_signal: str = "", exit_immediately: bool = False
    ) -> tuple[int, str, str]:
        _, command = self.generate_commands(env={"VLLM_PD_DEPLOYMENT_MODE": "coupled"})
        stubs = r"""
VLLM_COMMON_ARGS=(--common-test common)
VLLM_PREFILL_ARGS=(--prefill-test producer)
run_service() {
    local role=$1
    trap 'printf "%s-stopped\n" "$role"; exit 0' TERM
    printf '%s-started\n' "$role"
    touch "$TEST_STATE_DIR/$role-ready"
    if [[ "$role" == "$TEST_EXIT_ROLE" ]]; then
        # Synchronize on real process startup, not an assumed sleep duration.
        if [[ "$TEST_EXIT_IMMEDIATELY" == 0 ]]; then wait_for_services; fi
        printf '%s-exited\n' "$role"
        return "$TEST_EXIT_STATUS"
    fi
    if [[ "$role" == router && -n "$TEST_SHUTDOWN_SIGNAL" ]]; then
        wait_for_services
        # $$ remains the supervising shell's PID inside a background function.
        kill -s "$TEST_SHUTDOWN_SIGNAL" "$$"
    fi
    while true; do "$TEST_SLEEP" 0.01; done
}
vllm() { run_service prefill; }
vllm-router() { run_service router; }
wait_for_services() {
    while [[ ! -f "$TEST_STATE_DIR/prefill-ready" || ! -f "$TEST_STATE_DIR/router-ready" ]]; do
        "$TEST_SLEEP" 0.01
    done
}
sleep() { "$TEST_SLEEP" 0.01; }
hostname() { printf 'node0\n'; }
"""
        with TemporaryDirectory(prefix="gym-coupled-lifecycle-") as state_dir:
            return self.run_shell(
                stubs + command,
                env={
                    "TEST_STATE_DIR": state_dir,
                    "TEST_EXIT_ROLE": exit_role,
                    "TEST_EXIT_STATUS": str(exit_status),
                    "TEST_SHUTDOWN_SIGNAL": shutdown_signal,
                    "TEST_EXIT_IMMEDIATELY": str(int(exit_immediately)),
                },
            )

    def test_prefill_exit_after_router_start_stops_router(self):
        """Treat a prefill exit after router startup as a failure and stop the router."""
        for exit_status in (0, 7):
            with self.subTest(exit_status=exit_status):
                status, stdout, stderr = self.run_coupled_lifecycle(exit_role="prefill", exit_status=exit_status)
                expected_status = exit_status or 1
                self.assertEqual(status, expected_status, stderr)
                self.assertIn("router-started\n", stdout)
                self.assertIn("prefill-exited\n", stdout)
                self.assertIn("router-stopped\n", stdout)
                self.assertIn(f"prefill process exited after startup (status={expected_status})", stderr)

    def test_router_exit_after_startup_stops_prefill(self):
        """Propagate unexpected router exits as failures and stop prefill."""
        for exit_status in (0, 9):
            with self.subTest(exit_status=exit_status):
                status, stdout, stderr = self.run_coupled_lifecycle(exit_role="router", exit_status=exit_status)
                expected_status = exit_status or 1
                self.assertEqual(status, expected_status, stderr)
                self.assertIn("router-exited\n", stdout)
                self.assertIn("prefill-stopped\n", stdout)
                self.assertIn(f"router process exited after startup (status={expected_status})", stderr)

    def test_coupled_shutdown_signals_stop_both_processes(self):
        """Stop both services on SIGTERM or SIGINT while preserving the cancellation exit status."""
        for signal_name, expected_status in (("TERM", 143), ("INT", 130)):
            with self.subTest(signal=signal_name):
                status, stdout, stderr = self.run_coupled_lifecycle(shutdown_signal=signal_name)
                self.assertEqual(status, expected_status, stderr)
                self.assertIn("prefill-stopped\n", stdout)
                self.assertIn("router-stopped\n", stdout)
                self.assertNotIn("ERROR:", stderr)

    def test_immediate_prefill_exit_is_detected(self) -> None:
        """Propagate a prefill startup failure without waiting for worker readiness."""
        status, _, stderr = self.run_coupled_lifecycle(exit_role="prefill", exit_status=7, exit_immediately=True)
        self.assertEqual(status, 7, stderr)
        self.assertIn("prefill process exited after startup (status=7)", stderr)
