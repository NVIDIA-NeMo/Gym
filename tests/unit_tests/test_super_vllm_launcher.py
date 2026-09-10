# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh"


class TestSuperVllmLauncher(unittest.TestCase):
    def setUp(self):
        # Never inherit cluster credentials, tuning overrides, or real sbatch commands.
        self.env = {
            "PATH": os.defpath,
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

    def run_shell(self, command, *args, env=None):
        proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-c", command, "launcher-test", *args],
            env=self.env | (env or {}),
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

    def eval_arguments(self, *overrides, env=None):
        command, _ = self.generate_commands(*overrides, env=env)
        command = command.replace("source /opt/Gym_venv/bin/activate", ":").replace("cd /opt/Gym\n", ":\n")
        stubs = r"""
gym() {
    if [[ "$2" == run ]]; then printf '%s\0' "$@"; fi
}
date() { printf '20260909_120000\n'; }
getent() { printf '10.0.0.1 node0\n'; }
"""
        status, stdout, stderr = self.run_shell(stubs + command, env=env)
        self.assertEqual(status, 0, stderr)
        return stdout.rstrip("\0").split("\0")

    def settings(self, args, key):
        return [arg for arg in args if arg.lstrip("+").startswith(key + "=")]

    def serving_arguments(self, command, *, rank, coupled_head=False, env=None):
        # Record argv separately for vLLM and the router; marker files synchronize startup.
        stubs = r"""
VLLM_COMMON_ARGS=(--common-test 'value with spaces')
VLLM_PREFILL_ARGS=(--prefill-test producer)
VLLM_DECODE_ARGS=(--decode-test consumer)
vllm() {
    printf '%s\0' "$VLLM_NIXL_SIDE_CHANNEL_HOST" "$VLLM_NIXL_SIDE_CHANNEL_PORT" "$@"
    touch "$TEST_STATE_DIR/service-ready"
    if [[ "$TEST_COUPLED_HEAD" == 1 ]]; then
        while true; do command sleep 0.01; done
    fi
    if (( SLURM_PROCID == 0 )); then
        while [[ ! -f "$TEST_STATE_DIR/router-ready" ]]; do command sleep 0.01; done
    fi
}
vllm-router() {
    printf '%s\0' "$@" >&2
    touch "$TEST_STATE_DIR/router-ready"
    if [[ "$TEST_COUPLED_HEAD" == 1 ]]; then
        kill -TERM "$$"
    else
        # Independent mode checks router liveness before starting vLLM.
        while true; do command sleep 0.01; done
    fi
}
curl() { [[ -f "$TEST_STATE_DIR/service-ready" ]]; }
sleep() { command sleep 0.01; }
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

    def assert_router_arguments(self, args, prefill_urls, decode_urls):
        expected = [
            "--prefill-policy",
            "cache_aware",
            "--decode-policy",
            "cache_aware",
            "--balance-abs-threshold",
            "4",
            "--balance-rel-threshold",
            "1.1",
            "--vllm-pd-disaggregation",
            "--host",
            "node0",
            "--port",
            "8000",
            "--intra-node-data-parallel-size",
            "1",
            "--request-timeout-secs",
            "86400",
            "--log-level",
            "error",
        ]
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
        self.assertEqual(args, expected)

    def test_independent_mode_preserves_per_node_engines(self):
        """Keep one engine per node and all router destinations with default or explicit independent mode."""
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

    def test_coupled_nodes_use_correct_tier_roles_and_ranks(self):
        """Assign the correct API heads, headless ranks, tier sizes, and ports for each coupled node."""
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
                        expected += ["--host", host, "--port", "8001" if is_prefill else "8002"]
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
                            router, ["http://node0:8001"], [f"http://node{prefill_count}:8002"]
                        )
                    else:
                        self.assertEqual(router, [])

    def test_router_overrides_apply_to_both_deployment_modes(self):
        """Preserve main's configurable router policies and local data-parallel size in both modes."""
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

    def test_independent_router_startup_failure_stops_launch(self):
        """Stop independent startup when the router exits during main's five-second startup check."""
        stubs = r"""
VLLM_COMMON_ARGS=()
VLLM_PREFILL_ARGS=()
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
        """Keep the four-hour default and one segment spanning every allocated node."""
        for prefill_count, decode_count in ((1, 1), (1, 4), (4, 4)):
            with self.subTest(prefill=prefill_count, decode=decode_count):
                _, _, _, calls = self.capture_submission(
                    "--config",
                    "benchmark.yaml",
                    env={"NUM_PREFILL_NODES": str(prefill_count), "NUM_DECODE_NODES": str(decode_count)},
                )
                self.assertEqual(len(calls), 2)
                self.assertIn(f"--nodes={prefill_count + decode_count}", calls[0])
                self.assertIn(f"--segment={prefill_count + decode_count}", calls[0])
                self.assertIn("--time=04:00:00", calls[0])
                self.assertIn("--ntasks-per-node=1", calls[0])
                self.assertIn("--exclusive", calls[0])
                self.assertIn("--dependency=afterany:12345", calls[1])

    def test_submission_overrides_do_not_change_cleanup_allocation(self):
        """Apply custom walltime and segment size only to the main job, leaving cleanup CPU-only and short."""
        _, _, _, calls = self.capture_submission(
            "--config", "benchmark.yaml", env={"SBATCH_TIME": "7-00:00:00", "VLLM_SLURM_SEGMENT": "4"}
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("--time=7-00:00:00", calls[0])
        self.assertIn("--segment=4", calls[0])
        self.assertIn("--time=00:30:00", calls[1])
        self.assertIn("--nodes=1", calls[1])
        self.assertIn("--partition=cpu", calls[1])
        self.assertIn("--gres=none", calls[1])
        self.assertFalse(any(arg.startswith("--segment=") for arg in calls[1]))

    def test_invalid_launcher_controls_are_rejected_before_submission(self):
        """Reject invalid concurrency, deployment modes, and segment sizes without submitting any job."""
        stub = 'sbatch() { printf "unexpected-submission\\n"; }; source "$@"'
        cases = {
            "NUM_SAMPLES_IN_PARALLEL": (("0", "-1", "1.5", "bad"), 1),
            "VLLM_PD_DEPLOYMENT_MODE": (("bad", "COUPLED"), 1),
            "VLLM_SLURM_SEGMENT": (("0", "-1", "1.5", "bad"), 2),
        }
        for name, (values, expected_status) in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    status, stdout, stderr = self.run_shell(stub, str(SCRIPT), env={name: value})
                    self.assertEqual(status, expected_status, stderr)
                    self.assertNotIn("unexpected-submission", stdout)
                    self.assertIn(name, stderr)

    def test_serving_only_skips_evaluation_and_cleanup_submission(self):
        """Run only the serving step without eval arguments and preserve its success or failure status."""
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

    def test_environment_fallbacks_are_preserved(self):
        """Apply concurrency and resume environment defaults, including the stable results path."""
        args = self.eval_arguments(env={"NUM_SAMPLES_IN_PARALLEL": "16", "RESUME_EVAL_ON_REQUEUE": "1"})
        self.assertEqual(self.settings(args, "num_samples_in_parallel"), ["++num_samples_in_parallel=16"])
        self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
        self.assertIn("++output_jsonl_fpath=results/launcher-test/resumable.jsonl", args)

    def test_no_environment_defaults_leave_gym_settings_untouched(self):
        """Preserve Gym settings and timestamped output naming when environment overrides are unset."""
        args = self.eval_arguments()
        self.assertEqual(self.settings(args, "num_samples_in_parallel"), [])
        self.assertEqual(self.settings(args, "resume_from_cache"), [])
        self.assertIn("++output_jsonl_fpath=results/launcher-test/slurm_job_id_12345/date_20260909_120000.jsonl", args)

    def test_explicit_concurrency_overrides_environment(self):
        """Let explicit concurrency arguments override the environment without duplicate settings."""
        for prefix in ("", "+", "++"):
            with self.subTest(prefix=prefix):
                override = prefix + "num_samples_in_parallel=512"
                args = self.eval_arguments(override, env={"NUM_SAMPLES_IN_PARALLEL": "16"})
                self.assertEqual(self.settings(args, "num_samples_in_parallel"), [override])

    def test_explicit_resume_overrides_environment(self):
        """Honor explicit resume arguments while retaining the switch's stable output naming."""
        for prefix in ("", "+", "++"):
            for value in ("true", "false"):
                with self.subTest(prefix=prefix, value=value):
                    override = prefix + "resume_from_cache=" + value
                    args = self.eval_arguments(override, env={"RESUME_EVAL_ON_REQUEUE": "1"})
                    self.assertEqual(self.settings(args, "resume_from_cache"), [override])
                    self.assertIn("++output_jsonl_fpath=results/launcher-test/resumable.jsonl", args)

    def test_resume_switch_keeps_output_path_across_job_ids(self):
        """Keep resumable output and W&B names stable when the Slurm job ID changes."""
        for job_id in ("12345", "12346"):
            with self.subTest(job_id=job_id):
                args = self.eval_arguments(env={"RESUME_EVAL_ON_REQUEUE": "1", "SLURM_JOB_ID": job_id})
                self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
                self.assertIn("++output_jsonl_fpath=results/launcher-test/resumable.jsonl", args)
                self.assertIn("+wandb_name=launcher-test/resumable", args)

    def test_invalid_resume_switch_is_rejected_before_submission(self):
        """Reject an invalid resume switch before any Slurm job is submitted."""
        stub = r"""
sbatch() { printf 'unexpected-submission\n'; }
source "$@"
"""
        status, stdout, stderr = self.run_shell(stub, str(SCRIPT), env={"RESUME_EVAL_ON_REQUEUE": "invalid"})
        self.assertNotEqual(status, 0)
        self.assertNotIn("unexpected-submission", stdout)
        self.assertIn("RESUME_EVAL_ON_REQUEUE must be 0 or 1", stderr)

    def test_explicit_resume_keeps_output_path_across_job_ids(self):
        """Preserve a manually supplied resume path while keeping the default per-job run naming."""
        for job_id in ("12345", "12346"):
            with self.subTest(job_id=job_id):
                args = self.eval_arguments(
                    "++resume_from_cache=true",
                    env={"ROLLOUTS_FPATH": "results/existing/resumable.jsonl", "SLURM_JOB_ID": job_id},
                )
                self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
                self.assertIn("++output_jsonl_fpath=results/existing/resumable.jsonl", args)
                self.assertIn(f"+wandb_name=launcher-test/slurm_job_id_{job_id}/date_20260909_120000", args)

    def test_unrelated_override_does_not_suppress_concurrency_default(self):
        """Keep global environment defaults when overrides only target nested agent settings."""
        args = self.eval_arguments(
            "++agent.num_samples_in_parallel=8",
            "++agent.resume_from_cache=false",
            env={"NUM_SAMPLES_IN_PARALLEL": "16", "RESUME_EVAL_ON_REQUEUE": "1"},
        )
        self.assertEqual(self.settings(args, "num_samples_in_parallel"), ["++num_samples_in_parallel=16"])
        self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
        self.assertIn("++agent.num_samples_in_parallel=8", args)
        self.assertIn("++agent.resume_from_cache=false", args)

    def test_prefill_exit_is_detected_during_both_health_checks(self):
        """Detect prefill exits during either startup health check, including after request timeouts."""
        _, command = self.generate_commands(env={"VLLM_PD_DEPLOYMENT_MODE": "coupled"})
        stubs = r"""
VLLM_COMMON_ARGS=()
VLLM_PREFILL_ARGS=()
vllm() { command sleep 0.1; return "$TEST_PREFILL_STATUS"; }
curl() {
    if [[ "$TEST_WAIT_ROLE" == decode && "$*" == *':8001/health'* ]]; then
        return 0
    fi
    return "$TEST_HEALTH_STATUS"
}
sleep() { command sleep 0.01; }
hostname() { printf 'node0\n'; }
vllm-router() { printf 'unexpected-router-start\n'; }
"""
        for role in ("prefill", "decode"):
            for exit_status in (0, 7):
                # curl returns 28 for a timed-out request. That must not hide the
                # prefill exit status or abort the polling loop prematurely.
                for health_status in (1, 28):
                    with self.subTest(role=role, exit_status=exit_status, health_status=health_status):
                        status, stdout, stderr = self.run_shell(
                            stubs + command,
                            env={
                                "TEST_WAIT_ROLE": role,
                                "TEST_PREFILL_STATUS": str(exit_status),
                                "TEST_HEALTH_STATUS": str(health_status),
                            },
                        )
                        expected_status = exit_status or 1
                        self.assertEqual(status, expected_status, stderr)
                        self.assertNotIn("unexpected-router-start", stdout)
                        self.assertIn(
                            f"prefill vLLM process exited while waiting for {role} health (status={expected_status})",
                            stderr,
                        )

    def test_coupled_health_probes_have_timeouts_and_retry_until_ready(self):
        """Bound both tiers' health probes and retry timed-out requests until the servers are ready."""
        _, command = self.generate_commands(env={"VLLM_PD_DEPLOYMENT_MODE": "coupled"})
        stubs = r"""
VLLM_COMMON_ARGS=()
VLLM_PREFILL_ARGS=()
vllm() { while true; do command sleep 0.01; done; }
probe_count=0
curl() {
    printf '%s\0' "$@" >&2
    printf '\0' >&2
    probe_count=$((probe_count + 1))
    # Both tiers time out once, then become healthy while prefill stays alive.
    if (( probe_count % 2 )); then return 28; fi
    return 0
}
sleep() { :; }
hostname() { printf 'node0\n'; }
vllm-router() { printf 'router-started\n'; kill -TERM "$$"; }
"""
        status, stdout, stderr = self.run_shell(stubs + command)
        self.assertEqual(status, 143, stderr)
        self.assertEqual(stdout, "router-started\n")
        probes = [probe.split("\0") for probe in stderr.rstrip("\0").split("\0\0")]
        self.assertEqual(len(probes), 4)
        self.assertEqual(
            [args[-1] for args in probes],
            ["http://node0:8001/health"] * 2 + ["http://node4:8002/health"] * 2,
        )
        for args in probes:
            self.assertIn("--connect-timeout", args)
            self.assertEqual(args[args.index("--connect-timeout") + 1], "5")
            self.assertIn("--max-time", args)
            self.assertEqual(args[args.index("--max-time") + 1], "10")

    def test_healthy_coupled_servers_start_router(self):
        """Start the router with the correct prefill and decode URLs once both servers are healthy."""
        _, command = self.generate_commands(env={"VLLM_PD_DEPLOYMENT_MODE": "coupled"})
        stubs = r"""
VLLM_COMMON_ARGS=()
VLLM_PREFILL_ARGS=()
vllm() { while true; do command sleep 0.01; done; }
curl() { return 0; }
hostname() { printf 'node0\n'; }
vllm-router() { printf '%s\0' "$@"; kill -TERM "$$"; }
"""
        status, stdout, stderr = self.run_shell(stubs + command)
        self.assertEqual(status, 143, stderr)
        args = stdout.rstrip("\0").split("\0")
        self.assertEqual(args[args.index("--prefill") + 1], "http://node0:8001")
        self.assertEqual(args[args.index("--decode") + 1], "http://node4:8002")

    def run_coupled_lifecycle(
        self, *, exit_role="", exit_status=0, shutdown_signal="", startup_shutdown=False, exit_before_monitoring=False
    ):
        _, command = self.generate_commands(env={"VLLM_PD_DEPLOYMENT_MODE": "coupled"})
        stubs = r"""
VLLM_COMMON_ARGS=()
VLLM_PREFILL_ARGS=()
run_service() {
    local role=$1
    trap 'printf "%s-stopped\n" "$role"; exit 0' TERM
    trap 'exit "$TEST_EXIT_STATUS"' USR1
    printf '%s-started\n' "$role"
    touch "$TEST_STATE_DIR/$role-ready"
    if [[ "$role" == "$TEST_EXIT_ROLE" ]]; then
        # Synchronize on real process startup, not an assumed sleep duration.
        while [[ ! -f "$TEST_STATE_DIR/prefill-ready" || ! -f "$TEST_STATE_DIR/router-ready" ]]; do
            command sleep 0.01
        done
        printf '%s-exited\n' "$role"
        return "$TEST_EXIT_STATUS"
    fi
    if [[ "$role" == router && -n "$TEST_SHUTDOWN_SIGNAL" ]]; then
        # $$ remains the supervising shell's PID inside a background function.
        kill -s "$TEST_SHUTDOWN_SIGNAL" "$$"
    fi
    while true; do command sleep 0.01; done
}
vllm() { run_service prefill; }
vllm-router() { run_service router; }
curl() {
    [[ -f "$TEST_STATE_DIR/prefill-ready" ]] || return 1
    if [[ "$TEST_STARTUP_SHUTDOWN" == 1 ]]; then
        kill -s "$TEST_SHUTDOWN_SIGNAL" "$$"
    fi
    if [[ "$TEST_EXIT_BEFORE_MONITORING" == 1 && "$*" == *':8002/health'* ]]; then
        # Make prefill exit before the final health request returns success.
        kill -USR1 "$prefill_pid"
        wait "$prefill_pid" || true
    fi
    return 0
}
sleep() { command sleep 0.01; }
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
                    "TEST_STARTUP_SHUTDOWN": str(int(startup_shutdown)),
                    "TEST_EXIT_BEFORE_MONITORING": str(int(exit_before_monitoring)),
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

    def test_coupled_shutdown_during_readiness_stops_prefill(self):
        """Stop prefill on cancellation during readiness without starting the router."""
        for signal_name, expected_status in (("TERM", 143), ("INT", 130)):
            with self.subTest(signal=signal_name):
                status, stdout, stderr = self.run_coupled_lifecycle(shutdown_signal=signal_name, startup_shutdown=True)
                self.assertEqual(status, expected_status, stderr)
                self.assertIn("prefill-stopped\n", stdout)
                self.assertNotIn("router-started\n", stdout)
                self.assertNotIn("ERROR:", stderr)

    def test_prefill_exit_before_runtime_monitoring_is_detected(self):
        """Catch a prefill exit between the final health response and runtime monitoring."""
        status, _, stderr = self.run_coupled_lifecycle(exit_status=7, exit_before_monitoring=True)
        self.assertEqual(status, 7, stderr)
        self.assertIn("prefill process exited after startup (status=7)", stderr)
