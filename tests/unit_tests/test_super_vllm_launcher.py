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

    def generate_commands(self, *overrides, env=None):
        # Capture both commands using a shell function; never submit a Slurm job.
        stub = r"""
sbatch() {
    if [[ -n "${vllm_command:-}" ]]; then
        printf '%s\0' "$eval_command" "$vllm_command" >&2
    fi
    printf '12345\n'
}
source "$@"
"""
        status, _, captured = self.run_shell(stub, str(SCRIPT), "--config", "benchmark.yaml", *overrides, env=env)
        self.assertEqual(status, 0, captured)
        eval_command, pd_command, terminator = captured.split("\0")
        self.assertEqual(terminator, "")
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
