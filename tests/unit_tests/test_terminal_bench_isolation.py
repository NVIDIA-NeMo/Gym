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
"""Terminal-Bench agent/evaluator isolation.

Terminal-Bench grades the agent's *live* filesystem: the agent and the verifier
share one Apptainer overlay. That makes the usual "the tests are in a separate
image" guarantee unavailable, so these tests pin the properties that keep the
hidden tests, the verifier reward file, and the authoritative instance row out of
the agent container.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# swe_agents/app.py imports tomlkit at module level; it is a server-level dep
# (requirements.txt) absent from the core lock — skip rather than break a
# clean-env `pytest tests/unit_tests/` run.
pytest.importorskip("tomlkit")

from responses_api_agents.swe_agents.app import (  # noqa: E402
    _TB_TERMINAL_TOOL_OUTPUT,
    AGENT_VISIBLE_INSTANCE_FIELDS,
    ExecuteContainerCommandArgs,
    SWEBenchWrapper,
    TerminalBenchDatasetProcessor,
    _append_terminal_tool_outputs,
    _effective_agent_timeout_sec,
    _redact_instance_dict_for_agent,
    _tb_flatten_multipart_content,
    _tb_hidden_tests_install_script,
    _tb_net_bridge_cfg,
    _tb_repair_trajectory,
    _tb_tests_host_dir,
    _tb_verifier_host_dir,
    _validated_tb_tests_dir,
)


class TerminalBenchIsolationTests(unittest.TestCase):
    def test_tests_dir_is_limited_to_known_destructive_targets(self):
        self.assertEqual(_validated_tb_tests_dir("/tests", "/app"), "/tests")
        self.assertEqual(_validated_tb_tests_dir("/app/tests", "/app"), "/app/tests")
        for unsafe in ("tests", "/", "/app", "/tmp/tests", "/app/../tests"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    _validated_tb_tests_dir(unsafe, "/app")
        with self.assertRaises(ValueError):
            _validated_tb_tests_dir("/tests", "/tests/workspace")

    def test_hidden_test_install_replaces_agent_planted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            tests_dir = workspace / "tests"
            tests_dir.mkdir(parents=True)
            (tests_dir / "conftest.py").write_text("raise RuntimeError('agent')\n")
            (tests_dir / "stale.txt").write_text("agent\n")
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "test.sh").write_text("#!/bin/bash\nexit 0\n")
            (bundle / "test_outputs.py").write_text("def test_ok(): pass\n")

            validated = _validated_tb_tests_dir(str(tests_dir), str(workspace))
            script = _tb_hidden_tests_install_script(validated, "test.sh", bundle_dir=str(bundle))
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((tests_dir / "conftest.py").exists())
            self.assertFalse((tests_dir / "stale.txt").exists())
            self.assertEqual((tests_dir / "test_outputs.py").read_text(), "def test_ok(): pass\n")
            self.assertTrue((tests_dir / "test.sh").stat().st_mode & 0o100)

    def test_hidden_test_install_orders_delete_recreate_copy_and_chmod(self):
        script = _tb_hidden_tests_install_script("/tests", "test.sh")
        operations = ("rm -rf --", "mkdir -p --", "cp -a --", "chmod +x --")
        offsets = [script.index(operation) for operation in operations]
        self.assertEqual(offsets, sorted(offsets))

    def test_trajectory_repairs_only_apply_to_terminal_bench(self):
        """The repairs must never touch a non-terminal-bench trajectory."""

        def trajectory():
            return [
                {"role": "user", "content": [{"type": "text", "text": "multipart"}]},
                {"role": "assistant", "tool_calls": [{"id": "dangling", "type": "function"}]},
            ]

        tb_messages = trajectory()
        synthesized = _tb_repair_trajectory({"dataset_name": "terminal-bench"}, tb_messages)
        self.assertEqual(synthesized, 1)
        self.assertEqual(tb_messages[0]["content"], "multipart")
        self.assertEqual(
            tb_messages[-1],
            {"role": "tool", "tool_call_id": "dangling", "content": _TB_TERMINAL_TOOL_OUTPUT},
        )

        for dataset_name in ("swe-bench", "deepswe", "R2E-Gym-Subset"):
            with self.subTest(dataset_name=dataset_name):
                other = trajectory()
                self.assertIsNone(_tb_repair_trajectory({"dataset_name": dataset_name}, other))
                # Untouched: still multipart, still no synthesized tool result.
                self.assertEqual(other, trajectory())

    def test_multipart_content_flattens_to_the_text_the_model_consumed(self):
        messages = [
            {"role": "user", "content": "plain string stays"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Attached image(s) from tool result:"},
                    {
                        "type": "text",
                        "text": (
                            "ERROR: Cannot read image (this model does not support image input). Inform the user."
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "assistant untouched"}]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "t"}, "raw"]},
            {"role": "user", "content": [{"type": "text", "text": 5}, {"type": "text", "text": None}, None]},
        ]
        self.assertEqual(_tb_flatten_multipart_content(messages), 3)
        self.assertEqual(messages[0]["content"], "plain string stays")
        self.assertEqual(
            messages[1]["content"],
            "Attached image(s) from tool result:\n"
            "ERROR: Cannot read image (this model does not support image input). Inform the user.",
        )
        self.assertEqual(messages[2]["content"], [{"type": "text", "text": "assistant untouched"}])
        self.assertEqual(messages[3]["content"], "t\nraw")
        self.assertEqual(messages[4]["content"], "5")

    def test_terminal_bench_trusted_sif_root_is_exact_and_fail_closed(self):
        wrapper = SimpleNamespace(config=SimpleNamespace(debug=False))
        data_point = {
            "dataset_name": "terminal-bench",
            "instance_id": "task-a",
            "container_formatter": "/untrusted/{instance_id}.sif",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exact = root / "task-a_amd64.sif"
            exact.write_text("trusted\n")
            with patch.dict("os.environ", {"TB_TRUSTED_SIF_ROOT": str(root)}):
                self.assertEqual(SWEBenchWrapper._find_container(wrapper, data_point), str(exact))

            exact.unlink()
            (root / "prefix-task-a_amd64.sif").write_text("fuzzy\n")
            with patch.dict("os.environ", {"TB_TRUSTED_SIF_ROOT": str(root)}):
                with self.assertRaisesRegex(FileNotFoundError, "fuzzy fallback disabled"):
                    SWEBenchWrapper._find_container(wrapper, data_point)

    def test_only_final_dangling_tool_calls_get_explicit_terminal_results(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "old", "type": "function"}]},
            {"role": "assistant", "content": "continued"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "done", "type": "function"}, {"id": "dangling", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "done", "content": "ok"},
        ]
        self.assertEqual(_append_terminal_tool_outputs(messages), ["dangling"])
        self.assertEqual(
            messages[-1],
            {"role": "tool", "tool_call_id": "dangling", "content": _TB_TERMINAL_TOOL_OUTPUT},
        )

        terminal = [
            {"role": "user", "content": "repair"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "done", "type": "function"}, {"id": "dangling", "type": "function"}],
            },
        ]
        self.assertEqual(_append_terminal_tool_outputs(terminal), ["done", "dangling"])
        self.assertEqual(
            terminal[-1],
            {"role": "tool", "tool_call_id": "dangling", "content": _TB_TERMINAL_TOOL_OUTPUT},
        )

    def test_agent_instance_dict_excludes_terminal_bench_evaluator_fields(self):
        """The agent's instance row must never carry the hidden test payload.

        Redaction is an allowlist, so this asserts both that the allowlist cannot
        grow to admit an evaluator field and that a realistic Terminal-Bench row
        loses every one of them.
        """
        tb_evaluator_only = {"test_files", "test_files_b64", "test_entrypoint", "tests_dir", "verifier_timeout_sec"}
        self.assertEqual(tb_evaluator_only & AGENT_VISIBLE_INSTANCE_FIELDS, set())

        source = {
            "instance_id": "example",
            "workspace_path": "/app",
            "test_files": {"test.sh": "secret"},
            "test_files_b64": {"fixture.bin": "AA=="},
            "test_entrypoint": "test.sh",
            "tests_dir": "/tests",
            "verifier_timeout_sec": 30,
        }
        safe = _redact_instance_dict_for_agent(source)
        self.assertEqual(safe, {"instance_id": "example", "workspace_path": "/app"})
        self.assertNotIn("secret", json.dumps(safe))
        # Non-destructive: the host-side authoritative row is still intact.
        self.assertIn("test_files", source)

    def test_two_exec_commands_keep_tests_out_of_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = root / "results" / "task_run"
            persistent.mkdir(parents=True)
            eval_private = persistent / "eval_private"
            (eval_private / ".empty").mkdir(parents=True)
            private_tests = _tb_tests_host_dir(eval_private)
            private_tests.mkdir(parents=True)
            private_verifier = _tb_verifier_host_dir(eval_private)
            authoritative_dataset = eval_private / "instance.jsonl"
            authoritative_dataset.write_text("{}\n")
            safe_dataset = persistent / "instance_datasets" / "agent_instance.jsonl"
            safe_dataset.parent.mkdir(parents=True)
            safe_dataset.write_text("{}\n")

            setup = root / "opencode_setup"
            (setup / "opencode" / "evaluation" / "oh").mkdir(parents=True)
            (setup / "opencode" / "packages" / "opencode" / "migration").mkdir(parents=True)
            (setup / "bun").mkdir(parents=True)

            bridge = root / "bridge"
            (bridge / "libs").mkdir(parents=True)
            (bridge / "socat").write_text("")
            (bridge / "ld-linux.so").write_text("")

            params = SimpleNamespace(
                problem_info={
                    "dataset_name": "terminal-bench",
                    "instance_dict": json.dumps(
                        {"allow_internet": False, "workspace_path": "/app", "tests_dir": "/tests"}
                    ),
                },
                instance_dataset_path=authoritative_dataset,
                agent_instance_dataset_path=safe_dataset,
                eval_private_dir=eval_private,
                agent_framework="opencode",
                opencode_setup_dir=setup,
                openhands_setup_dir=None,
                persistent_dir=persistent,
                agent_run_id="example-run",
                resolved_system_prompt_template=None,
                resolved_user_prompt_template=None,
                swebench_setup_dir=root / "swebench",
                swebench_multilingual_setup_dir=root / "multilingual",
                swe_rebench_setup_dir=root / "rebench",
                model_patch_path=persistent / "patch.diff",
                container="/images/example.sif",
            )
            agent = ExecuteContainerCommandArgs(
                command="echo agent",
                expected_file_pattern=str(persistent / "agent.json"),
                mode="agent",
                timeout=60,
            )
            evaluate = ExecuteContainerCommandArgs(
                command="echo eval",
                expected_file_pattern=str(persistent / "eval.json"),
                mode="eval",
                timeout=60,
            )

            with patch.dict("os.environ", {"TB_NET_BRIDGE_PKG": str(bridge)}):
                agent_cmd = SWEBenchWrapper._build_apptainer_command(None, params, agent)
                eval_cmd = SWEBenchWrapper._build_apptainer_command(None, params, evaluate)
            agent_script = (persistent / "container_scripts" / "agent_script.sh").read_text()
            eval_script = (persistent / "container_scripts" / "eval_script.sh").read_text()

            self.assertIn(f"src={safe_dataset},dst=/root/dataset/data.jsonl", agent_cmd)
            self.assertNotIn(str(authoritative_dataset), agent_cmd)
            self.assertNotIn(str(private_tests), agent_cmd)
            self.assertNotIn(str(private_verifier), agent_cmd)
            self.assertNotIn("dst=/root/tb_tests", agent_cmd)
            self.assertNotIn("dst=/logs/verifier", agent_cmd)

            self.assertIn(f"src={authoritative_dataset},dst=/root/dataset/data.jsonl", eval_cmd)
            self.assertIn(f"src={private_tests},dst=/root/tb_tests,ro", eval_cmd)
            self.assertIn(f"src={private_verifier},dst=/logs/verifier", eval_cmd)

            self.assertIn("--net --network=none", agent_cmd)
            self.assertIn("--net --network=none", eval_cmd)
            self.assertIn("TCP-LISTEN:18900", agent_script)
            self.assertNotIn("TCP-LISTEN:18900", eval_script)
            self.assertIn("/.singularity.d/runscript", agent_script)
            self.assertIn("/.singularity.d/runscript", eval_script)

    def test_terminal_bench_uses_a_shared_persistent_overlay_not_tmpfs(self):
        """Agent and eval must share one overlay image, or live-state grading breaks."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = root / "results" / "task_run"
            persistent.mkdir(parents=True)
            eval_private = persistent / "eval_private"
            (eval_private / ".empty").mkdir(parents=True)
            _tb_tests_host_dir(eval_private).mkdir(parents=True)
            dataset = eval_private / "instance.jsonl"
            dataset.write_text("{}\n")

            setup = root / "opencode_setup"
            (setup / "opencode" / "evaluation" / "oh").mkdir(parents=True)
            (setup / "opencode" / "packages" / "opencode" / "migration").mkdir(parents=True)
            (setup / "bun").mkdir(parents=True)

            params = SimpleNamespace(
                problem_info={
                    "dataset_name": "terminal-bench",
                    # allow_internet defaults to true -> shared host netns, no bridge.
                    "instance_dict": json.dumps({"workspace_path": "/app", "tests_dir": "/tests"}),
                },
                instance_dataset_path=dataset,
                agent_instance_dataset_path=dataset,
                eval_private_dir=eval_private,
                agent_framework="opencode",
                opencode_setup_dir=setup,
                openhands_setup_dir=None,
                persistent_dir=persistent,
                agent_run_id="example-run",
                resolved_system_prompt_template=None,
                resolved_user_prompt_template=None,
                swebench_setup_dir=root / "swebench",
                swebench_multilingual_setup_dir=root / "multilingual",
                swe_rebench_setup_dir=root / "rebench",
                model_patch_path=persistent / "patch.diff",
                container="/images/example.sif",
            )
            commands = {
                mode: SWEBenchWrapper._build_apptainer_command(
                    None,
                    params,
                    ExecuteContainerCommandArgs(
                        command=f"echo {mode}",
                        expected_file_pattern=str(persistent / f"{mode}.json"),
                        mode=mode,
                        timeout=60,
                    ),
                )
                for mode in ("agent", "eval")
            }
            overlay = persistent / "agent_overlay.img"
            for mode, cmd in commands.items():
                with self.subTest(mode=mode):
                    self.assertIn(f"--overlay {overlay}", cmd)
                    self.assertNotIn("--writable-tmpfs", cmd)
            # allow_internet defaults to true -> no private netns.
            self.assertNotIn("--network=none", commands["agent"])

    def test_terminal_bench_verifier_timeout_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            persistent = Path(tmp) / "results" / "task_run"
            persistent.mkdir(parents=True)
            eval_private = persistent / "eval_private"
            eval_private.mkdir(parents=True)
            params = SimpleNamespace(
                problem_info={
                    "instance_dict": json.dumps(
                        {
                            "workspace_path": "/app",
                            "tests_dir": "/tests",
                            "test_entrypoint": "test.sh",
                            "test_files": {"test.sh": "#!/bin/bash\nexit 0\n"},
                            "verifier_timeout_sec": 60,
                        }
                    )
                },
                persistent_dir=persistent,
                eval_private_dir=eval_private,
                final_eval_apptainer_spinup_timestamp_mounted_fpath=(
                    "/trajectories_mount/final_eval_apptainer_spinup_timestamp"
                ),
                instance_id="task-a",
                swebench_tests_timeout=300,
                tb_verifier_timeout_floor_sec=1500,
            )
            processor = TerminalBenchDatasetProcessor.model_construct(config=params)
            command = processor.get_run_command()
            self.assertEqual(command.timeout, 1500)
            self.assertTrue(_tb_verifier_host_dir(eval_private).is_dir())
            self.assertIn("VERIFIER_STARTED_EPOCH", command.command)
            self.assertIn('REWARD_SOURCE="exit_code"', command.command)
            self.assertIn('REWARD_SOURCE="verifier_file"', command.command)
            self.assertIn("[ ! -L /logs/verifier/reward.txt ]", command.command)
            self.assertIn("export PYTHONDONTWRITEBYTECODE=1", command.command)
            self.assertLess(
                command.command.index("rm -rf -- /tests"),
                command.command.index("cp -a -- /root/tb_tests/bundle/. /tests/"),
            )

    def test_terminal_bench_stages_binary_test_fixtures_byte_identically(self):
        """`test_files_b64` fixtures must land byte-for-byte in the staged bundle."""
        payload = bytes(range(256))
        with tempfile.TemporaryDirectory() as tmp:
            persistent = Path(tmp) / "results" / "task_run"
            persistent.mkdir(parents=True)
            eval_private = persistent / "eval_private"
            eval_private.mkdir(parents=True)
            import base64

            params = SimpleNamespace(
                problem_info={
                    "instance_dict": json.dumps(
                        {
                            "workspace_path": "/app",
                            "test_files": {"test.sh": "#!/bin/bash\nexit 0\n"},
                            "test_files_b64": {"fixtures/ref.bin": base64.b64encode(payload).decode()},
                        }
                    )
                },
                persistent_dir=persistent,
                eval_private_dir=eval_private,
                final_eval_apptainer_spinup_timestamp_mounted_fpath="/trajectories_mount/ts",
                instance_id="task-a",
                swebench_tests_timeout=300,
                tb_verifier_timeout_floor_sec=None,
            )
            processor = TerminalBenchDatasetProcessor.model_construct(config=params)
            command = processor.get_run_command()
            self.assertEqual(command.timeout, 300)

            bundle = _tb_tests_host_dir(eval_private) / "bundle"
            self.assertEqual((bundle / "fixtures" / "ref.bin").read_bytes(), payload)
            self.assertEqual((bundle / "test.sh").read_text(), "#!/bin/bash\nexit 0\n")

    def test_terminal_bench_requires_the_entrypoint_to_be_in_the_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            persistent = Path(tmp) / "results" / "task_run"
            persistent.mkdir(parents=True)
            eval_private = persistent / "eval_private"
            eval_private.mkdir(parents=True)
            params = SimpleNamespace(
                problem_info={
                    "instance_dict": json.dumps(
                        {"workspace_path": "/app", "test_entrypoint": "run.sh", "test_files": {"test.sh": "x"}}
                    )
                },
                persistent_dir=persistent,
                eval_private_dir=eval_private,
                final_eval_apptainer_spinup_timestamp_mounted_fpath="/trajectories_mount/ts",
                instance_id="task-a",
                swebench_tests_timeout=300,
                tb_verifier_timeout_floor_sec=None,
            )
            processor = TerminalBenchDatasetProcessor.model_construct(config=params)
            with self.assertRaisesRegex(ValueError, "not present in instance_dict"):
                processor.get_run_command()

            params.problem_info["instance_dict"] = json.dumps({"workspace_path": "/app", "test_files": {}})
            with self.assertRaisesRegex(ValueError, "is missing"):
                processor.get_run_command()

    def test_single_exec_mounts_the_opencode_harness_and_exposes_hidden_tests(self):
        """`agent_eval` (tb_single_exec) needs the agent harness AND the verifier mounts.

        It runs both phases in one exec, so it must mount the opencode harness like an
        agent exec and the test bundle like an eval exec. The second half of that is a
        deliberate, documented trade: because Apptainer fixes mounts at exec start, the
        hidden tests are readable by the agent for the whole run. Two-exec does not have
        this property (see test_two_exec_commands_keep_tests_out_of_agent); single-exec
        buys daemon persistence for service tasks at the cost of test isolation.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = root / "results" / "task_run"
            persistent.mkdir(parents=True)
            eval_private = persistent / "eval_private"
            (eval_private / ".empty").mkdir(parents=True)
            _tb_tests_host_dir(eval_private).mkdir(parents=True)
            dataset = eval_private / "instance.jsonl"
            dataset.write_text("{}\n")
            setup = root / "opencode_setup"
            (setup / "opencode" / "evaluation" / "oh").mkdir(parents=True)
            (setup / "opencode" / "packages" / "opencode" / "migration").mkdir(parents=True)
            (setup / "bun").mkdir(parents=True)

            params = SimpleNamespace(
                problem_info={
                    "dataset_name": "terminal-bench",
                    "instance_dict": json.dumps({"workspace_path": "/app", "tests_dir": "/tests"}),
                },
                instance_dataset_path=dataset,
                agent_instance_dataset_path=dataset,
                eval_private_dir=eval_private,
                agent_framework="opencode",
                opencode_setup_dir=setup,
                # Deliberately None: an opencode run has no OpenHands setup dir, so if the
                # harness dispatch ever stops recognising "agent_eval" this assertion fires.
                openhands_setup_dir=None,
                persistent_dir=persistent,
                agent_run_id="example-run",
                resolved_system_prompt_template=None,
                resolved_user_prompt_template=None,
                swebench_setup_dir=root / "swebench",
                swebench_multilingual_setup_dir=root / "multilingual",
                swe_rebench_setup_dir=root / "rebench",
                model_patch_path=persistent / "patch.diff",
                container="/images/example.sif",
            )
            merged = SWEBenchWrapper._build_apptainer_command(
                None,
                params,
                ExecuteContainerCommandArgs(
                    command="echo merged",
                    expected_file_pattern=str(persistent / "eval.json"),
                    mode="agent_eval",
                    timeout=60,
                ),
            )

            # Agent side: the opencode harness must be present or the agent phase cannot run.
            self.assertIn("dst=/opencode_setup/opencode,ro", merged)
            self.assertIn("dst=/root/dataset/data.jsonl", merged)
            # Eval side: verifier mounts must be present or the grading phase cannot run.
            self.assertIn(f"src={_tb_tests_host_dir(eval_private)},dst=/root/tb_tests,ro", merged)
            self.assertIn("dst=/logs/verifier", merged)
            # Same shared overlay as every other terminal-bench exec.
            self.assertIn(f"--overlay {persistent / 'agent_overlay.img'}", merged)

    def test_per_instance_agent_timeout_overrides_the_config_default(self):
        """TB rows declare their own agent budget; it must beat the config-wide value."""
        config = SimpleNamespace(
            problem_info={"instance_dict": json.dumps({"agent_timeout_sec": 2400})},
            swebench_agent_timeout=7200,
        )
        self.assertEqual(_effective_agent_timeout_sec(config), 2400)

        # Rows without the key (every non-TB dataset) keep the config value.
        for instance_dict in ({}, {"agent_timeout_sec": 0}, {"agent_timeout_sec": None}):
            with self.subTest(instance_dict=instance_dict):
                config.problem_info = {"instance_dict": json.dumps(instance_dict)}
                self.assertEqual(_effective_agent_timeout_sec(config), 7200)

        # A malformed instance_dict must not take the rollout down.
        config.problem_info = {"instance_dict": "{not json"}
        self.assertEqual(_effective_agent_timeout_sec(config), 7200)

    def test_offline_task_without_a_configured_bridge_fails_closed(self):
        """An offline task must never silently fall back to the shared network."""
        offline = SimpleNamespace(
            problem_info={
                "dataset_name": "terminal-bench",
                "instance_dict": json.dumps({"allow_internet": False}),
            },
            agent_run_id="example-run",
            tb_net_bridge_pkg=None,
            tb_net_bridge_port=18900,
        )
        with patch.dict("os.environ", {}, clear=False) as _env:
            os.environ.pop("TB_NET_BRIDGE_PKG", None)
            with self.assertRaisesRegex(RuntimeError, "no .*bundle is configured"):
                _tb_net_bridge_cfg(offline)

        # An online task needs no bridge at all, configured or not.
        online = SimpleNamespace(
            problem_info={
                "dataset_name": "terminal-bench",
                "instance_dict": json.dumps({"allow_internet": True}),
            },
            agent_run_id="example-run",
            tb_net_bridge_pkg=None,
            tb_net_bridge_port=18900,
        )
        self.assertIsNone(_tb_net_bridge_cfg(online))

    def test_bridge_port_comes_from_config_and_env_overrides_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = Path(tmp) / "bridge"
            (bridge / "libs").mkdir(parents=True)
            (bridge / "socat").write_text("")
            (bridge / "ld-linux.so").write_text("")
            config = SimpleNamespace(
                problem_info={
                    "dataset_name": "terminal-bench",
                    "instance_dict": json.dumps({"allow_internet": False}),
                },
                agent_run_id="example-run",
                tb_net_bridge_pkg=str(bridge),
                tb_net_bridge_port=19100,
            )
            os.environ.pop("TB_NET_BRIDGE_PKG", None)
            os.environ.pop("TB_NET_BRIDGE_PORT", None)
            self.assertEqual(_tb_net_bridge_cfg(config)["port"], 19100)

            with patch.dict("os.environ", {"TB_NET_BRIDGE_PORT": "19200"}):
                self.assertEqual(_tb_net_bridge_cfg(config)["port"], 19200)


if __name__ == "__main__":
    unittest.main()
