import inspect
import json
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
    H9_OPENCODE_BUN_ARTIFACTS,
    H9_OPENCODE_BUNDLE_ARTIFACTS,
    OPENCODE_SETUP_RECEIPT_SCHEMA,
    ExecuteContainerCommandArgs,
    SWEBenchWrapper,
    TerminalBenchDatasetProcessor,
    _append_terminal_tool_outputs,
    _build_opencode_setup_receipt,
    _opencode_receipt_is_authorized,
    _opencode_receipt_matches_runtime,
    _opencode_tree_digest,
    _tb_agent_safe_instance_dict,
    _tb_flatten_multipart_content,
    _tb_hidden_tests_install_script,
    _tb_private_tests_host_dir,
    _tb_private_verifier_host_dir,
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
            script = _tb_hidden_tests_install_script(
                validated, "test.sh", bundle_dir=str(bundle)
            )
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((tests_dir / "conftest.py").exists())
            self.assertFalse((tests_dir / "stale.txt").exists())
            self.assertEqual(
                (tests_dir / "test_outputs.py").read_text(), "def test_ok(): pass\n"
            )
            self.assertTrue((tests_dir / "test.sh").stat().st_mode & 0o100)

    def test_hidden_test_install_orders_delete_recreate_copy_and_chmod(self):
        script = _tb_hidden_tests_install_script("/tests", "test.sh")
        operations = ("rm -rf --", "mkdir -p --", "cp -a --", "chmod +x --")
        offsets = [script.index(operation) for operation in operations]
        self.assertEqual(offsets, sorted(offsets))

    def test_hsg_arm64_bun_artifact_is_pinned(self):
        self.assertEqual(
            H9_OPENCODE_BUN_ARTIFACTS["aarch64"],
            {
                "size": 101309000,
                "sha256": "67892bb6734ef352f90a964f0a2a680d5f6e6d15bdb5510e56b8065ac4d599ae",
            },
        )
        self.assertEqual(
            H9_OPENCODE_BUNDLE_ARTIFACTS["aarch64"],
            {
                "size": 23260101,
                "sha256": "027862bbde53b41acc2814d8d2b7a48168c7aa74f8a34326d51ab76a6a1c2709",
            },
        )

    def test_opencode_setup_paths_are_gated_on_the_preauthorized_pin(self):
        setup_script = (
            Path(__file__).resolve().parents[2]
            / "responses_api_agents"
            / "swe_agents"
            / "setup_scripts"
            / "opencode.sh"
        ).read_text()
        # Hardened (pinned-archive) path stays intact and preauthorized...
        self.assertIn(
            "expected_commit=0c088fd18b5ef6ff9b2a949ac2a41204b1ac8046",
            setup_script,
        )
        self.assertIn(
            "archive_sha256=2f417d942189d2d2b93f74fd2b87e977c96e5db951191c2103643c816cd73d62",
            setup_script,
        )
        self.assertIn('handle.extractall(extract_root, filter="data")', setup_script)
        self.assertIn(
            'BENCH_CLI="$OPENCODE_DIR/.bench-build/bench-cli.js"', setup_script
        )
        self.assertIn('"schema_version": "h8_opencode_setup_receipt_v2"', setup_script)
        self.assertIn('"source_tree": tree_identity(', setup_script)
        self.assertIn('"build_tree": tree_identity(', setup_script)
        # ...but ONLY behind the (repo, commit) == pin gate: any other config
        # (e.g. stock SWE configs pinned to sdd/dev) takes the original
        # git-clone path and is never hard-failed.
        self.assertIn(
            '[ "$agent_framework_repo" = "$expected_repo" ] '
            '&& [ "$agent_framework_commit" = "$expected_commit" ]',
            setup_script,
        )
        self.assertIn('git clone "$agent_framework_repo"', setup_script)
        self.assertIn('git checkout "$agent_framework_commit"', setup_script)
        self.assertNotIn("ERROR: unrecognized opencode commit", setup_script)
        # Default path keeps the fork-point non-frozen bun-install fallback and
        # the non-fatal bench-cli warm bundle.
        self.assertIn('[ "$hardened" != "1" ] && bun install --ignore-scripts', setup_script)
        self.assertIn('--entry-naming "bench-cli.js" || true', setup_script)

        app_source = (
            Path(__file__).resolve().parents[2]
            / "responses_api_agents"
            / "swe_agents"
            / "app.py"
        ).read_text()
        # Receipt enforcement is opt-in config (default False) and setup dirs
        # are not relocatable via ambient environment variables.
        self.assertIn("opencode_setup_receipt_enforce: bool = Field(", app_source)
        self.assertNotIn("NEMO_GYM_OPENCODE_SETUP_DIR", app_source)
        # Enforced setups use their own directory so receipt-invalidation
        # rmtree can never poison the shared default-path cache.
        self.assertIn('"swe_opencode_setup_preauthorized"', app_source)

    def test_opencode_setup_receipt_enforcement_defaults_off(self):
        from responses_api_agents.swe_agents.app import SWEBenchWrapperConfig

        field = SWEBenchWrapperConfig.model_fields["opencode_setup_receipt_enforce"]
        self.assertFalse(field.default)

    def test_trajectory_repairs_are_terminal_bench_gated_in_inner_responses(self):
        source = inspect.getsource(SWEBenchWrapper._inner_responses)
        gate = 'if params.problem_info.get("dataset_name") == "terminal-bench":'
        self.assertIn(gate, source)
        gate_at = source.index(gate)
        self.assertGreater(source.index("_append_terminal_tool_outputs("), gate_at)
        self.assertGreater(source.index("_tb_flatten_multipart_content("), gate_at)
        self.assertGreater(
            source.index('metrics_to_update["synthetic_terminal_tool_outputs"]'), gate_at
        )

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
                            "ERROR: Cannot read image (this model does not support "
                            "image input). Inform the user."
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
            "ERROR: Cannot read image (this model does not support image input). "
            "Inform the user.",
        )
        self.assertEqual(messages[2]["content"], [{"type": "text", "text": "assistant untouched"}])
        self.assertEqual(messages[3]["content"], "t\nraw")
        self.assertEqual(messages[4]["content"], "5")

    def test_opencode_receipt_v2_rejects_source_and_build_tree_drift(self):
        repo = "https://github.com/sdevare-nv/nv-opencode.git"
        commit = "0c088fd18b5ef6ff9b2a949ac2a41204b1ac8046"
        mutations = (
            "evaluation/benchmarks/swe_bench/scripts/run_infer.sh",
            "packages/opencode/src/bench/cli.ts",
            "packages/opencode/migration/001_init.sql",
        )
        for relative in mutations:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                opencode = root / "opencode"
                for source_relative in mutations:
                    path = opencode / source_relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"source:{source_relative}\n")
                (opencode / ".source_commit").write_text(commit + "\n")
                (opencode / "node_modules").mkdir()
                build = opencode / ".bench-build"
                build.mkdir()
                (build / "bench-cli.js").write_text("bench bundle\n")
                (build / "opencode.js").write_text("opencode bundle\n")
                bun = root / "bun"
                bun.write_text("bun\n")
                receipt = _build_opencode_setup_receipt(opencode, bun, repo, commit)

                self.assertEqual(receipt["schema_version"], OPENCODE_SETUP_RECEIPT_SCHEMA)
                self.assertTrue(
                    _opencode_receipt_matches_runtime(receipt, opencode, bun, repo, commit)
                )
                self.assertTrue(_opencode_receipt_is_authorized(receipt, receipt))
                (opencode / relative).write_text("mutated\n")
                self.assertFalse(
                    _opencode_receipt_matches_runtime(receipt, opencode, bun, repo, commit)
                )

    def test_opencode_receipt_v2_rejects_additions_missing_commit_and_build_drift(self):
        repo = "https://github.com/sdevare-nv/nv-opencode.git"
        commit = "0c088fd18b5ef6ff9b2a949ac2a41204b1ac8046"

        def fixture(root: Path):
            opencode = root / "opencode"
            source = opencode / "packages" / "opencode" / "src" / "bench" / "cli.ts"
            source.parent.mkdir(parents=True)
            source.write_text("source\n")
            (opencode / ".source_commit").write_text(commit + "\n")
            (opencode / "node_modules").mkdir()
            build = opencode / ".bench-build"
            build.mkdir()
            (build / "bench-cli.js").write_text("bench bundle\n")
            (build / "opencode.js").write_text("opencode bundle\n")
            bun = root / "bun"
            bun.write_text("bun\n")
            receipt = _build_opencode_setup_receipt(opencode, bun, repo, commit)
            return opencode, bun, receipt

        for mutation in ("source_addition", "missing_source_commit", "build_addition"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                opencode, bun, receipt = fixture(Path(tmp))
                if mutation == "source_addition":
                    (opencode / "added-source.ts").write_text("injected\n")
                elif mutation == "missing_source_commit":
                    (opencode / ".source_commit").unlink()
                else:
                    (opencode / ".bench-build" / "added.js").write_text("injected\n")
                self.assertFalse(
                    _opencode_receipt_matches_runtime(receipt, opencode, bun, repo, commit)
                )

    def test_receipt_authorization_pins_inputs_not_bundle_bytes(self):
        repo = "https://github.com/sdevare-nv/nv-opencode.git"
        commit = "0c088fd18b5ef6ff9b2a949ac2a41204b1ac8046"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opencode = root / "opencode"
            source = opencode / "packages" / "opencode" / "src" / "bench" / "cli.ts"
            source.parent.mkdir(parents=True)
            source.write_text("source\n")
            (opencode / ".source_commit").write_text(commit + "\n")
            (opencode / "node_modules").mkdir()
            build = opencode / ".bench-build"
            build.mkdir()
            (build / "bench-cli.js").write_text("bench bundle\n")
            (build / "opencode.js").write_text("opencode bundle\n")
            bun = root / "bun"
            bun.write_text("bun\n")
            receipt = _build_opencode_setup_receipt(opencode, bun, repo, commit)

            # Bundle BYTE drift (a rebuild on another node) stays authorized...
            bundle_drift = json.loads(json.dumps(receipt))
            for member in bundle_drift["build_tree"]["members"]:
                if member["relative_path"] == "opencode.js":
                    member["sha256"] = "0" * 64
                    member["size"] = 1
            bundle_drift["build_tree"]["sha256"] = _opencode_tree_digest(
                bundle_drift["build_tree"]["members"]
            )
            self.assertTrue(_opencode_receipt_is_authorized(bundle_drift, receipt))
            # ...but INPUT drift and build-tree SHAPE drift are rejected.
            bun_drift = json.loads(json.dumps(receipt))
            bun_drift["bun"]["sha256"] = "0" * 64
            self.assertFalse(_opencode_receipt_is_authorized(bun_drift, receipt))
            source_drift = json.loads(json.dumps(receipt))
            source_drift["source_tree"]["members"][0]["sha256"] = "0" * 64
            self.assertFalse(_opencode_receipt_is_authorized(source_drift, receipt))
            shape_drift = json.loads(json.dumps(receipt))
            shape_drift["build_tree"]["members"].append(
                {"relative_path": "evil.js", "type": "file", "mode": 420, "size": 1, "sha256": "0" * 64}
            )
            self.assertFalse(_opencode_receipt_is_authorized(shape_drift, receipt))

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
                self.assertEqual(
                    SWEBenchWrapper._find_container(wrapper, data_point), str(exact)
                )

            exact.unlink()
            (root / "prefix-task-a_amd64.sif").write_text("fuzzy\n")
            with patch.dict("os.environ", {"TB_TRUSTED_SIF_ROOT": str(root)}):
                with self.assertRaisesRegex(FileNotFoundError, "fuzzy fallback disabled"):
                    SWEBenchWrapper._find_container(wrapper, data_point)

    def test_only_final_dangling_tool_calls_get_explicit_terminal_results(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "old", "type": "function"}],
            },
            {"role": "assistant", "content": "continued"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "done", "type": "function"},
                    {"id": "dangling", "type": "function"},
                ],
            },
            {"role": "tool", "tool_call_id": "done", "content": "ok"},
        ]
        self.assertEqual(_append_terminal_tool_outputs(messages), ["dangling"])
        self.assertEqual(
            messages[-1],
            {
                "role": "tool",
                "tool_call_id": "dangling",
                "content": _TB_TERMINAL_TOOL_OUTPUT,
            },
        )

        terminal = [
            {"role": "user", "content": "repair"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "done", "type": "function"},
                    {"id": "dangling", "type": "function"},
                ],
            },
        ]
        self.assertEqual(
            _append_terminal_tool_outputs(terminal),
            ["done", "dangling"],
        )
        self.assertEqual(
            terminal[-1],
            {
                "role": "tool",
                "tool_call_id": "dangling",
                "content": _TB_TERMINAL_TOOL_OUTPUT,
            },
        )

    def test_agent_instance_dict_excludes_evaluator_fields(self):
        source = {
            "instance_id": "example",
            "workspace_path": "/app",
            "test_files": {"test.sh": "secret"},
            "test_files_b64": {"fixture.bin": "AA=="},
            "test_entrypoint": "test.sh",
            "tests_dir": "/tests",
            "verifier_timeout_sec": 30,
        }
        safe = _tb_agent_safe_instance_dict(source)
        self.assertEqual(safe, {"instance_id": "example", "workspace_path": "/app"})
        self.assertIn("test_files", source)

    def test_two_exec_commands_keep_tests_out_of_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = root / "results" / "task_run"
            persistent.mkdir(parents=True)
            private_tests = _tb_private_tests_host_dir(persistent)
            private_tests.mkdir(parents=True)
            private_verifier = _tb_private_verifier_host_dir(persistent)
            authoritative_dataset = private_tests.parent / "instance.jsonl"
            authoritative_dataset.write_text("{}\n")
            safe_dataset = persistent / "agent_instance.jsonl"
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
                        {
                            "allow_internet": False,
                            "workspace_path": "/app",
                            "tests_dir": "/tests",
                        }
                    ),
                },
                instance_dataset_path=authoritative_dataset,
                agent_instance_dataset_path=safe_dataset,
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
                apptainer_memory_limit_mb=32768,
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

    def test_terminal_bench_verifier_timeout_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            persistent = Path(tmp) / "results" / "task_run"
            persistent.mkdir(parents=True)
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
            self.assertTrue(_tb_private_verifier_host_dir(persistent).is_dir())
            self.assertIn("VERIFIER_STARTED_EPOCH", command.command)
            self.assertIn('REWARD_SOURCE="exit_code"', command.command)
            self.assertIn('REWARD_SOURCE="verifier_file"', command.command)
            self.assertIn('[ ! -L /logs/verifier/reward.txt ]', command.command)
            self.assertIn("export PYTHONDONTWRITEBYTECODE=1", command.command)
            self.assertLess(
                command.command.index("rm -rf -- /tests"),
                command.command.index("cp -a -- /root/tb_tests/bundle/. /tests/"),
            )


if __name__ == "__main__":
    unittest.main()
