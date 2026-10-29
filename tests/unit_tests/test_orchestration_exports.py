# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.script_templates import render_driver_entrypoint, render_export_tail
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.exports import effective_exports, export_install_extras, export_run_settings


COMPUTE = {"cluster": {"type": "slurm", "account": "acct", "hostname": "host"}}
KUBERNETES = {
    "k8s": {"type": "kubernetes", "namespace": "ns", "gpus_per_node": 8, "context": "ctx", "pvc_name": "pvc"}
}
SERVICE = {"container": "gym:latest", "type": "vllm", "model": "org/model"}
JOB_ID = "gym-job-20261023T101500Z-abc123"
BENCH_DIR = Path("/jobs") / JOB_ID / "gpqa"


def _config(
    *,
    job_exports: dict | None = None,
    benchmarks: dict | None = None,
    compute: dict | None = None,
    gym_install: bool = False,
) -> dict[str, Any]:
    driver: dict[str, Any] = {"container": "gym:latest", "benchmarks": benchmarks or {"gpqa": {}}}
    if gym_install:
        driver["gym_install"] = {"ref": "abc123"}
    return {
        "services": {"svc": SERVICE},
        "compute": compute or COMPUTE,
        "driver": driver,
        "job": {"output_path": "/jobs", **({"export": job_exports} if job_exports is not None else {})},
    }


def _submit(**kwargs: Any) -> SubmitConfig:
    return SubmitConfig.model_validate(_config(**kwargs))


class TestEffectiveExports:
    def test_a_job_level_export_applies_to_every_benchmark(self) -> None:
        config = _submit(job_exports={"evalrdl": {"label": "nightly"}}, benchmarks={"a": {}, "b": {}})

        assert effective_exports(config, "a") == {"evalrdl": {"label": "nightly"}}
        assert effective_exports(config, "b") == {"evalrdl": {"label": "nightly"}}

    def test_a_benchmark_overrides_the_job_level_settings_one_by_one(self) -> None:
        config = _submit(
            job_exports={"evalrdl": {"label": "nightly", "tags": ["a"], "meta": {"k": "1", "j": "2"}}},
            benchmarks={"gpqa": {"export": {"evalrdl": {"label": "gpqa", "meta": {"k": "9"}}}}, "other": {}},
        )

        assert effective_exports(config, "gpqa") == {
            "evalrdl": {"label": "gpqa", "tags": ["a"], "meta": {"k": "9", "j": "2"}}
        }
        assert effective_exports(config, "other")["evalrdl"]["label"] == "nightly"

    def test_null_drops_an_inherited_export_for_that_benchmark_only(self) -> None:
        config = _submit(
            job_exports={"evalrdl": {"label": "nightly"}},
            benchmarks={"gpqa": {"export": {"evalrdl": None}}, "other": {}},
        )

        assert effective_exports(config, "gpqa") == {}
        assert list(effective_exports(config, "other")) == ["evalrdl"]

    def test_a_benchmark_can_add_an_export_the_job_does_not_have(self) -> None:
        config = _submit(benchmarks={"gpqa": {"export": {"evalrdl": {"label": "x"}}}, "other": {}})

        assert list(effective_exports(config, "gpqa")) == ["evalrdl"]
        assert effective_exports(config, "other") == {}

    def test_a_command_benchmark_takes_none_of_the_job_level_defaults(self) -> None:
        config = _submit(
            job_exports={"evalrdl": {"label": "nightly"}}, benchmarks={"custom": {"command": "echo hi"}, "gpqa": {}}
        )

        assert effective_exports(config, "custom") == {}
        assert list(effective_exports(config, "gpqa")) == ["evalrdl"]

    def test_the_merge_does_not_change_the_config(self) -> None:
        config = _submit(
            job_exports={"evalrdl": {"meta": {"k": "1"}}},
            benchmarks={"gpqa": {"export": {"evalrdl": {"meta": {"j": "2"}}}}},
        )

        effective_exports(config, "gpqa")["evalrdl"]["meta"]["k"] = "changed"

        assert config.job.export == {"evalrdl": {"meta": {"k": "1"}}}
        assert config.driver.benchmarks["gpqa"].export == {"evalrdl": {"meta": {"j": "2"}}}


class TestRunSettings:
    @staticmethod
    def _settings(name: str, settings: dict[str, Any]) -> dict[str, Any]:
        return export_run_settings(name, settings, benchmark_name="gpqa", gym_job_id=JOB_ID, bench_dir=BENCH_DIR)

    def test_every_export_reads_the_benchmarks_rollouts_and_reports_to_its_own_result_file(self) -> None:
        settings = self._settings("evalrdl", {})

        assert settings["rollouts_jsonl_fpath"] == f"/jobs/{JOB_ID}/gpqa/artifacts/rollouts.jsonl"
        assert settings["result_json_fpath"] == f"/jobs/{JOB_ID}/gpqa/exports/evalrdl.result.json"

    def test_mlflow_and_wandb_runs_are_named_after_the_benchmark_and_the_submission(self) -> None:
        assert self._settings("mlflow", {})["mlflow_run_name"] == f"gpqa-{JOB_ID}"
        assert self._settings("wandb", {})["wandb_name"] == f"gpqa-{JOB_ID}"

    def test_atif_gets_a_session_id_and_an_output_folder_beside_the_result(self) -> None:
        settings = self._settings("atif", {"agent_version": "1.0.0"})

        assert settings["session_id"] == "gym-submit-gpqa-20261023T101500Z-abc123"
        assert settings["output_dirpath"] == f"/jobs/{JOB_ID}/gpqa/exports/atif"
        assert settings["agent_version"] == "1.0.0"

    def test_evalrdl_metadata_names_the_benchmark(self) -> None:
        assert self._settings("evalrdl", {})["meta"] == {"benchmark": "gpqa"}
        assert self._settings("evalrdl", {"meta": {"benchmark": "mine", "k": "v"}})["meta"] == {
            "benchmark": "mine",
            "k": "v",
        }

    def test_names_the_submit_config_sets_are_kept(self) -> None:
        assert self._settings("mlflow", {"mlflow_run_name": "mine"})["mlflow_run_name"] == "mine"
        assert self._settings("atif", {"session_id": "s", "output_dirpath": "/out"})["output_dirpath"] == "/out"

    def test_the_settings_passed_in_are_not_changed(self) -> None:
        original = {"meta": {"k": "v"}}

        self._settings("evalrdl", original)

        assert original == {"meta": {"k": "v"}}

    def test_only_in_process_exporters_need_install_extras(self) -> None:
        assert export_install_extras(["atif", "evalrdl", "mlflow", "wandb"]) == ("mlflow", "wandb")


class TestValidation:
    def test_valid_exports_are_accepted(self) -> None:
        _submit(
            job_exports={
                "evalrdl": {"label": "nightly", "tags": ["a"], "url": "stg"},
                "atif": {"agent_version": "1.0.0"},
                "mlflow": {"mlflow_tracking_uri": "https://mlflow.example", "mlflow_experiment_name": "gym"},
                "wandb": {"wandb_project": "p", "wandb_api_key": "${oc.env:WANDB_API_KEY}"},
            }
        )

    def test_an_unknown_exporter_is_rejected_with_the_choices(self) -> None:
        with pytest.raises(ValidationError, match=r"Benchmark 'gpqa' export 'nope': unknown exporter.*atif"):
            _submit(job_exports={"nope": {}})

    def test_an_unknown_setting_is_rejected_with_the_known_ones(self) -> None:
        with pytest.raises(ValidationError, match=r"unknown settings lable\. Known settings:.*label"):
            _submit(job_exports={"evalrdl": {"lable": "typo"}})

    @pytest.mark.parametrize("setting", ["rollouts_jsonl_fpath", "result_json_fpath", "exporter"])
    def test_settings_the_orchestrator_owns_are_rejected(self, setting: str) -> None:
        with pytest.raises(ValidationError, match=f"{setting} are set per benchmark by the orchestrator"):
            _submit(job_exports={"evalrdl": {setting: "x"}})

    def test_a_setting_of_the_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"invalid settings \(parallel"):
            _submit(job_exports={"evalrdl": {"parallel": 0}})

    def test_atif_needs_an_agent_version(self) -> None:
        with pytest.raises(ValidationError, match=r"export 'atif': invalid settings \(agent_version"):
            _submit(job_exports={"atif": {}})

    def test_mlflow_needs_a_tracking_uri_and_experiment(self) -> None:
        with pytest.raises(ValidationError, match=r"export 'mlflow': the settings are incomplete"):
            _submit(job_exports={"mlflow": {"mlflow_tracking_uri": "https://mlflow.example"}})

    def test_wandb_export_without_a_key_setting_uses_the_environment(self) -> None:
        config = _submit(job_exports={"wandb": {"wandb_project": "p"}})

        assert effective_exports(config, "gpqa")["wandb"] == {"wandb_project": "p"}

    @pytest.mark.parametrize("key", ["", "****"])
    def test_wandb_export_rejects_an_empty_or_masked_key(self, key: str) -> None:
        with pytest.raises(ValidationError, match=r"export 'wandb': the settings are incomplete"):
            _submit(job_exports={"wandb": {"wandb_project": "p", "wandb_api_key": key}})

    def test_wandb_export_still_needs_a_project(self) -> None:
        with pytest.raises(ValidationError, match=r"export 'wandb': the settings are incomplete"):
            _submit(job_exports={"wandb": {}})

    def test_wandb_export_without_a_key_is_still_refused_next_to_the_same_live_exporter(self) -> None:
        run = {"wandb_project": "p", "wandb_name": "n", "wandb_api_key": "k"}  # pragma: allowlist secret
        with pytest.raises(ValidationError, match="already enable the live wandb exporter"):
            _submit(benchmarks={"gpqa": {"run": run}}, job_exports={"wandb": {"wandb_project": "p"}})

    def test_exports_on_a_command_benchmark_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sets both `command` and `export`"):
            _submit(benchmarks={"custom": {"command": "echo hi", "export": {"evalrdl": {}}}})

    def test_dropping_an_export_on_a_command_benchmark_is_not_an_export(self) -> None:
        _submit(benchmarks={"custom": {"command": "echo hi", "export": {"evalrdl": None}}})

    def test_the_live_exporter_and_the_same_export_cannot_both_be_on(self) -> None:
        run = {"wandb_project": "p", "wandb_name": "n", "wandb_api_key": "k"}  # pragma: allowlist secret
        with pytest.raises(ValidationError, match="already enable the live wandb exporter"):
            _submit(
                benchmarks={"gpqa": {"run": run}},
                job_exports={"wandb": {"wandb_project": "p", "wandb_api_key": "k"}},  # pragma: allowlist secret
            )

    def test_a_live_exporter_and_a_different_export_can_be_combined(self) -> None:
        run = {"wandb_project": "p", "wandb_name": "n", "wandb_api_key": "k"}  # pragma: allowlist secret
        _submit(benchmarks={"gpqa": {"run": run}}, job_exports={"evalrdl": {}})

    def test_kubernetes_compute_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not supported on Kubernetes compute yet"):
            _submit(job_exports={"evalrdl": {}}, compute=KUBERNETES)

    def test_kubernetes_compute_without_exports_is_unaffected(self) -> None:
        _submit(compute=KUBERNETES)


def _script(config: SubmitConfig, name: str = "gpqa") -> str:
    compute = next(iter(config.compute.values()))
    return build_sbatch_script(config, name, config.driver.benchmarks[name], compute, BENCH_DIR)


class TestSbatchScript:
    def test_a_benchmark_without_exports_keeps_exec_and_has_no_tail(self) -> None:
        script = _script(_submit(gym_install=True))

        assert 'exec "$@"' in script
        assert "gym eval export" not in script
        assert "eval.exit" not in script

    def test_exports_run_after_the_eval_in_the_same_step_with_the_derived_settings(self) -> None:
        script = _script(_submit(gym_install=True, job_exports={"evalrdl": {"label": "nightly", "tags": ["a", "b"]}}))

        assert 'exec "$@"' not in script
        assert "gym eval export evalrdl" in script
        for expected in (
            "+label=nightly",
            "+tags=[a,b]",
            f"+rollouts_jsonl_fpath={BENCH_DIR}/artifacts/rollouts.jsonl",
            f"+result_json_fpath={BENCH_DIR}/exports/evalrdl.result.json",
            "+meta.benchmark=gpqa",
        ):
            assert expected in script
        assert script.index("gym eval export evalrdl") > script.index("uv pip install")

    def test_every_effective_export_gets_a_command_in_order(self) -> None:
        script = _script(_submit(gym_install=True, job_exports={"atif": {"agent_version": "1"}, "evalrdl": {}}))

        assert script.index("gym eval export atif") < script.index("gym eval export evalrdl")

    @staticmethod
    def _install_extras(script: str) -> list[str]:
        match = re.search(r'uv pip install -e "\.\[([^\]]*)\]"', script)
        return match.group(1).split(",") if match else []

    def test_in_process_exporters_add_their_extra_to_the_gym_install_once(self) -> None:
        config = _submit(
            gym_install=True,
            job_exports={"mlflow": {"mlflow_tracking_uri": "https://m", "mlflow_experiment_name": "e"}},
        )

        assert self._install_extras(_script(config)).count("mlflow") == 1

    def test_an_isolated_exporter_adds_no_extra(self) -> None:
        extras = self._install_extras(_script(_submit(gym_install=True, job_exports={"evalrdl": {}})))

        assert not {"evalrdl", "mlflow", "wandb"} & set(extras)

    def test_the_tail_is_used_even_when_the_driver_installs_and_prepares_nothing(self) -> None:
        script = _script(_submit(job_exports={"evalrdl": {}}))

        assert "gym eval export evalrdl" in script
        assert "bash -c" in script

    def test_a_command_benchmark_script_has_no_tail(self) -> None:
        config = _submit(job_exports={"evalrdl": {}}, benchmarks={"custom": {"command": "echo hi"}})

        assert "gym eval export" not in _script(config, "custom")


# A real bash run of the tail, with a `gym` that records what it is asked to export.
FAKE_GYM = """#!/bin/bash
echo "$@" >> "$FAKE_GYM_LOG"
printf '%s\\0' "$@" > "$FAKE_GYM_LOG.args"
if [ "$1 $2" = "eval export" ] && [ "${FAKE_EXPORT_FAILS:-}" = "$3" ]; then exit 1; fi
"""


@pytest.fixture
def tail_env(tmp_path: Path) -> dict[str, Any]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gym").write_text(FAKE_GYM)
    (bin_dir / "gym").chmod(0o755)
    run = tmp_path / "run.sh"
    run.write_text('#!/bin/bash\necho run >> "$FAKE_GYM_LOG"\nexit "${FAKE_RUN_RC:-0}"\n')
    run.chmod(0o755)
    bench = tmp_path / "bench"
    bench.mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_GYM_LOG": str(tmp_path / "calls.log")}
    return {"bench": bench, "run": run, "env": env, "log": tmp_path / "calls.log"}


def _run_tail(
    tail_env: dict[str, Any], exports: list[tuple[str, list[str]]], **env: str
) -> subprocess.CompletedProcess:
    tail = render_export_tail(tail_env["bench"], exports)
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{tail}", "--", str(tail_env["run"])],
        env={**tail_env["env"], **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _calls(tail_env: dict[str, Any]) -> list[str]:
    return tail_env["log"].read_text().splitlines() if tail_env["log"].exists() else []


class TestExportTail:
    def test_a_successful_run_is_followed_by_each_export_in_order(self, tail_env: dict[str, Any]) -> None:
        completed = _run_tail(tail_env, [("atif", ["+a=1"]), ("evalrdl", ["+label=x"])])

        assert completed.returncode == 0
        assert _calls(tail_env) == ["run", "eval export atif +a=1", "eval export evalrdl +label=x"]
        assert (tail_env["bench"] / "eval.exit").read_text().strip() == "0"
        assert (tail_env["bench"] / "exports").is_dir()

    def test_a_failed_run_is_not_exported_and_its_exit_code_is_the_scripts(self, tail_env: dict[str, Any]) -> None:
        completed = _run_tail(tail_env, [("evalrdl", [])], FAKE_RUN_RC="3")

        assert completed.returncode == 3
        assert _calls(tail_env) == ["run"]
        assert (tail_env["bench"] / "eval.exit").read_text().strip() == "3"

    def test_a_failing_export_neither_changes_the_exit_code_nor_stops_the_others(
        self, tail_env: dict[str, Any]
    ) -> None:
        completed = _run_tail(tail_env, [("atif", []), ("evalrdl", [])], FAKE_EXPORT_FAILS="atif")

        assert completed.returncode == 0
        assert _calls(tail_env) == ["run", "eval export atif", "eval export evalrdl"]

    def test_the_run_is_not_exec_d_so_arguments_with_quotes_survive(self, tail_env: dict[str, Any]) -> None:
        completed = _run_tail(tail_env, [("evalrdl", ["'+label=a b'"])])

        assert completed.returncode == 0
        assert _calls(tail_env)[-1] == "eval export evalrdl +label=a b"

    def test_term_reaches_the_run_which_decides_the_exit_code_and_nothing_is_exported(
        self, tail_env: dict[str, Any]
    ) -> None:
        tail_env["run"].write_text(
            "#!/bin/bash\ntrap 'echo got-term >> \"$FAKE_GYM_LOG\"; exit 143' TERM\n"
            'echo started >> "$FAKE_GYM_LOG"\nwhile true; do sleep 0.1; done\n'
        )
        tail = render_export_tail(tail_env["bench"], [("evalrdl", [])])
        process = subprocess.Popen(
            ["bash", "-c", f"set -euo pipefail\n{tail}", "--", str(tail_env["run"])], env=tail_env["env"]
        )
        deadline = time.monotonic() + 10
        while "started" not in _calls(tail_env) and time.monotonic() < deadline:
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 143

        assert _calls(tail_env) == ["started", "got-term"]
        assert (tail_env["bench"] / "eval.exit").read_text().strip() == "143"


def test_the_entrypoint_hands_the_run_to_the_tail_instead_of_exec() -> None:
    entrypoint = render_driver_entrypoint("https://repo", "ref", None, export_tail="TAIL-MARKER")

    assert "TAIL-MARKER" in entrypoint
    assert 'exec "$@"' not in entrypoint
    assert "TAIL-MARKER" not in render_driver_entrypoint("https://repo", "ref", None)


def _parse_overrides(args: list[str]) -> dict[str, Any]:
    """What Hydra makes of `args`, nested back into a dict the way the exporter's settings are."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    settings: dict[str, Any] = {}
    for override in OverridesParser.create().parse_overrides(args):
        *path, leaf = override.key_or_group.split(".")
        node = settings
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = override.value()
    return settings


AWKWARD = {
    "agent_version": "1",
    "float_like": "1.0",
    "bool_like": "true",
    "null_like": "null",
    "label": "nightly(run)",
    "owner": "o'brien",
    "spaced": "nightly run",
    "padded": " padded ",
    "empty": "",
    "listy": "a,b",
    "bracket": "x]y",
    "backslash": "C:\\path\\",
    "unicode": "caf\u00e9",
    "interpolation": "${oc.env:EVALRDL_URL}",
    "tags": ["1", "true", "a b", "x,y", "o'brien"],
    "count": 3,
    "ratio": 2.5,
    "enabled": True,
    "nothing": None,
    "meta": {"benchmark": "gpqa", "owner": "o'brien", "version": "2"},
}


def test_export_settings_reach_hydra_with_their_values_and_types() -> None:
    from nemo_gym.orchestration.executors.utils import flatten_export_args

    tokens = [shlex.split(arg)[0] for arg in flatten_export_args(AWKWARD)]

    assert _parse_overrides(tokens) == AWKWARD
    assert isinstance(_parse_overrides(tokens)["agent_version"], str)


def test_the_arguments_of_a_submitted_export_validate_against_the_exporters_own_model() -> None:
    from nemo_gym.exporters import load_run_exporter_config_model
    from nemo_gym.orchestration.executors.utils import flatten_export_args

    for name, settings, field, expected in (
        ("atif", {"agent_version": "1"}, "agent_version", "1"),
        ("evalrdl", {"label": "nightly(run)", "tags": ["1", "a b"]}, "label", "nightly(run)"),
    ):
        full = export_run_settings(name, settings, benchmark_name="gpqa", gym_job_id=JOB_ID, bench_dir=BENCH_DIR)
        tokens = [shlex.split(arg)[0] for arg in flatten_export_args(full)]

        validated = load_run_exporter_config_model(name).model_validate(_parse_overrides(tokens))

        assert getattr(validated, field) == expected, name


@pytest.mark.parametrize(
    "settings",
    [{"tags": [{"a": 1}]}, {"bad key": "x"}, {"meta": {"a=b": "x"}}, {"": "x"}],
)
def test_export_settings_that_cannot_be_written_as_overrides_are_refused(settings: dict[str, Any]) -> None:
    from nemo_gym.orchestration.executors.utils import flatten_export_args

    with pytest.raises(ValueError, match="cannot be written as a Hydra override"):
        flatten_export_args(settings)


def test_the_rendered_entrypoint_passes_awkward_settings_to_gym_eval_export_unchanged(
    tail_env: dict[str, Any],
) -> None:
    from nemo_gym.orchestration.executors.utils import flatten_export_args

    settings = export_run_settings(
        "evalrdl",
        {
            "label": "nightly(run)",
            "tags": ["a b", "1"],
            "meta": {"owner": "o'brien", "version": "2"},
            "url": "${oc.env:EVALRDL_URL}",
        },
        benchmark_name="gpqa",
        gym_job_id=JOB_ID,
        bench_dir=tail_env["bench"],
    )
    tail = render_export_tail(tail_env["bench"], [("evalrdl", flatten_export_args(settings))])
    entrypoint = render_driver_entrypoint(None, None, None, export_tail=tail)

    completed = subprocess.run(
        ["bash", "-c", f'GYM_CMD=("{tail_env["run"]}")\n{entrypoint}'],
        env=tail_env["env"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    received = Path(f"{tail_env['log']}.args").read_bytes().decode().split("\0")[:-1]
    assert received[:3] == ["eval", "export", "evalrdl"]
    assert _parse_overrides(received[3:]) == {
        "label": "nightly(run)",
        "tags": ["a b", "1"],
        "meta": {"owner": "o'brien", "version": "2", "benchmark": "gpqa"},
        "url": "${oc.env:EVALRDL_URL}",
        "rollouts_jsonl_fpath": f"{tail_env['bench']}/artifacts/rollouts.jsonl",
        "result_json_fpath": f"{tail_env['bench']}/exports/evalrdl.result.json",
    }
