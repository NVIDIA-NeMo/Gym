# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CICD_MAIN_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cicd-main.yml"
CLASSIFY_CHANGES_ACTION = REPO_ROOT / ".github" / "actions" / "classify-changes" / "action.yml"
FULL_TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "full-test-suite.yml"
GPU_E2E_CONFIG = REPO_ROOT / "tests" / "e2e" / "gpu_e2e.yaml"
GPU_E2E_DATASET = REPO_ROOT / "tests" / "e2e" / "gpu_smoke.jsonl"
GPU_E2E_VERIFIER = REPO_ROOT / "tests" / "e2e" / "verify_gpu_rollout.py"
GPU_E2E_SCRIPT = REPO_ROOT / "tests" / "e2e" / "gpu_e2e_test.sh"
INFERENCE_PROVIDER_E2E_SCRIPT = REPO_ROOT / "tests" / "e2e" / "run_inference_provider_e2e.sh"
INFERENCE_PROVIDER_ROLLOUT_VERIFIER = REPO_ROOT / "tests" / "e2e" / "verify_inference_provider_rollout.py"
GITLAB_PIPELINE = REPO_ROOT / ".gitlab-ci.yml"
IS_RETRYABLE_FULL_SUITE_FAILURE = REPO_ROOT / "scripts" / "ci" / "is_retryable_full_suite_failure.sh"
RECLAIM_RUNNER_DISK = REPO_ROOT / "scripts" / "ci" / "reclaim_runner_disk.sh"
RETRY_FULL_TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "retry-full-test-suite.yml"
SANITIZER = REPO_ROOT / "scripts" / "ci" / "sanitize_env.sh"
SERVER_TESTS = REPO_ROOT / "scripts" / "ci" / "server_tests.sh"
SETUP_DEV = REPO_ROOT / "scripts" / "ci" / "setup_dev.sh"
TEST_TEMPLATE_ACTION = REPO_ROOT / ".github" / "actions" / "test-template" / "action.yml"
UNIT_TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "unit-tests.yml"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

BEHAVIOR_CHANGING_ENV = {
    "GYM_CI_DEV_VENV_DIR": "/tmp/injected-driver-venv",
    "SKIP": "ruff",
    "NEMO_GYM_EXTRA_ROOTS": "/tmp/external-gym",
    "NEMO_GYM_CONFIG_DICT": '{"search_dir": "/tmp/external-gym"}',
    "NEMO_GYM_ALLOW_PRERELEASE": "true",
    "PYTHONPATH": "/tmp/python",
    "PYTHONSAFEPATH": "1",
    "PYTEST_ADDOPTS": "-m injected-selection",
}


def _environment_after_sanitizing(stage: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(BEHAVIOR_CHANGING_ENV)
    env["GYM_CI_PRESERVED_SENTINEL"] = "preserved"
    command = f'source "{SANITIZER}"; gym_ci_sanitize_environment "{stage}"; env -0'
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        env=env,
    )
    return {
        key.decode(): value.decode()
        for item in result.stdout.split(b"\0")
        if item
        for key, value in [item.split(b"=", 1)]
    }


@pytest.mark.parametrize(
    ("stage", "removed"),
    [
        ("lint", {"SKIP"}),
        (
            "core",
            {"GYM_CI_DEV_VENV_DIR", "NEMO_GYM_EXTRA_ROOTS", "NEMO_GYM_CONFIG_DICT", "PYTHONPATH"},
        ),
        (
            "server",
            {
                "GYM_CI_DEV_VENV_DIR",
                "NEMO_GYM_EXTRA_ROOTS",
                "NEMO_GYM_CONFIG_DICT",
                "NEMO_GYM_ALLOW_PRERELEASE",
                "PYTHONPATH",
                "PYTHONSAFEPATH",
                "PYTEST_ADDOPTS",
            },
        ),
    ],
)
def test_ci_stage_removes_only_its_behavior_changing_environment(stage: str, removed: set[str]) -> None:
    sanitized = _environment_after_sanitizing(stage)

    assert removed.isdisjoint(sanitized)
    assert sanitized["GYM_CI_PRESERVED_SENTINEL"] == "preserved"
    for name, value in BEHAVIOR_CHANGING_ENV.items():
        if name not in removed:
            assert sanitized[name] == value


def test_ci_environment_sanitizer_rejects_unknown_stage() -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{SANITIZER}"; gym_ci_sanitize_environment unknown'],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown Gym CI stage: unknown" in result.stderr


@pytest.mark.parametrize("venv_dir", ["relative-driver-venv", "/"])
def test_setup_dev_rejects_unsafe_driver_venv(venv_dir: str) -> None:
    env = os.environ.copy()
    env["GYM_CI_DEV_VENV_DIR"] = venv_dir

    result = subprocess.run([str(SETUP_DEV)], capture_output=True, text=True, env=env)

    assert result.returncode == 2
    assert f"GYM_CI_DEV_VENV_DIR must be an absolute non-root path: {venv_dir}" in result.stderr


def test_gitlab_adapter_selects_current_contract_version() -> None:
    expected_version = (REPO_ROOT / "scripts" / "ci" / "contract-version").read_text().strip()

    assert f'GYM_CI_CONTRACT_VERSION: "{expected_version}"' in GITLAB_PIPELINE.read_text()


def test_gitlab_adapter_selects_cpu_short_partition() -> None:
    assert 'GYM_SLURM_PARTITION: "cpu_short"' in GITLAB_PIPELINE.read_text()


@pytest.mark.parametrize(
    "failure",
    [
        "fatal: server certificate verification failed. CAfile: none CRLfile: none",
        "fatal: unable to access repository: Could not resolve host: github.com",
        "error: No space left on device (os error 28)",
        "Insufficient runner disk space: 1024 KiB available; 10485760 KiB required",
        "The hosted runner lost communication with the server",
    ],
)
def test_full_suite_failure_classifier_accepts_infrastructure_failures(tmp_path: Path, failure: str) -> None:
    log_path = tmp_path / "failed.log"
    log_path.write_text(failure)

    result = subprocess.run([str(IS_RETRYABLE_FULL_SUITE_FAILURE), str(log_path)], capture_output=True, text=True)

    assert result.returncode == 0
    assert "recognized runner or transport failure" in result.stdout


@pytest.mark.parametrize(
    "failure",
    [
        "FAILED tests/unit_tests/test_cli.py::test_command - AssertionError",
        "No matching distribution found for deterministic-package==1.2.3",
        "ruff check failed: unused import",
    ],
)
def test_full_suite_failure_classifier_rejects_deterministic_failures(tmp_path: Path, failure: str) -> None:
    log_path = tmp_path / "failed.log"
    log_path.write_text(failure)

    result = subprocess.run([str(IS_RETRYABLE_FULL_SUITE_FAILURE), str(log_path)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "do not contain a recognized" in result.stdout


def test_full_suite_retry_is_bounded_and_reruns_only_failed_jobs() -> None:
    workflow = RETRY_FULL_TEST_WORKFLOW.read_text()

    assert "github.event.workflow_run.run_attempt == 1" in workflow
    assert "head_repository.full_name == github.repository" in workflow
    assert "head_branch == github.event.repository.default_branch" in workflow
    assert "is_retryable_full_suite_failure.sh" in workflow
    assert "rerun-failed-jobs" in workflow
    assert 'select(.conclusion == "failure" and .name != "Server suite")' in workflow
    assert "At least one failed job was deterministic" in workflow
    assert "actions: write" in workflow


def test_github_full_test_jobs_reclaim_disk_before_dependency_restore() -> None:
    for workflow_path, expected_jobs in [(FULL_TEST_WORKFLOW, 2), (UNIT_TEST_WORKFLOW, 2)]:
        workflow = workflow_path.read_text()
        assert workflow.count("run: ./scripts/ci/reclaim_runner_disk.sh") == expected_jobs
        assert workflow.count("Reclaim runner disk") == expected_jobs
        sections = workflow.split("- name: Reclaim runner disk")
        for section in sections[1:]:
            assert section.index("reclaim_runner_disk.sh") < section.index("Cache uv dependencies")


def test_cicd_main_wires_preflight_cpu_and_gpu_workflows() -> None:
    workflow = CICD_MAIN_WORKFLOW.read_text()
    results_path = (
        "${{ runner.temp }}/nemo-gym-gpu-e2e/"
        "${{ github.run_id }}-${{ github.run_attempt }}/${{ matrix.artifact_name }}"
    )

    assert "      - main\n" in workflow
    assert '      - "pull-request/[0-9]+"\n' in workflow
    assert "deploy-release" not in workflow
    assert "schedule:" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert "  contents: read\n" in workflow
    assert "  pull-requests: read\n" in workflow
    assert "id-token:" not in workflow
    assert "uses: ./.github/workflows/unit-tests.yml" in workflow
    assert workflow.count("base-ref: ${{ needs.pre-flight.outputs.base_ref }}") == 2
    assert "uses: ./.github/actions/classify-changes" in workflow
    assert "uses: ./.github/actions/test-template" in workflow
    assert "base-ref: ${{ needs.pre-flight.outputs.base_ref }}" in workflow
    assert "needs: [pre-flight, classify_changes, unit_tests]" in workflow
    assert "needs: [pre-flight, classify_changes, unit_tests, container_build]" in workflow
    assert workflow.count("if: needs.classify_changes.outputs.docs_only != 'true'") == 4
    assert "needs.pre-flight.outputs.docs_only" not in workflow
    assert "runs-on: ${{ needs.pre-flight.outputs.runner_prefix }}" in workflow
    assert "matrix:" in workflow
    assert "script: ${{ matrix.script }}" in workflow
    assert "test-type: ${{ matrix.test_type }}" in workflow
    assert "test-data-path: ${{ needs.pre-flight.outputs.test_data_path }}" in workflow
    assert "container-image: ${{ needs.container_build.outputs.image }}" in workflow
    assert "model: ${{ matrix.model }}" in workflow
    assert "model-revision: ${{ matrix.model_revision }}" in workflow
    assert "hf-cache-path:" not in workflow
    assert f"results-path: {results_path}" in workflow
    assert f"path: {results_path}" in workflow


def test_cicd_container_build_pushes_sha_image_after_unit_tests() -> None:
    workflow = CICD_MAIN_WORKFLOW.read_text()

    assert "name: Build Gym container" in workflow
    assert "runs-on: ${{ needs.pre-flight.outputs.runner_prefix }}" in workflow
    assert "uses: docker/setup-buildx-action@8d2750c68a42422c14e847fe6c8ac0403b4cbd6f" in workflow
    assert "uses: docker/build-push-action@ca052bb54ab0790a636c9b5f226502c73d547a25" in workflow
    assert "build-contexts: nemo-gym=." in workflow
    assert "target: release" in workflow
    assert "push: true" in workflow
    assert 'echo "image=$REGISTRY/gym:$GITHUB_SHA"' in workflow
    assert "cache-from: type=registry" in workflow
    assert "cache-to: type=registry" in workflow
    assert "NEMO_GYM_PREFETCH_CONFIGS=tests/e2e/gpu_e2e.yaml" in workflow


def test_cicd_summary_accepts_only_expected_docs_only_skips() -> None:
    workflow = CICD_MAIN_WORKFLOW.read_text()

    assert (
        "needs: [pre-flight, classify_changes, unit_tests, container_build, gpu_e2e_tests, provider_e2e_tests]"
        in workflow
    )
    assert "if: always() && !cancelled()" in workflow
    assert '"$PREFLIGHT_RESULT" != "success"' in workflow
    assert '"$CLASSIFY_RESULT" != "success"' in workflow
    assert '"$DOCS_ONLY" == "true"' in workflow
    assert '"$CONTAINER_BUILD_RESULT" == "skipped"' in workflow
    assert '"$CONTAINER_BUILD_RESULT" == "success"' in workflow
    assert "PROVIDER_E2E_TEST_RESULT: ${{ needs.provider_e2e_tests.result }}" in workflow
    assert 'echo "Provider E2E tests: $PROVIDER_E2E_TEST_RESULT"' in workflow
    assert '"$PROVIDER_E2E_TEST_RESULT" == "skipped"' not in workflow
    assert '"$PROVIDER_E2E_TEST_RESULT" == "success"' not in workflow


def test_shared_change_classifier_matches_gym_docs_and_server_paths() -> None:
    action = CLASSIFY_CHANGES_ACTION.read_text()
    unit_workflow = UNIT_TEST_WORKFLOW.read_text()

    for path in ("**.md", "fern/**", "LICENSE", "benchmarks/**"):
        assert path in action
    for path in ("resources_servers/**", "responses_api_agents/**", "responses_api_models/**"):
        assert path in action

    assert "uses: ./.github/actions/classify-changes" in unit_workflow
    assert "base-ref: ${{ inputs.base-ref || github.event.pull_request.base.sha || '' }}" in unit_workflow
    assert "force-run-all: ${{ inputs.base-ref == '' && github.event_name != 'pull_request' }}" in unit_workflow
    assert "gh pr view" not in action
    assert "DOCS_ONLY_LABEL" not in action


def test_test_template_runs_cpu_or_gpu_script_in_container() -> None:
    action = TEST_TEMPLATE_ACTION.read_text()

    assert '        case "$TEST_TYPE" in' in action
    assert "          cpu)" in action
    assert "          gpu)" in action
    assert "gpu_args=(--runtime=nvidia --gpus all)" in action
    assert 'docker pull "$CONTAINER_IMAGE"' in action
    assert '--volume "$TEST_DATA_PATH:/home/TestData"' in action
    assert '--env "TEST_DATA_PATH=/home/TestData"' in action
    assert '--env "HF_HOME=/home/TestData/HF_HOME"' in action
    assert '--volume "$RESULTS_PATH:$CONTAINER_RESULTS_DIR"' in action
    assert '--env "MODEL=$MODEL"' in action
    assert '--env "MODEL_REVISION=$MODEL_REVISION"' in action
    assert 'TEST_DATA_PATH="$(prepare_mount_source test-data-path "$TEST_DATA_PATH")"' in action
    assert '-e -u -o pipefail "$TEST_SCRIPT"' in action
    assert "continue-on-error: true" in action
    assert "      if: always()" in action
    assert "TEST_OUTCOME: ${{ steps.test.outcome }}" in action
    assert 'echo "::notice title=Test result::$TEST_SCRIPT — PASSED"' in action
    assert 'echo "::error title=Test result::$TEST_SCRIPT — FAILED"' in action
    assert '} >> "$GITHUB_STEP_SUMMARY"' in action
    assert "┌─ launching test ─" in action
    assert "║   ✅  PASSED" in action
    assert "║   ❌  FAILED" in action
    assert action.count("required: true") == 4


def test_gpu_e2e_matrix_uses_qwen_smoke_config() -> None:
    workflow = CICD_MAIN_WORKFLOW.read_text()
    config = GPU_E2E_CONFIG.read_text()
    dataset = json.loads(GPU_E2E_DATASET.read_text())

    assert "fail-fast: false" in workflow
    assert "- name: GPU E2E - Qwen vLLM rollout" in workflow
    assert "script: ./tests/e2e/gpu_e2e_test.sh" in workflow
    assert "test_type: gpu" in workflow

    assert "resources_servers/string_match/configs/string_match.yaml" in config
    assert "responses_api_models/vllm_model/configs/vllm_model.yaml" in config
    assert "skip_venv_if_present: true" in config
    assert QWEN_MODEL in config
    assert dataset["expected_answer"] == "Paris"
    assert dataset["extraction_mode"] == "final_answer"
    assert dataset["case_sensitive"] is False


def _valid_gpu_rollout() -> dict:
    return {
        "response": {
            "status": "completed",
            "error": None,
            "model": QWEN_MODEL,
            "usage": {"input_tokens": 12, "output_tokens": 6},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Final answer: Paris"}],
                }
            ],
        },
        "reward": 1.0,
        "expected_answer": "Paris",
        "extracted_answer": "Paris",
        "agent_ref": {"name": "string_match_simple_agent"},
    }


def _run_gpu_rollout_verifier(tmp_path: Path, rollout: dict) -> subprocess.CompletedProcess[str]:
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollouts_path.write_text(json.dumps(rollout) + "\n")
    return subprocess.run(
        [
            sys.executable,
            str(GPU_E2E_VERIFIER),
            "--rollouts",
            str(rollouts_path),
            "--expected-model",
            QWEN_MODEL,
            "--expected-answer",
            "Paris",
        ],
        capture_output=True,
        text=True,
    )


def test_gpu_e2e_verifier_accepts_successful_qwen_rollout(tmp_path: Path) -> None:
    result = _run_gpu_rollout_verifier(tmp_path, _valid_gpu_rollout())

    assert result.returncode == 0, result.stderr


def test_gpu_e2e_verifier_accepts_case_insensitive_reward(tmp_path: Path) -> None:
    rollout = _valid_gpu_rollout()
    rollout["response"]["output"][0]["content"][0]["text"] = "Final answer: paris"
    rollout["extracted_answer"] = "paris"

    result = _run_gpu_rollout_verifier(tmp_path, rollout)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", ["wrong-model", "zero-reward"])
def test_gpu_e2e_verifier_rejects_vacuous_rollout(tmp_path: Path, failure: str) -> None:
    rollout = _valid_gpu_rollout()
    if failure == "wrong-model":
        rollout["response"]["model"] = "unexpected/model"
    else:
        rollout["reward"] = 0.0

    result = _run_gpu_rollout_verifier(tmp_path, rollout)

    assert result.returncode != 0


def test_provider_e2e_matrix_selects_config_model_and_secret_by_name() -> None:
    workflow = CICD_MAIN_WORKFLOW.read_text()

    assert "provider_e2e_tests:" in workflow
    assert "name: ${{ matrix.provider }}-e2e" in workflow
    assert "timeout-minutes: 20" in workflow
    assert "provider: fireworks" in workflow
    assert "config: responses_api_models/inference_provider/configs/fireworks.yaml" in workflow
    assert "model: accounts/fireworks/models/gpt-oss-20b" in workflow
    assert "model_api_key_secret_name: FIREWORKS" in workflow
    assert "MODEL_API_KEY: ${{ secrets[matrix.model_api_key_secret_name] }}" in workflow
    assert workflow.count("secrets[matrix.model_api_key_secret_name]") == 1
    assert "model-api-key:" not in workflow
    assert "needs: [classify_changes, unit_tests]" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "UV_VERSION=\"$(sed -n 's/^ARG UV_VERSION=//p' docker/Dockerfile)\"" in workflow
    assert 'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh' in workflow
    assert "https://astral.sh/uv/0.11.29/install.sh" not in workflow
    assert "E2E_MODEL: ${{ matrix.model }}" in workflow
    assert "E2E_PROVIDER_CONFIG: ${{ matrix.config }}" in workflow
    assert "run: bash tests/e2e/run_inference_provider_e2e.sh" in workflow

    script = INFERENCE_PROVIDER_E2E_SCRIPT.read_text()
    assert "E2E_PROVIDER_CONFIG" in script
    assert "E2E_MODEL" in script
    assert "MODEL_API_KEY" in script
    assert "--model-api-key" not in script
    assert "resources_servers/example_single_tool_call/data/example.jsonl" in script
    assert "--max-output-tokens 4096" in script

    env_config = (REPO_ROOT / "tests" / "e2e" / "inference_provider_env.yaml").read_text()
    assert "max_steps: 2" in env_config


def _valid_inference_provider_rollout() -> dict:
    return {
        "reward": 1.0,
        "response": {
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city": "San Francisco"}',
                    "call_id": "weather-call",
                },
                {
                    "type": "function_call_output",
                    "call_id": "weather-call",
                    "output": (
                        '{"city": "San Francisco", "weather_description": "The weather in San Francisco is cold."}'
                    ),
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "It is cold in San Francisco."}],
                },
            ],
        },
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rollout: rollout["response"].update(status="incomplete"),
        lambda rollout: rollout["response"]["usage"].update(input_tokens=0),
        lambda rollout: rollout["response"]["usage"].update(output_tokens=0),
        lambda rollout: rollout["response"]["output"].__setitem__(slice(0, 1), []),
        lambda rollout: rollout["response"]["output"][1].update(call_id="wrong-call"),
        lambda rollout: rollout["response"]["output"][-1].update(content=[]),
    ],
)
def test_inference_provider_rollout_verifier_rejects_invalid_tool_loop(tmp_path: Path, mutate) -> None:
    rollout = _valid_inference_provider_rollout()
    mutate(rollout)
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollouts_path.write_text(f"{json.dumps(rollout)}\n")

    result = subprocess.run(
        [sys.executable, str(INFERENCE_PROVIDER_ROLLOUT_VERIFIER), "--rollouts", str(rollouts_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_inference_provider_rollout_verifier_accepts_completed_tool_loop(tmp_path: Path) -> None:
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollouts_path.write_text(f"{json.dumps(_valid_inference_provider_rollout())}\n")

    result = subprocess.run(
        [sys.executable, str(INFERENCE_PROVIDER_ROLLOUT_VERIFIER), "--rollouts", str(rollouts_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_inference_provider_rollout_verifier_reports_incomplete_reason(tmp_path: Path) -> None:
    rollout = _valid_inference_provider_rollout()
    rollout["response"].update(
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollouts_path.write_text(f"{json.dumps(rollout)}\n")

    result = subprocess.run(
        [sys.executable, str(INFERENCE_PROVIDER_ROLLOUT_VERIFIER), "--rollouts", str(rollouts_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "max_output_tokens" in result.stderr


def test_runner_disk_reclamation_fails_fast_when_space_is_still_low(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "sudo", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "df",
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-Pk" ]]; then\n'
        "  printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/root 1000 901 99 91%% /\\n'\n"
        "else\n"
        "  printf 'Filesystem Size Used Avail Use%% Mounted on\\n/dev/root 1G 901M 99M 91%% /\\n'\n"
        "fi\n",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "GITHUB_WORKSPACE": str(tmp_path),
            "GYM_CI_MIN_FREE_DISK_KB": "100",
        }
    )

    result = subprocess.run([str(RECLAIM_RUNNER_DISK)], capture_output=True, text=True, env=env)

    assert result.returncode == 1
    assert "Insufficient runner disk space: 99 KiB available; 100 KiB required" in result.stderr


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_server_tests_propagates_absolute_cache_and_venv_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    ci_dir = repo_root / "scripts" / "ci"
    shutil.copytree(REPO_ROOT / "scripts" / "ci", ci_dir)
    shutil.copy2(REPO_ROOT / ".python-version", repo_root / ".python-version")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "ng-test-all.args"

    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
cat <<'INSTALL'
set -eu
mkdir -p "${UV_UNMANAGED_INSTALL}"
cat > "${UV_UNMANAGED_INSTALL}/uv" <<'UV'
#!/usr/bin/env bash
set -eu
case "${1:-}" in
    --version) printf '%s\\n' 'uv 0.11.29' ;;
    cache) printf '%s\\n' "${UV_CACHE_DIR:-${HOME}/.cache/uv}" ;;
    venv)
        venv_dir="${@: -1}"
        mkdir -p "${venv_dir}/bin"
        : > "${venv_dir}/bin/activate"
        : > "${venv_dir}/bin/python"
        chmod +x "${venv_dir}/bin/python"
        ;;
    sync) ;;
    *) printf 'unexpected fake uv command: %s\\n' "$*" >&2; exit 2 ;;
esac
UV
chmod +x "${UV_UNMANAGED_INSTALL}/uv"
INSTALL
""",
    )
    _write_executable(
        bin_dir / "ng_test_all",
        """#!/usr/bin/env bash
set -eu
printf 'UV_CACHE_DIR=%s\\n' "${UV_CACHE_DIR}" > "${GYM_CI_CAPTURE}"
printf 'UV_LINK_MODE=%s\\n' "${UV_LINK_MODE:-}" >> "${GYM_CI_CAPTURE}"
printf 'GYM_CI_DEV_VENV_DIR=%s\\n' "${GYM_CI_DEV_VENV_DIR:-}" >> "${GYM_CI_CAPTURE}"
printf 'ARG=%s\\n' "$@" >> "${GYM_CI_CAPTURE}"
""",
    )

    node_local_root = tmp_path / "node-local"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "HOME": str(tmp_path / "home"),
            "UV_CACHE_DIR": "relative-cache",
            "GYM_CI_UV_VENV_DIR": str(node_local_root),
            "GYM_CI_CAPTURE": str(capture_path),
        }
    )
    subprocess.run([str(ci_dir / "server_tests.sh"), "2", "8"], check=True, env=env)

    captured = capture_path.read_text().splitlines()
    assert f"UV_CACHE_DIR={repo_root}/relative-cache" in captured
    assert "UV_LINK_MODE=copy" in captured
    assert f"GYM_CI_DEV_VENV_DIR={node_local_root}/.driver-venv" in captured
    assert "ARG=+uv_cache_dir=" + str(repo_root / "relative-cache") in captured
    assert "ARG=+uv_venv_dir=" + str(node_local_root) in captured
    assert "ARG=+shard_index=2" in captured
    assert "ARG=+num_shards=8" in captured
    assert not (node_local_root / ".driver-venv").exists()


@pytest.mark.parametrize("venv_root", ["relative-venvs", "/"])
def test_server_tests_rejects_unsafe_venv_root(venv_root: str) -> None:
    env = os.environ.copy()
    env["GYM_CI_UV_VENV_DIR"] = venv_root

    result = subprocess.run([str(SERVER_TESTS), "0", "8"], capture_output=True, text=True, env=env)

    assert result.returncode == 2
    assert f"GYM_CI_UV_VENV_DIR must be an absolute non-root path: {venv_root}" in result.stderr
