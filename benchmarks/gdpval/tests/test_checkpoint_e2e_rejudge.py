# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"


def test_rejudge_shell_entrypoints_parse() -> None:
    scripts = [
        PACKAGE / "judge.sbatch",
        PACKAGE / "transport_prebuild.sbatch",
        PACKAGE / "prepare_rejudge_fingerprint.sh",
        PACKAGE / "rejudge_bootstrap.sbatch",
        PACKAGE / "rejudge_controller.sbatch",
        PACKAGE / "launch_rejudge.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_judge_override_bundle_cannot_alias_original_campaign() -> None:
    script = (PACKAGE / "judge.sbatch").read_text()
    assert "override_count == 0 || override_count == 4" in script
    assert 'JUDGE_DIR_SUFFIX != e2e' in script
    assert 'TRANSPORT_VIEWS_PY=$E2E_DIR/transport_views.py' in script
    assert 'TRANSPORT_VIEW_ROOT == "$RUN_DIR"/*' in script
    assert 'TRANSPORT_VIEW_ROOT != "$CAMPAIGN_TRANSPORT_VIEW_ROOT"' in script
    assert 'REQUESTED_JUDGE_RUNTIME_OVERLAY == "$RUN_DIR"/*' in script
    assert 'JUDGE_RUNTIME_OVERLAY=$MARS_RUNTIME' in script
    assert 'JUDGE_RUNTIME_OVERLAY == /raid/scratch/*' in script
    assert 'JUDGE_TRANSPORT_OVERLAY == "$E2E_DIR/true3_transport.yaml"' in script
    assert 'TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX' in script


def test_corrected_transport_prebuild_requires_sibling_receipt_bundle() -> None:
    script = (PACKAGE / "transport_prebuild.sbatch").read_text()
    assert "override_count == 0 || override_count == 3" in script
    assert 'TRANSPORT_VIEW_ROOT == "$RUN_DIR"/*' in script
    assert 'TRANSPORT_VIEW_ROOT != "$CAMPAIGN_TRANSPORT_VIEW_ROOT"' in script
    assert 'TRANSPORT_VIEWS_PY=$E2E_DIR/transport_views.py' in script
    assert 'TRANSPORT_PREBUILD_PASS_$TRANSPORT_RECEIPT_SUFFIX' in script


def test_fingerprint_preparation_is_locked_and_revalidates_existing_receipts() -> None:
    script = (PACKAGE / "prepare_rejudge_fingerprint.sh").read_text()
    assert 'flock 9' in script
    assert 'existing fingerprint receipt no longer matches current runtime/profile inputs' in script
    assert 'discovery_rc=$?' in script
    assert '[[ $discovery_rc == 1 ]]' in script
    assert 'original_fingerprint=' in script
    assert 'fingerprint != "$original_fingerprint"' in script
    assert '"$CAMPAIGN_E2E_SCRIPT" _compute-preflight "$RUN_DIR"' in script
    assert 'SOURCE_PREPROCESSED=$RUN_DIR/judge_e2e/preprocessed_datasets/benchmark.jsonl' in script
    assert 'PREPROCESSED=$JUDGE_DIR/preprocessed_datasets/benchmark.jsonl' in script
    assert 'cmp -s "$SOURCE_PREPROCESSED" "$PREPROCESSED"' in script
    assert 'chmod 0400 "$preprocessed_temporary"' in script
    assert 'd["preprocessed_input"]==p' in script
    assert 'd["preprocessed_input_sha256"]==s' in script


def test_fingerprint_probe_loads_the_same_judge_environment_without_provider_access() -> None:
    script = (PACKAGE / "prepare_rejudge_fingerprint.sh").read_text()
    env_load = 'for try in 1 2 3 4 5 6; do'
    assert env_load in script
    assert '[[ $line == export\\ * ]] || continue' in script
    assert '[[ -n ${!name:-} ]] || eval "$line"' in script
    assert 'JUDGE_API_KEY must be the sk- LiteLLM key' in script
    assert ': "${JUDGE_BASE_URL:?JUDGE_BASE_URL is required}"' in script
    assert script.index(env_load) < script.index('probe() {')
    assert 'models_json=$(curl' not in script


def test_judge_disables_bytecode_before_runtime_validation() -> None:
    script = (PACKAGE / "judge.sbatch").read_text()
    assert 'export PYTHONDONTWRITEBYTECODE=1' in script
    assert script.index('export PYTHONDONTWRITEBYTECODE=1') < script.index(
        'source "$MARS_JOB_ROOT/settings.env"'
    )


def test_judge_reuses_the_fingerprinted_sibling_preprocessed_input_before_provider_access() -> None:
    script = (PACKAGE / "judge.sbatch").read_text()
    assert 'PREPROCESSED_INPUT=$JUDGE_DIR/preprocessed_datasets/benchmark.jsonl' in script
    assert 'd["preprocessed_input_sha256"]==s' in script
    assert 'REUSE_PREPROCESSED_ARGS+=(++reuse_existing_data_preparation=true)' in script
    assert '"${REUSE_PREPROCESSED_ARGS[@]}"' in script
    assert script.index('REUSE_PREPROCESSED_ARGS+=(++reuse_existing_data_preparation=true)') > script.index(
        'if [[ $CORRECTED_REJUDGE == true ]]'
    )
    assert script.index('provider-free fingerprint input/distribution path/hash drift') < script.index(
        ': "${JUDGE_BASE_URL:?JUDGE_BASE_URL is required}"'
    )
    assert script.index('provider-free fingerprint input/distribution path/hash drift') < script.index(
        'models_json=$(curl'
    )


def test_rejudge_controller_gates_fingerprint_and_publishes_atomically() -> None:
    script = (PACKAGE / "rejudge_controller.sbatch").read_text()
    assert 'validate_original_campaign' in script
    assert 'corrected journal fingerprint drift' in script
    assert 'd["schema"]=="gdpval.multistage-fingerprint-probe.v1"' in script
    assert 'os.path.realpath(d["runtime_root"])==os.path.realpath(r)' in script
    assert 'os.path.realpath(d["preprocessed_input"])==os.path.realpath(p)' in script
    assert 'd["preprocessed_input_sha256"]==s' in script
    assert "waivable_failure_classes: [timeout_exceeded, transient]" in script
    assert "{num_tasks: 220, num_models: 4}" in script
    assert 'publish_immutable "$FINAL_TEMP" "$FINAL_RECEIPT"' in script
    assert 'publish_immutable "$FINAL_SHA_TEMP" "$FINAL_SHA"' in script
    assert 'publish_immutable "$FINAL_MARKER_TEMP" "$FINAL_MARKER"' in script
    assert script.index('publish_immutable "$FINAL_TEMP"') < script.index(
        'publish_immutable "$FINAL_MARKER_TEMP"'
    )
    assert 'validate_corrected_final' in script


def test_rejudge_launcher_can_replace_a_terminal_controller_safely() -> None:
    script = (PACKAGE / "launch_rejudge.sh").read_text()
    assert 'rejudge_controller_${JUDGE_DIR_SUFFIX}_submissions' in script
    assert 'case "$(job_liveness "$latest_job")"' in script
    assert 'REJUDGE_ALREADY_RUNNING' in script
    assert 'controller-a${attempt}' in script
    assert 'slurm_publish_job_receipt "$current_receipt" "$job" true' in script


def test_rejudge_bootstrap_is_n3_guarded_and_dependency_safe() -> None:
    script = (PACKAGE / "rejudge_bootstrap.sbatch").read_text()
    assert "provider calls were not explicitly authorized" in script
    assert '[[ $ACCOUNT == nemotron_n3_post ]]' in script
    assert "TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX" in script
    assert "prepare_rejudge_fingerprint.sh" in script
    assert script.index("prepare_rejudge_fingerprint.sh") < script.index("launch_rejudge.sh")
