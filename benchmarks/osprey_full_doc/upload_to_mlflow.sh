#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-${SCRIPT_DIR}/data/osprey_full_doc_benchmark.jsonl}"
DATASET_NAME="${DATASET_NAME:-osprey_full_doc}"
DATASET_VERSION="${DATASET_VERSION:-0.0.1}"
NG_BIN_DIR="${NG_BIN_DIR:-${REPO_ROOT}/.venv/bin}"

resolve_bin() {
    local binary_name="$1"
    local candidate=""

    if candidate="$(command -v "${binary_name}" 2>/dev/null)"; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="${REPO_ROOT}/.venv/bin/${binary_name}"
    if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="${NG_BIN_DIR}/${binary_name}"
    if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    echo "Could not find ${binary_name} on PATH or in a known .venv location." >&2
    exit 1
}

if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "Missing prepared benchmark JSONL: ${INPUT_JSONL}" >&2
    echo "Run ng_prepare_benchmark or benchmarks/osprey_full_doc/prepare.py first." >&2
    exit 1
fi

NG_UPLOAD_BIN="$(resolve_bin ng_upload_dataset_to_gitlab)"

upload_args=(
    "+dataset_name=${DATASET_NAME}"
    "+version=${DATASET_VERSION}"
    "+input_jsonl_fpath=${INPUT_JSONL}"
)

if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    upload_args+=("+mlflow_tracking_uri=${MLFLOW_TRACKING_URI}")
fi

if [[ -n "${MLFLOW_TRACKING_TOKEN:-}" ]]; then
    upload_args+=("+mlflow_tracking_token=${MLFLOW_TRACKING_TOKEN}")
fi

exec "${NG_UPLOAD_BIN}" "${upload_args[@]}"
