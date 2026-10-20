# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned SciCode-Verified v2 release for NeMo Gym.

The released problems and corrected HDF5 targets are downloaded from the versioned Hugging Face
dataset. The three official unscored-step implementations are downloaded from the pinned source
repository revision and embedded in their corresponding Gym rows as cumulative context.
"""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "scicode_verified_benchmark.jsonl"
MANIFEST_FPATH = DATA_DIR / "manifest.json"
TEST_DATA_FPATH = DATA_DIR / "test_data_cleaned.h5"

HF_DATASET_ID = "shhu2001/SciCode-Verified"
HF_REVISION = "eea11a866be6860725258702b39ef8651ed26abd"  # pragma: allowlist secret
UPSTREAM_GIT_REVISION = "ddab4a92f8d80a7113ab946628e994b52354d838"  # pragma: allowlist secret
DATASET_VERSION = "v2"

EXPECTED_PROBLEMS = 64
EXPECTED_TOTAL_SUBPROBLEMS = 290
EXPECTED_SCORED_SUBPROBLEMS = 287
EXPECTED_PROBLEMS_JSONL_MD5 = "5c604d8dbf52642bd94e13b92c8f52eb"  # pragma: allowlist secret
EXPECTED_PROBLEMS_JSONL_SHA256 = (
    "427771cb8bceb5058e8b510af0ee2c8210827a1491e0cdf8e6db56d4ed1440ba"  # pragma: allowlist secret
)
EXPECTED_MANIFEST_SHA256 = (
    "5e17afe722e127d120d6e48793cdf0379d430068eea8efb94a25503c6b3e83f9"  # pragma: allowlist secret
)
EXPECTED_H5_MD5 = "2b41a7df40ddc23ce651ec05b8ecb6f8"  # pragma: allowlist secret

# Official SciCode skip steps. They are inserted into cumulative context but never generated or scored.
PREFILLED_STEP_SHA256 = {
    "13.6": "795a2b57c2d9bb12ca4eaf16d6b8e1f202015a89a886628858abf42a1b18a94e",  # pragma: allowlist secret
    "62.1": "bc9931d88a7d5950091b72a996a25b8be6c936fd136b01005e22c3d45b0008a2",  # pragma: allowlist secret
    "76.3": "4758300d96ea726cdc0fbf749f1bc437030d2a3e8b43bde232ecdeaf636cd367",  # pragma: allowlist secret
}
PREFILLED_STEP_LOCATIONS = {
    ("13", 5): "13.6",
    ("62", 0): "62.1",
    ("76", 2): "76.3",
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as contents:
        for chunk in iter(lambda: contents.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(path: Path, algorithm: str, expected: str) -> None:
    actual = _digest(path, algorithm)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}.")


def _download_release_file(filename: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=HF_REVISION,
            token=token,
        )
    )


def _download_prefilled_steps() -> Dict[str, str]:
    prefilled_dir = DATA_DIR / "prefilled_steps"
    prefilled_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for step_number, expected_sha256 in PREFILLED_STEP_SHA256.items():
        destination = prefilled_dir / f"{step_number}.txt"
        if not destination.exists():
            url = (
                "https://raw.githubusercontent.com/flyingwagner/scicode-verified/"
                f"{UPSTREAM_GIT_REVISION}/eval_clean/vendor/eval_data/{step_number}.txt"
            )
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned HTTPS source
                destination.write_bytes(response.read())
        _require_digest(destination, "sha256", expected_sha256)
        result[step_number] = destination.read_text()
    return result


def _stage_file(source: Path, destination: Path, algorithm: str, expected: str) -> None:
    if destination.exists():
        _require_digest(destination, algorithm, expected)
        return
    if destination.is_symlink():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)
    _require_digest(destination, algorithm, expected)


def _validate_release(problems_path: Path, manifest_path: Path, h5_path: Path) -> tuple[list[dict], dict]:
    _require_digest(problems_path, "md5", EXPECTED_PROBLEMS_JSONL_MD5)
    _require_digest(problems_path, "sha256", EXPECTED_PROBLEMS_JSONL_SHA256)
    _require_digest(manifest_path, "sha256", EXPECTED_MANIFEST_SHA256)
    _require_digest(h5_path, "md5", EXPECTED_H5_MD5)

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != DATASET_VERSION:
        raise RuntimeError(f"Expected SciCode-Verified {DATASET_VERSION}, got {manifest.get('version')!r}.")
    if manifest.get("n_problems") != EXPECTED_PROBLEMS:
        raise RuntimeError(f"Expected {EXPECTED_PROBLEMS} problems in the release manifest.")
    if manifest.get("problems_test_jsonl_md5") != EXPECTED_PROBLEMS_JSONL_MD5:
        raise RuntimeError("The release manifest does not bind the expected problems JSONL.")
    if manifest.get("h5_md5") != EXPECTED_H5_MD5:
        raise RuntimeError("The release manifest does not bind the expected corrected HDF5 targets.")

    rows = [json.loads(line) for line in problems_path.read_text().splitlines() if line.strip()]
    problem_ids = [str(row["problem_id"]) for row in rows]
    expected_order = [str(problem_id) for problem_id in manifest["problem_order"]]
    if problem_ids != expected_order or len(rows) != EXPECTED_PROBLEMS:
        raise RuntimeError("SciCode-Verified problem count or order does not match the release manifest.")
    if "2" in problem_ids:
        raise RuntimeError("SciCode-Verified must exclude under-specified original problem 2.")

    step_numbers = [step["step_number"] for row in rows for step in row["sub_steps"]]
    if len(step_numbers) != EXPECTED_TOTAL_SUBPROBLEMS:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_SUBPROBLEMS} total subproblems, got {len(step_numbers)}.")
    actual_prefilled_locations = {
        (str(row["problem_id"]), step_index): step["step_number"]
        for row in rows
        for step_index, step in enumerate(row["sub_steps"])
        if step["step_number"] in PREFILLED_STEP_SHA256
    }
    if actual_prefilled_locations != PREFILLED_STEP_LOCATIONS:
        raise RuntimeError(
            "SciCode-Verified official prefilled-step locations do not match the pinned protocol: "
            f"expected {PREFILLED_STEP_LOCATIONS}, got {actual_prefilled_locations}."
        )
    scored_subproblems = len(step_numbers) - len(PREFILLED_STEP_LOCATIONS)
    if scored_subproblems != EXPECTED_SCORED_SUBPROBLEMS:
        raise RuntimeError(f"Expected {EXPECTED_SCORED_SUBPROBLEMS} scored subproblems, got {scored_subproblems}.")
    return rows, manifest


def prepare() -> Path:
    """Download, integrity-check, and convert SciCode-Verified v2 to Gym JSONL."""
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    problems_path = _download_release_file("data/problems_test.jsonl", hf_token)
    manifest_path = _download_release_file("manifest.json", hf_token)
    h5_path = _download_release_file("test_data_cleaned.h5", hf_token)

    rows, _ = _validate_release(problems_path, manifest_path, h5_path)
    prefilled_steps = _download_prefilled_steps()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _stage_file(manifest_path, MANIFEST_FPATH, "sha256", EXPECTED_MANIFEST_SHA256)
    _stage_file(h5_path, TEST_DATA_FPATH, "md5", EXPECTED_H5_MD5)

    output_rows = []
    for source_row in rows:
        row = dict(source_row)
        row["responses_create_params"] = {"input": []}
        row["uuid"] = str(row["problem_id"])
        row_prefills = {
            step["step_number"]: prefilled_steps[step["step_number"]]
            for step in row["sub_steps"]
            if step["step_number"] in prefilled_steps
        }
        if row_prefills:
            row["prefilled_steps_code"] = row_prefills
        output_rows.append(json.dumps(row, ensure_ascii=False) + "\n")

    OUTPUT_FPATH.write_text("".join(output_rows))
    print(
        f"Wrote {len(output_rows)} SciCode-Verified {DATASET_VERSION} problems to {OUTPUT_FPATH}; "
        f"corrected targets staged at {TEST_DATA_FPATH}."
    )
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
