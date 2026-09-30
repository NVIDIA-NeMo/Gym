# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the complete pinned public SOL-ExecBench evaluation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
REPO_ID = "nvidia/SOL-ExecBench"
SOURCE_REVISION = "63699402f003496acc3af4eb534a5304a8ac1ea9"
FLASHINFER_TRACE_REPO_ID = "flashinfer-ai/flashinfer-trace"
FLASHINFER_TRACE_REVISION = "4ee6fc905cdef5ef6b941b73ff4a220c92aec470"
FLASHINFER_TRACE_LOCAL_PREFIX = Path("data/flashinfer-trace")
SUBSET_COUNTS = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}
LANGUAGES = ("cuda_cpp", "triton")


def _json_object(value: object, field: str, task_id: str) -> dict[str, object]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError(f"{task_id}: {field} must decode to an object")
    return parsed


def _workloads(value: object, task_id: str) -> list[dict[str, object]]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, dict) for item in parsed):
        raise ValueError(f"{task_id}: workloads must decode to a nonempty list of objects")
    uuids = [item.get("uuid") for item in parsed]
    if any(not isinstance(uuid, str) or not uuid for uuid in uuids) or len(uuids) != len(set(uuids)):
        raise ValueError(f"{task_id}: workloads require unique nonempty UUIDs")
    return parsed


def definition_from_row(row: Mapping[str, object], task_id: str) -> dict[str, object]:
    """Reconstruct the public Definition while preserving argument insertion order."""
    definition = {
        "name": row["name"],
        "description": row.get("description") or "",
        "axes": _json_object(row["axes"], "axes", task_id),
        "inputs": _json_object(row["inputs"], "inputs", task_id),
        "outputs": _json_object(row["outputs"], "outputs", task_id),
        "reference": row["reference"],
    }
    for field in ("hf_id", "custom_inputs_entrypoint"):
        if row.get(field):
            definition[field] = row[field]
    return definition


def safetensors_asset_paths(workloads: Sequence[Mapping[str, object]], task_id: str) -> tuple[str, ...]:
    """Allow only relative files within the pinned FlashInfer asset prefix."""
    paths: set[str] = set()
    for workload in workloads:
        inputs = workload.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError(f"{task_id}: workload inputs must be an object")
        for spec in inputs.values():
            if not isinstance(spec, Mapping) or spec.get("type") != "safetensors":
                continue
            if "shards" in spec:
                raise ValueError(f"{task_id}: the pinned native runtime does not support safetensors shards")
            raw_path = spec.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{task_id}: safetensors input has no path")
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts or "\\" in raw_path or path.as_posix() != raw_path:
                raise ValueError(f"{task_id}: unsafe safetensors path {raw_path!r}")
            if not path.is_relative_to(FLASHINFER_TRACE_LOCAL_PREFIX) or path.suffix != ".safetensors":
                raise ValueError(f"{task_id}: asset is outside the pinned FlashInfer safetensors root")
            paths.add(path.as_posix())
    return tuple(sorted(paths))


def download_safetensors_assets(
    subset_rows: Mapping[str, Sequence[Mapping[str, object]]],
    output_dir: Path,
    *,
    token: str | bool | None = None,
) -> None:
    """Download exactly the referenced files from the immutable FlashInfer revision."""
    from huggingface_hub import snapshot_download

    paths = {
        path
        for subset, rows in subset_rows.items()
        for row in rows
        for path in safetensors_asset_paths(_workloads(row["workloads"], f"{subset}/{row['name']}"), str(row["name"]))
    }
    if not paths:
        return
    snapshot_download(
        FLASHINFER_TRACE_REPO_ID,
        repo_type="dataset",
        revision=FLASHINFER_TRACE_REVISION,
        local_dir=output_dir / FLASHINFER_TRACE_LOCAL_PREFIX,
        allow_patterns=[Path(path).relative_to(FLASHINFER_TRACE_LOCAL_PREFIX).as_posix() for path in sorted(paths)],
        token=token,
    )
    for relative in paths:
        path = output_dir / relative
        if not path.is_file() or not path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Missing or unsafe pinned asset: {relative}")


def _prompt(definition: Mapping[str, object], workloads: Sequence[Mapping[str, object]], language: str) -> str:
    source_path = "main.cu" if language == "cuda_cpp" else "main.py"
    example = {
        "name": "candidate",
        "definition": definition["name"],
        "author": "model",
        "spec": {
            "languages": [language],
            "target_hardware": ["B200"],
            "entry_point": f"{source_path}::run",
            "destination_passing_style": True,
        },
        "sources": [{"path": source_path, "content": "<complete source code>"}],
    }
    return (
        f"Implement this GPU kernel in {language} for an NVIDIA B200. Optimize latency while preserving correctness "
        "for every workload below. Use the Definition input order for function arguments, followed by output "
        "tensors in Definition output order. Write into the supplied output tensors (destination-passing style). "
        "Use a torch extension binding for CUDA C++.\n\n"
        "Return exactly one native SOL-ExecBench Solution JSON object, with complete source file contents and "
        "no Markdown fences or surrounding prose. Source paths must be relative and cannot contain '..'. "
        "The entry point must name a function in one of the sources. The required object structure is:\n"
        + json.dumps(example, ensure_ascii=False, indent=2)
        + "\n\nDefinition:\n"
        + json.dumps(definition, ensure_ascii=False, indent=2)
        + "\n\nWorkloads (all are evaluated):\n"
        + json.dumps(workloads, ensure_ascii=False, indent=2)
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def materialize(
    subset_rows: Mapping[str, Sequence[Mapping[str, object]]],
    output_dir: Path = DATA_DIR,
    *,
    language: str = "cuda_cpp",
    expected_counts: Mapping[str, int] = SUBSET_COUNTS,
) -> dict[str, Path]:
    """Write all tasks, complete workloads, a trusted manifest, and one generated example."""
    from resources_servers.sol_execbench.problem_store import NATIVE_REVISION, problem_digest

    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}")
    if set(subset_rows) != set(SUBSET_COUNTS) or set(expected_counts) != set(SUBSET_COUNTS):
        raise ValueError(f"Expected exactly these source subsets: {list(SUBSET_COUNTS)}")
    problems = []
    dataset = []
    seen: set[str] = set()
    for subset in SUBSET_COUNTS:
        rows = subset_rows[subset]
        if len(rows) != expected_counts[subset]:
            raise ValueError(f"{subset}: expected {expected_counts[subset]} tasks, found {len(rows)}")
        for row in sorted(rows, key=lambda row: str(row["name"])):
            task_id = f"{subset}/{row['name']}"
            if task_id in seen:
                raise ValueError(f"Duplicate task ID: {task_id}")
            seen.add(task_id)
            definition = definition_from_row(row, task_id)
            workloads = _workloads(row["workloads"], task_id)
            assets = []
            for relative in safetensors_asset_paths(workloads, task_id):
                path = output_dir / relative
                if not path.is_file() or not path.resolve().is_relative_to(output_dir.resolve()):
                    raise ValueError(f"{task_id}: missing or unsafe pinned asset {relative!r}")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                assets.append({"path": relative, "sha256": digest})
            problem = {"task_id": task_id, "definition": definition, "workloads": workloads, "assets": assets}
            problem["problem_digest"] = problem_digest(task_id, definition, workloads, assets)
            problems.append(problem)
            dataset.append(
                {
                    "task_id": task_id,
                    "responses_create_params": {
                        "input": [{"role": "user", "content": _prompt(definition, workloads, language)}]
                    },
                    "verifier_metadata": {"task_id": task_id, "problem_digest": problem["problem_digest"]},
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manifest": output_dir / "problem_manifest.json",
        "manifest_sha256": output_dir / "problem_manifest.sha256",
        "dataset": output_dir / f"solexecbench_{language}.jsonl",
        "example": output_dir / f"example_{language}.jsonl",
    }
    _write_json(
        outputs["manifest"],
        {
            "schema_version": 1,
            "source": {"repository": REPO_ID, "revision": SOURCE_REVISION},
            "native_revision": NATIVE_REVISION,
            "asset_source": {
                "repository": FLASHINFER_TRACE_REPO_ID,
                "revision": FLASHINFER_TRACE_REVISION,
                "local_prefix": FLASHINFER_TRACE_LOCAL_PREFIX.as_posix(),
            },
            "problems": problems,
        },
    )
    outputs["manifest_sha256"].write_text(hashlib.sha256(outputs["manifest"].read_bytes()).hexdigest() + "\n")
    lines = [json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in dataset]
    outputs["dataset"].write_text("".join(lines), encoding="utf-8")
    outputs["example"].write_text(lines[0], encoding="utf-8")
    return outputs


def prepare(*, language: str = "cuda_cpp", output_dir: str | Path = DATA_DIR) -> Path:
    """Fetch every pinned task and asset; return Gym's complete benchmark JSONL."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}")
    from datasets import load_dataset

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

    token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    output_dir = Path(output_dir)
    subset_rows = {
        subset: list(load_dataset(REPO_ID, name=subset, split="train", revision=SOURCE_REVISION, token=token))
        for subset in SUBSET_COUNTS
    }
    download_safetensors_assets(subset_rows, output_dir, token=token)
    return materialize(subset_rows, output_dir, language=language)["dataset"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--language", choices=LANGUAGES, default="cuda_cpp")
    args = parser.parse_args()
    # This entrypoint owns its argparse flags; Gym's Hydra parser must not consume them.
    from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict

    get_global_config_dict(GlobalConfigDictParserConfig(skip_load_from_cli=True))
    print(prepare(output_dir=args.output_dir, language=args.language))


if __name__ == "__main__":
    main()
