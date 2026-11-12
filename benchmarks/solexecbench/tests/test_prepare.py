# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.solexecbench import prepare as prepare_module
from benchmarks.solexecbench.prepare import (
    FLASHINFER_TRACE_REVISION,
    LANGUAGES,
    SOURCE_REVISION,
    SUBSET_COUNTS,
    download_safetensors_assets,
    materialize,
    safetensors_asset_paths,
)
from nemo_gym.base_responses_api_agent import BaseRunRequest
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from resources_servers.sol_execbench.problem_store import checked_asset
from resources_servers.sol_execbench.problem_store import load_manifest as load_problem_manifest


ROOT = Path(__file__).resolve().parents[3]
ASSET_PATH = "data/flashinfer-trace/blob/workloads/example/left.safetensors"


def source_row(name: str = "synthetic_add") -> dict:
    """Original synthetic input; no upstream evaluation-dataset content is embedded."""
    return {
        "name": name,
        "description": "Synthetic ordered vector sum for preparation tests.",
        "axes": json.dumps({"N": {"type": "var"}}),
        "inputs": json.dumps({"z": {"shape": ["N"], "dtype": "float32"}, "a": {"shape": ["N"], "dtype": "float32"}}),
        "outputs": json.dumps({"result": {"shape": ["N"], "dtype": "float32"}}),
        "reference": "def run(z, a):\n    return z + a\n",
        "workloads": json.dumps(
            [
                {
                    "uuid": f"synthetic-{n}",
                    "axes": {"N": n},
                    "inputs": {"z": {"type": "random"}, "a": {"type": "random"}},
                }
                for n in (17, 33)
            ]
        ),
    }


def subsets() -> dict:
    return {subset: [source_row()] for subset in SUBSET_COUNTS}


def prepare_fixture(tmp_path: Path, **kwargs) -> dict:
    return materialize(subsets(), tmp_path, expected_counts={subset: 1 for subset in SUBSET_COUNTS}, **kwargs)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("language", LANGUAGES)
def test_complete_rows_and_manifest_preserve_native_inputs(tmp_path, language):
    outputs = prepare_fixture(tmp_path, language=language)
    digest = outputs["manifest_sha256"].read_text().strip()
    manifest = load_problem_manifest(outputs["manifest"], digest)
    assert manifest.source["revision"] == SOURCE_REVISION
    assert manifest.asset_source["revision"] == FLASHINFER_TRACE_REVISION
    assert len(manifest.problems) == 4
    rows = read_rows(outputs["dataset"])
    assert read_rows(outputs["example"]) == rows[:1]
    assert rows[0]["task_id"] == "L1/synthetic_add"
    for row, problem in zip(rows, manifest.problems, strict=True):
        BaseRunRequest.model_validate(row)
        assert row["verifier_metadata"] == {"task_id": problem.task_id, "problem_digest": problem.problem_digest}
        assert list(problem.definition["inputs"]) == ["z", "a"]
        assert [workload["axes"]["N"] for workload in problem.workloads] == [17, 33]
        prompt = row["responses_create_params"]["input"][0]["content"]
        assert f'"languages": [\n      "{language}"' in prompt
        assert '"sources"' in prompt and '"destination_passing_style": true' in prompt
        assert all(workload["uuid"] in prompt for workload in problem.workloads)


def test_digest_tracks_workload_and_argument_order(tmp_path):
    original = prepare_fixture(tmp_path)
    initial = read_rows(original["dataset"])[0]["verifier_metadata"]["problem_digest"]
    source = subsets()
    row = source["L1"][0]
    row["inputs"] = json.dumps(dict(reversed(list(json.loads(row["inputs"]).items()))))
    swapped = materialize(source, tmp_path, expected_counts={subset: 1 for subset in SUBSET_COUNTS})
    assert read_rows(swapped["dataset"])[0]["verifier_metadata"]["problem_digest"] != initial
    source = subsets()
    source["L1"][0]["workloads"] = json.dumps(list(reversed(json.loads(source["L1"][0]["workloads"]))))
    reordered = materialize(source, tmp_path, expected_counts={subset: 1 for subset in SUBSET_COUNTS})
    assert read_rows(reordered["dataset"])[0]["verifier_metadata"]["problem_digest"] != initial


def test_full_corpus_counts_are_enforced_before_writing(tmp_path):
    with pytest.raises(ValueError, match="expected 94 tasks"):
        materialize(subsets(), tmp_path)
    assert not list(tmp_path.iterdir())
    missing = subsets()
    del missing["Quant"]
    with pytest.raises(ValueError, match="Expected exactly"):
        materialize(missing, tmp_path)


def test_duplicate_tasks_and_workloads_rejected(tmp_path):
    source = subsets()
    source["L1"] *= 2
    with pytest.raises(ValueError, match="Duplicate task ID"):
        materialize(source, tmp_path, expected_counts={subset: 2 if subset == "L1" else 1 for subset in SUBSET_COUNTS})
    source = subsets()
    workloads = json.loads(source["L1"][0]["workloads"])
    workloads[1]["uuid"] = workloads[0]["uuid"]
    source["L1"][0]["workloads"] = json.dumps(workloads)
    with pytest.raises(ValueError, match="unique nonempty UUIDs"):
        materialize(source, tmp_path, expected_counts={subset: 1 for subset in SUBSET_COUNTS})


def test_safetensors_allowlist_matches_download_and_hashes(tmp_path, monkeypatch):
    source = subsets()
    workloads = json.loads(source["FlashInfer-Bench"][0]["workloads"])
    workloads[0]["inputs"]["z"] = {"type": "safetensors", "path": ASSET_PATH, "tensor_key": "z"}
    source["FlashInfer-Bench"][0]["workloads"] = json.dumps(workloads)
    calls = []

    def fake_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        path = tmp_path / ASSET_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(b"original synthetic tensor fixture")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    download_safetensors_assets(source, tmp_path, token=False)
    assert calls == [
        (
            "flashinfer-ai/flashinfer-trace",
            {
                "repo_type": "dataset",
                "revision": FLASHINFER_TRACE_REVISION,
                "local_dir": tmp_path / "data/flashinfer-trace",
                "allow_patterns": ["blob/workloads/example/left.safetensors"],
                "token": False,
            },
        )
    ]
    outputs = materialize(source, tmp_path, expected_counts={subset: 1 for subset in SUBSET_COUNTS})
    manifest = load_problem_manifest(outputs["manifest"], outputs["manifest_sha256"].read_text().strip())
    asset = manifest.problems[-1].assets[0]
    assert asset.sha256 == hashlib.sha256(b"original synthetic tensor fixture").hexdigest()
    checked_asset(tmp_path, asset)
    (tmp_path / ASSET_PATH).write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checked_asset(tmp_path, asset)


@pytest.mark.parametrize(
    "unsafe",
    [
        "/tmp/file.safetensors",
        "../file.safetensors",
        "other/file.safetensors",
        "data/flashinfer-trace/../../file.safetensors",
    ],
)
def test_rejects_asset_paths_outside_pinned_root(unsafe):
    with pytest.raises(ValueError, match="unsafe|outside"):
        safetensors_asset_paths([{"inputs": {"x": {"type": "safetensors", "path": unsafe}}}], "test")


def test_sharded_safetensors_rejected_by_pinned_native_contract():
    second = ASSET_PATH.replace("left", "right")
    inputs = {"x": {"type": "safetensors", "shards": [{"path": ASSET_PATH}, {"path": second}]}}
    with pytest.raises(ValueError, match="does not support safetensors shards"):
        safetensors_asset_paths([{"inputs": inputs}], "test")


def test_prepare_uses_every_pinned_subset_and_returns_configured_language(tmp_path, monkeypatch):
    calls = []

    def fake_load(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return [source_row(f"synthetic_{i}") for i in range(SUBSET_COUNTS[kwargs["name"]])]

    monkeypatch.setattr("datasets.load_dataset", fake_load)
    monkeypatch.setattr("nemo_gym.global_config.get_global_config_dict", lambda: {})
    monkeypatch.setattr(prepare_module, "download_safetensors_assets", lambda *args, **kwargs: None)
    result = prepare_module.prepare(language="triton", output_dir=tmp_path)
    assert result.name == "solexecbench_triton.jsonl"
    assert len(read_rows(result)) == 235
    assert [kwargs["name"] for _, kwargs in calls] == list(SUBSET_COUNTS)
    assert all(
        repo == "nvidia/SOL-ExecBench" and args["revision"] == SOURCE_REVISION and args["split"] == "train"
        for repo, args in calls
    )


def test_canonical_benchmark_manifest_and_config():
    manifest = load_manifest(ROOT / "benchmarks/solexecbench/manifest.yaml")
    assert manifest.licensing == "LicenseRef-NVIDIA-Evaluation"
    assert manifest.prompt_source == "prepared"
    benchmark = BenchmarkConfig.from_config_path(ROOT / "benchmarks/solexecbench/config.yaml", strict=False)
    assert benchmark.name == "solexecbench"
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/solexecbench/data/solexecbench_cuda_cpp.jsonl")
    assert benchmark.dataset.prepare_script == Path("benchmarks/solexecbench/prepare.py")
    assert benchmark.num_repeats == 1


def test_standalone_arguments_do_not_reach_hydra(tmp_path, monkeypatch):
    parser_configs = []
    selections = []
    monkeypatch.setattr("sys.argv", ["prepare", "--language", "triton", "--output-dir", str(tmp_path)])
    monkeypatch.setattr("nemo_gym.global_config.get_global_config_dict", lambda config: parser_configs.append(config))
    monkeypatch.setattr(prepare_module, "prepare", lambda **kwargs: selections.append(kwargs))
    prepare_module.main()
    assert parser_configs[0].skip_load_from_cli is True
    assert selections == [{"language": "triton", "output_dir": tmp_path}]
