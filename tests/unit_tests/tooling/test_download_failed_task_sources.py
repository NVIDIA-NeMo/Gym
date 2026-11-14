# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry-backed task sources retain original data without invoking policy models."""

import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "download_failed_task_sources.py"
spec = importlib.util.spec_from_file_location("original_task_download_tests", SCRIPT)
utility = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utility
spec.loader.exec_module(utility)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    actual_exists = Path.exists
    shared_temp_marker = Path(tempfile.gettempdir()) / ".git"
    monkeypatch.setattr(Path, "exists", lambda path: False if path == shared_temp_marker else actual_exists(path))
    root = tmp_path / "fictional-gym"
    root.mkdir()
    (root / ".git").mkdir()
    revision = "b" * 40

    def git_reads_only(command, **kwargs):
        if command[0] != "git":
            raise AssertionError("Preparing a download plan must not launch a downloader, model, or server")
        return SimpleNamespace(stdout=revision + "\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", git_reads_only)
    identifier = {
        "backend": "gitlab",
        "dataset_name": "fictional_registered_tasks",
        "version": "0.0.3",
        "artifact_fpath": "fictional-validation.jsonl",
    }
    selected_dataset = {
        "name": "fictional_validation",
        "type": "validation",
        "jsonl_fpath": "data/fictional-validation.jsonl",
        "license": "MIT",
        "num_repeats": 4,
        "gitlab_identifier": {name: value for name, value in identifier.items() if name != "backend"},
    }
    original_config = {
        "fictional_agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "entrypoint": "app.py",
                    "max_steps": 17,
                    "datasets": [
                        selected_dataset,
                        {
                            **copy.deepcopy(selected_dataset),
                            "name": "unrequested_variant",
                            "gitlab_identifier": {
                                "dataset_name": "fictional_registered_tasks",
                                "version": "0.0.9",
                                "artifact_fpath": "other.jsonl",
                            },
                        },
                    ],
                }
            }
        },
        "unrequested_agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "entrypoint": "app.py",
                    "datasets": [{**copy.deepcopy(selected_dataset), "name": "unrequested_agent_dataset"}],
                }
            }
        },
        "fictional_resource": {"resources_servers": {"fictional_native": {"entrypoint": "app.py"}}},
    }
    config = root / "original-config.yaml"
    write_json(config, original_config)
    routes = {
        "schema_version": "original-task-download/routes/v1",
        "gym_revision": revision,
        "benchmarks": {
            "fictional_benchmark": {
                "config_paths": [str(config)],
                "agent_name": "fictional_agent",
                "dataset_name": "fictional_validation",
                "dataset_type": "validation",
                "registry_identifier": identifier,
            }
        },
    }
    routes_path = tmp_path / "download-routes.json"
    write_json(routes_path, routes)
    original_rows = [
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Use the complete original fictional task."}],
                "tools": [
                    {
                        "type": "function",
                        "name": "original_lookup",
                        "strict": False,
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    }
                ],
                "temperature": 0.6,
            },
            "verifier_metadata": {
                "reference_answer": "Original fictional answer",
                "rubric": [{"id": "F1", "criterion": "Keep every native reference"}],
            },
            "ground_truth": {"source": "fictional", "value": [1, 2, 3]},
            "native_extra": {"nested": ["retain", {"all_fields": True}]},
        }
    ]
    return {
        "root": tmp_path,
        "gym_root": root,
        "revision": revision,
        "config": config,
        "original_config": original_config,
        "routes": routes,
        "routes_path": routes_path,
        "identifier": identifier,
        "selected_dataset": selected_dataset,
        "original_rows": original_rows,
    }


def install_native_runtime(source, monkeypatch, *, failure=None, changed_identifier=False):
    from omegaconf import OmegaConf

    calls = []
    resolved = []
    monkeypatch.chdir(source["gym_root"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("NEMO_GYM_CONFIG_DICT", raising=False)
    monkeypatch.delenv("NEMO_GYM_CONFIG_PATH", raising=False)

    def forbidden_execution(*args, **kwargs):
        raise AssertionError("Task-source download must never start policy models, graders, or environment servers")

    class ParserConfig:
        NO_MODEL_GLOBAL_CONFIG_DICT = OmegaConf.create(
            {"policy_model": {"responses_api_models": {"dummy_model": {"entrypoint": "app.py"}}}}
        )

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Parser:
        def load_extra_config_paths(self, paths):
            calls.append("original_config_loader")
            assert [str(path) for path in paths] == [str(source["config"])]
            values = [OmegaConf.load(path) for path in paths]
            if changed_identifier:
                values[0].fictional_agent.responses_api_agents.simple_agent.datasets[
                    0
                ].gitlab_identifier.version = "0.0.4"
            return list(paths), values

        def _recursively_swap_keys(self, config):
            return None

    def get_original_config(*, global_config_dict_parser_config=None):
        if global_config_dict_parser_config is None:
            return resolved[-1]
        parser = global_config_dict_parser_config
        assert parser.skip_load_from_cli is True
        value = OmegaConf.create(parser.initial_global_config_dict)
        resolved.append(value)
        return value

    class OriginalProcessor:
        def run(self, config):
            calls.append("original_data_processor")
            value = OmegaConf.to_container(config, resolve=True)
            assert value["should_download"] is True
            assert value["mode"] == "train_preparation"
            assert "unrequested_agent" not in value
            assert "fictional_resource" not in value
            agent = value["fictional_agent"]["responses_api_agents"]["simple_agent"]
            assert len(agent["datasets"]) == 1, "Unrequested dataset versions must never be downloaded"
            selected = agent["datasets"][0]
            assert selected["name"] == source["selected_dataset"]["name"]
            assert selected["type"] == "validation"
            assert selected["num_repeats"] == 1, (
                "Source tasks must be cached once; failed attempts repeat at rerun time"
            )
            backend = source["identifier"]["backend"]
            assert value["data_source"] == backend
            assert selected[backend + "_identifier"] == source["selected_dataset"][backend + "_identifier"]
            original_file = Path(selected["jsonl_fpath"])
            assert original_file.is_absolute() and original_file.is_relative_to(source["root"])
            assert not original_file.is_relative_to(source["gym_root"])
            original_file.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(original_file, source["original_rows"])
            if failure:
                raise RuntimeError(failure)
            output = Path(value["output_dirpath"])
            output.mkdir(parents=True, exist_ok=True)
            rows = copy.deepcopy(source["original_rows"])
            for row in rows:
                row["agent_ref"] = {"type": "responses_api_agents", "name": "fictional_agent"}
            write_jsonl(output / "validation.jsonl", rows)

    modules = {}
    for name in (
        "nemo_gym",
        "nemo_gym.global_config",
        "nemo_gym.train_data_utils",
        "nemo_gym.cli",
        "nemo_gym.cli.env",
        "nemo_gym.cli.eval",
        "nemo_gym.rollout_collection",
    ):
        module = ModuleType(name)
        module.__file__ = str(source["gym_root"] / Path(*name.split(".")) / "__init__.py")
        module.__path__ = [str(Path(module.__file__).parent)]
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["nemo_gym.global_config"].GlobalConfigDictParserConfig = ParserConfig
    modules["nemo_gym.global_config"].GlobalConfigDictParser = Parser
    modules["nemo_gym.global_config"].get_global_config_dict = get_original_config
    modules["nemo_gym.train_data_utils"].TrainDataProcessor = OriginalProcessor
    modules["nemo_gym.cli.env"].RunHelper = forbidden_execution
    modules["nemo_gym.cli.eval"].e2e_rollout_collection = forbidden_execution
    modules["nemo_gym.rollout_collection"].RolloutCollectionHelper = forbidden_execution
    return calls


def download_fixture(source, monkeypatch):
    output = source["root"] / "private-cache"
    plan = utility.prepare_download_plan(source["routes_path"], source["gym_root"], output)
    calls = install_native_runtime(source, monkeypatch)
    job = plan["jobs"][0]
    result = utility.download_original_source(job["worker_spec"])
    job.update(result)
    manifest = output / "download_manifest.json"
    write_json(manifest, plan)
    return plan, manifest, calls


def test_download_plan_pins_original_registered_source_without_invoking_native_download(source):
    output = source["root"] / "uncreated-private-cache"
    plan = utility.download_sources(source["routes_path"], source["gym_root"], output, execute=False)
    assert not output.exists()
    assert len(plan["jobs"]) == 1
    selected = plan["jobs"][0]["worker_spec"]
    assert selected["agent_name"] == "fictional_agent"
    assert selected["dataset_name"] == "fictional_validation"
    assert selected["registry_identifier"] == source["identifier"]
    assert selected["config_files"] == [{"path": str(source["config"]), "sha256": file_hash(source["config"])}]
    assert plan["routes_sha256"] == file_hash(source["routes_path"])


def test_native_download_preflight_selects_only_requested_original_agent_and_dataset(source, monkeypatch):
    output = source["root"] / "preflight-cache"
    plan = utility.prepare_download_plan(source["routes_path"], source["gym_root"], output)
    calls = install_native_runtime(source, monkeypatch)
    result = utility.download_original_source(plan["jobs"][0]["worker_spec"], preflight_only=True)
    assert result["status"] == "preflight_passed"
    assert result["registry_identifier"] == source["identifier"]
    assert calls == ["original_config_loader"]
    assert not output.exists()


def test_native_processor_download_keeps_full_original_rows_and_publishes_verified_private_cache(source, monkeypatch):
    plan, _, calls = download_fixture(source, monkeypatch)
    job = plan["jobs"][0]
    assert calls == ["original_config_loader", "original_data_processor"]
    assert job["status"] == "completed"
    cache = Path(job["worker_spec"]["cache_dir"])
    receipt = job["receipt"]
    assert receipt["num_rows"] == 1
    assert receipt["source_dataset_num_repeats"] == 4 and receipt["cache_dataset_num_repeats"] == 1
    assert receipt["request_sha256"] == utility.rerun.canonical_sha256(job["worker_spec"])
    assert receipt["registry_identifier"] == source["identifier"]
    assert (cache.stat().st_mode & 0o077) == 0
    assert not list(cache.parent.glob(".download-*")), "Staging must disappear after atomic publication"
    prepared = cache / receipt["prepared_dataset_path"]
    assert file_hash(prepared) == receipt["prepared_dataset_sha256"]
    rows = read_jsonl(prepared)
    expected = copy.deepcopy(source["original_rows"])
    expected[0]["agent_ref"] = {"type": "responses_api_agents", "name": "fictional_agent"}
    assert rows == expected
    for item in receipt["files"]:
        assert not Path(item["path"]).is_absolute()
        assert file_hash(cache / item["path"]) == item["sha256"]
    assert rows[0]["verifier_metadata"]["rubric"] == source["original_rows"][0]["verifier_metadata"]["rubric"]


def test_existing_verified_cache_is_reused_without_downloader_or_server_calls(source, monkeypatch):
    plan, _, calls = download_fixture(source, monkeypatch)
    calls.clear()
    result = utility.download_original_source(plan["jobs"][0]["worker_spec"])
    assert result["status"] == "cached"
    assert calls == []


@pytest.mark.parametrize(
    "change", ["raw_artifact", "prepared_artifact", "original_config", "request_binding", "extra_file", "file_mode"]
)
def test_cached_data_or_bindings_cannot_change_silently(source, monkeypatch, change):
    plan, _, calls = download_fixture(source, monkeypatch)
    job = plan["jobs"][0]
    cache = Path(job["worker_spec"]["cache_dir"])
    if change == "raw_artifact":
        (cache / "raw" / "dataset.jsonl").write_text("Changed raw task\n", encoding="utf-8")
    elif change == "prepared_artifact":
        (cache / job["receipt"]["prepared_dataset_path"]).write_text("Changed reference\n", encoding="utf-8")
    elif change == "original_config":
        source["config"].write_text("Changed original dataset declaration\n", encoding="utf-8")
    elif change == "extra_file":
        (cache / "unbound-reference.txt").write_text("Injected verifier reference", encoding="utf-8")
    elif change == "file_mode":
        (cache / "raw" / "dataset.jsonl").chmod(0o700)
    else:
        job["worker_spec"]["registry_identifier"]["version"] = "0.0.4"
    calls.clear()
    with pytest.raises(ValueError):
        utility.download_original_source(job["worker_spec"])
    assert calls == []


@pytest.mark.parametrize("mismatch", ["missing_agent", "missing_dataset", "duplicate_dataset", "registry_version"])
def test_native_source_selection_refuses_missing_ambiguous_or_changed_registered_variants(
    source, monkeypatch, mismatch
):
    config = source["original_config"]
    inner = config["fictional_agent"]["responses_api_agents"]["simple_agent"]
    if mismatch == "missing_agent":
        config.pop("fictional_agent")
    elif mismatch == "missing_dataset":
        inner["datasets"].pop(0)
    elif mismatch == "duplicate_dataset":
        inner["datasets"].append(copy.deepcopy(inner["datasets"][0]))
    else:
        inner["datasets"][0]["gitlab_identifier"]["version"] = "0.0.4"
    write_json(source["config"], config)
    output = source["root"] / "mismatched-source"
    plan = utility.prepare_download_plan(source["routes_path"], source["gym_root"], output)
    calls = install_native_runtime(source, monkeypatch)
    with pytest.raises(ValueError):
        utility.download_original_source(plan["jobs"][0]["worker_spec"], preflight_only=True)
    assert "original_data_processor" not in calls
    assert not output.exists()


def test_registry_authentication_failure_does_not_publish_partial_or_zero_row_cache(source, monkeypatch):
    output = source["root"] / "unauthenticated-cache"
    plan = utility.prepare_download_plan(source["routes_path"], source["gym_root"], output)
    calls = install_native_runtime(source, monkeypatch, failure="Fictional registry authentication unavailable")
    with pytest.raises(RuntimeError, match="registry authentication"):
        utility.download_original_source(plan["jobs"][0]["worker_spec"])
    assert calls == ["original_config_loader", "original_data_processor"]
    assert not Path(plan["jobs"][0]["worker_spec"]["cache_dir"]).exists()
    assert not (output / "receipt.json").exists()
    assert not list(output.glob(".download-*")), "Failed partial downloads must not become reusable task caches"


def test_unbound_existing_cache_or_git_destination_is_refused(source):
    existing = source["root"] / "unbound-existing-cache"
    existing.mkdir()
    with pytest.raises(ValueError):
        utility.prepare_download_plan(source["routes_path"], source["gym_root"], existing)
    with pytest.raises(ValueError):
        utility.prepare_download_plan(source["routes_path"], source["gym_root"], source["gym_root"] / "private-cache")


def test_route_config_paths_resolve_portably_from_the_original_checkout(source, monkeypatch):
    source["routes"]["benchmarks"]["fictional_benchmark"]["config_paths"] = [source["config"].name]
    write_json(source["routes_path"], source["routes"])
    other = source["root"] / "unrelated-working-directory"
    other.mkdir()
    monkeypatch.chdir(other)
    plan = utility.prepare_download_plan(source["routes_path"], source["gym_root"], source["root"] / "portable-cache")
    assert plan["jobs"][0]["worker_spec"]["config_files"][0]["path"] == str(source["config"])


def test_bound_failed_attempts_keep_native_reference_rows_and_repetition(source, monkeypatch):
    plan, manifest, _ = download_fixture(source, monkeypatch)
    job = plan["jobs"][0]
    dataset = Path(job["worker_spec"]["cache_dir"]) / job["receipt"]["prepared_dataset_path"]
    original_row = read_jsonl(dataset)[0]
    selection = [
        {
            "benchmark": "fictional_benchmark",
            "orig_trajectory_id": f"fictional-attempt-{rollout}",
            "orig_task_index": 7,
            "orig_rollout_index": rollout,
            "dataset_row_index": 0,
            "expected_task_sha256": utility.rerun.canonical_sha256(original_row),
        }
        for rollout in [0, 2]
    ]
    selected = source["root"] / "failure-selection.jsonl"
    write_jsonl(selected, selection)
    bound = utility.bind_selection(selected, manifest, source["root"] / "bound-selection")
    assert bound["num_attempts"] == 2
    assert read_jsonl(Path(bound["bound_selection_path"])) == selection
    routes = json.loads(Path(bound["bound_routes_path"]).read_text())
    route = routes["benchmarks"]["fictional_benchmark"]
    assert route["dataset_path"] == str(dataset)
    assert route["dataset_sha256"] == file_hash(dataset)
    assert route["agent_name"] == "fictional_agent"
    prepared = source["root"] / "native-rerun-input"
    result = utility.rerun.prepare_dataset(bound["bound_selection_path"], bound["bound_routes_path"], prepared)
    assert result["num_rows"] == 2 and result["num_task_groups"] == 1
    rows = read_jsonl(prepared / result["jobs"][0]["input_path"])
    assert [row["orig_rollout_index"] for row in rows] == [0, 2]
    assert all(row["verifier_metadata"] == original_row["verifier_metadata"] for row in rows)
    assert all(row["responses_create_params"] == original_row["responses_create_params"] for row in rows)


def test_binding_refuses_different_original_task_content_instead_of_choosing_a_near_match(source, monkeypatch):
    _, manifest, _ = download_fixture(source, monkeypatch)
    selected = source["root"] / "wrong-failure-selection.jsonl"
    write_jsonl(
        selected,
        [
            {
                "benchmark": "fictional_benchmark",
                "orig_trajectory_id": "fictional-original-attempt",
                "orig_task_index": 3,
                "orig_rollout_index": 0,
                "dataset_row_index": 0,
                "expected_task_sha256": "f" * 64,
            }
        ],
    )
    output = source["root"] / "must-not-bind"
    with pytest.raises(ValueError, match="bound uniquely"):
        utility.bind_selection(selected, manifest, output)
    assert not output.exists()


def test_huggingface_registered_source_uses_same_native_data_only_pipeline(source, monkeypatch):
    identifier = {
        "backend": "huggingface",
        "repo_id": "fictional-organization/fictional-tasks",
        "artifact_fpath": "validation.jsonl",
    }
    dataset = source["selected_dataset"]
    dataset.pop("gitlab_identifier")
    dataset["huggingface_identifier"] = {key: value for key, value in identifier.items() if key != "backend"}
    source["identifier"] = identifier
    source["routes"]["benchmarks"]["fictional_benchmark"]["registry_identifier"] = identifier
    write_json(source["routes_path"], source["routes"])
    write_json(source["config"], source["original_config"])
    plan, _, calls = download_fixture(source, monkeypatch)
    assert plan["jobs"][0]["receipt"]["registry_identifier"] == identifier
    assert calls == ["original_config_loader", "original_data_processor"]


def test_multiple_benchmarks_each_bind_their_requested_original_agent_and_variant(source, monkeypatch):
    first = copy.deepcopy(source["routes"]["benchmarks"]["fictional_benchmark"])
    source["routes"]["benchmarks"]["fictional_other_agent"] = {
        **copy.deepcopy(first),
        "agent_name": "unrequested_agent",
        "dataset_name": "unrequested_agent_dataset",
    }
    source["routes"]["benchmarks"]["fictional_other_version"] = {
        **copy.deepcopy(first),
        "dataset_name": "unrequested_variant",
        "registry_identifier": {
            "backend": "gitlab",
            "dataset_name": "fictional_registered_tasks",
            "version": "0.0.9",
            "artifact_fpath": "other.jsonl",
        },
    }
    write_json(source["routes_path"], source["routes"])
    plan = utility.prepare_download_plan(
        source["routes_path"], source["gym_root"], source["root"] / "multiple-original-sources"
    )
    calls = install_native_runtime(source, monkeypatch)
    assert len(plan["jobs"]) == 3
    for job in plan["jobs"]:
        receipt = utility.download_original_source(job["worker_spec"], preflight_only=True)
        assert (
            receipt["registry_identifier"] == source["routes"]["benchmarks"][job["benchmark"]]["registry_identifier"]
        )
    assert calls == ["original_config_loader"] * 3


def make_archive(path: Path, entries: list[tuple[str, bytes | None, str | None]]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content, link_type in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            if link_type:
                member.type = tarfile.SYMTYPE if link_type == "symlink" else tarfile.LNKTYPE
                member.linkname = "../../escape"
                stream.addfile(member)
            else:
                member.size = len(content or b"")
                stream.addfile(member, io.BytesIO(content or b""))


@pytest.mark.parametrize(
    "unsafe_entry",
    [
        ("../escape.txt", b"escaped", None),
        ("/absolute-escape.txt", b"escaped", None),
        ("source/tasks/case/link", None, "symlink"),
        ("source/tasks/case/link", None, "hardlink"),
    ],
)
def test_auxiliary_archive_rejects_path_traversal_and_links(source, unsafe_entry):
    archive = source["root"] / "unsafe.tar.gz"
    make_archive(archive, [("source/tasks/case/task.toml", b"original", None), unsafe_entry])
    with pytest.raises(ValueError):
        utility.extract_registry_archive(
            archive,
            source["root"] / "unsafe-expanded",
            {
                "format": "tar.gz",
                "strip_prefix": "source",
                "select_prefixes": ["tasks/case"],
            },
        )
    assert not (source["root"] / "escape.txt").exists()


@pytest.mark.parametrize("problem", ["duplicate_selected_path", "missing_selected_prefix"])
def test_auxiliary_archive_refuses_ambiguous_or_missing_task_members(source, problem):
    entries = [("source/tasks/case/task.toml", b"first", None)]
    prefixes = ["tasks/case"]
    if problem == "duplicate_selected_path":
        entries.append(("source/tasks/case/task.toml", b"second", None))
    else:
        prefixes.append("tasks/missing")
    archive = source["root"] / "ambiguous.tar.gz"
    make_archive(archive, entries)
    with pytest.raises(ValueError):
        utility.extract_registry_archive(
            archive,
            source["root"] / "ambiguous-expanded",
            {
                "format": "tar.gz",
                "strip_prefix": "source",
                "select_prefixes": prefixes,
            },
        )


def install_binary_registry(source, monkeypatch, artifacts):
    calls = []
    module = ModuleType("nemo_gym.gitlab_utils")
    module.__file__ = str(source["gym_root"] / "nemo_gym" / "gitlab_utils.py")
    config_module = ModuleType("nemo_gym.config_types")
    config_module.__file__ = str(source["gym_root"] / "nemo_gym" / "config_types.py")

    class BinaryConfig:
        @classmethod
        def model_validate(cls, value):
            return SimpleNamespace(**value)

    def original_binary_download(config):
        calls.append(config.artifact_fpath)
        Path(config.output_fpath).write_bytes(artifacts[config.artifact_fpath])

    module.download_binary_asset = original_binary_download
    config_module.DownloadBinaryAssetGitlabConfig = BinaryConfig
    monkeypatch.setitem(sys.modules, "nemo_gym.gitlab_utils", module)
    monkeypatch.setitem(sys.modules, "nemo_gym.config_types", config_module)
    return calls


def add_registry_asset(source, *, artifact_path, content, config_key, archive=None, runtime_copy=False):
    asset = {
        "name": "fictional-original-asset",
        "config_key": config_key,
        "registry_identifier": {
            "backend": "gitlab",
            "dataset_name": "fictional_original_assets",
            "version": "1.0.0",
            "artifact_fpath": artifact_path,
        },
        "sha256": hashlib.sha256(content).hexdigest(),
        "runtime_copy": runtime_copy,
    }
    if archive:
        asset["archive"] = archive
    source["routes"]["benchmarks"]["fictional_benchmark"].setdefault("registry_assets", []).append(asset)
    write_json(source["routes_path"], source["routes"])


def test_native_auxiliary_files_and_selected_archive_tasks_reach_original_rerun_routes(source, monkeypatch):
    archive = source["root"] / "original-tasks.tar.gz"
    make_archive(
        archive,
        [
            ("source/tasks/selected/task.toml", b"Original task definition", None),
            ("source/tasks/selected/tests/reference.txt", b"Original verifier reference", None),
            ("source/tasks/unselected/task.toml", b"Excluded unrelated task", None),
        ],
    )
    artifacts = {"trajectories.json": b'{"fictional_original": [1, 2]}', "tasks.tar.gz": archive.read_bytes()}
    add_registry_asset(
        source,
        artifact_path="trajectories.json",
        content=artifacts["trajectories.json"],
        config_key="fictional_resource.ground_truth_path",
    )
    add_registry_asset(
        source,
        artifact_path="tasks.tar.gz",
        content=artifacts["tasks.tar.gz"],
        config_key="fictional_resource.task_folder",
        runtime_copy=True,
        archive={
            "format": "tar.gz",
            "strip_prefix": "source",
            "select_prefixes": ["tasks/selected"],
        },
    )
    binary_calls = install_binary_registry(source, monkeypatch, artifacts)
    plan, manifest, _ = download_fixture(source, monkeypatch)
    assert binary_calls == ["trajectories.json", "tasks.tar.gz"]
    job = plan["jobs"][0]
    cache = Path(job["worker_spec"]["cache_dir"])
    row = read_jsonl(cache / job["receipt"]["prepared_dataset_path"])[0]
    selection = source["root"] / "asset-selection.jsonl"
    write_jsonl(
        selection,
        [
            {
                "benchmark": "fictional_benchmark",
                "orig_trajectory_id": "fictional-with-original-assets",
                "orig_task_index": 4,
                "orig_rollout_index": 0,
                "dataset_row_index": 0,
                "expected_task_sha256": utility.rerun.canonical_sha256(row),
            }
        ],
    )
    binding = utility.bind_selection(selection, manifest, source["root"] / "asset-bound")
    route = json.loads(Path(binding["bound_routes_path"]).read_text())["benchmarks"]["fictional_benchmark"]
    required = route["required_files"][0]
    assert file_hash(Path(required["path"])) == required["sha256"]
    overlay = json.loads(Path(route["config_paths"][-1]).read_text())
    assert overlay["fictional_resource"]["ground_truth_path"] == required["path"]
    tree = Path(route["runtime_assets"][0]["source_dir"])
    assert (tree / "tasks" / "selected" / "tests" / "reference.txt").read_bytes() == b"Original verifier reference"
    assert not (tree / "tasks" / "unselected").exists()


@pytest.mark.parametrize("problem", ["wrong_sha256", "archive_traversal", "archive_symlink"])
def test_invalid_auxiliary_registry_assets_never_publish_a_completed_cache(source, monkeypatch, problem):
    archive = source["root"] / "corrupt-original-tasks.tar.gz"
    if problem == "archive_traversal":
        entries = [("../escape.txt", b"unsafe", None)]
    elif problem == "archive_symlink":
        entries = [("source/tasks/case/link", None, "symlink")]
    else:
        entries = [("source/tasks/case/task.toml", b"original", None)]
    make_archive(archive, entries)
    content = archive.read_bytes()
    add_registry_asset(
        source,
        artifact_path="tasks.tar.gz",
        content=content,
        config_key="fictional_resource.task_folder",
        runtime_copy=True,
        archive={
            "format": "tar.gz",
            "strip_prefix": "source",
            "select_prefixes": ["tasks/case"],
        },
    )
    if problem == "wrong_sha256":
        source["routes"]["benchmarks"]["fictional_benchmark"]["registry_assets"][0]["sha256"] = "a" * 64
        write_json(source["routes_path"], source["routes"])
    output = source["root"] / "invalid-asset-cache"
    plan = utility.prepare_download_plan(source["routes_path"], source["gym_root"], output)
    install_native_runtime(source, monkeypatch)
    install_binary_registry(source, monkeypatch, {"tasks.tar.gz": content})
    with pytest.raises(ValueError):
        utility.download_original_source(plan["jobs"][0]["worker_spec"])
    assert not Path(plan["jobs"][0]["worker_spec"]["cache_dir"]).exists()
    assert not list(output.glob(".download-*"))


def test_download_failure_manifest_never_reports_completed_or_fabricated_zero_sources(source, monkeypatch):
    original_subprocess = subprocess.run

    def failed_native_download(command, **kwargs):
        if command[0] == "git":
            return original_subprocess(command, **kwargs)
        assert command[2] == "_worker"
        assert kwargs["cwd"] == str(source["gym_root"])
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(subprocess, "run", failed_native_download)
    output = source["root"] / "failed-native-download"
    result = utility.download_sources(source["routes_path"], source["gym_root"], output, execute=True)
    job = result["jobs"][0]
    assert job["status"] == "failed" and job["returncode"] == 17
    assert "receipt" not in job
    assert not Path(job["worker_spec"]["cache_dir"]).exists()
    assert json.loads((output / "download_manifest.json").read_text())["jobs"][0]["status"] == "failed"


def test_single_preparation_uses_native_download_then_binds_original_tasks_and_reuses_cache(source, monkeypatch):
    install_native_runtime(source, monkeypatch)
    git_subprocess = subprocess.run
    worker_calls = []

    def original_worker_subprocess(command, **kwargs):
        if command[0] == "git":
            return git_subprocess(command, **kwargs)
        assert command[:3] == [sys.executable, str(SCRIPT), "_worker"]
        assert kwargs["cwd"] == str(source["gym_root"])
        job = json.loads(Path(command[command.index("--job") + 1]).read_text())
        assert command[command.index("--expected-job-sha256") + 1] == utility.rerun.canonical_sha256(job)
        worker_calls.append(job)
        utility.download_original_source(job)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", original_worker_subprocess)
    expected = copy.deepcopy(source["original_rows"][0])
    expected["agent_ref"] = {"type": "responses_api_agents", "name": "fictional_agent"}
    selection = source["root"] / "original-failures.jsonl"
    write_jsonl(
        selection,
        [
            {
                "benchmark": "fictional_benchmark",
                "orig_trajectory_id": f"fictional-failure-{attempt}",
                "orig_task_index": 9,
                "orig_rollout_index": attempt,
                "dataset_row_index": 0,
                "expected_task_sha256": utility.rerun.canonical_sha256(expected),
            }
            for attempt in [1, 3]
        ],
    )
    workspace = source["root"] / "prepare-original-failures"
    receipt = utility.prepare_failed_tasks(selection, source["routes_path"], source["gym_root"], workspace)
    assert receipt["status"] == "completed" and receipt["num_rows"] == 2 and receipt["num_task_groups"] == 1
    assert len(worker_calls) == 1
    manifest = json.loads(Path(receipt["prepared_manifest"]).read_text())
    rows = read_jsonl(Path(receipt["prepared_manifest"]).parent / manifest["jobs"][0]["input_path"])
    assert [row["orig_rollout_index"] for row in rows] == [1, 3]
    assert all(row["verifier_metadata"] == expected["verifier_metadata"] for row in rows)
    download_manifest_before = (workspace / "downloads" / "download_manifest.json").read_bytes()
    cached = utility.prepare_failed_tasks(selection, source["routes_path"], source["gym_root"], workspace)
    assert cached["status"] == "cached" and len(worker_calls) == 1
    assert (workspace / "downloads" / "download_manifest.json").read_bytes() == download_manifest_before
