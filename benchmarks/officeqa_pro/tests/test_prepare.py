# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import json
import os
import shlex
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.officeqa_pro import prepare as preparation
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.client import ServerClient
from nemo_gym.config import (
    BaseServerConfig,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    taskset_environment_server_name,
)
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.data import TrainDataProcessor
from nemo_gym.environment import validation
from nemo_gym.server_runtime import SingleAgentTurnTaskInput
from resources_servers.officeqa.verifiers.officeqa_pro_task import OfficeQAProTask
from responses_api_agents.claude_code_agent.app import ClaudeCodeAgent, ClaudeCodeAgentConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    csv_path = tmp_path / "officeqa_pro.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "question", "answer"])
        writer.writeheader()
        writer.writerows(
            [
                {"uid": "UID0002", "question": "fixture question two", "answer": "43"},
            ]
        )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    document = corpus / "report.pdf"
    document.write_bytes(b"%PDF fixture document")
    monkeypatch.setattr(preparation, "CSV_SHA256", hashlib.sha256(csv_path.read_bytes()).hexdigest())
    hashes = {document.name: hashlib.sha256(document.read_bytes()).hexdigest()}
    monkeypatch.setattr(
        preparation, "CORPUS_SHA256", hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    )
    sizes = {document.name: document.stat().st_size}
    monkeypatch.setattr(
        preparation, "DOCUMENT_MANIFEST_SHA256", hashlib.sha256(json.dumps(sizes, sort_keys=True).encode()).hexdigest()
    )
    return {
        "source_csv": csv_path,
        "corpus_root": corpus,
        "output_fpath": tmp_path / "prepared.jsonl",
        "bundle_root": tmp_path / "bundle",
    }


def test_preparation_keeps_answers_outside_document_mount(artifacts):
    output = preparation.prepare(**artifacts)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["uid"] == "UID0002"
    row = rows[0]
    assert isinstance(row["task_id"], str) and "task_input" not in row
    assert "answer" not in row and "expected_answer" not in row
    task = OfficeQAProTask.model_validate_json(
        (artifacts["bundle_root"] / "tasks" / (row["task_id"] + ".json")).read_bytes()
    )
    assert task.expected_answer == "43" and task.question == row["question"]
    assert "agent_ref" not in row
    assert [path.name for path in (artifacts["bundle_root"] / "document_store").iterdir()] == ["report.pdf"]
    assert not (artifacts["bundle_root"] / "document_store" / "report.pdf").stat().st_mode & 0o222
    assert (artifacts["corpus_root"] / "report.pdf").stat().st_mode & 0o222  # source was not chmod'ed
    original_output = output.read_bytes()
    assert preparation.prepare(**artifacts).read_bytes() == original_output


def test_public_examples_use_pdf_prompts_and_trusted_answers_without_downloading_pro_questions(artifacts, monkeypatch):
    artifacts.pop("source_csv")

    def reject_question_download(**kwargs):
        raise AssertionError("public examples must not download the gated Pro questions")

    monkeypatch.setattr(preparation, "hf_hub_download", reject_question_download)
    output = preparation.prepare(**artifacts, examples=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert output.read_bytes() == (preparation.HERE / "data/example.jsonl").read_bytes()
    assert [row["uid"] for row in rows] == [f"officeqa_example_{i}" for i in range(1, 6)]
    answers = []
    for row in rows:
        task = OfficeQAProTask.model_validate_json(
            (artifacts["bundle_root"] / "tasks" / (row["task_id"] + ".json")).read_bytes()
        )
        answers.append(task.expected_answer)
        assert task.question == row["question"]
        assert "expected_answer" not in row
        assert "../docs/" in row["responses_create_params"]["input"][0]["content"]
        assert "treasury_bulletins_parsed" not in json.dumps(row)
    assert answers == ["217.3", "26.4", "172.6", "22.9", "7.68"]
    assert [path.name for path in (artifacts["bundle_root"] / "document_store").iterdir()] == ["report.pdf"]


def test_example_configuration_prepares_and_collates_five_single_attempt_tasks(artifacts, monkeypatch):
    monkeypatch.chdir(preparation.HERE.parents[1])
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": ["benchmarks/officeqa_pro/configs/examples.yaml"],
                    "policy_model_name": "fixture-model",
                    "policy_base_url": "http://127.0.0.1:8000/v1",
                    "policy_api_key": "fixture-key",
                    "officeqa_pro_workspace_root": str(artifacts["bundle_root"]),
                    "officeqa_pro_corpus_root": str(artifacts["corpus_root"]),
                    "prepare_script_args": {"output_fpath": str(artifacts["output_fpath"])},
                }
            ),
        )
    )
    output = preparation.prepare(**OmegaConf.to_container(config.prepare_script_args, resolve=True))
    settings = get_first_server_config_dict(config, "officeqa_pro_resources_server")
    assert len(settings.datasets) == 1 and settings.datasets[0].type == "example"
    assert settings.datasets[0].num_repeats == 1
    assert taskset_environment_server_name(config, "officeqa_pro_examples") == "officeqa_pro_environment_server"
    settings.datasets[0].jsonl_fpath = str(output)
    instances = [c for c in GlobalConfigDictParser().filter_for_server_instance_configs(config) if c.datasets]
    paths = TrainDataProcessor()._collate_samples_single_type(
        "example", instances, output_dir=output.parent / "collated", task_data_validation="error"
    )
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(rows) == 5
    assert all(row["task_id"]["taskset"] == "officeqa_pro_examples" for row in rows)
    for row in rows:
        task_input = SingleAgentTurnTaskInput.model_validate(row["task_input"])
        assert "expected_answer" not in task_input.task_data and "answer" not in task_input.task_data


def test_example_preparation_rejects_an_official_csv_override(artifacts):
    with pytest.raises(ValueError, match="source_csv must be unset"):
        preparation.prepare(**artifacts, examples=True)


def test_default_source_download_is_the_pinned_huggingface_pro_dataset(artifacts, monkeypatch):
    calls = []
    staged_csv = artifacts.pop("source_csv")

    def download(**kwargs):
        calls.append(kwargs)
        return str(staged_csv)

    monkeypatch.setattr(preparation, "hf_hub_download", download)
    preparation.prepare(**artifacts)
    assert calls[0]["repo_id"] == "databricks/officeqa"
    assert calls[0]["filename"] == "officeqa_pro.csv"
    assert calls[0]["revision"] == preparation.DATASET_REVISION
    assert calls[0]["repo_type"] == "dataset"


def test_corpus_download_uses_the_same_revision(artifacts, monkeypatch):
    corpus = artifacts.pop("corpus_root")
    corpus.rename(corpus.parent / "treasury_bulletin_pdfs")
    calls = []
    monkeypatch.setattr(preparation, "snapshot_download", lambda **kwargs: calls.append(kwargs) or str(corpus.parent))
    preparation.prepare(**artifacts)
    assert calls[0]["revision"] == preparation.DATASET_REVISION
    assert calls[0]["allow_patterns"] == ["treasury_bulletin_pdfs/*.pdf"]


@pytest.mark.parametrize("source_mode", [0o400, 0o440, 0o444, 0o644])
def test_huggingface_cache_symlinks_stage_as_regular_readonly_pdfs(artifacts, monkeypatch, source_mode):
    corpus = artifacts.pop("corpus_root")
    snapshot = corpus.parent / "hub-cache" / "snapshots" / "pinned-revision"
    documents = snapshot / "treasury_bulletin_pdfs"
    documents.mkdir(parents=True)
    blob = corpus.parent / "hub-cache" / "blobs" / "pinned-pdf"
    blob.parent.mkdir(parents=True)
    (corpus / "report.pdf").rename(blob)
    original = blob.read_bytes()
    blob.chmod(source_mode)
    cached = documents / "report.pdf"
    cached.symlink_to(os.path.relpath(blob, documents))
    monkeypatch.setattr(preparation, "snapshot_download", lambda **kwargs: str(snapshot))

    preparation.prepare(**artifacts)

    staged = artifacts["bundle_root"] / "document_store" / "report.pdf"
    assert staged.is_file() and not staged.is_symlink()
    assert staged.read_bytes() == original and staged.stat().st_mode & 0o777 == 0o444
    assert cached.is_symlink() and blob.stat().st_mode & 0o777 == source_mode
    cached.unlink()
    blob.unlink()
    assert staged.read_bytes() == original  # A PDF-only mount must not need the Hub cache.


def test_preparation_uses_every_question_in_the_upstream_csv(artifacts, monkeypatch):
    with artifacts["source_csv"].open("a") as handle:
        handle.write("UID0001,another question,42\n")
    monkeypatch.setattr(preparation, "CSV_SHA256", hashlib.sha256(artifacts["source_csv"].read_bytes()).hexdigest())
    rows = [json.loads(line) for line in preparation.prepare(**artifacts).read_text().splitlines()]
    assert [row["uid"] for row in rows] == ["UID0002", "UID0001"]


@pytest.mark.parametrize(
    "rows",
    ["", "UID0002,question,43\nUID0002,duplicate,42\n", "UID0002,question,\n", ",question,43\n"],
)
def test_preparation_rejects_empty_or_invalid_questions(artifacts, rows):
    artifacts["source_csv"].write_text("uid,question,answer\n" + rows)
    with pytest.raises(ValueError, match="no questions|unique UIDs"):
        preparation.prepare(**artifacts)


def test_prompt_materialization_cannot_include_hidden_answer(artifacts):
    output = preparation.prepare(**artifacts)
    row = json.loads(output.read_text())
    task_data = row
    messages = row["responses_create_params"]["input"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert messages[0]["content"].endswith("Question: " + task_data["question"])
    rendered = json.dumps(messages)
    assert "43" not in rendered and "officeqa" not in rendered.lower()
    assert "../docs/" in rendered and "<FINAL_ANSWER>" in rendered
    assert task_data["prompt_version"].startswith("officeqa-pro-")
    assert "temperature" not in row["responses_create_params"]
    assert "top_p" not in row["responses_create_params"]
    assert "tools" not in row["responses_create_params"]


@pytest.mark.parametrize("network_access", [None, "inherit"])
def test_default_binding_accepts_prepared_task_without_sampling_overrides(artifacts, monkeypatch, network_access):
    monkeypatch.chdir(preparation.HERE.parents[1])
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": ["benchmarks/officeqa_pro/config.yaml"],
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "fixture-model",
                    "officeqa_opencode_binary_sha256": "b" * 64,
                    **({"officeqa_network_access": network_access} if network_access is not None else {}),
                }
            ),
        )
    )
    settings = get_first_server_config_dict(config, "officeqa_pro_opencode_agent")
    agent_config = OpenCodeAgentConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": "officeqa_pro_opencode_agent"}
    )
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=config)
    agent = OpenCodeAgent(config=agent_config, server_client=client)
    assert agent.config.sandbox_executable_path == "/opt/cli/opencode"
    assert agent.config.sandbox_executable_sha256 == "b" * 64
    row = json.loads(preparation.prepare(**artifacts).read_text())
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    prompt, system = agent._session_input(body)
    assert prompt == body.input[0].content
    assert system == ""
    resource_settings = get_first_server_config_dict(config, "officeqa_pro_resources_server")
    assert resource_settings.network_access == (network_access or "model_only")
    permission = agent.config.opencode_config["permission"]
    expected_permission = "allow" if network_access == "inherit" else "deny"
    assert permission["webfetch"] == permission["websearch"] == expected_permission

    settings = get_first_server_config_dict(config, "policy_model")
    assert settings.get("sampling_overrides") is None

    settings = get_first_server_config_dict(config, "officeqa_pro_environment_server")
    SingleAgentTurnEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": "officeqa_pro_environment_server"}
    )


def test_prepared_source_collates_through_stock_gym_with_stable_task_identity(artifacts, monkeypatch):
    monkeypatch.chdir(preparation.HERE.parents[1])
    output = preparation.prepare(**artifacts)
    original = output.read_bytes()
    row = json.loads(original)
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": ["benchmarks/officeqa_pro/config.yaml"],
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "fixture-model",
                }
            ),
        )
    )
    config.officeqa_pro_resources_server.resources_servers.officeqa.datasets[0].jsonl_fpath = str(output)
    instances = [
        instance
        for instance in GlobalConfigDictParser().filter_for_server_instance_configs(config)
        if instance.datasets
    ]
    paths = TrainDataProcessor()._collate_samples_single_type(
        "benchmark", instances, output_dir=output.parent / "collated", task_data_validation="error"
    )
    assert len(paths) == 1
    tasks = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(tasks) == 5
    assert all(task["task_id"] == {"taskset": "officeqa_pro", "task_id": row["task_id"]} for task in tasks)
    task = tasks[0]
    assert task["task_id"] == {"taskset": "officeqa_pro", "task_id": row["task_id"]}
    task_input = SingleAgentTurnTaskInput.model_validate(task["task_input"])
    assert task_input.task_data["uid"] == "UID0002"
    assert "expected_answer" not in task_input.task_data and "answer" not in task_input.task_data
    assert task_input.responses_create_params.input[0].content == row["responses_create_params"]["input"][0]["content"]
    assert output.read_bytes() == original


@pytest.mark.parametrize("network_access", [None, "inherit"])
def test_claude_binding_uses_the_same_prepared_questions_and_stock_agents(artifacts, monkeypatch, network_access):
    monkeypatch.chdir(preparation.HERE.parents[1])
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": ["benchmarks/officeqa_pro/configs/claude.yaml"],
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "fixture-model",
                    **({"officeqa_network_access": network_access} if network_access is not None else {}),
                    "officeqa_claude_binary_sha256": "a" * 64,
                }
            ),
        )
    )
    assert {instance.name for instance in GlobalConfigDictParser().filter_for_server_instance_configs(config)} == {
        "officeqa_pro_claude_code_agent",
        "officeqa_pro_environment_server",
        "officeqa_pro_resources_server",
        "policy_model",
    }
    settings = get_first_server_config_dict(config, "officeqa_pro_claude_code_agent")
    agent = ClaudeCodeAgent(
        config=ClaudeCodeAgentConfig.model_validate(
            OmegaConf.to_container(settings, resolve=True) | {"name": "officeqa_pro_claude_code_agent"}
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=0), global_config_dict=config
        ),
    )
    row = json.loads(preparation.prepare(**artifacts).read_text())
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    prompt, system = agent._conversation_input(body)
    assert prompt == body.input[0].content and system is None
    resource_settings = get_first_server_config_dict(config, "officeqa_pro_resources_server")
    assert resource_settings.network_access == (network_access or "model_only")
    assert agent.config.disallowed_tools == (None if network_access == "inherit" else "WebSearch,WebFetch")
    assert agent.config.sandbox_executable_path == "/opt/cli/claude"
    assert agent.config.sandbox_executable_sha256 == "a" * 64
    assert agent.config.settings is None
    assert get_first_server_config_dict(config, "policy_model").is_responses_native
    assert get_first_server_config_dict(config, "policy_model").get("sampling_overrides") is None
    settings = get_first_server_config_dict(config, "officeqa_pro_environment_server")
    environment = SingleAgentTurnEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": "officeqa_pro_environment_server"}
    )
    assert environment.agent_server.name == "officeqa_pro_claude_code_agent"
    assert environment.resources_server.name == "officeqa_pro_resources_server"


def test_static_manifest_validation_uses_prepared_inputs_and_actual_wiring(artifacts, monkeypatch):
    output = preparation.prepare(**artifacts)
    original = validation._resolve_under_cwd_or_install

    def resolve(path):
        if str(path) == "benchmarks/officeqa_pro/data/benchmark.jsonl":
            return output
        return original(path)

    monkeypatch.setattr(validation, "_resolve_under_cwd_or_install", resolve)
    report = validation.validate_environment(preparation.HERE / "manifest.yaml")
    assert report.name == "officeqa_pro" and report.inferred_profile == "custom-gym-agent-loop"
    assert report.datasets[0].rows == 1
    assert {component.name for component in report.components} >= {
        "officeqa_pro_resources_server",
        "officeqa_pro_opencode_agent",
        "policy_model",
    }


def test_preparation_rejects_an_empty_corpus(artifacts):
    (artifacts["corpus_root"] / "report.pdf").unlink()
    with pytest.raises(ValueError, match="no PDFs"):
        preparation.prepare(**artifacts)


def test_offline_preparation_rejects_a_changed_document(artifacts):
    preparation.prepare(**artifacts)
    (artifacts["corpus_root"] / "report.pdf").write_bytes(b"updated document")
    with pytest.raises(ValueError, match="corpus differs from the pinned release"):
        preparation.prepare(**artifacts)


@pytest.mark.parametrize("change", ["extra_pdf", "unrelated_file", "directory", "file_symlink", "store_symlink"])
def test_reused_document_store_cannot_expose_unrelated_files_or_links(artifacts, change):
    preparation.prepare(**artifacts)
    store = artifacts["bundle_root"] / "document_store"
    if change == "extra_pdf":
        (store / "extra.pdf").write_bytes(b"unrelated document")
    elif change == "unrelated_file":
        (store / "answers.json").write_text('{"answer": "43"}')
    elif change == "directory":
        (store / "unrelated").mkdir()
    elif change == "file_symlink":
        document = store / "report.pdf"
        original = artifacts["corpus_root"] / "report.pdf"
        original.chmod(0o444)
        document.unlink()
        document.symlink_to(original)
    else:
        original_store = store.with_name("original_store")
        store.rename(original_store)
        store.symlink_to(original_store, target_is_directory=True)
    with pytest.raises(ValueError, match="document_store"):
        preparation.prepare(**artifacts)


def test_offline_preparation_rejects_a_changed_answer(artifacts):
    path = artifacts["source_csv"]
    path.write_text(path.read_text().replace(",43", ",99"))
    with pytest.raises(ValueError, match="CSV differs from the pinned release"):
        preparation.prepare(**artifacts)


@pytest.mark.parametrize("change", ["rename", "extra"])
def test_offline_preparation_checks_document_names_and_membership(artifacts, change):
    corpus = artifacts["corpus_root"]
    if change == "rename":
        (corpus / "report.pdf").rename(corpus / "renamed.pdf")
    else:
        (corpus / "extra.pdf").write_bytes(b"unexpected document")
    with pytest.raises(ValueError, match="corpus differs from the pinned release"):
        preparation.prepare(**artifacts)


def test_preparation_rejects_an_outdated_document_manifest(artifacts, monkeypatch):
    monkeypatch.setattr(preparation, "DOCUMENT_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="document manifest differs from the pinned release"):
        preparation.prepare(**artifacts)


def test_preparation_and_deployment_use_the_same_document_manifest():
    config = OmegaConf.load(preparation.HERE / "config.yaml")
    assert config.officeqa_document_manifest_sha256 == preparation.DOCUMENT_MANIFEST_SHA256


@pytest.mark.parametrize("change", [None, "missing", "extra", "renamed", "truncated", "symlink"])
def test_mount_probe_checks_filenames_and_sizes_without_reading_pdfs(artifacts, monkeypatch, capsys, change):
    preparation.prepare(**artifacts)
    store = artifacts["bundle_root"] / "document_store"
    document = store / "report.pdf"
    config = OmegaConf.load(preparation.HERE / "config.yaml")
    config.officeqa_document_manifest_sha256 = preparation.DOCUMENT_MANIFEST_SHA256
    source = shlex.split(config.sandbox.opensandbox.probe.command)[-1].replace('"/workspace/docs"', repr(str(store)))
    if change == "missing":
        document.unlink()
    elif change == "extra":
        (store / "extra.pdf").write_bytes(b"extra document")
    elif change == "renamed":
        document.rename(store / "renamed.pdf")
    elif change == "truncated":
        document.chmod(0o644)
        document.write_bytes(b"%PDF")
    elif change == "symlink":
        document.unlink()
        document.symlink_to(artifacts["corpus_root"] / "report.pdf")

    def reject_content_read(*args, **kwargs):
        raise AssertionError("readiness probe must not read PDF contents")

    with monkeypatch.context() as patch:
        # A local temporary directory stands in for the read-only provider mount.
        patch.setattr(os, "access", lambda path, mode: False)
        patch.setattr(Path, "open", reject_content_read)
        if change is None:
            exec(source)
        else:
            with pytest.raises(AssertionError, match="^$"):
                exec(source)
    assert capsys.readouterr().out == (
        config.sandbox.opensandbox.probe.expected_stdout + "\n" if change is None else ""
    )
