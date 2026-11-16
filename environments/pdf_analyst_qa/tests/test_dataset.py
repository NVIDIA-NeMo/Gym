# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from environments.pdf_analyst_qa.prepare_example import prepare_example
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.environment.validation import validate_environment
from resources_servers.pdf_analyst_qa.task_data import PdfAnalystQaSeedRequest
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig


def test_example_copy_preserves_ready_to_run_dataset(tmp_path):
    output = prepare_example(tmp_path / "runtime")
    source = Path(__file__).resolve().parents[3] / "resources_servers/pdf_analyst_qa/data/example_bundle"
    assert output.read_bytes() == (source / "gym_input.jsonl").read_bytes()
    row = json.loads(output.read_text())
    identity = PdfAnalystQaSeedRequest.model_validate(row)
    task = json.loads((output.parent / "tasks" / f"{identity.task_id}.json").read_text())
    assert task["workspace_id"] == identity.workspace_id
    assert row["responses_create_params"]["input"][-1]["content"] == task["question"]
    assert not ((output.parent / "document_store/example-observatory.pdf").stat().st_mode & 0o222)


@pytest.mark.parametrize("dataset_type", ["example", "train"])
def test_supplied_dataset_yaml_resolves_session_route(monkeypatch, tmp_path, dataset_type):
    monkeypatch.setenv("PDF_ANALYST_QA_DOCUMENT_IMAGE", "example@sha256:" + "a" * 64)
    monkeypatch.setenv("PDF_ANALYST_QA_OPENCODE_SHA256", "b" * 64)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "fixture-key")
    output = prepare_example(tmp_path / "dataset")
    dataset = {"name": "custom_pdf_qa", "type": dataset_type, "jsonl_fpath": str(output)}
    if dataset_type == "train":
        dataset["license"] = "Apache 2.0"
    config = {
        "config_paths": ["environments/pdf_analyst_qa/config.yaml"],
        "pdf_analyst_qa_dataset_root": str(output.parent),
        "pdf_analyst_qa_document_prefix": "custom-dataset",
        "pdf_analyst_qa_agent": {"responses_api_agents": {"opencode_agent": {"datasets": [dataset]}}},
    }
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config))
    manifest = yaml.safe_load(Path("environments/pdf_analyst_qa/manifest.yaml").read_text())
    manifest["datasets"] = [{key: value for key, value in dataset.items() if key != "license"}]
    manifest_path = tmp_path / "pdf_analyst_qa" / "manifest.yaml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(yaml.safe_dump(manifest))
    report = validate_environment(manifest_path, str(config_path))
    assert report.inferred_profile == "custom-gym-agent-loop"
    assert [(item.name, item.rows) for item in report.datasets] == [("custom_pdf_qa", 1)]
    assert {item.implementation for item in report.components} == {"pdf_analyst_qa", "opencode_agent", "vllm_model"}
    base = OmegaConf.load("environments/pdf_analyst_qa/config.yaml")
    resources = OmegaConf.load("resources_servers/pdf_analyst_qa/configs/pdf_analyst_qa.yaml")
    merged = OmegaConf.merge(resources, base, config)
    assert merged.pdf_analyst_qa_resources_server.resources_servers.pdf_analyst_qa.dataset_root == str(output.parent)
    assert merged.pdf_analyst_qa_agent.responses_api_agents.opencode_agent.datasets[0].jsonl_fpath == str(output)
    assert (
        merged.pdf_analyst_qa_resources_server.resources_servers.pdf_analyst_qa.rollout_timeout_s
        == merged.pdf_analyst_qa_agent.responses_api_agents.opencode_agent.timeout
    )


@pytest.mark.parametrize("deployment_configured", [True, False])
def test_bundled_dataset_yaml_matches_catalog(monkeypatch, deployment_configured):
    if deployment_configured:
        monkeypatch.setenv("PDF_ANALYST_QA_DOCUMENT_IMAGE", "example@sha256:" + "a" * 64)
        monkeypatch.setenv("PDF_ANALYST_QA_OPENCODE_SHA256", "b" * 64)
        monkeypatch.setenv("PDF_ANALYST_QA_OPENCODE_PLUGIN_SHA256", "c" * 64)
    else:
        monkeypatch.delenv("PDF_ANALYST_QA_DOCUMENT_IMAGE", raising=False)
        monkeypatch.delenv("PDF_ANALYST_QA_OPENCODE_SHA256", raising=False)
        monkeypatch.delenv("PDF_ANALYST_QA_OPENCODE_PLUGIN_SHA256", raising=False)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "fixture-key")
    report = validate_environment("environments/pdf_analyst_qa/manifest.yaml")
    assert [(item.name, item.rows) for item in report.datasets] == [("example", 1)]
    settings = OmegaConf.load("environments/pdf_analyst_qa/config.yaml")
    agent = settings.pdf_analyst_qa_agent.responses_api_agents.opencode_agent
    assert (
        agent.sandbox_plugin_dependencies_path
        == f"/opt/cli/opencode-{agent.opencode_version}-plugin-dependencies.tar.gz"
    )
    assert agent.sandbox_plugin_dependencies_sha256 == ("c" * 64 if deployment_configured else None)


def test_configured_opencode_accepts_training_request_sampling():
    settings = OmegaConf.load("environments/pdf_analyst_qa/config.yaml")
    agent = settings.pdf_analyst_qa_agent.responses_api_agents.opencode_agent
    harness = OpenCodeAgent.model_construct(
        config=OpenCodeAgentConfig(
            name="pdf_analyst_qa_agent",
            entrypoint="app.py",
            host="127.0.0.1",
            port=12345,
            model_server={"type": "responses_api_models", "name": "policy_model"},
            opencode_version=agent.opencode_version,
            max_output_tokens=agent.max_output_tokens,
        )
    )
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Read the selected PDF.", temperature=1.0, top_p=0.9, max_output_tokens=8192
    )
    prompt, _, media = harness._session_input(body)
    assert prompt == "Read the selected PDF."
    assert media == []


@pytest.mark.parametrize(
    ("overrides", "model_host", "head_host"),
    [
        ({}, "192.0.2.20", "127.0.0.1"),
        ({"default_host": "192.0.2.30"}, "192.0.2.30", "127.0.0.1"),
        (
            {"default_host": "192.0.2.30", "head_server": {"host": "192.0.2.30"}},
            "192.0.2.30",
            "192.0.2.30",
        ),
        (
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "192.0.2.40"}}}},
            "192.0.2.40",
            "127.0.0.1",
        ),
    ],
    ids=["standalone", "default-host", "nemo-rl-head-override", "explicit-model-host"],
)
def test_control_servers_stay_on_loopback(monkeypatch, overrides, model_host, head_host):
    monkeypatch.setenv("PDF_ANALYST_QA_DOCUMENT_IMAGE", "example@sha256:" + "a" * 64)
    monkeypatch.setenv("PDF_ANALYST_QA_OPENCODE_SHA256", "b" * 64)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "fixture-key")
    monkeypatch.setattr("nemo_gym.config.global_config.gethostbyname", lambda _: "192.0.2.20")
    monkeypatch.setattr("nemo_gym.config.global_config._find_open_port_using_range", lambda *args, **kwargs: 12345)
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": ["environments/pdf_analyst_qa/example.yaml"],
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "test-model",
                    **overrides,
                }
            ),
        )
    )
    assert config.policy_model.responses_api_models.vllm_model.host == model_host
    assert config.head_server.host == head_host
    assert config.pdf_analyst_qa_resources_server.resources_servers.pdf_analyst_qa.host == "127.0.0.1"
    assert config.pdf_analyst_qa_agent.responses_api_agents.opencode_agent.host == "127.0.0.1"
    assert config.pdf_analyst_qa_environment_server.environment_servers.single_agent_turn_legacy.host == "127.0.0.1"
