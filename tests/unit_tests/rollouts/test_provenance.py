# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import time
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import TypeAdapter

from nemo_gym.config.config_types import HeadServerUnreachableError
from nemo_gym.rollouts import provenance
from nemo_gym.utils.path_utils import materialized_path_for, provenance_path_for


@pytest.fixture(autouse=True)
def known_commit(monkeypatch):
    monkeypatch.setattr(provenance, "installed_gym_commit", lambda: "test-commit")


def test_provenance_hashes_materialized_inputs_and_selects_safe_metadata(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_bytes(
        orjson.dumps(
            {
                "task_source": "dataset-v2",
                "responses_create_params": {
                    "input": "private prompt",
                    "temperature": 0.7,
                    "metadata": {"extra_body": '{"seed": 23, "api_key": "secret"}'},
                },
            }
        )
        + b"\n"
    )
    config = {
        "seed": 42,
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "model": "policy",
                    "checkpoint": "checkpoint-10",
                    "served_model_name": "served",
                    "api_key": "secret",
                    "base_url": "https://user:secret@host/?token=secret",  # pragma: allowlist secret
                    "default_headers": {"Authorization": "secret"},
                    "sampling_overrides": {"temperature": 0.2},
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            }
        },
        "judge_model": {
            "responses_api_models": {"openai_model": {"openai_model": "judge", "openai_api_key": "secret"}}
        },
        "verifier": {"judge_prompt_version": "v3", "judge_prompt_template": "private judge prompt"},
        "env": {"model": "secret", "API_TOKEN": "secret"},
    }
    record = provenance.build_provenance(config, materialized_fpath=inputs)
    assert record.gym_commit == "test-commit"
    assert record.seed == 42
    assert record.datasets == ["dataset-v2"]
    assert record.materialized_inputs_sha256 == sha256(inputs.read_bytes()).hexdigest()
    assert record.models["policy_model.responses_api_models.vllm_model"] == {
        "model": "policy",
        "checkpoint": "checkpoint-10",
        "served_model_name": "served",
    }
    assert record.models["judge_model.responses_api_models.openai_model"]["openai_model"] == "judge"
    assert record.judge_prompt_versions == {"verifier.judge_prompt_version": "v3"}
    assert record.input_sampling_parameters == [{"temperature": 0.7, "seed": 23}]
    assert record.sampling_parameters["policy_model.responses_api_models.vllm_model.sampling_overrides"] == {
        "temperature": 0.2
    }
    serialized = record.model_dump_json()
    assert "secret" not in serialized
    assert "private" not in serialized
    assert "api_key" not in serialized
    inputs.write_bytes(inputs.read_bytes().replace(b"private prompt", b"different prompt"))
    assert (
        provenance.build_provenance(config, materialized_fpath=inputs).materialized_inputs_sha256
        != record.materialized_inputs_sha256
    )


def test_provenance_uses_applied_template_and_canonical_artifact_name(tmp_path: Path) -> None:
    output = tmp_path / "custom.jsonl"
    materialized_path_for(output).write_bytes(b"{}\n")
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text('user: "Question: {question}"\n')
    path = provenance.write_run_provenance(output, {}, prompt_config=str(prompt))
    assert path == provenance_path_for(output) == tmp_path / "custom_provenance.json"
    payload = orjson.loads(path.read_bytes())
    assert payload["prompt_config"]["path"] == str(prompt)
    assert len(payload["prompt_config"]["sha256"]) == 64
    assert payload["models"] == {}
    assert payload["seed"] is None
    assert payload["materialized_inputs_sha256"] == sha256(b"{}\n").hexdigest()


def test_provenance_accepts_hydra_template_settings() -> None:
    config = {
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "model": "policy",
                    "chat_template_kwargs": {"enable_thinking": True, "options": [1, 2]},
                }
            }
        }
    }
    assert provenance.build_provenance(OmegaConf.create(config)) == provenance.build_provenance(config)


def test_submission_provenance_keeps_benchmark_dataset_prompt_and_seed_settings() -> None:
    record = provenance.build_provenance(
        {
            "driver": {
                "benchmarks": {
                    "benchmark": {
                        "run": {
                            "input_jsonl_fpath": "dataset-v2.jsonl",
                            "prompt_config": "prompts/v2.yaml",
                            "seed": 7,
                            "policy_model_name": "served-policy",
                            "policy_api_key": "secret",
                        }
                    }
                }
            }
        }
    )
    assert record.datasets == ["dataset-v2.jsonl"]
    assert record.prompt_config_paths == {"driver.benchmarks.benchmark.run.prompt_config": "prompts/v2.yaml"}
    assert record.sampling_parameters["driver.benchmarks.benchmark.run"] == {"seed": 7}
    assert record.models["driver.benchmarks.benchmark.run"] == {"policy_model_name": "served-policy"}
    assert record.materialized_inputs_sha256 is None
    assert not provenance.build_provenance({"chat_template_kwargs": None}).template_sha256
    assert "secret" not in record.model_dump_json()


def test_review_provenance_failure_writes_safe_unavailable_record(tmp_path: Path, caplog) -> None:
    output = tmp_path / "rollouts.jsonl"
    materialized_path_for(output).write_bytes(b"{}\n")
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text("user: [secret]\n")
    path = provenance.write_run_provenance(output, {}, prompt_config=str(prompt))
    payload = orjson.loads(path.read_bytes())
    assert payload["status"] == "unavailable"
    assert payload["error_type"]
    assert "secret" not in path.read_text() + caplog.text


def test_review_provenance_bounds_row_decoding_but_hashes_all_bytes(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_bytes(b'{"responses_create_params":{"temperature":0.5}}\n' * 105)
    original = orjson.loads
    decoded = []

    def decode(payload):
        decoded.append(payload)
        return original(payload)

    monkeypatch.setattr(provenance.orjson, "loads", decode)
    record = provenance.build_provenance({}, materialized_fpath=inputs)
    assert len(decoded) <= 100
    assert record.materialized_inputs_sha256 == sha256(inputs.read_bytes()).hexdigest()
    assert record.input_sampling_rows == 100


def test_an_explicit_null_sampling_cap_reads_differently_from_an_unset_one() -> None:
    record = provenance.build_provenance(
        {"policy": {"max_tokens": None, "temperature": 0.6}, "judge": {"temperature": 0.0}}
    )
    assert record.sampling_parameters == {
        "policy": {"max_tokens": None, "temperature": 0.6},
        "judge": {"temperature": 0.0},
    }


def test_concurrency_limits_are_recorded_by_path() -> None:
    record = provenance.build_provenance(
        {
            "num_samples_in_parallel": 64,
            "policy_model": {"responses_api_models": {"openai_model": {"max_concurrent_requests": 128}}},
            "episodes": {"environment_servers": {"legacy_agent": {"max_concurrent_episodes": None}}},
            "judge": {"resources_servers": {"math_with_judge": {"library_verifier_max_concurrency": True}}},
        }
    )
    assert record.concurrency_limits == {
        provenance.TOP_LEVEL_PATH: {"num_samples_in_parallel": 64},
        "policy_model.responses_api_models.openai_model": {"max_concurrent_requests": 128},
        "episodes.environment_servers.legacy_agent": {"max_concurrent_episodes": None},
    }


def test_concurrency_limits_include_worker_counts_and_server_specific_limit_names() -> None:
    record = provenance.build_provenance(
        {
            "num_samples_in_parallel": 64,
            "max_resident_rollout_tasks": 256,
            "policy_model": {
                "responses_api_models": {
                    "openai_model": {"num_workers": 4, "max_concurrent_requests": 32, "tensor_parallel_size": 8}
                }
            },
            "stem_judge": {"resources_servers": {"stem_judge": {"judge_endpoint_max_concurrency": 64}}},
            "frontiermath": {"resources_servers": {"frontiermath": {"verifier_max_concurrency": 16}}},
            "gdpval": {"resources_servers": {"gdpval": {"num_workers": None, "preconvert_max_concurrent": 4}}},
            "bird_sql": {"resources_servers": {"bird_sql": {"max_concurrency": 32}}},
            "rna_design": {
                "responses_api_agents": {
                    "rna_design_agent": {"max_concurrent_rollouts": 8, "parallel_tool_calls": True}
                }
            },
            "goose": {"responses_api_agents": {"goose_agent": {"concurrency": 32, "max_concurrent_flag": True}}},
            "sandbox": {"docker": {"exec": {"default_timeout_s": 180, "concurrency": 32}}},
            "judge": {"resources_servers": {"sciknoweval": {"env": {"judge_max_concurrency": 3}}}},
        }
    )
    assert record.concurrency_limits == {
        provenance.TOP_LEVEL_PATH: {"num_samples_in_parallel": 64, "max_resident_rollout_tasks": 256},
        "policy_model.responses_api_models.openai_model": {"num_workers": 4, "max_concurrent_requests": 32},
        "stem_judge.resources_servers.stem_judge": {"judge_endpoint_max_concurrency": 64},
        "frontiermath.resources_servers.frontiermath": {"verifier_max_concurrency": 16},
        "gdpval.resources_servers.gdpval": {"num_workers": None, "preconvert_max_concurrent": 4},
        "bird_sql.resources_servers.bird_sql": {"max_concurrency": 32},
        "rna_design.responses_api_agents.rna_design_agent": {"max_concurrent_rollouts": 8},
        "goose.responses_api_agents.goose_agent": {"concurrency": 32},
        "sandbox.docker.exec": {"concurrency": 32},
    }


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("config_fpath", "path", "key", "limit"),
    [
        (
            "benchmarks/healthbench/configs/nemotron-ultra.yaml",
            "policy_model.responses_api_models.inference_provider",
            "num_concurrent_requests",
            4,
        ),
        (
            "responses_api_models/azure_openai_model/configs/azure_openai_model.yaml",
            "policy_model.responses_api_models.azure_openai_model",
            "num_concurrent_requests",
            8,
        ),
        (
            "benchmarks/lmarena_v3/config_super35_gemini38flash_judge.yaml",
            "lmarena_v3_benchmark_resources_server.resources_servers.arena",
            "judge_concurrency",
            32,
        ),
        ("nemo_gym/sandbox/providers/bwrap/configs/bwrap.yaml", "sandbox.bwrap", "exec_concurrency", 256),
        (
            "nemo_gym/sandbox/providers/openshell/configs/openshell.yaml",
            "sandbox.openshell.operations",
            "delete_concurrency",
            32,
        ),
        (
            "resources_servers/vision2code/configs/vision2code.yaml",
            "vision2code.resources_servers.vision2code",
            "render_concurrency",
            8,
        ),
        (
            "resources_servers/vision2code/configs/vision2code.yaml",
            "vision2code.resources_servers.vision2code",
            "image_metrics_concurrency",
            2,
        ),
        (
            "responses_api_agents/collab_intel_agent/configs/collab_intel_agent.yaml",
            "collab_intel_agent.responses_api_agents.collab_intel_agent",
            "max_concurrent",
            16,
        ),
        (
            "resources_servers/competitive_coding_challenges/configs/competitive_coding_challenges.yaml",
            "competitive_coding_challenges_resources_server.resources_servers.competitive_coding_challenges",
            "num_parallel_requests",
            16,
        ),
        ("resources_servers/legal_agent_bench/configs/legal_agent_bench.yaml", "lab", "judge_parallelism", 6),
        (
            "responses_api_agents/conversational_tool_use_policy_tool_generation/configs/"
            "conversational_tool_use_policy_tool_generation.yaml",
            "conversational_tool_use_policy_tool_generation.responses_api_agents."
            "conversational_tool_use_policy_tool_generation",
            "max_judge_concurrency",
            None,
        ),
        (
            "responses_api_agents/qwen_code_agent/configs/qwen_code_agent.yaml",
            "qwen_code_agent.responses_api_agents.qwen_code_agent",
            "max_parallel_agents",
            None,
        ),
    ],
)
def test_a_shipped_config_limit_under_its_own_name_is_recorded(config_fpath, path, key, limit) -> None:
    node = OmegaConf.select(OmegaConf.load(REPO_ROOT / config_fpath), path)
    config: dict = {}
    for part in reversed(path.split(".")):
        config = {part: config or {key: node[key]}}

    record = provenance.build_provenance(config)

    assert record.concurrency_limits == {path: {key: limit}}


def test_limits_named_only_in_their_config_models_are_recorded() -> None:
    record = provenance.build_provenance(
        {
            "global_aiohttp_intended_concurrency": 512,
            "global_aiohttp_intended_concurrency_per_host": 64,
            "mm_coding": {
                "resources_servers": {
                    "mm_coding_training": {"verification_concurrency": 8, "checkpoint": {"parallelism": 16}}
                }
            },
            "vcqa": {"responses_api_agents": {"vcqa_agent": {"sandbox_upload_concurrency": 4}}},
            "sandbox": {"ecs_fargate": {"build_parallelism": 50}},
        }
    )
    assert record.concurrency_limits == {
        provenance.TOP_LEVEL_PATH: {
            "global_aiohttp_intended_concurrency": 512,
            "global_aiohttp_intended_concurrency_per_host": 64,
        },
        "mm_coding.resources_servers.mm_coding_training": {"verification_concurrency": 8},
        "mm_coding.resources_servers.mm_coding_training.checkpoint": {"parallelism": 16},
        "vcqa.responses_api_agents.vcqa_agent": {"sandbox_upload_concurrency": 4},
        "sandbox.ecs_fargate": {"build_parallelism": 50},
    }


def _write_without_probing(output: Path, config, monkeypatch) -> dict:
    monkeypatch.setattr(provenance, "served_models", lambda servers: {})
    materialized_path_for(output).write_bytes(b"")
    return orjson.loads(provenance.write_run_provenance(output, config).read_bytes())


@pytest.mark.parametrize("configured", ["6", " 6 ", "6.0", None])
def test_gdpval_judge_request_limit_is_recorded_as_its_model_server_reads_it(
    configured, tmp_path: Path, monkeypatch
) -> None:
    from responses_api_models.openai_model.app import SimpleModelServerConfig

    if configured is None:
        monkeypatch.delenv("GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS", configured)
    judge = OmegaConf.load(REPO_ROOT / "benchmarks/gdpval/config.yaml").gdpval_gemini31_judge_model
    config = OmegaConf.create({"gdpval_gemini31_judge_model": judge})
    server = SimpleModelServerConfig.model_validate(
        {
            **OmegaConf.to_container(judge.responses_api_models.openai_model, resolve=True),
            "name": "gdpval_gemini31_judge_model",
            "host": "127.0.0.1",
            "port": 8000,
        }
    )

    record = _write_without_probing(tmp_path / "rollouts.jsonl", config, monkeypatch)

    assert server.max_concurrent_requests == (2 if configured is None else 6)
    assert record["concurrency_limits"] == {
        "gdpval_gemini31_judge_model.responses_api_models.openai_model": {
            "max_concurrent_requests": server.max_concurrent_requests
        }
    }


@pytest.mark.parametrize("configured", ["7", " 8 ", None])
@pytest.mark.parametrize(
    ("config_fpath", "path", "config_model"),
    [
        (
            "benchmarks/gdpval/config.yaml",
            "gdpval_resources_server.resources_servers.gdpval",
            "resources_servers.gdpval.app:GDPValResourcesServerConfig",
        ),
        (
            "benchmarks/aa_briefcase_lite/config.yaml",
            "aa_briefcase_lite_resources_server.resources_servers.aa_briefcase_lite",
            "resources_servers.aa_briefcase_lite.app:AABriefcaseLiteResourcesServerConfig",
        ),
    ],
)
def test_the_judge_sampling_seed_is_recorded_as_its_resources_server_reads_it(
    config_fpath, path, config_model, configured, tmp_path: Path, monkeypatch
) -> None:
    module, name = config_model.split(":")
    field = getattr(importlib.import_module(module), name).model_fields["judge_sampling_seed"]
    if configured is None:
        monkeypatch.delenv("JUDGE_SAMPLING_SEED", raising=False)
    else:
        monkeypatch.setenv("JUDGE_SAMPLING_SEED", configured)
    shipped = OmegaConf.to_container(OmegaConf.select(OmegaConf.load(REPO_ROOT / config_fpath), path), resolve=False)
    config: dict = {"judge_sampling_seed": shipped["judge_sampling_seed"]}
    for part in reversed(path.split(".")):
        config = {part: config}
    config = OmegaConf.create(config)

    record = _write_without_probing(tmp_path / "rollouts.jsonl", config, monkeypatch)

    seed = TypeAdapter(field.annotation).validate_python(OmegaConf.select(config, f"{path}.judge_sampling_seed"))
    assert seed == (None if configured is None else int(configured))
    assert record["sampling_parameters"] == {path: {"judge_sampling_seed": seed}}


def test_a_setting_that_cannot_be_resolved_is_left_out_without_losing_the_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PROVENANCE_UNSET_LIMIT", raising=False)
    config = OmegaConf.create(
        {
            "seed": "${oc.env:PROVENANCE_UNSET_LIMIT}",
            "num_samples_in_parallel": 16,
            "judge": {
                "responses_api_models": {
                    "openai_model": {
                        "openai_model": "judge",
                        "max_concurrent_requests": "${oc.env:PROVENANCE_UNSET_LIMIT}",
                        "num_workers": 2,
                        "persist_dir": "???",
                        "extra_body": {"temperature": "${missing.temperature}", "top_p": 0.9},
                    }
                }
            },
        }
    )

    record = _write_without_probing(tmp_path / "rollouts.jsonl", config, monkeypatch)

    assert record["status"] == "available"
    assert record["seed"] is None
    assert record["models"] == {"judge.responses_api_models.openai_model": {"openai_model": "judge"}}
    assert record["concurrency_limits"] == {
        provenance.TOP_LEVEL_PATH: {"num_samples_in_parallel": 16},
        "judge.responses_api_models.openai_model": {"num_workers": 2},
    }
    assert record["sampling_parameters"] == {"judge.responses_api_models.openai_model.extra_body": {"top_p": 0.9}}


def test_a_sampling_value_gym_forwards_as_given_is_recorded_only_when_it_is_a_number(
    tmp_path: Path, monkeypatch
) -> None:
    from responses_api_models.openai_model.app import SimpleModelServerConfig

    config = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "benchmarks/legal_agent_bench/model_sampling_openai.yaml"),
        {
            "lab": {"policy_temperature": "0.6", "policy_top_p": 0.95},
            "policy_model": {"responses_api_models": {"openai_model": {"request_overrides": {"seed": "7", "n": 1}}}},
        },
    )
    server = SimpleModelServerConfig.model_validate(
        {
            **OmegaConf.to_container(config.policy_model.responses_api_models.openai_model, resolve=True),
            "name": "policy_model",
            "host": "127.0.0.1",
            "port": 8000,
            "entrypoint": "app.py",
            "openai_base_url": "http://127.0.0.1:1/v1",
            "openai_api_key": "dummy",  # pragma: allowlist secret
            "openai_model": "policy",
        }
    )

    record = _write_without_probing(tmp_path / "rollouts.jsonl", config, monkeypatch)

    assert server.extra_body["temperature"] == "0.6"
    assert server.request_overrides["seed"] == "7"
    assert record["sampling_parameters"] == {
        "policy_model.responses_api_models.openai_model.extra_body": {"top_p": 0.95},
        "policy_model.responses_api_models.openai_model.request_overrides": {"n": 1},
    }


def test_a_quoted_number_in_row_metadata_is_left_out_without_losing_the_record(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    inputs = materialized_path_for(output)
    extra_body = orjson.dumps({"temperature": "123456789012345678901", "top_p": 0.9}).decode()
    inputs.write_bytes(
        orjson.dumps({"responses_create_params": {"temperature": "0.6", "metadata": {"extra_body": extra_body}}})
        + b"\n"
    )

    record = orjson.loads(provenance.write_run_provenance(output, {"model": "policy"}).read_bytes())

    assert record["status"] == "available"
    assert record["materialized_inputs_sha256"] == sha256(inputs.read_bytes()).hexdigest()
    assert record["input_sampling_parameters"] == [{"top_p": 0.9}]


def test_a_row_whose_settings_cannot_be_written_is_skipped_on_its_own(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "rollouts.jsonl"
    materialized_path_for(output).write_bytes(
        b'{"responses_create_params":{"seed":1}}\n{"responses_create_params":{"temperature":0.5}}\n'
    )
    sampling = provenance._sampling
    monkeypatch.setattr(
        provenance,
        "_sampling",
        lambda config, **kwargs: {"seed": 2**70} if config.get("seed") == 1 else sampling(config, **kwargs),
    )

    record = orjson.loads(provenance.write_run_provenance(output, {"model": "policy"}).read_bytes())

    assert record["status"] == "available"
    assert record["input_sampling_rows_skipped"] == 1
    assert record["input_sampling_parameters"] == [{"temperature": 0.5}]


def test_strings_are_read_as_numbers_only_where_a_server_config_validates_them() -> None:
    record = provenance.build_provenance(
        {
            "seed": "7",
            "temperature": "0.6",
            "judge": {
                "resources_servers": {
                    "judge": {
                        "max_concurrent_requests": "${oc.env:UNRESOLVED}",
                        "max_concurrency": "two",
                        "concurrency": "1.5",
                        "num_workers": "1_000",
                        "judge_max_concurrency": "-1",
                        "judge_endpoint_max_concurrency": 6.0,
                        "verifier_max_concurrency": 6.5,
                        "preconvert_max_concurrent": True,
                        "max_concurrent_episodes": None,
                        "max_tokens": "inf",
                        "temperature": " 0.6 ",
                        "top_p": "1e999",
                        "top_k": "40.5",
                        "min_p": float("inf"),
                        "presence_penalty": "-0.5",
                        "repetition_penalty": "true",
                        "seed": "42",
                        "judge_responses_create_params": {"temperature": "0.2", "top_p": 0.8},
                    }
                }
            },
        }
    )
    assert record.seed is None
    assert record.concurrency_limits == {
        "judge.resources_servers.judge": {
            "num_workers": 1000,
            "judge_max_concurrency": -1,
            "judge_endpoint_max_concurrency": 6,
            "max_concurrent_episodes": None,
        }
    }
    assert record.sampling_parameters == {
        "judge.resources_servers.judge": {"temperature": 0.6, "presence_penalty": -0.5, "seed": 42},
        "judge.resources_servers.judge.judge_responses_create_params": {"top_p": 0.8},
    }


def test_review_provenance_excludes_boolean_sampling_values() -> None:
    record = provenance.build_provenance({"seed": True, "temperature": False, "top_p": 0.9})
    assert record.sampling_parameters == {provenance.TOP_LEVEL_PATH: {"top_p": 0.9}}


@pytest.mark.parametrize(
    "identifier",
    ["user:secret@host/model?token=secret", "https://user:secret@host/model?token=secret"],  # pragma: allowlist secret
)
def test_review_provenance_removes_schemeless_userinfo(identifier) -> None:
    record = provenance.build_provenance({"model": identifier})
    assert "secret" not in record.model_dump_json()


@pytest.mark.parametrize("host", ["github.com/org/repo", "localhost:8080/model", "127.0.0.1/model"])
def test_provenance_strips_token_only_userinfo(host) -> None:
    record = provenance.build_provenance({"checkpoint": "ghp_TOKEN@" + host})
    assert record.models[provenance.TOP_LEVEL_PATH]["checkpoint"] == host
    assert "ghp_TOKEN" not in record.model_dump_json()


def test_provenance_skips_malformed_sampling_metadata_and_keeps_input_hash(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    inputs = materialized_path_for(output)
    inputs.write_bytes(
        b'{"responses_create_params":{"temperature":0.7,"metadata":"secret"}}\n'
        b'{"responses_create_params":{"temperature":0.5,"metadata":{}}}\n'
    )
    path = provenance.write_run_provenance(output, {"model": "policy"})
    record = orjson.loads(path.read_bytes())
    assert record["status"] == "available"
    assert record["materialized_inputs_sha256"] == sha256(inputs.read_bytes()).hexdigest()
    assert record["input_sampling_rows"] == 2
    assert record["input_sampling_rows_skipped"] == 1
    assert record["input_sampling_parameters"] == [{"temperature": 0.5}]
    assert record["models"] == {provenance.TOP_LEVEL_PATH: {"model": "policy"}}
    assert "secret" not in path.read_text()


def test_provenance_holds_no_config_copy_and_no_credentials() -> None:
    """The masked config is saved beside the rollouts on its own; provenance keeps only selected identifiers."""
    config = OmegaConf.create(
        {
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "base_url": "https://user:hunter2@inference.example/v1?token=abc",  # pragma: allowlist secret
                        "api_key": "sk-live",
                        "model": "nemotron",
                        "temperature": 0.6,
                    }
                }
            },
            "num_repeats": "${repeats}",
            "repeats": 4,
            "extra_headers": ["Authorization: Bearer xyz"],
            "env": {"HF_HOME": "/cache", "MY_SERVICE": "sk-unlabelled"},
        }
    )

    record = provenance.build_provenance(config)

    dumped = record.model_dump_json()
    assert "resolved_config" not in record.model_dump()
    assert all(secret not in dumped for secret in ("hunter2", "sk-live", "token=abc", "Bearer xyz", "sk-unlabelled"))
    assert record.models["policy_model.responses_api_models.vllm_model"]["model"] == "nemotron"


def test_top_level_settings_are_named_rather_than_keyed_by_an_empty_path() -> None:
    record = provenance.build_provenance({"policy_model_name": "gpt-6-luna", "temperature": 0.2})

    assert record.models == {provenance.TOP_LEVEL_PATH: {"policy_model_name": "gpt-6-luna"}}
    assert record.sampling_parameters == {provenance.TOP_LEVEL_PATH: {"temperature": 0.2}}


class _ModelsHandler(BaseHTTPRequestHandler):
    seen_authorization: list[str | None] = []

    def do_GET(self) -> None:
        type(self).seen_authorization.append(self.headers.get("Authorization"))
        if self.path != "/v1/models":
            self.send_error(404)
            return
        body = orjson.dumps({"data": [{"id": "nemotron-super-3.5", "root": "/checkpoints/super-3/step-900"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def test_served_models_records_what_each_backend_reports_not_the_configured_name(tmp_path: Path) -> None:
    """Marta's case: the config and the served name say 3.5, the loaded checkpoint says otherwise."""
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    endpoint_file = tmp_path / "endpoint"
    endpoint_file.write_text(base_url + "\n")
    config = {
        "policy_model": {"responses_api_models": {"vllm_model": {"base_url": [base_url], "model": "super-3.5"}}},
        "judge_model": {
            "responses_api_models": {"openai_model": {"openai_base_url": base_url, "openai_api_key": "sk-judge"}}
        },
        "tracked_model": {"responses_api_models": {"vllm_model": {"endpoint_file": str(endpoint_file)}}},
        "gone_model": {"responses_api_models": {"vllm_model": {"base_url": "http://127.0.0.1:9/v1"}}},
        "bare_model": {"responses_api_models": {"vllm_model": {}}},
        "resources": {"responses_api_agents": {"simple_agent": {"base_url": base_url}}},
    }
    _ModelsHandler.seen_authorization = []
    try:
        served = provenance.served_models(config)
    finally:
        server.shutdown()

    reported = provenance.ServedBackend(id="nemotron-super-3.5", root="/checkpoints/super-3/step-900")
    assert served["policy_model"] == [reported]
    assert served["judge_model"] == [reported]
    assert served["tracked_model"] == [reported]
    assert served["gone_model"][0].unknown_reason is not None and served["gone_model"][0].root is None
    assert served["bare_model"] == [provenance.ServedBackend(unknown_reason="no base_url")]
    assert "resources" not in served
    assert sorted(_ModelsHandler.seen_authorization, key=str) == ["Bearer sk-judge", None, None]


def test_written_provenance_keeps_the_served_models_and_the_efb_config_ref_but_not_the_key(tmp_path: Path) -> None:
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    config = {
        "efb_config_ref": "nvidia-eval-factory-benchmarking/configs/benchmarks/gpqa@v3",
        "policy_model": {
            "responses_api_models": {
                "openai_model": {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key": "sk-secret"}
            }
        },
    }
    materialized_path_for(tmp_path / "rollouts.jsonl").write_text("{}\n")
    try:
        path = provenance.write_run_provenance(tmp_path / "rollouts.jsonl", config)
    finally:
        server.shutdown()

    written = orjson.loads(path.read_bytes())
    assert written["efb_config_ref"] == "nvidia-eval-factory-benchmarking/configs/benchmarks/gpqa@v3"
    assert written["served_models"]["policy_model"] == [
        {"id": "nemotron-super-3.5", "root": "/checkpoints/super-3/step-900", "unknown_reason": None}
    ]
    assert "sk-secret" not in path.read_text()


def test_a_no_serve_collection_asks_the_head_server_for_the_model_servers(tmp_path: Path) -> None:
    """`gym eval run --no-serve` holds no model server entries; the running head server's config does."""
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    head_config = {
        "policy_model": {
            "responses_api_models": {"openai_model": {"base_url": f"http://127.0.0.1:{server.server_port}/v1"}}
        }
    }

    materialized_path_for(tmp_path / "rollouts.jsonl").write_text("{}\n")
    try:
        path = provenance.write_run_provenance(
            tmp_path / "rollouts.jsonl", {"policy_model_name": "super-3.5"}, model_server_config=lambda: head_config
        )
    finally:
        server.shutdown()

    written = orjson.loads(path.read_bytes())
    assert written["served_models"]["policy_model"][0]["root"] == "/checkpoints/super-3/step-900"
    assert written["served_models_unknown_reason"] is None


def test_an_unreachable_head_server_leaves_the_served_models_unknown_with_a_reason(tmp_path: Path) -> None:
    def unreachable() -> dict:
        raise HeadServerUnreachableError("Could not connect to the head server at http://10.0.0.1:11000")

    materialized_path_for(tmp_path / "rollouts.jsonl").write_text("{}\n")
    path = provenance.write_run_provenance(
        tmp_path / "rollouts.jsonl", {"policy_model_name": "super-3.5"}, model_server_config=unreachable
    )

    written = orjson.loads(path.read_bytes())
    assert (written["status"], written["served_models"]) == ("available", {})
    assert written["served_models_unknown_reason"] == "HeadServerUnreachableError"
    assert "10.0.0.1" not in path.read_text()


class _CatalogHandler(_ModelsHandler):
    def do_GET(self) -> None:
        body = orjson.dumps({"data": [{"id": "tts-1"}, {"id": "gpt-6-luna"}, {"id": "gpt-6-luna-2026-09-01"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def test_a_hosted_catalog_keeps_only_the_configured_model() -> None:
    server = HTTPServer(("127.0.0.1", 0), _CatalogHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    config = {
        "listed": {
            "responses_api_models": {"openai_model": {"openai_base_url": base_url, "openai_model": "gpt-6-luna"}}
        },
        "unlisted": {"responses_api_models": {"openai_model": {"openai_base_url": base_url, "openai_model": "other"}}},
    }
    try:
        served = provenance.served_models(config)
    finally:
        server.shutdown()

    assert served["listed"] == [provenance.ServedBackend(id="gpt-6-luna")]
    assert served["unlisted"] == [provenance.ServedBackend(unknown_reason="configured model not among 3 listed")]


def test_provenance_copies_no_identifier_from_a_credential_or_environment_mapping() -> None:
    config = {
        "harbor": {"secrets": {"model": "sk-harbor"}},
        "policy_model": {
            "responses_api_models": {
                "local_vllm_model": {"model": "nemotron", "vllm_serve_env_vars": {"model_name": "sk-env"}}
            }
        },
        "sandbox": {"docker": {"images": {"auth": {"model": "sk-registry"}}}},
        "router": {"session_id_headers": {"model": "sk-header"}},
    }

    dumped = provenance.build_provenance(config).model_dump_json()

    assert all(secret not in dumped for secret in ("sk-harbor", "sk-env", "sk-registry", "sk-header"))
    assert "nemotron" in dumped


def test_a_protocol_relative_or_at_sign_password_leaves_no_credential() -> None:
    record = provenance.build_provenance(
        {
            "checkpoint": "//user:hunter2@host.example/ckpt?token=abc",  # pragma: allowlist secret
            "model": "user:p@ss@host/model",  # pragma: allowlist secret
        }
    )

    assert record.models[provenance.TOP_LEVEL_PATH] == {"checkpoint": "//host.example/ckpt", "model": "host/model"}


class _MalformedHandler(_ModelsHandler):
    def do_GET(self) -> None:
        body = orjson.dumps({"data": 123})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def test_one_bad_backend_does_not_lose_the_others(tmp_path: Path) -> None:
    good = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    bad = HTTPServer(("127.0.0.1", 0), _MalformedHandler)
    for server in (good, bad):
        Thread(target=server.serve_forever, daemon=True).start()
    good_url = f"http://127.0.0.1:{good.server_port}/v1"
    rotating = tmp_path / "endpoint"
    rotating.write_text("")
    config = {
        "malformed": {"responses_api_models": {"vllm_model": {"base_url": f"http://127.0.0.1:{bad.server_port}/v1"}}},
        "unresolved": {"responses_api_models": {"vllm_model": {"base_url": "${oc.env:POLICY_URL}/v1"}}},
        "rotating": {"responses_api_models": {"vllm_model": {"endpoint_file": str(rotating), "base_url": [good_url]}}},
        "policy_model": {"responses_api_models": {"vllm_model": {"base_url": good_url}}},
    }
    try:
        served = provenance.served_models(config)
    finally:
        good.shutdown()
        bad.shutdown()

    reported = provenance.ServedBackend(id="nemotron-super-3.5", root="/checkpoints/super-3/step-900")
    assert served["malformed"] == [provenance.ServedBackend(unknown_reason="malformed /models answer")]
    assert served["unresolved"] == [provenance.ServedBackend(unknown_reason="unresolved base_url")]
    assert served["rotating"] == [reported]
    assert served["policy_model"] == [reported]


def test_backends_are_asked_at_once(monkeypatch) -> None:
    def slow(base_url: str, api_key: object, configured: set[str]) -> list[provenance.ServedBackend]:
        time.sleep(0.5)
        return [provenance.ServedBackend(id=base_url)]

    monkeypatch.setattr(provenance, "_ask_backend", slow)
    urls = [f"http://10.0.0.{index}/v1" for index in range(provenance.SERVED_MODEL_BACKEND_LIMIT)]
    config = {
        "policy_model": {"responses_api_models": {"vllm_model": {"base_url": urls}}},
        "judge_model": {"responses_api_models": {"vllm_model": {"base_url": urls[:2]}}},
    }

    started = time.monotonic()
    served = provenance.served_models(config)

    assert time.monotonic() - started < 2
    assert [backend.id for backend in served["policy_model"]] == urls
    assert [backend.id for backend in served["judge_model"]] == urls[:2]
