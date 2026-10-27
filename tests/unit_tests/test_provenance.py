# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import orjson
import pytest
from omegaconf import OmegaConf

from nemo_gym import provenance
from nemo_gym.path_utils import materialized_path_for, provenance_path_for
from nemo_gym.server_utils import HeadServerUnreachableError


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
                    "base_url": "https://user:secret@host/?token=secret",
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


def test_review_provenance_excludes_boolean_sampling_values() -> None:
    record = provenance.build_provenance({"seed": True, "temperature": False, "top_p": 0.9})
    assert record.sampling_parameters == {provenance.TOP_LEVEL_PATH: {"top_p": 0.9}}


@pytest.mark.parametrize(
    "identifier", ["user:secret@host/model?token=secret", "https://user:secret@host/model?token=secret"]
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
                        "base_url": "https://user:hunter2@inference.example/v1?token=abc",
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
