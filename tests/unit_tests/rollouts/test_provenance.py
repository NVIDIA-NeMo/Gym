# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import orjson
import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.config_types import HeadServerUnreachableError
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollouts import provenance
from nemo_gym.rollouts.provenance import AgentServerHarness
from nemo_gym.utils.path_utils import materialized_path_for, provenance_path_for


ROOT = Path(__file__).resolve().parents[3]


def _resolve(config_path: str, **overrides: object) -> DictConfig:
    """A repository config resolved as a run resolves it, with no model behind the policy."""
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(ROOT / config_path)], **overrides},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


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


@pytest.mark.parametrize(
    ("config_path", "agent_server", "expected"),
    [
        (
            "benchmarks/gpqa/config.yaml",
            "gpqa_mcqa_simple_agent",
            AgentServerHarness(agent_type="simple_agent", model_server="policy_model"),
        ),
        # OpenCode at the version the agent pins by default: the config pins none.
        (
            "benchmarks/swebench/verified/opencode.yaml",
            "swebench_verified_opencode_agent",
            AgentServerHarness(agent_type="opencode_agent", model_server="policy_model"),
        ),
        (
            "benchmarks/swebench/verified/opencode_legacy_agent.yaml",
            "swebench_verified_opencode_sandboxed_agent",
            AgentServerHarness(
                agent_type="opencode_sandboxed_agent",
                harness={"opencode_version": "1.17.11"},
                model_server="policy_model",
            ),
        ),
    ],
)
def test_a_benchmark_config_records_the_harness_of_its_agent_server(
    config_path: str, agent_server: str, expected: AgentServerHarness
) -> None:
    record = provenance.build_provenance(_resolve(config_path))

    assert record.agent_servers[agent_server] == expected


def test_each_agent_server_of_a_run_records_its_own_harness_and_model_server() -> None:
    config = _resolve("benchmarks/partner_if/config.yaml")

    agents = provenance.build_provenance(config).agent_servers

    assert agents["partner_if_simple_agent"] == AgentServerHarness(
        agent_type="simple_agent", model_server="partner_if_policy_model"
    )
    assert agents["partner_if_swe_agent"] == AgentServerHarness(
        agent_type="opencode_agent", harness={"opencode_version": "1.17.11"}, model_server="partner_if_policy_model"
    )
    assert agents["partner_if_prime_agent"] == AgentServerHarness(
        agent_type="prime_agent", harness={"prime_agent_version": "0.7.0"}, model_server="partner_if_prime_model"
    )
    # Each names a model server of the run, as `served_models` is keyed.
    model_servers = {
        name for name, node in config.items() if isinstance(node, DictConfig) and "responses_api_models" in node
    }
    assert {agent.model_server for agent in agents.values()} - {None} <= model_servers


def test_harness_presets_that_share_an_agent_type_record_which_harness_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSearchQA's harness presets all run `harness_exa_search`; only its `agent` and `agent_kwargs` differ."""
    monkeypatch.setenv("EXA_API_KEY", "exa-provenance-key")
    agents = {}
    for preset in ("opencode", "hermes"):
        config = _resolve(f"benchmarks/deepsearchqa/configs/deepsearchqa_{preset}.yaml")
        assert config["deepsearchqa_agent"]["responses_api_agents"]["harness_exa_search"]["exa_api_key"] == (
            "exa-provenance-key"
        )
        record = provenance.build_provenance(config)
        assert "exa-provenance-key" not in record.model_dump_json()
        agents[preset] = record.agent_servers["deepsearchqa_agent"]

    assert agents["opencode"] == AgentServerHarness(
        agent_type="harness_exa_search",
        harness={"agent": "opencode", "opencode_version": "1.17.11"},
        model_server="policy_model",
    )
    assert (agents["hermes"].agent_type, agents["hermes"].harness["agent"]) == ("harness_exa_search", "hermes")


def test_a_harness_setting_never_carries_url_credentials() -> None:
    config = {
        "swe_agent": {
            "responses_api_agents": {
                "swe_agents": {
                    "agent_framework": "openhands",
                    "agent_framework_repo": "https://user:hunter2@git.example/org/OpenHands.git",  # pragma: allowlist secret
                    "agent_framework_commit": "54d2a95",
                    "agent_kwargs": {"openai_api_key": "sk-wrapped", "opencode_version": "1.17.8"},
                    "openhands_token": "sk-own",
                }
            }
        }
    }

    record = provenance.build_provenance(config)

    assert record.agent_servers["swe_agent"].harness == {
        "agent_framework": "openhands",
        "agent_framework_commit": "54d2a95",
        "agent_framework_repo": "https://git.example/org/OpenHands.git",
        "opencode_version": "1.17.8",
    }
    assert all(secret not in record.model_dump_json() for secret in ("hunter2", "sk-wrapped", "sk-own"))


def test_a_no_serve_collection_records_the_head_servers_agent_servers(tmp_path: Path) -> None:
    """`gym eval run --no-serve` holds no agent server entries; the running head server's config does."""
    head_config = _resolve("benchmarks/gpqa/config.yaml")
    materialized_path_for(tmp_path / "rollouts.jsonl").write_text("{}\n")

    path = provenance.write_run_provenance(
        tmp_path / "rollouts.jsonl", {"policy_model_name": "super-3.5"}, model_server_config=lambda: head_config
    )

    written = orjson.loads(path.read_bytes())
    assert written["agent_servers"]["gpqa_mcqa_simple_agent"] == {
        "agent_type": "simple_agent",
        "harness": {},
        "model_server": "policy_model",
    }


# `*_version` settings of agent servers that version something other than the harness.
_NOT_HARNESS_VERSIONS = {"exact_trace_transport_version"}


def test_every_harness_version_a_repository_config_pins_is_recorded() -> None:
    pinned: dict[str, Path] = {}
    for pattern in (
        "benchmarks/**/*.yaml",
        "environments/**/*.yaml",
        "resources_servers/*/configs/**/*.yaml",
        "responses_api_agents/*/configs/**/*.yaml",
    ):
        for path in ROOT.glob(pattern):
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue
            for node in config.values() if isinstance(config, dict) else ():
                agents = node.get("responses_api_agents") if isinstance(node, dict) else None
                for settings in agents.values() if isinstance(agents, dict) else ():
                    wrapped = settings.get("agent_kwargs") if isinstance(settings, dict) else None
                    for block in (settings, wrapped):
                        for key in block if isinstance(block, dict) else ():
                            if str(key).endswith("_version"):
                                pinned.setdefault(str(key), path.relative_to(ROOT))

    missing = {key: str(path) for key, path in pinned.items() if key not in provenance._HARNESS_KEYS}
    assert set(missing) <= _NOT_HARNESS_VERSIONS, f"add each harness version key to _HARNESS_KEYS: {missing}"
