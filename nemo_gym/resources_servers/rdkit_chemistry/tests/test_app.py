# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the rdkit_chemistry resources server."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root

from nemo_gym.server_utils import ServerClient
from nemo_gym.resources_servers.rdkit_chemistry import sandbox_launcher
from nemo_gym.resources_servers.rdkit_chemistry.app import (
    _SUPPORTED_PROPERTY_TYPES,
    RDKitChemistryConfig,
    RDKitChemistryResourcesServer,
    compute_reward,
    extract_predicted_value,
)


# ---------------------------------------------------------------------------
# extract_predicted_value
# ---------------------------------------------------------------------------


class TestExtractPredictedValueStrict:
    """Non-boxed mode requires ((answer)) — bare text is rejected."""

    def test_bare_integer_rejected(self):
        assert extract_predicted_value("42", "count") is None

    def test_bare_decimal_rejected(self):
        assert extract_predicted_value("2.54", "count") is None

    def test_bare_text_with_number_rejected(self):
        assert extract_predicted_value("The count is approximately 5.", "count") is None

    def test_bool_text_rejected(self):
        assert extract_predicted_value("yes", "presence") is None

    def test_empty_string(self):
        assert extract_predicted_value("", "count") is None

    def test_non_string(self):
        assert extract_predicted_value(None, "count") is None


# ---------------------------------------------------------------------------
# extract_predicted_value — boxed format
# ---------------------------------------------------------------------------


class TestExtractPredictedValueBoxed:
    def test_boxed_integer(self):
        assert extract_predicted_value(r"\boxed{42}", "count", use_box_format=True) == 42.0

    def test_boxed_decimal(self):
        assert extract_predicted_value(r"\boxed{0.83}", "count", use_box_format=True) == pytest.approx(0.83)

    def test_boxed_negative(self):
        assert extract_predicted_value(r"\boxed{-1.5}", "count", use_box_format=True) == pytest.approx(-1.5)

    def test_boxed_zero_or_one(self):
        assert extract_predicted_value(r"\boxed{1}", "bool", use_box_format=True) == 1.0
        assert extract_predicted_value(r"\boxed{0}", "bool", use_box_format=True) == 0.0

    def test_boxed_with_surrounding_text(self):
        text = r"The atom count is \boxed{12}."
        assert extract_predicted_value(text, "count", use_box_format=True) == 12.0

    def test_boxed_last_occurrence_wins(self):
        text = r"First attempt: \boxed{1}. Correction: \boxed{3}"
        assert extract_predicted_value(text, "count", use_box_format=True) == 3.0

    def test_boxed_missing_returns_none(self):
        assert extract_predicted_value("42", "count", use_box_format=True) is None

    def test_boxed_empty_braces_returns_none(self):
        assert extract_predicted_value(r"\boxed{}", "count", use_box_format=True) is None

    def test_boxed_non_numeric_returns_none(self):
        assert extract_predicted_value(r"\boxed{hello}", "count", use_box_format=True) is None

    def test_boxed_not_required_when_flag_false(self):
        assert extract_predicted_value("((42))", "count", use_box_format=False) == 42.0

    def test_bare_number_rejected_when_boxed_required(self):
        assert extract_predicted_value("The answer is 42", "count", use_box_format=True) is None

    def test_boxed_with_whitespace_inside(self):
        assert extract_predicted_value(r"\boxed{ 7 }", "count", use_box_format=True) == 7.0


# ---------------------------------------------------------------------------
# extract_predicted_value — double-parentheses format (non-boxed)
# ---------------------------------------------------------------------------


class TestExtractPredictedValueDoubleParens:
    def test_double_parens_integer(self):
        assert extract_predicted_value("The answer is ((42))", "count") == 42.0

    def test_double_parens_decimal(self):
        assert extract_predicted_value("((0.83))", "count") == pytest.approx(0.83)

    def test_double_parens_negative(self):
        assert extract_predicted_value("((-1.5))", "count") == pytest.approx(-1.5)

    def test_double_parens_zero_or_one(self):
        assert extract_predicted_value("((1))", "bool") == 1.0
        assert extract_predicted_value("((0))", "bool") == 0.0

    def test_double_parens_with_surrounding_text(self):
        assert extract_predicted_value("After analysis, the count is ((8)).", "fragment") == 8.0

    def test_double_parens_last_occurrence_wins(self):
        text = "First ((3)), actually ((5))"
        assert extract_predicted_value(text, "count") == 5.0

    def test_double_parens_scientific_notation(self):
        assert extract_predicted_value("((1.5e-3))", "count") == pytest.approx(1.5e-3)

    def test_double_parens_whitespace_inside(self):
        assert extract_predicted_value("(( 7 ))", "count") == 7.0

    def test_double_parens_empty_returns_none(self):
        assert extract_predicted_value("(())", "count") is None

    def test_double_parens_non_numeric_returns_none(self):
        assert extract_predicted_value("((hello))", "count") is None

    def test_double_parens_preferred_over_bare_number(self):
        text = "The value 99 is wrong, the correct answer is ((42))"
        assert extract_predicted_value(text, "count") == 42.0

    def test_bare_number_rejected_without_double_parens(self):
        assert extract_predicted_value("42", "count") is None


# ---------------------------------------------------------------------------
# compute_reward — exact-match
# ---------------------------------------------------------------------------


class TestComputeReward:
    def test_count_correct(self):
        assert compute_reward(5.0, 5.0) == 1.0

    def test_count_wrong(self):
        assert compute_reward(4.0, 5.0) == 0.0

    def test_bool_correct(self):
        assert compute_reward(1.0, 1.0) == 1.0

    def test_bool_wrong(self):
        assert compute_reward(0.0, 1.0) == 0.0

    def test_presence_correct(self):
        assert compute_reward(0.0, 0.0) == 1.0

    def test_fragment_correct(self):
        assert compute_reward(3.0, 3.0) == 1.0

    def test_none_prediction(self):
        assert compute_reward(None, 5.0) == 0.0

    def test_nan_prediction(self):
        assert compute_reward(float("nan"), 5.0) == 0.0


class TestUnsupportedPropertyType:
    def test_float_not_supported(self):
        assert "float" not in _SUPPORTED_PROPERTY_TYPES

    def test_supported_types(self):
        assert _SUPPORTED_PROPERTY_TYPES == {"count", "bool", "presence", "fragment"}


class TestLocalNSToolsColocation:
    def test_rejects_cross_host_pairing(self):
        config = RDKitChemistryConfig(
            host="10.0.0.1",
            port=8000,
            entrypoint="app.py",
            name="rdkit_chemistry",
            domain="knowledge",
            sandbox_venv_path="/tmp/ns_tools/.venv",
            require_local_ns_tools_colocation=True,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = OmegaConf.create(
            {
                "rdkit_chemistry_ns_tools": {
                    "resources_servers": {
                        "ns_tools": {
                            "host": "10.0.0.2",
                            "port": 8001,
                            "entrypoint": "app.py",
                            "domain": "agent",
                            "sandbox_host": "127.0.0.1",
                        }
                    }
                }
            }
        )
        server = RDKitChemistryResourcesServer(config=config, server_client=server_client)

        with pytest.raises(RuntimeError, match="same host"):
            server._validate_local_ns_tools_colocation()

    def test_allows_same_host_pairing(self):
        config = RDKitChemistryConfig(
            host="10.0.0.1",
            port=8000,
            entrypoint="app.py",
            name="rdkit_chemistry",
            domain="knowledge",
            sandbox_venv_path="/tmp/ns_tools/.venv",
            require_local_ns_tools_colocation=True,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = OmegaConf.create(
            {
                "rdkit_chemistry_ns_tools": {
                    "resources_servers": {
                        "ns_tools": {
                            "host": "10.0.0.1",
                            "port": 8001,
                            "entrypoint": "app.py",
                            "domain": "agent",
                            "sandbox_host": "127.0.0.1",
                        }
                    }
                }
            }
        )
        server = RDKitChemistryResourcesServer(config=config, server_client=server_client)

        server._validate_local_ns_tools_colocation()

    def test_rejects_wrong_ns_tools_sandbox_port(self):
        config = RDKitChemistryConfig(
            host="10.0.0.1",
            port=8000,
            entrypoint="app.py",
            name="rdkit_chemistry",
            domain="knowledge",
            sandbox_venv_path="/tmp/ns_tools/.venv",
            sandbox_proxy_port=6001,
            require_local_ns_tools_colocation=True,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = OmegaConf.create(
            {
                "rdkit_chemistry_ns_tools": {
                    "resources_servers": {
                        "ns_tools": {
                            "host": "10.0.0.1",
                            "port": 8001,
                            "entrypoint": "app.py",
                            "domain": "agent",
                            "sandbox_host": "127.0.0.1",
                            "sandbox_port": 6000,
                        }
                    }
                }
            }
        )
        server = RDKitChemistryResourcesServer(config=config, server_client=server_client)

        with pytest.raises(RuntimeError, match="sandbox_port=6001"):
            server._validate_local_ns_tools_colocation()

    def test_setup_webserver_passes_startup_probe_config(self, monkeypatch):
        config = RDKitChemistryConfig(
            host="10.0.0.1",
            port=8000,
            entrypoint="app.py",
            name="rdkit_chemistry",
            domain="knowledge",
            sandbox_venv_path="/tmp/ns_tools/.venv",
            sandbox_proxy_port=6001,
            sandbox_startup_probe_enabled=True,
            sandbox_startup_probe_timeout_s=21.0,
            require_local_ns_tools_colocation=True,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = OmegaConf.create(
            {
                "rdkit_chemistry_ns_tools": {
                    "resources_servers": {
                        "ns_tools": {
                            "host": "10.0.0.1",
                            "port": 8001,
                            "entrypoint": "app.py",
                            "domain": "agent",
                            "sandbox_host": "127.0.0.1",
                            "sandbox_port": 6001,
                        }
                    }
                }
            }
        )
        server = RDKitChemistryResourcesServer(config=config, server_client=server_client)
        start_kwargs = {}

        def fake_start_sandbox(**kwargs):
            start_kwargs.update(kwargs)

        monkeypatch.setitem(sys.modules, "sandbox_launcher", SimpleNamespace(start_sandbox=fake_start_sandbox))
        monkeypatch.setattr(
            "resources_servers.rdkit_chemistry.app.SimpleResourcesServer.setup_webserver", lambda self: "web"
        )

        assert server.setup_webserver() == "web"
        assert start_kwargs["startup_probe_enabled"] is True
        assert start_kwargs["startup_probe_timeout_s"] == 21.0


class TestSandboxStartupProbe:
    def test_runs_stateful_rdkit_probe(self, monkeypatch):
        posted_payloads = []
        deleted_urls = []

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.probe_value = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, headers, json):
                posted_payloads.append((url, headers, json))
                code = json["generated_code"]
                if "probe_value = 42" in code:
                    self.probe_value = 42
                    return FakeResponse({"process_status": "completed", "stdout": "42\n", "stderr": ""})
                if "probe_value + 1" in code:
                    return FakeResponse(
                        {"process_status": "completed", "stdout": f"{self.probe_value + 1}\n", "stderr": ""}
                    )
                if "Chem.MolFromSmiles('CCO').GetNumAtoms()" in code:
                    return FakeResponse({"process_status": "completed", "stdout": "3\n", "stderr": ""})
                raise AssertionError(f"Unexpected probe code: {code}")

            def delete(self, url):
                deleted_urls.append(url)
                return FakeResponse({})

        monkeypatch.setattr(sandbox_launcher.httpx, "Client", FakeClient)
        sandbox_launcher._run_startup_probe(6001, timeout_s=9.0)

        assert len(posted_payloads) == 3
        assert all(url == "http://127.0.0.1:6001/execute" for url, _, _ in posted_payloads)
        assert deleted_urls == [
            posted_payloads[0][0].replace("/execute", f"/sessions/{posted_payloads[0][1]['X-Session-ID']}")
        ]

    def test_raises_when_probe_fails(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"process_status": "error", "stdout": "", "stderr": "boom"}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, headers, json):
                return FakeResponse()

            def delete(self, url):
                return FakeResponse()

        monkeypatch.setattr(sandbox_launcher.httpx, "Client", FakeClient)

        with pytest.raises(RuntimeError, match="Sandbox startup probe failed"):
            sandbox_launcher._run_startup_probe(6001, timeout_s=9.0)
