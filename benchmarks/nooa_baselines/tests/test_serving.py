# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport and identity checks; these tests make no network or model calls."""

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.nooa_baselines import serving


@pytest.fixture(autouse=True)
def policy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_POLICY_KEY", "fixture-secret")


def fixture(tmp_path: Path) -> argparse.Namespace:
    (tmp_path / "model.json").write_text(
        json.dumps(
            {
                "job_id": "123",
                "base_url": "http://test.invalid/v1",
                "served_model": "fixture-policy",
                "api_key_env": "TEST_POLICY_KEY",
            }
        )
    )
    return argparse.Namespace(run_dir=tmp_path, job_id="123")


def test_preflight_preserves_tool_call_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = fixture(tmp_path)
    calls = []
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "The tool will add the numbers.",
        "tool_calls": [
            {
                "id": "call-actual-id",
                "type": "function",
                "function": {"name": "add_integers", "arguments": '{"a":17,"b":25}'},
            }
        ],
    }

    def request(base_url: str, path: str, payload: dict | None = None, *, api_key: str) -> dict:
        assert api_key == "fixture-secret"
        assert base_url == "http://test.invalid/v1"
        calls.append((path, payload))
        if path == "/models":
            return {"data": [{"id": "fixture-policy"}]}
        message = assistant if len(calls) == 2 else {"role": "assistant", "content": "The tool returned 42."}
        return {"choices": [{"message": message}]}

    monkeypatch.setattr(serving, "request_json", request)
    serving.preflight(args)
    assert len(calls) == 3
    assert calls[1][1]["tool_choice"] == {"type": "function", "function": {"name": "add_integers"}}
    followup = calls[2][1]
    assert followup["messages"][1] == assistant
    assert followup["messages"][2] == {"role": "tool", "tool_call_id": "call-actual-id", "content": "42"}
    assert followup["tool_choice"] == "none"
    receipt = json.loads((tmp_path / "preflight.json").read_text())
    assert receipt["passed"] and len(receipt["responses"]) == 2
    assert receipt["model_manifest_sha256"] == serving.sha256(tmp_path / "model.json")
    with pytest.raises(FileExistsError):
        serving.preflight(args)
    assert len(calls) == 3


def test_preflight_rejects_stale_allocation_before_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = fixture(tmp_path)
    args.job_id = "456"
    monkeypatch.setattr(serving, "request_json", lambda *args: pytest.fail("Unexpected network request"))
    with pytest.raises(ValueError, match="live job"):
        serving.preflight(args)


def test_invalid_tool_arguments_do_not_advance_or_claim_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = fixture(tmp_path)
    calls = []

    def request(base_url: str, path: str, payload: dict | None = None, *, api_key: str) -> dict:
        assert api_key == "fixture-secret"
        calls.append(path)
        if path == "/models":
            return {"data": [{"id": "fixture-policy"}]}
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call-1", "function": {"name": "add_integers", "arguments": '{"a":17,"b":24}'}}
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(serving, "request_json", request)
    with pytest.raises(RuntimeError, match="arguments"):
        serving.preflight(args)
    assert calls == ["/models", "/chat/completions"]
    assert not (tmp_path / "preflight.json").exists()


def test_record_binds_allocation_image_checkpoint_and_scripts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architectures": ["test"]}')
    (model_dir / "configuration_custom.py").write_text("class CustomConfig: pass\n")
    (model_dir / "model.safetensors").write_bytes(b"fixture-weights")
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"fixture-image")
    for name in ("serve_hsg.sbatch", "serve_in_container.sh", "serving.py"):
        (tmp_path / name).write_text(name)
    (tmp_path / "allocation.txt").write_text("JobId=123\nAllocTRES=cpu=144,gres/gpu=4\n")
    monkeypatch.setattr(
        serving.subprocess,
        "check_output",
        lambda *args, **kwargs: "\n".join(f"NVIDIA GB200, GPU-{i}, 580.126.20" for i in range(4)),
    )
    monkeypatch.setattr(serving.socket, "gethostbyname", lambda _: "10.0.0.5")
    monkeypatch.setattr(serving.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(serving.importlib.metadata, "distributions", lambda: [])
    args = argparse.Namespace(
        run_dir=tmp_path,
        model=model_dir,
        image=image,
        job_id="123",
        served_model="fixture-policy",
        api_key_env="TEST_POLICY_KEY",
    )
    argv = ["--model", str(model_dir), "--max-model-len", "262144"]
    serving.record(args, argv)
    receipt = json.loads((tmp_path / "model.json").read_text())
    assert receipt["job_id"] == receipt["allocation"]["JobId"] == "123"
    assert receipt["base_url"] == "http://10.0.0.5:8000/v1"
    assert receipt["ready"] is False
    assert len(receipt["gpus"]) == 4 and receipt["server_argv"][-4:] == argv
    assert receipt["image"]["sha256"] == serving.sha256(image)
    assert receipt["checkpoint"]["metadata"]["config.json"]["sha256"] == serving.sha256(model_dir / "config.json")
    assert receipt["checkpoint"]["metadata"]["configuration_custom.py"]["sha256"] == serving.sha256(
        model_dir / "configuration_custom.py"
    )
    assert receipt["checkpoint"]["weight_content_hashed"] is False
    assert receipt["scripts"]["serving.py"] == serving.sha256(tmp_path / "serving.py")


def test_existing_endpoint_manifest_needs_no_slurm_and_never_saves_key(tmp_path: Path) -> None:
    args = argparse.Namespace(
        run_dir=tmp_path,
        base_url="https://api.example/v1",
        served_model="org/exact-alias",
        api_key_env="TEST_POLICY_KEY",
    )
    serving.register_endpoint(args)
    path = tmp_path / "model.json"
    model = json.loads(path.read_text())
    assert "job_id" not in model and "checkpoint" not in model
    assert model["served_model"] == "org/exact-alias"
    assert "fixture-secret" not in path.read_text()
    (tmp_path / "preflight.json").write_text(
        json.dumps({"passed": True, "model_manifest_sha256": serving.sha256(path)})
    )
    assert serving.validated_model(tmp_path) == model
    path.write_text(json.dumps(model | {"served_model": "different"}))
    with pytest.raises(RuntimeError, match="matching policy preflight"):
        serving.validated_model(tmp_path)


# Invalid fixture credentials exercise rejection; these are not a usable secret.
@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example/v1",  # pragma: allowlist secret
        "https://example/v1?key=secret",
        "file:///tmp/a",
    ],
)
def test_endpoint_registration_rejects_embedded_credentials(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="without embedded credentials"):
        serving.register_endpoint(
            argparse.Namespace(run_dir=tmp_path, base_url=url, served_model="m", api_key_env="KEY")
        )
    assert not (tmp_path / "model.json").exists()
