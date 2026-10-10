# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the shared policy identity and explicitly test its two-turn tool transport."""

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: dict) -> None:
    """Never replace a receipt from an earlier allocation or preflight."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without replacing earlier endpoint/preflight evidence.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record(args: argparse.Namespace, server_args: list[str]) -> None:
    """Capture actual runtime identity before starting vLLM; this is not readiness."""
    gpu_text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
        errors="replace",
        timeout=30,
    )
    gpu_rows = [line.split(", ") for line in gpu_text.strip().splitlines()]
    if not gpu_rows or any(len(row) != 3 for row in gpu_rows):
        raise RuntimeError("Unable to record allocated GPU identity")
    address = socket.gethostbyname(socket.gethostname())
    if ipaddress.ip_address(address).is_loopback:
        raise RuntimeError("Compute hostname resolved to loopback; select a routable allocation address")
    checkpoint_metadata = {}
    for path in sorted(args.model.iterdir()):
        if path.is_file() and path.suffix in {".json", ".jinja", ".jinja2", ".py"}:
            checkpoint_metadata[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    weights = {
        path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted(args.model.glob("*.safetensors"))
    }
    if not weights or "config.json" not in checkpoint_metadata:
        raise RuntimeError("Checkpoint lacks config.json or safetensors weights")
    receipt = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provenance": "local_checkpoint_serving",
        "job_id": args.job_id,
        "allocation": dict(line.split("=", 1) for line in (args.run_dir / "allocation.txt").read_text().splitlines()),
        "hostname": socket.gethostname(),
        "base_url": f"http://{address}:8000/v1",
        "served_model": args.served_model,
        "api_key_env": args.api_key_env,
        "ready": False,
        "python": sys.version,
        "platform": platform.platform(),
        "gpus": [dict(zip(("name", "uuid", "driver_version"), row, strict=True)) for row in gpu_rows],
        "server_argv": [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *server_args],
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "image": {"path": str(args.image), "bytes": args.image.stat().st_size, "sha256": sha256(args.image)},
        "checkpoint": {
            "path": str(args.model),
            "metadata": checkpoint_metadata,
            "weight_files": weights,
            "weight_content_hashed": False,
        },
        "scripts": {
            name: sha256(args.run_dir / name) for name in ("serve_hsg.sbatch", "serve_in_container.sh", "serving.py")
        },
    }
    write_new(args.run_dir / "model.json", receipt)


def request_json(base_url: str, path: str, payload: dict | None = None, *, api_key: str) -> dict:
    """Use the caller's normal network/proxy configuration and in-memory credential."""
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def preflight(args: argparse.Namespace) -> None:
    """Make exactly two policy calls with a forced harmless tool and recorded result."""
    if (args.run_dir / "preflight.json").exists():
        raise FileExistsError("This allocation already has a preflight receipt")
    model = json.loads((args.run_dir / "model.json").read_text())
    if args.job_id is not None and model.get("job_id") != args.job_id:
        raise ValueError("The requested live job does not match model.json")
    base_url = model["base_url"]
    model_name = model["served_model"]
    api_key = os.environ[model["api_key_env"]]
    models = request_json(base_url, "/models", api_key=api_key)
    if model_name not in {entry["id"] for entry in models["data"]}:
        raise RuntimeError("The endpoint does not advertise the recorded served_model")
    tool = {
        "type": "function",
        "function": {
            "name": "add_integers",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
    messages = [{"role": "user", "content": "Use add_integers to add 17 and 25, then report the tool result."}]
    first = request_json(
        base_url,
        "/chat/completions",
        {
            "model": model_name,
            "messages": messages,
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": "add_integers"}},
            "temperature": 0,
            "max_tokens": 4096,
        },
        api_key=api_key,
    )
    assistant = first["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "add_integers":
        raise RuntimeError("Forced tool did not produce exactly one parsed add_integers call")
    arguments = json.loads(calls[0]["function"]["arguments"])
    if arguments != {"a": 17, "b": 25}:
        raise RuntimeError("Forced tool arguments did not match the preflight request")
    second = request_json(
        base_url,
        "/chat/completions",
        {
            "model": model_name,
            "messages": [*messages, assistant, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "42"}],
            "tools": [tool],
            "tool_choice": "none",
            "temperature": 0,
            "max_tokens": 4096,
        },
        api_key=api_key,
    )
    answer = second["choices"][0]["message"]
    if answer.get("tool_calls") or "42" not in (answer.get("content") or ""):
        raise RuntimeError("Second turn did not return the tool result as assistant text")
    write_new(
        args.run_dir / "preflight.json",
        {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "job_id": model.get("job_id"),
            "model_manifest_sha256": sha256(args.run_dir / "model.json"),
            "base_url": base_url,
            "passed": True,
            "purpose": "transport_only_two_turn_tool_preflight_not_benchmark_sampling",
            "responses": [first, second],
        },
    )
    print(json.dumps({"passed": True, "job_id": model.get("job_id"), "base_url": base_url, "model_calls": 2}))


def register_endpoint(args: argparse.Namespace) -> None:
    """Record an existing endpoint without fabricating a checkpoint or Slurm job."""
    url = urlsplit(args.base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Use an HTTP(S) base URL without embedded credentials, query or fragment")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_new(
        args.run_dir / "model.json",
        {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "provenance": "existing_endpoint",
            "base_url": args.base_url.rstrip("/"),
            "served_model": args.served_model,
            "api_key_env": args.api_key_env,
            "ready": False,
        },
    )


def validated_model(model_dir: Path) -> dict:
    """Require a successful transport preflight of these exact endpoint metadata bytes."""
    model = json.loads((model_dir / "model.json").read_text())
    preflight = json.loads((model_dir / "preflight.json").read_text())
    if not preflight.get("passed") or preflight.get("model_manifest_sha256") != sha256(model_dir / "model.json"):
        raise RuntimeError("Missing matching policy preflight; run serving.py preflight for this model manifest")
    if preflight.get("job_id") != model.get("job_id"):
        raise RuntimeError("Preflight and model allocation identities differ")
    for field in ("base_url", "served_model", "api_key_env"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ValueError(f"Missing model manifest field: {field}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record")
    recorder.add_argument("--run-dir", type=Path, required=True)
    recorder.add_argument("--model", type=Path, required=True)
    recorder.add_argument("--image", type=Path, required=True)
    recorder.add_argument("--job-id", required=True)
    recorder.add_argument("--served-model", required=True)
    recorder.add_argument("--api-key-env", default="POLICY_API_KEY")
    check = commands.add_parser("preflight")
    check.add_argument("--run-dir", type=Path, required=True)
    check.add_argument("--job-id", help="Optional assertion for a locally served allocation")
    endpoint = commands.add_parser("endpoint", help="Register an existing endpoint; does not make model calls")
    endpoint.add_argument("--run-dir", type=Path, required=True)
    endpoint.add_argument("--base-url", required=True)
    endpoint.add_argument("--served-model", required=True)
    endpoint.add_argument("--api-key-env", default="POLICY_API_KEY")
    args, remainder = parser.parse_known_args()
    if args.command == "record":
        if not remainder or remainder[0] != "--":
            parser.error("Pass the actual vLLM arguments after --")
        record(args, remainder[1:])
    else:
        if remainder:
            parser.error("Unexpected preflight arguments")
        if args.command == "endpoint":
            register_endpoint(args)
        else:
            preflight(args)


if __name__ == "__main__":
    main()
