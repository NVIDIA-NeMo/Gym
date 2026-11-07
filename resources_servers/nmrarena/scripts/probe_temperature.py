# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Does the requested temperature reach the model on a given route?

The Gemini route of the gateway reports ``temperature: 0.0`` and
``max_output_tokens: null`` on every response regardless of the request, so the
echoed field is not evidence either way (see the server README, "Known
limitation"). This probe produces gateway-side evidence instead: one fixed
benchmark prompt, built with the server's own ``build_messages``, is sent N
times at ``temperature=0.0`` and N times at ``temperature=1.0`` through the
production path the sweeps used — ``responses_api_models/openai_model``'s
``SimpleModelServer.responses``, which forwards ``model_dump(exclude_unset=True)``
plus the configured model id — and the distinct output texts are counted per
setting. Identical outputs at 0.0 and varying outputs at 1.0 show the value is
applied; any other pattern is reported as observed.

Credentials come from ``env.yaml`` at the repository root and are never printed.

Usage (from the repository root, server venv active)::

    python resources_servers/nmrarena/scripts/probe_temperature.py \
        --model gcp/google/gemini-3.1-pro-preview --model openai/openai/gpt-5.5 \
        --n 8 --max-output-tokens 16384 --output results/nmrarena/temperature_probe.json
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import yaml


SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_DIR.parents[1]
sys.path.insert(0, str(SERVER_DIR))

from prompting import build_messages  # noqa: E402
from scoring import parse_candidates  # noqa: E402

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402


def _load_openai_model_module():
    spec = importlib.util.spec_from_file_location(
        "openai_model_app", REPO_ROOT / "responses_api_models" / "openai_model" / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_server(model: str):
    """The production ``openai_model`` server, configured as ``gym env start`` configures it from ``env.yaml``."""
    env = yaml.safe_load((REPO_ROOT / "env.yaml").read_text())
    mod = _load_openai_model_module()
    config = mod.SimpleModelServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="policy_model",
        openai_base_url=env["policy_base_url"],
        openai_api_key=env["policy_api_key"],
        openai_model=model,
    )
    return mod.SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def _secrets() -> list[str]:
    env = yaml.safe_load((REPO_ROOT / "env.yaml").read_text())
    return [str(v) for k, v in env.items() if "key" in k or "token" in k]


def _text(response: Any) -> str:
    texts = []
    for o in response.output:
        if getattr(o, "type", None) == "message":
            for c in getattr(o, "content", None) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    texts.append(t)
    return "\n".join(texts)


async def _draw(server, messages: list[dict], temperature: float, max_output_tokens: int, index: int) -> dict:
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=messages, temperature=temperature, max_output_tokens=max_output_tokens
    )
    response = await server.responses(body)
    text = _text(response)
    usage = response.usage.model_dump() if response.usage else {}
    details = usage.get("output_tokens_details") or {}
    candidates = parse_candidates(text).candidates
    return {
        "draw": index,
        "requested_temperature": temperature,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        # The answer the verifier would score, which varies less than the prose around it.
        "candidates_sha256": hashlib.sha256(json.dumps(candidates).encode("utf-8")).hexdigest(),
        "n_candidates": len(candidates),
        "text_chars": len(text),
        "echoed_temperature": getattr(response, "temperature", None),
        "echoed_top_p": getattr(response, "top_p", None),
        "echoed_max_output_tokens": getattr(response, "max_output_tokens", None),
        "status": getattr(response, "status", None),
        "model": getattr(response, "model", None),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


async def probe_model(model: str, messages: list[dict], n: int, max_output_tokens: int) -> dict:
    server = make_server(model)
    out: dict[str, Any] = {"model": model, "n_per_setting": n, "max_output_tokens": max_output_tokens, "settings": {}}
    # Is the value transmitted at all? A provider that reads it rejects an out-of-range
    # value with a range message; one that drops it answers normally.
    try:
        body = NeMoGymResponseCreateParamsNonStreaming(input=messages, temperature=3.5, max_output_tokens=16)
        response = await server.responses(body)
        out["out_of_range_temperature"] = {"accepted": True, "status": getattr(response, "status", None)}
    except Exception as exc:  # the gateway wraps the provider error; keep its class and a short message
        message = str(exc)
        for secret in _secrets():
            message = message.replace(secret, "***")
        out["out_of_range_temperature"] = {"accepted": False, "error": type(exc).__name__, "message": message[:300]}
    for temperature in (0.0, 1.0):
        draws = await asyncio.gather(*(_draw(server, messages, temperature, max_output_tokens, i) for i in range(n)))
        shas = [d["text_sha256"] for d in draws]
        out["settings"][str(temperature)] = {
            "draws": draws,
            "distinct_outputs": len(set(shas)),
            "distinct_candidate_lists": len({d["candidates_sha256"] for d in draws}),
            "empty_outputs": sum(d["text_chars"] == 0 for d in draws),
            "distinct_reasoning_token_counts": len({d["reasoning_tokens"] for d in draws}),
            "echoed_temperatures": sorted({str(d["echoed_temperature"]) for d in draws}),
        }
    return out


async def _probe_all(models: list[str], messages: list[dict], n: int, max_output_tokens: int) -> list[dict]:
    return [await probe_model(m, messages, n, max_output_tokens) for m in models]


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="Served model id; repeatable")
    parser.add_argument("--n", type=int, default=8, help="Draws per temperature setting")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument(
        "--input",
        default=str(SERVER_DIR / "data" / "nmrarena_105.jsonl"),
        help="Prepared benchmark JSONL; the first row's spectra are used",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    # Gym's server and client code reads the global config by parsing ``sys.argv`` through
    # Hydra; this script's own flags must not reach that parser.
    sys.argv = sys.argv[:1]

    row = json.loads(Path(args.input).read_text(encoding="utf-8").splitlines()[0])
    # Rebuild the prompt with the server's own code rather than reusing the row's
    # messages, so the probe is tied to the prompt construction under test.
    user = next(m["content"] for m in row["responses_create_params"]["input"] if m["role"] == "user")
    h_nmr = user.split("1H NMR: ", 1)[1].split("\n13C NMR: ", 1)[0]
    c_nmr = user.split("\n13C NMR: ", 1)[1].split("\n\nPropose", 1)[0]
    messages = build_messages(h_nmr, c_nmr)
    assert messages == row["responses_create_params"]["input"], "rebuilt prompt differs from the prepared row"

    report = {
        "compound_id": row["verifier_metadata"]["compound_id"],
        "prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "path": "responses_api_models/openai_model SimpleModelServer.responses (in-process, production class)",
        # One event loop for every model: the shared HTTP client binds to the first loop.
        "models": asyncio.run(_probe_all(args.model, messages, args.n, args.max_output_tokens)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for m in report["models"]:
        for t, s in m["settings"].items():
            print(
                f"{m['model']:40s} T={t}: distinct_outputs={s['distinct_outputs']}/{m['n_per_setting']} "
                f"distinct_candidate_lists={s['distinct_candidate_lists']} empty={s['empty_outputs']} "
                f"distinct_reasoning_counts={s['distinct_reasoning_token_counts']} echoed_T={s['echoed_temperatures']}"
            )
        print(f"{m['model']:40s} temperature=3.5: {m['out_of_range_temperature']}")
    return report


if __name__ == "__main__":
    main()
