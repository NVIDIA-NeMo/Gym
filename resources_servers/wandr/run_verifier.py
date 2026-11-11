# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one task's pinned WANDR verifier with Gym's judge endpoint and, optionally, a direct page fetch.

Usage: ``python run_verifier.py <task>/tests/verify.py [verify.py arguments]``, with PYTHONPATH holding the
upstream ``wandr_core`` directory and the task's ``tests`` directory (the layout ``benchmarks/wandr/prepare.py``
writes). The upstream verifier is unchanged; this entry point only swaps two clients before it is imported:

- ``WANDR_GYM_JUDGE_MODEL_TEMPLATE`` (default ``{model}``) renames every judge model, so the task configs'
  ``gpt-5.4`` reaches an OpenAI-compatible gateway (``OPENAI_BASE_URL``) under its own name.
- ``WANDR_GYM_FETCH_BACKEND`` picks the page fetch: ``exa`` uses Exa ``/contents`` (``WANDR_GYM_EXA_API_KEY``),
  ``direct`` a plain HTTP fetch, and ``perplexity`` keeps the upstream fetch, which needs ``PERPLEXITY_API_KEY``.

It also guards the score against a busy or failing judge endpoint:

- ``WANDR_GYM_LLM_TIMEOUT_SEC`` replaces the fixed per-call timeouts of the triage, canon, dedup and judge stages
  (upstream: 15, 60, 30 and 60 s). Upstream drops the record of a call that times out or fails.
- When a call failed and the verifier judged fewer records than ``WANDR_HEAL_MIN_COVERAGE`` (upstream's own
  threshold, default 0.90), the run exits 4 with ``error.json``: the score would rest on judging that never ran.
"""

import importlib.util
import json
import os
import sys
import types

# Run as a script, so this directory is first on sys.path.
from web import EXA_BASE_URL, DirectFetchClient, ExaFetchClient, gym_get, gym_post_json


def _stub_browser() -> None:
    # The upstream client module imports camoufox for its browser fallback. Gym runs with the fallback off
    # (WANDR_BROWSER_FALLBACK=0), so a stub keeps the browser dependency out of the server environment.
    if importlib.util.find_spec("camoufox") is not None:
        return

    class AsyncCamoufox:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("WANDR's browser fallback is not available in Gym; keep WANDR_BROWSER_FALLBACK=0")

    package = types.ModuleType("camoufox")
    async_api = types.ModuleType("camoufox.async_api")
    async_api.AsyncCamoufox = AsyncCamoufox
    package.async_api = async_api
    sys.modules["camoufox"] = package
    sys.modules["camoufox.async_api"] = async_api


def _rename_judge_models(template: str) -> None:
    import src.runtime.llm as llm

    upstream_completion = llm.llm_completion

    async def llm_completion(client, request):
        return await upstream_completion(client, request._replace(model=template.format(model=request.model)))

    llm.llm_completion = llm_completion


def _count_llm_failures() -> list[int]:
    import src.runtime.llm as llm

    upstream_completion = llm.llm_completion
    failures = [0]

    async def llm_completion(client, request):
        try:
            return await upstream_completion(client, request)
        except BaseException:
            # Includes the cancellation of a call that reached its stage timeout.
            failures[0] += 1
            raise

    llm.llm_completion = llm_completion
    return failures


def _set_llm_timeouts(seconds: float) -> None:
    import src.components as components

    for name in ("_TRIAGE_TIMEOUT", "_CANON_TIMEOUT", "_DEDUP_TIMEOUT", "_JUDGE_TIMEOUT"):
        setattr(components, name, seconds)


def _incomplete_judging(logs: str) -> str | None:
    """Name the shortfall when the verifier judged fewer records than upstream's healing coverage."""
    try:
        with open(os.path.join(logs, "wandr-details.json")) as stream:
            metadata = json.load(stream)["metadata"]
    except (OSError, ValueError, KeyError):
        return None
    submitted = metadata.get("submitted_record_count") or 0
    judged = metadata.get("judged_record_count") or 0
    min_coverage = float(os.environ.get("WANDR_HEAL_MIN_COVERAGE", "0.90"))
    if submitted and judged / submitted < min_coverage:
        return f"judged {judged} of {submitted} records after failed judge calls (coverage below {min_coverage:g})"
    return None


def _write_error(message: str) -> None:
    logs = os.environ.get("LOGS_DIR")
    if logs:
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, "error.json"), "w") as stream:
            json.dump({"error": message}, stream)


def _direct_fetch_client() -> type:
    timeout = float(os.environ.get("WANDR_GYM_FETCH_TIMEOUT_SEC", "45"))
    max_bytes = int(os.environ.get("WANDR_GYM_FETCH_MAX_BYTES", "20000000"))

    class VerifierFetchClient(DirectFetchClient):
        """Built by the upstream pool through ``FetchClient.from_env()``."""

        def __init__(self) -> None:
            super().__init__(gym_get, timeout=timeout, max_bytes=max_bytes)

        @classmethod
        def from_env(cls, *_args: object, **_kwargs: object) -> "VerifierFetchClient":
            return cls()

    return VerifierFetchClient


def _exa_fetch_client() -> type:
    api_key = os.environ["WANDR_GYM_EXA_API_KEY"]
    base_url = os.environ.get("WANDR_GYM_EXA_BASE_URL", EXA_BASE_URL)
    timeout = float(os.environ.get("WANDR_GYM_FETCH_TIMEOUT_SEC", "120"))
    max_age = os.environ.get("WANDR_GYM_EXA_MAX_AGE_HOURS")

    class VerifierFetchClient(ExaFetchClient):
        """Built by the upstream pool through ``FetchClient.from_env()``."""

        def __init__(self) -> None:
            super().__init__(
                gym_post_json,
                api_key=api_key,
                timeout=timeout,
                base_url=base_url,
                max_age_hours=int(max_age) if max_age else None,
            )

        @classmethod
        def from_env(cls, *_args: object, **_kwargs: object) -> "VerifierFetchClient":
            return cls()

    return VerifierFetchClient


def main() -> None:
    verify_path = sys.argv[1]
    _stub_browser()
    _rename_judge_models(os.environ.get("WANDR_GYM_JUDGE_MODEL_TEMPLATE", "{model}"))
    llm_failures = _count_llm_failures()
    if timeout := os.environ.get("WANDR_GYM_LLM_TIMEOUT_SEC"):
        _set_llm_timeouts(float(timeout))
    backend = os.environ.get("WANDR_GYM_FETCH_BACKEND", "direct")
    if backend in ("direct", "exa"):
        import src.runtime.clients as clients

        clients.FetchClient = _direct_fetch_client() if backend == "direct" else _exa_fetch_client()
    sys.argv = [verify_path, *sys.argv[2:]]
    # Load verify.py under its own name, not as __main__: the page-extraction workers start with the spawn
    # method, which re-imports __main__, and this guarded script is safe to re-import where verify.py is not.
    spec = importlib.util.spec_from_file_location("wandr_task_verify", verify_path)
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    code = verify.main()
    if ExaFetchClient.fatal_errors:
        # The scores would rest on pages that were never fetched; report an inconclusive verification instead.
        _write_error(ExaFetchClient.fatal_errors[0])
        raise SystemExit(3)
    if code == 0 and llm_failures[0] and (shortfall := _incomplete_judging(os.environ.get("LOGS_DIR", ""))):
        _write_error(shortfall)
        raise SystemExit(4)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
