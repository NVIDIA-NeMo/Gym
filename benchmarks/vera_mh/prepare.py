# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the VERA-MH recommended evaluation profile as NeMo Gym tasks.

VERA-MH (Belli et al., 2026, arXiv:2605.13318; https://github.com/SpringCare/VERA-MH) simulates conversations between
a clinician-authored user persona (played by a "user agent" LLM) and the chatbot under evaluation, then judges every
transcript against a clinical rubric. The repository's recommended, score-comparable profile (README "Recommended
settings", commit ``2c9d1fc``) is:

- all 100 personas in ``data/personas.tsv``;
- one 30-turn conversation per persona with GPT 5.2 as the user agent and one with Claude Opus 4.5;
- GPT 5.4 (``reasoning_effort=low``) as the judge, pooled over both user-agent suites.

This script pins the upstream commit, verifies the SHA-256 of every file it reads, re-checks the vendored rubric
assets under ``resources_servers/vera_mh/rubric/``, and emits one task per (persona, user simulator): 200 rows. The
persona system prompt is rendered exactly as upstream's ``load_prompts_from_csv`` does
(``template.format(**row)``), and the row carries the persona sheet fields as slices.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "vera_mh_recommended.jsonl"
RUBRIC_DIR = REPO_ROOT / "resources_servers" / "vera_mh" / "rubric"

UPSTREAM_REPOSITORY = "SpringCare/VERA-MH"
UPSTREAM_REVISION = "2c9d1fcbb68e1a2df64171c18b3e4d4c18b2f89e"  # pragma: allowlist secret
UPSTREAM_RUBRIC_TAG = "v1.2.0"
RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPOSITORY}/{UPSTREAM_REVISION}"
PERSONAS_SHA256 = "07f0aa92cde50469d18aff640ed03e0df93e8863aed75ea102e4ba97a124330b"  # pragma: allowlist secret
VENDORED_SHA256 = {
    "rubric.tsv": "5bc029ea47395c35d34dc28dc0f2eef907796e1bf0f9b78b5cf68d444809cdee",  # pragma: allowlist secret
    "rubric_prompt_beginning.txt": "9c5d1d6099a89b4d66cabef8320bb455732e80940a2e40b4baebb1f37f1e31f0",  # pragma: allowlist secret
    "question_prompt.txt": "6c7487face8e9e2e2348cce03afb713f8586b06f62627f1bab01a146a5b8e584",  # pragma: allowlist secret
    "persona_prompt_template.txt": "14ddd27dde3b15bd62db54a0b2e129bf040423b60c7fe3aff0fa420b71d3ca1f",  # pragma: allowlist secret
    "persona_prompt_reminder.txt": "9efd2a2bbfa1ede2bfb487d2f9a44b8900eabbd86a118a28d43d9e6ab54e194a",  # pragma: allowlist secret
}
UPSTREAM_PATHS = {name: f"data/{name}" for name in VENDORED_SHA256}
EXPECTED_PERSONAS = 100
USER_SIMULATORS = {
    # key -> upstream model id used by scripts/run_recommended_vera_pipeline.sh (VERA_USER_A / VERA_USER_B)
    "gpt-5.2": "gpt-5.2",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
}
MAX_TURNS = 30
PROVIDER_SYSTEM_PROMPT = "You are a helpful AI assistant."  # upstream default for the provider agent
LICENSE = "Other (Spring Care, Inc. VERA-MH license; see resources_servers/vera_mh/rubric/LICENSE)"
RISK_COLUMN = "Short Current Suicide Risk Level"


def _fetch(path: str, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(f"{RAW_BASE}/{path}", headers={"User-Agent": "nemo-gym-vera-mh-prepare/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_vendored_assets(rubric_dir: Path = RUBRIC_DIR, *, upstream: bool = True) -> dict[str, str]:
    """Check the vendored rubric assets against the pinned hashes (and, optionally, the pinned upstream files)."""
    digests: dict[str, str] = {}
    for name, expected in VENDORED_SHA256.items():
        local = (rubric_dir / name).read_bytes()
        digest = _sha256(local)
        if digest != expected:
            raise ValueError(f"vendored {name} SHA-256 {digest} does not match the pinned upstream file {expected}")
        if upstream:
            remote = _fetch(UPSTREAM_PATHS[name])
            if _sha256(remote) != expected:
                raise ValueError(f"upstream {UPSTREAM_PATHS[name]} at {UPSTREAM_REVISION} changed: {_sha256(remote)}")
        digests[name] = digest
    return digests


def render_persona_prompt(template: str, row: dict[str, str]) -> str:
    """Upstream ``generate_conversations.utils.load_prompts_from_csv``: ``template.format(**row)``."""
    return template.format(**row)


def _render(personas_tsv: bytes, template: str) -> tuple[str, int]:
    rows = list(csv.DictReader(io.StringIO(personas_tsv.decode("utf-8")), delimiter="\t"))
    if len(rows) != EXPECTED_PERSONAS:
        raise ValueError(f"expected {EXPECTED_PERSONAS} personas, found {len(rows)}")
    names = [row["Name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("persona names are not unique")
    output: list[str] = []
    for persona_index, row in enumerate(rows, start=1):
        prompt = render_persona_prompt(template, row)
        persona_sha256 = _sha256("\t".join(row[key] for key in row).encode("utf-8"))
        for simulator_key, simulator_model in USER_SIMULATORS.items():
            output.append(
                json.dumps(
                    {
                        "id": f"vera_mh__{row['Name']}__{simulator_key}",
                        "persona_index": persona_index,
                        "persona_name": row["Name"],
                        "persona": dict(row),
                        "persona_sha256": persona_sha256,
                        "persona_system_prompt": prompt,
                        "user_simulator": simulator_key,
                        "user_simulator_upstream_model": simulator_model,
                        "risk_level": row[RISK_COLUMN],
                        "disclosure_level": row["Disclosure of Suicide Risk"].split(" (", 1)[0],
                        "max_turns": MAX_TURNS,
                        "responses_create_params": {"input": [{"role": "system", "content": PROVIDER_SYSTEM_PROMPT}]},
                        "upstream": {
                            "repository": UPSTREAM_REPOSITORY,
                            "revision": UPSTREAM_REVISION,
                            "rubric_tag": UPSTREAM_RUBRIC_TAG,
                            "personas_sha256": PERSONAS_SHA256,
                            "license": LICENSE,
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return "".join(output), len(output)


def _atomic_write(content: str, output_fpath: Path) -> None:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=output_fpath.parent, prefix=f".{output_fpath.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_fpath)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(
    source_personas: Optional[str] = None,
    output_fpath: Optional[str] = None,
    rubric_dir: Optional[str] = None,
    verify_upstream: bool = True,
) -> Path:
    """Download (or read ``source_personas``), verify every pin, and write one task per persona and simulator."""
    rubric_path = Path(rubric_dir) if rubric_dir else RUBRIC_DIR
    verify_vendored_assets(rubric_path, upstream=verify_upstream and source_personas is None)
    personas = Path(source_personas).read_bytes() if source_personas else _fetch("data/personas.tsv")
    digest = _sha256(personas)
    if digest != PERSONAS_SHA256:
        raise ValueError(f"personas.tsv SHA-256 mismatch: expected {PERSONAS_SHA256}, got {digest}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if source_personas is None:
        (DATA_DIR / "personas.tsv").write_bytes(personas)
    template = (rubric_path / "persona_prompt_template.txt").read_text(encoding="utf-8")
    rendered, count = _render(personas, template)
    target = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    _atomic_write(rendered, target)
    print(
        f"Wrote {count} VERA-MH tasks ({EXPECTED_PERSONAS} personas x {len(USER_SIMULATORS)} user simulators) to {target}"
    )
    return target


if __name__ == "__main__":
    prepare()
