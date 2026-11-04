# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze UserSim identity_disclosure episodes for the model identity benchmark.

Every opening and every pressure follow-up comes from UserSim's packaged identity spec. This script
only overrides the expected identity and pins which tactic, strategy, locale and persona each row
gets, so every cell is covered evenly.
"""

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import yaml

from benchmarks.model_identity.report import DIMENSIONS


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
DATA = DIRECTORY / "data" / "benchmark.jsonl"
VERSION = "1.0.0"
REVISION = "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee"  # pragma: allowlist secret
PERSONAS_SHA256 = "0341192b00a376cf5643d98cb244e596529030fb3011694ca6ab381f149d3ae8"  # pragma: allowlist secret
REQUIREMENTS = ROOT / "environments/nemo_user_sim/requirements.txt"
# Other spellings that also count as the expected model, keyed by its casefolded name: the
# transliterations UserSim's vocabulary lists for Nemotron. Other models default to none.
DEFAULT_MODEL_ALIASES = {"nemotron": ("ネモトロン", "네모트론", "नेमोट्रॉन")}
# The locales UserSim's spec writes every verbatim opening in. Personas are en_US in all of them.
DEFAULT_LOCALES = ("en_US", "pt_BR", "fr_FR", "ja_JP", "ko_KR", "hi_Deva_IN", "hi_Latn_IN")


def _settings(
    *,
    single_per_tactic: int,
    multi_per_strategy: int,
    locales,
    tactics,
    seed: int,
    max_turns: int,
    temperature: float,
    expected_developer: str,
    expected_model: str,
    expected_model_aliases=None,
) -> dict:
    """Normalized preparation settings, as the manifest records them."""
    developer = expected_developer.strip().casefold()
    model = expected_model.strip()
    if not developer or not model:
        raise ValueError("Expected developer and model must be nonempty")
    if single_per_tactic < 0 or multi_per_strategy < 0 or not single_per_tactic + multi_per_strategy:
        raise ValueError("Row counts must be nonnegative and not both zero")
    if multi_per_strategy and max_turns < 2:
        raise ValueError("Multi-turn rows require max_turns >= 2")
    if not locales:
        raise ValueError("Expected at least one locale")
    if expected_model_aliases is None:
        expected_model_aliases = DEFAULT_MODEL_ALIASES.get(model.casefold(), ())
    return {
        "single_per_tactic": single_per_tactic,
        "multi_per_strategy": multi_per_strategy,
        "locales": list(locales),
        "tactics": None if tactics is None else list(tactics),
        "seed": seed,
        "max_turns": max_turns,
        "temperature": temperature,
        "expected_developer": developer,
        "expected_model": model,
        "expected_model_names": [model, *expected_model_aliases],
    }


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def _use_spec(path: Path):
    """Scope the native spec override to preparation, leaving the caller's environment intact."""
    key = "USERSIM_IDENTITY_DISCLOSURE_SPEC"
    previous = os.environ.get(key)
    os.environ[key] = str(path.resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _write_spec(path: Path, expected_developer: str, expected_model_names: list[str]) -> None:
    """Layer the expected identity over UserSim's packaged spec, keeping its tactics and strategies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": "1",
        "extends": "identity_disclosure",
        "bank_id": "model_identity",
        "bank_version": VERSION,
        "placeholder": True,
        # UserSim picks the expected identity by matching the evaluated model's name against `match`.
        # "*" matches any name, so every endpoint is graded as Nemotron, whatever it is called.
        "expected_identities": [
            {
                "match": "*",
                "developer": expected_developer,
                "model_names": list(expected_model_names),
            },
        ],
    }
    path.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n" + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
    )


def _cells(
    tactics: list, strategies: list[str], locales: list[str], *, single_per_tactic: int, multi_per_strategy: int
):
    """Every tactic single-turn, and every pressure strategy after tactics taken in turn, per locale."""
    cells = []
    for locale in locales:
        for tactic in tactics:
            cells.extend((locale, tactic, "none", i) for i in range(single_per_tactic))
        for s, strategy in enumerate(strategies):
            for i in range(multi_per_strategy):
                cells.append((locale, tactics[(s * multi_per_strategy + i) % len(tactics)], strategy, i))
    return cells


def _personas(*, seed: int, count: int, synthetic: bool) -> list[dict]:
    if synthetic:
        example = ROOT / "resources_servers/nemo_user_sim/data/example.jsonl"
        return [json.loads(example.read_text().splitlines()[0])["resolved_row"]["persona"]]
    import pyarrow.parquet as pq

    # Honor the same cache settings as the canonical UserSim preparer.
    from environments.nemo_user_sim.prepare import _managed_assets_path

    path = _managed_assets_path() / "datasets/en_US.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            "Pinned personas are not cached. Run environments/nemo_user_sim/prepare.py "
            "with NGC configured, then retry model_identity preparation."
        )
    if _digest(path) != PERSONAS_SHA256:
        raise ValueError("Persona asset does not match pinned Nemotron-Personas 0.0.2")
    parquet = pq.ParquetFile(path)
    indices = sorted(
        random.Random(seed).sample(range(parquet.metadata.num_rows), min(count, parquet.metadata.num_rows))
    )
    selected = []
    offset = 0
    for group in range(parquet.num_row_groups):
        end = offset + parquet.metadata.row_group(group).num_rows
        within = [i - offset for i in indices if offset <= i < end]
        if within:
            table = parquet.read_row_group(group)
            selected.extend(table.slice(i, 1).to_pylist()[0] for i in within)
        offset = end
    return selected


def _materialize(output: Path, **options) -> dict:
    """Build the panel beside the previous one and replace it only once all of it is built."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
    try:
        manifest = _build(staging / output.name, **options)
        for suffix in (".jsonl", ".spec.yaml", ".manifest.json"):
            os.replace((staging / output.name).with_suffix(suffix), output.with_suffix(suffix))
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _build(
    output: Path,
    *,
    single_per_tactic: int,
    multi_per_strategy: int,
    seed: int,
    max_turns: int,
    locales=("en_US",),
    tactics=None,
    synthetic: bool = False,
    temperature: float = 0.0,
    expected_developer: str = "nvidia",
    expected_model: str = "Nemotron",
    expected_model_aliases: list[str] | None = None,
) -> dict:
    from usersim.engine.config import ConversationSimulatorConfig
    from usersim.engine.core.identity_spec import load_identity_spec, reset_identity_spec_cache
    from usersim.engine.external import construct_probe_episode

    settings = _settings(
        single_per_tactic=single_per_tactic,
        multi_per_strategy=multi_per_strategy,
        locales=locales,
        tactics=tactics,
        seed=seed,
        max_turns=max_turns,
        temperature=temperature,
        expected_developer=expected_developer,
        expected_model=expected_model,
        expected_model_aliases=expected_model_aliases,
    )
    spec_path = output.with_suffix(".spec.yaml")
    _write_spec(spec_path, settings["expected_developer"], settings["expected_model_names"])
    spec = load_identity_spec(spec_path)
    by_id = {tactic.id: tactic for tactic in spec.tactics}
    if settings["tactics"] is not None:
        unknown = sorted(set(settings["tactics"]) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown UserSim tactics: {unknown}")
    selected = [by_id[t] for t in settings["tactics"]] if settings["tactics"] is not None else list(spec.tactics)
    strategies = [strategy.id for strategy in spec.strategies if strategy.id != "none"]
    cells = _cells(
        selected,
        strategies,
        settings["locales"],
        single_per_tactic=single_per_tactic,
        multi_per_strategy=multi_per_strategy,
    )
    personas = _personas(seed=seed, count=64, synthetic=synthetic)
    models = {
        alias: SimpleNamespace(model_name=name)
        for alias, name in (
            ("user_model", "user_model"),
            ("assistant_model", settings["expected_model"]),
            ("judge_model", "judge_model"),
            ("api_response_model", "api_response_model"),
            ("summary_model", "summary_model"),
        )
    }
    cases = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with _use_spec(spec_path), output.open("w") as destination:
        # The spec is cached per path; drop any earlier version of this one.
        reset_identity_spec_cache()
        for locale, tactic, strategy, index in cells:
            task_id = f"model-identity-v{VERSION}-{locale}-{tactic.id}-{strategy}-{index}"
            case = {
                "tactic": tactic.id,
                "tactic_kind": "neutral" if tactic.neutral else "leading",
                "strategy": strategy,
                "locale": locale,
                "mode": "single_turn" if strategy == "none" else "multi_turn",
            }
            cases.append(case)
            config = ConversationSimulatorConfig(
                name="conversation_messages",
                locale=locale,
                random_seed=seed,
                max_turns=1 if strategy == "none" else max_turns,
                max_assistant_attempts=1,
                store_reasoning=True,
            )
            episode = construct_probe_episode(
                {
                    "persona": personas[int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(personas)],
                    "probe_type": "identity_disclosure",
                    "identity_tactic_id": tactic.id,
                    "identity_strategy_id": strategy,
                    "theme": "__not_used__",
                    "trajectory_id": task_id,
                    "identity_case": case,
                },
                config=config,
                models=models,
            )
            row = episode.preamble.to_row()
            row["usersim_config"] = config.model_dump(mode="json")
            provenance = json.loads(row["usersim_provenance"])
            provenance.update(code_sha=REVISION, nemotron_personas_version="synthetic" if synthetic else "0.0.2")
            row["usersim_provenance"] = json.dumps(provenance, sort_keys=True)
            # A flat row: Gym adds the dataset's taskset and passes the rest to UserSim as the task input.
            destination.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "resolved_row": row,
                        "role_request_params": {
                            "assistant": {"temperature": temperature, "max_output_tokens": 4096},
                            "user": {"temperature": 0.7},
                            "judge": {"temperature": 0.0},
                        },
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    manifest = {
        "benchmark": "model_identity",
        "version": VERSION,
        "usersim_revision": REVISION,
        "usersim_spec": spec.version,
        "personas_version": "synthetic" if synthetic else "0.0.2",
        "personas_sha256": None if synthetic else PERSONAS_SHA256,
        "dataset_sha256": _digest(output),
        "spec_sha256": _digest(spec_path),
        "generator_sha256": _digest(Path(__file__)),
        "settings": settings,
        "task_count": len(cases),
        **{dimension: dict(Counter(case[dimension] for case in cases)) for dimension in DIMENSIONS},
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def is_prepared_data_current(path: str | Path = DATA) -> bool:
    """Whether the cached panel is intact and was prepared by this code."""
    path = Path(path)
    try:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        return (
            manifest["usersim_revision"] == REVISION
            and manifest["version"] == VERSION
            and manifest["generator_sha256"] == _digest(Path(__file__))
            and manifest["dataset_sha256"] == _digest(path)
            and manifest["spec_sha256"] == _digest(path.with_suffix(".spec.yaml"))
        )
    except (OSError, KeyError, ValueError):
        return False


def prepare(
    single_per_tactic: int = 5,
    multi_per_strategy: int = 10,
    locales: list[str] = DEFAULT_LOCALES,
    tactics: list[str] | None = None,
    seed: int = 1042,
    max_turns: int = 5,
    temperature: float = 0.0,
    expected_developer: str = "nvidia",
    expected_model: str = "Nemotron",
    expected_model_aliases: list[str] | None = None,
) -> Path:
    """Create a frozen panel; preparation performs no model calls.

    Per locale: `single_per_tactic` single-turn rows for each UserSim tactic (all of them, or
    `tactics`), and `multi_per_strategy` multi-turn rows for each pressure strategy.
    `expected_model_aliases` lists other spellings that also count as the expected model.
    When omitted, it defaults to `DEFAULT_MODEL_ALIASES` for that model, if any.
    """
    options = {
        "single_per_tactic": single_per_tactic,
        "multi_per_strategy": multi_per_strategy,
        "locales": list(locales),
        "tactics": None if tactics is None else list(tactics),
        "seed": seed,
        "max_turns": max_turns,
        "temperature": temperature,
        "expected_developer": expected_developer,
        "expected_model": expected_model,
        "expected_model_aliases": None if expected_model_aliases is None else list(expected_model_aliases),
    }
    subprocess.run(
        [
            "uv",
            "run",
            "--no-config",
            "--no-project",
            "--isolated",
            "--python",
            sys.executable,
            "--with-requirements",
            str(REQUIREMENTS),
            "python",
            "-m",
            "benchmarks.model_identity.prepare",
            "--worker",
            str(DATA),
            json.dumps(options),
        ],
        check=True,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "USERSIM_CODE_SHA": REVISION,
            "USERSIM_NEMOTRON_PERSONAS_VERSION": "0.0.2",
        },
        timeout=3600,
    )
    print(f"Prepared model identity tasks: {DATA}")
    return DATA


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(_materialize(Path(sys.argv[2]), **json.loads(sys.argv[3])), indent=2))
    else:
        prepare()
