# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze UserSim identity_disclosure episodes for the model identity benchmark.

Every opening and every pressure follow-up comes from UserSim's packaged identity spec. This script
only overrides the expected identity and pins the tactic, strategy and locale of each row, so every
cell the report breaks down by is covered evenly. Each row gets its own persona, and UserSim derives
the rest (the competitor named, the generated openings and follow-ups) from the persona and the seed.
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
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import yaml

from benchmarks.model_identity.report import DIMENSIONS, OPENINGS_NOT_ASKING_IDENTITY
from environments.nemo_user_sim.prepare import ENVIRONMENT_DIR as PREPARE_PROJECT
from environments.nemo_user_sim.prepare import NEMOTRON_PERSONAS_SHA256, NEMOTRON_PERSONAS_VERSIONS, USERSIM_REVISION


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
DATA = DIRECTORY / "data" / "benchmark.jsonl"
VERSION = "1.0.0"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
# Other spellings that also count as the expected model, keyed by its casefolded name: the
# transliterations UserSim's vocabulary lists for Nemotron. Other models default to none.
DEFAULT_MODEL_ALIASES = {"nemotron": ("ネモトロン", "네모트론", "नेमोट्रॉन")}


def _settings(
    *,
    single_per_tactic: int,
    multi_per_strategy: int,
    locales,
    tactics,
    seed: int,
    max_turns: int,
    temperature: float | None,
    top_p: float | None = None,
    expected_developer: str,
    expected_model: str,
    expected_model_aliases=None,
    expected_lineage=None,
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
    if locales is not None and not locales:
        raise ValueError("Expected at least one locale")
    if expected_model_aliases is None:
        expected_model_aliases = DEFAULT_MODEL_ALIASES.get(model.casefold(), ())
    return {
        "single_per_tactic": single_per_tactic,
        "multi_per_strategy": multi_per_strategy,
        "locales": None if locales is None else list(locales),
        "tactics": None if tactics is None else list(tactics),
        "seed": seed,
        "max_turns": max_turns,
        "temperature": temperature,
        "top_p": top_p,
        "expected_developer": developer,
        "expected_model": model,
        "expected_model_names": [model, *expected_model_aliases],
        "expected_lineage": sorted({d.strip().casefold() for d in expected_lineage or () if d.strip()}),
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


def _write_spec(
    path: Path, expected_developer: str, expected_model_names: list[str], expected_lineage: list[str]
) -> None:
    """Layer the expected identity over UserSim's packaged spec, keeping its tactics and strategies.

    ``expected_lineage`` lists the developers whose base model the evaluated model is built on,
    so a reply naming them as its base is not a false attribution.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": "1",
        "extends": "identity_disclosure",
        "bank_id": "model_identity",
        "bank_version": VERSION,
        "placeholder": True,
        # Endpoint model names can be arbitrary (e.g. "checkpoint-123"); "*" maps any name
        # to the configured expected identity. UserSim also applies this rule to the judge's
        # developer lookup, which scoring.py corrects when checking judge independence.
        # TODO: Replace "*" if UserSim can set the tested model's expected identity without affecting the judge.
        "expected_identities": [
            {
                "match": "*",
                "developer": expected_developer,
                "model_names": list(expected_model_names),
                **({"lineage": list(expected_lineage)} if expected_lineage else {}),
            },
        ],
    }
    path.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n" + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
    )


def _written_locales(spec) -> list[str]:
    """The locales UserSim's spec writes every verbatim opening in, in the spec's order."""
    verbatim = [tactic.text for tactic in spec.tactics if tactic.text is not None and tactic.mode.value == "verbatim"]
    if not verbatim:
        return ["en_US"]
    common = set.intersection(*(set(text.renderings) for text in verbatim))
    return [locale for locale in verbatim[0].renderings if locale in common]


def _cells(
    tactics: list, strategies: list[str], locales: list[str], *, single_per_tactic: int, multi_per_strategy: int
):
    """Every tactic single-turn, and every pressure strategy after tactics taken in turn, per locale.

    Each locale starts the strategies at a later tactic, so a strategy meets different openings in
    different locales rather than the same few in all of them.
    """
    cells = []
    for shift, locale in enumerate(locales):
        for tactic in tactics:
            cells.extend((locale, tactic, "none", i) for i in range(single_per_tactic))
        for s, strategy in enumerate(strategies):
            for i in range(multi_per_strategy):
                position = (s + shift) * multi_per_strategy + i
                cells.append((locale, tactics[position % len(tactics)], strategy, i))
    return cells


def _check_persona_assets(locales: list[str], *, synthetic: bool) -> None:
    """Reject unsupported locales and report every missing dataset before sampling."""
    unsupported = [locale for locale in locales if locale not in NEMOTRON_PERSONAS_VERSIONS]
    if unsupported:
        raise ValueError(f"Unsupported persona locales: {', '.join(unsupported)}")
    if synthetic:
        return
    from environments.nemo_user_sim.prepare import _managed_assets_path

    datasets = _managed_assets_path() / "datasets"
    missing = [locale for locale in locales if not (datasets / f"{locale}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pinned personas are not cached for: {', '.join(missing)}. Run gym eval prepare "
            "--config environments/nemo_user_sim/config.yaml "
            f"'++prepare_script_args.locales=[{','.join(missing)}]' with NGC and user-model settings configured, "
            "then retry model_identity preparation."
        )


def _personas(*, locale: str, seed: int, count: int, synthetic: bool) -> list[dict]:
    """Sample the locale's personas in a seeded order (one synthetic persona for tests)."""
    _check_persona_assets([locale], synthetic=synthetic)
    if synthetic:
        example = ROOT / "resources_servers/nemo_user_sim/data/example.jsonl"
        return [json.loads(example.read_text().splitlines()[0])["resolved_row"]["persona"]]
    import pyarrow.parquet as pq

    from environments.nemo_user_sim.prepare import _managed_assets_path

    path = _managed_assets_path() / "datasets" / f"{locale}.parquet"
    if _digest(path) != NEMOTRON_PERSONAS_SHA256[locale]:
        raise ValueError(
            f"Persona asset does not match pinned Nemotron-Personas {locale} {NEMOTRON_PERSONAS_VERSIONS[locale]}"
        )
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
    # The sample is read in file order; shuffle it so rows do not follow the dataset's ordering.
    random.Random(seed).shuffle(selected)
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
    locales=None,
    tactics=None,
    synthetic: bool = False,
    temperature: float | None = DEFAULT_TEMPERATURE,
    top_p: float | None = DEFAULT_TOP_P,
    expected_developer: str = "nvidia",
    expected_model: str = "Nemotron",
    expected_model_aliases: list[str] | None = None,
    expected_lineage: list[str] | None = None,
) -> dict:
    from usersim.engine.config import ConversationSimulatorConfig
    from usersim.engine.external import construct_probe_episode
    from usersim.engine.probes.identity_disclosure.spec import load_identity_spec, reset_identity_spec_cache

    settings = _settings(
        single_per_tactic=single_per_tactic,
        multi_per_strategy=multi_per_strategy,
        locales=locales,
        tactics=tactics,
        seed=seed,
        max_turns=max_turns,
        temperature=temperature,
        top_p=top_p,
        expected_developer=expected_developer,
        expected_model=expected_model,
        expected_model_aliases=expected_model_aliases,
        expected_lineage=expected_lineage,
    )
    spec_path = output.with_suffix(".spec.yaml")
    _write_spec(
        spec_path, settings["expected_developer"], settings["expected_model_names"], settings["expected_lineage"]
    )
    spec = load_identity_spec(spec_path)
    if settings["locales"] is None:
        settings["locales"] = _written_locales(spec)
    _check_persona_assets(settings["locales"], synthetic=synthetic)
    by_id = {tactic.id: tactic for tactic in spec.tactics}
    # The identity probe also tests indirect identity errors (e.g. knowledge-cutoff questions).
    # These answers need not name the developer or model family, but any identity claims must be correct.
    # Guard the list of these tactics: a renamed or removed id would silently require identity disclosure.
    # TODO: Replace this list and guard if UserSim adds a flag indicating whether a tactic asks for identity.
    stale = sorted(OPENINGS_NOT_ASKING_IDENTITY - set(by_id))
    if stale:
        raise ValueError(f"OPENINGS_NOT_ASKING_IDENTITY names tactics UserSim no longer has: {stale}")
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
    personas = {
        locale: cycle(_personas(locale=locale, seed=seed, count=count, synthetic=synthetic))
        for locale, count in Counter(locale for locale, *_ in cells).items()
    }
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
                    "persona": next(personas[locale]),
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
            provenance.update(
                code_sha=USERSIM_REVISION,
                nemotron_personas_version="synthetic" if synthetic else NEMOTRON_PERSONAS_VERSIONS[locale],
            )
            row["usersim_provenance"] = json.dumps(provenance, sort_keys=True)
            # A flat row: Gym adds the dataset's taskset and passes the rest to UserSim as the task input.
            destination.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "resolved_row": row,
                        "role_request_params": {
                            # The policy samples, so the rows of a cell are samples of its answers; a
                            # setting given as None is left to the endpoint.
                            "assistant": {
                                **({} if temperature is None else {"temperature": temperature}),
                                **({} if top_p is None else {"top_p": top_p}),
                                "max_output_tokens": 8192,
                            },
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
        "usersim_revision": USERSIM_REVISION,
        "usersim_spec": spec.version,
        "personas_versions": {
            locale: "synthetic" if synthetic else NEMOTRON_PERSONAS_VERSIONS[locale] for locale in settings["locales"]
        },
        "personas_sha256": (
            None if synthetic else {locale: NEMOTRON_PERSONAS_SHA256[locale] for locale in settings["locales"]}
        ),
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
            manifest["usersim_revision"] == USERSIM_REVISION
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
    locales: list[str] | None = None,
    tactics: list[str] | None = None,
    seed: int = 1042,
    max_turns: int = 5,
    temperature: float | None = DEFAULT_TEMPERATURE,
    top_p: float | None = DEFAULT_TOP_P,
    expected_developer: str = "nvidia",
    expected_model: str = "Nemotron",
    expected_model_aliases: list[str] | None = None,
    expected_lineage: list[str] | None = None,
) -> Path:
    """Create a frozen panel; preparation performs no model calls.

    Per locale (by default, every locale UserSim's spec writes all openings in): `single_per_tactic`
    single-turn rows for each UserSim tactic (all of them, or `tactics`), and `multi_per_strategy`
    multi-turn rows for each pressure strategy.
    `temperature` and `top_p` set how the policy samples (1.0 and 0.95 by default), so the rows of
    a cell are samples of its answers rather than copies of one; None leaves one to the endpoint.
    `expected_model_aliases` lists other spellings that also count as the expected model.
    When omitted, it defaults to `DEFAULT_MODEL_ALIASES` for that model, if any.
    `expected_lineage` lists UserSim developer ids whose base model the evaluated model is built on.
    """
    options = {
        "single_per_tactic": single_per_tactic,
        "multi_per_strategy": multi_per_strategy,
        "locales": None if locales is None else list(locales),
        "tactics": None if tactics is None else list(tactics),
        "seed": seed,
        "max_turns": max_turns,
        "temperature": temperature,
        "top_p": top_p,
        "expected_developer": expected_developer,
        "expected_model": expected_model,
        "expected_model_aliases": None if expected_model_aliases is None else list(expected_model_aliases),
        "expected_lineage": None if expected_lineage is None else list(expected_lineage),
    }
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError("'uv' is not on PATH; it is required to prepare model identity tasks.")
    # Run in Gym's UserSim preparation project, which pins UserSim, as its own preparer does.
    with tempfile.TemporaryDirectory(prefix="model-identity-prepare-") as working_dir:
        subprocess.run(
            [
                executable,
                "run",
                "--locked",
                "--project",
                str(PREPARE_PROJECT),
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
                # Never sync the preparation project into Gym's own environment.
                "UV_PROJECT_ENVIRONMENT": str(Path(working_dir) / ".venv"),
                "PYTHONPATH": str(ROOT),
                "USERSIM_CODE_SHA": USERSIM_REVISION,
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
