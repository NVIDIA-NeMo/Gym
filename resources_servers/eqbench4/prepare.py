# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build EQ-Bench 4 rows and the reference comparison graph from the pinned upstream repository.

Source: https://github.com/EQ-bench/eqbench4 at revision 93dcb7f5d430433ad49d501e672a98bcff2c9678, licensed under
BSD-3-Clause with the Commons Clause condition (Licensor: Samuel J. Paech). Nothing from that repository is stored in
Gym. Preparation downloads the scenario dataset, the prompt and rubric files, the prompt-building module and the
published result files, checks each against a pinned SHA-256, and runs the upstream prompt builder on the downloaded
copy, so the rendered prompts match the leaderboard runs exactly. The prepared files are gitignored.

Every row holds one scenario: the evaluated model's per-turn system prompts and opening message, the persona
simulator's system prompt, the pairwise and trait judge prompts split around their runtime slots, and the transcripts
of the leaderboard models the policy is compared against. The comparison graph holds every published pairwise
comparison; the policy's comparisons are added to it to place the policy on the leaderboard's rating scale.
"""

import hashlib
import importlib
import json
import re
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from resources_servers.eqbench4.judging import NO_FINAL_STATE
from resources_servers.eqbench4.rating import Outcome, canonical, fit_ratings


REVISION = "93dcb7f5d430433ad49d501e672a98bcff2c9678"
SOURCE_URL = f"https://raw.githubusercontent.com/EQ-bench/eqbench4/{REVISION}/"
NUM_TURNS = 8
POLICY_TEMPERATURE = 0.7
POLICY_MAX_OUTPUT_TOKENS = 10000
DEFAULT_OPPONENTS_PER_SCENARIO = 4

DATASET_FILE = "src/eqbench/datasets/test.json"
SOURCE_SHA256 = {
    DATASET_FILE: "afa074b85ad3e4606db3518b40d5d6bac872d6dc192c7dd213f2f03e5492f200",
    # The prompt builder and the modules it imports.
    "src/eqbench/__init__.py": "d16cfe7741f8d8f81344ab4e80eeb0b326c60752b54b94de1a0aaba071d579b7",
    "src/eqbench/core/prompts.py": "fd3021300c74298e7e3605ec065708794bc339d7f5a0b65c3464336489eb22f0",
    "src/eqbench/schema/__init__.py": "858886d7cf06124c4bb19d88c717f767e0a73528f6fcfa17e4a8d64e1ed88357",
    "src/eqbench/schema/base.py": "cc75a2c0bd4c1c8259588298426ec28599e3b9aa1a34381ab4dfdf3c7fc3f7f5",
    "src/eqbench/schema/persona.py": "aa8fe9c01373e5193351ce43073ff4631aa7a7f9fa5a8434ef6b6575f793a909",
    "src/eqbench/schema/registry.py": "0d32fb53c1de761a6bacaa6b133dbfe69005952aadd3d4dd012f7f7521c00795",
    "src/eqbench/schema/scenario.py": "5b4b97ac95820101f49fa06fa5c550c3946bac24db667577bea9a6e9518e6a11",
    "src/eqbench/prompts/evaluated_first_turn_section.txt": (
        "e967cc079b9435ba61bd7c7fc6e556326008d40b0fb1ad60ca2dac8edf7a1885"
    ),
    "src/eqbench/prompts/evaluated_system_prompt.txt": "44258e746938407ed5a729be19f63454becf3f385ca865f4bb0738265cb443a6",
    "src/eqbench/prompts/pairwise_judge_prompt.txt": "37d60b7ed8f1b63fb7335bda515abc597d3eae32453fa617ecd4c7605a2554eb",
    "src/eqbench/prompts/performance_judge_prompt.txt": (
        "dfc73c836c9df72b888e1c4530e618f84010ab5e9337d585b9cfe9881e12ac9e"
    ),
    "src/eqbench/prompts/persona_system_prompt_no_core_issue_no_state_no_unlock.txt": (
        "a0246e80772ddb658e4d45687156757fa218bb4022ce86f885ac91e04393729f"
    ),
    "src/eqbench/prompts/persona_system_prompt_no_state_no_unlock.txt": (
        "c77d4bd2c86dbe869043dc1e5db879d5161556fd33aa41e7f0de435afac93f7f"
    ),
    "src/eqbench/rubrics/eq_judging_rubric.txt": "6111b0d8bf126d370778678891cf2068bc2c9e1940cf7f42ebf2906efe9ccd6f",
    "src/eqbench/rubrics/eq_judging_rubric_ignore_for_scoring.txt": (
        "01c9e88b9fda21740b4a78e861b2bc37d36bf7f87d4b71a6cfe39b7d64d0ecd9"
    ),
    "src/eqbench/rubrics/pairwise_dimensions.txt": "a3da48ffcd5bf3902a944c3653fd6799dbdff8a5d0c523a6ef0cc437a623845f",
    "src/eqbench/rubrics/pairwise_dimensions_weights.txt": (
        "23c7166ca47128c4b12544cbed0b8cd14d386ea42ed04a381bab7c72768d7a83"
    ),
    # Published leaderboard runs: one evaluated model each, with its transcripts and the comparisons its rating run
    # added.
    "results/r23.json": "c4a589e2a366f85723032070ba59ae1088c5897854355c2f588a3e8b76fbd3d3",
    "results/r25.json": "a195a17ebe3131a5d6b0c378845ad132fc6f10132d063328fc6b1312437235c9",
    "results/r32.json": "281d4b81a65075afa2e4c2dd0dca89e0b243469ce1584744aff5a6fd55a9a1ce",
    "results/r34.json": "429e814063e8788dce9864ed87329df6b50c9897730c3610bc1a83cea58b172d",
    "results/r36.json": "d99ebd1a4280d73a2e0b9fc1a07a09d37833026ce76f26ef406c3c4b2286d7bc",
    "results/r37.json": "fdefc52b68cadbc4d1305872ff21cbdb9b46c143e108a5317308ba53dec292df",
    "results/r38.json": "5766abfdab9c6552ed3cf8603feeaebe9d6127aa85d3d384b559d312b79d7a1d",
    "results/r39.json": "c42d3b1f917da5e74b6efdf4ab4519ae913294de75b07ee6c5245195aa1df264",
    "results/r40.json": "cb813e5d28f283b96068904c9424fc387525a50ab7efd9e0ddc7ee42bc850d00",
    "results/r41.json": "c703b7edf114baeceba1fbee7f2e23051f81464d9f63153c367f544d44927559",
    "results/r42.json": "1ade7ff8db36a4f56b109b0aa437bae348905bd7cf7871ccbf08234bc3d5911c",
    "results/r43.json": "8e8001387434a1e6fa804547313c90c8cdfbfbabd951bc334f3c52070a0d09bf",
    "results/r44.json": "7ddc30a93fa2a20c37eef208a4f354a69c426fb939cffa3ac9e63155e990f71e",
    "results/r45.json": "02881c32da0dc1ff3c3ceff5e01aef1f619c92fa21323660005ee2a60214000b",
    "results/r46.json": "b194698f268fefee049693673fc1a2ec5d5371bb313468ad4be2aae83d7d7525",
    "results/r47.json": "9aa77077e2fe7a92c0d039c95ee393eb12685d91b23d14efcab793c7efe29ea1",
    "results/r48.json": "36966c550c7508cf75f278d301d806fac9a11113ceb5394f55e7c25121d378af",
    "results/r49.json": "01571f70acfb8fae525162dfeee350c5b9f1b7b270dec24105b2c3c067f59f8d",
    "results/r50.json": "c96998e82dfb38514c232b25d0ed4e6f4e96d15cc04ed44505cadb5d34815d82",
    "results/r51.json": "8242b96afe5b2663dbbb5620a69f7c563667dff314f00c610253c9be54e12029",
    "results/r52.json": "bfb22d30e6a3abd987ddbbb1ffb4ef3997cdc67422ae6c75c148307b3d6e24e6",
    "results/r53.json": "96e5d6a8ce4885d7dc9dc9cdde6129662a99e013e591b8ff6701d96c27c0fcb5",
    "results/r54.json": "beb4d6e7692ce0c2d1143520f9830773ce04ac7de1f6a95ee3bca9e00fc7e48c",
    "results/r56.json": "306d0921d7a533bcf65c3206cd4580aacdfc91d187ebe64fff94f3d8b3d1d63d",
    "results/r57.json": "c9da3d14ea6918232ef766b3baa94e5b309b235f526aa27c2e677fa85107a961",
    "results/r58.json": "36a991f9965ef01256965cae9c75dff4b3d21a6e8423198b6459eb19f8c35e27",
    "results/r59.json": "674cf655b33f32637b222c2974c8cee8a57d21d47dfec135f513d6e6232d04dd",
    "results/r60.json": "329bc602090ed7dccfeaffe8df04bb8b3ebf2b95e24133d98562d49cc005591a",
}
RESULT_FILES = sorted(path for path in SOURCE_SHA256 if path.startswith("results/"))
_PACKAGE_PREFIX = "src/"

_SLOT_MARK = "\x00"
_SLOT_PATTERN = re.compile(f"{_SLOT_MARK}(\\w+){_SLOT_MARK}")
_TRAIT_TRANSCRIPT_STAND_IN = f"Sentinel: {_SLOT_MARK}transcript{_SLOT_MARK}\n\n"
_INTERNAL_TAGGED = re.compile(r"<internal>\s*(.*?)\s*</internal>", re.DOTALL | re.IGNORECASE)
_INTERNAL_BRACKETED = re.compile(r"\[INTERNAL\]\s*(.*?)\s*\[/INTERNAL\]", re.DOTALL)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source(path: str, *, source_dir: Path | None = None, cache_dir: Path | None = None) -> bytes:
    """Return a pinned upstream file, read from a local checkout, the cache, or GitHub, after checking its hash.

    A Windows checkout may have converted line endings to CRLF; such a file is accepted when restoring LF line
    endings gives the pinned bytes.
    """
    expected = SOURCE_SHA256[path]
    cached = cache_dir / REVISION / path if cache_dir else None
    if source_dir is not None:
        data = (Path(source_dir) / path).read_bytes()
    elif cached is not None and cached.is_file():
        data = cached.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL + path, timeout=600) as response:
            data = response.read()
    if _sha256(data) != expected and b"\r\n" in data and _sha256(data.replace(b"\r\n", b"\n")) == expected:
        data = data.replace(b"\r\n", b"\n")
    if _sha256(data) != expected:
        raise ValueError(f"{path} does not match EQ-Bench 4 revision {REVISION} (SHA-256 {_sha256(data)})")
    if cached is not None and source_dir is None and not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    return data


def parse_rubric_lines(text: str) -> dict[str, str]:
    """``name: value`` lines of a rubric file, in order, skipping blanks and ``#`` comments."""
    entries: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and ":" in line:
            name, value = line.split(":", 1)
            entries[name.strip()] = value.strip()
    return entries


def split_prompt(rendered: str) -> dict[str, list[str]]:
    """Split a prompt rendered with slot markers into ``{"literals": [...], "slots": [...]}``."""
    if rendered.count(_SLOT_MARK) % 2:
        raise ValueError("a rendered prompt holds an unpaired slot marker")
    pieces = _SLOT_PATTERN.split(rendered)
    return {"literals": pieces[0::2], "slots": pieces[1::2]}


def _slot(name: str) -> str:
    return f"{_SLOT_MARK}{name}{_SLOT_MARK}"


class _BlankForMissing(dict):
    def __missing__(self, key: str) -> str:
        return ""


@contextmanager
def upstream_prompt_module(files: Mapping[str, bytes]) -> Iterator[ModuleType]:
    """Import the downloaded upstream prompt builder from a temporary package.

    ``eqbench.core.__init__`` is replaced by an empty module so that only the prompt builder and its schema imports
    load; nothing performs network or file access beyond the package's own prompt and rubric files.
    """
    saved = {name: module for name, module in sys.modules.items() if name == "eqbench" or name.startswith("eqbench.")}
    for name in saved:
        del sys.modules[name]
    with tempfile.TemporaryDirectory() as root:
        for path, data in files.items():
            if path.startswith(_PACKAGE_PREFIX + "eqbench/"):
                target = Path(root) / path[len(_PACKAGE_PREFIX) :]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        (Path(root) / "eqbench" / "core" / "__init__.py").write_text("")
        sys.path.insert(0, root)
        try:
            yield importlib.import_module("eqbench.core.prompts")
        finally:
            sys.path.remove(root)
            for name in [name for name in sys.modules if name == "eqbench" or name.startswith("eqbench.")]:
                del sys.modules[name]
            sys.modules.update(saved)


class ScenarioRenderer:
    """Renders one scenario's prompts with the upstream prompt builder and the pairwise judge template."""

    def __init__(self, prompts: ModuleType, files: Mapping[str, bytes]) -> None:
        self.prompts = prompts
        self.pairwise_template = files["src/eqbench/prompts/pairwise_judge_prompt.txt"].decode()
        self.dimension_descriptions = parse_rubric_lines(files["src/eqbench/rubrics/pairwise_dimensions.txt"].decode())
        weights = parse_rubric_lines(files["src/eqbench/rubrics/pairwise_dimensions_weights.txt"].decode())
        self.dimension_weights = {name: float(weights.get(name, 1.0)) for name in self.dimension_descriptions}

    def persona_system_prompt(self, doc: dict[str, Any]) -> str:
        return self.prompts.build_persona_system_prompt(doc, num_turns=NUM_TURNS)

    def evaluated_system_prompts(self, doc: dict[str, Any]) -> list[str]:
        return [
            self.prompts.build_evaluated_system_prompt(doc, current_turn=turn, num_turns=NUM_TURNS)
            for turn in range(1, NUM_TURNS + 1)
        ]

    def opening_message(self, doc: dict[str, Any]) -> str:
        presenting = self.prompts._get_respondent_presenting_description(doc)
        return f'[Presenting issue (what begins this conversation): "{presenting}"]'

    def assistant_label(self, doc: dict[str, Any]) -> str:
        return self.prompts._format_relationship_label(doc.get("assistant_relationship", "therapist"))

    def pairwise_prompt(self, doc: dict[str, Any], persona_system_prompt: str) -> dict[str, list[str]]:
        """The pairwise judge prompt, with the respondents' codes, transcripts and final states as slots."""
        prompts = self.prompts
        demographics, background, personality = prompts._extract_modifier_sections(doc.get("modifiers", []))
        sensitivities, preferences = prompts._extract_sensitivities_and_preferences(doc)
        core_issue = doc.get("core_issue") or {}
        if not core_issue:
            raise ValueError(f"scenario {doc.get('id')} has no core issue")
        adjacent = doc.get("core_adjacent_attributes", [])
        persona = doc.get("persona", {})
        trust_style = persona.get("trust_style", {})
        defense_style = persona.get("defense_style", {})
        relationship = doc.get("assistant_relationship", "supportive helper")
        values = {
            "assistant_relationship": relationship,
            "relationship_context": prompts._format_chatbot_relationship_context(relationship, for_judge=True),
            "presenting_issue": prompts._get_respondent_presenting_description(doc),
            "demographics_str": demographics,
            "background_str": background,
            "personality_str": personality,
            "core_issue_str": f"Label: {core_issue.get('label', 'Unknown')}\n"
            f"Description: {core_issue.get('description', '')}",
            "hand_authored_judge_context": prompts._format_hand_authored_judge_context(doc, persona_system_prompt),
            "adjacent_str": "\n".join(f"- {item.get('label', '')}: {item.get('content', '')}" for item in adjacent)
            or "(None)",
            "sensitivities_str": sensitivities,
            "preferences_str": preferences,
            "trust_style_name": trust_style.get("name", "Unknown"),
            "trust_style_description": trust_style.get("description", "(Not specified)"),
            "defense_style_name": defense_style.get("name", "Unknown"),
            "defense_style_description": defense_style.get("description", "(Not specified)"),
            "initial_state_str": prompts._build_initial_state_str(
                doc.get("initial_state", {}), prompts._state_exclude_fields(doc)
            ),
            "code_a": _slot("first_code"),
            "code_b": _slot("second_code"),
            "transcript_a": _slot("first_transcript"),
            "transcript_b": _slot("second_transcript"),
            "persona_a_final_state": _slot("first_final_state"),
            "persona_b_final_state": _slot("second_final_state"),
            "pairwise_dimensions": "\n\n".join(
                f"{number}. **{name.replace('_', ' ').title()}**: {description}"
                for number, (name, description) in enumerate(self.dimension_descriptions.items(), 1)
            ),
            "pairwise_dimensions_json": ",\n".join(
                f'  "{name}": "winner_code+disparity or tie"' for name in self.dimension_descriptions
            ),
        }
        return split_prompt(self.pairwise_template.format_map(_BlankForMissing(values)))

    def trait_prompt(self, doc: dict[str, Any], persona_system_prompt: str) -> dict[str, list[str]]:
        """The trait (pointwise) judge prompt for a run without persona state, with the transcript as its slot."""
        rendered = self.prompts.build_performance_judge_prompt(
            scenario=doc,
            transcript=[{"role": "sentinel", "content": _slot("transcript")}],
            final_state={},
            persona_system_prompt=persona_system_prompt,
            include_initial_state=True,
        )
        if rendered.count(_TRAIT_TRANSCRIPT_STAND_IN) != 1:
            raise ValueError(f"could not place the transcript in the trait prompt of scenario {doc.get('id')}")
        return split_prompt(rendered.replace(_TRAIT_TRANSCRIPT_STAND_IN, _slot("transcript")))


def final_state_text(state_history: list[dict[str, Any]]) -> str:
    """How the pairwise judge sees a transcript's final persona state."""
    if not state_history or not state_history[-1]:
        return NO_FINAL_STATE
    lines = []
    for key, value in state_history[-1].items():
        shown = f"{value:.1f}" if isinstance(value, float) else f"{value}"
        lines.append(f"- {key.replace('_', ' ').title()}: {shown}")
    return "\n".join(lines)


def reference_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, str | None]]:
    """A published transcript reduced to what the pairwise judge reads."""
    reduced: list[dict[str, str | None]] = []
    for message in transcript:
        entry: dict[str, str | None] = {"role": message["role"], "content": message.get("content") or ""}
        if message["role"] == "user":
            raw = message.get("raw_response") or ""
            match = _INTERNAL_TAGGED.search(raw) or _INTERNAL_BRACKETED.search(raw)
            entry["internal"] = match.group(1).strip() if match else None
        reduced.append(entry)
    return reduced


class ReferenceRuns:
    """Published transcripts by model and scenario, and the published comparison graph."""

    def __init__(self) -> None:
        self.transcripts: dict[str, dict[str, dict[str, Any]]] = {}
        self.comparisons: list[dict[str, Any]] = []
        self._signatures: set[tuple[str, str, str, str]] = set()

    def add_run_file(self, data: dict[str, Any]) -> None:
        for run in data.values():
            if not isinstance(run, dict) or "__meta__" not in run:
                continue
            model = run["__meta__"]["evaluated_model"]
            for key, conversations in run.items():
                if key.startswith("__") or not isinstance(conversations, list):
                    continue
                for conversation in conversations:
                    if conversation.get("completed") and conversation.get("convo_index", 0) == 0:
                        self.transcripts.setdefault(model, {})[conversation["scenario_id"]] = {
                            "transcript": reference_transcript(conversation["transcript"]),
                            "final_state": final_state_text(conversation.get("state_history") or []),
                            "persona_system_prompt": (conversation.get("prompts") or {}).get("persona_system_prompt"),
                            "doc": conversation.get("doc"),
                        }
            for comparison in run.get("__pairwise__", []):
                self._add_comparison(comparison)

    def _add_comparison(self, comparison: dict[str, Any]) -> None:
        if "error" in comparison:
            return
        pair = comparison.get("pair", {})
        model_a, model_b = pair.get("model_a"), pair.get("model_b")
        fraction = comparison.get("aggregated", {}).get("fraction_for_model_a")
        if not model_a or not model_b or model_a == model_b or fraction is None:
            return
        signature = (*sorted((model_a, model_b)), str(comparison["scenario_id"]), str(comparison["iteration_index"]))
        if signature in self._signatures:
            return
        self._signatures.add(signature)
        outcome = canonical(Outcome(model_a, model_b, min(max(float(fraction), 0.0), 1.0)))
        self.comparisons.append(
            {
                "model_a": outcome.model_a,
                "model_b": outcome.model_b,
                "fraction_a": outcome.fraction_a,
                "scenario_id": str(comparison["scenario_id"]),
            }
        )

    def ladder(self) -> list[str]:
        """Published models from the highest to the lowest fitted rating."""
        models = sorted(self.transcripts)
        outcomes = [Outcome(item["model_a"], item["model_b"], item["fraction_a"]) for item in self.comparisons]
        ratings = fit_ratings(models, outcomes)
        return sorted(models, key=lambda model: (-ratings[model], model))


def select_opponents(ladder: list[str], available: set[str], scenario_index: int, count: int) -> list[str]:
    """``count`` opponents spread evenly down the ladder, rotating with the scenario so every model is used."""
    eligible = [model for model in ladder if model in available]
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} published transcripts for scenario {scenario_index}, need {count}")
    stride = len(eligible) / count
    return [eligible[(scenario_index + int(step * stride)) % len(eligible)] for step in range(count)]


def build_rows(
    docs: list[dict[str, Any]],
    renderer: ScenarioRenderer,
    references: ReferenceRuns,
    *,
    opponents_per_scenario: int,
) -> list[dict[str, Any]]:
    ladder = references.ladder()
    rows = []
    for scenario_index, doc in enumerate(docs):
        scenario_id = doc["id"]
        persona_prompt = renderer.persona_system_prompt(doc)
        evaluated_prompts = renderer.evaluated_system_prompts(doc)
        available = {model for model, by_scenario in references.transcripts.items() if scenario_id in by_scenario}
        opponents = select_opponents(ladder, available, scenario_index, opponents_per_scenario)
        opening = renderer.opening_message(doc)
        rows.append(
            {
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": evaluated_prompts[0]},
                        {"role": "user", "content": opening},
                    ],
                    "temperature": POLICY_TEMPERATURE,
                    "max_output_tokens": POLICY_MAX_OUTPUT_TOKENS,
                },
                "scenario_id": scenario_id,
                "scenario_index": scenario_index,
                "source_type": doc.get("source_type", "generated"),
                "num_turns": NUM_TURNS,
                "evaluated_system_prompts": evaluated_prompts,
                "opening_message": opening,
                "persona_system_prompt": persona_prompt,
                "pairwise_prompt": renderer.pairwise_prompt(doc, persona_prompt),
                "pairwise_dimensions": renderer.dimension_weights,
                "trait_prompt": renderer.trait_prompt(doc, persona_prompt),
                "trait_assistant_label": renderer.assistant_label(doc),
                "opponents": [
                    {
                        "model": model,
                        "comparison_index": scenario_index * opponents_per_scenario + position,
                        "transcript": references.transcripts[model][scenario_id]["transcript"],
                        "final_state": references.transcripts[model][scenario_id]["final_state"],
                    }
                    for position, model in enumerate(opponents)
                ],
            }
        )
    return rows


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_eqbench4(
    output_path: Path,
    reference_comparisons_path: Path,
    *,
    source_dir: str | Path | None = None,
    cache_dir: Path | None = None,
    opponents_per_scenario: int = DEFAULT_OPPONENTS_PER_SCENARIO,
    log: Callable[[str], None] = print,
) -> Path:
    """Write the benchmark rows and the published comparison graph; return the rows' path."""
    if opponents_per_scenario < 1:
        raise ValueError("opponents_per_scenario must be at least 1")
    source = Path(source_dir) if source_dir is not None else None
    files = {
        path: load_source(path, source_dir=source, cache_dir=cache_dir)
        for path in SOURCE_SHA256
        if not path.startswith("results/")
    }
    docs = json.loads(files[DATASET_FILE])["docs"]
    references = ReferenceRuns()
    for path in RESULT_FILES:
        log(f"Reading {path}")
        references.add_run_file(json.loads(load_source(path, source_dir=source, cache_dir=cache_dir)))
    with upstream_prompt_module(files) as prompts:
        rows = build_rows(
            docs, ScenarioRenderer(prompts, files), references, opponents_per_scenario=opponents_per_scenario
        )
    _write_jsonl(reference_comparisons_path, references.comparisons)
    _write_jsonl(output_path, rows)
    log(
        f"Wrote {len(rows)} scenarios against {len(references.transcripts)} published models and "
        f"{len(references.comparisons)} published comparisons"
    )
    return output_path
