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
"""Gym's mirror of Harbor's ``task.toml`` and its copy of Harbor's hashing follow Harbor at a recorded revision.

``fixtures/harbor_task_format.json`` is Harbor's ``TaskConfig`` schema and the hashes of the definitions
``harbor_compat.py`` copies, written by ``scripts/harbor_task_format_snapshot.py`` from a Harbor checkout. These
tests compare Gym with that file, so they need neither the network nor the ``harbor`` package.

When Harbor changes its task format, refresh the snapshot (the script's docstring has the command). Every field
the mirror does not follow then fails here by name. Follow it in ``models.py``, or record it in
:data:`KNOWN_GAPS` (to fix) or :data:`ACCEPTED_DIFFERENCES` (kept on purpose).

The schema says which keys Harbor reads, what each accepts and its default. It does not carry the rules Harbor
checks in validators (for example which ``allowed_hosts`` entries are valid); the script's ``--check`` covers part
of that by parsing Harbor's own example tasks with the mirror.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.data.tasks.harbor import harbor_compat
from nemo_gym.data.tasks.harbor.build_compose import image_name
from nemo_gym.data.tasks.harbor.models import (
    HarborAgent,
    HarborArtifact,
    HarborCollectHook,
    HarborEnvironment,
    HarborHealthcheck,
    HarborMCPServer,
    HarborSettings,
    HarborSolution,
    HarborTaskConfig,
    HarborTaskStep,
    HarborTaskStepAgent,
    HarborTaskStepVerifier,
    HarborVerifier,
)


SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "harbor_task_format.json").read_text())
HARBOR_SCHEMA = SNAPSHOT["task_config_schema"]
MIRROR_SCHEMA = HarborTaskConfig.model_json_schema()

SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "harbor_task_format_snapshot.py"
_spec = importlib.util.spec_from_file_location("harbor_task_format_snapshot", SCRIPT)
snapshot_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snapshot_script)

# Each Harbor model with the mirror class that reads it. A step's agent and verifier are Harbor's `AgentConfig`
# and `VerifierConfig`; the mirror has a class of its own for each.
MIRRORED_MODELS: list[tuple[str, type[HarborSettings]]] = [
    ("TaskConfig", HarborTaskConfig),
    ("EnvironmentConfig", HarborEnvironment),
    ("AgentConfig", HarborAgent),
    ("AgentConfig", HarborTaskStepAgent),
    ("VerifierConfig", HarborVerifier),
    ("VerifierConfig", HarborTaskStepVerifier),
    ("VerifierCollectConfig", HarborCollectHook),
    ("StepConfig", HarborTaskStep),
    ("ArtifactConfig", HarborArtifact),
    ("HealthcheckConfig", HarborHealthcheck),
    ("MCPServerConfig", HarborMCPServer),
    ("SolutionConfig", HarborSolution),
]
# Harbor models the mirror has no class for: `[task]` is kept as a plain table (`PackageInfo` and its `Author`),
# and a `tpu` table is refused (`TpuSpec`).
# Their fields are pinned so that a change inside them is looked at too.
UNMIRRORED_MODELS = {
    "Author": {"email", "name"},
    "PackageInfo": {"authors", "description", "keywords", "name", "version"},
    "TpuSpec": {"topology", "type"},
}

# Fields the mirror does not follow yet: how each differs today, and why it matters. Each is an expected failure,
# and the failure is strict: once the mirror follows the field its case passes, which fails the run until the entry
# is removed. The difference is pinned as well, so that a further change to the field in Harbor is not hidden.
KNOWN_GAPS = {
    "EnvironmentConfig.cpus in HarborEnvironment": (
        ["Harbor accepts integer, number, the mirror integer"],
        "Harbor accepts fractional `cpus` (since 0.25.0); the mirror reads an integer and refuses the task",
    ),
    "MCPServerConfig.headers in HarborMCPServer": (
        ["not in the mirror"],
        "Harbor forwards `[[environment.mcp_servers]].headers` to the agent (since 0.25.0); the mirror ignores the "
        "key with an unknown-key warning",
    ),
}
# Definitions `harbor_compat.py` copies that Harbor has changed since the copy was taken.
KNOWN_VENDORED_GAPS = {
    "environment_content_hash": (
        "Harbor hashes files in chunks and widens the length prefix when a file is 4 GiB or larger (since 0.24.0); "
        "the copy is Harbor 0.22.0's, which reads each file whole and raises OverflowError on such a file. Both "
        "give the same hash for smaller files"
    ),
}

# Differences the mirror keeps on purpose, pinned so that a change on either side is noticed.
ACCEPTED_DIFFERENCES = {
    # `[task]` is not validated: Gym names a task after its folder, not after `[task].name`.
    "TaskConfig.task in HarborTaskConfig": [
        "Harbor accepts PackageInfo, the mirror object<any>",
        "Harbor's default is None, the mirror has none",
    ],
    # The mirror resolves at load the defaults Harbor resolves at run time: no steps, and the `mean` roll-up.
    "TaskConfig.steps in HarborTaskConfig": ["Harbor's default is None, the mirror has none"],
    "TaskConfig.multi_step_reward_strategy in HarborTaskConfig": ["Harbor's default is None, the mirror's 'mean'"],
    # A bare string is read as the artifact's `source` by a validator, which the mirror's schema does not show.
    "TaskConfig.artifacts in HarborTaskConfig": [
        "Harbor accepts array<ArtifactConfig|string>, the mirror array<ArtifactConfig>"
    ],
    "StepConfig.artifacts in HarborTaskStep": [
        "Harbor accepts array<ArtifactConfig|string>, the mirror array<ArtifactConfig>"
    ],
    # A `tpu` table is refused by name ("TPU environments are not supported yet"), not ignored.
    "EnvironmentConfig.tpu in HarborEnvironment": ["not in the mirror"],
    # Harbor leaves an unset agent timeout to the job; a Gym episode always has one.
    "AgentConfig.timeout_sec in HarborAgent": [
        "Harbor's default is None, the mirror's 28800",
        "Harbor has no bounds, the mirror {'exclusiveMinimum': 0}",
    ],
    # Harbor's step verifier is a `VerifierConfig` with the 600 s default; the mirror leaves it unset and resolves
    # the same 600 s in `HarborTaskConfig.step_verifier_timeout_sec`.
    "VerifierConfig.timeout_sec in HarborTaskStepVerifier": [
        "Harbor's default is 600.0, the mirror's None",
        "Harbor has no bounds, the mirror {'exclusiveMinimum': 0}",
    ],
    # The mirror refuses a budget of zero or less at load; Harbor's model has no bound.
    "EnvironmentConfig.build_timeout_sec in HarborEnvironment": [
        "Harbor has no bounds, the mirror {'exclusiveMinimum': 0}"
    ],
    "AgentConfig.timeout_sec in HarborTaskStepAgent": ["Harbor has no bounds, the mirror {'exclusiveMinimum': 0}"],
    "VerifierConfig.timeout_sec in HarborVerifier": ["Harbor has no bounds, the mirror {'exclusiveMinimum': 0}"],
    "VerifierCollectConfig.timeout_sec in HarborCollectHook": [
        "Harbor has no bounds, the mirror {'exclusiveMinimum': 0}"
    ],
    # A Gym extension, set from `dataset.toml` (see `HarborVerifier`).
    "VerifierConfig.inputs in HarborVerifier": ["not in Harbor"],
}

# The schema keywords that say what a value is; `_accepted` and the default comparison read them. Every other
# keyword limits the value (`exclusiveMinimum`, `pattern`, `minItems`, `format`, ...) and is compared as a bound.
_SHAPE_KEYS = {"type", "anyOf", "$ref", "default", "items", "additionalProperties", "enum", "description", "title"}
# What a model's schema says besides its fields, such as `additionalProperties: false` for a model that refuses
# unknown keys.
_MODEL_SHAPE_KEYS = {"type", "properties", "required", "description", "title", "$defs"}
_MIRROR_NAMES = {mirror.__name__: harbor for harbor, mirror in MIRRORED_MODELS}


def _model_schema(schema: dict[str, Any], name: str) -> dict[str, Any]:
    return schema["$defs"].get(name, schema)


def _accepted(prop: dict[str, Any], schema: dict[str, Any], names: dict[str, str]) -> set[str]:
    """What a field accepts, in words both schemas share: JSON types, enum values, and models under Harbor's
    names. ``null`` is left out: TOML cannot write it, so it only says how an unset key is stored."""
    if "$ref" in prop:
        name = prop["$ref"].rsplit("/", 1)[1]
        target = schema["$defs"][name]
        return _accepted(target, schema, names) if "enum" in target else {names.get(name, name)}
    if "anyOf" in prop:
        return set().union(*(_accepted(option, schema, names) for option in prop["anyOf"]))
    if "enum" in prop:
        return {"enum(" + "|".join(sorted(prop["enum"])) + ")"}
    kind = prop.get("type", "any")
    if kind == "array":
        return {"array<" + "|".join(sorted(_accepted(prop.get("items", {}), schema, names))) + ">"}
    if kind == "object":
        values = prop.get("additionalProperties", True)
        return {"object<" + ("any" if values is True else "|".join(sorted(_accepted(values, schema, names)))) + ">"}
    return set() if kind == "null" else {kind}


def _bounds(prop: dict[str, Any]) -> dict[str, Any]:
    """The limits a field's value must meet, whichever of its accepted types states them; a limit on the items of
    an array or the values of a table is named ``items.<limit>``."""
    bounds = {key: value for key, value in prop.items() if key not in _SHAPE_KEYS}
    for option in prop.get("anyOf", []):
        bounds |= _bounds(option)
    for nested in (prop.get("items"), prop.get("additionalProperties")):
        if isinstance(nested, dict):
            bounds |= {f"items.{key}": value for key, value in _bounds(nested).items()}
    return bounds


def _differences() -> dict[str, list[str]]:
    """Every field of every mirrored model, with how the mirror differs from Harbor (nothing when it does not)."""
    found: dict[str, list[str]] = {}
    for harbor_name, mirror in MIRRORED_MODELS:
        harbor = _model_schema(HARBOR_SCHEMA, harbor_name)
        ours = _model_schema(MIRROR_SCHEMA, mirror.__name__)
        harbor_fields, our_fields = harbor.get("properties", {}), ours.get("properties", {})
        for field in sorted(set(harbor_fields) | set(our_fields)):
            differences = found.setdefault(f"{harbor_name}.{field} in {mirror.__name__}", [])
            if field not in harbor_fields:
                differences.append("not in Harbor")
                continue
            if field not in our_fields:
                # A key the mirror renames before validation is read, under its current spelling.
                if field not in mirror.deprecated_keys:
                    differences.append("not in the mirror")
                continue
            theirs, mine = harbor_fields[field], our_fields[field]
            accepted = [
                ", ".join(sorted(_accepted(theirs, HARBOR_SCHEMA, {}))),
                ", ".join(sorted(_accepted(mine, MIRROR_SCHEMA, _MIRROR_NAMES))),
            ]
            if accepted[0] != accepted[1]:
                differences.append(f"Harbor accepts {accepted[0]}, the mirror {accepted[1]}")
            if (field in harbor.get("required", [])) != (field in ours.get("required", [])):
                differences.append("required in one, optional in the other")
            elif theirs.get("default") != mine.get("default") or ("default" in theirs) != ("default" in mine):
                defaults = [repr(prop["default"]) if "default" in prop else None for prop in (theirs, mine)]
                differences.append(
                    f"Harbor's default is {defaults[0]}, the mirror's {defaults[1]}"
                    if defaults[1]
                    else f"Harbor's default is {defaults[0]}, the mirror has none"
                    if defaults[0]
                    else f"Harbor has no default, the mirror's is {defaults[1]}"
                )
            bounds = [_bounds(theirs), _bounds(mine)]
            if bounds[0] != bounds[1]:
                differences.append(f"Harbor has {bounds[0] or 'no bounds'}, the mirror {bounds[1] or 'no bounds'}")
    return found


DIFFERENCES = _differences()


def test_snapshot_names_its_harbor_revision():
    assert SNAPSHOT["source"] == "https://github.com/harbor-framework/harbor"
    assert SNAPSHOT["version"] and len(SNAPSHOT["commit"]) == 40 and SNAPSHOT["revision"]
    # The snapshot is Harbor's whole task model, with a field named like a schema annotation kept.
    assert "description" in HARBOR_SCHEMA["$defs"]["PackageInfo"]["properties"]
    assert len(DIFFERENCES) > 70


@pytest.mark.parametrize(
    "field",
    [
        pytest.param(field, marks=pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[field][1]))
        if field in KNOWN_GAPS
        else field
        for field in DIFFERENCES
    ],
)
def test_mirror_reads_the_field_as_harbor_does(field):
    assert DIFFERENCES[field] == ACCEPTED_DIFFERENCES.get(field, [])


@pytest.mark.parametrize("field", sorted(KNOWN_GAPS))
def test_a_known_gap_differs_only_as_recorded(field):
    assert DIFFERENCES[field] == KNOWN_GAPS[field][0]


def test_recorded_gaps_and_differences_name_fields_that_exist():
    assert set(KNOWN_GAPS) | set(ACCEPTED_DIFFERENCES) <= set(DIFFERENCES)
    assert not set(KNOWN_GAPS) & set(ACCEPTED_DIFFERENCES)


def test_every_harbor_model_is_mirrored_or_named():
    harbor_models = {name for name, model in HARBOR_SCHEMA["$defs"].items() if "enum" not in model} | {"TaskConfig"}
    assert harbor_models == {name for name, _ in MIRRORED_MODELS} | set(UNMIRRORED_MODELS)
    for name, fields in UNMIRRORED_MODELS.items():
        assert set(HARBOR_SCHEMA["$defs"][name]["properties"]) == fields, name


@pytest.mark.parametrize(
    ("harbor_name", "mirror"), MIRRORED_MODELS, ids=lambda value: getattr(value, "__name__", value)
)
def test_mirror_model_has_harbors_model_level_rules(harbor_name, mirror):
    """Harbor ignores unknown keys, and so does the mirror: a model that starts to refuse them changes which tasks
    load."""
    harbor, ours = _model_schema(HARBOR_SCHEMA, harbor_name), _model_schema(MIRROR_SCHEMA, mirror.__name__)
    rules = [{key: value for key, value in model.items() if key not in _MODEL_SHAPE_KEYS} for model in (harbor, ours)]
    assert rules[0] == rules[1]


@pytest.mark.xfail(strict=True, reason=KNOWN_GAPS["EnvironmentConfig.cpus in HarborEnvironment"][1])
def test_fractional_cpus_load():
    assert HarborTaskConfig.model_validate({"environment": {"cpus": 0.5}}).environment.cpus == 0.5


@pytest.mark.xfail(strict=True, reason=KNOWN_GAPS["MCPServerConfig.headers in HarborMCPServer"][1])
def test_mcp_server_headers_are_read():
    data = {"environment": {"mcp_servers": [{"name": "s", "url": "http://s", "headers": {"X-Team": "a"}}]}}
    assert HarborTaskConfig.unknown_keys(data) == []
    assert HarborTaskConfig.model_validate(data).environment.mcp_servers[0].model_dump()["headers"] == {"X-Team": "a"}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=KNOWN_VENDORED_GAPS[name]))
        if name in KNOWN_VENDORED_GAPS
        else name
        for name in sorted(snapshot_script.VENDORED_DEFINITIONS)
    ],
)
def test_vendored_definition_is_harbors(name):
    """``harbor_compat.py`` promises Harbor's bodies unchanged, so that both name the same image for a task."""
    source = Path(harbor_compat.__file__).read_text()
    assert snapshot_script.definition_hash(source, name) == SNAPSHOT["vendored_definitions"][name]


def test_task_image_is_named_as_harbor_names_it():
    """Harbor's Docker environment names a task's image ``hb__<environment id>`` through its sanitiser; an image
    either framework built is found by the other only while both compose that name."""
    assert SNAPSHOT["image_name_expression"] == '_sanitize_docker_image_name(f"hb__{self.environment_id}")'
    assert image_name("0123ABCD", None) == harbor_compat.sanitize_docker_image_name("hb__0123ABCD") == "hb__0123abcd"
