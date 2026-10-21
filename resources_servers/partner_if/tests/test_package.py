# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task packages: the row contract, execution profiles and checksummed files. Synthetic packages only."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import BaseModel

from nemo_gym import PARENT_DIR
from resources_servers.partner_if.package import (
    CHECKSUMS,
    RepoPath,
    partition,
    read_jsonl,
    resolve_repo_path,
    select_rows,
    validate_rows,
    verify_package,
    write_jsonl,
)
from resources_servers.partner_if.tests.test_app import contract
from resources_servers.partner_if.tests.test_shared_adapter import shared_request


SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": "Search.",
    "parameters": {"type": "object"},
    "strict": False,
}
SWE_TASK = {
    "instance_id": "synthetic__calculator-1",
    "repo": "synthetic/calculator",
    "workdir": "/testbed",
    "image_url": "registry.invalid/synthetic-swe:1",
    "patch": "diff --git a/calc.py b/calc.py\n+FIXED = True\n",
    "f2p_script": "pytest tests/test_calc.py",
    "FAIL_TO_PASS": ["tests/test_calc.py::test_add"],
    "PASS_TO_PASS": [],
}
ACTOR_PROFILE = {
    "system_prompt": "You are a careful coding agent.\nQuote ${HOME} literally and never expand ${oc.env:HOME}.\n",
    "tool_description_additions": {"bash": "Prefer read-only commands first.", "edit": "Keep edits minimal."},
}


def row(task_id="C1", **kwargs):
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "Say HELLO.\n"}]},
        "verifier_metadata": {"partner_if": contract(task_id, data_kind="reported", **kwargs)},
    }


def live_row(task_id, *, task_name=None, tools=()):
    data = row(task_id)
    data["verifier_metadata"]["partner_if"]["evidence_mode"] = "live_agent"
    if tools:
        data["responses_create_params"]["tools"] = list(tools)
    if task_name:
        data["task_name"] = task_name
    return data


def native_row(identifier="native"):
    body = shared_request()
    c = body.verifier_metadata.partner_if
    c.task_id, c.data_kind, c.evidence_mode = identifier, "reconstruction", "live_agent"
    c.rubrics = [c.rubrics[0]]
    completion = c.rubrics[0].model_copy(deep=True)
    completion.id, completion.dimension = "completion", "completion"
    completion.verifier.params["literal"] = "BYE"
    c.rubrics.append(completion)
    return {
        "responses_create_params": body.responses_create_params.model_dump(),
        "verifier_metadata": body.verifier_metadata.model_dump(),
        "task_name": identifier,
    }, c


def package_rows():
    """(profile, row) in package order. Profiles interleave so selection order is observable."""
    return [
        ("text", row("text-1")),
        ("swe", live_row("swe-1", task_name="swe-1")),
        ("search_tools", live_row("search-1", tools=[SEARCH_TOOL])),
        ("text", row("text-2")),
        ("build_tools", live_row("build-1", tools=[SEARCH_TOOL])),
        ("markdown", live_row("markdown-1", task_name="markdown-1")),
    ]


def task_ids(rows):
    return [data["verifier_metadata"]["partner_if"]["task_id"] for data in rows]


def record_checksums(root: Path) -> None:
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / CHECKSUMS
    }
    (root / CHECKSUMS).write_text(json.dumps(hashes, indent=2))


def build_package(root: Path, entries: list[tuple[str, dict]] | None = None) -> Path:
    """A checksummed synthetic package, built from (profile, row) entries."""
    entries = package_rows() if entries is None else entries
    root.mkdir(parents=True)
    profiles = {}
    for profile, data in entries:
        profiles.setdefault(profile, []).append(data["verifier_metadata"]["partner_if"]["task_id"])
    write_jsonl(root / "expected.jsonl", [data for _, data in entries])
    (root / "profiles.json").write_text(json.dumps(profiles))
    for profile in profiles:
        write_jsonl(root / f"{profile}.jsonl", [data for name, data in entries if name == profile])
    (root / "swe-tasks").mkdir()
    for profile, data in entries:
        if profile == "swe":
            (root / "swe-tasks" / f"{data['task_name']}.json").write_text(json.dumps(SWE_TASK))
    (root / "swe-actor-profile.json").write_text(json.dumps(ACTOR_PROFILE))
    record_checksums(root)
    return root


@pytest.mark.parametrize("rows", [[], [row(), row()]])
def test_bad_membership_rejected(rows):
    with pytest.raises(ValueError):
        validate_rows(rows)


def test_synthetic_rows_are_not_benchmark_members():
    data = row()
    data["verifier_metadata"]["partner_if"]["data_kind"] = "synthetic_smoke"
    with pytest.raises(ValueError, match="synthetic"):
        validate_rows([data])
    assert validate_rows([data], allow_synthetic=True) == [data]


def test_validation_returns_rows_unchanged():
    rows = [row("C1"), native_row("C2")[0]]
    before = deepcopy(rows)
    assert validate_rows(rows) is rows
    assert rows == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["responses_create_params"].update(tools=[{"type": "function", "name": "run", "parameters": {}}]),
        lambda r: r.update(agent_ref={"name": "hardwired"}),
        lambda r: r["responses_create_params"].update(
            input=[{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]
        ),
        lambda r: r.update(task_name="text-tasks-have-no-native-package"),
    ],
)
def test_unsupported_inputs_not_silently_modified(mutation):
    data = row()
    mutation(data)
    with pytest.raises(ValueError):
        validate_rows([data])


@pytest.mark.parametrize("hidden", [{"conversation": "conv_123"}, {"prompt": {"id": "pmpt_123"}}])
def test_rows_with_hidden_context_are_rejected(hidden):
    source = row()
    source["responses_create_params"].update(hidden)
    with pytest.raises(ValueError):
        validate_rows([source])


def test_native_rows_are_accepted_but_not_arbitrary_harness_overrides():
    data, _ = native_row()
    assert validate_rows([data]) == [data]
    bad = {**data, "agent_ref": {"name": "arbitrary"}}
    with pytest.raises(ValueError):
        validate_rows([bad])


@pytest.mark.parametrize("name", ["../verify", "/grade", "close_session?admin=1"])
def test_live_tools_reject_unsafe_endpoint_names(name):
    data = live_row("live-1", tools=[SEARCH_TOOL | {"name": name}])
    with pytest.raises(ValueError, match="safe.*function"):
        validate_rows([data])


def test_live_tools_must_have_unique_names():
    with pytest.raises(ValueError, match="unique"):
        validate_rows([live_row("live-1", tools=[SEARCH_TOOL, SEARCH_TOOL])])


def test_suite_partitions_expected_tasks_once_without_rewriting_inputs():
    first, _ = native_row("A")
    second, _ = native_row("B")
    profiles = {"harbor_one": ["A"], "opencode_one": ["B"]}
    assert partition([first, second], profiles) == {"harbor_one": [first], "opencode_one": [second]}
    for bad in ({"one": ["A"]}, {"one": ["A", "B"], "two": ["A"]}, {"one": ["A", "B", "C"]}):
        with pytest.raises(ValueError):
            partition([first, second], bad)


def test_reserved_profile_name_cannot_overwrite_expected_membership():
    data, _ = native_row()
    with pytest.raises(ValueError, match="reserved"):
        partition([data], {"expected": ["native"]})


@pytest.mark.parametrize("profiles", [{"../escape": ["A", "B"]}, {"one": ["A", "B"], "two": []}])
def test_profiles_require_safe_names_and_members(profiles):
    rows = [native_row("A")[0], native_row("B")[0]]
    with pytest.raises(ValueError, match="safe names"):
        partition(rows, profiles)


def test_verified_package_returns_its_resolved_root(tmp_path):
    root = build_package(tmp_path / "package")
    assert verify_package(root) == root.resolve()
    assert verify_package(root / "swe-tasks" / "..") == root.resolve()


@pytest.mark.parametrize(
    "change,error",
    [
        ("changed", "checksum mismatch: text.jsonl"),
        ("missing", "checksum mismatch: profiles.json"),
        ("unrecorded", "unrecorded or missing"),
        ("unrecorded_nested", "unrecorded or missing"),
        ("link_inside", "checksum mismatch: swe-actor-profile.json"),
        ("link_outside", "Unsafe package path: swe-actor-profile.json"),
    ],
)
def test_package_rejects_changed_missing_unrecorded_or_linked_files(tmp_path, change, error):
    root = build_package(tmp_path / "package")
    if change == "changed":
        (root / "text.jsonl").write_text((root / "text.jsonl").read_text() + "\n")
    elif change == "missing":
        (root / "profiles.json").unlink()
    elif change == "unrecorded":
        (root / "notes.md").write_text("Not part of the package.")
    elif change == "unrecorded_nested":
        (root / "swe-tasks" / "extra.json").write_text("{}")
    else:
        # The same bytes behind a link are still rejected: a link can change after verification.
        target = (root / "swe-tasks" if change == "link_inside" else tmp_path) / "profile-copy.json"
        target.write_bytes((root / "swe-actor-profile.json").read_bytes())
        (root / "swe-actor-profile.json").unlink()
        (root / "swe-actor-profile.json").symlink_to(target)
    with pytest.raises(ValueError, match=error):
        verify_package(root)


@pytest.mark.parametrize("entry", ["../outside.txt", "/etc/hosts", "swe-tasks/../../outside.txt", "linked/doc.md"])
def test_package_manifest_cannot_reach_files_outside_the_package(tmp_path, entry):
    root = build_package(tmp_path / "package")
    (tmp_path / "outside.txt").write_text("outside")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "doc.md").write_text("outside")
    (root / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    hashes = json.loads((root / CHECKSUMS).read_text())
    hashes[entry] = hashlib.sha256(b"outside").hexdigest()
    (root / CHECKSUMS).write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="Unsafe package path"):
        verify_package(root)


@pytest.mark.parametrize("manifest", [{}, []])
def test_package_requires_a_nonempty_manifest(tmp_path, manifest):
    root = tmp_path / "package"
    root.mkdir()
    (root / CHECKSUMS).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonempty"):
        verify_package(root)


def test_selection_keeps_package_order_and_unchanged_rows(tmp_path):
    root = build_package(tmp_path / "package")
    entries = package_rows()
    selected = select_rows(root, ("text", "search_tools", "build_tools"))
    assert selected == [data for profile, data in entries if profile in {"text", "search_tools", "build_tools"}]
    assert task_ids(selected) == ["text-1", "search-1", "text-2", "build-1"]
    assert task_ids(select_rows(root, ("markdown", "swe"))) == ["swe-1", "markdown-1"]


def test_task_ids_from_other_profiles_are_skipped_and_unknown_ids_raise(tmp_path):
    root = build_package(tmp_path / "package")
    # One selection serves both benchmark configs: each keeps the IDs of its own profiles.
    assert task_ids(select_rows(root, ("text", "search_tools"), ["swe-1", "text-2"])) == ["text-2"]
    assert task_ids(select_rows(root, ("swe", "markdown"), ["swe-1", "text-2"])) == ["swe-1"]
    with pytest.raises(ValueError, match="Unknown task IDs: not-a-task"):
        select_rows(root, ("text",), ["text-1", "not-a-task"])
    with pytest.raises(ValueError, match="No selected tasks"):
        select_rows(root, ("text",), ["swe-1"])


@pytest.mark.parametrize("selection", [[], ["text-1", "text-1"]])
def test_task_selection_must_be_nonempty_and_unique(tmp_path, selection):
    root = build_package(tmp_path / "package")
    with pytest.raises(ValueError, match="nonempty, unique"):
        select_rows(root, ("text",), selection)


def test_unknown_profiles_are_rejected(tmp_path):
    root = build_package(tmp_path / "package")
    with pytest.raises(ValueError, match="Unknown execution profiles: hf"):
        select_rows(root, ("text", "hf"))


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda rows: rows[1:],
        lambda rows: [rows[0] | {"responses_create_params": {"input": [{"role": "user", "content": "Changed."}]}}]
        + rows[1:],
    ],
)
def test_profile_files_must_repeat_expected_membership(tmp_path, rewrite):
    root = build_package(tmp_path / "package")
    write_jsonl(root / "text.jsonl", rewrite(read_jsonl(root / "text.jsonl")))
    with pytest.raises(ValueError, match="Profile differs from expected membership: text"):
        select_rows(root, ("search_tools",))


def test_relative_package_paths_resolve_under_a_plugin_root_then_the_gym_root(tmp_path, monkeypatch):
    class Settings(BaseModel):
        path: RepoPath

    # Servers start in their own directories; the result must not depend on the cwd.
    monkeypatch.chdir(tmp_path)
    example = Path("resources_servers/partner_if/data/example.jsonl")
    assert resolve_repo_path(example) == PARENT_DIR / example
    assert Settings(path=str(example)).path == PARENT_DIR / example
    assert resolve_repo_path(tmp_path) == tmp_path
    # A benchmark kept outside Gym is found under its root (`--search-dir` / NEMO_GYM_EXTRA_ROOTS).
    plugin = tmp_path / "plugin"
    package = Path("benchmarks/synthetic/data/package")
    (plugin / package).mkdir(parents=True)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin))
    assert resolve_repo_path(package) == plugin / package
    assert Settings(path=str(package / "task.json")).path == plugin / package / "task.json"


def test_public_examples_are_five_synthetic_nonbenchmark_rows():
    rows = read_jsonl(PARENT_DIR / "resources_servers/partner_if/data/example.jsonl")
    validate_rows(rows, allow_synthetic=True)
    assert len(rows) == 5
    assert all(r["verifier_metadata"]["partner_if"]["data_kind"] == "synthetic_smoke" for r in rows)
    with pytest.raises(ValueError, match="synthetic"):
        validate_rows(rows)
