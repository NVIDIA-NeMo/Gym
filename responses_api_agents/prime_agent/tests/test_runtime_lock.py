# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from responses_api_agents.prime_agent import runtime_lock


ARCHIVES = ("prime-agent", "prime-agent-ai", "prime-agent-core", "prime-agent-tui")
TOKEN = "__NEMO_GYM_PRIME_ARCHIVES__"
REGISTRY_SRI = "sha512-" + base64.b64encode(bytes([91]) * 64).decode()


def integrities(seed: int = 1) -> dict[str, str]:
    return {
        name: "sha512-" + base64.b64encode(bytes([seed + index]) * 64).decode() for index, name in enumerate(ARCHIVES)
    }


@pytest.fixture
def templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "runtime"
    directory.mkdir()
    dependency = {"prime-agent": f"file:{TOKEN}/prime-agent.tgz"}
    manifest = {"name": "nemo-gym-prime-runtime", "version": "0.7.0", "private": True, "dependencies": dependency}
    packages = {"": {"name": manifest["name"], "version": manifest["version"], "dependencies": dependency}}
    for name in ARCHIVES:
        packages[f"node_modules/{name}"] = {
            "version": "0.7.0",
            "resolved": f"file:{TOKEN}/{name}.tgz",
            "integrity": "__NEMO_GYM_" + name.upper().replace("-", "_") + "_INTEGRITY__",
        }
    packages["node_modules/registry-example"] = {
        "version": "1.2.3",
        "resolved": "https://registry.npmjs.org/registry-example/-/registry-example-1.2.3.tgz",
        "integrity": REGISTRY_SRI,
    }
    lock = {"name": manifest["name"], "version": manifest["version"], "lockfileVersion": 3, "packages": packages}
    (directory / "package.json").write_text(json.dumps(manifest) + "\n")
    (directory / "package-lock.json").write_text(json.dumps(lock) + "\n")
    (directory / "python-constraints.txt").write_text("runtime-package==1.2.3\n")
    (directory / "python-build-constraints.txt").write_text("build-package==4.5.6\n")
    installer = tmp_path / "install_prime_runtime.sh"
    installer.write_text("#!/bin/sh\nset -eu\n")
    helper = tmp_path / "runtime_lock.py"
    helper.write_text("# synthetic runtime helper recipe, version one\n")
    monkeypatch.setattr(runtime_lock, "RUNTIME_DIR", directory)
    monkeypatch.setattr(runtime_lock, "INSTALLER_PATH", installer)
    monkeypatch.setattr(runtime_lock, "HELPER_PATH", helper)
    return directory


def stage(
    directory: str = "/tmp/session-a", seed: int = 1, identity: str = "pinned-prime-node-python-uv-archive-hashes"
) -> dict[str, bytes]:
    return runtime_lock.stage_runtime_inputs(
        archive_directory=directory, archive_integrities=integrities(seed), runtime_identity=identity
    )


def edit_json(path: Path, mutation: Callable[[dict], object]) -> None:
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value) + "\n")


def test_stage_binds_only_verified_release_archives_and_preserves_registry_integrity(templates: Path) -> None:
    staged = stage()
    assert set(staged) == {
        "runtime-package.json",
        "runtime-package-lock.json",
        "runtime-python-constraints.txt",
        "runtime-python-build-constraints.txt",
        "runtime-profile.txt",
    }
    assert all(isinstance(content, bytes) for content in staged.values())
    manifest = json.loads(staged["runtime-package.json"])
    assert manifest["dependencies"] == {"prime-agent": "file:/tmp/session-a/prime-agent.tgz"}
    locked = json.loads(staged["runtime-package-lock.json"])["packages"]
    for name in ARCHIVES:
        assert locked[f"node_modules/{name}"]["resolved"] == f"file:/tmp/session-a/{name}.tgz"
        assert locked[f"node_modules/{name}"]["integrity"] == integrities()[name]
    assert locked["node_modules/registry-example"] == {
        "version": "1.2.3",
        "resolved": "https://registry.npmjs.org/registry-example/-/registry-example-1.2.3.tgz",
        "integrity": REGISTRY_SRI,
    }
    assert staged["runtime-python-constraints.txt"] == (templates / "python-constraints.txt").read_bytes()
    assert staged["runtime-python-build-constraints.txt"] == (templates / "python-build-constraints.txt").read_bytes()
    assert len(staged["runtime-profile.txt"]) == 65
    assert staged["runtime-profile.txt"].strip()
    assert staged["runtime-profile.txt"] != b""  # The previous installer's empty readiness marker cannot match.


def test_profile_reuses_recipe_across_session_paths_and_localized_archive_byte_changes(templates: Path) -> None:
    first = stage("/tmp/session-a", seed=1)
    second = stage("/tmp/session-b", seed=30)
    assert first["runtime-package-lock.json"] != second["runtime-package-lock.json"]
    assert first["runtime-profile.txt"] == second["runtime-profile.txt"]


@pytest.mark.parametrize(
    "filename",
    [
        "package.json",
        "package-lock.json",
        "python-constraints.txt",
        "python-build-constraints.txt",
        "installer",
        "helper",
    ],
)
def test_profile_invalidates_cached_install_when_recipe_bytes_change(templates: Path, filename: str) -> None:
    previous = stage()["runtime-profile.txt"]
    path = {"installer": runtime_lock.INSTALLER_PATH, "helper": runtime_lock.HELPER_PATH}.get(
        filename, templates / filename
    )
    path.write_bytes(path.read_bytes() + b"\n")
    assert stage()["runtime-profile.txt"] != previous


def test_profile_invalidates_cached_install_when_pinned_runtime_identity_changes(templates: Path) -> None:
    assert (
        stage(identity="pinned-runtime-a")["runtime-profile.txt"]
        != stage(identity="pinned-runtime-b")["runtime-profile.txt"]
    )


def test_archive_directory_is_escaped_as_json_data(templates: Path) -> None:
    staged = stage('/tmp/session with "quotes"')
    manifest = json.loads(staged["runtime-package.json"])
    assert manifest["dependencies"]["prime-agent"] == 'file:/tmp/session with "quotes"/prime-agent.tgz'


@pytest.mark.parametrize(
    "directory",
    [
        "",
        "relative/session",
        "/",
        "/tmp/../other",
        "https://example.invalid/a",
        "/tmp/line\nbreak",
        "/tmp/null\x00byte",
        "/tmp/query?value",
        "/tmp/a\\b",
    ],
)
def test_reject_unsafe_or_remote_archive_directory(templates: Path, directory: str) -> None:
    with pytest.raises(ValueError, match="archive directory"):
        stage(directory)


@pytest.mark.parametrize("fault", ["missing", "extra", "not-sri", "bad-base64", "short-digest"])
def test_reject_incomplete_or_invalid_archive_integrities(templates: Path, fault: str) -> None:
    supplied = integrities()
    if fault == "missing":
        supplied.pop("prime-agent-ai")
    elif fault == "extra":
        supplied["unknown"] = supplied["prime-agent"]
    else:
        supplied["prime-agent"] = {"not-sri": "sha256-abc", "bad-base64": "sha512-!", "short-digest": "sha512-YQ=="}[
            fault
        ]
    with pytest.raises(ValueError):
        runtime_lock.stage_runtime_inputs(
            archive_directory="/tmp/session", archive_integrities=supplied, runtime_identity="pinned-runtime"
        )


def test_reject_unknown_template_placeholder(templates: Path) -> None:
    edit_json(templates / "package.json", lambda item: item.update(description="__UNEXPECTED_VALUE__"))
    with pytest.raises(ValueError, match="placeholder"):
        stage()


def test_reject_misplaced_known_placeholder(templates: Path) -> None:
    edit_json(templates / "package.json", lambda item: item.update(description=TOKEN))
    with pytest.raises(ValueError, match="placeholder"):
        stage()


def test_reject_unresolved_python_constraint_placeholder(templates: Path) -> None:
    (templates / "python-constraints.txt").write_text("package==__UNRESOLVED_VERSION__\n")
    with pytest.raises(ValueError, match="placeholders"):
        stage()


@pytest.mark.parametrize(
    "reference", ["file:/tmp/foreign.tgz", "file:../foreign.tgz", "file:https://example.invalid/archive.tgz"]
)
def test_reject_unexpected_local_artifact_reference(templates: Path, reference: str) -> None:
    edit_json(
        templates / "package-lock.json",
        lambda item: item["packages"]["node_modules/registry-example"].update(resolved=reference),
    )
    with pytest.raises(ValueError, match="file reference"):
        stage()


def test_reject_unexpected_remote_artifact(templates: Path) -> None:
    edit_json(
        templates / "package-lock.json",
        lambda item: item["packages"]["node_modules/registry-example"].update(
            resolved="https://example.invalid/unreviewed.tgz"
        ),
    )
    with pytest.raises(ValueError, match="remote artifact"):
        stage()


def test_reject_registry_artifact_without_integrity(templates: Path) -> None:
    edit_json(
        templates / "package-lock.json",
        lambda item: item["packages"]["node_modules/registry-example"].pop("integrity"),
    )
    with pytest.raises(ValueError, match="integrities"):
        stage()


def test_reject_archive_with_another_archives_integrity_placeholder(templates: Path) -> None:
    edit_json(
        templates / "package-lock.json",
        lambda item: item["packages"]["node_modules/prime-agent"].update(
            integrity="__NEMO_GYM_PRIME_AGENT_AI_INTEGRITY__"
        ),
    )
    with pytest.raises(ValueError, match="wrong integrity"):
        stage()


def test_reject_missing_verified_release_archive(templates: Path) -> None:
    edit_json(templates / "package-lock.json", lambda item: item["packages"].pop("node_modules/prime-agent-tui"))
    with pytest.raises(ValueError, match="all four"):
        stage()


def test_reject_manifest_lock_dependency_disagreement(templates: Path) -> None:
    edit_json(templates / "package-lock.json", lambda item: item["packages"][""].update(dependencies={}))
    with pytest.raises(ValueError, match="root dependencies disagree"):
        stage()


def test_reject_directory_link_dependency(templates: Path) -> None:
    edit_json(
        templates / "package-lock.json",
        lambda item: item["packages"]["node_modules/registry-example"].update(link=True),
    )
    with pytest.raises(ValueError, match="dependency link"):
        stage()
