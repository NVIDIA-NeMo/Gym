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
"""Exercise Harbor's actual verifier factory with private host evidence."""

import json
import shutil
from types import SimpleNamespace

import pytest
from harbor.models.trial.config import VerifierConfig
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.factory import VerifierFactory

from benchmarks.skill_trigger.prepare import digest, encode, tree_digest
from environments.skill_trigger.scoring import EvidenceError
from environments.skill_trigger.tests.test_scoring import SKILL
from environments.skill_trigger.tests.test_scoring import evidence as evidence
from environments.skill_trigger.verifier import EvidenceStore, load_case


@pytest.fixture
def prepared(tmp_path):
    release = tmp_path / "release"
    folder = release / "catalogs/catalog/target"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(SKILL)
    files = [("SKILL.md", digest(SKILL.encode()))]
    members = [{"name": "target", "files": files, "mounted_sha256": tree_digest(files)}]
    catalog_hash = digest(encode(members))
    case = {
        "task_name": "case-one",
        "id": "case",
        "prompt": "Use the relevant skill",
        "prompt_sha256": digest(b"Use the relevant skill"),
        "catalog": "catalog",
        "catalog_sha256": catalog_hash,
        "skill": "target",
    }
    manifest = {
        "artifact_kind": "prepared-release",
        "release_blockers": [],
        "cases": [case],
        "catalogs": {"catalog": {"members": members, "sha256": catalog_hash}},
    }
    (release / "manifest.json").write_bytes(encode(manifest))
    return release


def inputs(release):
    return dict(
        release=release,
        manifest_sha256=digest((release / "manifest.json").read_bytes()),
        task_name="case-one",
        model="model",
        harness="pi",
    )


def factory(tmp_path, prepared):
    return VerifierFactory.create_verifier_from_config(
        VerifierConfig(
            import_path="environments.skill_trigger.verifier:SkillOpeningVerifier",
            kwargs=dict(
                release=str(prepared),
                manifest_sha256=inputs(prepared)["manifest_sha256"],
                evidence_root=str(tmp_path / "private"),
                model="model",
                harness="pi",
            ),
        ),
        task=SimpleNamespace(short_name="case-one", instruction="Use the relevant skill"),
        trial_paths=TrialPaths(trial_dir=tmp_path / "harbor/trial"),
        environment=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("read", [True, False])
async def test_real_harbor_factory_returns_explicit_binary_reward(tmp_path, prepared, evidence, read):
    verifier = factory(tmp_path, prepared)
    evidence.identity = verifier.case.identity
    if not read:
        evidence.outputs = []
    verifier.store.begin()
    verifier.store.finish(evidence)
    assert (await verifier.verify()).rewards == {"reward": int(read)}


@pytest.mark.asyncio
async def test_agent_reward_file_cannot_replace_missing_host_evidence(tmp_path, prepared):
    verifier = factory(tmp_path, prepared)
    verifier.trial_paths.verifier_dir.mkdir(parents=True)
    (verifier.trial_paths.verifier_dir / "reward.json").write_text('{"reward": 1}')
    with pytest.raises(FileNotFoundError):
        await verifier.verify()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["provider_error", "trace_error", "observation_error", "cancelled"])
async def test_errors_raise_and_preserve_host_evidence(tmp_path, prepared, evidence, outcome):
    verifier = factory(tmp_path, prepared)
    evidence.identity = verifier.case.identity
    evidence.outcome = outcome
    verifier.store.begin()
    verifier.store.finish(evidence)
    with pytest.raises(EvidenceError, match="Unscored"):
        await verifier.verify()
    assert verifier.store.read().outcome == outcome
    # Harbor's own cleanup cannot remove records outside its trial directory.
    verifier.trial_paths.trial_dir.mkdir(parents=True)
    shutil.rmtree(verifier.trial_paths.trial_dir)
    assert verifier.store.read().outcome == outcome


def test_store_never_overwrites_an_attempt(tmp_path, evidence):
    store = EvidenceStore(root=tmp_path / "private", trial_dir=tmp_path / "trial")
    store.begin()
    store.finish(evidence)
    with pytest.raises(FileExistsError):
        store.begin()
    evidence.outputs = []
    with pytest.raises(FileExistsError):
        store.finish(evidence)
    assert store.read().outputs


def test_store_rejects_record_from_another_trial(tmp_path, evidence):
    stores = [EvidenceStore(root=tmp_path / "private", trial_dir=tmp_path / name) for name in ("a", "b")]
    for store in stores:
        store.begin()
    stores[0].finish(evidence)
    shutil.copyfile(stores[0].path / "evidence.json", stores[1].path / "evidence.json")
    (stores[1].path / "evidence.json").chmod(0o600)
    with pytest.raises(EvidenceError, match="different Harbor trial"):
        stores[1].read()


@pytest.mark.parametrize("kind", ["inside", "permissions", "symlink"])
def test_private_storage_boundaries(tmp_path, kind):
    root, trial = tmp_path / "private", tmp_path / "trial"
    if kind == "inside":
        root = trial / "agent/private"
    elif kind == "permissions":
        root.mkdir(mode=0o755)
    else:
        (tmp_path / "other").mkdir()
        root.symlink_to(tmp_path / "other", target_is_directory=True)
    with pytest.raises(EvidenceError):
        EvidenceStore(root=root, trial_dir=trial)


@pytest.mark.parametrize("change", ["audit", "blocked", "prompt", "catalog", "missing_target", "duplicate_case"])
def test_rejects_unusable_release(prepared, change):
    path = prepared / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if change == "audit":
        manifest["artifact_kind"] = "source-audit"
    elif change == "blocked":
        manifest["release_blockers"] = [{"reason": "unresolved"}]
    elif change == "prompt":
        manifest["cases"][0]["prompt"] += " changed"
    elif change == "catalog":
        manifest["catalogs"]["catalog"]["sha256"] = "wrong"
    elif change == "missing_target":
        manifest["cases"][0]["skill"] = "missing"
    else:
        manifest["cases"] *= 2
    path.write_bytes(encode(manifest))
    with pytest.raises(EvidenceError):
        load_case(**inputs(prepared))


@pytest.mark.parametrize("change", ["body", "extra_file", "symlink", "labels"])
def test_detects_changed_catalog_files(prepared, change):
    folder = prepared / "catalogs/catalog/target"
    if change == "body":
        (folder / "SKILL.md").write_text("changed")
    elif change == "extra_file":
        (folder / "extra.txt").write_text("extra")
    elif change == "symlink":
        (folder / "link").symlink_to(folder / "SKILL.md")
    else:
        (folder / "evals").mkdir()
        (folder / "evals/answers.json").write_text("{}")
    with pytest.raises(EvidenceError):
        load_case(**inputs(prepared))


def test_manifest_hash_and_harbor_prompt_are_bound(tmp_path, prepared):
    config = inputs(prepared)
    (prepared / "manifest.json").write_bytes((prepared / "manifest.json").read_bytes() + b" ")
    with pytest.raises(EvidenceError, match="release hash"):
        load_case(**config)
    verifier = factory(tmp_path, prepared)
    assert verifier.case.prompt == "Use the relevant skill"
    with pytest.raises(EvidenceError, match="frozen prompt"):
        type(verifier)(
            release=str(prepared),
            manifest_sha256=inputs(prepared)["manifest_sha256"],
            evidence_root=str(tmp_path / "private"),
            model="model",
            harness="pi",
            task=SimpleNamespace(short_name="case-one", instruction="changed"),
            trial_paths=verifier.trial_paths,
            environment=None,
        )
