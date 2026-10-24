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
"""Harbor verifier for host-collected evidence, never agent-written reward files.

The adapter and verifier share a private host directory outside Harbor's mounted
trial logs. The adapter must not mount it into any container. This module binds
evidence to a trial and a pinned prepared release; it does not make agent-supplied
JSON trustworthy or replace the adapter's independent collection.
"""

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.skill_trigger.prepare import digest, encode, source_path, tree_digest
from environments.skill_trigger.scoring import EvidenceError, Identity, TrialEvidence, score_trial


@dataclass(frozen=True)
class PreparedCase:
    """Expected identity and instructions read from the pinned host-only release."""

    identity: Identity
    prompt: str
    target: str
    sources: dict[str, str]
    mounted_hashes: dict[str, str]
    skill_directories: tuple[Path, ...]


def load_case(*, release: Path, manifest_sha256: str, task_name: str, model: str, harness: str) -> PreparedCase:
    """Reject audit artifacts and changed prompts, catalogs, or skill files."""
    raw = source_path(release, "manifest.json").read_bytes()
    if digest(raw) != manifest_sha256:
        raise EvidenceError("Prepared manifest differs from the configured release hash")
    manifest = json.loads(raw)
    if manifest["artifact_kind"] != "prepared-release" or manifest["release_blockers"]:
        raise EvidenceError("Only a prepared release with resolved file-level terms is runnable")
    matches = [case for case in manifest["cases"] if case["task_name"] == task_name]
    if len(matches) != 1:
        raise EvidenceError("Task lacks a unique case in the prepared release")
    case = matches[0]
    if digest(case["prompt"].encode()) != case["prompt_sha256"]:
        raise EvidenceError("Prepared prompt hash mismatch")
    catalog = manifest["catalogs"][case["catalog"]]
    if digest(encode(catalog["members"])) != catalog["sha256"] or catalog["sha256"] != case["catalog_sha256"]:
        raise EvidenceError("Prepared catalog hash mismatch")
    sources = {}
    for member in catalog["members"]:
        name = member["name"]
        if name in sources or not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise EvidenceError("Catalog contains a duplicate or unsafe skill name")
        folder = source_path(release, f"catalogs/{case['catalog']}/{name}")
        actual = []
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise EvidenceError("Prepared catalog contains a symlink")
            if path.is_file():
                relative = path.relative_to(folder).as_posix()
                if set(path.relative_to(folder).parts) & {"evals", ".git", "__pycache__"}:
                    raise EvidenceError("Prepared catalog exposes evaluation labels")
                actual.append((relative, digest(path.read_bytes())))
        if actual != [tuple(item) for item in member["files"]] or tree_digest(actual) != member["mounted_sha256"]:
            raise EvidenceError("Prepared skill files differ from the frozen catalog")
        sources[name] = source_path(folder, "SKILL.md").read_text(encoding="utf-8")
    if case["skill"] not in sources:
        raise EvidenceError("Prepared target is absent from the catalog")
    return PreparedCase(
        identity=Identity(
            task_id=case["id"],
            prompt_sha256=case["prompt_sha256"],
            catalog_sha256=case["catalog_sha256"],
            model=model,
            harness=harness,
        ),
        prompt=case["prompt"],
        target=case["skill"],
        sources=sources,
        skill_directories=tuple(source_path(release, f"catalogs/{case['catalog']}/{name}") for name in sources),
        mounted_hashes={member["name"]: member["mounted_sha256"] for member in catalog["members"]},
    )


class EvidenceStore:
    """Write each trial once, outside logs that Harbor exposes to the agent."""

    def __init__(self, *, root: Path, trial_dir: Path) -> None:
        self.trial_dir = trial_dir.resolve()
        root = root.absolute()
        if root.resolve() != root or root.is_relative_to(self.trial_dir):
            raise EvidenceError("Evidence storage must be a separate host path without symlinks")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = root.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise EvidenceError("Evidence root must be owned by the host user with mode 0700")
        self.path = root / digest(str(self.trial_dir).encode())

    def begin(self) -> Path:
        """Allocate a unique attempt before execution; never reuse an old trial."""
        self.path.mkdir(mode=0o700)
        return self.path

    def _check_directory(self) -> None:
        info = self.path.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
            raise EvidenceError("Trial evidence directory is not private host storage")

    def finish(self, evidence: TrialEvidence) -> None:
        """Preserve even unscored outcomes; publish a complete record atomically once."""
        self._check_directory()
        data = encode({"trial_dir": str(self.trial_dir), "evidence": evidence.model_dump()})
        fd, temporary = tempfile.mkstemp(dir=self.path)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.link(temporary, self.path / "evidence.json")
        finally:
            Path(temporary).unlink(missing_ok=True)

    def read(self) -> TrialEvidence:
        """Read a finished host record; missing or corrupted evidence raises."""
        self._check_directory()
        path = self.path / "evidence.json"
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise EvidenceError("Evidence file is not private host storage")
        data = json.loads(path.read_bytes())
        if data["trial_dir"] != str(self.trial_dir):
            raise EvidenceError("Evidence belongs to a different Harbor trial")
        return TrialEvidence.model_validate(data["evidence"])


class SkillOpeningVerifier(BaseVerifier):
    """Return an explicit binary reward only after complete host verification."""

    def __init__(
        self,
        *,
        release: str,
        manifest_sha256: str,
        evidence_root: str,
        model: str,
        harness: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.step_name is not None:
            raise EvidenceError("Skill-opening trials must contain one task, not Harbor steps")
        self.case = load_case(
            release=Path(release),
            manifest_sha256=manifest_sha256,
            task_name=self.task.short_name,
            model=model,
            harness=harness,
        )
        if self.task.instruction != self.case.prompt:
            raise EvidenceError("Harbor instruction differs from the frozen prompt")
        self.store = EvidenceStore(root=Path(evidence_root), trial_dir=self.trial_paths.trial_dir)

    async def verify(self) -> VerifierResult:
        evidence = self.store.read()
        score = score_trial(evidence, expected=self.case.identity, sources=self.case.sources, target=self.case.target)
        return VerifierResult(rewards={"reward": score.reward})
