# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""Load published Blobfish Harbor task packages as in-process, session-isolated worlds.

A *task package* is one directory produced by ``harbor datasets download blobfishai/<suite>``::

    <task_id>/
      task.toml               Harbor manifest; [metadata] carries benchmark / world_id / task_id
      instruction.md          the employee-facing assignment (becomes the user prompt)
      environment/
        runtime.py            the world: a class exposing create()/fresh(), call_tool(), snapshot(), trace
        task.json             seed data plus role / company context
        tools.json            MCP tool definitions (name, description, inputSchema, annotations)
        schema.sql, assets/
      tests/
        task.json             the verifier's view of the task (expected answers, rubric)
        verify.py             deterministic verifier: reads evidence.json, writes reward.json
        <bench>/evaluation.py optional in-process scorer exposing score_episode()
      solution/               oracle plan (plan.json, or reference.json with oracle_steps)

Every published suite ships the world as plain Python over SQLite, so one process can host
thousands of isolated episodes without Docker. Nothing in this module imports NeMo Gym; the
``server`` module layers that on top.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


__all__ = [
    "Episode",
    "Suite",
    "SuiteIntegrityError",
    "TaskPackage",
    "ToolSpec",
    "Verdict",
    "VerifierError",
    "WorldFactory",
    "json_schema_for",
    "package_digest",
    "public_tool_name",
    "score_episode",
]

# Responses-API function names admit letters, digits, underscores and hyphens only. Blobfish
# tools are namespaced with dots (``gmail.messages.list``); the mapping must be reversible per
# suite, which ``Suite.load`` enforces.
_PUBLIC_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")
_EVIDENCE_ENV_RE = re.compile(r"\b([A-Z][A-Z0-9_]*_EVIDENCE_PATH)\b")
_VERIFIER_TIMEOUT_SECONDS = 180.0


class VerifierError(RuntimeError):
    """The package's own verifier could not produce a verdict."""


class SuiteIntegrityError(RuntimeError):
    """A task package does not match the digest it was pinned to."""


def package_digest(root: Path) -> str:
    """SHA-256 over every file in a task package (relative path and bytes), byte-cache excluded.

    Task packages are executable code; the emitted suite manifest pins each package to this digest
    so a server refuses to run anything other than what was reviewed.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


class _no_bytecode:
    """Do not litter downloaded task packages with __pycache__ while importing their code."""

    def __enter__(self) -> None:
        self._saved = sys.dont_write_bytecode
        sys.dont_write_bytecode = True

    def __exit__(self, *_: Any) -> None:
        sys.dont_write_bytecode = self._saved


def public_tool_name(name: str) -> str:
    """Map a canonical Blobfish tool name onto a Responses-API-safe function name."""
    return _PUBLIC_NAME_RE.sub("__", name)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    public_name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool | None = None

    def as_responses_tool(self) -> dict[str, Any]:
        """The tool as a Responses API function declaration (what NeMo Gym rows carry)."""
        return {
            "type": "function",
            "name": self.public_name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": False,
        }


_DESCRIPTOR_TYPES = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


def json_schema_for(item: dict[str, Any]) -> dict[str, Any]:
    """The JSON-schema ``parameters`` object for one tool definition.

    Published suites declare tool inputs three ways: an MCP ``inputSchema`` object, a
    ``json_schema.parameters`` object next to a list of parameter descriptors (the companion-world
    convention), or the descriptor list alone (``[{"name", "type", "required", "description"}]``).
    Responses-API function tools need an object schema, so descriptor lists are converted.
    """
    nested = item.get("json_schema")
    if isinstance(nested, dict) and isinstance(nested.get("parameters"), dict):
        return nested["parameters"]
    for key in ("inputSchema", "parameters"):
        value = item.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            properties: dict[str, Any] = {}
            required: list[str] = []
            for descriptor in value:
                if not isinstance(descriptor, dict) or not descriptor.get("name"):
                    continue
                name = str(descriptor["name"])
                prop: dict[str, Any] = {
                    "type": _DESCRIPTOR_TYPES.get(str(descriptor.get("type", "string")).lower(), "string")
                }
                if prop["type"] == "array":
                    prop["items"] = {}
                if descriptor.get("description"):
                    prop["description"] = str(descriptor["description"])
                properties[name] = prop
                if descriptor.get("required"):
                    required.append(name)
            return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    return {"type": "object", "properties": {}}


def _load_tools(path: Path) -> tuple[ToolSpec, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "tools" in raw and isinstance(raw["tools"], list):
            items = raw["tools"]
        else:  # grouped by MCP server name
            items = [tool for group in raw.values() if isinstance(group, list) for tool in group]
    else:
        items = list(raw)
    specs: list[ToolSpec] = []
    seen: dict[str, str] = {}
    for item in items:
        name = str(item["name"])
        public = public_tool_name(name)
        if seen.get(public, name) != name:
            raise ValueError(
                f"tool name collision after mapping: {seen[public]!r} and {name!r} both map to {public!r}"
            )
        seen[public] = name
        parameters = json_schema_for(item)
        annotations = item.get("annotations") or {}
        specs.append(
            ToolSpec(
                name=name,
                public_name=public,
                description=str(item.get("description") or ""),
                parameters=parameters,
                read_only=annotations.get("readOnlyHint"),
            )
        )
    return tuple(specs)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TaskPackage:
    """One downloaded Harbor task directory, parsed but not yet instantiated."""

    root: Path
    task_id: str
    benchmark: str
    world_id: str
    instruction: str
    metadata: dict[str, Any]
    environment_task: dict[str, Any]
    verifier_task: dict[str, Any]
    tools: tuple[ToolSpec, ...]
    oracle_steps: tuple[dict[str, Any], ...] | None
    profile: str = "runtime"

    @property
    def environment_dir(self) -> Path:
        return self.root / "environment"

    @property
    def tests_dir(self) -> Path:
        return self.root / "tests"

    @property
    def role(self) -> str | None:
        value = self.environment_task.get("role")
        return str(value) if value else None

    @property
    def company(self) -> str | None:
        value = self.environment_task.get("company")
        if not value and isinstance(self.environment_task.get("world"), dict):
            value = self.environment_task["world"].get("company")
        return str(value) if value else None

    @property
    def public_to_canonical(self) -> dict[str, str]:
        return {tool.public_name: tool.name for tool in self.tools}

    @classmethod
    def load(cls, root: Path) -> "TaskPackage":
        root = Path(root)
        manifest = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
        metadata = dict(manifest.get("metadata") or {})
        from . import profiles as _profiles

        kind = _profiles.detect_profile(root)
        if kind != "runtime":
            profile = _profiles.profile_for(root, kind)
            instruction_path = root / "instruction.md"
            instruction = instruction_path.read_text(encoding="utf-8").strip() if instruction_path.exists() else ""
            spec = dict(profile.spec)
            return cls(
                root=root,
                task_id=str(metadata.get("task_id") or spec.get("task_id") or root.name),
                benchmark=str(metadata.get("benchmark") or spec.get("benchmark") or root.parent.name),
                world_id=str(metadata.get("world_id") or spec.get("world_id") or ""),
                instruction=instruction,
                metadata=metadata,
                environment_task=spec,
                verifier_task={},
                tools=profile.tools,
                oracle_steps=profile.oracle_steps(),
                profile=kind,
            )
        environment_task = _read_json(root / "environment" / "task.json")
        verifier_path = root / "tests" / "task.json"
        verifier_task = _read_json(verifier_path) if verifier_path.exists() else environment_task
        task_id = str(metadata.get("task_id") or environment_task.get("task_id") or root.name)
        instruction_path = root / "instruction.md"
        instruction = instruction_path.read_text(encoding="utf-8").strip() if instruction_path.exists() else ""
        if not instruction:
            instruction = str(environment_task.get("instruction") or environment_task.get("prompt") or "")
        return cls(
            root=root,
            task_id=task_id,
            benchmark=str(metadata.get("benchmark") or environment_task.get("benchmark") or root.parent.name),
            world_id=str(metadata.get("world_id") or environment_task.get("world_id") or ""),
            instruction=instruction,
            metadata=metadata,
            environment_task=environment_task,
            verifier_task=verifier_task,
            tools=_load_tools(root / "environment" / "tools.json"),
            oracle_steps=_load_oracle_steps(root, verifier_task),
        )


def _load_oracle_steps(root: Path, verifier_task: dict[str, Any]) -> tuple[dict[str, Any], ...] | None:
    plan = root / "solution" / "plan.json"
    if plan.exists():
        steps = json.loads(plan.read_text(encoding="utf-8"))
        return tuple(steps) if isinstance(steps, list) else None
    reference = root / "solution" / "reference.json"
    if reference.exists():
        data = json.loads(reference.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("oracle_steps"), list):
            return tuple(data["oracle_steps"])
    if isinstance(verifier_task.get("oracle_steps"), list):
        return tuple(verifier_task["oracle_steps"])
    return None


@dataclass
class Suite:
    """All task packages under one directory (typically one Harbor dataset download)."""

    root: Path
    name: str
    tasks: tuple[TaskPackage, ...]
    tools: tuple[ToolSpec, ...]
    _by_id: dict[str, TaskPackage] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {task.task_id: task for task in self.tasks}

    @classmethod
    def load(cls, root: Path, name: str | None = None, *, expected_digests: dict[str, str] | None = None) -> "Suite":
        root = Path(root)
        if (root / "task.toml").exists():  # a single task directory was given
            task_dirs = [root]
        else:
            task_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "task.toml").exists())
        if not task_dirs:
            raise FileNotFoundError(f"no Harbor task packages (task.toml) under {root}")
        if expected_digests is not None:
            found = {path.name: package_digest(path) for path in task_dirs}
            missing = sorted(set(expected_digests) - set(found))
            changed = sorted(
                name_ for name_, digest in found.items() if expected_digests.get(name_) not in (None, digest)
            )
            extra = sorted(set(found) - set(expected_digests))
            if missing or changed or extra:
                raise SuiteIntegrityError(
                    f"{root}: task packages differ from the pinned manifest "
                    f"(missing={missing[:3]}, changed={changed[:3]}, unexpected={extra[:3]})"
                )
        tasks = tuple(TaskPackage.load(path) for path in task_dirs)
        union: dict[str, ToolSpec] = {}
        for task in tasks:
            for tool in task.tools:
                previous = union.get(tool.public_name)
                if previous is not None and previous.name != tool.name:
                    raise ValueError(f"suite-wide tool name collision on {tool.public_name!r}")
                union.setdefault(tool.public_name, tool)
        suite_name = name or tasks[0].benchmark or root.name
        return cls(root=root, name=suite_name, tasks=tasks, tools=tuple(union.values()))

    def get(self, task_id: str) -> TaskPackage:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task {task_id!r} in suite {self.name!r}") from exc

    def __len__(self) -> int:
        return len(self.tasks)


class WorldFactory:
    """Import each package's ``environment/runtime.py`` once and build isolated worlds from it."""

    _lock = threading.Lock()
    _modules: dict[str, ModuleType] = {}

    @classmethod
    def module_for(cls, environment_dir: Path) -> ModuleType:
        runtime = environment_dir / "runtime.py"
        digest = hashlib.sha256()
        for path in sorted(environment_dir.rglob("*")):
            if (
                path.is_file()
                and (path.suffix in {".py", ".sql"} or path.name == "tools.json")
                and "assets" not in path.parts
                and "__pycache__" not in path.parts
            ):
                digest.update(str(path.relative_to(environment_dir)).encode())
                digest.update(path.read_bytes())
        key = digest.hexdigest()
        with cls._lock:
            module = cls._modules.get(key)
            if module is None:
                module_name = f"blobfish_suite_runtime_{key[:16]}"
                spec = importlib.util.spec_from_file_location(module_name, runtime)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot import world runtime at {runtime}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                with _no_bytecode(), _sibling_modules(environment_dir):
                    spec.loader.exec_module(module)
                cls._modules[key] = module
        return module

    @staticmethod
    def world_class(module: ModuleType) -> type:
        candidates = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type)
            and callable(getattr(obj, "call_tool", None))
            and callable(getattr(obj, "snapshot", None))
        ]
        local = [obj for obj in candidates if obj.__module__ == module.__name__]
        if local:
            candidates = local
        if len(candidates) != 1:
            names = [candidate.__name__ for candidate in candidates]
            raise ImportError(
                f"expected exactly one world class with call_tool()/snapshot() in {module.__name__}, found {names}"
            )
        return candidates[0]

    @classmethod
    def create(cls, task: TaskPackage, db_path: Path) -> Any:
        module = cls.module_for(task.environment_dir)
        world_cls = cls.world_class(module)
        task_json = deepcopy(task.environment_task)
        if callable(getattr(world_cls, "create", None)):
            world = world_cls.create(task_json, db_path)
        elif callable(getattr(world_cls, "fresh", None)):
            world = world_cls.fresh(task_json, db_path)
        else:
            seed = getattr(module, "seed_database", None)
            if seed is None:
                raise ImportError(
                    f"{world_cls.__name__} has no create()/fresh() and the runtime has no seed_database()"
                )
            seeded = seed(task_json, db_path)
            world = world_cls(task_json, seeded if isinstance(seeded, sqlite3.Connection) else db_path)
        _relax_thread_affinity(world, db_path)
        return world


class _sibling_modules:
    """Make ``environment/*.py`` siblings importable by bare name while a runtime is imported.

    Published runtimes import helpers such as ``from contracts import TOOL_BY_NAME``. Each suite
    ships its own copy, so the bare names are bound only for the duration of the import and the
    previous bindings are restored afterwards; the runtime keeps references to what it imported.
    """

    _SKIP = {"runtime", "service", "server", "__init__"}  # entrypoints instantiate worlds at import

    def __init__(self, environment_dir: Path):
        self.environment_dir = environment_dir
        self._saved: dict[str, ModuleType | None] = {}
        self._packages: list[str] = []
        self._path_added = False

    def __enter__(self) -> "_sibling_modules":
        try:
            return self._enter()
        except BaseException:
            self.__exit__()
            raise

    def _enter(self) -> "_sibling_modules":
        if str(self.environment_dir) not in sys.path:
            sys.path.insert(0, str(self.environment_dir))
            self._path_added = True
        self._packages = [
            p.name for p in self.environment_dir.iterdir() if p.is_dir() and (p / "__init__.py").exists()
        ]
        # Evict any same-named package another suite (or a tests/ copy) bound, so the runtime's own
        # absolute imports resolve against the copy that sits next to it.
        for package in self._packages:
            for key in [k for k in sys.modules if k == package or k.startswith(package + ".")]:
                self._saved[key] = sys.modules.pop(key)
        for path in sorted(self.environment_dir.glob("*.py")):
            name = path.stem
            if name in self._SKIP:
                continue
            self._saved.setdefault(name, sys.modules.get(name))
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return self

    def __exit__(self, *_: Any) -> None:
        for package in self._packages:
            for key in [k for k in sys.modules if k == package or k.startswith(package + ".")]:
                sys.modules.pop(key, None)
        for name, previous in self._saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if self._path_added:
            try:
                sys.path.remove(str(self.environment_dir))
            except ValueError:
                pass


def _relax_thread_affinity(world: Any, db_path: Path) -> None:
    """Reopen the world's SQLite connection so any worker thread may serve the episode.

    The published runtimes open SQLite with the default ``check_same_thread=True``. A NeMo Gym
    worker dispatches tool calls from a thread pool, so the connection is reopened on the same
    file with thread checks off; ``Episode`` serialises calls per episode with a lock.
    """
    connection = getattr(world, "connection", None)
    if not isinstance(connection, sqlite3.Connection):
        return
    connection.commit()
    connection.close()
    fresh = sqlite3.connect(db_path, check_same_thread=False)
    fresh.row_factory = sqlite3.Row
    fresh.execute("PRAGMA foreign_keys = ON")
    world.connection = fresh


class Episode:
    """One isolated attempt at one task: a fresh world, its baseline snapshot, and the trace."""

    def __init__(self, task: TaskPackage, workdir: Path | None = None):
        self.task = task
        self._tmp = tempfile.TemporaryDirectory(prefix=f"blobfish-{task.task_id}-") if workdir is None else None
        self.workdir = Path(self._tmp.name) if self._tmp is not None else Path(workdir)  # type: ignore[arg-type]
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self.by_public = task.public_to_canonical
        self._lock = threading.Lock()
        self._closed = False
        self.session: Any = None
        self.world: Any = None
        self.baseline: dict[str, Any] = {}
        if task.profile == "runtime":
            self.db_path = self.workdir / "world.sqlite3"
            self.world = WorldFactory.create(task, self.db_path)
            self.baseline = deepcopy(self.world.snapshot())
        else:
            from . import profiles as _profiles

            self.session = _profiles.profile_for(task.root, task.profile).open(self.workdir)

    def resolve_tool(self, name: str) -> str | None:
        if self.session is not None:
            return name if self.session.resolve(name) is not None else None
        if name in self.by_public:
            return self.by_public[name]
        if name in {tool.name for tool in self.task.tools}:
            return name
        return None

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.session is not None:
            return self.session.call(name, arguments)
        canonical = self.resolve_tool(name)
        if canonical is None:
            return {"error": f"unknown tool: {name}"}
        with self._lock:
            try:
                result = self.world.call_tool(canonical, dict(arguments or {}))
            except Exception as exc:  # the world's contract: errors are results the policy can read
                return {"error": f"{type(exc).__name__}: {exc}"}
        return result if isinstance(result, dict) else {"result": result}

    def replay(self, steps: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
        return [self.call(str(step["tool"]), step.get("arguments") or {}) for step in steps]

    @property
    def trace(self) -> list[dict[str, Any]]:
        if self.session is not None:
            return []
        return list(getattr(self.world, "trace", []) or [])

    def evidence(self) -> dict[str, Any]:
        if self.session is not None:
            return {"baseline": {}, "snapshot": {}, "trace": []}
        with self._lock:
            snapshot = self.world.snapshot()
        return {"baseline": self.baseline, "snapshot": snapshot, "trace": self.trace}

    def stats(self) -> tuple[int, int]:
        """``(tool_calls, failed_tool_calls)`` for the episode so far."""
        if self.session is not None:
            return self.session.stats()
        trace = self.trace
        failed = sum(1 for entry in trace if isinstance(entry, dict) and not entry.get("success", True))
        return len(trace), failed

    def score(self) -> "Verdict":
        """Run the package's own verifier over this episode's final state."""
        if self.session is not None:
            return self.session.score()
        return score_episode(self.task, self.evidence())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.session is not None:
            self.session.close()
        close = getattr(self.world, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self) -> "Episode":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class Criterion:
    """One graded criterion of a verdict, normalised across the suite verifier schemas."""

    name: str
    family: str
    weight: float
    earned: float
    passed: bool


def criteria_of(verdict: "Verdict") -> list[Criterion]:
    """Weighted criteria from a verdict's raw report, or [] when the schema has none."""
    raw = verdict.raw if isinstance(verdict.raw, dict) else {}
    items: list[dict[str, Any]] = []
    for key in ("checks", "milestones", "semantic_checks"):
        value = raw.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict) and "category" in value[0]:
            items = value
            break
    out: list[Criterion] = []
    for item in items:
        name = str(item.get("id") or item.get("name") or f"{item.get('category')}:{len(out)}")
        family = str(item.get("category") or name.split(".")[0].split(":")[0])
        weight = float(item.get("weight", item.get("points", 1.0)) or 0.0)
        if "earned_weight" in item:
            earned = float(item["earned_weight"] or 0.0)
        elif "earned_points" in item:
            earned = float(item["earned_points"] or 0.0)
        elif "earned" in item:
            earned = float(item["earned"] or 0.0)
        elif "points" in item and "weight" in item:
            earned = float(item["points"] or 0.0)
        else:
            earned = weight if item.get("passed") else 0.0
        passed = bool(item.get("passed", earned >= weight and weight > 0))
        out.append(Criterion(name=name, family=family, weight=weight, earned=earned, passed=passed))
    return out


# Criterion families that a wrong write breaks in at least half of a suite's tasks (measured by
# scripts/outcome_families.py, evidence in results/quality/outcome_families.json): the outcome of an
# episode, as opposed to the process around it (investigation, discovery, reasoning). The server's
# ``gated`` reward mode multiplies the criterion fraction by the earned share of these families, so a
# policy that does the process and writes the wrong thing is not paid for the process.
OUTCOME_FAMILIES: dict[str, frozenset[str]] = {
    "CounselBench-100": frozenset(["containment", "investigation", "reasoning", "state", "verification"]),
    "DealBench-100": frozenset(
        ["committed_state", "containment", "decision", "deliverable", "model_accuracy", "readback"]
    ),
    "DevOpsBench-100": frozenset(
        ["analysis", "answer", "decision", "execution", "investigation", "state", "verification"]
    ),
    "ERPBench-100": frozenset(["calculation", "containment", "decision", "erp_state", "handoff", "readback"]),
    "FactoryBench-100": frozenset(["analysis", "answer", "decision", "execution", "state", "verification"]),
    "LedgerBench-100": frozenset(["analysis", "answer", "decision", "execution", "state", "verification"]),
    "SalesBench-100": frozenset(
        ["answer", "containment", "decision", "execution", "investigation", "state", "verification"]
    ),
}


def outcome_fraction(verdict: "Verdict", suite_name: str, families: Iterable[str] | None = None) -> float | None:
    """Earned share of the outcome families' weight; None when the verdict has no criteria or no families apply."""
    chosen = frozenset(families) if families is not None else OUTCOME_FAMILIES.get(suite_name)
    if not chosen:
        return None
    criteria = [c for c in criteria_of(verdict) if c.family in chosen]
    possible = sum(c.weight for c in criteria)
    if possible <= 0:
        return None
    return max(0.0, min(1.0, sum(c.earned for c in criteria) / possible))


@dataclass(frozen=True)
class Verdict:
    reward_fraction: float
    strict_pass: bool
    raw: dict[str, Any]
    profile: str


_scorer_lock = threading.Lock()
_scorer_cache: dict[Path, Callable[..., dict[str, Any]] | None] = {}


def _package_digest(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".sql", ".json"}:
            digest.update(str(path.relative_to(package_dir)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _inprocess_scorer(tests_dir: Path) -> Callable[..., dict[str, Any]] | None:
    """``tests/<bench>/evaluation.py::score_episode`` when the package ships one.

    The package is imported under a content-hashed alias, so identical copies across the 100
    tasks of a suite import once and same-named packages from other suites never collide.
    The evaluation modules use relative imports only, which resolve inside the alias.
    """
    with _scorer_lock:
        if tests_dir in _scorer_cache:
            return _scorer_cache[tests_dir]
        scorer = None
        for evaluation in sorted(tests_dir.glob("*/evaluation.py")):
            package_dir = evaluation.parent
            alias = f"blobfish_suite_tests_{_package_digest(package_dir)[:16]}"
            if alias not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    alias, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)]
                )
                if spec is None or spec.loader is None:
                    continue
                package = importlib.util.module_from_spec(spec)
                sys.modules[alias] = package
                spec.loader.exec_module(package)
            module = importlib.import_module(f"{alias}.evaluation")
            candidate = getattr(module, "score_episode", None)
            if callable(candidate):
                scorer = candidate
                break
        _scorer_cache[tests_dir] = scorer
        return scorer


def _last_json_object(text: str) -> dict[str, Any]:
    start = text.rfind("\n{")
    start = 0 if start < 0 else start + 1
    if not text[start:].lstrip().startswith("{"):
        start = text.find("{")
    if start < 0:
        raise VerifierError("verifier printed no JSON verdict")
    return json.loads(text[start:])


def _normalize(verdict: dict[str, Any], reward_json: dict[str, Any] | None, profile: str) -> Verdict:
    fraction: float | None = None
    if reward_json and isinstance(reward_json.get("reward"), (int, float)):
        fraction = float(reward_json["reward"])
    elif isinstance(verdict.get("reward"), (int, float)):
        fraction = float(verdict["reward"])
    else:
        possible = verdict.get("points_possible")
        earned = verdict.get("points_earned")
        if isinstance(possible, (int, float)) and possible and isinstance(earned, (int, float)):
            fraction = float(earned) / float(possible)
        else:
            for key in ("criterion_fraction", "reward_fraction", "factory_score", "ledger_score", "score"):
                value = verdict.get(key)
                if isinstance(value, (int, float)):
                    fraction = float(value) / 100.0 if key.endswith("_score") else float(value)
                    break
    if fraction is None:
        raise VerifierError(f"verdict carries no reward: keys={sorted(verdict)}")
    fraction = min(max(fraction, 0.0), 1.0)
    strict = verdict.get("strict_pass")
    if strict is None:
        strict = verdict.get("passed")
    if strict is None:
        strict = fraction >= 1.0 - 1e-9
    return Verdict(reward_fraction=fraction, strict_pass=bool(strict), raw=verdict, profile=profile)


def score_episode(task: TaskPackage, evidence: dict[str, Any]) -> Verdict:
    """Run the package's own deterministic verifier over an episode's evidence.

    Two profiles exist in the published suites: an importable ``tests/<bench>/evaluation.py``
    with ``score_episode(task, baseline, snapshot, trace)``, and a self-contained
    ``tests/verify.py`` script that reads ``<BENCH>_EVIDENCE_PATH`` and writes ``reward.json``
    into ``VERIFIER_LOG_DIR``. Both produce the same two numbers: the criterion fraction and
    the strict pass.
    """
    scorer = _inprocess_scorer(task.tests_dir)
    if scorer is not None:
        verdict = scorer(deepcopy(task.verifier_task), evidence["baseline"], evidence["snapshot"], evidence["trace"])
        return _normalize(dict(verdict), None, "evaluation_module")
    verify = task.tests_dir / "verify.py"
    if not verify.exists():
        raise VerifierError(f"{task.task_id}: no tests/verify.py and no in-process scorer")
    env_names = sorted(set(_EVIDENCE_ENV_RE.findall(verify.read_text(encoding="utf-8")))) or ["BLOBFISH_EVIDENCE_PATH"]
    with tempfile.TemporaryDirectory(prefix=f"blobfish-verify-{task.task_id}-") as tmp:
        evidence_path = Path(tmp) / "evidence.json"
        evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
        logdir = Path(tmp) / "logs"
        logdir.mkdir()
        env = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
            if key in os.environ
        }
        env.update({name: str(evidence_path) for name in env_names})
        env["VERIFIER_LOG_DIR"] = str(logdir)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, str(verify)],
                cwd=task.tests_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=_VERIFIER_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VerifierError(f"{task.task_id}: verifier timed out") from exc
        if proc.returncode != 0:
            raise VerifierError(f"{task.task_id}: verifier exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")
        verdict = _last_json_object(proc.stdout)
        reward_json = None
        reward_path = logdir / "reward.json"
        if reward_path.exists():
            reward_json = json.loads(reward_path.read_text(encoding="utf-8"))
        return _normalize(verdict, reward_json, "verify_script")
