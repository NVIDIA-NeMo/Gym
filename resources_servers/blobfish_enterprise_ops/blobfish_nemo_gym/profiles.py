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
"""World-image profiles: published suites whose ``environment/world/`` carries the runtime.

Four of the published suites do not ship an ``environment/runtime.py``; their worlds live under
``environment/world/`` in one of four shapes, each hosted in-process here:

* ``image_framework``  (LedgerBench)  ``runtime/servers/<name>_server.py`` modules built on an
  env-var-driven ``framework.Server``; scored by ``server.build_report`` over the run directory.
* ``image_documents``  (CounselBench) ``world.py`` with a world class taking documents / output /
  state roots and a spec; ``verify(token)`` scores it.
* ``image_package``    (SalesBench)   a vendored package (``<pkg>/runtime/world.py``) whose world
  takes documents / output / state / spec / seed; tools are grouped per MCP server.
* ``image_companion``  (DevOpsBench)  ``server.py`` with a world class over ``environment.db`` and
  ``tools_combined.py``; ``verify(token)`` runs the packaged ``verify_task.py``.

Every profile yields the same session surface: ``call(public_name, arguments)``, ``score()``,
``stats()`` and ``close()``, with tool names made Responses-API safe by ``public_tool_name``.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .suite import (
    ToolSpec,
    Verdict,
    VerifierError,
    _no_bytecode,
    _normalize,
    _sibling_modules,
    json_schema_for,
    public_tool_name,
)


__all__ = ["IMAGE_PROFILES", "detect_profile", "profile_for", "verify_token"]

IMAGE_PROFILES = ("image_framework", "image_documents", "image_package", "image_companion")

_VERIFY_TOKEN_RE = re.compile(r'X-Verify-Token",\s*"([0-9a-f]{64})"')
_import_lock = threading.Lock()
_modules: dict[str, ModuleType] = {}
_profiles: dict[str, "ImageProfile"] = {}
_profiles_lock = threading.Lock()


def detect_profile(root: Path) -> str:
    env = root / "environment"
    if (env / "runtime.py").exists():
        return "runtime"
    world = env / "world"
    if (world / "runtime" / "lib" / "framework.py").exists():
        return "image_framework"
    if (world / "world.py").exists() and (world / "scoring.py").exists():
        return "image_documents"
    if (world / "server.py").exists() and (world / "tools_combined.py").exists():
        return "image_companion"
    if world.exists() and any((p / "runtime" / "world.py").exists() for p in world.iterdir() if p.is_dir()):
        return "image_package"
    raise ValueError(f"{root}: unrecognised task package profile")


def verify_token(root: Path) -> str | None:
    """The verifier capability token the Harbor test script presents to ``/verify``."""
    script = root / "tests" / "test.sh"
    if not script.exists():
        return None
    match = _VERIFY_TOKEN_RE.search(script.read_text(encoding="utf-8"))
    return match.group(1) if match else None


_PER_TASK_DIRS = {"state", "taskspec", "documents", "assets", "__pycache__"}


def _code_digest(directory: Path) -> str:
    """Digest of a world image's *code* (python and SQL), so identical runtimes import once per suite."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".sql"} and not (_PER_TASK_DIRS & set(path.parts)):
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_module(
    path: Path, alias: str, *, sibling_dir: Path | None = None, path_entries: tuple[Path, ...] = ()
) -> ModuleType:
    with _import_lock:
        if alias in _modules:
            return _modules[alias]
        added = [str(p) for p in path_entries if str(p) not in sys.path]
        for entry in added:
            sys.path.insert(0, entry)
        try:
            spec = importlib.util.spec_from_file_location(alias, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            with _no_bytecode():
                if sibling_dir is not None:
                    with _sibling_modules(sibling_dir):
                        spec.loader.exec_module(module)
                else:
                    spec.loader.exec_module(module)
        finally:
            for entry in added:
                try:
                    sys.path.remove(entry)
                except ValueError:
                    pass
        _modules[alias] = module
        return module


def _plain_result(result: Any) -> dict[str, Any]:
    """Flatten MCP-style ``{"content": [...], "isError": ...}`` results into what the agent reads."""
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        text = "\n".join(str(item.get("text", "")) for item in result["content"] if isinstance(item, dict))
        if result.get("isError"):
            return {"error": text or "tool error"}
        try:
            payload = json.loads(text) if text else None
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            return payload
        return {"result": payload if payload is not None else text}
    if isinstance(result, dict):
        return result
    return {"result": result}


def _is_failure(result: dict[str, Any]) -> bool:
    if "error" in result and result.get("ok") is not True and result.get("success") is not True:
        return bool(result["error"])
    return result.get("ok") is False


def _report_verdict(report: dict[str, Any], profile: str) -> Verdict:
    report = dict(report)
    if not isinstance(report.get("reward"), (int, float)):
        possible = report.get("points_possible")
        earned = report.get("points_earned")
        if isinstance(possible, (int, float)) and possible and isinstance(earned, (int, float)):
            report["reward"] = float(earned) / float(possible)
        elif "passed" in report:
            report["reward"] = 1.0 if report["passed"] else 0.0
    return _normalize(report, None, profile)


class ImageSession:
    """Common bookkeeping for image-profile sessions."""

    def __init__(self, profile: "ImageProfile", workdir: Path):
        self.profile = profile
        self.workdir = workdir
        self.calls = 0
        self.failed = 0
        self._lock = threading.Lock()

    def resolve(self, name: str) -> ToolSpec | None:
        return self.profile.by_public.get(name) or self.profile.by_name.get(name)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        tool = self.resolve(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        with self._lock:
            try:
                result = _plain_result(self._invoke(tool, dict(arguments or {})))
            except Exception as exc:  # the world's own contract: errors are results, not crashes
                result = {"error": f"{type(exc).__name__}: {exc}"}
            self.calls += 1
            if _is_failure(result):
                self.failed += 1
        return result

    def stats(self) -> tuple[int, int]:
        return self.calls, self.failed

    def _invoke(self, tool: ToolSpec, arguments: dict[str, Any]) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self) -> Verdict:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        pass


class ImageProfile:
    """Per-package view of a world image: tools, oracle plan, and a session factory."""

    name = "image"

    def __init__(self, root: Path):
        self.root = root
        self.world_dir = root / "environment" / "world"
        self.spec = (
            json.loads((self.world_dir / "spec.json").read_text(encoding="utf-8"))
            if (self.world_dir / "spec.json").exists()
            else {}
        )
        self.token = verify_token(root)
        self.tools: tuple[ToolSpec, ...] = tuple(self._tools())
        self.by_public = {tool.public_name: tool for tool in self.tools}
        self.by_name = {tool.name: tool for tool in self.tools}

    # -- to override
    def _tools(self) -> list[ToolSpec]:  # pragma: no cover - abstract
        raise NotImplementedError

    def oracle_steps(self) -> tuple[dict[str, Any], ...] | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def open(self, workdir: Path) -> ImageSession:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers
    @staticmethod
    def _spec_from_mcp(name: str, public: str, item: dict[str, Any]) -> ToolSpec:
        annotations = item.get("annotations") or {}
        return ToolSpec(
            name=name,
            public_name=public,
            description=str(item.get("description") or ""),
            parameters=json_schema_for(item),
            read_only=annotations.get("readOnlyHint"),
        )

    def _reference(self) -> dict[str, Any]:
        path = self.root / "solution" / "reference.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --------------------------------------------------------------------------- LedgerBench
class _ThreadLocalEnviron:
    """``os.environ`` stand-in: per-thread overrides first, the real environment second.

    The framework-style runtimes read WORLD_DB / WORLD_NOW / WORLD_ROLE / TRACE_FILE on every
    call. Binding those per thread lets sessions run concurrently without touching the process
    environment (which other servers, subprocesses and threads share).
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def overrides(self) -> dict[str, str]:
        if not hasattr(self._local, "values"):
            self._local.values = {}
        return self._local.values

    def __getitem__(self, key: str) -> str:
        overrides = self.overrides
        if key in overrides:
            return overrides[key]
        return os.environ[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.overrides.get(key, os.environ.get(key, default))

    def __contains__(self, key: object) -> bool:
        return key in self.overrides or key in os.environ

    def __setitem__(self, key: str, value: str) -> None:
        self.overrides[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        return self.overrides.pop(key, default)


class _OsProxy:
    """The ``os`` module with ``environ`` swapped for a thread-local mapping."""

    def __init__(self, environ: _ThreadLocalEnviron):
        self.environ = environ

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


class _FrameworkRuntime:
    """One imported copy of a framework-style world image, shared by every task with the same code."""

    _cache: dict[str, "_FrameworkRuntime"] = {}
    _lock = threading.Lock()

    def __init__(self, world_dir: Path):
        alias = f"blobfish_image_framework_{_code_digest(world_dir)[:16]}"
        self.module = _load_module(world_dir / "server.py", alias)
        self.runtime_dir = world_dir / "runtime"
        self.environ = _ThreadLocalEnviron()
        proxy = _OsProxy(self.environ)
        self.servers: dict[str, Any] = {}
        spec_servers = json.loads((world_dir / "spec.json").read_text(encoding="utf-8"))["servers"]
        lib_dir = str(self.runtime_dir / "lib")
        with _import_lock:
            if lib_dir not in sys.path:
                sys.path.insert(0, lib_dir)
            try:
                for name in spec_servers:
                    spec = importlib.util.spec_from_file_location(
                        f"{alias}_{name}_server", self.runtime_dir / "servers" / f"{name}_server.py"
                    )
                    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                    with _no_bytecode():
                        spec.loader.exec_module(module)  # type: ignore[union-attr]
                    for attr in ("os", "_os"):
                        if getattr(module, attr, None) is os:
                            setattr(module, attr, proxy)
                    self.servers[name] = module.S
                framework = sys.modules.get("framework")
                if framework is not None and getattr(framework, "os", None) is os:
                    framework.os = proxy  # type: ignore[attr-defined]
            finally:
                try:
                    sys.path.remove(lib_dir)
                except ValueError:
                    pass
        vcode_spec = importlib.util.spec_from_file_location(f"{alias}_vcode", self.runtime_dir / "vcode.py")
        self.vcode = importlib.util.module_from_spec(vcode_spec)  # type: ignore[arg-type]
        with _no_bytecode():
            vcode_spec.loader.exec_module(self.vcode)  # type: ignore[union-attr]
        self.score_lock = threading.Lock()

    @classmethod
    def for_world(cls, world_dir: Path) -> "_FrameworkRuntime":
        key = _code_digest(world_dir)
        with cls._lock:
            runtime = cls._cache.get(key)
            if runtime is None:
                runtime = cls._cache[key] = cls(world_dir)
            return runtime


class FrameworkProfile(ImageProfile):
    name = "image_framework"

    def __init__(self, root: Path):
        world_dir = root / "environment" / "world"
        self.runtime = _FrameworkRuntime.for_world(world_dir)
        self.module = self.runtime.module
        self.servers = self.runtime.servers
        self.runtime_dir = self.runtime.runtime_dir
        super().__init__(root)

    def _tools(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for server_name, server in self.servers.items():
            for tool_name, (_, schema) in server.tools.items():
                specs.append(self._spec_from_mcp(f"{server_name}.{tool_name}", f"{server_name}__{tool_name}", schema))
        return specs

    def oracle_steps(self) -> tuple[dict[str, Any], ...] | None:
        walk = self.root / "solution" / "walk.json"
        if not walk.exists():
            return None
        steps = json.loads(walk.read_text(encoding="utf-8"))
        return tuple(
            {"tool": f"{step['server']}__{step['tool']}", "arguments": step.get("args") or {}} for step in steps
        )

    def open(self, workdir: Path) -> ImageSession:
        return FrameworkSession(self, workdir)


class FrameworkSession(ImageSession):
    def __init__(self, profile: FrameworkProfile, workdir: Path):
        super().__init__(profile, workdir)
        self.run_dir = workdir / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = profile.world_dir / "state"
        target = self.run_dir / "world.sqlite"
        if (state / "world.sqlite").exists():
            shutil.copyfile(state / "world.sqlite", target)
        else:
            with gzip.open(state / "world.sqlite.gz", "rb") as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        shutil.copyfile(state / "initial_state.json", self.run_dir / "initial_state.json")
        (self.run_dir / "trace.jsonl").write_text("", encoding="utf-8")
        self._env = {
            "WORLD_DB": str(target),
            "WORLD_NOW": str(profile.spec.get("world_now", "")),
            "WORLD_ROLE": str(profile.spec.get("world_role", "")),
            "TRACE_FILE": str(self.run_dir / "trace.jsonl"),
        }

    def _with_env(self, fn: Callable[[], Any]) -> Any:
        """Bind this session's world to the current thread for the duration of one call."""
        environ = self.profile.runtime.environ  # type: ignore[attr-defined]
        overrides = environ.overrides
        saved = dict(overrides)
        overrides.update(self._env)
        try:
            return fn()
        finally:
            overrides.clear()
            overrides.update(saved)

    def _invoke(self, tool: ToolSpec, arguments: dict[str, Any]) -> Any:
        server_name, tool_name = tool.name.split(".", 1)
        server = self.profile.servers[server_name]  # type: ignore[attr-defined]
        return self._with_env(lambda: server.call(tool_name, arguments))

    def score(self) -> Verdict:
        profile: FrameworkProfile = self.profile  # type: ignore[assignment]
        runtime = profile.runtime

        def run() -> dict[str, Any]:
            # build_report imports ``vcode`` by bare name; bind this runtime's copy for the call.
            with runtime.score_lock:
                previous = sys.modules.get("vcode")
                sys.modules["vcode"] = runtime.vcode
                try:
                    return profile.module.build_report(
                        profile.world_dir / "taskspec", self.run_dir, profile.spec["task_id"]
                    )
                finally:
                    if previous is None:
                        sys.modules.pop("vcode", None)
                    else:
                        sys.modules["vcode"] = previous

        try:
            report = self._with_env(run)
        except Exception as exc:
            raise VerifierError(f"{profile.spec.get('task_id')}: {exc}") from exc
        return _report_verdict(report, self.profile.name)


# --------------------------------------------------------------------------- CounselBench
class DocumentsProfile(ImageProfile):
    name = "image_documents"

    def __init__(self, root: Path):
        world_dir = root / "environment" / "world"
        alias = f"blobfish_image_documents_{_code_digest(world_dir)[:16]}"
        self.module = _load_module(world_dir / "world.py", alias, sibling_dir=world_dir)
        self.world_cls = _world_class(self.module, ("call_tool", "verify"))
        super().__init__(root)

    def _tools(self) -> list[ToolSpec]:
        definitions = getattr(self.module, "tool_definitions", None)
        items = definitions() if callable(definitions) else []
        return [self._spec_from_mcp(str(item["name"]), public_tool_name(str(item["name"])), item) for item in items]

    def oracle_steps(self) -> tuple[dict[str, Any], ...] | None:
        calls = self._reference().get("calls")
        if not isinstance(calls, list):
            return None
        return tuple(
            {"tool": public_tool_name(str(call["name"])), "arguments": call.get("arguments") or {}} for call in calls
        )

    def open(self, workdir: Path) -> ImageSession:
        world = self.world_cls(
            self.root / "environment" / "documents",
            workdir / "output",
            workdir / "state",
            self.world_dir / "spec.json",
        )
        return WorldObjectSession(self, workdir, world, grouped=False)


# --------------------------------------------------------------------------- SalesBench
_package_runtimes: dict[str, tuple[Any, Any]] = {}


class PackageProfile(ImageProfile):
    name = "image_package"

    def __init__(self, root: Path):
        world_dir = root / "environment" / "world"
        package_dir = next(
            p for p in sorted(world_dir.iterdir()) if p.is_dir() and (p / "runtime" / "world.py").exists()
        )
        self.package = package_dir.name
        key = f"{self.package}:{_code_digest(world_dir)}"
        with _import_lock:
            cached = _package_runtimes.get(key)
            if cached is None:
                for name in [k for k in sys.modules if k == self.package or k.startswith(self.package + ".")]:
                    sys.modules.pop(name)
                sys.path.insert(0, str(world_dir))
                try:
                    with _no_bytecode():
                        world_module = importlib.import_module(f"{self.package}.runtime.world")
                        contracts = importlib.import_module(f"{self.package}.contracts")
                finally:
                    sys.path.remove(str(world_dir))
                cached = _package_runtimes[key] = (world_module, contracts)
        self.world_module, self.contracts = cached
        self.world_cls = _world_class(self.world_module, ("call_tool", "verify"))
        super().__init__(root)

    def _tools(self) -> list[ToolSpec]:
        by_server = getattr(self.contracts, "TOOLS_BY_SERVER", None) or {}
        specs: list[ToolSpec] = []
        for server_name, tools in by_server.items():
            items = tools.values() if isinstance(tools, dict) else tools
            for item in items:
                name = str(item["name"])
                specs.append(self._spec_from_mcp(f"{server_name}.{name}", f"{server_name}__{name}", item))
        return specs

    def oracle_steps(self) -> tuple[dict[str, Any], ...] | None:
        calls = self._reference().get("calls")
        if not isinstance(calls, list):
            return None
        return tuple(
            {"tool": f"{call['server']}__{call['name']}", "arguments": call.get("arguments") or {}} for call in calls
        )

    def open(self, workdir: Path) -> ImageSession:
        world = self.world_cls(
            self.root / "environment" / "documents",
            workdir / "output",
            workdir / "state",
            self.world_dir / "spec.json",
            self.world_dir / "seed.json",
        )
        return WorldObjectSession(self, workdir, world, grouped=True)


# --------------------------------------------------------------------------- DevOpsBench
class CompanionProfile(ImageProfile):
    name = "image_companion"

    def __init__(self, root: Path):
        world_dir = root / "environment" / "world"
        alias = f"blobfish_image_companion_{_code_digest(world_dir)[:16]}"
        self.module = _load_module(world_dir / "server.py", alias)
        self.world_cls = _world_class(self.module, ("call_tool", "verify"))
        super().__init__(root)

    def _tools(self) -> list[ToolSpec]:
        items = json.loads((self.world_dir / "tools.json").read_text(encoding="utf-8"))
        return [self._spec_from_mcp(str(item["name"]), public_tool_name(str(item["name"])), item) for item in items]

    def oracle_steps(self) -> tuple[dict[str, Any], ...] | None:
        calls = self._reference().get("expected_calls")
        if not isinstance(calls, list):
            return None
        return tuple(
            {"tool": public_tool_name(str(call["tool"])), "arguments": call.get("args") or {}} for call in calls
        )

    def open(self, workdir: Path) -> ImageSession:
        world = self.world_cls(self.world_dir, workdir / "state", self.spec.get("task_id"))
        return WorldObjectSession(self, workdir, world, grouped=False)


class WorldObjectSession(ImageSession):
    """Session over a world object exposing ``call_tool`` and ``verify(token)``."""

    def __init__(self, profile: ImageProfile, workdir: Path, world: Any, *, grouped: bool):
        super().__init__(profile, workdir)
        self.world = world
        self.grouped = grouped

    def _invoke(self, tool: ToolSpec, arguments: dict[str, Any]) -> Any:
        if self.grouped:
            server_name, tool_name = tool.name.split(".", 1)
            return self.world.call_tool(server_name, tool_name, arguments)
        return self.world.call_tool(tool.name, arguments)

    def score(self) -> Verdict:
        try:
            report = self.world.verify(self.profile.token)
        except Exception as exc:
            raise VerifierError(f"{self.profile.spec.get('task_id')}: {exc}") from exc
        if not isinstance(report, dict):
            raise VerifierError("verify() returned no report")
        return _report_verdict(report, self.profile.name)

    def close(self) -> None:
        close = getattr(self.world, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort
                pass


def _world_class(module: ModuleType, required: tuple[str, ...]) -> type:
    candidates = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and all(callable(getattr(obj, attr, None)) for attr in required)
    ]
    local = [obj for obj in candidates if obj.__module__ == module.__name__]
    if local:
        candidates = local
    if len(candidates) != 1:
        raise ImportError(
            f"expected one world class with {required} in {module.__name__}, found {[c.__name__ for c in candidates]}"
        )
    return candidates[0]


_PROFILE_CLASSES: dict[str, type[ImageProfile]] = {
    "image_framework": FrameworkProfile,
    "image_documents": DocumentsProfile,
    "image_package": PackageProfile,
    "image_companion": CompanionProfile,
}


def profile_for(root: Path, kind: str) -> ImageProfile:
    """One profile object per task package (cached; tools and oracle are static per package)."""
    key = str(Path(root).resolve())
    with _profiles_lock:
        profile = _profiles.get(key)
        if profile is None:
            profile = _PROFILE_CLASSES[kind](Path(root))
            _profiles[key] = profile
        return profile
