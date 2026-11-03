# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static sandbox policy of the Harbor path: task code runs in the sandbox, and the sandbox stays there.

The rules are derived from the code as it stands and checked on its syntax trees, so a change that moves
task execution onto the host or opens the sandbox to another caller fails here before it ships:

R1  Host processes. No module under ``environment_servers/harbor`` (``tests/`` aside) spawns a host process:
    none of ``subprocess.*``, ``os.system/popen/exec*/spawn*/fork*/posix_spawn*``,
    ``asyncio.create_subprocess_*``, ``loop.subprocess_exec/shell``, ``pty.spawn/fork`` or ``multiprocessing``
    is called, under any name: calls are matched after resolving the module's import aliases (``import
    subprocess as sp; sp.run`` is ``subprocess.run``, ``from os import execvp as e; e()`` is ``os.execvp``), a
    ``from`` import of any of those functions is refused, and so is an ``import`` of ``subprocess``,
    ``multiprocessing`` or ``pty`` outside the spawning modules, unless it sits under ``if TYPE_CHECKING:`` (never
    executed; ``build_plan`` names ``subprocess.CompletedProcess`` that way). Under ``nemo_gym/data/tasks/harbor`` only
    the dataset tooling spawns, each a fixed program: ``hub.py`` and ``manifest.py`` run ``git`` (argv literals
    whose first element is ``"git"``), ``build_tools.py`` runs the configured image build tool (``self.tool``,
    one of ``BUILD_TOOLS``) or ``docker`` through its one ``subprocess.run`` in ``_run_default``, and no spawn
    passes ``shell=True``. The checks are self-tested on snippets that must be flagged. Task files (``test.sh``,
    ``solve.sh``, Dockerfile ``RUN`` lines, collect hooks, entrypoints) can therefore run only where a sandbox
    runs them.
R2  No dynamic code loading in either tree: the builtins ``exec``, ``eval``, ``compile`` and ``__import__`` (bare
    or through ``builtins``), ``importlib.import_module`` and ``runpy`` are never called, under any name: calls are
    classified after resolving the module's import aliases, as in R1, so ``import importlib as il;
    il.import_module`` is flagged and ``from re import compile; compile(x)`` is ``re.compile`` (which, like
    ``sandbox.exec``, is another callable and not one of these). Self-tested on snippets that must be flagged and
    snippets that must pass.
R3  The verifier wrapper runs only inside a sandbox: every call that receives ``verifier_command(...)`` is the
    ``.exec`` method of a sandbox.
R4  HTTP surface. The Harbor environment server registers no route of its own (no ``app.<verb>``,
    ``add_api_route``, ``APIRouter`` or ``include_router`` under ``environment_servers/harbor``); its routes are the
    base's ``/run`` and ``/aggregate_metrics`` plus FastAPI's documentation routes, and neither request model has
    a field that names a command to run. The sandbox descriptor (``SandboxAccess``) leaves the server only in the
    harness's session seed: ``sandbox_access=`` is passed exactly once, inside the ``POST /v1/agent_sessions`` to
    the configured agent server; ``serialize()`` is called only by ``_sandbox_access``; and no response model of
    the server carries it.
R5  The host-execution provider (``local``) is never selected by a shipped Harbor config under
    ``environment_servers/harbor/configs`` or ``responses_api_agents/harbor*/configs``: the server runs a task on the
    top-level provider block its ``sandbox_provider`` names (``gpu_sandbox_provider`` for GPU tasks; the default
    name is ``sandbox``), so no block a config names in either key, under whatever name, has a ``local`` key, and
    neither has any ``sandbox*``-named block (the name the shipped provider configs use). Self-tested on synthetic
    configs that select a ``local`` block under another name, directly and from a nested server block.
"""

import ast
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
ENV_SERVER = ROOT / "environment_servers" / "harbor"
TASKS = ROOT / "nemo_gym" / "data" / "tasks" / "harbor"

# R1: the modules allowed to spawn, and the programs each may run (``self.tool`` is the configured build tool).
SPAWNING_MODULES = {
    TASKS / "hub.py": {"git"},
    TASKS / "manifest.py": {"git"},
    TASKS / "build_tools.py": {"docker", "buildah", "podman", "self.tool"},
}
SPAWN_PREFIXES = ("subprocess.", "multiprocessing.", "os.exec", "os.spawn", "os.posix_spawn")
SPAWN_NAMES = {
    "os.system",
    "os.popen",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
    "pty.fork",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
}
SPAWN_ATTRS = {"create_subprocess_exec", "create_subprocess_shell", "subprocess_exec", "subprocess_shell"}
SPAWN_IMPORT_MODULES = {"subprocess", "multiprocessing", "pty"}
# The functions a `from` import may not name: these, plus every `os.exec*`, `os.spawn*` and `os.posix_spawn*`.
SPAWN_IMPORT_NAMES = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "fork"),
    ("os", "forkpty"),
    ("pty", "spawn"),
    ("pty", "fork"),
    ("asyncio", "create_subprocess_exec"),
    ("asyncio", "create_subprocess_shell"),
}
SPAWN_OS_PREFIXES = ("exec", "spawn", "posix_spawn")
TYPE_CHECKING_GUARDS = {"TYPE_CHECKING", "typing.TYPE_CHECKING"}
# R2
DYNAMIC_CODE = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "importlib.import_module",
    "runpy.run_path",
    "runpy.run_module",
}
DYNAMIC_CODE |= {f"builtins.{name}" for name in ("exec", "eval", "compile", "__import__")}
# R4
ROUTE_REGISTRATION = {"add_api_route", "include_router", "APIRouter", "api_route", "websocket", "websocket_route"}
HTTP_VERBS = {"get", "post", "put", "delete", "patch", "options", "head"}
COMMAND_FIELDS = {"command", "commands", "argv", "cmd", "script", "stdin", "shell"}
# R5: the server settings that name the top-level provider block a task runs on (`HarborEnvironmentServerConfig`,
# `HarborSandboxMixin._provider_ref` and `_sidecar_provider_ref`).
PROVIDER_REF_KEYS = {"sandbox_provider", "gpu_sandbox_provider"}


def harbor_modules() -> list[Path]:
    modules = [p for p in ENV_SERVER.glob("*.py")] + [p for p in TASKS.rglob("*.py")]
    assert modules, "the Harbor path moved; update the policy test"
    return sorted(modules)


def dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, ``None`` for anything else (a call result, a subscript)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node, dotted(node.func)


def is_spawn(name: str | None, func: ast.AST) -> bool:
    if name is None:
        return isinstance(func, ast.Attribute) and func.attr in SPAWN_ATTRS
    return name in SPAWN_NAMES or name.startswith(SPAWN_PREFIXES) or name.rsplit(".", 1)[-1] in SPAWN_ATTRS


def is_spawn_import(module: str, name: str) -> bool:
    """Whether ``from <module> import <name>`` brings a spawning module or function into scope."""
    if module in SPAWN_IMPORT_MODULES or (module, name) in SPAWN_IMPORT_NAMES:
        return True
    return module == "os" and name.startswith(SPAWN_OS_PREFIXES)


def import_aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> the dotted name it stands for, from every absolute import of the module: ``import subprocess
    as sp`` gives ``sp: subprocess``, ``from os import execvp as e`` gives ``e: os.execvp``."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def canonical(name: str | None, aliases: dict[str, str]) -> str | None:
    """The dotted name with its first component resolved through the module's import aliases."""
    if name is None:
        return None
    head, _, rest = name.partition(".")
    head = aliases.get(head, head)
    return f"{head}.{rest}" if rest else head


def type_checking_guarded(tree: ast.AST) -> set[ast.AST]:
    """Every node under an ``if TYPE_CHECKING:`` body: never executed, so an import there spawns nothing."""
    guarded: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and dotted(node.test) in TYPE_CHECKING_GUARDS:
            for statement in node.body:
                guarded.update(ast.walk(statement))
    return guarded


def first_program(call: ast.Call) -> str | None:
    """What the spawned argv starts with: a string literal, ``self.tool``, or ``None`` when it is not a literal list."""
    if not call.args or not isinstance(call.args[0], ast.List) or not call.args[0].elts:
        return None
    head = call.args[0].elts[0]
    if isinstance(head, ast.Constant) and isinstance(head.value, str):
        return head.value
    return dotted(head)


def argv_literals(tree: ast.AST):
    """The argv lists ``build_tools`` composes: assigned to ``argv``, returned by an ``*_argv`` method, or passed first
    to its runner seam (``self._run``, ``self._check``, ``run``, ``_run_default``)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "argv" for t in node.targets):
            if isinstance(node.value, ast.List):
                yield node.value
        elif isinstance(node, ast.FunctionDef) and node.name.endswith("_argv"):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.List):
                    yield inner.value
        elif isinstance(node, ast.Call) and dotted(node.func) in {"self._run", "self._check", "run", "_run_default"}:
            if node.args and isinstance(node.args[0], ast.List):
                yield node.args[0]


def r1_offenders(tree: ast.AST, label: str, allowed: set[str] | None, *, argv_seam: bool = False) -> list[str]:
    """R1 on one module: ``allowed`` is the set of programs it may spawn (``None``: it may not spawn at all),
    ``argv_seam`` whether a non-literal argv is acceptable (``build_tools``, whose composed argvs are checked apart)."""
    offenders: list[str] = []
    aliases = import_aliases(tree)
    guarded = type_checking_guarded(tree)
    for node in ast.walk(tree):
        if node in guarded:
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            if any(is_spawn_import(top, alias.name) for alias in node.names):
                offenders.append(f"{label}:{node.lineno} imports a spawn function by name")
        elif isinstance(node, ast.Import) and allowed is None:
            if any(alias.name.split(".")[0] in SPAWN_IMPORT_MODULES for alias in node.names):
                offenders.append(f"{label}:{node.lineno} imports a spawning module")
    for call, name in calls(tree):
        name = canonical(name, aliases)
        if not is_spawn(name, call.func):
            continue
        where = f"{label}:{call.lineno} {name or ast.dump(call.func)}"
        if allowed is None:
            offenders.append(f"{where} spawns a host process")
            continue
        if any(kw.arg == "shell" for kw in call.keywords):
            offenders.append(f"{where} passes shell=")
        program = first_program(call)
        if program is not None and program not in allowed:
            offenders.append(f"{where} runs {program!r}, not one of {sorted(allowed)}")
        elif program is None and not argv_seam:
            offenders.append(f"{where} spawns an argv that is not a literal list")
    return offenders


def test_r1_no_host_process_spawning_outside_the_dataset_tooling():
    offenders: list[str] = []
    for path in harbor_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        label = str(path.relative_to(ROOT))
        offenders += r1_offenders(tree, label, SPAWNING_MODULES.get(path), argv_seam=path.name == "build_tools.py")
    assert offenders == []


# Code the R1 check must flag (and what it must say), whichever name the spawn goes by; the rule would regress
# silently otherwise, since the tree it guards contains none of these.
R1_MUST_FLAG = {
    "aliased module": ("import subprocess as sp\nsp.run(['ls'])\n", ["imports a spawning module", "spawns a host"]),
    "from os exec": ("from os import execvp, posix_spawn\n", ["imports a spawn function by name"]),
    "os alias": ("import os as o\no.system('ls')\n", ["spawns a host process"]),
    "os spawn": ("import os\nos.spawnvp(os.P_WAIT, 'ls', ['ls'])\n", ["spawns a host process"]),
    "from os alias": ("from os import execvp as e\ne('ls', ['ls'])\n", ["imports a spawn", "spawns a host"]),
    "aliased coroutine": (
        "from asyncio import create_subprocess_exec as start\nasync def f():\n    await start('x')\n",
        ["imports a spawn function by name", "spawns a host process"],
    ),
    "loop seam": (
        "import asyncio\nasync def f():\n    await asyncio.get_running_loop().subprocess_exec(None, 'x')\n",
        ["spawns a host"],
    ),
    "bare pty": ("import pty\n", ["imports a spawning module"]),
    "multiprocessing alias": ("import multiprocessing as mp\nmp.Process(target=print).start()\n", ["spawns a host"]),
    "from subprocess alias": ("from subprocess import Popen as P\nP(['x'])\n", ["imports a spawn", "spawns a host"]),
    "pty fork alias": ("from pty import fork as f\nf()\n", ["imports a spawn", "spawns a host"]),
}
R1_MUST_ACCEPT = {
    "type-only import": (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import subprocess\n"
        "def f(c: 'subprocess.CompletedProcess') -> None: ...\n"
    ),
    "typing-qualified guard": "import typing\nif typing.TYPE_CHECKING:\n    import subprocess as sp\n",
    "not a spawn": "import re, shutil\nre.compile('x'); shutil.which('git'); sandbox.exec('ls')\n",
    "relative import": "from .runner import run\nrun(['x'])\n",
}


@pytest.mark.parametrize("case", sorted(R1_MUST_FLAG))
def test_r1_flags_spawns_under_any_name(case: str):
    source, expected = R1_MUST_FLAG[case]
    offenders = r1_offenders(ast.parse(source), "snippet", None)
    assert offenders, case
    for fragment in expected:
        assert any(fragment in offender for offender in offenders), (case, offenders)


@pytest.mark.parametrize("case", sorted(R1_MUST_ACCEPT))
def test_r1_accepts_type_only_imports_and_non_spawns(case: str):
    assert r1_offenders(ast.parse(R1_MUST_ACCEPT[case]), "snippet", None) == []


def test_r1_holds_spawning_modules_to_their_programs():
    allowed = {"git"}
    assert r1_offenders(ast.parse("import subprocess\nsubprocess.run(['git', 'fetch'])\n"), "s", allowed) == []
    offenders = r1_offenders(
        ast.parse(
            "import subprocess as sp\nsp.run(['curl', 'x'])\nsp.run(['git'], shell=True)\nsp.run(argv)\n"
            "from subprocess import check_call\n"
        ),
        "s",
        allowed,
    )
    assert [o.partition(" ")[2] for o in offenders] == [
        "imports a spawn function by name",
        "subprocess.run runs 'curl', not one of ['git']",
        "subprocess.run passes shell=",
        "subprocess.run spawns an argv that is not a literal list",
    ]


def test_r1_build_tools_runs_only_the_build_tool_and_docker():
    path = TASKS / "build_tools.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    spawns = [(call, name) for call, name in calls(tree) if is_spawn(name, call.func)]
    assert [name for _, name in spawns] == ["subprocess.run"], "build_tools has one runner seam, `_run_default`"
    (seam,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_default"]
    assert any(c is spawns[0][0] for c in ast.walk(seam)), "the subprocess.run call lives in _run_default"
    heads = sorted({first_program_of_list(lst) for lst in argv_literals(tree)}, key=str)
    assert heads and set(heads) <= SPAWNING_MODULES[path] | {"*argv"}, heads
    # The build tool itself is one of a fixed set, validated where the configuration is read.
    from nemo_gym.data.tasks.harbor.build_plan import BUILD_TOOLS

    assert set(BUILD_TOOLS) <= {"docker", "buildah", "podman"}
    options = (TASKS / "build_options.py").read_text()
    assert "tool not in BUILD_TOOLS" in options


def first_program_of_list(node: ast.List) -> str | None:
    """The program a literal argv starts with; ``"*argv"`` when it extends a list named ``argv`` (checked on its own)."""
    head = node.elts[0] if node.elts else None
    if isinstance(head, ast.Constant) and isinstance(head.value, str):
        return head.value
    if isinstance(head, ast.Starred) and isinstance(head.value, ast.Name) and head.value.id == "argv":
        return "*argv"
    return dotted(head) if head is not None else None


def r2_offenders(paths: list[Path]) -> list[str]:
    """The dynamic code loading calls of the given modules, each call classified by the callable it resolves to
    through the module's import aliases (``import importlib as il; il.import_module`` is
    ``importlib.import_module``, ``from re import compile; compile`` is ``re.compile``), not by its local spelling."""
    offenders = []
    for path in paths:
        label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = import_aliases(tree)
        for call, name in calls(tree):
            name = canonical(name, aliases)
            if name in DYNAMIC_CODE or (name and name.startswith("runpy.")):
                offenders.append(f"{label}:{call.lineno} calls {name}")
    return offenders


def test_r2_no_dynamic_code_loading():
    assert r2_offenders(harbor_modules()) == []


R2_MUST_FLAG = {
    "exec": "exec(source)",
    "eval": "eval(expr)",
    "compile": "compile(source, 'task', 'exec')",
    "__import__": "__import__('task_module')",
    "builtins-alias": "import builtins as b\nb.exec(source)",
    "builtins-from": "from builtins import eval as ev\nev(expr)",
    "importlib": "import importlib\nimportlib.import_module('task_module')",
    "importlib-alias": "import importlib as il\nil.import_module('task_module')",
    "importlib-from": "from importlib import import_module\nimport_module('task_module')",
    "runpy-alias": "import runpy as rp\nrp.run_module('task_module')",
    "runpy-from-alias": "from runpy import run_path as run\nrun('task.py')",
}
R2_MUST_ACCEPT = {
    "re-compile": "import re\nre.compile(x)",
    "re-alias": "import re as r\nr.compile(x)",
    "re-from": "from re import compile\ncompile(x)",
    "sandbox-exec": "await sandbox.exec('ls')",
    "method-eval": "model.eval()",
}


@pytest.mark.parametrize("case", sorted(R2_MUST_FLAG))
def test_r2_flags_dynamic_code_loading_under_any_name(tmp_path, case: str):
    snippet = tmp_path / f"{case}.py"
    snippet.write_text(R2_MUST_FLAG[case] + "\n")
    offenders = r2_offenders([snippet])
    assert len(offenders) == 1 and offenders[0].startswith(f"{case}.py:"), offenders


@pytest.mark.parametrize("case", sorted(R2_MUST_ACCEPT))
def test_r2_accepts_other_callables_of_the_same_spelling(tmp_path, case: str):
    snippet = tmp_path / f"{case}.py"
    snippet.write_text("async def f():\n    " + R2_MUST_ACCEPT[case].replace("\n", "\n    ") + "\n")
    assert r2_offenders([snippet]) == []


def test_r3_the_verifier_wrapper_is_only_handed_to_a_sandbox_exec():
    seen = 0
    for path in ENV_SERVER.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for call, name in calls(tree):
            if not any(isinstance(a, ast.Call) and dotted(a.func) == "verifier_command" for a in call.args):
                continue
            seen += 1
            assert isinstance(call.func, ast.Attribute) and call.func.attr == "exec", (
                f"{path.relative_to(ROOT)}:{call.lineno} hands verifier_command(...) to {name}, not a sandbox exec"
            )
    assert seen >= 2, "the shared and the separate verifier each run test.sh through one sandbox exec"


def test_r4_no_routes_of_its_own_and_only_the_base_routes():
    for path in ENV_SERVER.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for call, name in calls(tree):
            attr = call.func.attr if isinstance(call.func, ast.Attribute) else name
            if attr in ROUTE_REGISTRATION:
                raise AssertionError(f"{path.relative_to(ROOT)}:{call.lineno} registers a route: {name}")
            if name and name.split(".", 1)[0] in {"app", "router"} and attr in HTTP_VERBS:
                raise AssertionError(f"{path.relative_to(ROOT)}:{call.lineno} registers a route: {name}")

    # The live half (the served routes and their bodies) runs with the server's own test suite:
    # environment_servers/harbor/tests/test_sandbox_policy_live.py, so the core coverage run never imports the
    # server package.


def test_r4_the_sandbox_descriptor_only_reaches_the_harness_session_seed():
    # The one `sandbox_access=` keyword of the whole server package, whichever module it sits in.
    parents: dict[ast.AST, ast.AST] = {}
    accesses: list[tuple[str, ast.keyword]] = []
    for path in sorted(ENV_SERVER.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents.update({child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)})
        accesses += [
            (path.name, n) for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "sandbox_access"
        ]
    assert [(name, kw.lineno > 0) for name, kw in accesses] == [("app.py", True)], (
        f"sandbox_access is handed over exactly once, in app.py: {[(n, kw.lineno) for n, kw in accesses]}"
    )
    node: ast.AST = accesses[0][1]
    post = None
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.Call) and dotted(node.func) == "self.server_client.post":
            post = node
            break
    assert post is not None, "the hand-over is inside a server_client.post"
    keywords = {kw.arg: kw.value for kw in post.keywords}
    assert isinstance(keywords["url_path"], ast.Constant) and keywords["url_path"].value == "/v1/agent_sessions"
    assert dotted(keywords["server_name"]) == "self.config.agent_server.name"

    serialize_sites = []
    for path in ENV_SERVER.glob("*.py"):
        module = ast.parse(path.read_text(), filename=str(path))
        for function in [n for n in ast.walk(module) if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef)]:
            for call, name in calls(function):
                if isinstance(call.func, ast.Attribute) and call.func.attr == "serialize":
                    serialize_sites.append((path.name, function.name))
    assert serialize_sites == [("session.py", "_sandbox_access")]

    # The model walk over HarborEpisodeResponse (nothing it carries names a sandbox descriptor) lives in
    # environment_servers/harbor/tests/test_sandbox_policy_live.py, beside the server it imports.


def r5_offenders(loaded) -> list[str]:
    """The ``local`` provider blocks of one loaded config: a top-level block that any ``sandbox_provider`` or
    ``gpu_sandbox_provider`` value in the document names (the block the server resolves, wherever the server's own
    settings sit in the document), and any ``sandbox*``-named block anywhere, each reported once by its path. The
    ``local`` key counts whatever its value: the provider registry turns null options into an empty mapping."""
    found: dict[str, None] = {}
    refs: list[str] = []
    stack = [((), loaded)]
    while stack:
        trail, value = stack.pop()
        if isinstance(value, dict):
            if "local" in value and trail and trail[-1].startswith("sandbox"):
                found[f"{'.'.join(trail)}.local"] = None
            for key, item in value.items():
                if key in PROVIDER_REF_KEYS and isinstance(item, str):
                    refs.append(item)
                stack.append(((*trail, str(key)), item))
        elif isinstance(value, list):
            stack.extend((trail, item) for item in value)
    if isinstance(loaded, dict):
        for ref in refs:
            block = loaded.get(ref)
            if isinstance(block, dict) and "local" in block:
                found[f"{ref}.local"] = None
    return list(found)


@pytest.mark.parametrize(
    ("config", "flagged"),
    [
        ({"sandbox_provider": "host_exec", "host_exec": {"local": {}}}, ["host_exec.local"]),
        (
            {
                "harbor_environment_server": {
                    "environment_servers": {"harbor": {"sandbox_provider": "sandbox", "gpu_sandbox_provider": "gpu"}}
                },
                "sandbox": {"docker": {}},
                "gpu": {"local": {"work_dir": "/tmp"}},
            },
            ["gpu.local"],
        ),
        ({"sandbox_provider": "sandbox", "sandbox": {"local": {}}}, ["sandbox.local"]),
        ({"nested": {"sandbox_gpu": {"local": {}}}}, ["nested.sandbox_gpu.local"]),
        # A null-valued provider block (YAML ``sandbox:\n  local:``) selects host execution as a mapping does, with
        # no explicit reference needed: ``sandbox`` is the server's default provider name.
        ({"sandbox": {"local": None}}, ["sandbox.local"]),
        ({"nested": {"sandbox_gpu": {"local": None}}}, ["nested.sandbox_gpu.local"]),
        ({"spare": {"local": None}}, []),
        ({"sandbox_provider": "sandbox", "sandbox": {"docker": {}}, "spare": {"opensandbox": {}}}, []),
        ({"sandbox_provider": "${sandbox}", "sandbox": {"opensandbox": {}}}, []),
        ([{"sandbox_provider": "x"}], []),
    ],
)
def test_r5_self_test(config, flagged):
    assert r5_offenders(config) == flagged


def test_r5_no_shipped_harbor_config_selects_the_host_execution_provider():
    configs = list((ENV_SERVER / "configs").glob("*.yaml"))
    configs += [
        p for agent in (ROOT / "responses_api_agents").glob("harbor*") for p in (agent / "configs").glob("*.yaml")
    ]
    assert configs
    offenders = []
    for path in configs:
        try:
            loaded = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        offenders += [f"{path.relative_to(ROOT)}: {found}" for found in r5_offenders(loaded)]
    assert offenders == []
