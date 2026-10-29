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
"""The old import paths kept by `nemo_gym._moved_paths` must give the same objects as the new paths."""

import importlib
import importlib.util
import json
import pickle
import pkgutil
import re
import sys
import textwrap
import types
import typing
import warnings
from pathlib import Path
from unittest import mock

import pytest
import typing_extensions
from packaging.version import Version

import nemo_gym._moved_paths as moved_paths
from nemo_gym._moved_paths import MovedModuleFinder, MovedPath, _resolve, moved_names_getattr
from nemo_gym.package_info import __version__


@pytest.fixture
def moved_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A package `newpkg` holding code that moved out of `oldpkg`, with a finder for the old paths.

    `newpkg.sub.mod` was `oldpkg.mod`.
    `newpkg.capture`, a package, was `oldpkg.capture`.
    Each module counts how many times it ran in `newpkg.counter.RUNS`.
    """
    files = {
        "newpkg/__init__.py": "",
        "newpkg/counter.py": "RUNS = {}\n",
        "newpkg/sub/__init__.py": "",
        "newpkg/sub/mod.py": """
            from newpkg.counter import RUNS
            RUNS[__name__] = RUNS.get(__name__, 0) + 1

            class Thing:
                pass

            def helper():
                return "real"

            def call_helper():
                return helper()
        """,
        "newpkg/capture/__init__.py": "",
        "newpkg/capture/staging/__init__.py": "",
        "newpkg/capture/staging/records.py": """
            from newpkg.counter import RUNS
            RUNS[__name__] = RUNS.get(__name__, 0) + 1

            class Record:
                pass
        """,
        "oldpkg/__init__.py": "",
    }
    for rel, body in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    monkeypatch.syspath_prepend(str(tmp_path))
    moves = {
        "oldpkg.mod": MovedPath(new="newpkg.sub.mod", since="0.8.0"),
        "oldpkg.capture": MovedPath(new="newpkg.capture", since="0.8.0"),
    }
    finder = MovedModuleFinder(moves)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    yield
    for name in [name for name in sys.modules if name.split(".")[0] in ("oldpkg", "newpkg")]:
        del sys.modules[name]


def _runs() -> dict:
    return importlib.import_module("newpkg.counter").RUNS


class TestMovedModuleFinder:
    def test_old_path_is_the_new_module(self, moved_tree) -> None:
        with pytest.warns(DeprecationWarning):
            import oldpkg.mod as old
        import newpkg.sub.mod as new

        assert old is new
        assert sys.modules["oldpkg.mod"] is sys.modules["newpkg.sub.mod"]
        assert old.Thing is new.Thing
        assert _runs() == {"newpkg.sub.mod": 1}

    def test_new_path_first_then_old_path(self, moved_tree) -> None:
        import newpkg.sub.mod as new

        with pytest.warns(DeprecationWarning):
            old = importlib.import_module("oldpkg.mod")
        assert old is new
        assert _runs() == {"newpkg.sub.mod": 1}

    def test_from_import_of_names_and_of_the_module(self, moved_tree) -> None:
        with pytest.warns(DeprecationWarning):
            from oldpkg import mod
            from oldpkg.mod import Thing
        import newpkg.sub.mod as new

        assert mod is new
        assert Thing is new.Thing
        assert importlib.import_module("oldpkg").mod is new

    def test_submodule_of_moved_package_is_not_executed_twice(self, moved_tree) -> None:
        import newpkg.capture.staging.records as new

        with pytest.warns(DeprecationWarning):
            import oldpkg.capture.staging.records as old
            from oldpkg.capture.staging.records import Record
        assert old is new
        assert Record is new.Record
        assert _runs() == {"newpkg.capture.staging.records": 1}

    def test_old_package_imported_before_its_submodule(self, moved_tree) -> None:
        with pytest.warns(DeprecationWarning):
            import oldpkg.capture  # noqa: F401
            import oldpkg.capture.staging.records as old
        assert old is importlib.import_module("newpkg.capture.staging.records")
        assert _runs() == {"newpkg.capture.staging.records": 1}

    def test_mock_patch_through_old_path_patches_running_code(self, moved_tree) -> None:
        import newpkg.sub.mod as new

        with pytest.warns(DeprecationWarning), mock.patch("oldpkg.mod.helper", return_value="patched"):
            assert new.call_helper() == "patched"
        assert new.call_helper() == "real"

    def test_dotted_config_string_resolves(self, moved_tree) -> None:
        module_path, _, class_name = "oldpkg.capture.staging.records:Record".partition(":")
        with pytest.warns(DeprecationWarning):
            factory = getattr(importlib.import_module(module_path), class_name)
        assert factory is importlib.import_module("newpkg.capture.staging.records").Record

    def test_pickle_written_with_old_path_loads(self, moved_tree) -> None:
        import newpkg.sub.mod as new

        with pytest.warns(DeprecationWarning):
            loaded = pickle.loads(b"coldpkg.mod\nThing\n.")
        assert loaded is new.Thing
        assert pickle.loads(pickle.dumps(new.Thing())).__class__ is new.Thing

    def test_warning_names_paths_versions_and_importer(self, moved_tree) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module("oldpkg.mod")
        [warning] = caught
        message = str(warning.message)
        assert warning.category is DeprecationWarning
        assert "`oldpkg.mod` moved to `newpkg.sub.mod` in nemo_gym 0.8.0" in message
        assert "removed in nemo_gym 0.10.0" in message
        assert f"running {__version__}" in message
        assert warning.filename == __file__

    def test_warns_once_per_old_path(self, moved_tree) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module("oldpkg.mod")
            importlib.import_module("oldpkg.mod")
        assert len(caught) == 1

    def test_find_spec_of_old_path_points_at_the_new_file(self, moved_tree, tmp_path: Path) -> None:
        # find_spec imports the parent packages, which are old paths too.
        with pytest.warns(DeprecationWarning):
            spec = importlib.util.find_spec("oldpkg.capture.staging.records")
        assert spec.name == "oldpkg.capture.staging.records"
        assert Path(spec.origin) == tmp_path / "newpkg/capture/staging/records.py"
        assert importlib.util.find_spec("oldpkg.capture").submodule_search_locations is not None

    def test_other_modules_are_untouched(self, moved_tree) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            import newpkg.sub.mod  # noqa: F401
            import oldpkg  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("oldpkg.never_existed")

    def test_nemo_gym_installs_the_finder_first(self) -> None:
        assert isinstance(sys.meta_path[0], MovedModuleFinder)


class TestMovedNames:
    def test_old_name_is_the_new_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            moved_paths.MOVED_NAMES,
            "nemo_gym.fake_old",
            {
                "PackageInfoVersion": MovedPath(new="nemo_gym.package_info:__version__", since="0.8.0"),
                "VERSION": MovedPath(new="nemo_gym.package_info", since="0.8.0"),
            },
        )
        getter = moved_names_getattr("nemo_gym.fake_old")
        with pytest.warns(
            DeprecationWarning, match=r"`nemo_gym.fake_old.VERSION` moved to `nemo_gym.package_info.VERSION`"
        ):
            assert getter("VERSION") is importlib.import_module("nemo_gym.package_info").VERSION
        with pytest.warns(DeprecationWarning, match=r"removed in nemo_gym 0.10.0"):
            assert getter("PackageInfoVersion") == __version__

    def test_each_moved_name_warns_once_and_is_looked_up_on_every_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            moved_paths.MOVED_NAMES,
            "nemo_gym.fake_old",
            {
                "VERSION": MovedPath(new="nemo_gym.package_info", since="0.8.0"),
                "PackageInfoVersion": MovedPath(new="nemo_gym.package_info:__version__", since="0.8.0"),
            },
        )
        getter = moved_names_getattr("nemo_gym.fake_old")
        package_info = importlib.import_module("nemo_gym.package_info")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getter("VERSION") is package_info.VERSION
            monkeypatch.setattr(package_info, "VERSION", "patched")
            assert getter("VERSION") == "patched"
            getter("PackageInfoVersion")
        assert [str(w.message).split("`")[1] for w in caught] == [
            "nemo_gym.fake_old.VERSION",
            "nemo_gym.fake_old.PackageInfoVersion",
        ]

    def test_unknown_name_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
            moved_names_getattr("nemo_gym.fake_old")("nothing")

    @pytest.mark.parametrize(
        ("old_module", "new_module", "running_as"),
        [
            ("nemo_gym.fake_old", "nemo_gym.pkg.fake_new", "nemo_gym.pkg.fake_new"),
            ("nemo_gym.fake_oldpkg", "nemo_gym.pkg.fake_newpkg", "nemo_gym.pkg.fake_newpkg.types"),
        ],
    )
    def test_names_listed_under_a_module_that_moved_later_still_resolve(
        self, monkeypatch: pytest.MonkeyPatch, old_module: str, new_module: str, running_as: str
    ) -> None:
        # A module that served moved names can move itself: its entries stay under its old path, and its
        # `__getattr__`, now built with the new module name, must still find them.
        listed_as = old_module + running_as[len(new_module) :]
        monkeypatch.setitem(
            moved_paths.MOVED_NAMES, listed_as, {"VERSION": MovedPath(new="nemo_gym.package_info", since="0.8.0")}
        )
        monkeypatch.setitem(moved_paths.MOVED_MODULES, old_module, MovedPath(new=new_module, since="0.8.0"))
        getter = moved_names_getattr(running_as)
        with pytest.warns(DeprecationWarning, match=rf"`{re.escape(listed_as)}.VERSION` moved to"):
            assert getter("VERSION") is importlib.import_module("nemo_gym.package_info").VERSION
        with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
            getter("nothing")


class TestDeprecationWindow:
    def test_removed_in_is_two_minor_releases_after_since(self) -> None:
        assert MovedPath(new="x", since="0.8.0").removed_in == "0.10.0"
        assert MovedPath(new="x", since="1.3.2").removed_in == "1.5.0"

    def test_no_old_path_outlives_its_removal_release(self) -> None:
        """Fails on the version bump that reaches an entry's removal release: delete the old path then."""
        running = Version(Version(__version__).base_version)
        entries = [(old, moved) for old, moved in moved_paths.MOVED_MODULES.items()]
        entries += [
            (f"{module}.{name}", moved)
            for module, names in moved_paths.MOVED_NAMES.items()
            for name, moved in names.items()
        ]
        overdue = [
            f"{old} (removed in {moved.removed_in})" for old, moved in entries if running >= Version(moved.removed_in)
        ]
        assert not overdue, f"nemo_gym {__version__} must no longer serve these old paths: {overdue}"


IMPORT_SURFACE_PATH = Path(__file__).parent / "fixtures" / "import_surface.json"


_VALUE_MODULES = {"builtins", "pathlib", "re", "enum", "_contextvars", "datetime", "decimal", "uuid"}


def _surface_names(module: types.ModuleType) -> list[str]:
    """Names an importer of `module` can rely on: what it defines and what it re-exports from nemo_gym.

    Modules, and classes, functions, and objects imported from other libraries (`Any`, `environ`), are left out.
    Type aliases the module defines, such as `X = Union[A, B]`, are kept.
    """
    names = []
    for name, value in vars(module).items():
        if not name.isidentifier() or name.startswith("__") or isinstance(value, types.ModuleType):
            continue
        if isinstance(value, type) or (callable(value) and not _is_type_alias(value)):
            # Classes and functions count only if nemo_gym defines them.
            if not str(getattr(value, "__module__", "")).startswith("nemo_gym"):
                continue
        elif _is_type_alias(value):
            # An alias counts unless it is a name imported from typing itself, such as `Any` or `Optional`.
            if any(getattr(source, name, None) is value for source in (typing, typing_extensions)):
                continue
        elif name == "TYPE_CHECKING" or type(value).__module__.split(".")[0] not in _VALUE_MODULES:
            # Values count if they are plain data (strings, numbers, paths, patterns) or nemo_gym objects.
            if not type(value).__module__.startswith("nemo_gym"):
                continue
        names.append(name)
    return sorted(names)


def _is_type_alias(value: object) -> bool:
    return isinstance(value, types.UnionType) or type(value).__module__ in ("typing", "typing_extensions")


def build_import_surface(modules: list[str] | None = None) -> dict[str, list[str]]:
    """Import every `nemo_gym` module (or only `modules`) and record its names."""
    import nemo_gym

    if modules is None:
        modules = [
            info.name
            for info in pkgutil.walk_packages(nemo_gym.__path__, "nemo_gym.")
            if not info.name.endswith(".__main__")
        ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {name: _surface_names(importlib.import_module(name)) for name in ["nemo_gym", *modules]}


def _is_moved(module: str) -> bool:
    return _resolve(module, moved_paths.MOVED_MODULES) is not None or module in moved_paths.MOVED_NAMES


class TestImportSurface:
    """Every module path and name recorded before the reorganization must still resolve at its old path.

    `fixtures/import_surface.json` records each module's names before any move.
    Only modules the reorganization touches are checked name by name, so ordinary changes elsewhere are unaffected.
    A PR that moves or splits a module first refreshes that module's entry on its parent commit:
    `python tests/unit_tests/test_moved_paths.py <module> ...`.
    """

    @property
    def surface(self) -> dict[str, list[str]]:
        return json.loads(IMPORT_SURFACE_PATH.read_text())

    def test_surface_keeps_type_aliases_and_drops_imported_library_names(self) -> None:
        module = types.ModuleType("nemo_gym.fake_surface")

        class Defined:
            pass

        Defined.__module__ = module.__name__
        module.__dict__.update(
            {
                "Defined": Defined,
                "UnionAlias": typing.Union[Defined, int],
                "PipeAlias": Defined | None,
                "LIMIT": 3,
                "Any": typing.Any,
                "Optional": typing.Optional,
                "BaseModel": __import__("pydantic").BaseModel,
                "Cached[int]": Defined,
            }
        )
        assert _surface_names(module) == ["Defined", "LIMIT", "PipeAlias", "UnionAlias"]

    def test_every_recorded_module_still_imports_at_its_old_path(self) -> None:
        missing = []
        for module in self.surface:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    importlib.import_module(module)
            except ModuleNotFoundError as error:
                if error.name is not None and error.name.split(".")[0] != "nemo_gym":
                    continue  # an optional dependency this environment lacks
                missing.append(module)
        assert not missing, (
            f"These modules no longer import at their old paths: {missing}. A moved module needs a MOVED_MODULES "
            f"entry in nemo_gym/_moved_paths.py. If one was deleted on purpose, remove it from {IMPORT_SURFACE_PATH.name}."
        )

    def test_surface_records_the_modules_added_before_the_reorganization(self) -> None:
        # Recorded from the whole tree the moves start from, so later moves cannot drop one unnoticed.
        surface = self.surface
        for module in (
            "nemo_gym.verifiers",
            "nemo_gym.verifiers.files",
            "nemo_gym.verifiers.runner",
            "nemo_gym.tasks.harbor.entries",
            "nemo_gym.tasks.harbor.parameterized",
            "nemo_gym.tasks.harbor.publish",
            "nemo_gym.tasks.harbor.python_verifier",
        ):
            assert module in surface
        assert "text_file_equals" in surface["nemo_gym.verifiers.files"]

    def test_moved_modules_keep_every_recorded_name(self) -> None:
        missing = []
        for module, names in self.surface.items():
            if not _is_moved(module):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                old = importlib.import_module(module)
                missing += [f"{module}.{name}" for name in names if not hasattr(old, name)]
        assert not missing, (
            f"These names no longer resolve at their old paths: {missing}. A name split out of a module needs a "
            "MOVED_NAMES entry in nemo_gym/_moved_paths.py."
        )

    def test_moved_module_is_the_same_object_under_both_names(self) -> None:
        for old, moved in moved_paths.MOVED_MODULES.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert importlib.import_module(old) is importlib.import_module(moved.new)

    def test_moved_names_are_the_same_objects_at_both_paths(self) -> None:
        for module, names in moved_paths.MOVED_NAMES.items():
            for name, moved in names.items():
                new_module, _, new_name = moved.new.partition(":")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    old_value = getattr(importlib.import_module(module), name)
                assert old_value is getattr(importlib.import_module(new_module), new_name or name)


# What NeMo RL imports from nemo_gym (nemo_rl/, tests/, examples/), plus the class it names in config.
NEMO_RL_IMPORTS = {
    "nemo_gym": ["config_types"],
    "nemo_gym.cli": ["GlobalConfigDictParserConfig", "RunHelper"],
    "nemo_gym.global_config": ["GlobalConfigDictParser", "GlobalConfigDictParserConfig", "get_global_config_dict"],
    "nemo_gym.rollout_collection": ["RolloutCollectionHelper"],
    "nemo_gym.server_utils": ["BaseServerConfig", "HEAD_SERVER_KEY_NAME", "ServerClient", "SimpleServer"],
    "nemo_gym.token_id_capture": [
        "NG_CAPTURE_FIELD",
        "NG_COMMIT_COORDS_FIELD",
        "UNCOMMITTED_CALL_REASON",
        "control_routes",
    ],
    "nemo_gym.token_id_capture.adapters.megatron": ["MegatronCaptureAdapter"],
    "nemo_gym.token_id_capture.adapters.vllm": ["MEDIA_SPANS_FIELD", "VLLMCaptureAdapter"],
    "nemo_gym.token_id_capture.control_routes": ["MAX_LEDGER_BATCH"],
    "nemo_gym.token_id_capture.external_capture": ["make_external_capture_handler"],
    "nemo_gym.token_id_capture.lineage": ["FileLineageStore", "InMemoryLineageStore"],
    "nemo_gym.token_id_capture.sink": ["CaptureContext", "reset_token_sink", "set_token_sink"],
    "nemo_gym.token_id_capture.staging": ["install_capture", "resolve_terminal"],
    "nemo_gym.token_id_capture.staging.capture": ["ActiveCall", "CaptureError", "RolloutTokenCapture"],
    "nemo_gym.token_id_capture.staging.conformance": ["assert_golden_vectors"],
    "nemo_gym.token_id_capture.staging.digest": [
        "EXTRAS_DIGEST_VERSION",
        "compute_chain_hash",
        "compute_extras_digest",
        "compute_staging_digest",
        "hash_token_ids",
    ],
    "nemo_gym.token_id_capture.staging.protocols": ["StagingSink", "StagingSource"],
    "nemo_gym.token_id_capture.staging.rebuild": ["RebuildError", "ReceiptVerificationError", "verify_and_linearize"],
    "nemo_gym.token_id_capture.staging.records": [
        "CallRecord",
        "CaptureAdmission",
        "RolloutReceipt",
        "StageResult",
        "StagedCallBaseSnapshot",
        "StagedCallRecord",
        "StagedCallSnapshot",
        "staging_key",
    ],
    "nemo_gym.token_id_capture.staging.routes": ["classify_route_span"],
    "nemo_gym.token_id_capture.staging.terminal": ["select_terminal_call"],
}


class TestNemoRLUsage:
    @pytest.mark.parametrize("module", sorted(NEMO_RL_IMPORTS))
    def test_from_imports_resolve(self, module: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            namespace: dict = {}
            exec(f"from {module} import {', '.join(NEMO_RL_IMPORTS[module])}", namespace)
        assert all(namespace[name] is not None for name in NEMO_RL_IMPORTS[module])

    def test_lineage_store_named_in_config_resolves(self) -> None:
        # NeMo RL sets token_id_capture.lineage_store to this string.
        module_path, _, class_name = "nemo_gym.token_id_capture.lineage:FileLineageStore".partition(":")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            factory = getattr(importlib.import_module(module_path), class_name)
        assert isinstance(factory, type)

    def test_patching_run_helper_on_the_cli_package_is_seen_by_importers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import nemo_gym.cli as gym_cli
            import nemo_gym.rollout_collection as gym_rollout_collection

        fake_helper, fake_collection = object(), object()
        # `RunHelper` reaches nemo_gym.cli through the CLI refactor's `__getattr__` shim. Patch the module's namespace,
        # so undoing the patch removes the name again; restoring it with setattr would leave it set for later tests.
        monkeypatch.setitem(vars(gym_cli), "RunHelper", fake_helper)
        monkeypatch.setattr(gym_rollout_collection, "RolloutCollectionHelper", fake_collection)
        namespace: dict = {}
        exec(
            "from nemo_gym.cli import RunHelper\nfrom nemo_gym.rollout_collection import RolloutCollectionHelper",
            namespace,
        )
        assert namespace["RunHelper"] is fake_helper
        assert namespace["RolloutCollectionHelper"] is fake_collection


if __name__ == "__main__":
    # Refresh the recorded import surface: all modules, or only the modules named on the command line.
    surface = json.loads(IMPORT_SURFACE_PATH.read_text()) if len(sys.argv) > 1 else {}
    surface.update(build_import_surface(sys.argv[1:] or None))
    IMPORT_SURFACE_PATH.write_text(
        "{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(surface.items())) + "\n}\n"
    )
