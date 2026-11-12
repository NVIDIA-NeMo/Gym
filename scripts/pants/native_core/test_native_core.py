# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard complete native target selection and uv/PEX dependency equivalence."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from collect import targets
from prepare import requirement_pins, snapshot_git, validate_lock, write_build_configuration
from run import combine_junit
from runtime import (
    exclude_instrumentation,
    expose_editable_to_children,
    restore_build_directories,
    restore_symlinks,
    restore_tracked_modes,
)


class NativeCoreTest(unittest.TestCase):
    def test_editable_build_directory_follows_sampled_project_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "pyproject.toml"
            project.write_text('[tool.distutils.egg_info]\negg_base = "cache/editable"\n')
            restore_build_directories(root)
            self.assertTrue((root / "cache/editable").is_dir())
            (root / "cache/editable/existing").write_text("preserved")
            restore_build_directories(root)
            self.assertEqual((root / "cache/editable/existing").read_text(), "preserved")
            project.write_text('[tool.distutils.egg_info]\negg_base = "../outside"\n')
            with self.assertRaisesRegex(ValueError, "inside"):
                restore_build_directories(root)

    def test_child_python_outside_checkout_uses_relocated_editable_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "sandbox"
            package = root / "native_fixture/cli"
            package.mkdir(parents=True)
            (root / "native_fixture/__init__.py").touch()
            (package / "__init__.py").touch()
            (package / "main.py").write_text("def main():\n    return 'sampled entrypoint'\n")
            installed = root / "native_fixture-1.2.dist-info"
            installed.mkdir()
            (installed / "METADATA").write_text("Name: native-fixture\nVersion: 1.2\n")
            (installed / "entry_points.txt").write_text("[console_scripts]\ngym = native_fixture.cli.main:main\n")
            distribution = base / "declared-distribution"
            distribution.mkdir()
            original = str(base / "unavailable-source")
            (distribution / "editable.json").write_text(
                json.dumps({"source_root": original, "hook": "editable.pth", "finder": "fixture_finder.py"})
            )
            (distribution / "editable.pth").write_text("import fixture_finder; fixture_finder.install()\n")
            (distribution / "fixture_finder.py").write_text(
                "import sys\nfrom importlib.util import spec_from_file_location\n"
                f"MAPPING = {{'native_fixture': {original + '/native_fixture'!r}}}\n"
                "class Finder:\n"
                "    @classmethod\n"
                "    def find_spec(cls, fullname, path=None, target=None):\n"
                "        if fullname in MAPPING:\n"
                "            return spec_from_file_location(fullname, MAPPING[fullname] + '/__init__.py')\n"
                "def install():\n    sys.meta_path.append(Finder)\n"
            )
            extra = base / "existing-pythonpath"
            extra.mkdir()
            (extra / "preserved_module.py").write_text("value = 42\n")
            # Pants installs the genuine distribution to supply its console
            # scripts. The sampled editable package must still win over it.
            wheel_package = extra / "native_fixture/cli"
            wheel_package.mkdir(parents=True)
            (extra / "native_fixture/__init__.py").touch()
            (wheel_package / "__init__.py").touch()
            (wheel_package / "main.py").write_text("def main():\n    return 'wrong wheel copy'\n")
            with patch.dict(os.environ, {"PYTHONPATH": str(extra)}):
                expose_editable_to_children(root, distribution, installed, wheel_site=extra)
                self.assertEqual(
                    os.environ["PYTHONPATH"].split(os.pathsep), [str(root / ".native-child-site"), str(extra)]
                )
                result = subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        "from native_fixture.cli.main import main; from importlib.metadata import distribution; "
                        "import preserved_module; d = distribution('native-fixture'); "
                        "assert d.version == '1.2'; assert next(iter(d.entry_points)).load()() == main(); "
                        "assert preserved_module.value == 42; print(main())",
                    ],
                    cwd=base,
                    text=True,
                )
                vendor = base / "vendored"
                vendor_package = vendor / "native_fixture/cli"
                vendor_package.mkdir(parents=True)
                (vendor / "native_fixture/__init__.py").touch()
                (vendor_package / "__init__.py").touch()
                (vendor_package / "main.py").write_text("def main():\n    return 'vendored entrypoint'\n")
                script = vendor / "runner.py"
                script.write_text("from native_fixture.cli.main import main; print(main())\n")
                with self.subTest(precedence="script directory before installed wheel"):
                    output = subprocess.check_output([sys.executable, str(script)], cwd=base, text=True)
                    self.assertEqual(output.strip(), "vendored entrypoint")
                with self.subTest(precedence="explicit sys.path before installed wheel"):
                    output = subprocess.check_output(
                        [
                            sys.executable,
                            "-c",
                            f"import sys; sys.path.insert(0, {str(vendor)!r}); " + script.read_text(),
                        ],
                        cwd=base,
                        text=True,
                    )
                    self.assertEqual(output.strip(), "vendored entrypoint")
                with self.subTest(precedence="ordinary editable fallback without wheel"):
                    shutil.rmtree(extra / "native_fixture")
                    output = subprocess.check_output([sys.executable, "-c", script.read_text()], cwd=base, text=True)
                    self.assertEqual(output.strip(), "sampled entrypoint")
            self.assertEqual(result.strip(), "sampled entrypoint")
            self.assertNotIn(original, (root / ".native-child-site/fixture_finder.py").read_text())

    def test_directory_and_dangling_symlinks_retain_their_git_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / ".agents/skills/example"
            canonical.mkdir(parents=True)
            (canonical / "data.txt").write_text("fixture")
            links = {".claude/skills/example": "../../.agents/skills/example", "dangling": "missing-target"}
            restore_symlinks(root, links)
            self.assertEqual((root / ".claude/skills/example").resolve(), canonical.resolve())
            self.assertEqual((root / ".claude/skills/example/data.txt").read_text(), "fixture")
            self.assertTrue((root / "dangling").is_symlink())
            self.assertFalse((root / "dangling").exists())
            self.assertEqual((root / "dangling").readlink(), Path("missing-target"))

    def test_batching_keeps_complete_target_scope_and_separate_resolves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts/pants/native_core").mkdir(parents=True)
            (root / "tests").mkdir()
            files = ["tests/test_a.py", "tests/test_b.py", "library.py"]
            for name in files:
                (root / name).touch()
            (root / "alias").symlink_to("tests", target_is_directory=True)
            (root / "dangling").symlink_to("missing-target")
            files.extend(["alias", "dangling"])
            (root / ".python-version").write_text("3.13.14")
            (root / "pyproject.toml").write_text(
                '[project]\nname = "fixture"\n[tool.uv]\n[tool.coverage.run]\ndata_file = "results/.coverage"\n'
            )

            def git_output(command, **kwargs):
                return "a" * 40 if kwargs.get("text") else "\0".join(files).encode()

            with (
                patch("prepare.snapshot_git"),
                patch("prepare.environment_names", return_value=["PATH"]),
                patch("prepare.subprocess.check_output", side_effect=git_output),
            ):
                write_build_configuration(root)
            tests = []
            exec(
                compile((root / "BUILD.native").read_text(), "BUILD.native", "exec"),
                {
                    "python_tests": lambda **values: tests.append(values),
                    "python_sources": lambda **values: None,
                    "files": lambda **values: None,
                    "relocated_files": lambda **values: None,
                    "python_distribution": lambda **values: None,
                    "python_artifact": lambda **values: values,
                },
            )
            self.assertEqual(len(tests), 2)
            self.assertEqual({item["resolve"] for item in tests}, {"core", "sandbox"})
            self.assertEqual(len({item["batch_compatibility_tag"] for item in tests}), 2)
            for item in tests:
                self.assertEqual(item["sources"], files[:2])
                self.assertIn(":native-project", item["dependencies"])
            self.assertEqual(tomllib.loads((root / "pants.native.toml").read_text())["test"]["batch_size"], 128)
            self.assertEqual(
                json.loads((root / "scripts/pants/native_core/symlinks.json").read_text()),
                {"alias": "tests", "dangling": "missing-target"},
            )

    def test_real_collection_skip_is_preserved_once_in_each_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test_optional.py").write_text(
                "import pytest\npytest.importorskip('native_fixture_missing_dependency')\n"
            )
            (root / "test_regular.py").write_text("def test_regular():\n    assert True\n")
            reports = root / "reports"
            reports.mkdir()
            collection = root / "collection.xml"
            environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
            common = [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "-q",
                "--junit-prefix=nemo_gym.core",
                "-o",
                "junit_family=xunit1",
            ]
            subprocess.run(
                common + ["--collect-only", "--junitxml=" + str(collection), str(root)],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                common + ["--junitxml=" + str(reports / "executed.xml"), str(root / "test_regular.py")],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
            )
            for shard in range(4):
                output = root / f"shard-{shard}.xml"
                combine_junit(reports, output, collection=collection)
                cases = list(ElementTree.parse(output).getroot().iter("testcase"))
                self.assertEqual(len(cases), 2)
                self.assertEqual(sum(case.find("skipped") is not None for case in cases), 1)
                self.assertEqual(sum(case.get("name") == "test_regular" for case in cases), 1)
            # A runner that also reports the collection event must not duplicate it.
            shutil.copy2(collection, reports / "duplicate-skip.xml")
            combine_junit(reports, root / "deduplicated.xml", collection=collection)
            self.assertEqual(len(list(ElementTree.parse(root / "deduplicated.xml").getroot().iter("testcase"))), 2)

    def test_collection_cannot_supply_unexecuted_passing_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports"
            reports.mkdir()
            (reports / "run.xml").write_text('<testsuite><testcase name="executed" /></testsuite>')
            collection = root / "collection.xml"
            collection.write_text('<testsuite><testcase name="collected-only" /></testsuite>')
            with self.assertRaisesRegex(ValueError, "unexpectedly contains"):
                combine_junit(reports, root / "result.xml", collection=collection)

    def test_repeated_collection_skip_identities_retain_their_multiplicity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports"
            reports.mkdir()
            case = '<testcase name="same-module"><skipped /></testcase>'
            (reports / "run.xml").write_text("<testsuite>" + case + "</testsuite>")
            collection = root / "collection.xml"
            collection.write_text("<testsuite>" + case * 2 + "</testsuite>")
            combine_junit(reports, root / "result.xml", collection=collection)
            self.assertEqual(len(list(ElementTree.parse(root / "result.xml").getroot().iter("testcase"))), 2)

    def test_declared_git_snapshot_preserves_tree_and_commit_without_parent_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "--quiet", str(source)], check=True)
            environment = dict(
                os.environ,
                GIT_AUTHOR_NAME="Fixture",
                GIT_AUTHOR_EMAIL="fixture@example.test",
                GIT_COMMITTER_NAME="Fixture",
                GIT_COMMITTER_EMAIL="fixture@example.test",
            )

            def git(*args, input=None):
                return subprocess.check_output(["git", *args], cwd=source, env=environment, input=input)

            blob = git("hash-object", "-w", "--stdin", input=b"sample\n").strip()
            tree = git("mktree", input=b"100644 blob " + blob + b"\tsource.py\n").strip()
            commit = git("commit-tree", tree.decode(), input=b"fixture\n").strip()
            git("update-ref", "HEAD", commit.decode())
            git("read-tree", "HEAD")
            (source / "source.py").write_bytes(b"sample\n")
            snapshot_git(source)
            exclude_instrumentation(source)
            (source / "BUILD.native").touch()
            self.assertEqual(git("status", "--porcelain"), b"")
            isolated = root / "isolated"
            isolated.mkdir()
            shutil.copytree(source / "scripts/pants/native_core/git-state", isolated / ".git")
            (isolated / "source.py").write_bytes(b"sample\n")
            shutil.rmtree(source)
            self.assertEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=isolated).strip(), commit)
            self.assertEqual(subprocess.check_output(["git", "ls-files"], cwd=isolated), b"source.py\n")
            self.assertEqual(subprocess.check_output(["git", "show", "HEAD:source.py"], cwd=isolated), b"sample\n")
            self.assertFalse((isolated / ".git/objects/info/alternates").exists())
            (isolated / "pytest_runner.pex").touch()
            (isolated / "nemo_gym-1.dist-info").mkdir()
            (isolated / "nemo_gym-1.dist-info/METADATA").write_text("sample metadata")
            exclude_instrumentation(isolated, ("nemo_gym-1.dist-info/",))
            self.assertEqual(subprocess.check_output(["git", "status", "--porcelain"], cwd=isolated), b"")
            content_store = root / "shared-content"
            content_store.write_bytes(b"sample\n")
            content_store.chmod(0o755)
            (isolated / "source.py").unlink()
            os.link(content_store, isolated / "source.py")
            restore_tracked_modes(isolated)
            self.assertEqual(content_store.stat().st_mode & 0o777, 0o755)
            self.assertNotEqual(content_store.stat().st_ino, (isolated / "source.py").stat().st_ino)
            self.assertEqual(subprocess.check_output(["git", "status", "--porcelain"], cwd=isolated), b"")
            (isolated / "source.py").chmod(0o755)
            self.assertIn(
                " M source.py", subprocess.check_output(["git", "status", "--porcelain"], cwd=isolated, text=True)
            )
            (isolated / "source.py").write_bytes(b"changed\n")
            (isolated / "unexpected.py").write_text("untracked source\n")
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=isolated, text=True)
            self.assertIn(" M source.py", status)
            self.assertIn("?? unexpected.py", status)
            (isolated / "source.py").unlink()
            self.assertIn(
                " D source.py", subprocess.check_output(["git", "status", "--porcelain"], cwd=isolated, text=True)
            )

    def test_four_shards_cover_each_globally_collected_file_once(self):
        files = {f"tests/unit_tests/test_{index}.py": [f"test_{index}"] for index in range(354)}
        assignments = {path: index % 4 for index, path in enumerate(files)}
        manifest = {"valid": True, "marker": "not sandbox", "files": files, "assignments": assignments}
        selected = [target for shard in range(4) for target in targets(manifest, resolve="core", shard=shard)]
        self.assertEqual(len(selected), 354)
        self.assertEqual(set(selected), {f"//{path}:../../core-tests" for path in files})
        self.assertEqual(len(set(selected)), len(selected))

    def test_wrong_marker_or_partial_collection_cannot_select_targets(self):
        for manifest in ({"valid": False}, {"valid": True, "marker": "sandbox"}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                targets(manifest, resolve="core", shard=0)

    def test_sandbox_has_no_core_sharding(self):
        manifest = {"valid": True, "marker": "sandbox", "files": {"tests/test_sandbox.py": ["x"]}}
        self.assertEqual(
            targets(manifest, resolve="sandbox", shard=None), ["//tests/test_sandbox.py:../sandbox-tests"]
        )
        with self.assertRaises(ValueError):
            targets(manifest, resolve="sandbox", shard=0)

    def test_mutable_vcs_and_unpinned_registry_requirements_fail(self):
        for line in ("foo>=1", "foo @ git+https://example.test/r.git@main", "foo @ https://example.test/foo.whl"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                requirement_pins(line)

    def lock_check(self, requirements, locked):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req = root / "requirements.txt"
            lock = root / "lock.json"
            req.write_text(requirements)
            lock.write_text(json.dumps({"locked_resolves": [{"locked_requirements": locked}]}))
            return validate_lock(req, lock)

    def test_excluded_dependency_reintroduced_by_pex_fails(self):
        with self.assertRaisesRegex(ValueError, "absent from uv: unwanted"):
            self.lock_check(
                "foo==1", [{"project_name": "foo", "version": "1"}, {"project_name": "unwanted", "version": "2"}]
            )

    def test_missing_or_drifted_pin_fails(self):
        for locked in ([], [{"project_name": "foo", "version": "2"}]):
            with self.subTest(locked=locked), self.assertRaises(ValueError):
                self.lock_check("foo==1", locked)

    def test_vcs_revision_and_artifact_integrity_are_required(self):
        url = "git+https://example.test/r.git@" + "a" * 40
        item = {"project_name": "foo", "version": "1", "artifacts": [{"url": url, "hash": "sha256"}]}
        self.assertEqual(self.lock_check(f"foo @ {url}", [item])["distributions"], 1)
        item["artifacts"][0]["url"] = url[:-1] + "b"
        with self.assertRaisesRegex(ValueError, "exact VCS source"):
            self.lock_check(f"foo @ {url}", [item])
        item["artifacts"][0] = {"url": url}
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.lock_check(f"foo @ {url}", [item])


if __name__ == "__main__":
    unittest.main()
