# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check canonical installation bytes, accurate relocated RECORDs and worker startup."""

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from editable import SOURCE_ROOT, record_bytes, restore_records, snapshot_editable
from runtime import expose_editable_to_children


def fixture(source: Path, *, timestamp: int) -> Path:
    site = source / ".venv/lib/python3.13/site-packages"
    metadata = site / "nemo_gym-1.0.dist-info"
    metadata.mkdir(parents=True)
    (source / "pyproject.toml").write_text('[project]\nname = "nemo-gym"\n')
    (metadata / "METADATA").write_text("Name: nemo-gym\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text("[console_scripts]\ngym = nemo_gym:main\n")
    (metadata / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}, "url": source.as_uri()}))
    (metadata / "uv_cache.json").write_text(
        json.dumps(
            {
                "timestamp": {"secs_since_epoch": timestamp, "nanos_since_epoch": 123},
                "commit": None,
                "env": {},
                "directories": {"src": None},
            }
        )
    )
    hook = site / "__editable__.nemo_gym-1.0.pth"
    hook.write_text("import fixture_finder; fixture_finder.install()\n")
    finder = site / "fixture_finder.py"
    finder.write_text(
        "import sys\nfrom importlib.util import spec_from_file_location\n"
        f"MAPPING = {{'nemo_gym': {str(source / 'nemo_gym')!r}}}\n"
        "class Finder:\n"
        "    @classmethod\n"
        "    def find_spec(cls, fullname, path=None, target=None):\n"
        "        if fullname in MAPPING:\n"
        "            return spec_from_file_location(fullname, MAPPING[fullname] + '/__init__.py')\n"
        "def install():\n    sys.meta_path.append(Finder)\n"
    )
    script = source / ".venv/bin/gym"
    script.parent.mkdir()
    script.write_text(f"#!{source}/.venv/bin/python\nfrom nemo_gym import main\nmain()\n")
    records = [(str(path.relative_to(site)), path.read_bytes()) for path in site.rglob("*") if path.is_file()]
    records.extend([("../../../bin/gym", script.read_bytes()), (metadata.name + "/RECORD", None)])
    (metadata / "RECORD").write_bytes(record_bytes(records))
    return metadata


def tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class EditableTest(unittest.TestCase):
    def test_distinct_installation_paths_and_timestamps_have_identical_canonical_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            snapshots = []
            for index, name in enumerate(("source", "longer source checkout")):
                source = base / name
                metadata = fixture(source, timestamp=100 + index)
                destination = base / f"snapshot-{index}"
                snapshot_editable(metadata, source=source, destination=destination)
                snapshots.append(destination)
                self.assertNotIn(str(source).encode(), b"".join(tree(destination).values()))
                manifest = json.loads((destination / "editable.json").read_text())
                self.assertEqual(manifest["source_root"], SOURCE_ROOT)
                rows = []
                for original, stored in manifest["record_files"].items():
                    rows.append(
                        (stored, None if original.endswith("/RECORD") else (destination / stored).read_bytes())
                    )
                self.assertEqual((destination / metadata.name / "RECORD").read_bytes(), record_bytes(rows))
            self.assertEqual(tree(snapshots[0]), tree(snapshots[1]))
            changed = snapshots[1] / "nemo_gym-1.0.dist-info/METADATA"
            changed.write_text("Name: nemo-gym\nVersion: 2.0\n")
            self.assertNotEqual(tree(snapshots[0]), tree(snapshots[1]))

    def test_modified_or_unsupported_original_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            metadata = fixture(source, timestamp=100)
            (metadata / "METADATA").write_text("Name: changed\n")
            with self.assertRaisesRegex(ValueError, "RECORD does not match"):
                snapshot_editable(metadata, source=source, destination=base / "snapshot")
            self.assertFalse((base / "snapshot").exists())

    def test_restored_metadata_records_real_child_files_and_genuine_scripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "source"
            metadata = fixture(source, timestamp=100)
            snapshot = base / "snapshot"
            snapshot_editable(metadata, source=source, destination=snapshot)
            root = base / "sandbox"
            root.mkdir()
            (root / "pyproject.toml").write_text('[project]\nname = "nemo-gym"\n')
            installed = root / metadata.name
            shutil.copytree(snapshot / metadata.name, installed)
            wheel_site = base / "real-wheel/lib/python3.13/site-packages"
            wheel_site.mkdir(parents=True)
            script = base / "real-wheel/bin/gym"
            script.parent.mkdir()
            script.write_text("#!genuine-wheel-interpreter\n# original installer generated launcher\n")
            before = script.read_bytes()
            with patch.dict(os.environ):
                expose_editable_to_children(root, snapshot, installed, wheel_site=wheel_site)
                restore_records(
                    snapshot,
                    root=root,
                    installed=installed,
                    child_site=root / ".native-child-site",
                    wheel_site=wheel_site,
                )
            self.assertEqual(script.read_bytes(), before)
            for restored in (installed, root / ".native-child-site" / metadata.name):
                records = list(csv.reader(io.StringIO((restored / "RECORD").read_text())))
                actual = [
                    (name, None if name.endswith("/RECORD") else (restored.parent / name).read_bytes())
                    for name, _, _ in records
                ]
                self.assertEqual((restored / "RECORD").read_bytes(), record_bytes(actual))
                self.assertEqual(json.loads((restored / "direct_url.json").read_text())["url"], root.as_uri())
                self.assertTrue(any((restored.parent / name).resolve() == script for name, _, _ in records))
            self.assertIn(str(root), (root / ".native-child-site/fixture_finder.py").read_text())
            self.assertNotIn(SOURCE_ROOT, (root / ".native-child-site/fixture_finder.py").read_text())

    def test_concurrent_worker_initialization_runs_once_and_reuses_pythonpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            native = root / "scripts/pants/native_core"
            native.mkdir(parents=True)
            controller = Path(__file__).parent
            script = """
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import runtime
root = Path(sys.argv[2])
def setup(root):
    with (root / 'initializations').open('a') as output:
        output.write('setup\\n')
    time.sleep(0.1)
    (root / 'scripts/pants/native_core/git-state').mkdir()
    (root / '.git').symlink_to(root / 'scripts/pants/native_core/git-state')
    (root / '.native-child-site').mkdir()
    (root / 'fixture.dist-info').mkdir()
    return 'fixture.dist-info'
runtime._initialize_sandbox = setup
runtime.initialize_sandbox(root)
runtime.initialize_sandbox(root)
assert os.environ['PYTHONPATH'].split(os.pathsep).count(str(root / '.native-child-site')) == 1
"""
            workers = [subprocess.Popen([sys.executable, "-c", script, str(controller), str(root)]) for _ in range(4)]
            self.assertEqual([worker.wait(timeout=20) for worker in workers], [0, 0, 0, 0])
            self.assertEqual((root / "initializations").read_text(), "setup\n")
