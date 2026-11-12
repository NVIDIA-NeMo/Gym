# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose declared checkout metadata inside each native pytest sandbox."""

import fcntl
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path


INSTRUMENTATION_PATHS = (
    "scripts/pants/native_core/",
    "BUILD.native",
    "pants.native.toml",
    "pants.toml",
    ".pants.d/",
    ".pids/",
    ".native-child-site/",
    "__run.sh",
    "__named_caches",
    "__immutable_inputs",
    ".python-build-standalone",
    "core_lockfile.pex",
    "sandbox_lockfile.pex",
    "pytest_runner.pex",
    "pytest_runner.pex_pex_shim.sh",
    "pytest_runner.pex_bin_python_shim.sh",
    "pytest.pex",
    "requirements.pex",
    "local_dists.pex",
)


def exclude_instrumentation(root: Path, extra_paths: tuple[str, ...] = (), *, git_dir: Path | None = None) -> None:
    """Hide only native runner artifacts from Git's untracked-file view.

    Git ignores never hide changes to indexed files. Keep ordinary untracked
    source visible too: clean-checkout guards must still reject real dirtiness.
    """
    if git_dir is None:
        path = subprocess.check_output(["git", "rev-parse", "--git-path", "info/exclude"], cwd=root, text=True)
        exclude = root / path.strip()
    else:
        exclude = git_dir / "info/exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text() if exclude.exists() else ""
    entries = []
    for relative in (*INSTRUMENTATION_PATHS, *extra_paths):
        if Path(relative).is_absolute() or ".." in Path(relative).parts or "\n" in relative:
            raise ValueError(f"Invalid instrumentation path: {relative!r}")
        # Exact root-relative names, not a wildcard exclusion for arbitrary files.
        escaped = "".join("\\" + char if char in "\\*?[]" else char for char in relative)
        entry = "/" + escaped
        if entry not in existing.splitlines():
            entries.append(entry)
    if entries:
        exclude.write_text(existing.rstrip("\n") + "\n" + "\n".join(entries) + "\n")


def restore_symlinks(root: Path, links: dict[str, str]) -> None:
    """Restore Git's exact link text, including directory aliases and dangling links."""
    for name, target in links.items():
        path = root / name
        if path.is_symlink():
            if os.readlink(path) != target:
                raise RuntimeError(f"Native sandbox has a different symlink: {name}")
            continue
        if path.exists():
            raise RuntimeError(f"Native sandbox symlink path is occupied: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)


def restore_build_directories(root: Path) -> None:
    """Restore the empty editable-build directory omitted by file digests."""
    project = tomllib.loads((root / "pyproject.toml").read_text())
    egg_base = project.get("tool", {}).get("distutils", {}).get("egg_info", {}).get("egg_base")
    if egg_base is not None:
        destination = root / egg_base
        if Path(egg_base).is_absolute() or not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("Editable egg_base must stay inside the native checkout")
        destination.mkdir(parents=True, exist_ok=True)


def restore_tracked_modes(root: Path) -> None:
    """Restore Git executable bits without chmod-ing Pants's shared CAS files."""
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).split(b"\0")
    for entry in filter(None, entries):
        metadata, name = entry.split(b"\t", 1)
        mode = metadata.split()[0]
        path = root / os.fsdecode(name)
        if mode == b"120000":
            continue  # Exact links were restored separately.
        if mode not in (b"100644", b"100755") or not path.is_file():
            raise RuntimeError(f"Native sandbox is missing a supported tracked file: {os.fsdecode(name)}")
        current_mode = path.stat().st_mode & 0o777
        executable = mode == b"100755"
        if bool(current_mode & 0o100) == executable:
            continue
        # A sandbox file may be a hardlink into the content store. Replace it
        # with a private copy before changing permissions; never alter its bytes.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".native-mode-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, stream)
                stream.flush()
                temporary.chmod((current_mode | 0o111) if executable else (current_mode & ~0o111))
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


def expose_editable_to_children(root: Path, distribution: Path, installed: Path, *, wheel_site: Path) -> None:
    """Relocate the actual editable finder into a sandbox-local child Python site."""
    from scripts.pants.native_core.editable import relocate_finder

    manifest = json.loads((distribution / "editable.json").read_text())
    relocated = relocate_finder(
        (distribution / manifest["finder"]).read_text(), original=manifest["source_root"], replacement=str(root)
    )
    child_site = root / ".native-child-site"
    child_site.mkdir()
    (child_site / manifest["finder"]).write_text(relocated)
    shutil.copy2(distribution / manifest["hook"], child_site / manifest["hook"])
    shutil.copytree(installed, child_site / installed.name)
    (child_site / "sitecustomize.py").write_text(
        "import importlib, site, sys\nfrom pathlib import Path\nfrom importlib.machinery import PathFinder\n"
        "previous = {id(finder) for finder in sys.meta_path}\n"
        "site.addsitedir(str(Path(__file__).parent))\n"
        "editable = [finder for finder in sys.meta_path if id(finder) not in previous]\n"
        f"mapping = importlib.import_module({Path(manifest['finder']).stem!r}).MAPPING\n"
        f"wheel_site = Path({str(wheel_site)!r})\n"
        "wheel_packages = [(wheel_site / name.replace('.', '/')).resolve() for name in mapping]\n"
        # The genuine wheel supplies console scripts, but only its own package
        # code should yield to the sampled editable sources. Preserve ordinary
        # script, vendor, and PYTHONPATH precedence and editable fallback order.
        "class SampledWheelOverlay:\n"
        "    @classmethod\n"
        "    def find_spec(cls, fullname, path=None, target=None):\n"
        "        regular = PathFinder.find_spec(fullname, path, target)\n"
        "        if regular is None or regular.origin is None:\n"
        "            return None\n"
        "        origin = Path(regular.origin).resolve()\n"
        "        if not any(origin.is_relative_to(package) for package in wheel_packages):\n"
        "            return None\n"
        "        for finder in editable:\n"
        "            sampled = finder.find_spec(fullname, path, target)\n"
        "            if sampled is not None:\n"
        "                return sampled\n"
        "        return None\n"
        "index = next((i for i, finder in enumerate(sys.meta_path) if finder is PathFinder), len(sys.meta_path))\n"
        "sys.meta_path.insert(index, SampledWheelOverlay)\n"
    )
    child_pythonpath(child_site)


def child_pythonpath(child_site: Path) -> None:
    """Expose child editable imports once per process, including xdist workers."""
    previous = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    paths = [str(child_site), *(path for path in previous if path and path != str(child_site))]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def _initialize_sandbox(root: Path) -> str:
    from scripts.pants.native_core.distribution import validate_distribution
    from scripts.pants.native_core.editable import restore_records

    restore_build_directories(root)
    metadata = root / "scripts/pants/native_core/git-state"
    restore_symlinks(root, json.loads((metadata.parent / "symlinks.json").read_text()))
    checkout_git = root / ".git"
    if checkout_git.exists():
        raise RuntimeError("Native test sandbox unexpectedly contains Git metadata")
    # Pants materializes files, so Git's required empty refs directory is recreated.
    (metadata / "refs").mkdir(exist_ok=True)
    checkout_git.symlink_to(metadata, target_is_directory=True)
    restore_tracked_modes(root)
    distributions = list((metadata.parent / "distribution").glob("*.dist-info"))
    if len(distributions) != 1:
        raise RuntimeError("Native sandbox is missing the sampled editable distribution")
    expected = importlib.metadata.Distribution.at(distributions[0])
    wheel = importlib.metadata.distribution(expected.metadata["Name"])
    validate_distribution(distributions[0], wheel)
    wheel_site = Path(wheel.locate_file(""))
    installed = root / distributions[0].name
    shutil.copytree(distributions[0], installed)
    # Relocate the original editable metadata alongside the relocated source.
    (installed / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}, "url": root.as_uri()}))
    expose_editable_to_children(root, metadata.parent / "distribution", installed, wheel_site=wheel_site)
    exclude_instrumentation(root, (installed.name + "/",))
    restore_records(
        metadata.parent / "distribution",
        root=root,
        installed=installed,
        child_site=root / ".native-child-site",
        wheel_site=wheel_site,
    )
    return installed.name


def initialize_sandbox(root: Path) -> None:
    """Initialize shared sandbox files exactly once across the xdist master/workers."""
    native = root / "scripts/pants/native_core"
    completed = native / ".runtime.json"
    with (native / ".runtime.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if completed.exists():
            state = json.loads(completed.read_text())
            if (
                state.get("root") != str(root)
                or not (root / ".git").is_symlink()
                or (root / ".git").resolve() != (native / "git-state").resolve()
                or not (root / state["distribution"]).is_dir()
                or not (root / ".native-child-site").is_dir()
            ):
                raise RuntimeError("Native worker found an inconsistent initialized sandbox")
        else:
            distribution = _initialize_sandbox(root)
            completed.write_text(json.dumps({"root": str(root), "distribution": distribution}) + "\n")
    child_pythonpath(root / ".native-child-site")


def pytest_load_initial_conftests(early_config, parser, args):
    initialize_sandbox(Path(__file__).resolve().parents[3])
