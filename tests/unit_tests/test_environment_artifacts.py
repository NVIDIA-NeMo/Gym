# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nemo_gym import PARENT_DIR
from nemo_gym.cli.setup_command import run_command
from nemo_gym.config_types import ConfigError
from nemo_gym.environment import artifacts


@pytest.fixture
def release(tmp_path, monkeypatch):
    root = tmp_path / "author"
    workload = root / "environments/example_single_tool_call"
    workload.mkdir(parents=True)
    (workload / "manifest.yaml").write_bytes(
        (PARENT_DIR / "environments/example_single_tool_call/manifest.yaml").read_bytes()
    )
    (workload / "config.yaml").write_text("example: {}\n")
    (workload / "package.yaml").write_text("include:\n- environments/example_single_tool_call\n")
    (workload / ".env").write_text("secret=must-not-publish")
    monkeypatch.setattr(artifacts, "validate_environment", lambda *args: None)
    return SimpleNamespace(manifest_path=workload / "manifest.yaml", config_path=workload / "config.yaml")


def test_deterministic_roundtrip_and_cache_integrity(release, tmp_path, monkeypatch):
    first = artifacts.build_environment_package(release, tmp_path / "first.tar.gz")
    second = artifacts.build_environment_package(release, tmp_path / "second.tar.gz")
    assert first.read_bytes() == second.read_bytes()
    monkeypatch.setattr(artifacts, "CACHE_DIR", tmp_path / "cache")
    root = artifacts.pull_environment_package(str(first))
    assert not list(root.rglob(".env"))
    metadata = json.loads((root / artifacts.PACKAGE_METADATA).read_text())
    assert metadata["name"] == "example_single_tool_call"
    assert (root / metadata["config_path"]).read_bytes() == release.config_path.read_bytes()
    assert artifacts.pull_environment_package(str(first)) == root
    (root / metadata["config_path"]).write_text("tampered")
    with pytest.raises(ConfigError, match="checksum mismatch"):
        artifacts.pull_environment_package(str(first))


@pytest.mark.parametrize("relative_root", [False, True])
def test_pulled_server_imports_packaged_sibling_before_gym(release, tmp_path, monkeypatch, relative_root):
    author = release.manifest_path.parents[2]
    server = author / "resources_servers/packaged_server"
    server.mkdir(parents=True)
    (server / "setup_server.py").write_text("VALUE = 'from registry package'\n")
    (server / "app.py").write_text(
        "from resources_servers.packaged_server.setup_server import VALUE\n"
        "from pathlib import Path\n"
        "Path('import-result.txt').write_text(VALUE)\n"
    )
    release.manifest_path.with_name("package.yaml").write_text(
        "include:\n- environments/example_single_tool_call\n- resources_servers/packaged_server\n"
    )
    archive = artifacts.build_environment_package(release, tmp_path / "server.tar.gz")
    installed = artifacts.pull_environment_package(str(archive), tmp_path / "installed")
    installed_server = installed / "resources_servers/packaged_server"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", "installed" if relative_root else str(installed))
    command = f"cd {shlex.quote(str(installed_server))} && {shlex.quote(sys.executable)} -S app.py"

    # -S prevents editable installs/site-packages from masking a missing package import path.
    with (tmp_path / "server.log").open("w") as log:
        process = run_command(
            command,
            installed_server,
            global_config_dict={"uv_cache_dir": str(tmp_path / "uv-cache")},
            stdout_target=log,
            stderr_target=log,
        )
        assert process.wait(timeout=30) == 0

    assert (installed_server / "import-result.txt").read_text() == "from registry package"


@pytest.mark.parametrize("include", ["../outside", "/etc/passwd", "environments/missing"])
def test_invalid_include_cannot_publish(release, tmp_path, include):
    release.manifest_path.with_name("package.yaml").write_text(yaml.safe_dump({"include": [include]}))
    with pytest.raises(ConfigError):
        artifacts.build_environment_package(release, tmp_path / "bad.tar.gz")
    assert not (tmp_path / "bad.tar.gz").exists()


def test_symlink_cannot_publish(release, tmp_path):
    release.manifest_path.with_name("outside").symlink_to(tmp_path)
    with pytest.raises(ConfigError, match="symbolic links"):
        artifacts.build_environment_package(release, tmp_path / "bad.tar.gz")


@pytest.mark.parametrize("attack", ["traversal", "absolute", "symlink", "duplicate", "oversized"])
def test_unsafe_archive_leaves_no_destination(tmp_path, attack):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo({"traversal": "../escape", "absolute": "/escape"}.get(attack, "file"))
        if attack == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
        archive.addfile(info)
        if attack == "duplicate":
            archive.addfile(info)
        if attack == "oversized":
            info = tarfile.TarInfo("huge")
            info.size = artifacts.MAX_PACKAGE_BYTES + 1
            archive.fileobj.write(info.tobuf())
    with pytest.raises(ConfigError):
        artifacts.pull_environment_package(str(path), tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "escape").exists()


def test_tampered_archive_is_rejected(release, tmp_path):
    original = artifacts.build_environment_package(release, tmp_path / "original.tar.gz")
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(original) as source, tarfile.open(tampered, "w:gz") as target:
        for member in source:
            data = source.extractfile(member).read()
            if member.name.endswith("config.yaml"):
                data = b"tampered"
                member.size = len(data)
            target.addfile(member, io.BytesIO(data))
    with pytest.raises(ConfigError, match="checksum mismatch"):
        artifacts.pull_environment_package(str(tampered), tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_existing_output_is_never_overwritten(release, tmp_path):
    path = artifacts.build_environment_package(release, tmp_path / "release.tar.gz")
    with pytest.raises(ConfigError, match="already exists"):
        artifacts.build_environment_package(release, path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_text("owned by user")
    with pytest.raises(ConfigError, match="already exists"):
        artifacts.pull_environment_package(str(path), output)
    assert (output / "keep").read_text() == "owned by user"


@pytest.mark.parametrize("result", [(0, ""), (1, "unauthorized"), (1, "network unavailable")])
def test_publish_never_overwrites_or_ignores_auth_failure(release, tmp_path, monkeypatch, result):
    package = artifacts.build_environment_package(release, tmp_path / "release.tar.gz")
    calls = []

    def oras(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, result[0], "sha256:existing", result[1])

    monkeypatch.setattr(artifacts, "_oras", oras)
    with pytest.raises(ConfigError):
        artifacts.push_environment_package(package, "registry.example/alice/env:0.1.0")
    assert [call[0] for call in calls] == ["resolve"]


@pytest.mark.parametrize("image_spec", ["v1.0", "v1.1"])
def test_registry_fetch_uses_resolved_digest(release, tmp_path, monkeypatch, image_spec):
    package = artifacts.build_environment_package(release, tmp_path / "release.tar.gz")
    payload = package.read_bytes()
    layer_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    type_fields = (
        {"artifactType": artifacts.ARTIFACT_TYPE}
        if image_spec == "v1.1"
        else {"annotations": {artifacts.ARTIFACT_ANNOTATION: artifacts.ARTIFACT_TYPE}}
    )
    manifest = json.dumps(
        {
            **type_fields,
            "layers": [{"mediaType": artifacts.LAYER_TYPE, "digest": layer_digest, "size": len(payload)}],
        }
    ).encode()
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    calls = []

    def oras(*args, **kwargs):
        calls.append(args)
        if args[0] == "resolve":
            return subprocess.CompletedProcess(args, 0, manifest_digest + "\n", "")
        Path(args[args.index("--output") + 1]).write_bytes(manifest if args[0] == "manifest" else payload)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(artifacts, "_oras", oras)
    monkeypatch.setattr(artifacts, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("GYM_ENV_REGISTRY", "registry.example/community")
    root = artifacts.pull_environment_package("alice/env:0.1.0")
    assert (root / artifacts.PACKAGE_METADATA).is_file()
    assert calls[0] == ("resolve", "registry.example/community/alice/env:0.1.0")
    assert calls[1][-1] == f"registry.example/community/alice/env@{manifest_digest}"
    assert calls[2][-1] == f"registry.example/community/alice/env@{layer_digest}"
    assert artifacts.pull_environment_package(f"registry.example/community/alice/env@{manifest_digest}") == root
    assert [call[0] for call in calls] == ["resolve", "manifest", "blob", "resolve", "manifest"]
    (root / "environments/example_single_tool_call/config.yaml").write_text("tampered")
    with pytest.raises(ConfigError, match="checksum mismatch"):
        artifacts.pull_environment_package("alice/env:0.1.0")
    assert sum(call[0] == "blob" for call in calls) == 1


@pytest.mark.parametrize("suffix", [":0.1.0", "@sha256:" + "a" * 64])
def test_namespaced_registry_reference(monkeypatch, suffix):
    monkeypatch.setenv("GYM_ENV_REGISTRY", "oci://registry.example:5000/community/")
    assert artifacts._oci_reference(f"nvidia/workplace_assistant{suffix}") == (
        f"registry.example:5000/community/nvidia/workplace_assistant{suffix}",
        "registry.example:5000/community/nvidia/workplace_assistant",
    )


@pytest.mark.parametrize("host", ["registry.example", "localhost", "localhost:5000"])
def test_full_reference_ignores_default_registry(monkeypatch, host):
    monkeypatch.setenv("GYM_ENV_REGISTRY", "invalid registry")
    assert artifacts._oci_reference(f"oci://{host}/alice/env:1") == (f"{host}/alice/env:1", f"{host}/alice/env")


@pytest.mark.parametrize("reference", ["alice/nested/env:1", "Alice/env:1", "alice/env:", "alice/env"])
def test_invalid_namespaced_reference_rejected(monkeypatch, reference):
    monkeypatch.setenv("GYM_ENV_REGISTRY", "registry.example/community")
    with pytest.raises(ConfigError):
        artifacts._oci_reference(reference)


@pytest.mark.parametrize("registry", ["", "https://registry.example/community", "registry.example/community:1"])
def test_invalid_registry_configuration_rejected(monkeypatch, registry):
    monkeypatch.setenv("GYM_ENV_REGISTRY", registry)
    with pytest.raises(ConfigError, match="GYM_ENV_REGISTRY"):
        artifacts._oci_reference("alice/env:1")


@pytest.mark.parametrize("reference", ["env", "registry.example/alice/env", "registry.example/env@sha256:bad"])
def test_missing_version_or_invalid_digest_rejected(reference, tmp_path):
    with pytest.raises(ConfigError):
        artifacts.pull_environment_package(reference, tmp_path / "output")


@pytest.mark.parametrize("change", ["schema", "version", "inventory", "identity", "config", "symlink", "corrupt_json"])
def test_invalid_release_metadata_rejected(release, tmp_path, change):
    package = artifacts.build_environment_package(release, tmp_path / "release.tar.gz")
    root = artifacts.pull_environment_package(str(package), tmp_path / "valid")
    metadata_path = root / artifacts.PACKAGE_METADATA
    metadata = json.loads(metadata_path.read_text())
    if change == "schema":
        metadata["schema_version"] = 99
    elif change == "version":
        metadata["gym_version"] = "99.0.0"
    elif change == "inventory":
        metadata["files"]["missing"] = "0" * 64
    elif change == "identity":
        metadata["name"] = "different"
    elif change == "config":
        metadata["config_path"] = "missing.yaml"
    elif change == "symlink":
        config = root / metadata["config_path"]
        config.unlink()
        config.symlink_to(release.config_path)
    metadata_path.write_text("{broken" if change == "corrupt_json" else json.dumps(metadata))
    with pytest.raises(ConfigError, match="Invalid environment package"):
        artifacts._verify_package(root)


def test_publish_exports_immutable_digest(release, tmp_path, monkeypatch):
    package = artifacts.build_environment_package(release, tmp_path / "release.tar.gz")
    calls = []
    exported = b'{"schemaVersion":2}'

    def oras(*args, **kwargs):
        calls.append(args)
        if args[0] == "resolve":
            return subprocess.CompletedProcess(args, 1, "", "not found")
        assert (kwargs["cwd"] / "environment.tar.gz").read_bytes() == package.read_bytes()
        (kwargs["cwd"] / "manifest.json").write_bytes(exported)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(artifacts, "_oras", oras)
    result = artifacts.push_environment_package(package, "oci://registry.example/alice/env:0.1.0")
    assert result == "registry.example/alice/env@sha256:" + hashlib.sha256(exported).hexdigest()
    assert calls[1][0:3] == ("push", "--image-spec", "v1.0")
    assert f"{artifacts.ARTIFACT_ANNOTATION}={artifacts.ARTIFACT_TYPE}" in calls[1]


@pytest.mark.parametrize("reference", ["registry.example/env:wrong", "registry.example/env@sha256:" + "a" * 64])
def test_publish_requires_manifest_version_tag(release, tmp_path, monkeypatch, reference):
    package = artifacts.build_environment_package(release, tmp_path / "release.tar.gz")
    monkeypatch.setattr(artifacts, "_oras", lambda *args, **kwargs: pytest.fail("must not contact registry"))
    with pytest.raises(ConfigError, match="version|tag"):
        artifacts.push_environment_package(package, reference)


@pytest.mark.parametrize(
    "failure", ["wrong_manifest_hash", "bad_json", "non_object", "wrong_type", "bad_layer", "bad_blob"]
)
def test_invalid_registry_content_is_rejected(tmp_path, monkeypatch, failure):
    payload = b"not a valid package"
    layer_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    manifest = {
        "artifactType": artifacts.ARTIFACT_TYPE,
        "layers": [{"mediaType": artifacts.LAYER_TYPE, "digest": layer_digest, "size": len(payload)}],
    }
    if failure == "wrong_type":
        manifest["artifactType"] = "another-artifact"
    elif failure == "bad_layer":
        manifest["layers"][0]["size"] = artifacts.MAX_PACKAGE_BYTES + 1
    encoded = (
        b"not JSON" if failure == "bad_json" else b"[]" if failure == "non_object" else json.dumps(manifest).encode()
    )
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()

    def oras(*args, **kwargs):
        if args[0] == "resolve":
            return subprocess.CompletedProcess(args, 0, digest, "")
        if args[0] == "manifest":
            data = b"tampered" if failure == "wrong_manifest_hash" else encoded
        else:
            data = b"tampered" if failure == "bad_blob" else payload
        Path(args[args.index("--output") + 1]).write_bytes(data)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(artifacts, "_oras", oras)
    with pytest.raises(ConfigError):
        artifacts.pull_environment_package("registry.example/env:1", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_oras_errors_are_actionable(monkeypatch):
    monkeypatch.setattr(artifacts.shutil, "which", lambda name: None)
    monkeypatch.setattr(artifacts.platform, "system", lambda: "Unknown")
    with pytest.raises(ConfigError, match="Automatic registry setup is unavailable"):
        artifacts._oras("resolve", "registry.example/env:1")
    monkeypatch.setattr(artifacts.shutil, "which", lambda name: "/tools/oras")
    monkeypatch.setattr(
        artifacts.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "unauthorized")
    )
    with pytest.raises(ConfigError, match="Registry access was denied"):
        artifacts._oras("resolve", "registry.example/env:1")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("oras", 600)

    monkeypatch.setattr(artifacts.subprocess, "run", timeout)
    with pytest.raises(ConfigError, match="Could not run ORAS"):
        artifacts._oras("resolve", "registry.example/env:1")


@pytest.mark.parametrize(
    "system,machine,target",
    [
        ("Darwin", "arm64", "darwin_arm64"),
        ("Darwin", "x86_64", "darwin_amd64"),
        ("Linux", "aarch64", "linux_arm64"),
        ("Linux", "x86_64", "linux_amd64"),
    ],
)
def test_oras_is_provisioned_once_and_atomically_cached(monkeypatch, tmp_path, system, machine, target):
    binary = b"test executable"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo("oras")
        member.size = len(binary)
        archive.addfile(member, io.BytesIO(binary))
    payload = stream.getvalue()
    monkeypatch.setattr(artifacts.shutil, "which", lambda name: None)
    monkeypatch.setattr(artifacts.platform, "system", lambda: system)
    monkeypatch.setattr(artifacts.platform, "machine", lambda: machine)
    monkeypatch.setattr(artifacts, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(artifacts.ORAS_CHECKSUMS, target, hashlib.sha256(payload).hexdigest())
    downloads = []

    def download(url, timeout):
        downloads.append(url)
        assert timeout == 60
        return io.BytesIO(payload)

    def run(args, **kwargs):
        executable = Path(args[0])
        assert executable.read_bytes() == binary
        assert executable.stat().st_mode & 0o111 == 0o111
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(args, 0, "1.3.4", "")

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", download)
    monkeypatch.setattr(artifacts.subprocess, "run", run)
    artifacts._oras("version", cwd=tmp_path)
    artifacts._oras("version", cwd=tmp_path)
    assert downloads == [f"https://github.com/oras-project/oras/releases/download/v1.3.4/oras_1.3.4_{target}.tar.gz"]
    assert not list(tmp_path.rglob(".install-*"))


@pytest.mark.parametrize("failure", ["checksum", "network", "symlink", "corrupt"])
def test_failed_oras_provisioning_leaves_no_executable(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(artifacts.shutil, "which", lambda name: None)
    monkeypatch.setattr(artifacts.platform, "system", lambda: "Linux")
    monkeypatch.setattr(artifacts.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(artifacts, "CACHE_DIR", tmp_path)
    payload = b"corrupted download"
    if failure == "symlink":
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            member = tarfile.TarInfo("oras")
            member.type = tarfile.SYMTYPE
            member.linkname = "/bin/sh"
            archive.addfile(member)
        payload = stream.getvalue()
    if failure != "checksum":
        monkeypatch.setitem(artifacts.ORAS_CHECKSUMS, "linux_amd64", hashlib.sha256(payload).hexdigest())

    def download(*args, **kwargs):
        if failure == "network":
            raise OSError("offline")
        return io.BytesIO(payload)

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", download)
    monkeypatch.setattr(artifacts.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not execute"))
    with pytest.raises(ConfigError):
        artifacts._oras("version")
    assert not list(tmp_path.rglob("oras"))


@pytest.mark.parametrize(
    "problem",
    [
        "no_manifest",
        "bad_root",
        "missing_spec",
        "bad_spec",
        "missing_config",
        "private_key",
        "reserved",
        "size",
        "count",
    ],
)
def test_unpublishable_packages_fail_before_writing(release, tmp_path, monkeypatch, problem):
    spec = release.manifest_path.with_name("package.yaml")
    if problem == "no_manifest":
        release.manifest_path = None
    elif problem == "bad_root":
        root = release.manifest_path.parent.parent
        root.rename(root.with_name("unsupported"))
        release.manifest_path = root.with_name("unsupported") / "example_single_tool_call/manifest.yaml"
    elif problem == "missing_spec":
        spec.unlink()
    elif problem == "bad_spec":
        spec.write_text("unexpected: []")
    elif problem == "missing_config":
        spec.write_text("include: [environments/example_single_tool_call/manifest.yaml]")
    elif problem == "private_key":
        release.manifest_path.with_name("private.key").write_text("must-not-publish")
    elif problem == "reserved":
        path = release.manifest_path.parent.parent.parent / artifacts.PACKAGE_METADATA
        path.write_text("{}")
        spec.write_text(yaml.safe_dump({"include": [artifacts.PACKAGE_METADATA]}))
    elif problem == "size":
        monkeypatch.setattr(artifacts, "MAX_PACKAGE_BYTES", 1)
    elif problem == "count":
        monkeypatch.setattr(artifacts, "MAX_PACKAGE_FILES", 1)
    with pytest.raises(ConfigError):
        artifacts.build_environment_package(release, tmp_path / "output.tar.gz")
    assert not (tmp_path / "output.tar.gz").exists()
