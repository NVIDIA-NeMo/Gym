# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A configured ``libreoffice_command`` must drive every Office -> PDF conversion.

The fake converter is a real executable, so these tests exercise argv
construction, executable lookup, and output discovery the way production does.
PATH is emptied where it matters, so a host ``libreoffice`` cannot satisfy them.
"""

import os
import shutil
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.gdpval import file_reader, preconvert, setup_libreoffice
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig


_SCRIPT = """#!/bin/sh
PATH=/usr/bin:/bin
# Output that is not UTF-8, as a container can print; every caller must still decode it.
printf 'container warning \\377\\n' >&2
printf '%s\\n' "$*" >> '@LOG@'
if [ "$1" = "--version" ]; then echo "LibreOffice 99.9 fake"; exit @VERSION_RC@; fi
outdir=""; src=""
while [ $# -gt 0 ]; do
  case "$1" in
    --outdir) outdir="$2"; shift ;;
    -env:UserInstallation=file://*) profile="${1#-env:UserInstallation=file://}" ;;
    *) src="$1" ;;
  esac
  shift
done
base=$(basename "$src")
@CONVERT@
"""


@pytest.fixture
def template_pdf(tmp_path: Path) -> Path:
    import fitz

    path = tmp_path / "template.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "rendered by the configured converter")
    document.save(path)
    document.close()
    return path


def _converter(
    directory: Path,
    template: Path,
    name: str = "fake-lo",
    *,
    converts=True,
    version_rc=0,
    unseen=None,
    workdir=None,
    locks_profile=False,
):
    """Write an executable that logs argv and writes ``<outdir>/<stem>.pdf`` like LibreOffice.

    Files under *unseen* produce nothing, like a container started without a bind for that directory. With
    *workdir* it runs there and converts only sources it can find, like a container with its own working directory.
    With *locks_profile* it leaves a file in a read-only profile directory, like the root-owned profile of a Docker
    command without ``--user``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{name}.log"
    convert = f"cp '{template}' \"$outdir/${{base%.*}}.pdf\"" if converts else ":"
    if unseen is not None:
        convert = f"case \"$src\" in '{unseen}'/*) exit 0 ;; esac\n{convert}"
    if workdir is not None:
        convert = f"cd '{workdir}' || exit 1\n[ -f \"$src\" ] || exit 0\n{convert}"
    if locks_profile:
        convert += '\nmkdir -p "$profile/user" && : > "$profile/user/registry.xcu" && chmod 555 "$profile/user"'
    script = directory / name
    script.write_text(
        _SCRIPT.replace("@LOG@", str(log)).replace("@VERSION_RC@", str(version_rc)).replace("@CONVERT@", convert)
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, log


def _calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def no_host_libreoffice(tmp_path: Path, monkeypatch) -> None:
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    # A failed host run sets file_reader's warn-once flag; keep that inside the test.
    monkeypatch.setattr(file_reader, "_libreoffice_unavailable_warned", False)


def test_ensure_accepts_working_command_without_host_checks(tmp_path, template_pdf, monkeypatch):
    script, log = _converter(tmp_path / "bin", template_pdf)
    # A prefix like `apptainer exec <image> libreoffice`: env finds the converter only on the PATH it sets.
    command = ["/usr/bin/env", f"PATH={script.parent}", script.name]
    real_run = setup_libreoffice._run

    def _spy(cmd, **kwargs):
        if cmd[0] == "apt-get":
            # Not an Exception, so ensure_libreoffice cannot swallow it.
            pytest.fail("apt-get ran for a configured command")
        return real_run(cmd, **kwargs)

    def _host_check(*_args):
        raise AssertionError("the host javaldx/java check ran for a configured command")

    monkeypatch.setattr(setup_libreoffice, "_run", _spy)
    monkeypatch.setattr(setup_libreoffice, "_javaldx_works", _host_check)
    monkeypatch.setattr(setup_libreoffice, "_java_runs", _host_check)

    assert setup_libreoffice.ensure_libreoffice(command) is True

    version, conversion = _calls(log)
    assert version == "--version"
    assert "--convert-to pdf" in conversion and conversion.endswith("probe.xlsx")


@pytest.mark.parametrize("converts, version_rc", [(False, 0), (True, 1)])
def test_ensure_rejects_command_that_fails(tmp_path, template_pdf, converts, version_rc):
    script, _ = _converter(tmp_path / "bin", template_pdf, converts=converts, version_rc=version_rc)

    assert setup_libreoffice.ensure_libreoffice([str(script)]) is False


def test_ensure_rejects_command_whose_output_is_not_a_pdf(tmp_path):
    # A converter can exit 0 and leave a .pdf that is really an error page; the probe checks the bytes.
    not_pdf = tmp_path / "error-page.html"
    not_pdf.write_text("<html>conversion failed</html>")
    script, log = _converter(tmp_path / "bin", not_pdf)

    assert setup_libreoffice.ensure_libreoffice([str(script)]) is False
    assert len(_calls(log)) == 2


def test_ensure_probes_every_directory_the_command_must_see(tmp_path, template_pdf):
    unbound = tmp_path / "deliverables"
    script, _ = _converter(tmp_path / "bin", template_pdf, unseen=unbound)

    # The temp-directory probe alone cannot tell that the command misses this directory.
    assert setup_libreoffice.ensure_libreoffice([str(script)]) is True
    assert setup_libreoffice.ensure_libreoffice([str(script)], probe_dirs=[unbound]) is False
    assert list(unbound.iterdir()) == []


def _gdpval_server(persist: Path, script: Path, **extra) -> GDPValResourcesServer:
    config = GDPValResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        name="resources",
        entrypoint="app.py",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        tavily_api_key="test-key",
        sandbox_provider="test",
        sandbox_config={},
        persist_deliverables_dir=str(persist),
        libreoffice_command=[str(script)],
        **extra,
    )
    return GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize("sees_persist_dir", [True, False])
def test_server_startup_probes_persist_deliverables_dir(tmp_path, template_pdf, sees_persist_dir):
    persist = tmp_path / "persist"
    script, _ = _converter(tmp_path / "bin", template_pdf, unseen=None if sees_persist_dir else persist)
    if sees_persist_dir:
        _gdpval_server(persist, script)
    else:
        with pytest.raises(RuntimeError, match="failed its startup check"):
            _gdpval_server(persist, script)
    # The probe cleans up after itself in the deliverables tree.
    assert list(persist.iterdir()) == []


@pytest.mark.parametrize(
    "reference, starts",
    [
        ("seen", True),
        ("unseen", False),
        ("missing", False),
        # Unseen too: the server cannot write it, so /verify converts nothing there and the probe is skipped.
        pytest.param(
            "read_only",
            True,
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory"),
        ),
    ],
)
def test_server_startup_probes_reference_deliverables_dirs(tmp_path, template_pdf, reference, starts):
    refs = tmp_path / "refs"
    if reference != "missing":
        refs.mkdir()
    script, _ = _converter(tmp_path / "bin", template_pdf, unseen=None if reference == "seen" else refs)
    if reference == "read_only":
        refs.chmod(0o555)
    try:
        if starts:
            _gdpval_server(
                tmp_path / "persist", script, reward_mode="comparison", reference_deliverables_dir=str(refs)
            )
        else:
            with pytest.raises(RuntimeError, match="startup check" if reference == "unseen" else "not a directory"):
                _gdpval_server(
                    tmp_path / "persist", script, reward_mode="comparison", reference_deliverables_dir=str(refs)
                )
    finally:
        if refs.exists():
            refs.chmod(0o755)
    # Neither creates a mistyped reference directory nor leaves probe files in the reference tree.
    assert not refs.exists() if reference == "missing" else list(refs.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_ensure_rejects_probe_directory_the_server_cannot_create(tmp_path, template_pdf, caplog):
    parent = tmp_path / "read-only"
    parent.mkdir(mode=0o555)
    script, log = _converter(tmp_path / "bin", template_pdf)

    try:
        # A warning and False, so the server raises its startup error instead of a bare PermissionError.
        assert setup_libreoffice.ensure_libreoffice([str(script)], probe_dirs=[parent / "persist"]) is False
    finally:
        parent.chmod(0o755)
    assert f"could not write the LibreOffice probe workbook in {parent / 'persist'}" in caplog.text
    # --version and the temp-directory probe ran; the persist-directory probe never reached the command.
    assert len(_calls(log)) == 2


def test_ensure_rejects_missing_command(tmp_path):
    missing = tmp_path / "missing"
    command = [str(missing), "exec", "image.sif", "libreoffice"]
    source = tmp_path / "memo.docx"
    source.write_bytes(b"office source")

    assert setup_libreoffice.ensure_libreoffice(command) is False
    # A conversion names the configured launcher instead of suggesting a host install.
    message = f"libreoffice_command executable not found: {missing}"
    assert preconvert.convert_to_pdf(source, libreoffice_command=command) == (source, False, message)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can delete files in a read-only directory")
@pytest.mark.parametrize("owned_by_another_user", [True, False])
def test_ensure_rejects_command_that_leaves_files_the_server_cannot_delete(
    tmp_path, template_pdf, monkeypatch, caplog, owned_by_another_user
):
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    # The probe's LibreOffice profile goes to the temp directory; keep the locked one out of the real one.
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    script, _ = _converter(tmp_path / "bin", template_pdf, locks_profile=True)
    owner, server_uid = os.getuid(), os.getuid() + 1 if owned_by_another_user else os.getuid()
    # The left files are the test user's; a different server uid makes them another user's, as root's are.
    monkeypatch.setattr(setup_libreoffice.os, "getuid", lambda: server_uid)

    assert setup_libreoffice.ensure_libreoffice([str(script)]) is False
    # --user only helps when the command runs as another user, so only that case suggests it.
    if owned_by_another_user:
        assert f"owned by uid {owner}, which the server (uid {server_uid}) cannot delete" in caplog.text
        assert f"--user {server_uid}:{os.getgid()}" in caplog.text
    else:
        assert "left files that the server cannot delete in" in caplog.text
        assert "--user" not in caplog.text
    for locked in scratch.glob("lo-profile-*/user"):
        locked.chmod(0o700)


def test_command_gets_absolute_paths_for_relative_deliverables(
    tmp_path, template_pdf, no_host_libreoffice, monkeypatch
):
    # A relative persist_deliverables_dir (the shipped default) gives relative deliverable paths, while a container
    # usually starts in its own working directory.
    script, log = _converter(tmp_path / "bin", template_pdf, workdir="/")
    command = ["/usr/bin/env", f"PATH={script.parent}", script.name]
    monkeypatch.chdir(tmp_path)
    comparison, rubric = Path("persist/task_a/repeat_0"), Path("persist/task_b/repeat_0")
    for directory in (comparison, rubric):
        directory.mkdir(parents=True)
    for path in (comparison / "memo.docx", comparison / "legacy.doc", rubric / "report.docx"):
        path.write_bytes(b"office source")

    assert preconvert.preconvert_dir(comparison, libreoffice_command=command) == (2, 0, [])
    blocks = file_reader.convert_deliverables_to_content_blocks(str(rubric), libreoffice_command=command)

    assert "\nreport.docx (converted to PDF):" in [block.get("text") for block in blocks]
    # One run per file: none needed the round-trip retry from a temp copy.
    assert len(_calls(log)) == 3


def test_preconvert_dir_uses_command_on_every_path(tmp_path, template_pdf, no_host_libreoffice):
    script, log = _converter(tmp_path / "bin", template_pdf)
    deliverables = tmp_path / "deliverables"
    deliverables.mkdir()
    # Parallel path, whitespace staging, and the serial same-stem sidecar path.
    for name in ("memo.docx", "Q1 notes.docx", "Plan.docx", "Plan.pptx"):
        (deliverables / name).write_bytes(b"office source")

    result = preconvert.preconvert_dir(deliverables, libreoffice_command=[str(script)])

    assert result == (4, 0, [])
    for name in ("memo.pdf", "Q1 notes.pdf", "Plan.docx.pdf", "Plan.pptx.pdf"):
        assert (deliverables / name).read_bytes() == template_pdf.read_bytes()
    assert len(_calls(log)) == 4


def test_unconfigured_default_runs_host_libreoffice_with_historical_argv(tmp_path, template_pdf, monkeypatch):
    bin_dir = tmp_path / "bin"
    _, log = _converter(bin_dir, template_pdf, name="libreoffice")
    monkeypatch.setenv("PATH", str(bin_dir))
    # Relative paths reach the host libreoffice unchanged; only a configured command gets absolute ones.
    monkeypatch.chdir(tmp_path)
    source = Path("work") / "memo.docx"
    source.parent.mkdir()
    source.write_bytes(b"office source")

    _, ok, _ = preconvert.convert_to_pdf(source)
    rendered = file_reader._convert_office_to_pdf(source, out_dir=tmp_path)

    assert ok and rendered == tmp_path / "memo.pdf"
    for call, outdir in zip(_calls(log), (source.parent, tmp_path), strict=True):
        parts = call.split(" ")
        assert parts[:5] == ["--headless", "--nologo", "--nolockcheck", "--nodefault", "--norestore"]
        assert parts[5].startswith("-env:UserInstallation=file:///")
        assert parts[6:] == ["--convert-to", "pdf", "--outdir", str(outdir), str(source)]


@pytest.mark.skipif(shutil.which("libreoffice") is None, reason="LibreOffice is not installed")
def test_real_libreoffice_through_wrapper_command(tmp_path):
    from docx import Document

    log = tmp_path / "lo-wrapper.log"
    wrapper = tmp_path / "lo-wrapper"
    wrapper.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{log}'\nexec libreoffice \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    deliverables = tmp_path / "deliverables"
    deliverables.mkdir()
    document = Document()
    document.add_paragraph("Rendered through a wrapper command.")
    document.save(deliverables / "report.docx")

    assert setup_libreoffice.ensure_libreoffice([str(wrapper)], probe_dirs=[tmp_path / "persist"]) is True
    assert preconvert.preconvert_dir(deliverables, libreoffice_command=[str(wrapper)]) == (1, 0, [])
    assert (deliverables / "report.pdf").read_bytes().startswith(b"%PDF")
    # --version, the temp-directory and persist-directory probes, then the deliverable.
    calls = _calls(log)
    assert len(calls) == 4 and calls[-1].endswith(str(deliverables / "report.docx"))
