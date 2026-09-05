# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "benchmarks/gdpval/hsg/checkpoint_e2e"
RUNTIME_SOURCE = PACKAGE / "runtime_sources/transport_assignment.py"
REFERENCE_SCOPE = REPO / "benchmarks/gdpval/tests/fixtures/gdpval_reference_asset_scope.v1.json"


def _fixture_cache_identity(version: str = "v1") -> dict[str, str]:
    return {"fixture_converter": version}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cost(module, compatible: bool, wire_bytes: int = 1000):
    return module.PairCost(
        compatible=compatible,
        wire_bytes=wire_bytes,
        raw_bytes=wire_bytes,
        max_file_bytes=wire_bytes,
        reasons=() if compatible else ("serialized_request_over_cap",),
    )


def test_zip_footprint_counts_uncompressed_binary_members_and_av(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_assignment_zip_footprint", RUNTIME_SOURCE)
    repeat = tmp_path / "repeat_0"
    reference_files = repeat / "reference_files"
    reference_files.mkdir(parents=True)

    with zipfile.ZipFile(repeat / "candidate.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("audio/source.wav", b"w" * 257)
        archive.writestr("documents/report.pdf", b"p" * 113)
        archive.writestr("notes/readme.txt", b"ignored")
    with zipfile.ZipFile(reference_files / "reference.zip", "w") as archive:
        archive.writestr("images/chart.png", b"i" * 389)

    footprint = module._footprint(repeat)

    assert footprint == module.Footprint(
        raw_bytes=257 + 113 + 389,
        max_file_bytes=389,
        has_av=True,
        file_count=3,
    )
    assert (repeat / "candidate.zip").stat().st_size < footprint.raw_bytes


def test_footprint_recurses_only_for_reference_assets(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_assignment_recursive_refs", RUNTIME_SOURCE)
    repeat = tmp_path / "repeat_0"
    nested = repeat / "reference_files/asset-a"
    nested.mkdir(parents=True)
    (repeat / "candidate.pdf").write_bytes(b"candidate")
    (nested / "Plan.docx").write_bytes(b"office source is not a native attachment")
    (nested / "Plan.docx.pdf").write_bytes(b"reference render")

    candidate = module._footprint(repeat, include_reference_files=False)
    reference = module._footprint(repeat, include_reference_files=True)

    assert candidate == module.Footprint(
        raw_bytes=len(b"candidate"),
        max_file_bytes=len(b"candidate"),
        has_av=False,
        file_count=1,
    )
    assert reference == module.Footprint(
        raw_bytes=len(b"candidate") + len(b"reference render"),
        max_file_bytes=len(b"reference render"),
        has_av=False,
        file_count=2,
    )


def test_frozen_dataset_reference_asset_scope_is_exactly_125_of_220() -> None:
    receipt = json.loads(REFERENCE_SCOPE.read_text())
    assert receipt == {
        "all_paths_nested": True,
        "projection_sha256": "77904a9b323e909bb58e6738ebe90edf6d091591448a654fc4e9b1e896df7c26",
        "reference_occurrences": 261,
        "schema": "gdpval.reference-asset-scope.v1",
        "source_dataset_sha256": ("85f4b36317292f417c79dcad97af41c49704e6d0553b5f79f8b54349d0641774"),
        "source_rows": 220,
        "tasks_with_references": 125,
        "tasks_without_references": 95,
    }
    assert receipt["tasks_with_references"] + receipt["tasks_without_references"] == 220


def test_nested_reference_zip_over_cap_is_rejected_before_assignment(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_assignment_nested_zip_cap", RUNTIME_SOURCE)
    repeat = tmp_path / "repeat_0"
    nested = repeat / "reference_files/asset-zip"
    nested.mkdir(parents=True)
    with zipfile.ZipFile(nested / "evidence.zip", "w") as archive:
        archive.writestr("documents/one.pdf", b"a" * 700)
        archive.writestr("documents/two.pdf", b"b" * 701)

    reference = module._footprint(repeat, include_reference_files=True)
    assert reference.raw_bytes == 1401
    assert reference.file_count == 2
    pair = module._pair_cost(
        module.Footprint(raw_bytes=10, max_file_bytes=10, has_av=True, file_count=1),
        reference,
        max_file_bytes=10_000,
        max_raw_bytes=1000,
        max_wire_bytes=1900,
        framing_reserve_bytes=100,
    )
    assert pair.compatible is False
    assert pair.reasons == ("aggregate_raw_over_cap", "serialized_request_over_cap")


def test_exact_float_wav_pair_fits_only_the_guarded_gemini_near_cap_route() -> None:
    module = _load("checkpoint_transport_assignment_float_wav_cap", RUNTIME_SOURCE)
    candidate = module.Footprint(
        raw_bytes=315_977_664,
        max_file_bytes=52_662_944,
        has_av=True,
        file_count=6,
    )
    qwen36 = module.Footprint(
        raw_bytes=48_034_279,
        max_file_bytes=14_941_511,
        has_av=True,
        file_count=2,
    )

    old = module._pair_cost(
        candidate,
        qwen36,
        max_file_bytes=335_544_320,
        max_raw_bytes=330_301_440,
        max_wire_bytes=440_401_920,
        framing_reserve_bytes=4_194_304,
    )
    guarded = module._pair_cost(
        candidate,
        qwen36,
        max_file_bytes=335_544_320,
        max_raw_bytes=368_000_000,
        max_wire_bytes=495_000_000,
        framing_reserve_bytes=4_194_304,
    )

    assert old.compatible is False
    assert guarded == module.PairCost(
        compatible=True,
        wire_bytes=489_543_564,
        raw_bytes=364_011_943,
        max_file_bytes=52_662_944,
        reasons=(),
    )


def test_footprint_deduplicates_only_exact_av_payloads_across_zip_and_loose_files(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_assignment_exact_av_dedup", RUNTIME_SOURCE)
    repeat = tmp_path / "repeat_0"
    repeat.mkdir()
    shared = b"float-wav-payload" * 41
    different = b"other-wav-payload".ljust(len(shared), b"!")
    (repeat / "loose.wav").write_bytes(shared)
    with zipfile.ZipFile(repeat / "submission.zip", "w") as archive:
        archive.writestr("stems/duplicate.wav", shared)
        archive.writestr("stems/same-size-different.wav", different)

    footprint = module._footprint(repeat)

    assert footprint == module.Footprint(
        raw_bytes=len(shared) + len(different),
        max_file_bytes=len(shared),
        has_av=True,
        file_count=2,
    )


def test_zip_rewrite_losslessly_replaces_only_eligible_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("checkpoint_transport_views_zip_rewrite", PACKAGE / "transport_views.py")
    source = tmp_path / "submission.zip"
    destination = tmp_path / "submission.__gdpval_lossless__.zip"
    original_audio = b"uncompressed-pcm" * 32
    original_pdf = b"%PDF fixture"
    original_small_audio = b"small"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("media/recording.wav", original_audio)
        archive.writestr("media/small.wav", original_small_audio)
        archive.writestr("evidence/report.pdf", original_pdf)

    converted_audio = b"flac" * 11

    def fake_convert_lossless(source_member: Path, destination_member: Path):
        assert source_member.read_bytes() == original_audio
        destination_member.write_bytes(converted_audio)
        return {
            "source_sha256": hashlib.sha256(original_audio).hexdigest(),
            "output_sha256": hashlib.sha256(converted_audio).hexdigest(),
            "source_bytes": len(original_audio),
            "output_bytes": len(converted_audio),
            "audio_identity": {"decoded_pcm_int32_sha256": "fixture"},
        }

    monkeypatch.setattr(module, "_convert_lossless", fake_convert_lossless)
    result = module._convert_zip_lossless(source, destination, min_audio_bytes=64)

    assert result is not None
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert result["member_conversions"] == [
        {
            "source_sha256": hashlib.sha256(original_audio).hexdigest(),
            "output_sha256": hashlib.sha256(converted_audio).hexdigest(),
            "source_bytes": len(original_audio),
            "output_bytes": len(converted_audio),
            "audio_identity": {"decoded_pcm_int32_sha256": "fixture"},
            "source_member": "media/recording.wav",
            "output_member": "media/recording.__gdpval_lossless__.flac",
            "kind": "lossless_flac",
            "derivative_profile": module.DERIVATIVE_PROFILE,
        }
    ]
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "media/recording.__gdpval_lossless__.flac",
            "media/small.wav",
            "evidence/report.pdf",
        }
        assert archive.read("media/recording.__gdpval_lossless__.flac") == converted_audio
        assert archive.read("media/small.wav") == original_small_audio
        assert archive.read("evidence/report.pdf") == original_pdf


def test_lossless_audio_conversion_has_an_independent_ffmpeg_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("checkpoint_transport_views_ffmpeg_timeout", PACKAGE / "transport_views.py")
    source = tmp_path / "source.wav"
    destination = tmp_path / "destination.flac"
    source.write_bytes(b"source-pcm" * 100)
    identity = {
        "samplerate": 48000,
        "channels": 2,
        "frames": 100,
        "subtype": "PCM_16",
        "decoded_pcm_int32_sha256": "fixture",
    }
    monkeypatch.setattr(module, "_audio_identity", lambda _path: dict(identity))
    observed: dict[str, int] = {}

    def fake_run(arguments, *, check, timeout):
        assert check is True
        observed["timeout"] = timeout
        Path(arguments[-1]).write_bytes(b"flac")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module._convert_lossless(source, destination)

    assert result is not None
    assert observed == {"timeout": 1800}
    assert destination.read_bytes() == b"flac"


def test_transport_derivative_profile_is_exact_and_accepts_plus() -> None:
    module = _load("checkpoint_transport_views_profile", PACKAGE / "transport_views.py")

    assert module.DEFAULT_MIN_VIDEO_BYTES == 8 * 1024 * 1024
    assert module._validate_derivative_profile(module.DERIVATIVE_PROFILE) == (
        "reference-pdf-v1+video-h264-720p-crf26-aac128-min8m+ref-video-bundle8-v1"
    )
    with pytest.raises(ValueError, match="unsupported derivative_profile"):
        module._validate_derivative_profile("reference-pdf-v2")
    with pytest.raises(ValueError, match="must match"):
        module._validate_derivative_profile("../unsafe")


def test_h264_video_proxy_uses_exact_profile_and_validates_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("checkpoint_transport_views_h264", PACKAGE / "transport_views.py")
    source = tmp_path / "source.mov"
    destination = tmp_path / "proxy.mp4"
    source.write_bytes(b"source-video")
    source_identity = {
        "duration_seconds": 10.0,
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "prores", "width": 1920, "height": 1080},
            {"index": 1, "codec_type": "audio", "codec_name": "pcm_s16le", "channels": 2},
        ],
    }
    output_identity = {
        "duration_seconds": 10.01,
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
            },
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 2},
        ],
    }
    probes = iter((source_identity, output_identity))
    monkeypatch.setattr(module, "_probe_video", lambda _path: next(probes))
    observed: dict[str, object] = {}

    def fake_run(arguments, *, check, timeout):
        observed.update(arguments=arguments, check=check, timeout=timeout)
        Path(arguments[-1]).write_bytes(b"deterministic-mp4")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module._convert_video(source, destination)

    assert observed["check"] is True
    assert observed["timeout"] == module.FFMPEG_TIMEOUT_SECONDS
    arguments = observed["arguments"]
    assert "libx264" in arguments
    assert arguments[arguments.index("-crf") + 1] == "26"
    assert arguments[arguments.index("-b:a") + 1] == "128k"
    assert "force_original_aspect_ratio=decrease" in arguments[arguments.index("-vf") + 1]
    assert result["ffmpeg_args"] == list(module.VIDEO_FFMPEG_ARGS_TEMPLATE)
    assert result["ffprobe_args"] == list(module.VIDEO_FFPROBE_ARGS_TEMPLATE)
    assert result["source_video_identity"] == source_identity
    assert result["output_video_identity"] == output_identity
    assert destination.read_bytes() == b"deterministic-mp4"


def test_office_handoff_is_injective_when_converter_leaves_libreoffice_stem_pdf(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fitz")
    module = _load("checkpoint_transport_views_office_handoff", PACKAGE / "transport_views.py")
    source_root = tmp_path / "sources"
    output_root = tmp_path / "view"
    source_root.mkdir()
    output_root.mkdir()
    sources = [source_root / "Plan.docx", source_root / "Plan.pptx"]
    for source in sources:
        source.write_bytes(source.suffix.encode())

    requested_outputs: list[str] = []

    def convert_to_pdf(input_path: Path, output_pdf: Path | None = None):
        assert output_pdf is not None
        requested_outputs.append(output_pdf.name)
        # This is the spelling LibreOffice's batch converter actually emits,
        # even when an adapter accepts a more specific output_pdf argument.
        input_path.with_suffix(".pdf").write_bytes(module._deterministic_text_pdf([input_path.suffix.encode()]))
        return (
            input_path,
            False,
            f"libreoffice rc=0 did not produce {output_pdf.name}: fixture",
        )

    preconvert = types.SimpleNamespace(convert_to_pdf=convert_to_pdf)
    results = {}
    for source in sources:
        destination = output_root / f"{source.name}.pdf"
        results[source.suffix] = module._convert_office_pdf(
            source,
            destination,
            preconvert,
            "a" * 64,
        )

    assert requested_outputs == ["Plan.docx.pdf", "Plan.pptx.pdf"]
    assert {result["converter_output_handoff"] for result in results.values()} == {"libreoffice_stem"}
    assert {result["converter_status_override"] for result in results.values()} == {"libreoffice_rc0_stem_handoff"}
    assert (output_root / "Plan.docx.pdf").is_file()
    assert (output_root / "Plan.pptx.pdf").is_file()
    assert results[".docx"]["output_sha256"] != results[".pptx"]["output_sha256"]
    assert not (output_root / "Plan.pdf").exists()


def test_office_handoff_rejects_conflicting_requested_and_libreoffice_outputs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fitz")
    module = _load("checkpoint_transport_views_office_ambiguity", PACKAGE / "transport_views.py")
    source = tmp_path / "Plan.docx"
    destination = tmp_path / "Plan.docx.pdf"
    source.write_bytes(b"document")

    def convert_to_pdf(input_path: Path, output_pdf: Path | None = None):
        assert output_pdf is not None
        output_pdf.write_bytes(module._deterministic_text_pdf([b"requested output"]))
        input_path.with_suffix(".pdf").write_bytes(module._deterministic_text_pdf([b"different LibreOffice output"]))
        return input_path, True, "fixture"

    with pytest.raises(RuntimeError, match="ambiguous PDF outputs"):
        module._convert_office_pdf(
            source,
            destination,
            types.SimpleNamespace(convert_to_pdf=convert_to_pdf),
            "b" * 64,
        )
    assert not destination.exists()


def test_office_conversion_repairs_only_missing_external_relationship_closures(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fitz")
    module = _load("checkpoint_transport_views_ooxml_repair", PACKAGE / "transport_views.py")
    source = tmp_path / "Recreare_Contract_Outline.docx"
    destination = tmp_path / "Recreare_Contract_Outline.docx.pdf"
    malformed_relationships = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId1" Type="http://example.test/relationships/hyperlink" '
        b'Target="https://one.example" TargetMode="External"\n'
        b'<Relationship Id="rId2" Type="http://example.test/relationships/hyperlink" '
        b'Target="https://two.example" TargetMode="External"\n'
        b'<Relationship Id="rId3" Type="http://example.test/relationships/hyperlink" '
        b'Target="https://three.example" TargetMode="External"\n'
        b'<Relationship Id="rId4" Type="styles" Target="styles.xml"/>'
        b"</Relationships>"
    )
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types />")
        archive.writestr("word/document.xml", b"<document />")
        archive.writestr("word/_rels/document.xml.rels", malformed_relationships)
    source_bytes = source.read_bytes()
    observed: dict[str, object] = {}

    def convert_to_pdf(input_path: Path, output_pdf: Path | None = None):
        assert output_pdf is not None
        observed["staged_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
        with zipfile.ZipFile(input_path) as archive:
            relationships = archive.read("word/_rels/document.xml.rels")
        root = module.ElementTree.fromstring(relationships)
        observed["relationship_count"] = len(root)
        observed["external_count"] = sum(child.attrib.get("TargetMode") == "External" for child in root)
        output_pdf.write_bytes(module._deterministic_text_pdf([b"repaired document"]))
        return input_path, True, "fixture"

    result = module._convert_office_pdf(
        source,
        destination,
        types.SimpleNamespace(convert_to_pdf=convert_to_pdf),
        "c" * 64,
    )

    repair = result["ooxml_repair"]
    assert source.read_bytes() == source_bytes
    assert repair == {
        "schema": module.OOXML_REPAIR_SCHEMA,
        "repair_scope": "staged_copy_only",
        "before_package_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "after_package_sha256": observed["staged_sha256"],
        "members": repair["members"],
        "member_count": 1,
        "insertion_count": 3,
    }
    assert repair["members"][0]["member"] == "word/_rels/document.xml.rels"
    assert repair["members"][0]["repair"] == ("close_missing_external_relationship_empty_element")
    assert repair["members"][0]["inserted_bytes_hex"] == "2f3e"
    assert repair["members"][0]["insertion_count"] == 3
    assert repair["members"][0]["after_bytes"] == (repair["members"][0]["before_bytes"] + 6)
    assert observed["relationship_count"] == 4
    assert observed["external_count"] == 3
    assert destination.is_file()


def test_office_conversion_rejects_other_malformed_ooxml_without_calling_converter(
    tmp_path: Path,
) -> None:
    module = _load("checkpoint_transport_views_ooxml_repair_closed", PACKAGE / "transport_views.py")
    source = tmp_path / "unsupported.docx"
    destination = tmp_path / "unsupported.docx.pdf"
    malformed_relationships = (
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId1" Type="internal" Target="styles.xml"\n'
        b'<Relationship Id="rId2" Type="styles" Target="styles.xml"/>'
        b"</Relationships>"
    )
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types />")
        archive.writestr("word/document.xml", b"<document />")
        archive.writestr("word/_rels/document.xml.rels", malformed_relationships)
    source_bytes = source.read_bytes()
    calls = 0

    def convert_to_pdf(_input_path: Path):
        nonlocal calls
        calls += 1
        raise AssertionError("unsupported malformed package reached converter")

    with pytest.raises(RuntimeError, match="unsupported malformed OOXML relationships member"):
        module._convert_office_pdf(
            source,
            destination,
            types.SimpleNamespace(convert_to_pdf=convert_to_pdf),
            "d" * 64,
        )
    assert calls == 0
    assert source.read_bytes() == source_bytes
    assert not destination.exists()


def test_zip_rewrite_proxies_video_alongside_lossless_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("checkpoint_transport_views_zip_av", PACKAGE / "transport_views.py")
    source = tmp_path / "submission.zip"
    destination = tmp_path / "submission.__gdpval_transport__.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("media/audio.wav", b"audio-source")
        archive.writestr("media/video.mov", b"video-source")
        archive.writestr("notes/readme.txt", b"preserved")

    def fake_audio(input_path: Path, output_path: Path):
        output_path.write_bytes(b"flac-output")
        return {
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "source_bytes": input_path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
            "audio_identity": {"decoded_pcm_int32_sha256": "fixture"},
        }

    def fake_video(input_path: Path, output_path: Path):
        output_path.write_bytes(b"mp4-output")
        return {
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "source_bytes": input_path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
            "ffmpeg_args": list(module.VIDEO_FFMPEG_ARGS_TEMPLATE),
            "ffprobe_args": list(module.VIDEO_FFPROBE_ARGS_TEMPLATE),
            "source_video_identity": {"duration_seconds": 1.0, "streams": []},
            "output_video_identity": {"duration_seconds": 1.0, "streams": []},
        }

    monkeypatch.setattr(module, "_convert_lossless", fake_audio)
    monkeypatch.setattr(module, "_convert_video", fake_video)
    result = module._convert_zip_lossless(
        source,
        destination,
        min_audio_bytes=1,
        min_video_bytes=1,
    )

    assert result is not None
    assert result["member_conversion_counts"] == {"h264_video": 1, "lossless_flac": 1}
    assert [item["kind"] for item in result["member_conversions"]] == [
        "lossless_flac",
        "h264_video",
    ]
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "media/audio.__gdpval_lossless__.flac",
            "media/video.__gdpval_h264__.mp4",
            "notes/readme.txt",
        }
        assert archive.read("media/audio.__gdpval_lossless__.flac") == b"flac-output"
        assert archive.read("media/video.__gdpval_h264__.mp4") == b"mp4-output"
        assert archive.read("notes/readme.txt") == b"preserved"


def test_zip_mp4_proxy_uses_distinct_workspace_output_and_preserves_source_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("checkpoint_transport_views_zip_mp4_source", PACKAGE / "transport_views.py")
    source = tmp_path / "submission.zip"
    destination = tmp_path / "submission.__gdpval_transport__.zip"
    source_video = b"original-1080p-mp4"
    proxy_video = b"derived-720p-h264"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("reel footage/clip.mp4", source_video)
    source_zip_bytes = source.read_bytes()
    source_identity = {
        "duration_seconds": 3.0,
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p",
            }
        ],
    }
    output_identity = {
        "duration_seconds": 3.0,
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
            }
        ],
    }
    observed: dict[str, str] = {}

    def fake_video(input_path: Path, output_path: Path):
        observed["input"] = input_path.name
        observed["output"] = output_path.name
        assert input_path != output_path
        assert input_path.read_bytes() == source_video
        output_path.write_bytes(proxy_video)
        assert input_path.read_bytes() == source_video
        return {
            "source_sha256": hashlib.sha256(source_video).hexdigest(),
            "output_sha256": hashlib.sha256(proxy_video).hexdigest(),
            "source_bytes": len(source_video),
            "output_bytes": len(proxy_video),
            "ffmpeg_args": list(module.VIDEO_FFMPEG_ARGS_TEMPLATE),
            "ffprobe_args": list(module.VIDEO_FFPROBE_ARGS_TEMPLATE),
            "source_video_identity": source_identity,
            "output_video_identity": output_identity,
        }

    def fake_probe(path: Path):
        payload = path.read_bytes()
        if payload == source_video:
            return source_identity
        if payload == proxy_video:
            return output_identity
        raise AssertionError(f"unexpected video payload: {payload!r}")

    monkeypatch.setattr(module, "_convert_video", fake_video)
    monkeypatch.setattr(module, "_probe_video", fake_probe)
    result = module._convert_zip_lossless(
        source,
        destination,
        min_audio_bytes=1024,
        min_video_bytes=1,
    )

    assert result is not None
    assert observed == {"input": "member_0.mp4", "output": "member_0.derived.mp4"}
    assert source.read_bytes() == source_zip_bytes
    conversion = result["member_conversions"][0]
    assert conversion["source_sha256"] == hashlib.sha256(source_video).hexdigest()
    assert conversion["output_sha256"] == hashlib.sha256(proxy_video).hexdigest()
    assert conversion["source_sha256"] != conversion["output_sha256"]
    with zipfile.ZipFile(source) as archive:
        assert archive.read("reel footage/clip.mp4") == source_video
    with zipfile.ZipFile(destination) as archive:
        assert archive.read("reel footage/clip.__gdpval_h264__.mp4") == proxy_video
    module._validate_transport_zip(result, source, destination)


def test_recursive_reference_zip_bundles_all_13_videos_into_eight_audited_attachments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("checkpoint_transport_views_reference_video_bundles", PACKAGE / "transport_views.py")
    source = tmp_path / "reel footage.zip"
    destination = tmp_path / "reel footage.__gdpval_transport__.zip"
    video_names = [
        "reel footage/logos.mp4",
        "reel footage/CastleExplosion(TyFlow+Phoenix).mp4",
        "reel footage/Shores_Comp_04222020.mp4",
        "reel footage/BuildingExplosion+Destruction(TyFlow+Phoenix).mp4",
        "reel footage/Helicopter_DustSim(TyFlow+Phoenix).mp4",
        "reel footage/4 Rooms(rotoScopingTest_AfterEffects).mp4",
        *(f"reel footage/clip_{index:02d}.mp4" for index in range(6, 12)),
        "reel footage/logo_2.mp4",
    ]
    assert len(video_names) == 13
    payload_by_name = {name: f"source-video-{index:02d}".encode() for index, name in enumerate(video_names)}
    # Deliberately write the archive in reverse order: the transport contract is
    # normalized member-path order rather than incidental central-directory order.
    with zipfile.ZipFile(source, "w") as archive:
        for name in reversed(video_names):
            archive.writestr(name, payload_by_name[name])
        archive.writestr("reel footage/readme.txt", b"preserved")
    source_bytes = source.read_bytes()

    probe_by_payload: dict[bytes, dict[str, object]] = {}

    def source_identity(index: int) -> dict[str, object]:
        duration = float(index + 1)
        return {
            "duration_seconds": duration,
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "prores",
                    "width": 1920,
                    "height": 1080,
                    "duration": str(duration),
                }
            ],
        }

    def normalized_identity(duration: float) -> dict[str, object]:
        return {
            "duration_seconds": duration,
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "pix_fmt": "yuv420p",
                    "duration": str(duration),
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "48000",
                },
            ],
        }

    index_by_payload = {payload: index for index, payload in enumerate(payload_by_name.values())}
    for payload, index in index_by_payload.items():
        probe_by_payload[payload] = source_identity(index)

    def fake_normalize(input_path: Path, output_path: Path):
        source_payload = input_path.read_bytes()
        index = index_by_payload[source_payload]
        output_payload = b"normalized:" + source_payload
        output_path.write_bytes(output_payload)
        source_video_identity = source_identity(index)
        output_video_identity = normalized_identity(float(index + 1))
        probe_by_payload[output_payload] = output_video_identity
        return {
            "source_sha256": hashlib.sha256(source_payload).hexdigest(),
            "source_bytes": len(source_payload),
            "source_video_identity": source_video_identity,
            "source_duration_seconds": float(index + 1),
            "normalized_sha256": hashlib.sha256(output_payload).hexdigest(),
            "normalized_bytes": len(output_payload),
            "normalized_video_identity": output_video_identity,
            "normalization_audio": "deterministic_silence",
            "normalization_filter": module.REFERENCE_VIDEO_BUNDLE_FILTER,
            "normalization_ffmpeg_args": ["fixture"],
        }

    def fake_concatenate(normalized, output_path: Path):
        output_payload = b"bundle:" + b"|".join(path.read_bytes() for path, _receipt in normalized)
        output_path.write_bytes(output_payload)
        duration = sum(
            float(receipt["normalized_video_identity"]["duration_seconds"]) for _path, receipt in normalized
        )
        identity = normalized_identity(duration)
        probe_by_payload[output_payload] = identity
        return {
            "output_sha256": hashlib.sha256(output_payload).hexdigest(),
            "output_bytes": len(output_payload),
            "output_video_identity": identity,
            "expected_duration_seconds": duration,
            "concat_ffmpeg_args": ["fixture"],
        }

    def fake_probe(path: Path):
        return probe_by_payload[path.read_bytes()]

    monkeypatch.setattr(module, "_normalize_reference_video_for_bundle", fake_normalize)
    monkeypatch.setattr(module, "_concatenate_reference_video_bundle", fake_concatenate)
    monkeypatch.setattr(module, "_probe_video", fake_probe)

    result = module._convert_zip_lossless(
        source,
        destination,
        min_audio_bytes=1024,
        min_video_bytes=module.DEFAULT_MIN_VIDEO_BYTES,
        reference_pdf_derivatives=True,
    )

    assert result is not None
    assert source.read_bytes() == source_bytes
    receipt = result["reference_video_bundles"]
    expected_order = sorted(video_names)
    assert receipt["source_order"] == expected_order
    assert receipt["source_video_count"] == 13
    assert receipt["output_video_count"] == 8
    assert [bundle["clip_count"] for bundle in receipt["bundles"]] == [2, 2, 2, 2, 2, 1, 1, 1]
    assert [clip["source_member"] for bundle in receipt["bundles"] for clip in bundle["clips"]] == expected_order
    assert result["member_conversion_counts"] == {"reference_video_bundle": 8}

    with zipfile.ZipFile(destination) as archive:
        physical_videos = sorted(name for name in archive.namelist() if name.lower().endswith(".mp4"))
        assert physical_videos == sorted(bundle["output_member"] for bundle in receipt["bundles"])
        assert len(physical_videos) == 8
        assert archive.read("reel footage/readme.txt") == b"preserved"
        manifest = archive.read(module.REFERENCE_VIDEO_BUNDLE_MANIFEST).decode()
        assert all(name in manifest for name in expected_order)
        assert all(name not in archive.namelist() for name in video_names)

    module._validate_transport_zip(result, source, destination)


def test_nested_reference_zip_adds_ambiguous_office_step_and_psd_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fitz")
    image_module = pytest.importorskip("PIL.Image")
    module = _load("checkpoint_transport_views_zip_reference_pdfs", PACKAGE / "transport_views.py")
    source = tmp_path / "references.zip"
    destination = tmp_path / "references.__gdpval_transport__.zip"
    legacy_pdf = module._deterministic_text_pdf([b"ambiguous legacy render"])
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("docs/Plan.docx", b"document")
        archive.writestr("docs/Plan.pptx", b"presentation")
        archive.writestr("docs/Plan.pdf", legacy_pdf)
        archive.writestr("cad/model.step", b"ISO-10303-21;\nEND-ISO-10303-21;\n")
        archive.writestr("art/Artwork.psd", b"fixture-psd")
        archive.writestr("notes/readme.txt", b"preserved")

    office_calls: list[str] = []

    def convert_to_pdf(input_path: Path, output_pdf: Path | None = None):
        assert output_pdf is not None
        office_calls.append(input_path.name)
        output_pdf.write_bytes(module._deterministic_text_pdf([input_path.name.encode()]))
        return input_path, True, "fixture"

    original_open = image_module.open
    fixture = image_module.new("RGBA", (4, 3), (20, 40, 60, 200))

    def fake_open(value, *args, **kwargs):
        if isinstance(value, (str, os.PathLike)) and Path(value).suffix.lower() == ".psd":
            opened = fixture.copy()
            opened.format = "PSD"
            return opened
        return original_open(value, *args, **kwargs)

    monkeypatch.setattr(image_module, "open", fake_open)
    result = module._convert_zip_lossless(
        source,
        destination,
        min_audio_bytes=1024,
        min_video_bytes=module.DEFAULT_MIN_VIDEO_BYTES,
        derivative_cache=module._DerivativeCache(
            tmp_path / "cache", module.DERIVATIVE_PROFILE, _fixture_cache_identity()
        ),
        reference_pdf_derivatives=True,
        preconvert_module=types.SimpleNamespace(convert_to_pdf=convert_to_pdf),
        preconvert_module_sha256="b" * 64,
    )

    assert result is not None
    assert sorted(office_calls) == ["Plan.docx", "Plan.pptx"]
    assert result["member_conversion_counts"] == {
        "office_pdf": 2,
        "psd_flattened_pdf": 1,
        "step_text_pdf": 1,
    }
    assert all(
        conversion["derivative_profile"] == module.DERIVATIVE_PROFILE for conversion in result["member_conversions"]
    )
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "docs/Plan.docx",
            "docs/Plan.pptx",
            "docs/Plan.pdf",
            "docs/Plan.docx.pdf",
            "docs/Plan.pptx.pdf",
            "cad/model.step",
            "cad/model.step.pdf",
            "art/Artwork.psd",
            "art/Artwork.psd.pdf",
            "notes/readme.txt",
        }
        assert archive.read("docs/Plan.pdf") == legacy_pdf
        assert archive.read("notes/readme.txt") == b"preserved"

    source_stat = source.stat()
    entry = {
        **result,
        "source": str(source),
        "source_mtime_ns": source_stat.st_mtime_ns,
        "kind": "transport_zip",
    }
    module._validate_derived_file(entry, destination)


def test_nested_reference_zip_fails_closed_on_unsupported_member(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_views_zip_unsupported", PACKAGE / "transport_views.py")
    source = tmp_path / "references.zip"
    destination = tmp_path / "references.__gdpval_transport__.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("cad/model.blend", b"unsupported")

    with pytest.raises(RuntimeError, match="unsupported recursive reference ZIP member: cad/model.blend"):
        module._convert_zip_lossless(
            source,
            destination,
            min_audio_bytes=1024,
            min_video_bytes=module.DEFAULT_MIN_VIDEO_BYTES,
            reference_pdf_derivatives=True,
        )
    assert not destination.exists()


def test_derivative_cache_converts_identical_assets_once(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_views_cache", PACKAGE / "transport_views.py")
    cache = module._DerivativeCache(tmp_path / "cache", module.DERIVATIVE_PROFILE, _fixture_cache_identity())
    source = tmp_path / "source.bin"
    first = tmp_path / "one/output.bin"
    second = tmp_path / "two/output.bin"
    source.write_bytes(b"identical-source")
    calls = 0

    def converter(input_path: Path, output_path: Path):
        nonlocal calls
        calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes() + b"-derived")
        return {
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "source_bytes": input_path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
        }

    first_receipt = cache.materialize("fixture", source, first, {"version": 1}, converter)
    second_receipt = cache.materialize("fixture", source, second, {"version": 1}, converter)

    assert calls == 1
    assert first.read_bytes() == second.read_bytes() == b"identical-source-derived"
    assert first_receipt["cache"] == {
        "schema": module.DERIVATIVE_CACHE_SCHEMA,
        "key": second_receipt["cache"]["key"],
        "reused": False,
        "materialization": "generated",
    }
    assert second_receipt["cache"]["reused"] is True
    assert second_receipt["cache"]["materialization"] in {"hardlink", "copy"}
    assert first.stat().st_mode & 0o222 == 0
    assert second.stat().st_mode & 0o222 == 0


def test_derivative_cache_reuses_objects_across_instances_and_separates_identities(
    tmp_path: Path,
) -> None:
    module = _load("checkpoint_transport_views_persistent_cache", PACKAGE / "transport_views.py")
    root = tmp_path / "shared-cache"
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable-source")
    calls = 0

    def converter(input_path: Path, output_path: Path):
        nonlocal calls
        calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes() + b"-derived")
        return {
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "source_bytes": input_path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
        }

    first_cache = module._DerivativeCache(
        root, module.DERIVATIVE_PROFILE, _fixture_cache_identity("one"), persistent=True
    )
    first = first_cache.materialize("fixture", source, tmp_path / "first.bin", {"version": 1}, converter)
    second_cache = module._DerivativeCache(
        root, module.DERIVATIVE_PROFILE, _fixture_cache_identity("one"), persistent=True
    )
    second = second_cache.materialize("fixture", source, tmp_path / "second.bin", {"version": 1}, converter)
    changed_cache = module._DerivativeCache(
        root, module.DERIVATIVE_PROFILE, _fixture_cache_identity("two"), persistent=True
    )
    changed = changed_cache.materialize("fixture", source, tmp_path / "changed.bin", {"version": 1}, converter)

    assert calls == 2
    assert first["cache"]["reused"] is False
    assert second["cache"]["reused"] is True
    assert second["cache"]["key"] == first["cache"]["key"]
    assert changed["cache"]["key"] != first["cache"]["key"]
    assert second_cache.summary()["hits"] == 1
    assert changed_cache.summary()["misses"] == 1


def test_derivative_cache_quarantines_corruption_and_rebuilds(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_views_cache_corruption", PACKAGE / "transport_views.py")
    root = tmp_path / "shared-cache"
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable-source")
    calls = 0

    def converter(input_path: Path, output_path: Path):
        nonlocal calls
        calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes() + b"-derived")
        return {
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "source_bytes": input_path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
        }

    first_cache = module._DerivativeCache(root, module.DERIVATIVE_PROFILE, _fixture_cache_identity(), persistent=True)
    first = first_cache.materialize("fixture", source, tmp_path / "first.bin", {"version": 1}, converter)
    object_path = first_cache._object_path(first["cache"]["key"])
    artifact = next(path for path in object_path.iterdir() if path.name.startswith("artifact"))
    artifact.chmod(0o600)
    artifact.write_bytes(b"corrupt")

    repaired_cache = module._DerivativeCache(
        root, module.DERIVATIVE_PROFILE, _fixture_cache_identity(), persistent=True
    )
    repaired_output = tmp_path / "repaired.bin"
    repaired = repaired_cache.materialize("fixture", source, repaired_output, {"version": 1}, converter)

    assert calls == 2
    assert repaired_output.read_bytes() == b"stable-source-derived"
    assert repaired["cache"]["reused"] is False
    assert repaired["cache"]["repaired_corruption"] is True
    assert repaired_cache.summary()["corruptions_quarantined"] == 1
    assert len(list((root / "quarantine").iterdir())) == 1


def test_derivative_cache_key_lock_converges_concurrent_builders(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_views_cache_lock", PACKAGE / "transport_views.py")
    root = tmp_path / "shared-cache"
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable-source")
    caches = [
        module._DerivativeCache(root, module.DERIVATIVE_PROFILE, _fixture_cache_identity(), persistent=True)
        for _ in range(2)
    ]
    calls = 0

    def converter(input_path: Path, output_path: Path):
        nonlocal calls
        calls += 1
        time.sleep(0.1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes() + b"-derived")
        return {
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "source_bytes": input_path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
        }

    def materialize(index: int):
        return caches[index].materialize(
            "fixture",
            source,
            tmp_path / f"output-{index}.bin",
            {"version": 1},
            converter,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(executor.map(materialize, range(2)))

    assert calls == 1
    assert sorted(receipt["cache"]["reused"] for receipt in receipts) == [False, True]
    assert (tmp_path / "output-0.bin").read_bytes() == b"stable-source-derived"
    assert (tmp_path / "output-1.bin").read_bytes() == b"stable-source-derived"


def test_step_reference_pdf_is_full_deterministic_and_self_validating(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    module = _load("checkpoint_transport_views_step", PACKAGE / "transport_views.py")
    source = tmp_path / "assembly.step"
    first = tmp_path / "assembly.step.pdf"
    second = tmp_path / "assembly-copy.step.pdf"
    body = ["ISO-10303-21;", "HEADER;"]
    body.extend(f"#{index}=CARTESIAN_POINT('',({index}.0,0.0,0.0));" for index in range(250))
    body.extend(["ENDSEC;", "END-ISO-10303-21;"])
    source.write_text("\n".join(body) + "\n", encoding="utf-8")

    first_receipt = module._convert_step_pdf(source, first)
    second_receipt = module._convert_step_pdf(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["output_sha256"] == second_receipt["output_sha256"]
    assert first_receipt["logical_line_count"] == len(body) + 1
    assert first_receipt["pdf_identity"]["page_count"] > 1
    with fitz.open(first) as document:
        rendered = "".join(page.get_text() for page in document)
    assert "ISO-10303-21" in rendered
    assert "END-ISO-10303-21" in rendered

    source_stat = source.stat()
    entry = {
        **first_receipt,
        "source": str(source),
        "source_mtime_ns": source_stat.st_mtime_ns,
        "kind": "step_text_pdf",
    }
    module._validate_derived_file(entry, first)
    first.chmod(0o600)
    with pytest.raises(ValueError, match="output is writable"):
        module._validate_derived_file(entry, first)


def test_psd_reference_pdf_embeds_lossless_flattened_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fitz")
    image_module = pytest.importorskip("PIL.Image")
    module = _load("checkpoint_transport_views_psd", PACKAGE / "transport_views.py")
    source = tmp_path / "layered.psd"
    first = tmp_path / "layered.psd.pdf"
    second = tmp_path / "layered-copy.psd.pdf"
    source.write_bytes(b"fixture-psd-source")
    original_open = image_module.open
    fixture = image_module.new("RGBA", (7, 5), (10, 20, 30, 128))

    def fake_open(value, *args, **kwargs):
        if isinstance(value, (str, os.PathLike)) and Path(value) == source:
            opened = fixture.copy()
            opened.format = "PSD"
            return opened
        return original_open(value, *args, **kwargs)

    monkeypatch.setattr(image_module, "open", fake_open)
    first_receipt = module._convert_psd_pdf(source, first)
    second_receipt = module._convert_psd_pdf(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["flattened_image_identity"] == {
        "width": 7,
        "height": 5,
        "rgb_sha256": first_receipt["flattened_image_identity"]["rgb_sha256"],
    }
    assert first_receipt["flattened_png_sha256"] == second_receipt["flattened_png_sha256"]
    assert first_receipt["output_sha256"] == second_receipt["output_sha256"]


def test_nested_office_reuses_unambiguous_plain_pdf_but_derives_ambiguous_sidecars(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fitz")
    module = _load("checkpoint_transport_views_office_provenance", PACKAGE / "transport_views.py")
    source_root = tmp_path / "source"
    asset = source_root / "task_fixture/repeat_0/reference_files/asset"
    asset.mkdir(parents=True)
    (source_root / "top.docx").write_bytes(b"not nested")
    (asset / "Plan.docx").write_bytes(b"document")
    (asset / "Plan.pdf").write_bytes(module._deterministic_text_pdf([b"trusted render"]))
    calls = 0

    def convert_to_pdf(input_path: Path, output_pdf: Path | None = None):
        nonlocal calls
        calls += 1
        assert output_pdf is not None
        output_pdf.write_bytes(module._deterministic_text_pdf([input_path.name.encode()]))
        return input_path, True, "fixture"

    preconvert = types.SimpleNamespace(convert_to_pdf=convert_to_pdf)
    output = tmp_path / "view"
    output.mkdir()
    entries, _directories = module._mirror_tree(
        source_root,
        output,
        min_audio_bytes=1024,
        derivative_cache=module._DerivativeCache(
            tmp_path / "cache", module.DERIVATIVE_PROFILE, _fixture_cache_identity()
        ),
        preconvert_module=preconvert,
        preconvert_module_sha256="a" * 64,
    )
    assert calls == 0
    assert not any(entry["kind"] == "office_pdf" for entry in entries)
    assert not (output / "top.docx.pdf").exists()

    (asset / "Plan.pptx").write_bytes(b"presentation")
    ambiguous_output = tmp_path / "ambiguous-view"
    ambiguous_output.mkdir()
    entries, _directories = module._mirror_tree(
        source_root,
        ambiguous_output,
        min_audio_bytes=1024,
        derivative_cache=module._DerivativeCache(
            tmp_path / "ambiguous-cache", module.DERIVATIVE_PROFILE, _fixture_cache_identity()
        ),
        preconvert_module=preconvert,
        preconvert_module_sha256="a" * 64,
    )
    assert calls == 2
    assert {entry["output_relative_path"] for entry in entries if entry["kind"] == "office_pdf"} == {
        "task_fixture/repeat_0/reference_files/asset/Plan.docx.pdf",
        "task_fixture/repeat_0/reference_files/asset/Plan.pptx.pdf",
    }


def test_transport_view_inventory_allows_only_candidate_judge_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("checkpoint_transport_views_inventory", PACKAGE / "transport_views.py")
    monkeypatch.setattr(
        module,
        "check_tools",
        lambda: {
            "ffmpeg": "/usr/bin/ffmpeg",
            "ffprobe": "/usr/bin/ffprobe",
            "fitz": "fixture",
            "pillow": "fixture",
            "soundfile": "fixture",
        },
    )
    candidate = tmp_path / "candidate-source"
    reference = tmp_path / "reference-source"
    (candidate / "task_fixture" / "repeat_0").mkdir(parents=True)
    (reference / "task_fixture" / "repeat_0").mkdir(parents=True)
    (candidate / "task_fixture" / "repeat_0" / "answer.txt").write_text("candidate")
    (reference / "task_fixture" / "repeat_0" / "answer.txt").write_text("reference")
    reference_overlay = tmp_path / "references.yaml"
    reference_overlay.write_text(
        yaml.safe_dump(
            {
                "gdpval_resources_server": {
                    "resources_servers": {
                        "gdpval": {
                            "reference_models": {
                                "fixture_ref": {
                                    "deliverables_dir": str(reference),
                                    "elo": 1234.5,
                                }
                            }
                        }
                    }
                }
            }
        )
    )
    output = tmp_path / "views"
    module.build(
        candidate,
        reference_overlay,
        output,
        min_audio_bytes=1024,
        derivative_profile=module.DERIVATIVE_PROFILE,
    )

    dynamic_cache = output / "candidate/task_fixture/repeat_0_verify_response_0123456789ab.json"
    dynamic_cache.write_text("{}")
    module._validate(output)

    unexpected_reference = output / "references/fixture_ref/unexpected.json"
    unexpected_reference.write_text("{}")
    with pytest.raises(ValueError, match="unexpected transport outputs"):
        module._validate(output)
    unexpected_reference.unlink()

    unexpected_reference_group = output / "references/unexpected_ref"
    unexpected_reference_group.mkdir()
    with pytest.raises(ValueError, match="reference-group inventory drift"):
        module._validate(output)
    unexpected_reference_group.rmdir()

    unexpected_candidate = output / "candidate/task_fixture/arbitrary.json"
    unexpected_candidate.write_text("{}")
    with pytest.raises(ValueError, match="unexpected transport outputs"):
        module._validate(output)


def test_transport_build_reuses_shared_cache_across_campaign_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("checkpoint_transport_views_cross_campaign", PACKAGE / "transport_views.py")
    tools = {
        "ffmpeg": "/usr/bin/ffmpeg",
        "ffprobe": "/usr/bin/ffprobe",
        "fitz": "fixture",
        "pillow": "fixture",
        "soundfile": "fixture",
    }
    monkeypatch.setattr(module, "check_tools", lambda: tools)
    source = tmp_path / "source"
    nested = source / "task_fixture/repeat_0/reference_files/asset"
    nested.mkdir(parents=True)
    (nested / "model.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
    reference_overlay = tmp_path / "references.yaml"
    reference_overlay.write_text(
        yaml.safe_dump(
            {"gdpval": {"reference_models": {"fixture_ref": {"deliverables_dir": str(source), "elo": 1234.5}}}}
        )
    )
    cache_root = tmp_path / "shared-cache"
    common = {
        "candidate_root": source,
        "reference_overlay": reference_overlay,
        "min_audio_bytes": 1024,
        "derivative_profile": module.DERIVATIVE_PROFILE,
        "derivative_cache_root": cache_root,
        "derivative_cache_label": "/shared/transport-cache",
        "container_sha256": "a" * 64,
    }

    first = module.build(output_root=tmp_path / "campaign-one", **common)
    second = module.build(output_root=tmp_path / "campaign-two", **common)

    assert first["derivative_cache"]["persistent"] is True
    assert first["derivative_cache"]["misses"] == 1
    assert first["derivative_cache"]["hits"] == 1
    assert second["derivative_cache"]["misses"] == 0
    assert second["derivative_cache"]["hits"] == 2
    assert second["derivative_cache"]["cache_label"] == "/shared/transport-cache"
    assert module._validate(tmp_path / "campaign-one") == first
    assert module._validate(tmp_path / "campaign-two") == second


def test_known_true3_stage0_pair_is_rebalanced_without_task_specific_code() -> None:
    module = _load("checkpoint_transport_assignment_stage0", RUNTIME_SOURCE)
    a941 = "a941b6d8-4289-4500-b45a-f8e4fc94a724"
    partner = "15ddd28d-8445-4baa-ac7f-f41372e1344e"
    original = {a941: "deepseek_v4_pro", partner: "qwen36_35b"}
    costs = {
        (a941, "deepseek_v4_pro"): _cost(module, False),
        (a941, "qwen36_35b"): _cost(module, True, 200),
        (partner, "deepseek_v4_pro"): _cost(module, True, 100),
        (partner, "qwen36_35b"): _cost(module, True, 100),
    }
    repaired = module._solve_capacity_assignment(list(original), list(set(original.values())), original, costs)
    assert repaired == {a941: "qwen36_35b", partner: "deepseek_v4_pro"}


def test_known_true3_stage1_cycles_are_rebalanced_with_exact_capacities() -> None:
    module = _load("checkpoint_transport_assignment_stage1", RUNTIME_SOURCE)
    zero = "0e386e32-df20-4d1f-b536-7159bc409ad5"
    audio = "38889c3b-e3d4-49c8-816a-3cc8e5313aba"
    ninety = "90edba97-74f0-425a-8ff6-8b93182eb7cb"
    video = "a941b6d8-4289-4500-b45a-f8e4fc94a724"
    refs = ["nemotron3_ultra", "glm51_fp8", "kimi_k26"]
    original = {
        zero: "nemotron3_ultra",
        audio: "glm51_fp8",
        ninety: "kimi_k26",
        video: "glm51_fp8",
    }
    allowed = {
        zero: {"glm51_fp8"},
        audio: {"nemotron3_ultra"},
        ninety: {"glm51_fp8"},
        video: {"kimi_k26"},
    }
    costs = {
        (task, reference): _cost(module, reference in allowed[task], 100 + refs.index(reference))
        for task in original
        for reference in refs
    }
    repaired = module._solve_capacity_assignment(list(original), refs, original, costs)
    assert repaired == {
        zero: "glm51_fp8",
        audio: "nemotron3_ultra",
        ninety: "glm51_fp8",
        video: "kimi_k26",
    }
    assert sorted(repaired.values()) == sorted(original.values())


def test_assignment_accepts_null_finish_marker_but_rejects_other_json_shapes(tmp_path: Path) -> None:
    module = _load("checkpoint_transport_assignment_null_finish", RUNTIME_SOURCE)
    repeat = tmp_path / "task" / "repeat_0"
    repeat.mkdir(parents=True)
    marker = repeat / "finish_params.json"

    marker.write_text("null\n", encoding="utf-8")
    assert module._has_valid_finish_marker([repeat])

    marker.write_text("[]\n", encoding="utf-8")
    assert not module._has_valid_finish_marker([repeat])


def test_assignment_routes_around_incomplete_reference_without_changing_counts(
    tmp_path: Path,
) -> None:
    module = _load("checkpoint_transport_assignment_reference_coverage", RUNTIME_SOURCE)
    candidate = tmp_path / "candidate"
    ref_a = tmp_path / "ref-a"
    ref_b = tmp_path / "ref-b"
    task_a = "task-a"
    task_b = "task-b"
    for root in (candidate, ref_a, ref_b):
        for task in (task_a, task_b):
            (root / f"task_{task}" / "repeat_0").mkdir(parents=True)

    # A syntactically present but invalid marker must not make a reference
    # eligible. Transport views expose valid source markers as symlinks, which
    # should still count because their resolved target is a regular JSON file.
    (ref_a / f"task_{task_a}" / "repeat_0" / "finish_params.json").write_text("[]")
    marker_source = tmp_path / "valid_finish_params.json"
    marker_source.write_text("{}")
    for repeat in (
        ref_a / f"task_{task_b}" / "repeat_0",
        ref_b / f"task_{task_a}" / "repeat_0",
        ref_b / f"task_{task_b}" / "repeat_0",
    ):
        (repeat / "finish_params.json").symlink_to(marker_source)

    global_config = {
        "gdpval_resources_server": {
            "resources_servers": {
                "gdpval": {
                    "persist_deliverables_dir": str(candidate),
                    "reference_models": {
                        "ref_a": {"deliverables_dir": str(ref_a), "elo": 1000},
                        "ref_b": {"deliverables_dir": str(ref_b), "elo": 900},
                    },
                }
            }
        }
    }
    repair = module.make_assignment_repair(global_config, {})
    repaired, receipt = repair(
        0,
        ["ref_a", "ref_b"],
        {task_a: "ref_a", task_b: "ref_b"},
    )

    assert repaired == {task_a: "ref_b", task_b: "ref_a"}
    assert receipt["reference_counts"] == {"ref_a": 1, "ref_b": 1}
    assert receipt["initially_incompatible"] == [
        {
            "task_id": task_a,
            "reference_id": "ref_a",
            "reasons": ["reference_incomplete"],
        }
    ]


def test_true3_overlay_has_distinct_reasoning_providers_and_dpi81() -> None:
    overlay = yaml.safe_load((PACKAGE / "true3_transport.yaml").read_text())
    resource_config = overlay["gdpval_resources_server"]["resources_servers"]["gdpval"]
    panel = resource_config["judge_panel"]
    assert resource_config["strict_comparison_trials"] is True
    assert [member["model_server"]["name"] for member in panel] == [
        "gdpval_gpt55_judge_model",
        "gdpval_gemini31_judge_model",
        "gdpval_claude48_judge_model",
    ]
    assert panel[0]["create_params_overrides"] == {"reasoning_effort": "medium"}
    assert 81 in panel[0]["raster_dpi_tiers"]  # TRUE3 task 83d semantic-preserving tier.
    assert panel[0]["max_total_image_base64_bytes"] == 12_000_000
    assert panel[0]["max_serialized_request_bytes"] == 30_000_000
    assert panel[1]["media_mode"] == "native_pdf_overflow_images"
    assert panel[1]["handles_audio"] and panel[1]["handles_video"]
    assert panel[1]["max_video_files"] == 10
    assert panel[1]["max_serialized_request_bytes"] == 495_000_000
    assert overlay["multistage"]["transport_assignment_repair"] == {
        "enabled": True,
        "max_file_bytes": 335_544_320,
        "max_raw_bytes": 368_000_000,
        "max_wire_bytes": 495_000_000,
        "framing_reserve_bytes": 4_194_304,
    }
    assert overlay["gdpval_gemini31_judge_model"]["responses_api_models"]["openai_model"]["extra_body"] == {
        "reasoning_effort": "high"
    }
    assert (
        overlay["gdpval_gemini31_judge_model"]["responses_api_models"]["openai_model"]["max_concurrent_requests"]
        == "${oc.env:GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS,2}"
    )
    assert overlay["gdpval_claude48_judge_model"]["responses_api_models"]["openai_model"]["extra_body"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "timeout": 900,
        "request_timeout": 900,
    }
    assert panel[2]["create_params_overrides"]["temperature"] is None
    assert panel[2]["max_image_base64_bytes"] == 5 * 1024 * 1024
    assert overlay["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_reference_files_recursive"] is True


def test_runtime_patch_is_generic_and_preserves_exact_pr_base() -> None:
    runtime = _load("checkpoint_transport_runtime", PACKAGE / "transport_runtime.py")
    patch = (PACKAGE / "runtime_sources/pr2588_true3_transport.patch").read_text()
    assert runtime.REVISION == "d3f146d386c7dfe07d4fabce32c4c8b14c7917d2"
    assert "resources_servers/gdpval/multistage_elo.py" in runtime.BASE_HASHES
    assert "responses_api_models/openai_model/app.py" in runtime.BASE_HASHES
    assert "responses_api_models/openai_model/requirements.txt" in runtime.BASE_HASHES
    for task_id in (
        "a941b6d8-4289-4500-b45a-f8e4fc94a724",
        "15ddd28d-8445-4baa-ac7f-f41372e1344e",
        "0e386e32-df20-4d1f-b536-7159bc409ad5",
        "38889c3b-e3d4-49c8-816a-3cc8e5313aba",
        "90edba97-74f0-425a-8ff6-8b93182eb7cb",
        "83d10b06-26d1-4636-a32c-23f92c57f30b",
    ):
        assert task_id not in patch
        assert task_id not in RUNTIME_SOURCE.read_text()
    for relative, expected in runtime.BASE_HASHES.items():
        try:
            data = subprocess.check_output(["git", "show", f"{runtime.REVISION}:{relative}"], cwd=REPO)
        except subprocess.CalledProcessError:
            pytest.skip("pinned PR object is unavailable in this checkout")
        assert hashlib.sha256(data).hexdigest() == expected


def test_runtime_prompt_and_strict_trial_gate_apply_to_exact_pr_base(tmp_path: Path) -> None:
    runtime_module = _load("checkpoint_transport_runtime_verdict_prompt", PACKAGE / "transport_runtime.py")
    checkout = tmp_path / "pinned-gym"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(REPO), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", runtime_module.REVISION],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    overlay = tmp_path / "runtime"
    runtime_module.materialize(checkout, overlay, PACKAGE)
    manifest = json.loads((overlay / "runtime_manifest.json").read_text())
    assert manifest["schema"] == "gdpval.transport-runtime.v3"
    assert manifest["revision"] == "d3f146d386c7dfe07d4fabce32c4c8b14c7917d2"
    assert "waivable_failure_classes" in (overlay / "resources_servers/gdpval/multistage_elo.py").read_text()
    assert (
        manifest["strict_comparison_trials_patch_sha256"]
        == hashlib.sha256((PACKAGE / "runtime_sources/strict_comparison_trials.patch").read_bytes()).hexdigest()
    )
    assert (
        manifest["provider_context_fallback_patch_sha256"]
        == hashlib.sha256((PACKAGE / "runtime_sources/provider_context_fallback.patch").read_bytes()).hexdigest()
    )
    assert (
        manifest["provider_rate_limit_backoff_patch_sha256"]
        == hashlib.sha256((PACKAGE / "runtime_sources/provider_rate_limit_backoff.patch").read_bytes()).hexdigest()
    )
    assert (
        manifest["partial_pdf_overflow_patch_sha256"]
        == hashlib.sha256((PACKAGE / "runtime_sources/partial_pdf_overflow.patch").read_bytes()).hexdigest()
    )
    expected_requirement = f"-e nemo-gym[dev] @ {checkout.resolve().as_uri()}"
    for relative in runtime_module.COMPONENT_REQUIREMENTS:
        assert (overlay / relative).read_text().splitlines()[0] == expected_requirement
    app_source = (overlay / "resources_servers/gdpval/app.py").read_text()
    assert "except JudgeContextWindowError as error:" in app_source
    assert "failed_names.add(error.judge_name)" in app_source
    assert "rng.setstate(trial_rng_state)" in app_source
    assert "max_retries=0" in app_source
    model_adapter_source = (overlay / "responses_api_models/openai_model/app.py").read_text()
    assert "class BoundedRateLimitOpenAI" in model_adapter_source
    assert "RATE_LIMIT_MAX_ATTEMPTS = 6" in model_adapter_source
    comparison_source = (overlay / "resources_servers/gdpval/comparison.py").read_text()
    assert "RATE_LIMIT_ERROR_MARKERS" in comparison_source
    assert "standby" not in (PACKAGE / "runtime_sources/provider_context_fallback.patch").read_text()

    script = r"""
from types import SimpleNamespace

from resources_servers.gdpval.comparison import (
    A_WIN_RESPONSE,
    B_WIN_RESPONSE,
    FINAL_VERDICT_REMINDER,
    Judge,
    JudgeContextWindowError,
    SUBMISSION_B_CLOSE,
    TIE_RESPONSE,
    construct_judge_messages,
    is_context_window_error,
    parse_judgement,
    run_trials,
)
from resources_servers.gdpval.app import (
    GDPValResourcesServerConfig,
    _strict_comparison_trial_failure,
)
from resources_servers.gdpval.multistage_orchestrator import parse_multistage_config

stage_config = parse_multistage_config(
    {
        "enabled": True,
        "stages": [
            {
                "num_tasks": 45,
                "partial_completion": {
                    "min_success_fraction": 0.9,
                    "min_per_reference_success_fraction": 0.5,
                    "min_successful_rows_per_reference": 1,
                    "waivable_failure_classes": ["timeout_exceeded", "transient"],
                },
            },
            {"num_tasks": 220, "num_models": 4},
        ],
    }
)
assert stage_config.stages[0].partial_completion.waivable_failure_classes == (
    "timeout_exceeded",
    "transient",
)
assert stage_config.stages[1].partial_completion is None

messages = construct_judge_messages(
    task_prompt="fixture task",
    refs=[{"type": "text", "text": "reference evidence"}],
    submission_a=[{"type": "text", "text": "submission A evidence"}],
    submission_b=[{"type": "text", "text": "submission B evidence"}],
)
content = messages[0]["content"]
assert content[-2] == {"type": "text", "text": SUBMISSION_B_CLOSE}
assert content[-1] == {"type": "text", "text": FINAL_VERDICT_REMINDER}
assert FINAL_VERDICT_REMINDER.endswith("invalid.\n")
assert all(FINAL_VERDICT_REMINDER.count(token) == 1 for token in (A_WIN_RESPONSE, B_WIN_RESPONSE, TIE_RESPONSE))

# Keep the scientific gate strict: prose that appears to prefer one side is not
# normalized into a vote without the protocol's explicit verdict token.
assert parse_judgement("Submission **B** is stronger overall.") is None
assert parse_judgement(f"reasoning\n{A_WIN_RESPONSE}") == A_WIN_RESPONSE
assert parse_judgement(f"reasoning\n{B_WIN_RESPONSE}") == B_WIN_RESPONSE
assert parse_judgement(f"reasoning\n{TIE_RESPONSE}") == TIE_RESPONSE

# The compatibility default remains off. The TRUE3 overlay opts in separately.
assert GDPValResourcesServerConfig.model_fields["strict_comparison_trials"].default is False
assert _strict_comparison_trial_failure(
    attempted_matchups=1,
    num_trials=4,
    total_judged=4,
    total_invalid=0,
    ref_errors={},
) is None
for failure in (
    _strict_comparison_trial_failure(
        attempted_matchups=1,
        num_trials=4,
        total_judged=3,
        total_invalid=1,
        ref_errors={},
    ),
    _strict_comparison_trial_failure(
        attempted_matchups=2,
        num_trials=4,
        total_judged=4,
        total_invalid=0,
        ref_errors={"anchor": ["provider timeout"]},
    ),
    _strict_comparison_trial_failure(
        attempted_matchups=0,
        num_trials=4,
        total_judged=0,
        total_invalid=0,
        ref_errors={},
    ),
):
    assert failure is not None


class StubCompletions:
    def __init__(self, *, error=None, response="BOXED[A]"):
        self.error = error
        self.response = response

    def create(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response))]
        )


def stub_client(*, error=None, response="BOXED[A]"):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=StubCompletions(error=error, response=response))
    )


# A nested provider context rejection is converted to a typed, named failure so
# the resources server can exclude that provider and replay the whole matchup.
context_cause = RuntimeError("Input is too long for requested model")
context_error = RuntimeError("provider wrapper")
context_error.__cause__ = context_cause
assert is_context_window_error(context_error)
try:
    run_trials(
        judges=[Judge(name="claude", client=stub_client(error=context_error), model="fixture")],
        task_prompt="fixture task",
        refs=[],
        submission_a=[],
        submission_b=[],
        num_trials=1,
    )
except JudgeContextWindowError as error:
    assert error.judge_name == "claude"
    assert error.original is context_error
else:
    raise AssertionError("context rejection was not converted to a named failure")

# Every unrelated provider error remains fatal and retains its original identity.
unrelated_error = RuntimeError("provider permission denied")
assert not is_context_window_error(unrelated_error)
try:
    run_trials(
        judges=[Judge(name="claude", client=stub_client(error=unrelated_error), model="fixture")],
        task_prompt="fixture task",
        refs=[],
        submission_a=[],
        submission_b=[],
        num_trials=1,
    )
except RuntimeError as error:
    assert error is unrelated_error
else:
    raise AssertionError("unrelated provider error was swallowed")
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(overlay), str(checkout))),
    }
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=60,
    )

    # A partial adapter overlay must fail closed instead of silently launching
    # the unpatched built-in adapter from the pinned Gym checkout.
    (overlay / "responses_api_models/openai_model/requirements.txt").unlink()
    with pytest.raises(ValueError, match="component resolver selected"):
        runtime_module._validate_component_resolution(checkout, overlay)


def test_partial_pdf_overflow_retains_all_pages_with_exact_image_budget(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    runtime_module = _load("checkpoint_transport_runtime_partial_pdf", PACKAGE / "transport_runtime.py")
    checkout = tmp_path / "pinned-gym"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(REPO), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", runtime_module.REVISION],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    overlay = tmp_path / "runtime"
    runtime_module.materialize(checkout, overlay, PACKAGE)

    script = r"""
import base64
from copy import deepcopy

import fitz

from resources_servers.gdpval.comparison import (
    apply_native_pdf_overflow,
    plan_native_pdf_overflow,
)
from resources_servers.gdpval.media_conversion import pdf_page_count


PDF_PREFIX = "data:application/pdf;base64,"


def make_pdf(page_count, *, labels=False):
    document = fitz.open()
    for index in range(page_count):
        page = document.new_page(width=72, height=72)
        if labels:
            page.insert_text((8, 36), f"page-{index}")
    payload = document.tobytes()
    document.close()
    return payload


def pdf_block(payload):
    return {
        "type": "image_url",
        "image_url": {"url": PDF_PREFIX + base64.b64encode(payload).decode("ascii")},
    }


# Reproduce the pathological task's exact page geometry without task-specific
# code: 1001 candidate + 414 benchmark reference + 9 anchor = 1424 pages.
candidate = make_pdf(1001)
benchmark_reference = make_pdf(414)
anchor = make_pdf(9)
sections = {
    "submission_b": [pdf_block(candidate)],
    "refs": [pdf_block(benchmark_reference), pdf_block(anchor)],
    "submission_a": [],
}
plan = plan_native_pdf_overflow(
    sections, native_page_cap=1000, image_cap=450, native_pdf_byte_cap=50_000_000
)
assert plan["eligible"]
assert plan["total_pdf_pages"] == 1424
assert plan["native_pages_after"] == 1000
assert plan["raster_pages"] == 424
assert plan["existing_images"] == 0
assert plan["total_images_after"] == 424
assert [
    (row["pages"], row["raster_page_start"], row["raster_page_count"], row["native_page_count"])
    for row in plan["selected"]
] == [(9, 0, 9, 0), (414, 0, 414, 0), (1001, 0, 1, 1000)]

# The cap is request-wide: 26 existing images leave exactly enough space,
# while a 27th makes the same otherwise-lossless plan ineligible.
with_images = deepcopy(sections)
with_images["submission_a"] = [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    for _ in range(26)
]
exact = plan_native_pdf_overflow(
    with_images, native_page_cap=1000, image_cap=450, native_pdf_byte_cap=50_000_000
)
assert exact["eligible"] and exact["total_images_after"] == 450
with_images["submission_a"].append(
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
)
over = plan_native_pdf_overflow(
    with_images, native_page_cap=1000, image_cap=450, native_pdf_byte_cap=50_000_000
)
assert not over["eligible"] and over["total_images_after"] == 451

# Exercise the actual prefix split and render on a small labeled PDF. The
# raster is source page 0; the native suffix contains source pages 1..3.
small = make_pdf(4, labels=True)
small_sections = {"refs": [pdf_block(small)], "submission_a": [], "submission_b": []}
small_before = deepcopy(small_sections)
small_plan = plan_native_pdf_overflow(
    small_sections, native_page_cap=3, image_cap=1, native_pdf_byte_cap=50_000_000
)
assert small_plan["eligible"]
assert small_plan["selected"][0]["raster_page_count"] == 1
assert small_plan["selected"][0]["native_page_count"] == 3
converted = apply_native_pdf_overflow(
    small_sections,
    small_plan,
    render_dpi=36,
    max_pages=1,
    include_text=False,
)
assert small_sections == small_before
assert len(converted["refs"]) == 2
raster_url = converted["refs"][0]["image_url"]["url"]
native_url = converted["refs"][1]["image_url"]["url"]
assert raster_url.startswith("data:image/png;base64,")
assert native_url.startswith(PDF_PREFIX)
rendered_pixmap = fitz.Pixmap(base64.b64decode(raster_url.split(",", 1)[1], validate=True))
with fitz.open(stream=small, filetype="pdf") as original:
    source_pixmap = original[0].get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False)
assert (rendered_pixmap.width, rendered_pixmap.height, rendered_pixmap.samples) == (
    source_pixmap.width,
    source_pixmap.height,
    source_pixmap.samples,
)
native_payload = base64.b64decode(native_url[len(PDF_PREFIX) :], validate=True)
assert pdf_page_count(native_payload) == 3
with fitz.open(stream=native_payload, filetype="pdf") as suffix:
    assert [page.get_text().strip() for page in suffix] == ["page-1", "page-2", "page-3"]

# Immutable planning is fail-closed: changing the selected source after
# preflight cannot silently apply the old split.
drifted = deepcopy(small_sections)
drifted["refs"][0] = pdf_block(make_pdf(4, labels=True) + b"\n")
try:
    apply_native_pdf_overflow(
        drifted,
        small_plan,
        render_dpi=36,
        max_pages=1,
        include_text=False,
    )
except ValueError as error:
    assert "overflow plan hash drift" in str(error)
else:
    raise AssertionError("overflow plan accepted source drift")
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(overlay), str(checkout))),
    }
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=60,
    )


def test_provider_aggregate_caps_select_safe_tier_and_bound_video_count(tmp_path: Path) -> None:
    runtime_module = _load("checkpoint_transport_runtime_aggregate_caps", PACKAGE / "transport_runtime.py")
    checkout = tmp_path / "pinned-gym"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(REPO), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", runtime_module.REVISION],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    overlay = tmp_path / "runtime"
    runtime_module.materialize(checkout, overlay, PACKAGE)

    script = r"""
from resources_servers.gdpval.comparison import Judge, preflight_judge_transport

task_like_base64_bytes = {
    120: 23_826_112,
    108: 26_767_424,
    96: 19_767_704,
    90: 11_444_976,
}
aggregate_judge = Judge(
    name="fixture-gpt",
    client=None,
    model="fixture",
    max_total_image_base64_bytes=12_000_000,
    max_serialized_request_bytes=50_000_000,
)
chosen_dpi = None
for dpi, payload_bytes in task_like_base64_bytes.items():
    payload = "A" * payload_bytes
    sections = {
        "refs": [],
        "submission_a": [],
        "submission_b": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
        ],
    }
    receipt = preflight_judge_transport(aggregate_judge, "fixture", sections)
    assert receipt["total_image_base64_bytes"] == payload_bytes
    if receipt["eligible"]:
        chosen_dpi = dpi
        break
    assert receipt["reasons"] == ["provider_total_image_byte_cap"]
assert chosen_dpi == 90
assert 12_000_000 - task_like_base64_bytes[chosen_dpi] == 555_024

video_judge = Judge(name="fixture-gemini", client=None, model="fixture", max_video_files=10)
video_blocks = [
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
    for _ in range(11)
]
video_sections = {"refs": video_blocks, "submission_a": [], "submission_b": []}
receipt = preflight_judge_transport(video_judge, "fixture", video_sections)
assert receipt["video_file_count"] == 11
assert receipt["reasons"] == ["provider_video_count_cap"]
video_sections["refs"].pop()
assert preflight_judge_transport(video_judge, "fixture", video_sections)["eligible"]
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(overlay), str(checkout))),
    }
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=60,
    )


def test_provider_image_cap_losslessly_tiles_and_preflights(tmp_path: Path) -> None:
    runtime_module = _load("checkpoint_transport_runtime_image_cap", PACKAGE / "transport_runtime.py")
    checkout = tmp_path / "pinned-gym"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(REPO), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", runtime_module.REVISION],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    overlay = tmp_path / "runtime"
    runtime_module.materialize(checkout, overlay, PACKAGE)

    script = r"""
import base64
from copy import deepcopy
import fitz
import random

from resources_servers.gdpval.comparison import (
    Judge,
    _image_data_url,
    apply_judge_image_byte_cap,
    build_file_section,
    preflight_judge_transport,
)
from resources_servers.gdpval.multistage_orchestrator import (
    compute_fingerprint,
    parse_multistage_config,
)

width, height = 320, 192
samples = random.Random(7).randbytes(width * height * 3)
source = fitz.Pixmap(fitz.csRGB, width, height, samples, False)
source_png = source.tobytes("png")
source_payload = base64.b64encode(source_png).decode("ascii")
cap = max(4096, len(source_payload) // 4)
sections = {
    "refs": [],
    "submission_a": [],
    "submission_b": [
        {"type": "text", "text": "moodboard.png:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{source_payload}"}},
    ],
}
judge = Judge(
    name="fixture-claude",
    client=None,
    model="fixture",
    max_image_base64_bytes=cap,
    max_serialized_request_bytes=50_000_000,
)

unprepared = preflight_judge_transport(judge, "fixture", sections)
assert not unprepared["eligible"]
assert unprepared["reasons"] == ["provider_image_byte_cap"]

prepared, receipt = apply_judge_image_byte_cap(judge, sections, max_images=100)
assert receipt["eligible"]
assert receipt["original_image_count"] == 1
assert receipt["output_image_count"] > 1
assert len(receipt["transformations"]) == 1
assert all(tile["base64_bytes"] <= cap for tile in receipt["transformations"][0]["tiles"])
postflight = preflight_judge_transport(judge, "fixture", prepared)
assert postflight["eligible"]
assert postflight["oversize_image_count"] == 0

reconstructed = bytearray(len(source.samples))
image_blocks = [block for block in prepared["submission_b"] if _image_data_url(block) is not None]
tiles = receipt["transformations"][0]["tiles"]
assert len(image_blocks) == len(tiles)
for block, tile_receipt in zip(image_blocks, tiles):
    _mime, encoded = _image_data_url(block)
    tile = fitz.Pixmap(base64.b64decode(encoded, validate=True))
    x0, y0, x1, y1 = tile_receipt["crop"]
    assert (tile.width, tile.height) == (x1 - x0, y1 - y0)
    for row in range(tile.height):
        source_start = row * tile.stride
        target_start = ((y0 + row) * width + x0) * source.n
        reconstructed[target_start : target_start + tile.width * source.n] = tile.samples[
            source_start : source_start + tile.width * tile.n
        ]
assert bytes(reconstructed) == source.samples

# GPT's provider-wide image size is distinct from both one-image and serialized
# request limits. Reproduce the provider-free sweep around the configured
# 12,000,000-decimal-byte cap: 120/108/96 DPI are rejected and 90 DPI is the
# first eligible tier, with 555,024 bytes of base64 headroom.
task_like_base64_bytes = {
    120: 23_826_112,
    108: 26_767_424,
    96: 19_767_704,
    90: 11_444_976,
}
aggregate_judge = Judge(
    name="fixture-gpt",
    client=None,
    model="fixture",
    max_total_image_base64_bytes=12_000_000,
    max_serialized_request_bytes=50_000_000,
)
eligible_tiers = []
for dpi, payload_bytes in task_like_base64_bytes.items():
    payload = "A" * payload_bytes
    task_like_sections = {
        "refs": [],
        "submission_a": [],
        "submission_b": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
        ],
    }
    aggregate_receipt = preflight_judge_transport(aggregate_judge, "fixture", task_like_sections)
    assert aggregate_receipt["total_image_base64_bytes"] == payload_bytes
    if dpi == 90:
        assert aggregate_receipt["eligible"]
        eligible_tiers.append(dpi)
    else:
        assert aggregate_receipt["reasons"] == ["provider_total_image_byte_cap"]
    del payload, task_like_sections
assert eligible_tiers == [90]

# Gemini's native-video count is measured on the final request blocks. The
# guard accepts exactly ten and rejects eleven before any provider dispatch.
video_judge = Judge(name="fixture-gemini", client=None, model="fixture", max_video_files=10)
video_blocks = [
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
    for _ in range(11)
]
video_sections = {"refs": video_blocks, "submission_a": [], "submission_b": []}
video_receipt = preflight_judge_transport(video_judge, "fixture", video_sections)
assert video_receipt["video_file_count"] == 11
assert video_receipt["reasons"] == ["provider_video_count_cap"]
video_sections["refs"].pop()
assert preflight_judge_transport(video_judge, "fixture", video_sections)["eligible"]

# Nested benchmark references use logical relative names and Office-derived
# PDFs exactly once. Equal basenames in different asset directories cannot
# collide, and the legacy flat mode remains unchanged.
from pathlib import Path
import resources_servers.gdpval.comparison as comparison

comparison._ignore_files = lambda: frozenset()
reference_files = Path("reference_files")
for directory, text in (("asset-b", "second"), ("asset-a", "first")):
    target = reference_files / directory
    target.mkdir(parents=True)
    (target / "Plan.docx").write_bytes(f"{text}-source".encode())
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(target / "Plan.docx.pdf")
    document.close()

flat = build_file_section(str(reference_files), [], recursive=False)
assert flat == [{"type": "text", "text": "None"}]
recursive = build_file_section(str(reference_files), [], recursive=True)
headers = [block["text"] for block in recursive if block["type"] == "text"]
assert headers == ["\nasset-a/Plan.docx:\n", "\nasset-b/Plan.docx:\n"]
attachments = [block for block in recursive if block["type"] == "image_url"]
assert len(attachments) == 2
assert len({block["image_url"]["url"] for block in attachments}) == 2
assert not any("Plan.docx.pdf" in header for header in headers)

unrendered = reference_files / "asset-c/Model.step"
unrendered.parent.mkdir(parents=True)
unrendered.write_bytes(b"unsupported CAD source")
try:
    build_file_section(str(reference_files), [], recursive=True)
except ValueError as exc:
    assert str(exc) == (
        "recursive reference asset has no provenance-safe PDF render: asset-c/Model.step"
    )
else:
    raise AssertionError("unrendered recursive reference asset was silently omitted")

# The cap is a provider transport-compatibility knob: adding it must preserve a
# frozen campaign fingerprint.  Judge identity and semantic presentation knobs
# remain fingerprinted, and the same field in materialized data is not ignored.
cfg = parse_multistage_config({"enabled": True, "stages": ["1", "3:2"], "seed": 42})
distribution = {"all": {"percentage": 1.0, "task_ids": ["t0", "t1", "t2"]}}
runtime = {
    "gdpval_resources_server": {
        "resources_servers": {
            "gdpval": {
                "num_comparison_trials": 4,
                "judge_panel": [
                    {
                        "name": "claude-opus-4.8",
                        "model": "aws/anthropic/bedrock-claude-opus-4-8",
                        "media_mode": "native_pdf",
                        "max_serialized_request_bytes": 495000000,
                    }
                ],
            }
        }
    }
}
baseline_fingerprint = compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=runtime)
changed_wire_cap = deepcopy(runtime)
changed_wire_cap["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_panel"][0][
    "max_serialized_request_bytes"
] = 440401920
assert (
    compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=changed_wire_cap)
    == baseline_fingerprint
)
lowered_wire_cap = deepcopy(runtime)
lowered_wire_cap["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_panel"][0][
    "max_serialized_request_bytes"
] = 420000000
assert (
    compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=lowered_wire_cap)
    != baseline_fingerprint
)
capped_runtime = deepcopy(runtime)
capped_runtime["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_panel"][0][
    "max_image_base64_bytes"
] = 5 * 1024 * 1024
assert (
    compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=capped_runtime)
    == baseline_fingerprint
)

# Credentials authenticate an otherwise identical runtime and must not fork a
# resumable campaign when they rotate.  Only the two supported credential field
# names are ignored: endpoint and model identity stay scientifically bound.
credential_runtime = deepcopy(capped_runtime)
credential_runtime["gdpval_gpt55_judge_model"] = {
    "responses_api_models": {
        "openai_model": {
            "entrypoint": "app.py",
            "openai_base_url": "https://judge-a.example/v1",
            "openai_api_key": "openai-key-a",
            "openai_model": "openai/openai/gpt-5.5",
        },
        "generic_model": {
            "api_key": "generic-key-a",
            "base_url": "https://generic-a.example/v1",
            "model": "fixture/model-a",
        },
    }
}
credential_fingerprint = compute_fingerprint(
    cfg,
    {"ref": 1200.0},
    distribution,
    resolved_global_config=credential_runtime,
)
rotated_openai_key = deepcopy(credential_runtime)
rotated_openai_key["gdpval_gpt55_judge_model"]["responses_api_models"]["openai_model"][
    "openai_api_key"
] = "openai-key-b"
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        resolved_global_config=rotated_openai_key,
    )
    == credential_fingerprint
)
rotated_generic_key = deepcopy(credential_runtime)
rotated_generic_key["gdpval_gpt55_judge_model"]["responses_api_models"]["generic_model"][
    "api_key"
] = "generic-key-b"
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        resolved_global_config=rotated_generic_key,
    )
    == credential_fingerprint
)
changed_endpoint = deepcopy(credential_runtime)
changed_endpoint["gdpval_gpt55_judge_model"]["responses_api_models"]["openai_model"][
    "openai_base_url"
] = "https://judge-b.example/v1"
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        resolved_global_config=changed_endpoint,
    )
    != credential_fingerprint
)
changed_response_model = deepcopy(credential_runtime)
changed_response_model["gdpval_gpt55_judge_model"]["responses_api_models"]["openai_model"][
    "openai_model"
] = "openai/openai/gpt-5.6"
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        resolved_global_config=changed_response_model,
    )
    != credential_fingerprint
)

# Aggregate image and video-count caps can change raster evidence or judge
# eligibility, so both remain campaign-fingerprint inputs.
aggregate_runtime = deepcopy(capped_runtime)
aggregate_runtime["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_panel"][0][
    "max_total_image_base64_bytes"
] = 12_000_000
assert (
    compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=aggregate_runtime)
    != baseline_fingerprint
)
video_runtime = deepcopy(capped_runtime)
video_runtime["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_panel"][0][
    "max_video_files"
] = 10
assert (
    compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=video_runtime)
    != baseline_fingerprint
)

# Recursive reference inclusion changes scientific evidence and therefore must
# change the fingerprint (unlike the provider-only image byte cap above).
recursive_runtime = deepcopy(capped_runtime)
recursive_runtime["gdpval_resources_server"]["resources_servers"]["gdpval"][
    "judge_reference_files_recursive"
] = True
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        resolved_global_config=recursive_runtime,
    )
    != baseline_fingerprint
)

changed_judge = deepcopy(capped_runtime)
changed_judge["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_panel"][0][
    "model"
] = "aws/anthropic/a-different-judge"
assert (
    compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=changed_judge)
    != baseline_fingerprint
)
changed_media_mode = deepcopy(capped_runtime)
changed_media_mode["gdpval_resources_server"]["resources_servers"]["gdpval"]["judge_panel"][0][
    "media_mode"
] = "images_and_text"
assert (
    compute_fingerprint(cfg, {"ref": 1200.0}, distribution, resolved_global_config=changed_media_mode)
    != baseline_fingerprint
)
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0", "prompt": "original task prompt"}],
        resolved_global_config=runtime,
    )
    != compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0", "prompt": "changed task prompt"}],
        resolved_global_config=runtime,
    )
)
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0", "max_serialized_request_bytes": 495000000}],
        resolved_global_config=runtime,
    )
    != compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0", "max_serialized_request_bytes": 420000000}],
        resolved_global_config=runtime,
    )
)
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0"}],
        resolved_global_config=runtime,
    )
    != compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0"}, {"task_id": "t0"}],
        resolved_global_config=runtime,
    )
)
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0", "max_image_base64_bytes": 5 * 1024 * 1024}],
        resolved_global_config=runtime,
    )
    != compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0"}],
        resolved_global_config=runtime,
    )
)
assert (
    compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0", "openai_api_key": "row-key-a"}],
        resolved_global_config=runtime,
    )
    != compute_fingerprint(
        cfg,
        {"ref": 1200.0},
        distribution,
        materialized_rows=[{"task_id": "t0", "openai_api_key": "row-key-b"}],
        resolved_global_config=runtime,
    )
)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(overlay), str(checkout))),
    }
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=60,
    )


def test_judge_uses_campaign_runtime_views_and_bounded_process_group() -> None:
    script = (PACKAGE / "judge.sbatch").read_text()
    prebuild = (PACKAGE / "transport_prebuild.sbatch").read_text()
    process_helper = (PACKAGE / "judge_process_group.sh").read_text()
    controller = (PACKAGE / "controller.sbatch").read_text()
    assert 'NEMO_GYM_EXTRA_ROOTS="$JUDGE_RUNTIME_OVERLAY"' in script
    assert "REQUESTED_JUDGE_DIR_SUFFIX=${JUDGE_DIR_SUFFIX:-}" in script
    assert "JUDGE_DIR_SUFFIX=${REQUESTED_JUDGE_DIR_SUFFIX:-e2e}" in script
    assert 'JUDGE_DIR="$RUN_DIR/judge_${JUDGE_DIR_SUFFIX}"' in script
    assert "CHECKPOINT_E2E_ACTIVE_PACKAGE" in script
    assert "CHECKPOINT_E2E_JUDGE_RUNTIME_OVERLAY" in script
    assert "CHECKPOINT_E2E_JUDGE_TRANSPORT_OVERLAY" in script
    assert "CHECKPOINT_E2E_TRANSPORT_VIEW_ROOT" in script
    assert "TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX" in script
    assert '"$CAMPAIGN_E2E_SCRIPT" _compute-preflight "$RUN_DIR"' in script
    assert '"$E2E_DIR/transport_runtime.py" validate' in script
    assert '--config "$JUDGE_TRANSPORT_OVERLAY"' in script
    assert '--config "$REFERENCE_VIEW_OVERLAY"' in script
    assert "configure_gdpval_container_python" in script
    assert "CONTAINER_PYTHON=(" in script
    assert '"${GDPVAL_CONTAINER_PYTHON[@]:0:$container_tail_start}"' in script
    assert "TRANSPORT_PREBUILD_PASS" in script
    assert '"$TRANSPORT_VIEWS_PY" build' not in script
    assert "#SBATCH --time=02:00:00" in prebuild
    assert '"$TRANSPORT_VIEWS_PY" build' in prebuild
    assert '"$TRANSPORT_VIEWS_PY" validate' not in prebuild
    assert "--derivative-cache-root" in prebuild
    assert "--derivative-cache-label" in prebuild
    assert "--container-sha256" in prebuild
    assert "configure_gdpval_container_python" in prebuild
    assert "CONTAINER_CACHE_ROOT=/gdpval-transport-derivative-cache" in prebuild
    assert '--bind "$DERIVATIVE_CACHE_ROOT:$CONTAINER_CACHE_ROOT"' in prebuild
    assert "container_identity_receipt" in prebuild
    assert "container_stat_after" in prebuild
    assert "TRANSPORT_PREBUILD_PASS" in prebuild
    assert "JUDGE_API_KEY" not in prebuild
    assert "CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS" not in prebuild
    assert 'source "$E2E_DIR/judge_process_group.sh"' in script
    assert 'start_judge_process_group "$JUDGE_LOG" "$GYM_PYTHON" "$GYM_ENTRYPOINT" eval run' in script
    assert "stop_judge_process_group" in script
    assert 'kill -TERM -- "-$GYM_PID"' in process_helper
    assert 'kill -KILL -- "-$GYM_PID"' in process_helper
    assert 'wait "$GYM_PID"' in process_helper
    assert "| tee" not in script
    assert 'STIRRUP_PER_TASK_TIMEOUT_S="${JUDGE_TASK_TIMEOUT_SECONDS:-1500}"' in script
    assert "JUDGE_NO_PROGRESS_SECONDS >= JUDGE_TASK_TIMEOUT_SECONDS + 300" in controller
    assert 'source "$E2E_DIR/judge_process_group.sh"' in controller
    assert "configure_gdpval_container_python" in controller


def test_transport_container_contract_binds_only_required_roots(tmp_path: Path) -> None:
    helper = PACKAGE / "judge_process_group.sh"
    apptainer = tmp_path / "bin" / "apptainer"
    apptainer.parent.mkdir()
    apptainer.write_text("#!/bin/sh\n", encoding="utf-8")
    apptainer.chmod(0o755)
    sif = tmp_path / "gdpval.sif"
    sif.touch()
    e2e = tmp_path / "e2e"
    run = tmp_path / "run"
    e2e.mkdir()
    run.mkdir()
    roots = []
    for index in range(9):
        root = tmp_path / "refs" / f"model_{index}"
        root.mkdir(parents=True)
        roots.append(root)
    overlay = tmp_path / "reference.yaml"
    overlay.write_text(
        "gdpval:\n  reference_models:\n"
        + "".join(f"    model_{index}:\n      deliverables_dir: {root}\n" for index, root in enumerate(roots)),
        encoding="utf-8",
    )
    script = (
        f'source "{helper}"; '
        f'configure_gdpval_container_python "{apptainer}" "{sif}" "{e2e}" "{run}" "{overlay}" && '
        "printf '%s\\n' \"${GDPVAL_CONTAINER_PYTHON[@]}\""
    )

    result = subprocess.run(["bash", "-c", script], check=True, text=True, capture_output=True)

    assert result.stdout.splitlines() == [
        str(apptainer),
        "exec",
        "--bind",
        f"{e2e}:{e2e}:ro",
        "--bind",
        f"{run}:{run}",
        "--bind",
        f"{overlay}:{overlay}:ro",
        *[item for root in roots for item in ("--bind", f"{root}:{root}:ro")],
        str(sif),
        "python3",
    ]


def test_judge_process_group_captures_rc_and_leaves_no_orphan(tmp_path: Path) -> None:
    helper = PACKAGE / "judge_process_group.sh"
    log = tmp_path / "judge.log"
    group_receipt = tmp_path / "group.pid"
    child_receipt = tmp_path / "child.pid"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    setsid = fake_bin / "setsid"
    setsid.write_text(f"#!{sys.executable}\nimport os, sys\nos.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n")
    setsid.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "HELPER": str(helper),
        "JUDGE_LOG": str(log),
        "GROUP_RECEIPT": str(group_receipt),
        "CHILD_RECEIPT": str(child_receipt),
    }
    normal = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$HELPER"; '
            'start_judge_process_group "$JUDGE_LOG" bash -c "exit 7"; '
            "wait_judge_process_group; [[ $GYM_RC == 7 && -z $GYM_PID ]]",
        ],
        env=environment,
        check=False,
        timeout=10,
    )
    assert normal.returncode == 0

    stopped = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$HELPER"; '
            "JUDGE_PROCESS_TERM_GRACE_SECONDS=1; "
            'start_judge_process_group "$JUDGE_LOG" bash -c '
            '\'trap "" TERM; sleep 60 & echo $! > "$CHILD_RECEIPT"; wait\'; '
            'echo "$GYM_PID" > "$GROUP_RECEIPT"; '
            "while [[ ! -s $CHILD_RECEIPT ]]; do sleep 0.05; done; "
            "stop_judge_process_group; [[ -z $GYM_PID ]]",
        ],
        env=environment,
        check=False,
        timeout=10,
    )
    assert stopped.returncode == 0
    process_group = int(group_receipt.read_text())
    try:
        for _ in range(40):
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"judge process group {process_group} survived bounded cleanup")
    finally:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_launcher_uses_versioned_root_and_pins_transport_contract() -> None:
    script = (PACKAGE / "run_checkpoint_e2e.sh").read_text()
    assert (PACKAGE / "VERSION").read_text().strip() == "1.4.13"
    assert "checkpoint_e2e_true3_v1_4_13_runs" in script
    assert "gym-pr2588-d3f146d" in script
    assert "d3f146d386c7dfe07d4fabce32c4c8b14c7917d2" in script
    assert 'cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P' in script
    for name in (
        "transport_runtime.py",
        "transport_views.py",
        "fingerprint_probe.py",
        "true3_transport.yaml",
        "pr2588_true3_transport.patch",
        "provider_image_caps.patch",
        "provider_aggregate_media_caps.patch",
        "recursive_reference_assets.patch",
        "strict_comparison_trials.patch",
        "provider_context_fallback.patch",
        "provider_rate_limit_backoff.patch",
        "partial_pdf_overflow.patch",
        "transport_assignment.py",
        "transport_prebuild.sbatch",
        "judge_state.py",
        "judge_process_group.sh",
        "benchmarks/gdpval/prepare.py",
        "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ".venv/bin/gym",
    ):
        assert name in script
    assert "JUDGE_DELIVERABLES=$RUN_DIR/judge_transport_views/candidate" in script


def test_fingerprint_probe_is_provider_free_and_stops_before_resume_state() -> None:
    script = (PACKAGE / "fingerprint_probe.py").read_text()
    assert "GlobalConfigDictParser" in script
    assert "RolloutCollectionHelper()._preprocess_rows_from_config" in script
    assert "compute_fingerprint(" in script
    assert '"waivable_failure_classes": ["timeout_exceeded", "transient"]' in script
    assert "_prepare_resume" not in script
    assert "run_multistage_stages" not in script
    assert "subprocess" not in script
    assert "requests" not in script
    assert "aiohttp" not in script
    assert "curl" not in script
